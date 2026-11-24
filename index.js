require('dotenv').config();
const fs = require('fs');
const path = require('path');
const {
  default: makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  downloadMediaMessage,
} = require('@whiskeysockets/baileys');
const pino = require('pino');
const qrcode = require('qrcode-terminal');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// =========================================================================
// KONFIGURASI MEMORI PERMANEN 2 BULAN (60 HARI) & KNOWLEDGE FILES
// =========================================================================
const MEMORY_FILE = path.join(__dirname, 'chat_memory.json');
const RULES_FILE = path.join(__dirname, 'knowledge', 'rules.md');
const PRODUCT_FILE = path.join(__dirname, 'knowledge', 'product_knowledge.md');
const RETENTION_MS = 60 * 24 * 60 * 60 * 1000; // 60 hari dalam milidetik

// 1. Inisialisasi Gemini AI (Membaca rules.md & product_knowledge.md secara dinamis)
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

function loadSystemInstruction() {
  let rulesContent = '';
  let productContent = '';

  try {
    if (fs.existsSync(RULES_FILE)) {
      rulesContent = fs.readFileSync(RULES_FILE, 'utf8').trim();
    } else {
      console.warn('⚠️ File knowledge/rules.md tidak ditemukan!');
    }

    if (fs.existsSync(PRODUCT_FILE)) {
      productContent = fs.readFileSync(PRODUCT_FILE, 'utf8').trim();
    } else {
      console.warn('⚠️ File knowledge/product_knowledge.md tidak ditemukan!');
    }
  } catch (err) {
    console.error('❌ Gagal membaca file knowledge (.md):', err.message);
  }

  return `${rulesContent}\n\n${productContent}`.trim();
}

function getKnowledgeLastModified() {
  let latestMtime = 0;
  try {
    if (fs.existsSync(RULES_FILE)) {
      latestMtime = Math.max(latestMtime, fs.statSync(RULES_FILE).mtimeMs);
    }
    if (fs.existsSync(PRODUCT_FILE)) {
      latestMtime = Math.max(latestMtime, fs.statSync(PRODUCT_FILE).mtimeMs);
    }
  } catch (_) {}
  return latestMtime;
}

function createGeminiModel() {
  return genAI.getGenerativeModel({
    model: 'gemini-3.5-flash-lite',
    systemInstruction: loadSystemInstruction(),
    generationConfig: {
      maxOutputTokens: 2048,
      temperature: 0.7,
    },
  });
}

// Penyimpanan permanen di disk: { [jid]: { status: 'AI'|'HUMAN', updatedAt: number, messages: [{ role, text, timestamp }] } }
let persistentStore = {};

// Memory session aktif di RAM
// Format: { [jid]: { status: 'AI' | 'HUMAN', chatSession, loadedKnowledgeMtime, messageBuffer: [], imageBuffer: [], replyTimer: null } }
const userSessions = {};

// Menyimpan ID pesan yang dikirim oleh bot agar tidak terdeteksi sebagai pesan manual CS
const botSentMessageIds = new Set();

// Helper jeda waktu (ms)
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// =========================================================================
// FUNGSI PENGELOLA MEMORI 60 HARI (LOAD, SAVE, PRUNE)
// =========================================================================
function loadMemoryFromDisk() {
  try {
    if (fs.existsSync(MEMORY_FILE)) {
      const raw = fs.readFileSync(MEMORY_FILE, 'utf8');
      persistentStore = JSON.parse(raw);
      cleanupExpiredMemories();
      console.log(`🧠 Memori percakapan dimuat (${Object.keys(persistentStore).length} kontak aktif dalam 60 hari terakhir).`);
    }
  } catch (err) {
    console.error('⚠️ Gagal memuat file chat_memory.json, memulai memori baru:', err.message);
    persistentStore = {};
  }
}

function saveMemoryToDisk() {
  try {
    fs.writeFileSync(MEMORY_FILE, JSON.stringify(persistentStore, null, 2), 'utf8');
  } catch (err) {
    console.error('❌ Gagal menyimpan memori ke disk:', err.message);
  }
}

// Menghapus pesan atau sesi yang sudah berusia lebih dari 60 hari
function cleanupExpiredMemories() {
  const now = Date.now();
  let isModified = false;

  for (const jid of Object.keys(persistentStore)) {
    const record = persistentStore[jid];
    if (!record) continue;

    // Filter hanya pesan yang usianya <= 60 hari
    const validMessages = (record.messages || []).filter(
      (m) => now - m.timestamp <= RETENTION_MS
    );

    if (validMessages.length !== (record.messages || []).length) {
      record.messages = validMessages;
      isModified = true;
    }

    // Jika tidak ada pesan tersisa dan interaksi terakhir sudah > 60 hari, hapus kontak dari memori
    const lastActivity = record.updatedAt || 0;
    if (validMessages.length === 0 && now - lastActivity > RETENTION_MS) {
      delete persistentStore[jid];
      delete userSessions[jid];
      isModified = true;
    }
  }

  if (isModified) {
    saveMemoryToDisk();
    console.log('🧹 Pembersihan otomatis memori > 60 hari selesai dijalankan.');
  }
}

// Menyusun riwayat valid untuk Gemini (wajib diawali 'user' dan bergantian 'user' <-> 'model')
function buildValidGeminiHistory(messages) {
  const validHistory = [];
  for (const msg of messages) {
    if (validHistory.length === 0 && msg.role !== 'user') {
      continue; // Gemini mewajibkan history pertama berasal dari 'user'
    }
    const lastEntry = validHistory[validHistory.length - 1];
    if (lastEntry && lastEntry.role === msg.role) {
      // Gabungkan jika ada role yang sama berurutan
      lastEntry.parts[0].text += `\n${msg.text}`;
    } else {
      validHistory.push({
        role: msg.role,
        parts: [{ text: msg.text }],
      });
    }
  }

  // Jika pesan terakhir di history adalah 'user' (belum sempat dibalas bot), hapus agar tidak bentrok dengan pesan baru
  if (validHistory.length > 0 && validHistory[validHistory.length - 1].role === 'user') {
    validHistory.pop();
  }

  return validHistory;
}

// Mencatat pasangan pesan ke penyimpanan 60 hari
function appendConversationTurn(jid, userText, modelText) {
  const now = Date.now();
  if (!persistentStore[jid]) {
    persistentStore[jid] = {
      status: 'AI',
      updatedAt: now,
      messages: [],
    };
  }
  persistentStore[jid].updatedAt = now;
  persistentStore[jid].messages.push(
    { role: 'user', text: userText, timestamp: now },
    { role: 'model', text: modelText, timestamp: now }
  );
  saveMemoryToDisk();
}

function updatePersistedStatus(jid, status) {
  const now = Date.now();
  if (!persistentStore[jid]) {
    persistentStore[jid] = {
      status: status,
      updatedAt: now,
      messages: [],
    };
  } else {
    persistentStore[jid].status = status;
    persistentStore[jid].updatedAt = now;
  }
  saveMemoryToDisk();
}

// Helper mendapatkan atau memulihkan sesi user dari memori 60 hari (sekaligus hot-reload .md jika diubah)
function getOrCreateSession(jid) {
  const now = Date.now();
  const currentKnowledgeMtime = getKnowledgeLastModified();

  if (!userSessions[jid]) {
    const saved = persistentStore[jid];
    let restoredStatus = 'AI';
    let restoredHistory = [];

    if (saved) {
      // Bersihkan pesan > 60 hari milik user ini terlebih dahulu
      saved.messages = (saved.messages || []).filter(
        (m) => now - m.timestamp <= RETENTION_MS
      );

      if (saved.messages.length > 0 || now - (saved.updatedAt || 0) <= RETENTION_MS) {
        restoredStatus = saved.status || 'AI';
        restoredHistory = buildValidGeminiHistory(saved.messages);
      } else {
        delete persistentStore[jid];
        saveMemoryToDisk();
      }
    }

    userSessions[jid] = {
      status: restoredStatus,
      chatSession: createGeminiModel().startChat({ history: restoredHistory }),
      loadedKnowledgeMtime: currentKnowledgeMtime,
      messageBuffer: [],
      imageBuffer: [],
      replyTimer: null,
    };
  } else if (userSessions[jid].loadedKnowledgeMtime !== currentKnowledgeMtime) {
    // Jika file rules.md atau product_knowledge.md baru saja diedit, muat ulang instruksi tanpa mematikan bot
    const saved = persistentStore[jid];
    const restoredHistory = saved ? buildValidGeminiHistory(saved.messages || []) : [];
    userSessions[jid].chatSession = createGeminiModel().startChat({ history: restoredHistory });
    userSessions[jid].loadedKnowledgeMtime = currentKnowledgeMtime;
    console.log(`🔄 Knowledge (.md) diperbarui otomatis untuk sesi ${jid}.`);
  }

  return userSessions[jid];
}

// Muat memori dari disk saat aplikasi pertama kali dijalankan & jadwalkan pembersihan tiap 12 jam
loadMemoryFromDisk();
setInterval(cleanupExpiredMemories, 12 * 60 * 60 * 1000);

// 2. Fungsi Utama Menjalankan Koneksi WhatsApp Baileys
async function startKiniBot() {
  console.log('⏳ Menghubungkan ke server WhatsApp...');
  const { state, saveCreds } = await useMultiFileAuthState('auth_info_baileys');
  const { version } = await fetchLatestBaileysVersion();

  const sock = makeWASocket({
    version,
    auth: state,
    logger: pino({ level: 'error' }),
    browser: ['KiniTrip Bot', 'Chrome', '1.0.0'],
  });

  // Helper kirim pesan yang otomatis mencatat ID pesan bot
  async function sendBotMessage(jid, content, options = {}) {
    const sentMsg = await sock.sendMessage(jid, content, options);
    if (sentMsg?.key?.id) {
      botSentMessageIds.add(sentMsg.key.id);
      // Hapus ID dari set setelah 2 menit agar memori tetap ringan
      setTimeout(() => botSentMessageIds.delete(sentMsg.key.id), 120000);
    }
    return sentMsg;
  }

  // Simpan kredensial otomatis saat diperbarui
  sock.ev.on('creds.update', saveCreds);

  // Pantau status koneksi & tampilkan QR Code
  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      console.log('\n📱 Scan QR Code ini menggunakan WhatsApp Bisnis Kini Trip:');
      qrcode.generate(qr, { small: true });
    }

    if (connection === 'close') {
      const statusCode = lastDisconnect?.error?.output?.statusCode;
      const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
      console.log('⚠️ Koneksi WhatsApp terputus. Alasan:', lastDisconnect?.error?.message || statusCode);
      if (shouldReconnect) {
        startKiniBot();
      } else {
        console.log('❌ Sesi logout dari HP. Hapus folder auth_info_baileys lalu jalankan ulang untuk scan QR baru.');
      }
    } else if (connection === 'open') {
      console.log('✅ WhatsApp KiniBot (MinKI) berhasil terhubung dan siap melayani!');
    }
  });

  // 3. Event Listener Pesan Masuk & Pesan Keluar CS
  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;

    const msg = messages[0];
    if (!msg.message) return;

    const remoteJid = msg.key.remoteJid;

    // Abaikan grup (@g.us) dan status WhatsApp
    if (remoteJid.endsWith('@g.us') || remoteJid === 'status@broadcast') {
      return;
    }

    // Deteksi apakah pesan mengandung gambar
    const isImageMessage = Boolean(msg.message.imageMessage);

    // Ambil isi teks pesan (teks biasa, balasan/quote, atau caption gambar)
    const messageText = (
      msg.message.conversation ||
      msg.message.extendedTextMessage?.text ||
      msg.message.imageMessage?.caption ||
      ''
    ).trim();

    // Abaikan jika bukan teks dan bukan gambar
    if (!messageText && !isImageMessage) return;

    const senderNumber = remoteJid.replace('@s.whatsapp.net', '');

    // =========================================================================
    // KONTROL MANUAL OLEH CS DARI WA WEB / HP (fromMe === true)
    // =========================================================================
    if (msg.key.fromMe) {
      // Abaikan jika pesan ini dikirim oleh bot sendiri
      if (botSentMessageIds.has(msg.key.id)) {
        return;
      }

      const session = getOrCreateSession(remoteJid);

      // Cek apakah pesan CS diakhiri emoji gunung (⛰️⛰️) -> MATIKAN AI
      if (messageText.endsWith('⛰️⛰️') || messageText.endsWith('⛰⛰')) {
        session.status = 'HUMAN';
        updatePersistedStatus(remoteJid, 'HUMAN');
        if (session.replyTimer) {
          clearTimeout(session.replyTimer);
          session.replyTimer = null;
        }
        session.messageBuffer = [];
        session.imageBuffer = [];
        console.log(`🛑 [MANUAL OFF] CS mematikan AI untuk ${senderNumber} (emoji ⛰️⛰️ terdeteksi).`);
        return;
      }

      // Cek apakah pesan CS diakhiri emoji tenda (🏕️🏕️) -> NYALAKAN AI KEMBALI
      if (messageText.endsWith('🏕️🏕️') || messageText.endsWith('🏕🏕')) {
        if (session.replyTimer) {
          clearTimeout(session.replyTimer);
        }
        // Reset memori percakapan lama untuk sesi baru
        persistentStore[remoteJid] = {
          status: 'AI',
          updatedAt: Date.now(),
          messages: [],
        };
        saveMemoryToDisk();

        userSessions[remoteJid] = {
          status: 'AI',
          chatSession: createGeminiModel().startChat({ history: [] }),
          loadedKnowledgeMtime: getKnowledgeLastModified(),
          messageBuffer: [],
          imageBuffer: [],
          replyTimer: null,
        };
        console.log(`🟢 [MANUAL ON] CS menyalakan kembali AI untuk ${senderNumber} (emoji 🏕️🏕️ terdeteksi).`);
        return;
      }

      // Pesan manual biasa dari CS tanpa emoji pemicu -> abaikan
      return;
    }

    // =========================================================================
    // PENANGANAN PESAN MASUK DARI CUSTOMER
    // =========================================================================
    if (isImageMessage) {
      console.log(`\n🖼️ Gambar masuk dari ${senderNumber}${messageText ? ` dengan caption: "${messageText}"` : ' (tanpa caption)'}`);
    } else {
      console.log(`\n💬 Pesan masuk dari ${senderNumber}: "${messageText}"`);
    }

    const session = getOrCreateSession(remoteJid);

    // Jika chat sudah diambil alih CS manusia, bot berhenti merespons nomor ini
    if (session.status === 'HUMAN') {
      console.log(`⏸️ Status ${senderNumber} sudah HUMAN. Bot AI diam.`);
      return;
    }

    // Deteksi apakah customer sudah mengisi Form Pemesanan
    const isFormSubmission =
      messageText.toLowerCase().includes('form pemesanan kini trip') ||
      (messageText.toLowerCase().includes('nama lengkap:') &&
        messageText.toLowerCase().includes('destinasi'));

    if (isFormSubmission) {
      // Batalkan antrean balasan AI biasa jika masih menunggu
      if (session.replyTimer) {
        clearTimeout(session.replyTimer);
        session.replyTimer = null;
      }
      session.messageBuffer = [];
      session.imageBuffer = [];

      // Ubah status ke HUMAN agar AI tidak membalas lagi & simpan ke memori permanen
      session.status = 'HUMAN';
      updatePersistedStatus(remoteJid, 'HUMAN');

      const confirmationText =
        "Terima kasih sudah mengisi form pemesanan! 🏔️\n\nData kamu sudah kami teruskan ke Tim CS Kini Trip. CS kami akan segera menghubungi kamu di chat ini untuk verifikasi dan petunjuk pembayaran DP. Mohon tunggu sebentar ya!";

      appendConversationTurn(remoteJid, messageText, confirmationText);

      // 1. Kirim konfirmasi ke Customer
      await sendBotMessage(
        remoteJid,
        { text: confirmationText },
        { quoted: msg }
      );

      // 2. Kirim Alert Notifikasi ke Nomor WA CS Asli
      const csNumberClean = process.env.CS_WA_NUMBER.replace(/[^0-9]/g, '');
      const csJid = `${csNumberClean}@s.whatsapp.net`;

      const alertMessage = 
`🚨 *ALERT LEAD BARU KINI TRIP* 🚨
Customer telah mengisi Form Pemesanan!

*Kontak Customer:* wa.me/${senderNumber}
*Isi Form:*
${messageText}

⚠️ *Sistem AI telah DINONAKTIFKAN untuk nomor ini.* Silakan CS langsung ambil alih chat di WA Web! (Kirim pesan diakhiri 🏕️🏕️ ke customer jika urusan selesai untuk mengaktifkan AI kembali)`;

      try {
        await sendBotMessage(csJid, { text: alertMessage });
        console.log(`🚨 [ALERT CS SENT] Form dari ${senderNumber} berhasil dikirim ke CS (${csNumberClean}).`);
      } catch (err) {
        console.error('❌ Gagal mengirim notifikasi ke nomor CS:', err);
      }
      return;
    }

    // =========================================================================
    // UNDUH GAMBAR (JIKA ADA) & MASUKKAN KE ANTREAN
    // =========================================================================
    if (isImageMessage) {
      try {
        const imageBuffer = await downloadMediaMessage(
          msg,
          'buffer',
          {},
          {
            logger: pino({ level: 'error' }),
            reuploadRequest: sock.updateMediaMessage,
          }
        );

        const mimeType = msg.message.imageMessage.mimetype || 'image/jpeg';

        session.imageBuffer.push({
          inlineData: {
            data: imageBuffer.toString('base64'),
            mimeType: mimeType,
          },
        });

        // Tambahkan konteks teks ke buffer agar AI tahu ada gambar yang dikirim
        if (messageText) {
          session.messageBuffer.push(`[Customer mengirim gambar dengan caption]: ${messageText}`);
        } else {
          session.messageBuffer.push(`[Customer mengirim gambar tanpa caption]`);
        }
      } catch (imgErr) {
        console.error('❌ Gagal mengunduh gambar dari WhatsApp:', imgErr);
        session.messageBuffer.push(
          messageText
            ? `[Customer mengirim gambar yang gagal dimuat dengan caption]: ${messageText}`
            : `[Customer mengirim gambar namun gagal dimuat oleh sistem]`
        );
      }
    } else {
      session.messageBuffer.push(messageText);
    }

    // =========================================================================
    // PENGAMAN PESAN BERUNTUN & JEDA RANDOM (9 - 35 DETIK)
    // =========================================================================
    if (session.replyTimer) {
      const totalItems = session.messageBuffer.length;
      console.log(`📥 Pesan/gambar tambahan dari ${senderNumber} digabungkan ke antrean (${totalItems} item).`);
      return;
    }

    // Hitung durasi acak antara 9.000 ms (9 detik) hingga 35.000 ms (35 detik)
    const totalDelayMs = Math.floor(Math.random() * (35000 - 9000 + 1)) + 9000;
    const typingDurationMs = 4500; // Status "mengetik..." aktif di 4.5 detik terakhir
    const initialWaitMs = totalDelayMs - typingDurationMs;

    console.log(`⏳ Menunggu ${(totalDelayMs / 1000).toFixed(1)} detik sebelum membalas ${senderNumber}...`);

    session.replyTimer = setTimeout(async () => {
      try {
        // Cek ulang status jika CS tiba-tiba mematikan AI saat bot sedang menunggu
        if (session.status === 'HUMAN') {
          session.messageBuffer = [];
          session.imageBuffer = [];
          session.replyTimer = null;
          return;
        }

        // Fase 4.5 detik terakhir: Tampilkan indikator "sedang mengetik..."
        await sock.sendPresenceUpdate('composing', remoteJid);
        await sleep(typingDurationMs);

        // Cek sekali lagi setelah selesai mengetik
        if (session.status === 'HUMAN') {
          await sock.sendPresenceUpdate('paused', remoteJid);
          session.messageBuffer = [];
          session.imageBuffer = [];
          session.replyTimer = null;
          return;
        }

        // Gabungkan seluruh pesan beruntun & gambar menjadi satu payload utuh
        const combinedMessage = session.messageBuffer.join('\n');
        const imagesToSend = [...session.imageBuffer];

        session.messageBuffer = [];
        session.imageBuffer = [];
        session.replyTimer = null;

        // Jika ada gambar, kirim array [teks, ...gambar]; jika tidak ada, kirim string teks biasa
        const geminiPayload =
          imagesToSend.length > 0 ? [combinedMessage, ...imagesToSend] : combinedMessage;

        let result = await session.chatSession.sendMessage(geminiPayload);
        let aiResponse = result.response.text().trim();

        // Pengaman jika respons pertama kosong
        if (!aiResponse) {
          console.warn(
            '⚠️ Jawaban Gemini kosong. FinishReason:',
            result.response.candidates?.[0]?.finishReason
          );
          result = await session.chatSession.sendMessage(geminiPayload);
          aiResponse = result.response.text().trim();
        }

        await sock.sendPresenceUpdate('paused', remoteJid);

        if (aiResponse) {
          await sendBotMessage(remoteJid, { text: aiResponse });
          // Simpan percakapan ke memori permanen 60 hari
          appendConversationTurn(remoteJid, combinedMessage, aiResponse);
          console.log(`🤖 MinKI membalas ke ${senderNumber}: "${aiResponse}"`);
        } else {
          console.error('❌ Gemini tetap mengembalikan teks kosong setelah percobaan ulang.');
          await sendBotMessage(remoteJid, {
            text: 'Siap kak, boleh infoin lebih detail rencananya biar aku bantu cekin? 😊',
          });
        }
      } catch (error) {
        session.messageBuffer = [];
        session.imageBuffer = [];
        session.replyTimer = null;
        await sock.sendPresenceUpdate('paused', remoteJid);
        console.error('❌ Error saat memanggil Gemini API:', error);
        await sendBotMessage(remoteJid, {
          text: 'Mohon maaf kak, jaringan sedang agak gangguan sebentar. Tim kami segera bantu ya 🙏',
        });
      }
    }, initialWaitMs);
  });
}

// Jalankan Bot
startKiniBot();