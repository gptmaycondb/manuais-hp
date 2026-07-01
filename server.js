require('dotenv').config();
const express = require('express');
const cors = require('cors');
const admin = require('firebase-admin');

const app = express();
app.use(cors());
app.use(express.json({ limit: '4mb' }));

const PORT = process.env.PORT || 3000;
const SEMANTIC = process.env.SEMANTIC_SEARCH !== '0';
const REPO_RAW = 'https://raw.githubusercontent.com/gptmaycondb/techguide-ia/main/assets/embeddings';

// ── Model IDs (override via env vars, no rebuild required) ────────────────────
const GEMINI_MODEL    = process.env.GEMINI_MODEL    || 'gemini-2.5-flash';
const OPENAI_MODEL    = process.env.OPENAI_MODEL    || 'gpt-4o-mini';
const ANTHROPIC_MODEL = process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-6';

function parseDailyLimit(value) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : 10;
}

const DAILY_LIMIT = parseDailyLimit(process.env.DAILY_LIMIT);
const RESERVATION_TTL_MS = 5 * 60 * 1000;

const AUTH_ERROR = {
  error: 'auth_required',
  message: 'Sessão expirada, faça login novamente.',
};

function initializeFirebaseServices(serviceAccountJson = process.env.FIREBASE_SERVICE_ACCOUNT) {
  if (!serviceAccountJson) {
    console.error('Firebase Admin disabled: FIREBASE_SERVICE_ACCOUNT is missing');
    return null;
  }
  try {
    const serviceAccount = JSON.parse(serviceAccountJson);
    const firebaseApp = admin.apps.length
      ? admin.app()
      : admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    console.log('Firebase Admin initialized');
    return {
      auth: firebaseApp.auth(),
      firestore: firebaseApp.firestore(),
    };
  } catch (error) {
    console.error(`Firebase Admin disabled: invalid FIREBASE_SERVICE_ACCOUNT (${error.message})`);
    return null;
  }
}

function createRequireAuth(firebaseAuth) {
  return async function requireAuth(req, res, next) {
    const match = req.headers.authorization?.match(/^Bearer\s+(.+)$/i);
    if (!firebaseAuth || !match) {
      return res.status(401).json(AUTH_ERROR);
    }
    try {
      const decoded = await firebaseAuth.verifyIdToken(match[1]);
      req.uid = decoded.uid;
      req.email = decoded.email || null;
      return next();
    } catch {
      return res.status(401).json(AUTH_ERROR);
    }
  };
}

function getSaoPauloDay(value = new Date()) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Sao_Paulo',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(value);
  const get = type => parts.find(part => part.type === type)?.value;
  return `${get('year')}-${get('month')}-${get('day')}`;
}

function isMasterEmail(email, masterEmail) {
  return Boolean(email && masterEmail)
    && email.trim().toLowerCase() === masterEmail.trim().toLowerCase();
}

function isValidRequestId(requestId) {
  return typeof requestId === 'string'
    && /^[A-Za-z0-9_-]{16,128}$/.test(requestId);
}

function cleanExpiredReservations(requests, nowMs) {
  const cleaned = {};
  for (const [requestId, entry] of Object.entries(requests || {})) {
    if (entry?.state === 'reserved' && Number(entry.expiresAtMs || 0) <= nowMs) continue;
    cleaned[requestId] = entry;
  }
  return cleaned;
}

function createUsageLimit({
  firestore,
  fieldValue,
  masterEmail = process.env.MASTER_EMAIL,
  dailyLimit = DAILY_LIMIT,
  now = () => new Date(),
  reservationTtlMs = RESERVATION_TTL_MS,
}) {
  return async function requireUsage(req, res, next) {
    const requestId = req.body?.requestId;
    if (!isValidRequestId(requestId)) {
      return res.status(400).json({
        error: 'request_id_required',
        message: 'Identificador da consulta ausente ou inválido.',
      });
    }
    if (isMasterEmail(req.email, masterEmail)) {
      req.usage = { unlimited: true };
      req.quotaReservation = { unlimited: true };
      return next();
    }
    if (!firestore) {
      return res.status(503).json({
        error: 'usage_unavailable',
        message: 'Controle de consultas temporariamente indisponível. Tente novamente.',
      });
    }

    const reservationTime = now();
    const day = getSaoPauloDay(reservationTime);
    const ref = firestore.collection('usage').doc(`${req.uid}_${day}`);
    const nowMs = reservationTime.getTime();
    let confirmed = 0;
    try {
      await firestore.runTransaction(async transaction => {
        const snapshot = await transaction.get(ref);
        const data = snapshot.exists ? snapshot.data() : {};
        confirmed = Number(data?.confirmed ?? data?.count ?? 0);
        const requests = cleanExpiredReservations(data?.requests, nowMs);
        const existing = requests[requestId];

        if (existing?.state === 'confirmed') return;
        if (existing?.state === 'reserved') {
          requests[requestId] = {
            ...existing,
            attempts: Number(existing.attempts || 1) + 1,
            expiresAtMs: nowMs + reservationTtlMs,
          };
        } else {
          const activeReservations = Object.values(requests)
            .filter(entry => entry?.state === 'reserved').length;
          if (confirmed + activeReservations >= dailyLimit) {
            const error = new Error('daily_limit_reached');
            error.code = 'daily_limit_reached';
            throw error;
          }
          requests[requestId] = {
            state: 'reserved',
            attempts: 1,
            createdAtMs: nowMs,
            expiresAtMs: nowMs + reservationTtlMs,
          };
        }

        transaction.set(ref, {
          uid: req.uid,
          day,
          confirmed,
          requests,
          updatedAt: fieldValue.serverTimestamp(),
        }, { merge: true });
      });
    } catch (error) {
      if (error.code === 'daily_limit_reached') {
        return res.status(429).json({
          error: 'rate_limit',
          message: 'Limite diário de consultas atingido. Libera à meia-noite.',
          used: confirmed,
          limit: dailyLimit,
        });
      }
      console.error(`Usage reservation failed for uid=${req.uid}:`, error.message);
      return res.status(503).json({
        error: 'usage_unavailable',
        message: 'Controle de consultas temporariamente indisponível. Tente novamente.',
      });
    }

    req.usage = { used: confirmed, limit: dailyLimit, remaining: dailyLimit - confirmed };
    req.quotaReservation = {
      firestore,
      fieldValue,
      ref,
      requestId,
      dailyLimit,
      now,
    };
    return next();
  };
}

async function confirmUsage(reservation) {
  if (reservation?.unlimited) return { unlimited: true };
  const { firestore, fieldValue, ref, requestId, dailyLimit, now } = reservation;
  let confirmed = 0;
  await firestore.runTransaction(async transaction => {
    const snapshot = await transaction.get(ref);
    const data = snapshot.exists ? snapshot.data() : {};
    confirmed = Number(data?.confirmed ?? data?.count ?? 0);
    const requests = cleanExpiredReservations(data?.requests, now().getTime());
    const existing = requests[requestId];
    if (existing?.state === 'confirmed') return;
    if (existing?.state !== 'reserved') {
      const error = new Error('usage_reservation_expired');
      error.code = 'usage_reservation_expired';
      throw error;
    }

    confirmed += 1;
    requests[requestId] = {
      state: 'confirmed',
      confirmedAtMs: now().getTime(),
    };
    transaction.set(ref, {
      confirmed,
      requests,
      updatedAt: fieldValue.serverTimestamp(),
    }, { merge: true });
  });
  return { used: confirmed, limit: dailyLimit, remaining: dailyLimit - confirmed };
}

async function releaseUsage(reservation) {
  if (!reservation || reservation.unlimited) {
    return reservation?.unlimited ? { unlimited: true } : null;
  }
  const { firestore, fieldValue, ref, requestId, dailyLimit, now } = reservation;
  let confirmed = 0;
  await firestore.runTransaction(async transaction => {
    const snapshot = await transaction.get(ref);
    const data = snapshot.exists ? snapshot.data() : {};
    confirmed = Number(data?.confirmed ?? data?.count ?? 0);
    const requests = cleanExpiredReservations(data?.requests, now().getTime());
    const existing = requests[requestId];
    if (existing?.state !== 'reserved') return;

    const attempts = Number(existing.attempts || 1);
    if (attempts > 1) {
      requests[requestId] = { ...existing, attempts: attempts - 1 };
    } else {
      delete requests[requestId];
    }
    transaction.set(ref, {
      requests,
      updatedAt: fieldValue.serverTimestamp(),
    }, { merge: true });
  });
  return { used: confirmed, limit: dailyLimit, remaining: dailyLimit - confirmed };
}

function isProviderRateLimit(error) {
  return error?.status === 429
    || error?.statusCode === 429
    || /(?:429|rate.?limit|quota|resource_exhausted)/i.test(error?.message || '');
}

const firebaseServices = initializeFirebaseServices();
const requireAuth = createRequireAuth(firebaseServices?.auth);
const requireUsage = createUsageLimit({
  firestore: firebaseServices?.firestore,
  fieldValue: admin.firestore.FieldValue,
});

// ── Instrução de formato universal (todos os providers) ───────────────────────
const FORMAT_INSTRUCTION = '\n\nEstruture a resposta em markdown com seções (Defeito, Causas, Solução passo a passo, Recuperação, SPs/Peças), tabelas onde couber, e marque explicitamente como \'(Complemento)\' qualquer informação que não esteja nos trechos fornecidos.';

// ── Embeddings ────────────────────────────────────────────────────────────────
const KEYS = [
  'e52645_guia','cpmd','service',
  'e62655_guia','e62655_cpmd','e62655_service',
  'ricoh_imc3000_guia','ricoh_imc3000_service','ricoh_imc3000_parts',
  'ricoh_mpc3004_guia','ricoh_mpc3004_service',
];

const embeddingsStore = {};
let embedder = null;

async function loadEmbeddings() {
  if (!SEMANTIC) { console.log('SEMANTIC_SEARCH=0 — skipping embeddings'); return; }
  console.log('Loading embeddings sequentially...');
  for (const key of KEYS) {
    try {
      const res = await fetch(`${REPO_RAW}/${key}.json`);
      if (!res.ok) { console.warn(`skip ${key}: HTTP ${res.status}`); continue; }
      const data = await res.json();
      const chunks = data[key] || [];
      if (!chunks.length) continue;
      const dim = chunks[0].e.length;
      const vecs = new Float32Array(chunks.length * dim);
      chunks.forEach((c, i) => vecs.set(c.e, i * dim));
      embeddingsStore[key] = { texts: chunks.map(c => c.t), vecs, dim };
      console.log(`  ${key}: ${chunks.length} chunks`);
      if (global.gc) global.gc();
    } catch (e) { console.warn(`skip ${key}:`, e.message); }
  }
  console.log('Embeddings loaded.');
}

async function loadEmbedder() {
  if (!SEMANTIC) return;
  try {
    const { pipeline } = await import('@xenova/transformers');
    embedder = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', { quantized: true });
    console.log('Embedder ready.');
  } catch (e) { console.warn('Embedder failed:', e.message); }
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

async function semanticSearch(query, key, topK = 5) {
  const store = embeddingsStore[key];
  if (!store || !embedder) return [];
  const out = await embedder(query, { pooling: 'mean', normalize: true });
  const qVec = Array.from(out.data);
  const { texts, vecs, dim } = store;
  const scores = texts.map((t, i) => ({
    t, score: cosineSim(qVec, Array.from(vecs.subarray(i * dim, (i+1) * dim)))
  }));
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topK).map(s => s.t);
}

function keywordSearch(query, key, topK = 5) {
  const store = embeddingsStore[key];
  if (!store) return [];
  const q = query.toLowerCase().split(/\s+/);
  const scored = store.texts
    .map(t => ({ t, score: q.reduce((s, w) => s + (t.toLowerCase().includes(w) ? 1 : 0), 0) }))
    .filter(x => x.score > 0);
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(s => s.t);
}

// ── Lazy AI providers ─────────────────────────────────────────────────────────
async function callClaude(systemPrompt, messages, maxTokens, model, onDelta) {
  const { default: Anthropic } = await import('@anthropic-ai/sdk');
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  let fullText = '';
  const stream = await client.messages.stream({
    model: model || ANTHROPIC_MODEL,
    max_tokens: maxTokens || 3072,
    system: systemPrompt,
    messages,
  });
  for await (const event of stream) {
    if (event.type === 'content_block_delta' && event.delta?.type === 'text_delta') {
      onDelta(event.delta.text);
      fullText += event.delta.text;
    }
  }
  return fullText;
}

async function callOpenAI(systemPrompt, messages, maxTokens, onDelta) {
  const { OpenAI } = await import('openai');
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  let fullText = '';
  const stream = await openai.chat.completions.create({
    model: OPENAI_MODEL, max_tokens: maxTokens || 3072, stream: true,
    messages: [{ role: 'system', content: systemPrompt }, ...messages],
  });
  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content || '';
    if (text) { onDelta(text); fullText += text; }
  }
  return fullText;
}

// Pure function — testable without API key (SDK validates history client-side)
function buildGeminiMessages(messages) {
  const mapped = messages.map(m => ({
    role: m.role === 'user' ? 'user' : 'model',
    parts: [{ text: m.content }],
  }));
  const lastMsg = mapped.pop() || null;
  // Gemini requires history[0].role === 'user'; strip any leading model turns
  while (mapped.length > 0 && mapped[0].role !== 'user') mapped.shift();
  return { history: mapped, userText: lastMsg?.parts[0]?.text || '' };
}

async function callGemini(systemPrompt, messages, maxTokens, onDelta) {
  const { GoogleGenAI } = await import('@google/genai');
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  let fullText = '';
  const { history, userText } = buildGeminiMessages(messages);
  try {
    const chat = ai.chats.create({
      model: GEMINI_MODEL,
      config: { systemInstruction: systemPrompt },
      history,
    });
    const stream = await chat.sendMessageStream({ message: userText });
    for await (const chunk of stream) {
      const text = chunk.text;
      if (text) { onDelta(text); fullText += text; }
    }
  } catch (e) {
    if (e.status) console.error(`Gemini HTTP ${e.status} ${e.statusText || ''}:`, String(e.message).slice(0, 500));
    throw e;
  }
  return fullText;
}

// ── Index map ─────────────────────────────────────────────────────────────────
const MANUAL_KEY_MAP = {
  'mfpe52645':'e52645_guia', 'e52645_guia':'e52645_guia',
  'cpmd':'cpmd', 'service':'service',
  'mfpe62655':'e62655_guia', 'e62655_guia':'e62655_guia',
  'e62655_cpmd':'e62655_cpmd', 'e62655_service':'e62655_service',
  'ricoh_imc3000':'ricoh_imc3000_guia', 'ricoh_imc3000_guia':'ricoh_imc3000_guia',
  'ricoh_imc3000_service':'ricoh_imc3000_service', 'ricoh_imc3000_parts':'ricoh_imc3000_parts',
  'ricoh_mpc3004':'ricoh_mpc3004_guia', 'ricoh_mpc3004_guia':'ricoh_mpc3004_guia',
  'ricoh_mpc3004_service':'ricoh_mpc3004_service',
};

// ── Routes ────────────────────────────────────────────────────────────────────
app.get('/ping', (req, res) => {
  res.json({ ok: true, ts: Date.now(), semantic: !!embedder });
});

app.get('/providers', (req, res) => {
  const providers = [];
  if (process.env.ANTHROPIC_API_KEY) providers.push('claude');
  if (process.env.OPENAI_API_KEY) providers.push('openai');
  if (process.env.GEMINI_API_KEY) providers.push('gemini');
  res.json({ providers });
});

app.post('/chat', requireAuth, requireUsage, async (req, res) => {
  const t0 = Date.now();
  const wantsStream = (req.headers.accept || '').includes('text/event-stream');
  let clientDisconnected = false;
  res.on('close', () => {
    if (!res.writableEnded) clientDisconnected = true;
  });

  if (wantsStream) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.flushHeaders();
  }

  const sendDelta = (text) => {
    if (wantsStream && !res.destroyed && !res.writableEnded) {
      res.write(`data: ${JSON.stringify({ type: 'delta', text })}\n\n`);
    }
  };
  if (wantsStream) {
    res.write(`data: ${JSON.stringify({ type: 'usage', ...req.usage })}\n\n`);
  }

  try {
    const isNew = !!req.body.query;
    const provider = req.body.provider || 'claude';
    const maxTokens = req.body.max_tokens || 3072;

    let systemPrompt, apiMessages, foundInManual;

    if (isNew) {
      const { query, systemBase, history = [], manualId } = req.body;
      const primaryKey = MANUAL_KEY_MAP[manualId] || manualId || 'e52645_guia';
      const chunks = embedder
        ? await semanticSearch(query, primaryKey, 5)
        : keywordSearch(query, primaryKey, 5);

      foundInManual = chunks.length > 0;
      const cap = c => c.length > 700 ? c.substring(0, 700) + '…' : c;
      const contextBlock = foundInManual
        ? '\n\nTRECHOS DO MANUAL:\n\n' + chunks.map((c, i) => `[${i+1}]\n${cap(c)}`).join('\n\n---\n\n')
          + '\n\nResponda baseando-se nos trechos acima.'
        : '\n\nNenhum trecho encontrado nos manuais. Responda com conhecimento tecnico geral.';

      systemPrompt = (systemBase || '') + contextBlock;
      apiMessages = [...history.slice(-6), { role: 'user', content: query }];
    } else {
      systemPrompt = req.body.system || '';
      apiMessages = req.body.messages || [];
      foundInManual = true;
    }

    systemPrompt += FORMAT_INSTRUCTION;

    const modelMap = { 'claude-opus': 'claude-opus-4-8', claude: ANTHROPIC_MODEL };
    let fullText = '';

    if (provider === 'claude' || provider === 'claude-opus') {
      fullText = await callClaude(systemPrompt, apiMessages, maxTokens, modelMap[provider], sendDelta);
    } else if (provider === 'openai') {
      fullText = await callOpenAI(systemPrompt, apiMessages, maxTokens, sendDelta);
    } else if (provider === 'gemini') {
      fullText = await callGemini(systemPrompt, apiMessages, maxTokens, sendDelta);
    } else {
      throw new Error(`Unknown provider: ${provider}`);
    }

    if (clientDisconnected || res.destroyed) {
      const error = new Error('client_disconnected_before_completion');
      error.code = 'client_disconnected';
      throw error;
    }

    req.usage = await confirmUsage(req.quotaReservation);
    console.log(JSON.stringify({ provider, ms: Date.now()-t0, chars: fullText.length, foundInManual }));

    if (wantsStream) {
      res.write(`data: ${JSON.stringify({ type: 'usage', ...req.usage })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'done', foundInManual })}\n\n`);
      res.end();
    } else {
      res.json({ content: [{ text: fullText }], foundInManual, usage: req.usage });
    }
  } catch (err) {
    let usageAfterRelease = req.usage;
    try {
      usageAfterRelease = await releaseUsage(req.quotaReservation) || usageAfterRelease;
    } catch (releaseError) {
      console.error(`Usage release failed for uid=${req.uid}:`, releaseError.message);
    }
    const providerRateLimit = isProviderRateLimit(err);
    const status = providerRateLimit ? 429 : 500;
    const error = providerRateLimit ? 'provider_rate_limit' : 'provider_error';
    const message = providerRateLimit
      ? 'Serviço de IA temporariamente indisponível. Tente em instantes.'
      : err.message;
    console.error(`chat error status=${err.status || err.statusCode || status}:`, err.message);
    if (wantsStream) {
      if (!res.destroyed && !res.writableEnded) {
        res.write(`data: ${JSON.stringify({ type: 'usage', ...usageAfterRelease })}\n\n`);
        res.write(`data: ${JSON.stringify({ type: 'error', error, message })}\n\n`);
        res.end();
      }
    } else {
      res.status(status).json({ error, message, usage: usageAfterRelease });
    }
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, async () => {
  console.log(`Server listening on port ${PORT}`);

  if (process.env.ANTHROPIC_API_KEY) console.log(`provider=claude model=${ANTHROPIC_MODEL}`);
  if (process.env.OPENAI_API_KEY)    console.log(`provider=openai model=${OPENAI_MODEL}`);
  if (process.env.GEMINI_API_KEY) {
    console.log(`provider=gemini model=${GEMINI_MODEL}`);
    try {
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GEMINI_API_KEY}`);
      if (r.ok) {
        const data = await r.json();
        const names = (data.models || []).map(m => m.name.replace('models/', ''));
        console.log(`gemini models (${names.length}): ${names.slice(0, 8).join(', ')}${names.length > 8 ? '…' : ''}`);
      } else {
        console.warn(`gemini ListModels: HTTP ${r.status}`);
      }
    } catch (e) { console.warn('gemini ListModels failed:', e.message); }
  }

  await loadEmbeddings();
  await loadEmbedder();
});
