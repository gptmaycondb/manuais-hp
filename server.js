'use strict';

const express = require('express');
const cors    = require('cors');
const fs      = require('fs');
const path    = require('path');
const https   = require('https');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// ── Embeddings & Search ───────────────────────────────────────────────────────

const EMBEDDINGS_BASE =
  'https://raw.githubusercontent.com/gptmaycondb/techguide-ia/main/assets/embeddings';

const MANUAL_SEARCH_KEYS = {
  mfpe52645:     ['e52645_guia', 'cpmd', 'service'],
  mfpe62655:     ['e62655_guia', 'e62655_cpmd', 'e62655_service'],
  ricoh_imc3000: ['ricoh_imc3000_guia', 'ricoh_imc3000_service'],
  ricoh_mpc3004: ['ricoh_mpc3004_guia', 'ricoh_mpc3004_service'],
};

const embeddingsCache = {};
let embedder = null;
let embedderReady = false;

function fetchJSON(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}: ${url}`));
      let raw = '';
      res.on('data', d => raw += d);
      res.on('end', () => {
        try { resolve(JSON.parse(raw)); } catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

async function loadKey(key) {
  if (embeddingsCache[key]) return embeddingsCache[key];
  try {
    const data = await fetchJSON(`${EMBEDDINGS_BASE}/${key}.json`);
    embeddingsCache[key] = data[key] || [];
    console.log(JSON.stringify({ event: 'embedding_key_loaded', key, chunks: embeddingsCache[key].length }));
    return embeddingsCache[key];
  } catch (e) {
    console.log(JSON.stringify({ event: 'embedding_key_failed', key, error: e.message }));
    return [];
  }
}

async function preloadKeys() {
  const allKeys = [...new Set(Object.values(MANUAL_SEARCH_KEYS).flat())];
  await Promise.all(allKeys.map(loadKey));
  console.log(JSON.stringify({ event: 'embeddings_preloaded', keys: allKeys.length }));
}

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function keywordSearch(chunks, query, topK = 5) {
  const words = query.toLowerCase().split(/\W+/).filter(w => w.length > 2);
  const scored = chunks
    .map(c => ({ score: words.reduce((s, w) => s + (c.t.toLowerCase().includes(w) ? 1 : 0), 0), text: c.t }))
    .filter(c => c.score > 0);
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(c => c.text);
}

async function search(query, keys, topK = 5) {
  const allChunks = (await Promise.all(keys.map(loadKey))).flat();
  if (!allChunks.length) return [];

  if (embedderReady) {
    try {
      const out = await embedder(query, { pooling: 'mean', normalize: true });
      const qVec = Array.from(out.data);
      const scored = allChunks
        .filter(c => c.e)
        .map(c => ({ sim: cosineSim(qVec, c.e), text: c.t }));
      scored.sort((a, b) => b.sim - a.sim);
      return scored.slice(0, topK).map(c => c.text);
    } catch (e) {
      console.error('semantic_search_error:', e.message);
    }
  }
  return keywordSearch(allChunks, query, topK);
}

async function initEmbedder() {
  preloadKeys().catch(e => console.error('preload_error:', e.message));
  try {
    const { pipeline } = await import('@xenova/transformers');
    embedder = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');
    embedderReady = true;
    console.log(JSON.stringify({ event: 'embedder_ready' }));
  } catch (e) {
    console.log(JSON.stringify({ event: 'embedder_unavailable', error: e.message }));
  }
}

// ── AI Providers — instanciação lazy (sem API key não quebra o start) ─────────

const MODELS = {
  'claude':      'claude-sonnet-4-6',
  'claude-opus': 'claude-opus-4-8',
  'openai':      'gpt-4o',
  'gemini':      'gemini-1.5-pro',
};

async function callClaude(modelId, system, messages, maxTokens) {
  if (!process.env.ANTHROPIC_API_KEY) throw new Error('ANTHROPIC_API_KEY não configurada no servidor.');
  const { default: Anthropic } = await import('@anthropic-ai/sdk');
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  const res = await client.messages.create({ model: modelId, max_tokens: maxTokens, system, messages });
  return res.content[0].text;
}

async function callOpenAI(system, messages, maxTokens) {
  if (!process.env.OPENAI_API_KEY) throw new Error('OPENAI_API_KEY não configurada no servidor.');
  const { default: OpenAI } = await import('openai');
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const msgs = [{ role: 'system', content: system }, ...messages];
  const res = await client.chat.completions.create({ model: MODELS.openai, max_tokens: maxTokens, messages: msgs });
  return res.choices[0].message.content;
}

async function callGemini(system, messages, maxTokens) {
  if (!process.env.GEMINI_API_KEY) throw new Error('GEMINI_API_KEY não configurada no servidor.');
  const { GoogleGenerativeAI } = await import('@google/generative-ai');
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const gModel = genAI.getGenerativeModel({
    model: MODELS.gemini,
    systemInstruction: system,
    generationConfig: { maxOutputTokens: maxTokens },
  });
  const history = messages.slice(0, -1).map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }));
  const chat = gModel.startChat({ history });
  const last = messages[messages.length - 1]?.content || '';
  const res = await chat.sendMessage(last);
  return res.response.text();
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get('/ping', (_req, res) => {
  res.json({ ok: true, ts: Date.now() });
});

app.post('/chat', async (req, res) => {
  const t0 = Date.now();
  const {
    systemBase, query, history = [], manualId,
    system: legacySystem, messages: legacyMessages,
    max_tokens = 1024, provider = 'claude',
  } = req.body;

  const isNew = !!query;

  try {
    let systemPrompt, chatMessages, foundInManual = false, chunksFound = 0;

    if (isNew) {
      const keys = MANUAL_SEARCH_KEYS[manualId] || [];
      const chunks = keys.length ? await search(query, keys, 5) : [];
      chunksFound   = chunks.length;
      foundInManual = chunksFound > 0;

      const ctx = chunks.length
        ? '\n\nTrechos relevantes do manual:\n' + chunks.map((c, i) => `[${i + 1}] ${c}`).join('\n\n')
        : '';
      systemPrompt  = (systemBase || '') + ctx;
      chatMessages  = [
        ...history.map(m => ({
          role: m.role === 'ai' ? 'assistant' : (m.role || 'user'),
          content: m.text || m.content || '',
        })),
        { role: 'user', content: query },
      ];
    } else {
      systemPrompt  = legacySystem || '';
      chatMessages  = (legacyMessages || []).map(m => ({
        role: m.role === 'assistant' ? 'assistant' : (m.role || 'user'),
        content: typeof m.content === 'string' ? m.content : (m.content?.[0]?.text || ''),
      }));
      foundInManual = systemPrompt.length > 200;
      chunksFound   = -1;
    }

    let text;
    if (provider === 'openai') {
      text = await callOpenAI(systemPrompt, chatMessages, max_tokens);
    } else if (provider === 'gemini') {
      text = await callGemini(systemPrompt, chatMessages, max_tokens);
    } else {
      const model = MODELS[provider] || MODELS.claude;
      text = await callClaude(model, systemPrompt, chatMessages, max_tokens);
    }

    if (!text) throw new Error('Resposta vazia do modelo');

    console.log(JSON.stringify({
      ts: new Date().toISOString(), provider, manualId: manualId || null,
      queryLen: query?.length || 0, chunksFound,
      durationMs: Date.now() - t0, status: 'ok',
    }));

    res.json({ content: [{ text }], foundInManual });

  } catch (err) {
    console.log(JSON.stringify({
      ts: new Date().toISOString(), provider, manualId: manualId || null,
      queryLen: query?.length || 0, durationMs: Date.now() - t0,
      status: 'error', error: err.message,
    }));
    res.status(500).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(JSON.stringify({ event: 'server_start', port: PORT }));
  initEmbedder();
});
