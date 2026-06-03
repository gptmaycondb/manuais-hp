const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

let searchIndex = null;
const indexPath = path.join(__dirname, 'search_index.json');
if (fs.existsSync(indexPath)) {
  searchIndex = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
  console.log('Search index loaded:', Object.keys(searchIndex).map(k => k + ':' + searchIndex[k].length).join(', '));
} else {
  console.warn('WARNING: search_index.json not found - RAG disabled');
}

const MANUAL_INDEX_MAP = {
  'e52645_guia': 'e52645_guia', 'cpmd': 'cpmd',
  'm501_catalog': 'service', 'm527_catalog': 'service',
  'e50045_catalog': 'service', 'e52545_catalog': 'service',
};

function tokenize(text) {
  return text.toLowerCase().match(/[a-z0-9]+(?:\.[a-z0-9]+)*/g) || [];
}

function searchChunks(query, indexKey, topK = 6) {
  if (!searchIndex || !searchIndex[indexKey]) return [];
  const qTokens = new Set(tokenize(query));
  const scored = searchIndex[indexKey].map(chunk => ({
    score: [...qTokens].filter(t => new Set(chunk.k.split(' ')).has(t)).length * 2
         + [...qTokens].filter(t => new Set(tokenize(chunk.t)).has(t)).length,
    text: chunk.t,
  })).filter(c => c.score > 0);
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(s => s.text);
}

async function callClaude(model, fullSystem, messages, maxTokens) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY não configurada');
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model, max_tokens: maxTokens, system: fullSystem, messages }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message || JSON.stringify(data.error));
  return { content: [{ text: data.content[0].text }] };
}

async function callOpenAI(fullSystem, messages, maxTokens) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY não configurada');
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + apiKey },
    body: JSON.stringify({
      model: 'gpt-4o', max_tokens: maxTokens,
      messages: [{ role: 'system', content: fullSystem }, ...messages],
    }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message || JSON.stringify(data.error));
  return { content: [{ text: data.choices[0].message.content }] };
}

async function callGemini(fullSystem, messages, maxTokens) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY não configurada');
  const geminiMessages = messages.map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }));
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: fullSystem }] },
        contents: geminiMessages,
        generationConfig: { maxOutputTokens: maxTokens },
      }),
    }
  );
  const data = await res.json();
  if (data.error) throw new Error(data.error.message || JSON.stringify(data.error));
  return { content: [{ text: data.candidates[0].content.parts[0].text }] };
}

app.post('/chat', async (req, res) => {
  const { system, messages, manualId, max_tokens = 1024, provider = 'claude' } = req.body;
  if (!messages || !messages.length) return res.status(400).json({ error: 'messages obrigatorio' });

  let contextBlock = '';
  if (searchIndex && manualId) {
    const indexKey = MANUAL_INDEX_MAP[manualId] || null;
    if (indexKey) {
      const chunks = searchChunks(messages[messages.length - 1].content, indexKey, 6);
      if (chunks.length > 0) {
        contextBlock = '\n\n---\nTRECHOS RELEVANTES DO MANUAL (use como base para responder):\n\n'
          + chunks.map((c, i) => `[Trecho ${i+1}]:\n${c}`).join('\n\n') + '\n---\n';
      }
    }
  }
  const fullSystem = system + contextBlock;

  try {
    let result;
    if (provider === 'claude-opus') result = await callClaude('claude-opus-4-8',  fullSystem, messages, max_tokens);
    else if (provider === 'openai')  result = await callOpenAI(fullSystem, messages, max_tokens);
    else if (provider === 'gemini')  result = await callGemini(fullSystem, messages, max_tokens);
    else                             result = await callClaude('claude-sonnet-4-6', fullSystem, messages, max_tokens);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/', (req, res) => res.send('Manuais HP RAG API OK'));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('Servidor rodando na porta ' + PORT));
