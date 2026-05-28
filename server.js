const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// Load search index at startup
let searchIndex = null;
const indexPath = path.join(__dirname, 'search_index.json');
if (fs.existsSync(indexPath)) {
  searchIndex = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
  console.log('Search index loaded:', Object.keys(searchIndex).map(k => k + ':' + searchIndex[k].length).join(', '));
} else {
  console.warn('WARNING: search_index.json not found - RAG disabled');
}

// Manual ID -> index key mapping
const MANUAL_INDEX_MAP = {
  'e52645_guia':    'e52645_guia',
  'cpmd':           'cpmd',
  'm501_catalog':   'service',
  'm527_catalog':   'service',
  'e50045_catalog': 'service',
  'e52545_catalog': 'service',
};

function tokenize(text) {
  return text.toLowerCase().match(/[a-z0-9]+(?:\.[a-z0-9]+)*/g) || [];
}

function searchChunks(query, indexKey, topK = 6) {
  if (!searchIndex || !searchIndex[indexKey]) return [];
  const qTokens = new Set(tokenize(query));
  const chunks = searchIndex[indexKey];
  const scored = [];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const kTokens = new Set(chunk.k.split(' '));
    const tTokens = new Set(tokenize(chunk.t));
    const score = [...qTokens].filter(t => kTokens.has(t)).length * 2
                + [...qTokens].filter(t => tTokens.has(t)).length;
    if (score > 0) scored.push({ score, text: chunk.t });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(s => s.text);
}

app.post('/chat', async (req, res) => {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'API key nao configurada' });

  const { system, messages, manualId } = req.body;
  if (!messages || !messages.length) return res.status(400).json({ error: 'messages obrigatorio' });

  // RAG: find relevant chunks from the manual
  let contextBlock = '';
  if (searchIndex && manualId) {
    const indexKey = MANUAL_INDEX_MAP[manualId] || null;
    if (indexKey) {
      const lastQuestion = messages[messages.length - 1].content;
      const chunks = searchChunks(lastQuestion, indexKey, 6);
      if (chunks.length > 0) {
        contextBlock = '\n\n---\nTRECHOS RELEVANTES DO MANUAL (use como base para responder):\n\n'
          + chunks.map((c, i) => `[Trecho ${i+1}]:\n${c}`).join('\n\n')
          + '\n---\n';
      }
    }
  }

  // Inject context into system prompt
  const fullSystem = system + contextBlock;

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-6',
        max_tokens: 1024,
        system: fullSystem,
        messages: messages
      })
    });
    const data = await response.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/', (req, res) => res.send('Manuais HP RAG API OK'));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('Servidor rodando na porta ' + PORT));
