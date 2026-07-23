function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', ['search', 'benchmark', 'stats'][i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  if (name === 'stats') loadStats();
}

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function buildCards(results) {
  return results.map(hit => `
    <div class="card">
      <div class="card-header">
        <span class="label">${hit.label_name}</span>
        <span class="similarity">${(hit.similarity * 100).toFixed(1)}%</span>
      </div>
      <div class="card-text">${escapeHtml(hit.document)}</div>
    </div>`).join('');
}

// ── Search ──────────────────────────────────────────────────────────────────
async function doSearch() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const btn = document.getElementById('searchBtn');
  const output = document.getElementById('searchOutput');
  btn.disabled = true;
  btn.textContent = 'Searching…';
  output.innerHTML = '<div class="loading">Searching…</div>';

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    if (!res.ok) throw new Error();
    const data = await res.json();

    const hitClass = data.cache_hit ? 'badge-hit' : 'badge-miss';
    const hitLabel = data.cache_hit ? '✓ Cache Hit' : '✗ Cache Miss';

    let html = `
      <div class="result-meta">
        <span class="badge ${hitClass}">${hitLabel}</span>
        <span class="meta-item">${data.retrieval_time_ms} ms</span>
        <span class="meta-item">Cluster: <span>${data.cluster_label}</span></span>
      </div>`;

    if (data.cache_hit && data.matched_query) {
      html += `<div class="matched-query">Matched: <strong>"${data.matched_query}"</strong> · similarity ${data.similarity_score}</div>`;
    }

    html += `<div class="results">${buildCards(data.result)}</div>`;
    output.innerHTML = html;

  } catch (e) {
    output.innerHTML = '<div class="error">Something went wrong. Make sure the server is running.</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Search';
  }
}

// ── Benchmark ────────────────────────────────────────────────────────────────
async function doBenchmark() {
  const query = document.getElementById('benchInput').value.trim();
  if (!query) return;

  const btn = document.getElementById('benchBtn');
  const output = document.getElementById('benchOutput');
  btn.disabled = true;
  btn.textContent = 'Running…';
  output.innerHTML = '<div class="loading">Running both lookups…</div>';

  try {
    const res = await fetch('/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    if (!res.ok) throw new Error();
    const data = await res.json();

    const cacheWins = data.cache_time_ms < data.vector_db_time_ms;
    const hitClass = data.cache_hit ? 'badge-hit' : 'badge-miss';
    const hitLabel = data.cache_hit ? '✓ Cache Hit' : '✗ Cache Miss';

    let html = `
      <div class="result-meta" style="margin-bottom:20px">
        <span class="badge ${hitClass}">${hitLabel}</span>
      </div>
      <div class="benchmark-grid">
        <div class="bench-box">
          <div class="bench-label">Cache Lookup</div>
          <div class="bench-time ${cacheWins ? 'bench-winner' : ''}">${data.cache_time_ms}</div>
          <div class="bench-unit">ms</div>
        </div>
        <div class="bench-box">
          <div class="bench-label">ChromaDB Search</div>
          <div class="bench-time ${!cacheWins ? 'bench-winner' : ''}">${data.vector_db_time_ms}</div>
          <div class="bench-unit">ms</div>
        </div>
      </div>`;

    if (data.cache_hit && data.cache_result && data.cache_result.length) {
      html += `<div class="section-title">Cache Results</div>
               <div class="results">${buildCards(data.cache_result)}</div>`;
    }

    if (data.vector_db_result && data.vector_db_result.length) {
      html += `<div class="section-title" style="margin-top:24px">Vector DB Results</div>
               <div class="results">${buildCards(data.vector_db_result)}</div>`;
    }

    output.innerHTML = html;

  } catch (e) {
    output.innerHTML = '<div class="error">Something went wrong. Make sure the server is running.</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Benchmark';
  }
}

// ── Cache Stats ───────────────────────────────────────────────────────────────
async function loadStats() {
  const output = document.getElementById('statsOutput');
  try {
    const res = await fetch('/cache/stats');
    const d = await res.json();
    output.innerHTML = `
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">${d.total_entries}</div><div class="stat-label">Total Entries</div></div>
        <div class="stat-card"><div class="stat-value">${d.hit_count}</div><div class="stat-label">Cache Hits</div></div>
        <div class="stat-card"><div class="stat-value">${d.miss_count}</div><div class="stat-label">Cache Misses</div></div>
        <div class="stat-card"><div class="stat-value">${(d.hit_rate * 100).toFixed(0)}%</div><div class="stat-label">Hit Rate</div></div>
        <div class="stat-card"><div class="stat-value">${d.similarity_threshold}</div><div class="stat-label">Similarity Threshold</div></div>
        <div class="stat-card"><div class="stat-value">${d.max_entries}</div><div class="stat-label">Max Entries (LRU)</div></div>
        <div class="stat-card"><div class="stat-value">${d.ttl_seconds / 3600}h</div><div class="stat-label">TTL</div></div>
      </div>`;
  } catch (e) {
    output.innerHTML = '<div class="error">Could not load cache stats.</div>';
  }
}

async function flushCache() {
  if (!confirm('Flush all cache entries?')) return;
  await fetch('/cache', { method: 'DELETE' });
  loadStats();
}

// Enter key listeners
document.getElementById('queryInput').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
document.getElementById('benchInput').addEventListener('keydown', e => { if (e.key === 'Enter') doBenchmark(); });
