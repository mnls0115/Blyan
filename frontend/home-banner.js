// Homepage usage banner: shows 24h inferences, p95 latency, and nodes online
(function () {
  const API = (window.API_CONFIG && window.API_CONFIG.baseURL) || '';

  async function fetchJSON(url, timeoutMs = 6000) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const r = await fetch(url, { signal: ctrl.signal });
      clearTimeout(to);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return await r.json();
    } catch (e) {
      clearTimeout(to);
      return null;
    }
  }

  async function loadBanner() {
    const banner = document.getElementById('usage-banner');
    const text = document.getElementById('usage-banner-text');
    if (!banner || !text) return;

    const [summary] = await Promise.all([
      fetchJSON(`${API}/analytics/summary?window=24h`)
    ]);

    // Compose banner text
    let parts = [];
    if (summary && typeof summary.inference_count === 'number') {
      parts.push(`24h inferences ${summary.inference_count}`);
    }
    if (summary && summary.backend_latency_ms && typeof summary.backend_latency_ms.p95 === 'number') {
      parts.push(`p95 ${Math.round(summary.backend_latency_ms.p95)} ms`);
    }
    // Intentionally no node counts on homepage banner

    if (parts.length) {
      text.textContent = `Network Status: ` + parts.join(' • ');
      banner.style.display = 'block';
    } else {
      // Hide if nothing to show
      banner.style.display = 'none';
    }
  }

  // Initial load and periodic refresh
  window.addEventListener('load', () => {
    loadBanner();
    setInterval(loadBanner, 60000);
  });
})();
