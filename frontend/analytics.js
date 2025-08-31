// Simple, privacy-aware frontend analytics
// - Generates an anonymous client_id (UUID v4) stored in localStorage
// - Sends small events to POST /analytics/event
// - Never sends prompt or response text

(function () {
  const STORAGE_KEY = 'blyan_client_id';
  const API_BASE = (window.API_CONFIG && window.API_CONFIG.baseURL) || '';

  function uuidv4() {
    // RFC4122 version 4 compliant UUID using crypto.getRandomValues
    const buf = new Uint8Array(16);
    if (window.crypto && window.crypto.getRandomValues) {
      window.crypto.getRandomValues(buf);
    } else {
      // Fallback: Math.random (less ideal, but avoids deps)
      for (let i = 0; i < 16; i++) buf[i] = Math.floor(Math.random() * 256);
    }
    buf[6] = (buf[6] & 0x0f) | 0x40; // version
    buf[8] = (buf[8] & 0x3f) | 0x80; // variant
    const bth = Array.from(buf, (b) => ('0' + b.toString(16)).slice(-2));
    return (
      bth[0] + bth[1] + bth[2] + bth[3] + '-' +
      bth[4] + bth[5] + '-' + bth[6] + bth[7] + '-' +
      bth[8] + bth[9] + '-' + bth[10] + bth[11] + bth[12] + bth[13] + bth[14] + bth[15]
    );
  }

  function getClientId() {
    try {
      let id = localStorage.getItem(STORAGE_KEY);
      if (!id) {
        id = uuidv4();
        localStorage.setItem(STORAGE_KEY, id);
      }
      return id;
    } catch (_) {
      return undefined;
    }
  }

  async function postEvent(body) {
    try {
      const resp = await fetch(`${API_BASE}/analytics/event`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        keepalive: true // allow sending on unload
      });
      // ignore response errors silently
      return resp.ok;
    } catch (_) {
      return false;
    }
  }

  const Analytics = {
    clientId: getClientId(),
    track: function (eventType, payload) {
      const body = Object.assign({
        event_type: eventType,
        client_id: this.clientId,
        route: window.location.pathname,
        ts_ms: Date.now()
      }, sanitizePayload(payload));
      return postEvent(body);
    }
  };

  function sanitizePayload(p) {
    const out = {};
    if (!p) return out;
    // allow only safe keys
    const allowed = ['status', 'duration_ms', 'model', 'meta'];
    for (const k of allowed) {
      if (p[k] !== undefined) out[k] = p[k];
    }
    if (out.meta) {
      // remove disallowed keys in meta
      const banned = ['prompt', 'response', 'messages', 'content'];
      for (const b of banned) delete out.meta[b];
    }
    return out;
  }

  // Expose
  window.Analytics = Analytics;

  // Track page_view on load and on hash changes
  function trackPageView() {
    Analytics.track('page_view', { meta: { title: document.title } });
  }
  window.addEventListener('load', trackPageView, { once: true });
  window.addEventListener('hashchange', trackPageView);
})();
