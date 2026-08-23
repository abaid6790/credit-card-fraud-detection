// Theme toggle
(function () {
  const toggle = document.getElementById('themeToggle');
  const root = document.documentElement;
  const saved = localStorage_safe_get('fraudguard-theme') || 'light';
  root.setAttribute('data-theme', saved);
  if (toggle) {
    toggle.addEventListener('click', () => {
      const current = root.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage_safe_set('fraudguard-theme', next);
    });
  }
  // Guard localStorage in case of restricted contexts
  function localStorage_safe_get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function localStorage_safe_set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
})();

function riskBadgeClass(level) {
  const map = {
    VERY_LOW: 'badge-very-low', LOW: 'badge-low', MODERATE: 'badge-moderate',
    HIGH: 'badge-high', CRITICAL: 'badge-critical',
  };
  return map[level] || 'badge-low';
}

function riskLabel(level) {
  const map = {
    VERY_LOW: 'Very Low', LOW: 'Low', MODERATE: 'Moderate', HIGH: 'High', CRITICAL: 'Critical',
  };
  return map[level] || level;
}

async function postJSON(url, body, extraHeaders = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.message || 'Request failed');
    err.data = data;
    throw err;
  }
  return data;
}

async function getJSON(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.message || 'Request failed');
    err.data = data;
    throw err;
  }
  return data;
}
