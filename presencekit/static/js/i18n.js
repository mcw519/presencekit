// Internationalization: string tables live in static/i18n/<lang>.js (loaded right after this file).
// Language: ?lang= in the URL (saved), else localStorage "presence-lang", else the browser language (zh* -> zh-TW),
// else English.
const I18N = {};
const LANGS = {en: 'EN', 'zh-TW': '中文'};
const LANG = (() => {
  const q = new URLSearchParams(location.search).get('lang');
  if (LANGS[q]) localStorage.setItem('presence-lang', q);
  const saved = localStorage.getItem('presence-lang');
  if (LANGS[saved]) return saved;
  for (const l of navigator.languages || [navigator.language || 'en']) {
    if (/^zh/i.test(l)) return 'zh-TW';
    if (/^en/i.test(l)) return 'en';
  }
  return 'en';
})();
/** Translate `key`, filling {placeholders} from `p`. Falls back to English, then to the key itself. */
function t(key, p) {
  let s = (I18N[LANG] || {})[key];
  if (s == null) s = (I18N.en || {})[key];
  if (s == null) return key;
  return p ? s.replace(/\{(\w+)\}/g, (_, n) => (p[n] ?? '')) : s;
}
const T = t;  // alias for places where `t` is shadowed by a local variable (e.g. a track)
/** Apply data-i18n (text) and data-i18n-<attr> (title, aria-label, placeholder) in the static markup. */
function applyI18n(root = document) {
  document.documentElement.lang = LANG === 'zh-TW' ? 'zh-Hant-TW' : 'en';
  root.querySelectorAll('[data-i18n]').forEach(e => { e.textContent = t(e.dataset.i18n); });
  for (const a of ['title', 'aria-label', 'placeholder'])
    root.querySelectorAll(`[data-i18n-${a}]`).forEach(e => e.setAttribute(a, t(e.getAttribute('data-i18n-' + a))));
}
/** Switch language: saved per browser; the page reloads so every panel is rebuilt in the new language. */
function setLang(l) {
  if (!LANGS[l] || l === LANG) return;
  localStorage.setItem('presence-lang', l);
  location.reload();
}
