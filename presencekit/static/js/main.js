// Start-up: wire global buttons, build the settings UI and connect.
// Classic script; all files share one global scope and load in order from index.html.
applyI18n();
document.querySelectorAll('#langSw button').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.l===LANG));b.onclick=()=>setLang(b.dataset.l)});
setTheme(document.documentElement.dataset.themeChoice||localStorage.getItem('presence-theme')||'auto');document.querySelectorAll('#theme button').forEach(b=>b.onclick=()=>setTheme(b.dataset.t));$('navSet').onclick=()=>$('openSet').click();
$('navEv').onclick=()=>{const p=$('evPanel');p.open=!p.open;if(p.open)p.scrollIntoView({block:'nearest'})};buildUI();addCalib();addAsrKey();addReidKey();renderRV();setMonitor('off');new ResizeObserver(fit).observe(cv);connect();
