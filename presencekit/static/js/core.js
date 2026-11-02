// Core helpers and theme (tokens: docs/ui.md); charts read the same CSS variables.
// Classic script; all files share one global scope and load in order from index.html.
const $=id=>document.getElementById(id);
const K={};
function pal(){const cs=getComputedStyle(document.documentElement);
 for(const n of['ok','warn','bad','accent','ink','ink-muted','violet','end','idle','viz-raw','viz-clean','viz-grid','viz-empty','viz-ink','line-strong'])K[n.replace(/-/g,'_')]=cs.getPropertyValue('--'+n).trim();
 if(typeof DC!=='undefined'){Object.assign(DC,{agree:K.ok,leak:K.bad,false_reject:K.warn,false_accept:K.violet,silence:K.end,acoustic_only:K.viz_raw});
  Object.assign(SC,{ACTIVE:K.ok,IDLE:K.idle,MAYBE_AWAY:K.warn,AWAY:K.bad,ENDED:K.end});const l=$('tlLeg');if(l&&typeof legend==='function')legend()}}
function resolveTheme(c){return c==='dark'||(c==='auto'&&matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light'}
function setTheme(t){if(!['auto','light','dark'].includes(t))t='auto';localStorage.setItem('presence-theme',t);
 document.documentElement.dataset.themeChoice=t;document.documentElement.dataset.theme=resolveTheme(t);
 document.querySelectorAll('#theme button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.t===t)));pal()}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if((localStorage.getItem('presence-theme')||'auto')==='auto'){document.documentElement.dataset.theme=resolveTheme('auto');pal()}});
// Bounded client measurements. Timings are milliseconds; JPEG encode measures async toBlob wall time.
const FEPerf={samples:{capture:[],encode:[],render:[]},counts:{dom_writes:0,canvas_resizes:0,canvas_draws:0,video_encoded:0,video_sent:0,video_dropped:0,video_backpressure:0},
 record(k,ms){const a=this.samples[k];if(!a||!Number.isFinite(ms))return;a.push(ms);if(a.length>200)a.shift()},
 snapshot(){const out={...this.counts};for(const[k,a]of Object.entries(this.samples)){const s=[...a].sort((x,y)=>x-y);out[k]={n:a.length,avg:a.length?a.reduce((x,y)=>x+y,0)/a.length:0,p95:s.length?s[Math.min(s.length-1,Math.ceil(s.length*.95)-1)]:0}}return out}};
const HTML_CACHE=new WeakMap();
function writeHTML(id,markup){const e=typeof id==='string'?$(id):id;if(!e)return false;const old=HTML_CACHE.get(e);if(old?.markup===markup&&e.innerHTML===old.html)return false;e.innerHTML=markup;HTML_CACHE.set(e,{markup,html:e.innerHTML});FEPerf.counts.dom_writes++;return true}
function resizeCanvas(c,w,h){w=Math.max(0,Math.round(w));h=Math.max(0,Math.round(h));if(c.width===w&&c.height===h)return false;c.width=w;c.height=h;FEPerf.counts.canvas_resizes++;return true}
function canvasVisible(c){if(document.hidden)return false;const r=c.getBoundingClientRect();return r.width>0&&r.height>0&&r.bottom>0&&r.right>0&&r.top<(globalThis.innerHeight||Infinity)&&r.left<(globalThis.innerWidth||Infinity)}
function measureRender(fn){const t0=performance.now();try{return fn()}finally{FEPerf.record('render',performance.now()-t0)}}
