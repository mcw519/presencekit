// Side panels: speech recognition, cross-session re-ID card, API-key rows, event log, error banner.
// Classic script; all files share one global scope and load in order from index.html.
const ASR=[];
// Caption overlay: one line, updated word by word (partials overwrite in place), new turns fade in, finals stay then fade.
// Lives outside the canvas, so mirroring never flips it
const CAP_HOLD_MS=2500;let capTimer=null,capKey=null;
function caption(m){const el=$('cap');if(!el)return;if(cfg.asr_provider==='none'||!m||!m.text){el.hidden=true;return}
 const key=m.session+':'+m.turn_order;if(key!==capKey){capKey=key;el.classList.remove('is-in');void el.offsetWidth}
 el.hidden=false;$('capTxt').textContent=m.text;el.classList.add('is-in');el.classList.toggle('is-final',m.kind==='final');
 clearTimeout(capTimer);capTimer=setTimeout(()=>el.classList.remove('is-in'),m.kind==='final'?CAP_HOLD_MS:CAP_HOLD_MS*3)}
function captionOff(){const el=$('cap');if(el){el.classList.remove('is-in');el.hidden=true}capKey=null}
function onAsr(m){if(m.kind==='status'&&m.status==='terminated')captionOff();if(m.kind==='status'){$('asrSt').textContent=m.status==='begin'?t('asr.begin',{m:m.model?' · '+m.model:''}):['connected','terminated'].includes(m.status)?t('asr.'+m.status):'';return}
 if(m.kind==='error'){$('asrSt').textContent=t('asr.error',{e:m.error});return}
 const key=m.session+':'+m.turn_order;let e=ASR.find(x=>x.key===key);if(!e){e={key};ASR.push(e);if(ASR.length>100)ASR.shift()}Object.assign(e,m);
 caption(m);
 $('asrList').innerHTML=ASR.slice().reverse().map(x=>`<div style="padding:3px 0;border-bottom:1px solid var(--line);${x.kind==='partial'?'color:var(--mut)':''}"><small style="color:var(--mut)">#${x.session} · ${x.start_s!=null?x.start_s.toFixed(1)+'–'+(x.end_s??0).toFixed(1)+' s':''}${x.language?' · '+x.language:''}</small> ${x.text.replace(/[<&]/g,c=>c==='<'?'&lt;':'&amp;')}</div>`).join('')}
// Cross-session re-ID: decision, calls and estimated cost; comparison thumbnails only with "show comparison photos" (memory only)
let REID={};const RV=[];const RV_MAX=4;
function clearReidDisplay(session=REID.session){REID={};if(Number.isInteger(session))REID.session=session;rvClear();$('reidSt').innerHTML='';$('reidSt').hidden=true}
function onReidSession(session){if(!Number.isInteger(session)||session<0)return true;
 if(Number.isInteger(REID.session)&&session<REID.session)return false;
 if(REID.session!==session)clearReidDisplay(session);return true}
const RVR=r=>{const k='reid.r.'+r,s=t(k);return s===k?r:s};
function rvClear(){RV.length=0;renderRV()}
function renderRV(){const card=$('reidCard');card.hidden=!(cfg.reid_enabled&&cfg.reid_show_crops);if(card.hidden){RV.length=0}
 const img=(b,l)=>b?`<img src="data:image/jpeg;base64,${b}" alt="${l}" width="72" height="72">`:`<span class="rv-ph">${l}<br>${t('rv.none')}</span>`;
 $('reidList').innerHTML=RV.length?RV.map(v=>`<div class="rv-row"><figure>${img(v.probe,t('rv.probe'))}<figcaption>${t('rv.probe_n',{n:v.session})}</figcaption></figure><span class="rv-vs">↔</span><figure>${img(v.ref,t('rv.ref'))}<figcaption>${v.ref_session!=null?t('rv.ref_n',{n:v.ref_session}):t('rv.ref')}</figcaption></figure>
  <div class="rv-info"><b class="${v.decision==='resume'?'ok':''}">${v.decision==='resume'?t('reid.resume',{n:v.resumed_from_session}):t('reid.new')}</b>
  <span>${t('rv.score',{s:`<code>${v.score!=null?v.score.toFixed(2):'—'}</code>`,t:`<code>${(+v.threshold).toFixed(2)}</code>`})}</span>
  <span>${RVR(v.reason)} · ${new Date(v.t*1000).toLocaleTimeString()}</span></div></div>`).join(''):`<p class="panel-help">${t('rv.empty')}</p>`}
function onReidView(m){if(m.type==='reid_view_clear')return clearReidDisplay();if(!cfg.reid_enabled||!cfg.reid_consent||!cfg.reid_show_crops||!onReidSession(m.session))return;RV.unshift(m);RV.length=Math.min(RV.length,RV_MAX);renderRV()}
function onReid(m){const el=$('reidSt');if(!onReidSession(m.session))return;
 if(!cfg.reid_enabled||!cfg.reid_consent||m.enabled===false){clearReidDisplay();return}
 if(m.type==='handoff'){REID.handoff=m}else REID.st=m;
 const st=REID.st||{},h=REID.handoff;el.hidden=false;
 const dec=st.phase==='await_query'||st.phase==='calling'||st.phase==='await_retry'?`<b>${t('reid.identifying')}</b>`:h?(h.decision==='resume'?`<b class="ok">${t('reid.resume',{n:h.resumed_from_session})}</b>${h.score!=null?' ('+h.score.toFixed(2)+')':''}`:`<b>${t('reid.new')}</b> <small>${RVR(h.reason)}</small>`):'—';
 el.innerHTML=`<span>${t('reid.line')}</span>${dec}${h?` <code>${h.conversation_id.slice(0,8)}</code>`:''}<span class="sp">${t('reid.stats',{c:st.calls||0,t:st.tokens||0,cost:st.cost_est?t('reid.cost',{c:st.cost_est}):'',r:st.refs||0,ttl:st.refs&&Number.isFinite(st.ttl_left)?t('reid.ttl',{m:Math.round((st.ttl_left||0)/60)}):''})}</span>`;
 if(st.key_set!=null&&$('reidKeySt')&&$('reidKeySt').dataset.src!=='env')keyBadge('reidKey',{set:st.key_set});renderRV()}
function keyRow(mod,id,label,tip,extra=''){const det=$('mod_'+mod);if(!det)return false;const host=det.querySelector('.mbody')||det;
 host.insertAdjacentHTML('beforeend',`<div class="row keyrow"><div class="top"><label for="${id}In">${label}</label><span id="${id}St" class="kbadge">—</span></div>
  <input type="password" id="${id}In" class="control" autocomplete="off" placeholder="${t('key.placeholder')}">
  <div class="btns"><button class="button button-secondary button-small" id="${id}Set" type="button">${t('key.btn_set')}</button><button class="button button-ghost button-small" id="${id}Clr" type="button">${t('key.btn_clear')}</button>${extra}</div>
  <div class="tip">${tip}</div></div>`);return true}
function keyBadge(id,j){const e=$(id+'St');if(!e)return;e.textContent=j.set?(j.source==='env'?t('key.env'):t('key.set')):t('key.unset');e.dataset.src=j.set&&j.source==='env'?'env':'';e.classList.toggle('is-set',!!j.set)}
function wireKey(id,api){const post=k=>fetch('/api/'+api+'_key'+location.search,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k})}).then(r=>r.json()).then(j=>keyBadge(id,j));
 $(id+'Set').onclick=()=>{const v=$(id+'In').value;$(id+'In').value='';post(v).then(()=>toast(t('toast.key_set')))};$(id+'Clr').onclick=()=>post('');
 fetch('/api/'+api+'_status'+location.search).then(r=>r.json()).then(j=>keyBadge(id,j)).catch(()=>{})}
function addReidKey(){if(!keyRow('reid','reidKey',t('reid.key_label'),t('reid.key_tip'),
  `<button class="button button-ghost button-small" id="reidWipe" type="button">${t('reid.wipe')}</button>`))return;
 wireKey('reidKey','reid');
 $('reidWipe').onclick=()=>{if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'reid_wipe'}));rvClear();toast(t('toast.wiped'))}}
function addAsrKey(){if(!keyRow('asr','asrKey','AssemblyAI API key',t('asr.key_tip')))return;
 wireKey('asrKey','asr')}
$('reidViewClr').onclick=rvClear;

const EVL=[];function addEv(s){EVL.push(s);if(EVL.length>200)EVL.shift();const ul=document.getElementById('evList');const li=document.createElement('li');li.className=s.level||'info';
const kv=Object.entries(s).filter(([k])=>!['type','level','event','ts','t'].includes(k)).map(([k,v])=>k+'='+(typeof v==='object'?JSON.stringify(v):v)).join(' ');
li.textContent=new Date().toLocaleTimeString()+' '+(s.event||s.type)+' '+kv;ul.prepend(li);while(ul.children.length>200)ul.lastChild.remove();document.getElementById('evCount').textContent=EVL.length;document.getElementById('navEvCount').textContent=EVL.length}
function showErr(m){document.getElementById('errText').textContent=t('err.recovered',{m});document.getElementById('errBanner').style.display='block'}
document.getElementById('errClose').onclick=()=>document.getElementById('errBanner').style.display='none';
