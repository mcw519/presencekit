// Audio: AudioWorklet capture, clean-audio playback/monitoring, opt-in client-side recording, tracks view.
// Classic script; all files share one global scope and load in order from index.html.
// ---------- audio: AudioWorklet capture (16 kHz PCM16, 20 ms) -> WS binary; clean audio back -> player ----------
// PRIVACY: audio exists only in memory (worklet buffers, WebSocket), never stored.
const WORKLET=`class Cap extends AudioWorkletProcessor{constructor(){super();this.r=sampleRate/16000;this.p=0;this.prev=0;this.buf=new Int16Array(320);this.n=0}
 process(i){const x=i[0]&&i[0][0];if(!x)return true;let p=this.p;
  // linear-interpolation resampler with a 2-tap pre-average (box low-pass) when downsampling
  for(;p<x.length;p+=this.r){const k=Math.floor(p),f=p-k,a=k>0?x[k-1]:this.prev,b=x[k],c=k+1<x.length?x[k+1]:b;
   let v=this.r>1.5?((a+b)/2*(1-f)+(b+c)/2*f):(b+(c-b)*f);v=Math.max(-1,Math.min(1,v));this.buf[this.n++]=v*32767;
   if(this.n===320){this.port.postMessage(this.buf.slice(0));this.n=0}}
  this.p=p-x.length;this.prev=x[x.length-1];return true}}
registerProcessor('cap',Cap);
// Low-latency player: small fixed jitter buffer (start at 60 ms, target 80 ms). Above 160 ms the oldest audio is
// dropped back to the target, so bursts after network stalls never turn into permanent delay (and clock drift
// between mic and speaker cannot accumulate). Underrun -> silence + re-prebuffer.
class Play extends AudioWorkletProcessor{constructor(){super();this.q=[];this.n=0;this.cur=null;this.i=0;this.r=16000/sampleRate;this.play=false;
  this.START=960;this.TGT=1280;this.MAX=2560;this.under=0;this.drop=0;this.k=0;this.hi=0;
  this.port.onmessage=e=>{this.q.push(e.data);this.n+=e.data.length;
   if(this.n>this.MAX){while(this.n>this.TGT&&this.q.length>1){this.drop+=this.q[0].length;this.n-=this.q.shift().length}}}}
 depth(){return this.n+(this.cur?this.cur.length-this.i:0)}
 process(_,o){const out=o[0][0];if(!this.play&&this.n>=this.START)this.play=true;
  for(let j=0;j<out.length;j++){if(!this.play){out[j]=0;continue}
   if(!this.cur||this.i>=this.cur.length){const rest=this.cur?this.i-this.cur.length:0;this.cur=this.q.shift()||null;if(this.cur)this.n-=this.cur.length;this.i=Math.max(0,rest)}
   if(!this.cur){this.play=false;this.under++;out[j]=0;continue}
   const k=Math.floor(this.i),f=this.i-k,a=this.cur[k],b=k+1<this.cur.length?this.cur[k+1]:(this.q.length?this.q[0][0]:a);out[j]=a+(b-a)*f;this.i+=this.r}
  // slow drain: buffer above target+20 ms for 0.5 s -> drop one chunk (keeps latency near the target after bursts)
  if(this.play&&this.depth()>this.TGT+320){if(++this.hi*out.length>sampleRate/2&&this.q.length){this.drop+=this.q[0].length;this.n-=this.q.shift().length;this.hi=0}}else this.hi=0;
  if(++this.k%40===0)this.port.postMessage({buf_ms:this.depth()/16,under:this.under,drop_ms:this.drop/16,playing:this.play});return true}}
registerProcessor('play',Play);`;
let player=null,rawMon=null,cleanMon=null;const WR=[],WC=[];
async function startAudio(ac){await ac.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET],{type:'application/javascript'})));
 const src=ac.createMediaStreamSource(stream),cap=new AudioWorkletNode(ac,'cap');src.connect(cap);
 const sink=ac.createGain();sink.gain.value=0;cap.connect(sink).connect(ac.destination);// keep the graph pulling, silent
 rawMon=ac.createGain();rawMon.gain.value=0;src.connect(rawMon).connect(ac.destination);
 anRaw=ac.createAnalyser();anRaw.fftSize=256;src.connect(anRaw);
 player=new AudioWorkletNode(ac,'play',{outputChannelCount:[1]});player.port.onmessage=e=>{pst=e.data};cleanMon=ac.createGain();cleanMon.gain.value=0;player.connect(cleanMon).connect(ac.destination);
 anClean=ac.createAnalyser();anClean.fftSize=256;player.connect(anClean);setMonitor(cfg.monitor);
 cap.port.onmessage=e=>{const pcm=e.data;meterPush(WR,pcm);
  if(cfg.audio_pipeline!=='puresound'||!ws||ws.readyState!==1)return;
  const b=new ArrayBuffer(9+pcm.byteLength),dv=new DataView(b);dv.setUint8(0,1);dv.setFloat64(1,performance.now(),true);new Uint8Array(b,9).set(new Uint8Array(pcm.buffer,pcm.byteOffset,pcm.byteLength));ws.send(b)}}
function setMonitor(m){cfg.monitor=m;if(rawMon)rawMon.gain.value=m==='raw'?1:0;if(cleanMon)cleanMon.gain.value=m==='clean'?1:0;document.querySelectorAll('#monBtns button').forEach(b=>{const on=b.dataset.m===m;b.classList.toggle('is-active',on);b.setAttribute('aria-checked',String(on))})}
document.querySelectorAll('#monBtns button').forEach(b=>b.onclick=()=>{setMonitor(b.dataset.m);if(b.dataset.m!=='off')toast(t('toast.monitoring'))});
// ---------- optional client-side recording (opt-in, default off) ----------
// PRIVACY: the three aligned tracks live only in this page's memory and are saved to the user's own computer as a
// zip download. The server only streams them while recording is on; it never writes audio.
const REC_MAX_S=600;let rec=null;
function recToggle(){if(rec)return recStop();
 if(!running){toast(t('toast.press_start'));return}if(cfg.audio_pipeline!=='puresound'){toast(t('toast.rec_needs_ps'));return}
 rec={t0:new Date(),n:0,raw:[],vi:[],out:[],blocks:[],states:[],settings:JSON.parse(JSON.stringify(cfg)),dropped0:aMeta?aMeta.dropped:0};
 wsSend({type:'record',on:true});recUI();rec.timer=setInterval(recUI,500)}
function onRec(dv,buf){if(!rec)return;const n=dv.getUint32(9,true),o=13;
 rec.out.push(new Int16Array(buf.slice(o,o+2*n)));rec.vi.push(new Int16Array(buf.slice(o+2*n,o+4*n)));rec.raw.push(new Int16Array(buf.slice(o+4*n,o+6*n)));rec.n+=n;
 if(rec.n>=REC_MAX_S*16000){recStop();toast(t('toast.rec_max'))}}
function recMeta(s){if(!rec)return;const t=+(rec.n/16000).toFixed(3);
 if(s.type==='audio'&&s.available!==false)rec.blocks.push({t,decision:s.decision,why:s.why,speech:s.speech,clean_vad:s.clean_vad,raw_vad:s.raw_vad,gain:s.gain,mix:s.mix,state:s.state,p_group:s.p_group,target_track:s.target_track,distance:s.distance});
 if(s.type==='state'){const l=rec.states[rec.states.length-1];if(!l||l.state!==s.state||l.session!==s.session)rec.states.push({t,state:s.state,session:s.session,participants:(s.tracks||[]).filter(x=>x.participant).length})}}
function recUI(){const on=!!rec;$('recInd').classList.toggle('on',on);$('recBtn').textContent=on?t('rec.stop'):t('rec.start');$('recBtn').classList.toggle('button-danger',on);
 if(on){const sec=rec.n/16000,f=x=>`${Math.floor(x/60)}:${String(Math.floor(x%60)).padStart(2,'0')}`;$('recInd').textContent=t('rec.ind',{t:f(sec),m:f(REC_MAX_S)});$('recSt').textContent=t('rec.st',{t:f(sec),m:f(REC_MAX_S)})}}
function wav(chunks,n){const b=new ArrayBuffer(44+2*n),v=new DataView(b),w=(o,t)=>[...t].forEach((c,i)=>v.setUint8(o+i,c.charCodeAt(0)));
 w(0,'RIFF');v.setUint32(4,36+2*n,true);w(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,16000,true);v.setUint32(28,32000,true);v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,'data');v.setUint32(40,2*n,true);
 let o=44;const u=new Uint8Array(b);for(const c of chunks){u.set(new Uint8Array(c.buffer,c.byteOffset,c.byteLength),o);o+=c.byteLength}return u}
const CRC=(()=>{const t=new Uint32Array(256);for(let i=0;i<256;i++){let c=i;for(let k=0;k<8;k++)c=c&1?0xedb88320^(c>>>1):c>>>1;t[i]=c>>>0}return t})();
function crc32(u){let c=0xffffffff;for(let i=0;i<u.length;i++)c=CRC[(c^u[i])&255]^(c>>>8);return(c^0xffffffff)>>>0}
function zip(files){const parts=[],cd=[];let off=0;const enc=new TextEncoder();
 for(const[name,data]of files){const nm=enc.encode(name),crc=crc32(data),h=new DataView(new ArrayBuffer(30));
  h.setUint32(0,0x04034b50,true);h.setUint16(4,20,true);h.setUint32(14,crc,true);h.setUint32(18,data.length,true);h.setUint32(22,data.length,true);h.setUint16(26,nm.length,true);
  parts.push(new Uint8Array(h.buffer),nm,data);const c=new DataView(new ArrayBuffer(46));c.setUint32(0,0x02014b50,true);c.setUint16(4,20,true);c.setUint16(6,20,true);
  c.setUint32(16,crc,true);c.setUint32(20,data.length,true);c.setUint32(24,data.length,true);c.setUint16(28,nm.length,true);c.setUint32(42,off,true);cd.push(new Uint8Array(c.buffer),nm);off+=30+nm.length+data.length}
 const cdl=cd.reduce((a,x)=>a+x.length,0),e=new DataView(new ArrayBuffer(22));e.setUint32(0,0x06054b50,true);e.setUint16(8,files.length,true);e.setUint16(10,files.length,true);e.setUint32(12,cdl,true);e.setUint32(16,off,true);
 return new Blob([...parts,...cd,new Uint8Array(e.buffer)],{type:'application/zip'})}
function recStop(){const r=rec;if(!r)return;rec=null;clearInterval(r.timer);wsSend({type:'record',on:false});recUI();
 $('recSt').textContent=t('rec.stopping');
 if(!r.n){toast(t('toast.rec_empty'));$('recSt').textContent='';return}
 const p=x=>String(x).padStart(2,'0'),d=r.t0,stamp=`${d.getFullYear()}${p(d.getMonth()+1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`,base='presence-'+stamp;
 const side={format:'16 kHz mono PCM16 WAV',tracks:{raw:'microphone input (PureSound latency compensated, sample-aligned with vi/out)',vi:'PureSound voice isolation output (before the vision cross-check)',out:'final output (after the vision cross-check; the clean monitored track)'},
  started_at:d.toISOString(),duration_s:+(r.n/16000).toFixed(3),samples:r.n,algo_latency_ms:aMeta?aMeta.algo_latency_ms:null,
  server_dropped_chunks:aMeta?aMeta.dropped-r.dropped0:null,note:'t = seconds from the start of the recording (by sample count); blocks dropped by the server are not in the files.',blocks:r.blocks,states:r.states,settings:r.settings};
 const enc=new TextEncoder(),z=zip([[base+'_raw.wav',wav(r.raw,r.n)],[base+'_vi.wav',wav(r.vi,r.n)],[base+'_out.wav',wav(r.out,r.n)],[base+'.json',enc.encode(JSON.stringify(side,null,1))]]);
 const a=document.createElement('a');a.href=URL.createObjectURL(z);a.download=base+'.zip';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),60000);
 $('recSt').textContent=t('rec.done',{f:base,s:(r.n/16000).toFixed(1)});toast(t('toast.rec_done'))}
$('recBtn').onclick=recToggle;
let lastLat=null,pst=null;const LATS=[];
const outLat=()=>ac?((ac.baseLatency||0)+(ac.outputLatency||0))*1000:0;
function onClean(buf){const dv=new DataView(buf);if(dv.getUint8(0)===3)return onRec(dv,buf);if(dv.getUint8(0)!==2)return;lastLat=performance.now()-dv.getFloat64(1,true);LATS.push(lastLat);if(LATS.length>200)LATS.shift();
 const pcm=new Int16Array(buf.slice(9));meterPush(WC,pcm);
 if(player){const f=new Float32Array(pcm.length);for(let i=0;i<pcm.length;i++)f[i]=pcm[i]/32768;player.port.postMessage(f,[f.buffer])}}
function meterPush(arr,pcm){let mx=0;for(let i=0;i<pcm.length;i+=4)mx=Math.max(mx,Math.abs(pcm[i]));arr.push(mx/32768);if(arr.length>250)arr.shift()}
function wave(id,arr,col){const c=$(id),d=devicePixelRatio||1,r=c.getBoundingClientRect();if(!canvasVisible(c))return;resizeCanvas(c,r.width*d,r.height*d);const x=c.getContext('2d'),W=c.width,H=c.height;x.clearRect(0,0,W,H);FEPerf.counts.canvas_draws++;
 x.fillStyle=col;arr.forEach((v,i)=>{const h=Math.max(1,Math.sqrt(v)*H);x.fillRect(i/250*W,(H-h)/2,Math.max(1,W/250),h)})}
const db=a=>{const v=a.length?a[a.length-1]:0;return v>0?(20*Math.log10(v)).toFixed(0)+' dBFS':'—'};
// Tracks view: waveform / spectrogram / both, kept in localStorage; hidden canvases are not redrawn
const TRK_KEY='presence.trackView',TRK=['wave','spec','both'];let trkView=TRK.includes(localStorage.getItem(TRK_KEY))?localStorage.getItem(TRK_KEY):'both';
const showWave=()=>trkView!=='spec',showSpec=()=>trkView!=='wave';
function setTrkView(v){trkView=TRK.includes(v)?v:'both';localStorage.setItem(TRK_KEY,trkView);$('sigStack').dataset.view=trkView;$('stage').dataset.view=trkView;
 $('trkView').querySelectorAll('button').forEach(b=>{const on=b.dataset.v===trkView;b.classList.toggle('is-active',on);b.setAttribute('aria-checked',on)});if(typeof fit==='function')fit()}
$('trkView').addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(b)setTrkView(b.dataset.v)});setTrkView(trkView);
const dbv=a=>{const v=a.length?a[a.length-1]:0;return v>0?(20*Math.log10(v)).toFixed(0):'—'};
setInterval(()=>{if(!running||document.hidden)return;$('dbRaw').textContent=dbv(WR);$('dbClean').textContent=dbv(WC)},250);
setInterval(()=>{if(!running||!showWave())return;wave('wRaw',WR,K.viz_raw);wave('wClean',WC,aMeta&&aMeta.clean_vad?K.ok:K.viz_clean)},100);
function monLat(){if(lastLat==null)return'—';const s=[...LATS].sort((x,y)=>x-y),p=q=>s[Math.min(s.length-1,Math.floor(q*s.length))];
 const buf=pst?pst.buf_ms:0,tot=20+aMeta.algo_latency_ms+p(0.5)+buf+outLat();
 return t('lat.mon',{tot:tot.toFixed(0),a:aMeta.algo_latency_ms,rt:p(0.5).toFixed(0),p95:p(0.95).toFixed(0),b:buf.toFixed(0),o:outLat().toFixed(0)})+(pst?t('lat.mon_under',{u:pst.under,d:pst.drop_ms.toFixed(0)}):'')+t('lat.mon_queue',{q:aMeta.queue,m:aMeta.backlog_max??0})}
// Metadata and recording history are consumed immediately; only the visual update is coalesced.
let audioRenderTimer=null,audioRenderAt=-Infinity,audioRenderPending=false;
function flushAudioRender(){clearTimeout(audioRenderTimer);audioRenderTimer=null;if(!audioRenderPending||document.hidden)return;audioRenderPending=false;audioRenderAt=performance.now();measureRender(renderAudio)}
function queueAudioRender(){audioRenderPending=true;if(document.hidden||audioRenderTimer!==null)return;const delay=Math.max(0,200-(performance.now()-audioRenderAt));if(!delay)flushAudioRender();else audioRenderTimer=setTimeout(flushAudioRender,delay)}
document.addEventListener('visibilitychange',()=>{if(document.hidden){clearTimeout(audioRenderTimer);audioRenderTimer=null}else if(audioRenderPending)queueAudioRender()});
// Ratios use all processed PCM time, including silence/audio-only, never transition counts or wall time.
function acousticDurations(a){const labels=['leak','false_reject','false_accept','agree','silence','acoustic_only'];
 if(!Number.isFinite(a.classified_audio_s)||a.classified_audio_s<0)return'—';
 if(labels.some(k=>!Number.isFinite(a.decision_duration_s?.[k])||a.decision_duration_s[k]<0||!Number.isFinite(a.decision_ratio?.[k])||a.decision_ratio[k]<0||a.decision_ratio[k]>1))return'—';
 return labels.map(k=>t('akv.duration_v',{label:XC_LABEL[k],s:a.decision_duration_s[k].toFixed(2),p:(100*a.decision_ratio[k]).toFixed(1)})).join(' · ')}
function visionAlignment(a){const v=a.vision_timing,gap=Number.isFinite(v?.capture_gap_ms)?v.capture_gap_ms:
 Number.isFinite(a.t_client)&&Number.isFinite(a.t_vision_client)?a.t_client-a.t_vision_client:null;
 if(gap===null)return'—';
 const text=t(gap>=0?'akv.align_v':'akv.align_future_v',{s:(Math.abs(gap)/1000).toFixed(2)}),parts=[text],fps=typeof curFps==='function'?curFps():null;
 if(Number.isFinite(fps)&&fps>0)parts.push(t('akv.align_sampling_v',{fps,interval:(1/fps).toFixed(2)}));
 if(v&&['processed_age_ms','receive_to_processed_ms','queue_wait_ms','effective_age_ms'].every(k=>Number.isFinite(v[k])&&v[k]>=0)){
  parts.push(t('akv.align_timing_v',{phase:(v.processed_age_ms/1000).toFixed(2),server:v.receive_to_processed_ms.toFixed(1),queue:v.queue_wait_ms.toFixed(1),age:(v.effective_age_ms/1000).toFixed(2)}));
  parts.push(t(v.stale?'akv.align_stale':'akv.align_fresh'))}
 return parts.join(' · ')}
function renderAudio(){const a=aMeta;if(!a)return;
 if(a.available===false){$('aSt').textContent=t('a.unavailable');writeHTML('aKv',`<span>${t('a.reason')}</span><span>${a.error}</span>`);return}
 const LB=XC_LABEL;
 $('aSt').textContent=(LB[a.decision]||a.decision)+(a.speech?t('a.speech'):'');
 const kv={[t('akv.decision')]:`${LB[a.decision]||a.decision}${a.why?' ('+a.why+')':''}`,
  [t('akv.counts')]:t('akv.counts_v',{l:a.counts.leak,fr:a.counts.false_reject,fa:a.counts.false_accept,a:a.counts.agree}),
  [t('akv.duration')]:acousticDurations(a),
  [t('akv.duration_total')]:Number.isFinite(a.classified_audio_s)&&a.classified_audio_s>=0?t('akv.duration_total_v',{s:a.classified_audio_s.toFixed(2)}):'—',
  [t('akv.vad')]:`${a.clean_vad?t('akv.speaking'):'—'} / ${a.raw_vad?t('akv.speaking'):'—'}${t('akv.onsets',{n:a.onsets})}${a.vad_source==='energy_on_enhanced'?t('akv.vad_energy'):t('akv.vad_model')}`,
  [t('akv.level')]:t('akv.level_v',{r:a.raw_db,c:a.clean_db,wr:db(WR),wc:db(WC)}),
  [t('akv.gate')]:a.gain<0.5?t('akv.gate_down',{db:(20*Math.log10(Math.max(a.gain,1e-4))).toFixed(0)}):t('akv.gate_pass'),
  [t('akv.mix')]:t('akv.mix_v',{e:(a.mix*100).toFixed(0),o:(100-a.mix*100).toFixed(0),g:a.gain}),
  [t('akv.align')]:visionAlignment(a),
  [t('akv.asr')]:a.asr?t('akv.asr_v',{p:a.asr.provider==='none'?t('akv.asr_off'):a.asr.provider,k:a.asr.key_set?t('key.set'):t('key.unset'),c:a.asr.connected?t('akv.connected'):t('akv.disconnected'),s:a.asr.sent_s}):'—',
  [t('akv.target')]:a.target_track!=null?`#${a.target_track}${a.distance!=null?' · '+a.distance+' m':''}`:'—','P(group)':a.p_group??'—',
  [t('akv.rtf')]:t('akv.rtf_v',{all:a.rtf??'—',recent:a.rtf_recent??'—'}),[t('akv.proc_p95')]:t('akv.proc_p95_v',{p:a.proc_p95_ms??'—',a:a.queue_wait_ms_avg??'—',q:a.queue_wait_ms_p95??'—'}),[t('akv.proc')]:t('akv.proc_v',{ms:a.proc_ms,q:a.queue,d:a.dropped,r:a.resets}),
  [t('akv.latency')]:t('akv.latency_v',{a:a.algo_latency_ms,l:lastLat!=null?lastLat.toFixed(0):'—'}),
  [t('akv.mon_latency')]:monLat()};
 writeHTML('aKv',Object.entries(kv).map(([k,v])=>`<span>${k}</span><span>${v}</span>`).join(''))}
const XC_LABEL={get agree(){return t('xc.agree')},get leak(){return t('xc.leak')},get false_reject(){return t('xc.false_reject')},get false_accept(){return t('xc.false_accept')},get silence(){return t('xc.silence')},get acoustic_only(){return t('xc.acoustic_only')}};
const DC={};
const SC={};pal();
const TLH=[];
function legend(){$('tlLeg').innerHTML=Object.entries(XC_LABEL).map(([k,v])=>[k,v.replace(/^✅ /,'')]).map(([k,v])=>`<span style="color:${DC[k]}">■</span>${v}`).join(' ')}legend();
function drawTL(){const c=$('tl'),d=devicePixelRatio||1,r=c.getBoundingClientRect();if(!canvasVisible(c))return;resizeCanvas(c,r.width*d,r.height*d);const g=c.getContext('2d'),W=c.width,H=c.height,now=performance.now(),rows=5,rh=H/rows;
 g.clearRect(0,0,W,H);FEPerf.counts.canvas_draws++;TLH.forEach((e,i)=>{const n=TLH[i+1];const x0=(e.t-now+30000)/30000*W,x1=((n?n.t:now)-now+30000)/30000*W,w=Math.max(1,x1-x0);
  if(e.raw){g.fillStyle=K.viz_raw;g.fillRect(x0,0,w,rh-1)}if(e.clean){g.fillStyle=K.viz_clean;g.fillRect(x0,rh,w,rh-1)}if(e.sp){g.fillStyle=K.ok;g.fillRect(x0,2*rh,w,rh-1)}
  g.fillStyle=DC[e.dec]||K.viz_empty;g.fillRect(x0,3*rh,w,rh-1);g.fillStyle=SC[e.st]||K.viz_empty;g.fillRect(x0,4*rh,w,rh-1)});
 g.strokeStyle=K.accent;g.lineWidth=1.5*d;g.beginPath();let on=false;TLH.forEach(e=>{if(e.p==null){on=false;return}const x=(e.t-now+30000)/30000*W,y=H-e.p*H;on?g.lineTo(x,y):g.moveTo(x,y);on=true});g.stroke();
 g.fillStyle=K.viz_ink;g.font=`${9*d}px sans-serif`;['raw','clean','speech','decision','state'].forEach((k,i)=>g.fillText(t('tl.'+k),2*d,i*rh+rh*0.75))}
let anRaw=null,anClean=null;const SGB=new Uint8Array(128);
function sg(id,an){if(!an)return;const c=$(id);if(!canvasVisible(c)||c.width<2)return;const g=c.getContext('2d');g.drawImage(c,1,0,c.width-1,c.height,0,0,c.width-1,c.height);g.clearRect(c.width-1,0,1,c.height);FEPerf.counts.canvas_draws++;an.getByteFrequencyData(SGB);
 for(let y=0;y<c.height;y++){const v=SGB[Math.floor((1-y/c.height)*SGB.length*0.5)];g.fillStyle=`hsl(${240-v*0.9},80%,${v/5.5}%)`;g.fillRect(c.width-1,y,1,1)}}
setInterval(()=>{if(!running)return;if(showSpec()){sg('sgRaw',anRaw);sg('sgClean',anClean)}drawTL()},100);
