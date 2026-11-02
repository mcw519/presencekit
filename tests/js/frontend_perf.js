// Browser scheduling/DOM/canvas regressions, without camera or audio devices.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const root=path.resolve(__dirname,'../../presencekit/static/js');
const read=n=>fs.readFileSync(path.join(root,n+'.js'),'utf8');
let now=0,next=0,writes=0,clears=0,draws=0;const timers=new Map(),listeners={};
const context2d=new Proxy({clearRect(){clears++},drawImage(){draws++},getImageData(){throw Error('GPU readback forbidden')}},{get:(o,k)=>o[k]||(()=>{})});
const element={style:{},_html:'',get innerHTML(){return this._html},set innerHTML(x){writes++;this._html=x},width:10,height:10,getContext:()=>context2d,getBoundingClientRect:()=>({width:100,height:40,top:0,left:0,right:100,bottom:40}),classList:{add(){},remove(){}},textContent:''};
const sandbox={console,Blob,ArrayBuffer,DataView,document:{hidden:false,documentElement:{},getElementById:()=>element,querySelectorAll:()=>[],addEventListener:(n,fn)=>{listeners[n]=fn}},localStorage:{getItem:()=>null},matchMedia:()=>({addEventListener(){}}),performance:{now:()=>now},setInterval:(fn,ms)=>{timers.set(++next,{fn,ms});return next},clearInterval:id=>timers.delete(id),setTimeout:(fn,ms)=>{timers.set(++next,{fn,ms});return next},clearTimeout:id=>timers.delete(id),devicePixelRatio:1,innerHeight:800,innerWidth:1000,location:{protocol:'http:',host:'test',search:''},getComputedStyle:()=>({getPropertyValue:()=>''})};
sandbox.URLSearchParams=URLSearchParams;sandbox.crypto=require('node:crypto').webcrypto;
sandbox.navigator={sendBeacon:()=>true};sandbox.addEventListener=(n,fn)=>{listeners[n]=fn};
vm.createContext(sandbox);const run=s=>vm.runInContext(s,sandbox);
run(read('core'));
run("writeHTML('x','same');writeHTML('x','same')");assert.equal(writes,1);element.innerHTML='external';run("writeHTML('x','same')");assert.equal(writes,3);
run("for(let i=0;i<300;i++)FEPerf.record('render',i)");const stats=run('FEPerf.snapshot()');assert.equal(stats.render.n,200);assert.equal(stats.render.avg,199.5);assert.equal(stats.render.p95,289);
run("resizeCanvas($('x'),100,40);resizeCanvas($('x'),100,40)");assert.equal(run('FEPerf.counts.canvas_resizes'),1);
assert.equal(run("canvasVisible($('x'))"),true);sandbox.document.hidden=true;assert.equal(run("canvasVisible($('x'))"),false);sandbox.document.hidden=false;
const app=read('app'),frames=app.slice(app.indexOf('// Only replace the timer'),app.indexOf('function setUI()'));
run(app.slice(app.indexOf('const PAGE_CAPABILITY='),app.indexOf('// Browser-side reports')));
assert.match(run('PAGE_CAPABILITY'),/^[0-9a-f]{32}$/);
run("let cfg={adaptive_fps:true,fps:1,fps_low:.5,fps_high:2},running=true,frameTimer=null;let ws=null,v={videoWidth:0,videoHeight:480};let callbacks=[],sent=[];let off={width:0,height:0,toBlob:(cb,type,q)=>{callbacks.push(cb);if(type!=='image/jpeg'||q!==.7)throw Error('wrong JPEG options')}},og={drawImage(){}};");run(frames);
run('restartFrames()');assert.equal([...timers.values()][0].ms,1000);
run('updateRecommendedFps(2)');assert.equal(timers.size,1);assert.equal([...timers.values()][0].ms,500);
const timerId=next;run('updateRecommendedFps(undefined);updateRecommendedFps(2);restartFrames()');assert.equal(next,timerId);
run('cfg.adaptive_fps=false;restartFrames()');assert.equal([...timers.values()][0].ms,1000);
run('running=false;restartFrames()');assert.equal(timers.size,0);run('running=true;restartFrames()');assert.equal(timers.size,1);timers.clear();
// Actual socket message routing must consume stats recommendations and preserve missing state fields.
run("let rec=null,TLH=[],aMeta=null,last=null,lastRtt=null;let recMeta=()=>{},render=()=>{},renderStats=()=>{},t=x=>x,sendCfg=()=>{},report=()=>{};");
// Run production stop(), then restart capture: cleared interval IDs must not suppress a new timer.
run("let vadTimer=null,pingTimer=null,stream=null,ac=null,player=null,rawMon=null,cleanMon=null,anRaw=null,anClean=null,speech=false,autoPaused=false;function setUI(){}function draw(){}function wsSend(){}");
run(app.slice(app.indexOf('function stop(auto)'),app.indexOf("$('start').onclick=")));
run('running=false;restartFrames();running=true;restartFrames()');const beforeStop=next;assert.equal(timers.size,1);
run('stop()');assert.equal(timers.size,0);assert.equal(run('frameTimer'),null);assert.equal(run('frameRate'),0);assert.equal(run('recFps'),0);
run('running=true;restartFrames()');assert.equal(timers.size,1);assert.ok(next>beforeStop);assert.equal([...timers.values()][0].ms,1000);
// Extract only connect(), retaining its production routing and closures.
sandbox.clearReidDisplay=()=>{};sandbox.onReidSession=()=>{};
const connect=app.slice(app.indexOf('function connect()'),app.indexOf('// ---------- render ----------'));
sandbox.WebSocket=class{constructor(url,protocols){sandbox.wsUrl=url;sandbox.wsProtocols=protocols;this.readyState=1;this.bufferedAmount=0}send(x){sandbox.sentForTest.push(x)}};sandbox.sentForTest=[];
run("let wsTries=0,wsDownAt=0,lastRx=0,REPQ=[],DEF={},BOXES=[];");run(connect);run('connect();cfg.adaptive_fps=true;ws.onmessage({data:JSON.stringify({type:"stats",conn:{optim:{recommended_fps:.5}}})})');assert.equal(run('curFps()'),.5);assert.ok(!sandbox.wsUrl.includes('page_token'));assert.equal(sandbox.wsProtocols[0],'presence-v1');assert.equal(sandbox.wsProtocols[1],'page-'+run('PAGE_CAPABILITY'));
run('ws.onmessage({data:JSON.stringify({type:"state",detections:[]})})');assert.equal(run('curFps()'),.5);assert.ok(!sandbox.wsUrl.includes('page_token'));assert.equal(sandbox.wsProtocols[0],'presence-v1');assert.equal(sandbox.wsProtocols[1],'page-'+run('PAGE_CAPABILITY'));
// Async JPEG encoding: raw camera pixels, one encode in flight and stale callback guards.
run("v.videoWidth=640;speech=true;captureFrame();captureFrame()");assert.equal(run('callbacks.length'),1);
assert.equal(run('FEPerf.counts.video_dropped'),1);
run("callbacks.shift()(new Blob([new Uint8Array([255,216,255,217])],{type:'image/jpeg'}))");
assert.equal(sandbox.sentForTest.length,1);assert.ok(sandbox.sentForTest[0] instanceof Blob);assert.equal(sandbox.sentForTest[0].size,14);
// Verify header bytes asynchronously below, after all synchronous regression checks.
const packet=sandbox.sentForTest[0];sandbox.sentForTest.length=0;
run('captureFrame();stop();running=true;restartFrames();captureFrame()');assert.equal(run('callbacks.length'),1);
run('callbacks.shift()(new Blob(["old"]));captureFrame()');assert.equal(sandbox.sentForTest.length,0);assert.equal(run('callbacks.length'),1);
run('connect();callbacks.shift()(new Blob(["oldsocket"]))');assert.equal(sandbox.sentForTest.length,0);
run('ws.bufferedAmount=VIDEO_BACKPRESSURE_BYTES+1;captureFrame()');assert.equal(run('callbacks.length'),0);
run('ws.bufferedAmount=0;captureFrame();ws.bufferedAmount=VIDEO_BACKPRESSURE_BYTES+1;callbacks.shift()(new Blob(["late"]))');assert.equal(sandbox.sentForTest.length,0);
run('ws.bufferedAmount=0;captureFrame();callbacks.shift()(null);captureFrame();ws.readyState=3;callbacks.shift()(new Blob(["closed"]))');assert.equal(sandbox.sentForTest.length,0);
run('ws.readyState=1;cfg.adaptive_fps=true;updateRecommendedFps(1);captureFrame();updateRecommendedFps(2);callbacks.shift()(new Blob(["oldrate"]))');assert.equal(sandbox.sentForTest.length,0);
run('ws.readyState=1;off.toBlob=()=>{throw Error("encode failure")};captureFrame()');assert.equal(run('frameEncode'),null);
assert.ok(!frames.includes('toDataURL'));assert.ok(!frames.includes('scale(-1'));assert.ok(!frames.includes('MIR()'));
run('cfg.adaptive_fps=true;updateRecommendedFps(1);captureFrame()');assert.equal(run('frameEncode'),null);
// 10 Hz metadata is reduced to 5 Hz visual work, latest state wins, hidden pending work resumes.
const audio=read('audio');run(audio.slice(audio.indexOf('let audioRenderTimer='),audio.indexOf('function renderAudio()')));
run('let rendered=[];function renderAudio(){rendered.push(aMeta.value)}');timers.clear();
for(let i=0;i<10;i++){now=i*100;run(`aMeta={value:${i}};queueAudioRender()`);for(const[id,t]of [...timers])if(t.ms<=100&&i%2===0){timers.delete(id);t.fn()}}
run('flushAudioRender()');assert.ok(run('rendered.length')<=6);assert.equal(run('rendered.at(-1)'),9);
sandbox.document.hidden=true;run('aMeta={value:99};queueAudioRender();flushAudioRender()');assert.equal(run('rendered.at(-1)'),9);sandbox.document.hidden=false;run('flushAudioRender()');assert.equal(run('rendered.at(-1)'),99);
// Canvas redraw keeps backing store stable and explicitly clears old waveform pixels.
run("let SGB=new Uint8Array(128);");run(audio.slice(audio.indexOf('function wave('),audio.indexOf('const db=')));
const resizeBefore=run('FEPerf.counts.canvas_resizes');run("wave('x',[.5],'red');wave('x',[],'red')");assert.equal(run('FEPerf.counts.canvas_resizes'),resizeBefore);assert.equal(clears,2);
run(audio.slice(audio.indexOf('function sg('),audio.lastIndexOf('setInterval(')));run("sg('x',{getByteFrequencyData(){}})");assert.equal(draws,1);assert.equal(clears,3);
run('rec={};let recordedMessages=0;recMeta=()=>recordedMessages++;TLH.length=0');
for(let i=0;i<25;i++){now+=10;run(`ws.onmessage({data:JSON.stringify({type:'audio',value:${i},available:true})})`)}
assert.equal(run('recordedMessages'),25);assert.equal(run('TLH.length'),25);run('flushAudioRender()');assert.equal(run('rendered.at(-1)'),24);
packet.arrayBuffer().then(buf=>{const v=new DataView(buf);assert.equal(v.getUint8(0),4);assert.equal(v.getFloat64(1,true),0);assert.equal(v.getUint8(9),1);assert.deepEqual([...new Uint8Array(buf,10)],[255,216,255,217]);console.log('frontend scheduling, metrics, DOM/canvas and binary JPEG regressions passed')}).catch(e=>{console.error(e);process.exitCode=1});
