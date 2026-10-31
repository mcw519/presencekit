const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.resolve(__dirname,'../../presencekit/static/js/real-acceptance-recorder.js'),'utf8');
class Socket {
 constructor(){this.readyState=1;this.listeners={message:new Set(),close:new Set()}}
 addEventListener(k,f){this.listeners[k].add(f)}removeEventListener(k,f){this.listeners[k].delete(f)}
 emit(m){for(const f of this.listeners.message)f({data:JSON.stringify(m)})}
 send(){throw Error('Observer must never send')}
 close(){this.readyState=3;for(const f of this.listeners.close)f()}
}
function setup(seconds=10){let now=0,callback=null;const events={error:new Set(),unhandledrejection:new Set()},s={ws:new Socket(),running:true,
 performance:{now:()=>now},setInterval:f=>(callback=f,1),clearInterval:()=>callback=null,
 addEventListener:(k,f)=>events[k].add(f),removeEventListener:(k,f)=>events[k].delete(f),
 fetch:()=>{throw Error('Observer must not fetch')},navigator:{mediaDevices:{getUserMedia(){throw Error('No capture')}}}};
 vm.createContext(s);vm.runInContext(source,s);const recorder=s.RealAcceptanceRecorder;
 assert.equal(s.ws.listeners.message.size,0);recorder.start({seconds,warmup:5});
 function tick(i,{rtf=.3,queue=0,dropped=0,sendDropped=0,rss=300,stats=true,audio=true}={}){
  now=i*1000;if(audio)s.ws.emit({type:'audio',available:true,rtf_recent:rtf,queue,dropped,send_dropped:sendDropped,conversation_id:'PRIVATE',jpeg:'PRIVATE'});
  if(stats)s.ws.emit({type:'stats',system:{proc_rss:rss*(1<<20)},secret:'PRIVATE'});callback?.();
 }
 return {s,recorder,tick,events};
}
// A short run can measure numeric gates but can never accept a real 30-minute test.
{
 const x=setup();for(let i=1;i<=15;i++)x.tick(i);
 const r=x.recorder.report();assert.equal(r.status,'pending_real');assert.equal(r.samples.length,10);
 assert.equal(r.gates.rtf.status,'passed');assert.equal(r.gates.thirty_minute_coverage.status,'not_run');
 assert.equal(r.gates.rss_growth.status,'not_run');assert.equal(r.active,false);
 assert.equal(r.manual_checks.server_uncaught_errors,'unverified');assert.equal(x.s.ws.listeners.message.size,0);
 assert.equal(x.events.error.size,0);assert.ok(!JSON.stringify(r).includes('PRIVATE'));
 assert.throws(()=>x.recorder.start({seconds:2}),/Invalid duration/);
}
// Full wall-time coverage, queue, p95 and first/last ten-minute RSS computations.
{
 const x=setup(1800);for(let i=1;i<=1805;i++)x.tick(i,{rss:i>1205?320:300});
 const r=x.recorder.report();assert.equal(r.valid_sample_count,1800);
 assert.equal(r.gates.thirty_minute_coverage.status,'passed');assert.equal(r.gates.rss_growth.status,'passed');
 assert.equal(r.gates.rss_growth.actual.growth_mib,20);assert.equal(r.status,'pending_real');
}
// Nonzero drops, slow inference, growing queues and observed errors fail their gates.
{
 const x=setup();for(let i=1;i<=15;i++)x.tick(i,{rtf:.6,queue:i>12?3:0,dropped:i>8?1:0,sendDropped:i>8?1:0});
 const r=x.recorder.report();for(const k of ['rtf','audio_dropped','queue'])assert.equal(r.gates[k].status,'failed');
}
// Missing/stale telemetry cannot be turned into passing percentile measurements.
{
 const x=setup();for(let i=1;i<=15;i++)x.tick(i,{stats:i<9,audio:i<9});
 const r=x.recorder.report();assert.equal(r.gates.coverage.status,'failed');assert.equal(r.gates.rtf.status,'not_run');
}
// A reset counter and a transport change invalidate continuity, even with enough fresh samples.
{
 const x=setup(30);for(let i=1;i<=35;i++){
  if(i===15){x.s.ws.close();x.s.ws=new Socket()}
  x.tick(i,{dropped:i<20?2:0});
 }
 const r=x.recorder.report();assert.equal(r.gates.coverage.status,'passed');assert.equal(r.gates.continuity.status,'failed');
 assert.equal(r.disconnections,1);assert.equal(r.connection_changes,1);assert.ok(r.counter_resets>0);
}
// Observe only numeric Jev evidence, never identity, media, aliases or raw event records.
{
 const x=setup();x.s.ws.emit({type:'event',event:'reid_call',latency_ms:25,in_tok:20,model:'jev-1.13.0',conversation_id:'PRIVATE'});
 x.s.ws.emit({type:'event',event:'reid_call',latency_ms:50,in_tok:30,model:'private-model-SECRET'});
 for(let i=1;i<=15;i++)x.tick(i);
 const r=x.recorder.report();assert.equal(r.observed_jev.calls,2);assert.equal(r.observed_jev.input_tokens,50);
 assert.equal(r.observed_jev.latency_p95_ms,50);assert.deepEqual([...r.observed_jev.models],['jev-1.13.0']);
 assert.ok(!JSON.stringify(r).includes('SECRET'));assert.equal(r.status,'pending_real');
}
// Early stop retains evidence but cannot qualify its planned measurement window.
{
 const x=setup(1800);for(let i=1;i<=10;i++)x.tick(i);const r=x.recorder.stop();
 assert.equal(r.gates.coverage.status,'not_run');assert.equal(r.gates.thirty_minute_coverage.status,'not_run');
 assert.equal(r.active,false);assert.equal(x.s.ws.listeners.message.size,0);
}
console.log('real acceptance passive numeric recorder contracts passed');
