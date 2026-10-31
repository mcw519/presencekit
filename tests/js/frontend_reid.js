// Re-ID display follows the current session and consent; no browser or devices.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const root=path.resolve(__dirname,'../..'),read=n=>fs.readFileSync(path.join(root,'presencekit/static/js',n+'.js'),'utf8');
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,{innerHTML:'',hidden:true,textContent:'',dataset:{},classList:{add(){},remove(){}}});return elements.get(id)};
const sandbox={console,Date,Number,URLSearchParams,performance:{now:()=>0},location:{search:'',protocol:'http:',host:'localhost'},
 $:element,t:k=>k,keyBadge(){},setTimeout(){return 1},clearTimeout(){},save(){},syncUI(){},sendCfg(){},restartFrames(){},draw(){},
 invalidateFrames(){},updateRecommendedFps(){},measureRender:fn=>fn(),render(){}};
sandbox.WebSocket=class{constructor(){this.readyState=1}send(){}};
vm.createContext(sandbox);const run=s=>vm.runInContext(s,sandbox);
run("let cfg={reid_enabled:true,reid_consent:true,reid_show_crops:false};let sendTimer;");
const panels=read('panels');run(panels.slice(panels.indexOf('let REID='),panels.indexOf('function keyRow')));
const settings=read('settings');run(settings.slice(settings.indexOf('function setCfg('),settings.indexOf('function serverCfg(')));
const handoff=session=>({type:'handoff',session,decision:'resume',conversation_id:'previous-conversation',resumed_from_session:1,score:.99});
const deliver=message=>{sandbox.message=message;run('onReid(message)')};
const resumed=session=>{run('cfg.reid_enabled=true;cfg.reid_consent=true');deliver(handoff(session));assert.match(element('reidSt').innerHTML,/reid.resume/);assert.equal(element('reidSt').hidden,false)};
const cleared=()=>{assert.equal(run('REID.handoff'),undefined);assert.equal(element('reidSt').innerHTML,'');assert.equal(element('reidSt').hidden,true)};
// Revoke locally: stale result vanishes before any server acknowledgement.
resumed(2);deliver({type:'reid',session:2,enabled:true,phase:'idle',calls:1,refs:1});
run('setCfg({reid_consent:false})');cleared();
deliver(handoff(2));deliver({type:'reid',session:2,enabled:false,phase:'idle',calls:1,refs:0});cleared();
// Disabling also clears a prior decision despite nonzero historical call counters.
resumed(2);run('setCfg({reid_enabled:false})');cleared();
// A wipe response clears textual handoff and thumbnails, not only the photos.
resumed(2);run("onReidView({type:'reid_view_clear'})");cleared();
// A newer session drops the old result; delayed older messages cannot restore it.
resumed(2);deliver({type:'reid',session:3,enabled:true,phase:'calling',calls:1,refs:1});
assert.equal(run('REID.handoff'),undefined);assert.doesNotMatch(element('reidSt').innerHTML,/previous-conversation|reid.resume/);
deliver(handoff(2));assert.equal(run('REID.handoff'),undefined);
deliver(handoff(3));assert.equal(run('REID.handoff.session'),3);
deliver({type:'reid',session:2,enabled:false,phase:'idle',calls:1});assert.equal(run('REID.handoff.session'),3);
run("cfg.reid_show_crops=true;onReidView({type:'reid_view',session:2,probe:'obsolete'})");assert.equal(run('RV.length'),0);
// Exercise production connection and state routing, including lower counters after restart.
run("let ws,last=null,aMeta=null,wsTries=0,wsDownAt=0,lastRx=0,REPQ=[],rec=null,running=false,DEF={},BOXES=[];const PAGE_CAPABILITY='a'.repeat(32);");
const app=read('app');run(app.slice(app.indexOf('function connect()'),app.indexOf('// ---------- render ----------')));
run('connect()');cleared();
resumed(3);run('ws.onclose()');cleared();
run('connect()');resumed(1); // New server can restart its session numbering.
sandbox.message={type:'state',session:2,detections:[]};run('ws.onmessage({data:JSON.stringify(message)})');cleared();
assert.equal(run('REID.session'),2);
// Import/preset changes also go through the real sendCfg() path.
run(settings.slice(settings.indexOf('function serverCfg()'),settings.indexOf("document.querySelectorAll('#presets")));
resumed(2);run('cfg.reid_consent=false;sendCfg()');cleared();
console.log('re-ID consent, session, wipe and reconnect display regressions passed');
