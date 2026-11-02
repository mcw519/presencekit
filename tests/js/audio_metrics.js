// Audio diagnostics rendering uses processed PCM duration, not transition counts or wall time.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const root=path.resolve(__dirname,'../../presencekit/static'),source=fs.readFileSync(path.join(root,'js/audio.js'),'utf8');
for(const locale of ['en','zh-TW']){
 const ctx={I18N:{},locale,output:'',fps:.5};vm.createContext(ctx);
 const run=s=>vm.runInContext(s,ctx);
 run(fs.readFileSync(path.join(root,`i18n/${locale}.js`),'utf8'));
 run("function t(k,p={}){return I18N[locale][k].replace(/\\{(\\w+)\\}/g,(_,x)=>String(p[x]))};function curFps(){return fps}");
 run(source.slice(source.indexOf('const XC_LABEL='),source.indexOf('const DC=')));
 run(source.slice(source.indexOf('function acousticDurations('),source.indexOf('function renderAudio()')));
 run("let a={counts:{leak:122},mix:.9,gain:1,classified_audio_s:4,decision_duration_s:{leak:1,false_reject:2,false_accept:0,agree:0,silence:1,acoustic_only:0},decision_ratio:{leak:.25,false_reject:.5,false_accept:0,agree:0,silence:.25,acoustic_only:0}}");
 let text=run('acousticDurations(a)');assert.ok(text.includes('2.00 s'));assert.ok(text.includes('50.0%'));assert.ok(!text.includes('122'));
 assert.equal((text.match(/[0-9]+\.[0-9]{2} s/g)||[]).length,6); // All labels, including silence and acoustic-only.
 assert.equal(run('acousticDurations({counts:{leak:122}})'),'—');
 assert.equal(run('acousticDurations({...a,decision_ratio:{...a.decision_ratio,leak:NaN}})'),'—');
 assert.equal(run('acousticDurations({...a,classified_audio_s:Infinity})'),'—');
 run("a.t_client=2370;a.t_vision_client=1000;a.vision_timing={capture_gap_ms:1370,processed_age_ms:1360,receive_to_processed_ms:10,queue_wait_ms:1,effective_age_ms:1370,stale:false}");
 text=run('visionAlignment(a)');assert.ok(text.includes('1.37'));assert.ok(text.includes('0.5 FPS'));assert.ok(text.includes('2.00 s'));assert.ok(text.includes('10.0 ms'));
 run('a.vision_timing.stale=true');assert.ok(run('visionAlignment(a)').includes(run("t('akv.align_stale')")));
 run('a.vision_timing.capture_gap_ms=-100');assert.ok(run('visionAlignment(a)').includes(run("t('akv.align_future_v',{s:'0.10'})")));
 assert.equal(run('visionAlignment({t_client:1,t_vision_client:null})'),'—');
 assert.equal(run('visionAlignment({t_client:NaN,t_vision_client:0})'),'—');
 assert.ok(run('visionAlignment({t_client:1370,t_vision_client:0})').includes('1.37'));
 // Exercise the actual audio panel render with the new server contract and an older server.
 run("let aMeta=a,lastLat=null,WR=[],WC=[];function db(){return '—'};function monLat(){return '—'};function $(id){return {}};function writeHTML(id,html){output=html}");
 run(source.slice(source.indexOf('function renderAudio()'),source.indexOf('const XC_LABEL=')));
 run('renderAudio()');assert.ok(ctx.output.includes('50.0%'));assert.ok(ctx.output.includes('4.00'));
 run('aMeta={counts:{},mix:.9,gain:1};renderAudio()');assert.ok(!ctx.output.includes('NaN'));assert.ok(!ctx.output.includes('Infinity'));
}
console.log('audio duration shares, alignment diagnostics and bilingual panel rendering passed');
