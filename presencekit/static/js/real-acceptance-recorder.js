// Optional, operator-loaded observer. Never requests devices, changes config or sends network traffic.
(() => {
  'use strict';
  if (globalThis.RealAcceptanceRecorder) throw new Error('Recorder already loaded');
  let run=null;
  const finite=x=>typeof x==='number'&&Number.isFinite(x), mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
  const p95=a=>a.length?[...a].sort((a,b)=>a-b)[Math.ceil(a.length*.95)-1]:null;
  function unbind(r){if(r.bound){r.bound.removeEventListener('message',r.message);r.bound.removeEventListener('close',r.closed)}r.bound=null}
  function bind(r){const current=typeof ws==='undefined'?null:ws;if(current===r.bound)return;
    if(r.bound)r.connectionChanges++;unbind(r);if(!current)return;r.bound=current;
    current.addEventListener('message',r.message);current.addEventListener('close',r.closed)}
  function finish(r){if(!r.active)return;r.active=false;r.end=performance.now();clearInterval(r.timer);unbind(r);
    removeEventListener('error',r.error);removeEventListener('unhandledrejection',r.error)}
  function observe(r,event){if(!r.active||typeof event.data!=='string')return;
    let m;try{m=JSON.parse(event.data)}catch{r.errors++;return}const now=performance.now();
    if(m.type==='audio'&&m.available===true){const a={};
      for(const k of ['rtf_recent','queue','dropped','send_dropped'])a[k]=finite(m[k])?m[k]:null;
      if(r.audio&&(a.dropped<r.audio.dropped||a.send_dropped<r.audio.send_dropped))r.counterResets++;r.audio=a;r.audioAt=now;
    }else if(m.type==='stats'){r.rss=finite(m.system?.proc_rss)?m.system.proc_rss:null;r.statsAt=now;
    }else if(m.type==='state'&&m.running===false){r.captureStops++;
    }else if(m.type==='error'||(m.type==='event'&&['error','critical'].includes(m.level))){r.errors++}
    if(m.type==='event'&&m.event==='reid_call'){
      r.jevCalls++;if(finite(m.in_tok)&&m.in_tok>=0)r.jevTokens+=m.in_tok;
      if(finite(m.latency_ms)&&m.latency_ms>=0){if(r.jevLatency.length<6000)r.jevLatency.push(m.latency_ms);else r.overflow=true}
      if(typeof m.model==='string'&&/^jev-[0-9]+(?:\.[0-9]+){1,3}$/.test(m.model)&&r.models.size<10)r.models.add(m.model);
    }
  }
  function sample(r){if(!r.active)return;bind(r);const now=performance.now(),t=(now-r.measureStart)/1000;
    if(t<0)return;
    if(!r.baseline){r.baseline={dropped:r.audio?.dropped,send_dropped:r.audio?.send_dropped};
      r.baselineValid=finite(r.baseline.dropped)&&finite(r.baseline.send_dropped)}
    if(t<1||Math.floor(t)===r.lastBucket)return;r.lastBucket=Math.floor(t);
    const a=r.audio||{},audioAge=(now-r.audioAt)/1000,statsAge=(now-r.statsAt)/1000;
    const valid=r.bound?.readyState===1&&(typeof running!=='undefined'&&running)&&audioAge<=1&&statsAge<=2
      &&audioAge>=0&&statsAge>=0&&finite(a.rtf_recent)&&a.rtf_recent>=0&&finite(a.queue)&&a.queue>=0&&a.queue<=25
      &&finite(a.dropped)&&a.dropped>=0&&finite(a.send_dropped)&&a.send_dropped>=0&&finite(r.rss)&&r.rss>0;
    if(r.samples.length<3601)r.samples.push({t:+t.toFixed(3),valid:+!!valid,
      rtf_recent:a.rtf_recent??null,queue:a.queue??null,audio_dropped:a.dropped??null,send_dropped:a.send_dropped??null,
      rss_bytes:r.rss??null,audio_age_seconds:finite(audioAge)?+audioAge.toFixed(3):null,
      stats_age_seconds:finite(statsAge)?+statsAge.toFixed(3):null});else r.overflow=true;
    if(t>=r.seconds){r.completed=true;finish(r)}
  }
  function report(){if(!run)throw new Error('No measurement');const r=run,valid=r.samples.filter(x=>x.valid),
    elapsed=Math.max(0,((r.end??performance.now())-r.measureStart)/1000),expected=Math.max(r.seconds,elapsed),
    coverage=valid.length/expected,adequate=r.completed&&r.baselineValid&&valid.length>=3&&coverage>=.95&&!r.overflow,
    continuous=r.connectionChanges===0&&r.disconnections===0&&r.captureStops===0&&r.counterResets===0,
    gate=(ok,actual)=>({status:!adequate?'not_run':ok?'passed':'failed',actual});
    const rt=p95(valid.map(x=>x.rtf_recent)),quarter=Math.max(1,Math.floor(valid.length/4)),
      early=valid.length?mean(valid.slice(0,quarter).map(x=>x.queue)):null,
      late=valid.length?mean(valid.slice(-quarter).map(x=>x.queue)):null,
      peak=valid.length?Math.max(...valid.map(x=>x.queue)):null,
      dropped=valid.length&&r.baselineValid?Math.max(...valid.map(x=>x.audio_dropped))-r.baseline.dropped:null,
      sendDropped=valid.length&&r.baselineValid?Math.max(...valid.map(x=>x.send_dropped))-r.baseline.send_dropped:null,
      first=valid.filter(x=>x.t<=600),last=valid.filter(x=>x.t>r.seconds-600),
      memoryReady=adequate&&r.seconds>=1800&&first.length>=570&&last.length>=570,
      growth=memoryReady?(mean(last.map(x=>x.rss_bytes))-mean(first.map(x=>x.rss_bytes)))/(1<<20):null;
    return {schema_version:1,status:'pending_real',method:'Passive page WebSocket numeric observer; operator review required',
      configured_seconds:r.seconds,warmup_seconds:r.warmup,elapsed_seconds:+elapsed.toFixed(3),
      sample_count:r.samples.length,valid_sample_count:valid.length,sample_coverage:Math.min(1,coverage),
      completed:r.completed,active:r.active,connection_changes:r.connectionChanges,disconnections:r.disconnections,
      capture_stops:r.captureStops,counter_resets:r.counterResets,overflow:r.overflow,
      gates:{coverage:{status:!r.completed?'not_run':adequate?'passed':'failed',actual:Math.min(1,coverage)},
        rtf:gate(rt!==null&&rt<=.5,{p95:rt}),audio_dropped:gate(dropped===0&&sendDropped===0,{dropped,send_dropped:sendDropped}),
        queue:gate(peak<25&&late<=early+1,{early_mean:early,late_mean:late,peak}),
        continuity:gate(continuous,{continuous}),observed_errors:gate(r.errors===0,{count:r.errors}),
        rss_growth:{status:!memoryReady?'not_run':growth<=30?'passed':'failed',actual:{growth_mib:growth}},
        thirty_minute_coverage:{status:!adequate||r.seconds<1800?'not_run':continuous?'passed':'failed',actual:{seconds:r.seconds}}},
      manual_checks:{server_uncaught_errors:'unverified',participant_ground_truth:'unverified',provider_billing:'unverified'},
      observed_jev:{calls:r.jevCalls,input_tokens:r.jevTokens,latency_p95_ms:p95(r.jevLatency),models:[...r.models]},
      limitations:['Observed events are not an authoritative billing or server-error audit.',
        'Numeric gates do not establish device accuracy, participant consent or Jev identity quality.',
        'A short or interrupted run cannot qualify the continuous 30-minute test.'],samples:r.samples.map(x=>({...x}))};
  }
  globalThis.RealAcceptanceRecorder=Object.freeze({
    start({seconds=1800,warmup=10}={}){
      if(run?.active)throw new Error('Measurement already active');
      if(!Number.isInteger(seconds)||seconds<3||seconds>3600||!Number.isInteger(warmup)||warmup<5||warmup>120)throw new Error('Invalid duration');
      if(typeof ws==='undefined'||ws?.readyState!==1||typeof running==='undefined'||!running)throw new Error('Start the production page first');
      const r={active:true,completed:false,seconds,warmup,measureStart:performance.now()+warmup*1000,end:null,lastBucket:0,
        audio:null,audioAt:-Infinity,rss:null,statsAt:-Infinity,baseline:null,baselineValid:false,samples:[],bound:null,
        errors:0,connectionChanges:0,disconnections:0,captureStops:0,counterResets:0,overflow:false,
        jevCalls:0,jevTokens:0,jevLatency:[],models:new Set()};
      r.message=e=>observe(r,e);r.closed=()=>{if(r.active)r.disconnections++};r.error=()=>{if(r.active)r.errors++};
      run=r;bind(r);addEventListener('error',r.error);addEventListener('unhandledrejection',r.error);
      r.timer=setInterval(()=>sample(r),1000);return {status:'pending_real',seconds,warmup};
    },
    stop(){if(run)finish(run);return report()},report,
    download(){const data=report(),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');
      a.href=url;a.download='real-acceptance-metrics.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);return {status:'pending_real'};}
  });
})();
