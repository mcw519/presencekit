// Opt-in recording of the camera and its overlays, assembled entirely in the browser.
const REC_MAX_S=600,REC_FPS=24,REC_MAX_EDGE=960,REC_SAMPLE_RATE=16000;
const REC_FORMATS=['video/mp4;codecs=avc1.42001f,mp4a.40.2','video/webm;codecs=vp8,opus','video/mp4','video/webm'];
const REC_TRACKS=['out','vi','raw']; // order in the PureSound 0x03 packet
let rec=null,recUrl=null,recArchive=null;

function requestRecTracks(){
 if(rec?.tracks&&rec.stopAt===null)wsSend({type:'record',on:true});
}

function onRecTracks(packet){
 const tracks=rec?.tracks;
 if(!tracks||rec.stopAt!==null||packet.byteLength<13)return;
 // The outgoing audio timestamp uses the same browser clock. Ignore late packets from before this clip.
 const capturedAt=packet.getFloat64(1,true);
 if(!Number.isFinite(capturedAt)||capturedAt<rec.startAt)return;
 const n=packet.getUint32(9,true),size=2*n;
 if(!n||packet.byteLength!==13+size*REC_TRACKS.length||tracks.samples+n>REC_MAX_S*REC_SAMPLE_RATE)return;
 REC_TRACKS.forEach((name,i)=>{
  const offset=packet.byteOffset+13+i*size;
  tracks[name].push(packet.buffer.slice(offset,offset+size));
 });
 tracks.samples+=n;
}

function recTrackMeta(s){
 const tracks=rec?.tracks;
 if(!tracks||rec.stopAt!==null)return;
 const t=+(tracks.samples/REC_SAMPLE_RATE).toFixed(3);
 if(s.type==='audio'&&s.available!==false){
  const {decision,why,speech,clean_vad,raw_vad,gain,mix,state,p_group,target_track,distance,recovery}=s;
  tracks.blocks.push({t,decision,why,speech,clean_vad,raw_vad,gain,mix,state,p_group,target_track,distance,recovery});
 }else if(s.type==='state'){
  const prev=tracks.states.at(-1);
  if(!prev||prev.state!==s.state||prev.session!==s.session){
   tracks.states.push({t,state:s.state,session:s.session,participants:(s.tracks||[]).filter(x=>x.participant).length});
  }
 }
}

function drawRecordedFrame(r){
 const {ctx,canvas:{width:w,height:h}}=r;
 ctx.save();
 if(MIR()){ctx.translate(w,0);ctx.scale(-1,1)}
 ctx.drawImage(v,0,0,w,h);
 ctx.restore();
 drawOverlay(ctx,w,h,w/640);
}

function recUI(){
 const active=!!rec&&rec.stopAt===null,button=$('recBtn');
 $('recInd').classList.toggle('on',active);
 button.textContent=t(active?'rec.stop':'rec.start');
 button.classList.toggle('button-danger',active);
 button.disabled=!!rec&&!active;
 if(active){
  const format=sec=>`${Math.floor(sec/60)}:${String(sec%60).padStart(2,'0')}`;
  const clock={t:format(Math.floor((performance.now()-rec.startAt)/1000)),m:format(REC_MAX_S)};
  $('recInd').textContent=t('rec.ind',clock);
  $('recSt').textContent=t('rec.st',clock);
 }
}

function releaseRecording(r){
 r.videoTrack?.stop();
 if(r.destination){
  // Detach only this recording output. The live player and its monitor still use the node.
  try{r.source.disconnect(r.destination)}catch{}
  r.audioTrack?.stop();
 }
}

function recToggle(){
 if(rec){recStop();return}
 if(!running||v.readyState<2||!v.videoWidth||!v.videoHeight){toast(t('toast.rec_wait'));return}
 const canvas=document.createElement('canvas');
 if(typeof MediaRecorder==='undefined'||!canvas.captureStream){toast(t('toast.rec_unsupported'));return}
 const pipeline=cfg.audio_pipeline,clean=pipeline==='puresound';
 const mic=stream?.getAudioTracks().find(track=>track.readyState==='live');
 if(clean?(!player||!ac||aMeta?.available===false):!mic){toast(t('toast.rec_audio'));return}
 const scale=Math.min(1,REC_MAX_EDGE/Math.max(v.videoWidth,v.videoHeight));
 canvas.width=Math.max(2,2*Math.round(v.videoWidth*scale/2));
 canvas.height=Math.max(2,2*Math.round(v.videoHeight*scale/2));
 const r={canvas,ctx:canvas.getContext('2d'),pipeline,chunks:[],stamp:new Date(),stopAt:null,failed:false};
 if(clean){
  const settings=Object.fromEntries(Object.keys(DEF).map(key=>[key,cfg[key]]));
  r.tracks={samples:0,raw:[],vi:[],out:[],blocks:[],states:[],settings:JSON.parse(JSON.stringify(settings)),dropped:aMeta?.dropped};
 }
 try{
  const media=canvas.captureStream(REC_FPS);
  r.videoTrack=media.getVideoTracks()[0];
  if(!r.videoTrack)throw new Error('Missing canvas video track');
  if(clean){
   r.source=player;
   r.destination=ac.createMediaStreamDestination();
   r.audioTrack=r.destination.stream.getAudioTracks()[0];
   player.connect(r.destination);
  }else r.audioTrack=mic;
  media.addTrack(r.audioTrack);
  const mimeType=REC_FORMATS.find(m=>MediaRecorder.isTypeSupported(m));
  r.recorder=new MediaRecorder(media,{videoBitsPerSecond:1500000,audioBitsPerSecond:96000,...(mimeType?{mimeType}:{})});
  r.recorder.ondataavailable=e=>{if(e.data.size)r.chunks.push(e.data)};
  r.recorder.onerror=()=>{r.failed=true;if(rec===r)recStop()};
  r.recorder.onstop=()=>recFinish(r);
  drawRecordedFrame(r);
  r.startAt=performance.now();
  r.recorder.start(1000);
  rec=r;
  requestRecTracks();
  r.frames=setInterval(()=>{
   if(cfg.audio_pipeline!==r.pipeline){recStop();toast(t('toast.rec_pipeline'));return}
   if(v.readyState>=2)drawRecordedFrame(r);
  },1000/REC_FPS);
  r.clock=setInterval(recUI,500);
  r.limit=setTimeout(()=>{recStop();toast(t('toast.rec_max'))},REC_MAX_S*1000);
  recUI();
 }catch{
  releaseRecording(r);
  toast(t('toast.rec_failed'));
 }
}

function recStop(){
 const r=rec;
 if(!r||r.stopAt!==null)return;
 r.stopAt=performance.now();
 if(r.tracks)wsSend({type:'record',on:false});
 clearInterval(r.frames);clearInterval(r.clock);clearTimeout(r.limit);
 recUI();
 $('recSt').textContent=t('rec.stopping');
 if(r.recorder.state!=='inactive')r.recorder.stop();
}

function recFinish(r){
 if(rec!==r)return;
 recStop();
 rec=null;
 releaseRecording(r);
 recUI();
 if(r.failed||!r.chunks.length){
  const key=r.failed?'toast.rec_failed':'toast.rec_empty';
  $('recSt').textContent=t(key);
  toast(t(key));
  return;
 }
 const type=r.chunks[0].type||r.recorder.mimeType;
 const suffix=type.startsWith('video/mp4')?'mp4':'webm';
 const stamp=r.stamp.toISOString().slice(0,19).replace(/:/g,'-');
 const base=`presence-${stamp}`,filename=`${base}.${suffix}`;
 const video=new Blob(r.chunks,{type}),url=URL.createObjectURL(video);
 const preview=$('recVideo'),download=$('recDownload');
 preview.src=url;
 download.href=url;
 download.download=filename;
 $('recResult').hidden=false;
 if(recUrl)URL.revokeObjectURL(recUrl);
 recUrl=url;
 if(recArchive?.url)URL.revokeObjectURL(recArchive.url);
 const tracks=r.tracks,duration=(r.stopAt-r.startAt)/1000;
 recArchive=null;
 if(tracks?.samples){
  recArchive={base,filename,video,tracks,metadata:{
   format:'16 kHz mono PCM16 WAV',
   tracks:{raw:'Microphone input aligned to the PureSound output',vi:'PureSound isolation output',out:'Final cross-checked output'},
   video:{file:filename,duration_s:+duration.toFixed(3),audio:'PureSound final output'},
   started_at:r.stamp.toISOString(),duration_s:+(tracks.samples/REC_SAMPLE_RATE).toFixed(3),samples:tracks.samples,
   algo_latency_ms:aMeta?.algo_latency_ms??null,
   server_dropped_chunks:Number.isFinite(aMeta?.dropped-tracks.dropped)?Math.max(0,aMeta.dropped-tracks.dropped):null,
   note:'WAV tracks use the same sample clock. Blocks dropped by the server are absent. Video uses the browser clock.',
   blocks:tracks.blocks,states:tracks.states,settings:tracks.settings,
  }};
 }
 const bundle=$('recDownloadZip');
 bundle.hidden=!recArchive;
 bundle.disabled=false;
 bundle.textContent=t('rec.download_zip');
 $('recSt').textContent=t('rec.done',{s:duration.toFixed(1)})+(tracks&&!tracks.samples?' '+t('rec.no_tracks'):'');
}

function recordingWav(chunks,n){
 const header=new DataView(new ArrayBuffer(44));
 const text=(offset,tag)=>[...tag].forEach((ch,i)=>header.setUint8(offset+i,ch.charCodeAt(0)));
 text(0,'RIFF');header.setUint32(4,36+2*n,true);text(8,'WAVEfmt ');
 header.setUint32(16,16,true);header.setUint16(20,1,true);header.setUint16(22,1,true);
 header.setUint32(24,REC_SAMPLE_RATE,true);header.setUint32(28,2*REC_SAMPLE_RATE,true);
 header.setUint16(32,2,true);header.setUint16(34,16,true);text(36,'data');header.setUint32(40,2*n,true);
 return new Blob([header.buffer,...chunks],{type:'audio/wav'});
}

const REC_CRC=(()=>{
 const table=new Uint32Array(256);
 for(let i=0;i<256;i++){
  let c=i;
  for(let k=0;k<8;k++)c=c&1?0xedb88320^(c>>>1):c>>>1;
  table[i]=c;
 }
 return table;
})();
async function recordingCrc32(blob){
 const reader=blob.stream().getReader();
 let crc=0xffffffff;
 try{
  for(;;){
   const {value,done}=await reader.read();
   if(done)return (crc^0xffffffff)>>>0;
   for(const b of value)crc=REC_CRC[(crc^b)&255]^(crc>>>8);
  }
 }finally{reader.releaseLock()}
}

// Store the already encoded video as a Blob and read it in chunks for the ZIP checksum.
async function recordingZip(files){
 const local=[],central=[],encoder=new TextEncoder();
 let offset=0,centralSize=0;
 for(const [name,blob] of files){
  const filename=encoder.encode(name),crc=await recordingCrc32(blob),size=blob.size;
  const head=new DataView(new ArrayBuffer(30)),dir=new DataView(new ArrayBuffer(46));
  head.setUint32(0,0x04034b50,true);head.setUint16(4,20,true);head.setUint32(14,crc,true);
  head.setUint32(18,size,true);head.setUint32(22,size,true);head.setUint16(26,filename.length,true);
  local.push(head.buffer,filename,blob);
  dir.setUint32(0,0x02014b50,true);dir.setUint16(4,20,true);dir.setUint16(6,20,true);
  dir.setUint32(16,crc,true);dir.setUint32(20,size,true);dir.setUint32(24,size,true);
  dir.setUint16(28,filename.length,true);dir.setUint32(42,offset,true);
  central.push(dir.buffer,filename);
  offset+=head.byteLength+filename.length+size;
  centralSize+=dir.byteLength+filename.length;
 }
 const end=new DataView(new ArrayBuffer(22));
 end.setUint32(0,0x06054b50,true);end.setUint16(8,files.length,true);end.setUint16(10,files.length,true);
 end.setUint32(12,centralSize,true);end.setUint32(16,offset,true);
 return new Blob([...local,...central,end.buffer],{type:'application/zip'});
}

async function downloadRecordingZip(){
 const result=recArchive,button=$('recDownloadZip');
 if(!result||button.disabled)return;
 button.disabled=true;
 button.textContent=t('rec.zip_wait');
 try{
  if(!result.url){
   const {base,filename,video,tracks,metadata}=result;
   const files=[[filename,video],...REC_TRACKS.map(name=>[`${base}_${name}.wav`,recordingWav(tracks[name],tracks.samples)]),
    [base+'.json',new Blob([JSON.stringify(metadata,null,1)],{type:'application/json'})]];
   const bundle=await recordingZip(files);
   if(result!==recArchive)return;
   result.url=URL.createObjectURL(bundle);
   result.tracks=null;
  }
  const link=document.createElement('a');
  link.href=result.url;link.download=result.base+'.zip';
  document.body.appendChild(link);link.click();link.remove();
 }catch{toast(t('toast.rec_zip_failed'))}
 finally{
  if(result===recArchive){
   button.disabled=false;
   button.textContent=t('rec.download_zip');
  }
 }
}
$('recBtn').onclick=recToggle;
$('recDownloadZip').onclick=downloadRecordingZip;
