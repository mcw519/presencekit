// Browser-local input choices. Device IDs never become part of the server config or recording metadata.
const captureInputs=[
 {id:'cameraDevice',kind:'videoinput',key:'presence-camera-device',name:'device.camera'},
 {id:'microphoneDevice',kind:'audioinput',key:'presence-microphone-device',name:'device.microphone'},
].map(input=>({...input,selected:localStorage.getItem(input.key)||'',devices:[],listed:false}));
let mediaStarting=false,readingDevices=false,deviceListRevision=0,deviceError='';

function captureConstraints(){
 const [camera,mic]=captureInputs.map(input=>input.selected);
 return {
  video:{width:{ideal:640},height:{ideal:480},...(camera?{deviceId:{exact:camera}}:{facingMode:'user'})},
  audio:{echoCancellation:true,noiseSuppression:true,...(mic?{deviceId:{exact:mic}}:{})}
 };
}

function syncInputControls(){
 const available=typeof navigator.mediaDevices?.enumerateDevices==='function';
 const busy=mediaStarting||readingDevices;
 const needsAccess=captureInputs.some(input=>!input.devices.some(device=>device.label));
 for(const input of captureInputs)$(input.id).disabled=!available||running||busy;
 $('start').disabled=busy&&!running;
 $('loadDevices').hidden=!available||running||!needsAccess;
 $('loadDevices').disabled=busy;
 const status=$('deviceStatus');
 status.textContent=readingDevices?t('device.loading'):!available?t('device.unavailable'):
  running?t('device.stop_to_change'):deviceError;
 status.hidden=!status.textContent;
}

async function refreshInputDevices(){
 const media=navigator.mediaDevices;
 if(typeof media?.enumerateDevices!=='function'){syncInputControls();return}
 const revision=++deviceListRevision;
 try{
  const list=await media.enumerateDevices();
  if(revision!==deviceListRevision)return;
  for(const input of captureInputs){
   const reported=list.filter(device=>device.kind===input.kind);
   const identifiable=reported.filter(device=>device.deviceId);
   // Some browsers redact IDs again after the temporary permission stream is released.
   if(identifiable.length||!reported.length){
    input.listed=Boolean(identifiable.length);
    input.devices=identifiable.filter(device=>device.deviceId!=='default');
   }
   const options=[new Option(t('device.default'),'')];
   input.devices.forEach((device,index)=>options.push(
    new Option(device.label||t('device.numbered',{kind:t(input.name),n:index+1}),device.deviceId)
   ));
   if(input.selected&&!input.devices.some(device=>device.deviceId===input.selected)){
    options.push(new Option(t(input.listed?'device.disconnected':'device.saved'),input.selected));
   }
   const select=$(input.id);
   select.replaceChildren(...options);
   select.value=input.selected;
  }
 }catch{
  // System defaults remain selectable if the browser withholds or fails enumeration.
 }
 syncInputControls();
}

async function requestInputAccess(){
 if(running||mediaStarting||readingDevices)return;
 readingDevices=true;deviceError='';syncInputControls();
 let probe;
 try{
  probe=await navigator.mediaDevices.getUserMedia({video:true,audio:true});
  await refreshInputDevices();
 }catch(e){
  deviceError=t('device.access_failed',{e:e.message||e.name});
 }finally{
  probe?.getTracks().forEach(track=>track.stop());
  readingDevices=false;syncInputControls();
 }
}

function initInputDevices(){
 for(const input of captureInputs)$(input.id).onchange=()=>{
  input.selected=$(input.id).value;
  if(input.selected)localStorage.setItem(input.key,input.selected);
  else localStorage.removeItem(input.key);
  // Picking a replacement explicitly leaves the next start to the user.
  autoPaused=false;deviceError='';setUI();
 };
 $('loadDevices').onclick=requestInputAccess;
 navigator.mediaDevices?.addEventListener?.('devicechange',refreshInputDevices);
 refreshInputDevices();
}

function inputStartError(error,fallback){
 if(captureInputs.some(input=>input.selected)&&['NotFoundError','OverconstrainedError'].includes(error.name)){
  deviceError=t('device.selection_failed');syncInputControls();refreshInputDevices();
  return deviceError;
 }
 return t(fallback,{e:error.message||error.name});
}
