// Settings: schema (defaults, ranges, help text), modules in data-flow order, presets, settings panel.
// Classic script; all files share one global scope and load in order from index.html.
// ---------- settings schema ----------
const DEF={face_roi:[0.2,0.05,0.8,0.95],mirror_view:true,show_face_roi:true,pose_enabled:false,pose_conf:0.4,pose_interval:2,pose_roi:[0,0,1,1],show_pose_roi:true,roi_rule:'either',
 turned_away_early:true,turned_away_seconds:2,recede_ratio:0.1,distance_enabled:false,camera_hfov:60,face_width_m:0.15,shoulder_width_m:0.38,
 distance_calib:1,distance_ema:0.3,zone_min:0.4,zone_max:2,leave_distance:2.5,min_conf:0.4,embed_backend:'fp32',npu_precision:'int8',cos_match_int8:0.4,cos_match:0.4,cos_weak:0.2,iou_match:0.3,embed_ema:0.15,join_seconds:2,grace_seconds:15,
 present_frames:2,absent_frames:3,idle_after:3,present_timeout:15,empty_timeout:5,base_timeout:20,maybe_away_frac:0.5,end_after_away:5,new_group_seconds:3,
 priority_mode:'vision',speech_join:'auto',presence_model:'rules',intent_model:'rules',intent_tau:1,join_threshold:0.6,drop_threshold:0.35,drop_seconds:8,intent_weighted_presence:false,w_facing:0.45,w_distance:0.1,w_dwell:0.1,w_approach:0.05,w_still:0.15,w_center:0.15,leave_halflife:1.5,detect_gain:4,lowconf_factor:0.15,p_active:0.8,p_maybe:0.6,p_away:0.3,away_hold_seconds:1,tracker_mode:'legacy',coast_seconds:2,coast_present_conf:0.4,gate_mahalanobis:9.21,appearance_min_sim:0.6,appearance_ema:0.2,body_min_conf:0.4,present_max_silence:0,motion_gate:true,motion_threshold:0.015,force_detect_seconds:2,embed_throttle:true,embed_interval:3,embed_move_thresh:0.15,adaptive_fps:true,fps_low:0.5,fps_high:2,roi_crop:true,crop_margin:0.1,max_width:640,fps:1,vad_k:2.5,vad_min:0.005,vad_hang:400,
 audio_pipeline:'energy_vad',iso_onset_guard:false,iso_reset_on_change:true,onset_needs_vision:true,onset_entry_only:true,clean_vad_margin_db:10,clean_vad_min_db:-50,clean_vad_hang:0.3,xcheck_enabled:true,near_range:1,xcheck_hold:0.75,xcheck_stale:2,xcheck_require_calib:false,xcheck_floor_db:-40,xcheck_recover_mix:0.3,xcheck_attack:0.01,xcheck_release:0.05,asr_provider:'none',asr_source:'clean',asr_feed:'speech',asr_region:'global',asr_speech_model:'universal-3-6-pro',asr_language_codes:'zh,en',asr_keyterms:'',asr_format_turns:true,asr_eot_confidence:0.4,asr_min_turn_silence:400,asr_max_turn_silence:1280,reid_enabled:false,reid_consent:false,reid_show_crops:true,reid_provider:'jev',reid_model:'jev-latest',reid_threshold:0.9,reid_uncertain_low:0.4,reid_ttl:600,reid_max_refs:3,reid_min_face:0.06,reid_decision_timeout:10,reid_face_wait:3,reid_timeout:2,reid_cooldown:5,reid_cap_per_hour:30,reid_cap_total:200,reid_cost_per_call:0,monitor:'off'};
const CLIENT=['monitor','mirror_view','fps','vad_k','vad_min','vad_hang','show_face_roi','show_pose_roi'];
const BOXES=['face_roi','pose_roi'];
// Setting metadata (range/step/unit/options). Labels, help text and option names come from the string tables
// (static/i18n/*.js) under s.<key>, s.<key>.h and s.<key>.o.<value>; units starting with "@" are unit.<name>.
const META={
 priority_mode:{opts:{vision:1,balanced:1,audio:1}},
 speech_join:{opts:{auto:1,off:1,single_candidate:1,all:1}},
 present_max_silence:{min:0,max:3600,step:10,u:"s"},
 face_roi:{min:0,max:1,step:0.01,box:true},
 mirror_view:{bool:true},
 show_face_roi:{bool:true},
 min_conf:{min:0,max:1,step:0.01},
 cos_match:{min:0,max:1,step:0.01},
 cos_weak:{min:0,max:1,step:0.01},
 embed_backend:{opts:{fp32:1,int8:1,npu:1}},
 npu_precision:{opts:{int8:1,fp16:1}},
 cos_match_int8:{min:0,max:1,step:0.01},
 iou_match:{min:0,max:1,step:0.01},
 embed_ema:{min:0,max:1,step:0.01},
 join_seconds:{min:0,max:30,step:0.5,u:"s"},
 grace_seconds:{min:0,max:120,step:1,u:"s"},
 present_frames:{min:1,max:20,step:1,u:"@frames"},
 absent_frames:{min:1,max:20,step:1,u:"@frames"},
 idle_after:{min:0,max:60,step:0.5,u:"s"},
 present_timeout:{min:1,max:600,step:1,u:"s"},
 empty_timeout:{min:1,max:120,step:0.5,u:"s"},
 base_timeout:{min:1,max:300,step:1,u:"s"},
 maybe_away_frac:{min:0.05,max:0.95,step:0.05},
 end_after_away:{min:0,max:120,step:1,u:"s"},
 new_group_seconds:{min:0,max:60,step:0.5,u:"s"},
 intent_model:{opts:{rules:1,score:1}},
 join_threshold:{min:0.05,max:0.99,step:0.01},
 drop_threshold:{min:0,max:0.95,step:0.01},
 drop_seconds:{min:0,max:120,step:1,u:"s"},
 intent_tau:{min:0.1,max:20,step:0.1,u:"s"},
 intent_weighted_presence:{bool:true},
 w_facing:{min:0,max:1,step:0.05},
 w_distance:{min:0,max:1,step:0.05},
 w_dwell:{min:0,max:1,step:0.05},
 w_approach:{min:0,max:1,step:0.05},
 w_still:{min:0,max:1,step:0.05},
 w_center:{min:0,max:1,step:0.05},
 presence_model:{opts:{rules:1,hmm:1}},
 leave_halflife:{min:0.1,max:30,step:0.1,u:"s"},
 detect_gain:{min:0.1,max:20,step:0.1},
 lowconf_factor:{min:0,max:1,step:0.05},
 p_active:{min:0.05,max:0.99,step:0.01},
 p_maybe:{min:0.02,max:0.98,step:0.01},
 p_away:{min:0.01,max:0.95,step:0.01},
 away_hold_seconds:{min:0,max:30,step:0.5,u:"s"},
 tracker_mode:{opts:{legacy:1,kalman:1}},
 coast_seconds:{min:0,max:10,step:0.5,u:"s"},
 coast_present_conf:{min:0,max:1,step:0.05},
 gate_mahalanobis:{min:1,max:100,step:0.5},
 appearance_min_sim:{min:0,max:1,step:0.05},
 appearance_ema:{min:0,max:1,step:0.05},
 pose_enabled:{bool:true},
 pose_interval:{min:0,max:2,step:0.1,u:"s"},
 pose_conf:{min:0.05,max:1,step:0.05},
 body_min_conf:{min:0.05,max:1,step:0.05},
 pose_roi:{min:0,max:1,step:0.01,box:true},
 show_pose_roi:{bool:true},
 roi_rule:{opts:{either:1,face:1,body:1}},
 turned_away_early:{bool:true},
 turned_away_seconds:{min:0,max:30,step:0.5,u:"s"},
 recede_ratio:{min:0.01,max:0.9,step:0.01},
 distance_enabled:{bool:true},
 camera_hfov:{min:20,max:170,step:1,u:"\u00b0"},
 face_width_m:{min:0.08,max:0.3,step:0.005,u:"m"},
 shoulder_width_m:{min:0.2,max:0.6,step:0.01,u:"m"},
 distance_calib:{min:0.2,max:5,step:0.01},
 distance_ema:{min:0.01,max:1,step:0.01},
 zone_min:{min:0,max:10,step:0.1,u:"m"},
 zone_max:{min:0.1,max:20,step:0.1,u:"m"},
 leave_distance:{min:0.2,max:30,step:0.1,u:"m"},
 motion_gate:{bool:true},
 motion_threshold:{min:0,max:0.2,step:0.001},
 force_detect_seconds:{min:0.5,max:30,step:0.5,u:"s"},
 embed_throttle:{bool:true},
 embed_interval:{min:0,max:30,step:0.5,u:"s"},
 embed_move_thresh:{min:0,max:1,step:0.01},
 adaptive_fps:{bool:true},
 fps_low:{min:0.2,max:5,step:0.1,u:"fps"},
 fps_high:{min:0.2,max:5,step:0.1,u:"fps"},
 roi_crop:{bool:true},
 crop_margin:{min:0,max:0.5,step:0.01},
 max_width:{min:96,max:1920,step:16,u:"px"},
 fps:{min:0.5,max:5,step:0.5,u:"fps"},
 vad_k:{min:1.2,max:8,step:0.1,u:"\u00d7"},
 vad_min:{min:0.001,max:0.05,step:0.001},
 vad_hang:{min:100,max:1500,step:50,u:"ms"},
 audio_pipeline:{opts:{energy_vad:1,puresound:1}},
 onset_entry_only:{bool:true},
 onset_needs_vision:{bool:true},
 iso_onset_guard:{bool:true},
 iso_reset_on_change:{bool:true},
 clean_vad_margin_db:{min:3,max:30,step:0.5,u:"dB"},
 clean_vad_min_db:{min:-80,max:-10,step:1,u:"dBFS"},
 clean_vad_hang:{min:0,max:2,step:0.05,u:"s"},
 xcheck_enabled:{bool:true},
 near_range:{min:0.2,max:10,step:0.1,u:"m"},
 xcheck_hold:{min:0,max:5,step:0.05,u:"s"},
 xcheck_stale:{min:0.2,max:30,step:0.1,u:"s"},
 xcheck_require_calib:{bool:true},
 xcheck_floor_db:{min:-80,max:0,step:1,u:"dB"},
 xcheck_recover_mix:{min:0,max:0.9,step:0.05},
 xcheck_attack:{min:0.001,max:0.5,step:0.001,u:"s"},
 xcheck_release:{min:0.005,max:2,step:0.005,u:"s"},
 asr_provider:{opts:{none:1,assemblyai:1}},
 asr_source:{opts:{clean:1,raw:1}},
 asr_feed:{opts:{speech:1,session:1}},
 asr_region:{opts:{global:1,us:1,eu:1}},
 asr_speech_model:{opts:{"universal-3-6-pro":1,"universal-3-5-pro":1,"universal-streaming-english":1,"universal-streaming-multilingual":1}},
 asr_language_codes:{text:true},
 asr_keyterms:{text:true},
 asr_format_turns:{bool:true},
 asr_eot_confidence:{min:0,max:1,step:0.05},
 asr_min_turn_silence:{min:50,max:10000,step:50,u:"ms"},
 asr_max_turn_silence:{min:100,max:10000,step:50,u:"ms"},
 reid_enabled:{bool:true},
 reid_consent:{bool:true},
 reid_show_crops:{bool:true},
 reid_provider:{opts:{jev:1}},
 reid_model:{opts:{"jev-latest":1,"jev-1.13.0":1}},
 reid_threshold:{min:0.5,max:0.99,step:0.01},
 reid_uncertain_low:{min:0,max:0.95,step:0.05},
 reid_ttl:{min:30,max:600,step:30,u:"s"},
 reid_max_refs:{min:1,max:5,step:1,u:"@people"},
 reid_min_face:{min:0.02,max:0.5,step:0.01},
 reid_face_wait:{min:0.5,max:15,step:0.5,u:"s"},
 reid_decision_timeout:{min:0.5,max:30,step:0.5,u:"s"},
 reid_timeout:{min:0.3,max:10,step:0.1,u:"s"},
 reid_cooldown:{min:0,max:120,step:1,u:"s"},
 reid_cap_per_hour:{min:1,max:600,step:1,u:"@times"},
 reid_cap_total:{min:1,max:10000,step:1,u:"@times"},
 reid_cost_per_call:{min:0,max:10,step:0.0001}};
// "Recommended" = the full set tuned in a field test (low-angle robot camera, one person talking up close);
// the scenario presets start from it and adjust a few values
const REC={tracker_mode:'kalman',presence_model:'hmm',intent_model:'score',pose_enabled:true,distance_enabled:true,
 audio_pipeline:'puresound',xcheck_enabled:true,near_range:1.6,min_conf:0.4,leave_halflife:1.5,force_detect_seconds:2,
 join_seconds:1.5,grace_seconds:10,face_roi:[0.25,0.1,0.8,0.89],pose_roi:[0.18,0.04,0.85,0.92]};
const PRESETS={
 lowcam:{...REC,min_conf:0.35,leave_halflife:2,force_detect_seconds:1.5,face_roi:[0.2,0.05,0.85,0.95],pose_roi:[0.1,0,0.9,1],body_min_conf:0.35},
 single:{...REC,face_roi:[0.25,0.05,0.75,0.95],present_timeout:45,empty_timeout:5,new_group_seconds:3,absent_frames:3},
 group:{...REC,face_roi:[0.08,0.02,0.92,0.98],join_seconds:3,grace_seconds:20,present_frames:2,absent_frames:4,present_timeout:90,empty_timeout:8,new_group_seconds:5,cos_match:0.42,fps:2,near_range:2},
 noisy:{...REC,priority_mode:'vision',vad_k:3.5,vad_min:0.02,vad_hang:300,min_conf:0.5,near_range:1.2,base_timeout:15,present_timeout:45,empty_timeout:5}};
const KEY='presence-cfg-v1';
const migrate=o=>{o={...o};if(!o._v091){if(o.max_width===320)o.max_width=640;o._v091=1}if(o.roi&&!o.face_roi)o.face_roi=o.roi;delete o.roi;return o};
let cfg=Object.assign(structuredClone({...DEF,_v091:1}),migrate(JSON.parse(localStorage.getItem(KEY)||'{}')));cfg.monitor='off';// monitoring always starts off (avoids feedback)
const save=()=>localStorage.setItem(KEY,JSON.stringify(cfg));
const toast=t=>{const e=$('toast');e.textContent=t;e.classList.add('is-visible');clearTimeout(e._t);e._t=setTimeout(()=>e.classList.remove('is-visible'),2200)};
const clamp=(k,v)=>{const m=META[k];if(m.text)return String(v).slice(0,2000);if(m.bool)return v===true||v==='true'||v===1;if(m.opts)return m.opts[v]?v:DEF[k];return Math.min(m.max,Math.max(m.min,Number(v)))};
// ---------- settings: modules in data-flow order (v0.10) ----------
// Each module: enable/mode key at the top, then parameters. `adv` = shown only in advanced mode.
// Rules per key: show(c) -> hidden when false (irrelevant), need:[fn,reason] -> shown disabled with the reason.
const ON=k=>c=>!!c[k],EQ=(k,v)=>c=>c[k]===v,PS=c=>c.audio_pipeline==='puresound',DIST=c=>!!c.distance_enabled;
const MODULES=[
 {id:'input',keys:['fps','vad_k','vad_min','vad_hang'],adv:['vad_min','vad_hang'],
  rules:{fps:{need:[c=>!c.adaptive_fps,'need.fps_adaptive']},vad_k:{need:[c=>!PS(c),'need.vad_k_ps']},vad_min:{need:[c=>!PS(c),'need.not_with_ps']},vad_hang:{need:[c=>!PS(c),'need.not_with_ps']}},
  status:c=>`${c.adaptive_fps?t('st.adaptive'):c.fps+' fps'} · ${PS(c)?t('st.pcm'):t('st.browser_vad')}`},
 {id:'detect',keys:['face_roi','min_conf','embed_backend','npu_precision','cos_match_int8'],adv:['cos_match_int8','npu_precision'],
  rules:{npu_precision:{show:EQ('embed_backend','npu')},cos_match_int8:{show:c=>c.embed_backend!=='fp32'}},status:c=>`YuNet · ${c.embed_backend}`},
 {id:'pose',gate:'pose_enabled',keys:['pose_conf','pose_interval','body_min_conf','pose_roi','roi_rule','turned_away_early','turned_away_seconds','recede_ratio'],adv:['pose_interval','body_min_conf','recede_ratio','turned_away_seconds'],
  rules:{turned_away_seconds:{show:ON('turned_away_early')}},status:c=>c.pose_enabled?t('st.on'):t('st.off')},
 {id:'distance',gate:'distance_enabled',keys:['camera_hfov','distance_calib','zone_min','zone_max','leave_distance','face_width_m','shoulder_width_m','distance_ema'],adv:['face_width_m','shoulder_width_m','distance_ema'],
  rules:{shoulder_width_m:{need:[ON('pose_enabled'),'need.shoulder_pose']}},status:c=>c.distance_enabled?t('st.on'):t('st.off')},
 {id:'track',gate:'tracker_mode',keys:['join_seconds','grace_seconds','cos_match','cos_weak','iou_match','embed_ema','coast_seconds','coast_present_conf','gate_mahalanobis','appearance_min_sim','appearance_ema'],
  adv:['cos_weak','iou_match','embed_ema','gate_mahalanobis','appearance_min_sim','appearance_ema','coast_present_conf'],
  rules:Object.fromEntries(['coast_seconds','coast_present_conf','gate_mahalanobis','appearance_min_sim','appearance_ema'].map(k=>[k,{show:EQ('tracker_mode','kalman')}])),status:c=>c.tracker_mode==='kalman'?'Kalman':t('st.classic')},
 {id:'intent',gate:'intent_model',keys:['join_threshold','drop_threshold','drop_seconds','intent_tau','intent_weighted_presence','w_facing','w_distance','w_dwell','w_approach','w_still','w_center'],
  adv:['intent_tau','w_facing','w_distance','w_dwell','w_approach','w_still','w_center','drop_seconds'],
  rules:{...Object.fromEntries(['join_threshold','drop_threshold','drop_seconds','intent_tau','intent_weighted_presence','w_facing','w_distance','w_dwell','w_approach','w_still','w_center'].map(k=>[k,{show:EQ('intent_model','score')}])),
   w_distance:{show:EQ('intent_model','score'),need:[DIST,'need.w_distance']},intent_weighted_presence:{show:EQ('intent_model','score'),need:[EQ('presence_model','hmm'),'need.iwp_hmm']}},
  status:c=>c.intent_model==='score'?t('st.intent_score'):t('st.rules')},
 {id:'presence',gate:'presence_model',keys:['priority_mode','speech_join','present_frames','absent_frames','leave_halflife','detect_gain','lowconf_factor','p_active','p_maybe','p_away','away_hold_seconds'],
  adv:['detect_gain','lowconf_factor','away_hold_seconds','p_active','speech_join'],
  rules:{...Object.fromEntries(['leave_halflife','detect_gain','lowconf_factor','p_active','p_maybe','p_away','away_hold_seconds'].map(k=>[k,{show:EQ('presence_model','hmm')}])),present_frames:{show:EQ('presence_model','rules')},absent_frames:{show:EQ('presence_model','rules')}},
  status:c=>`${c.presence_model==='hmm'?t('st.hmm'):t('st.rules')} · ${t('prio.'+c.priority_mode)}`},
 {id:'state',keys:['idle_after','present_timeout','empty_timeout','base_timeout','present_max_silence','maybe_away_frac','end_after_away','new_group_seconds'],adv:['maybe_away_frac','base_timeout'],
  rules:{present_max_silence:{need:[EQ('priority_mode','vision'),'need.vision_only']},present_timeout:{need:[c=>c.priority_mode!=='vision','need.not_vision']}},status:c=>t('st.timeout',{a:c.present_timeout,b:c.empty_timeout})},
 {id:'audio',gate:'audio_pipeline',keys:['onset_entry_only','onset_needs_vision','iso_onset_guard','iso_reset_on_change','clean_vad_margin_db','clean_vad_min_db','clean_vad_hang','xcheck_enabled','near_range','xcheck_hold','xcheck_stale','xcheck_require_calib','xcheck_floor_db','xcheck_recover_mix','xcheck_attack','xcheck_release'],
  adv:['iso_reset_on_change','clean_vad_hang','xcheck_stale','xcheck_attack','xcheck_release','xcheck_recover_mix','clean_vad_min_db'],
  rules:{...Object.fromEntries(['onset_entry_only','onset_needs_vision','iso_onset_guard','iso_reset_on_change','clean_vad_margin_db','clean_vad_min_db','clean_vad_hang','xcheck_enabled'].map(k=>[k,{show:PS}])),
   ...Object.fromEntries(['near_range','xcheck_hold','xcheck_stale','xcheck_floor_db','xcheck_recover_mix','xcheck_attack','xcheck_release'].map(k=>[k,{show:c=>PS(c)&&c.xcheck_enabled}])),
   near_range:{show:c=>PS(c)&&c.xcheck_enabled,need:[DIST,'need.near_range']},
   xcheck_require_calib:{show:c=>PS(c)&&c.xcheck_enabled,need:[DIST,'need.distance']}},
  status:c=>PS(c)?`PureSound${c.xcheck_enabled?t('st.xcheck'):''}`:t('st.browser_vad')},
 {id:'asr',gate:'asr_provider',keys:['asr_source','asr_feed','asr_region','asr_speech_model','asr_language_codes','asr_keyterms','asr_format_turns','asr_eot_confidence','asr_min_turn_silence','asr_max_turn_silence'],
  adv:['asr_eot_confidence','asr_min_turn_silence','asr_max_turn_silence','asr_keyterms','asr_format_turns'],
  rules:{...Object.fromEntries(['asr_source','asr_feed','asr_region','asr_speech_model','asr_language_codes','asr_keyterms','asr_min_turn_silence','asr_max_turn_silence'].map(k=>[k,{show:EQ('asr_provider','assemblyai')}])),
   asr_language_codes:{show:c=>c.asr_provider==='assemblyai'&&c.asr_speech_model.endsWith('-pro')},
   asr_format_turns:{show:c=>c.asr_provider==='assemblyai'&&c.asr_speech_model.startsWith('universal-streaming')},asr_eot_confidence:{show:c=>c.asr_provider==='assemblyai'&&c.asr_speech_model.startsWith('universal-streaming')}},
  gateNeed:[PS,'need.asr_ps'],status:c=>c.asr_provider==='none'?t('st.off'):(PS(c)?'AssemblyAI':t('st.asr_needs_ps'))},
 {id:'reid',gate:'reid_enabled',keys:['reid_consent','reid_show_crops','reid_provider','reid_model','reid_threshold','reid_uncertain_low','reid_ttl','reid_max_refs','reid_min_face','reid_face_wait','reid_decision_timeout','reid_timeout','reid_cooldown','reid_cap_per_hour','reid_cap_total','reid_cost_per_call'],
  adv:['reid_provider','reid_uncertain_low','reid_min_face','reid_face_wait','reid_decision_timeout','reid_timeout','reid_cooldown','reid_cap_total'],rules:{},
  status:c=>!c.reid_enabled?t('st.off'):c.reid_consent?t('st.on'):t('st.no_consent')},
 {id:'perf',keys:['motion_gate','motion_threshold','force_detect_seconds','embed_throttle','embed_interval','embed_move_thresh','adaptive_fps','fps_low','fps_high','roi_crop','crop_margin','max_width'],
  adv:['motion_threshold','force_detect_seconds','embed_interval','embed_move_thresh','crop_margin'],
  rules:{motion_threshold:{show:ON('motion_gate')},force_detect_seconds:{show:ON('motion_gate')},embed_interval:{show:ON('embed_throttle')},embed_move_thresh:{show:ON('embed_throttle')},fps_low:{show:ON('adaptive_fps')},fps_high:{show:ON('adaptive_fps')},crop_margin:{show:ON('roi_crop')}},
  status:c=>`${t('st.gate',{v:c.motion_gate?t('st.on'):t('st.off')})} · ${c.max_width}px`},
 {id:'display',keys:['mirror_view','show_face_roi','show_pose_roi'],adv:[],rules:{show_pose_roi:{show:ON('pose_enabled')}},status:c=>c.mirror_view?t('st.mirror'):t('st.raw')}];
const MOD_PRESETS={
 recommended:{...REC},
 lowpower:{motion_gate:true,embed_throttle:true,embed_interval:5,adaptive_fps:true,fps_low:0.5,fps_high:1,roi_crop:true,max_width:480,pose_enabled:false,distance_enabled:false}};
let ADV=localStorage.getItem('presence-adv')==='1';
function ruleOf(m,k){return(m.rules||{})[k]||{}}
function buildUI(){
 const all=new Set(MODULES.flatMap(m=>[m.gate,...m.keys].filter(Boolean)));const miss=Object.keys(META).filter(k=>!all.has(k));if(miss.length)console.error('settings keys without module:',miss);
 $('flow').innerHTML=MODULES.map((m,i)=>`${i?'<span class="arr">→</span>':''}<button type="button" class="chip" data-mod="${m.id}"><b>${t('mod.'+m.id+'.short')}</b><small id="fs_${m.id}"></small></button>`).join('');
 $('flow').querySelectorAll('[data-mod]').forEach(b=>b.onclick=()=>{const d=$('mod_'+b.dataset.mod);d.open=true;d.scrollIntoView({behavior:'smooth',block:'start'})});
 $('groups').innerHTML=MODULES.map(m=>`<details class="mod" id="mod_${m.id}" ${m.id==='detect'?'open':''}><summary>${t('mod.'+m.id)} <span class="mst" id="ms_${m.id}"></span><button type="button" class="button button-ghost button-small mreset" data-mreset="${m.id}" title="${t('mod.reset_title')}">${t('mod.reset')}</button></summary>`
  +(m.gate?`<div class="gate">${rowHTML(m.gate)}<div class="gnote" id="gn_${m.id}"></div></div>`:'')+`<div class="mbody">`+m.keys.map(rowHTML).join('')+'</div></details>').join('');
 $('groups').querySelectorAll('[data-mreset]').forEach(b=>b.onclick=e=>{e.preventDefault();e.stopPropagation();const m=MODULES.find(x=>x.id===b.dataset.mreset);
  const part={};[m.gate,...m.keys].filter(Boolean).forEach(k=>part[k]=structuredClone(DEF[k]));setCfg(part);restartFrames();toast(t('toast.mod_reset',{m:t('mod.'+m.id)}))});
 $('adv').checked=ADV;$('adv').onchange=()=>{ADV=$('adv').checked;localStorage.setItem('presence-adv',ADV?'1':'0');applyDeps()};
 $('groups').querySelectorAll('input,select').forEach(inp=>{
  const ev=inp.type==='range'?'input':'change';
  if(inp.type==='checkbox'){inp.addEventListener('change',()=>setCfg({[inp.dataset.k]:inp.checked}));return}
  if(inp.type==='text'){inp.addEventListener('change',()=>setCfg({[inp.dataset.k]:inp.value}));return}
  if(inp.tagName==='SELECT'){inp.addEventListener('change',()=>setCfg({[inp.dataset.k]:inp.value}));return}
  inp.addEventListener(ev,()=>{const k=inp.dataset.k;
   if(BOXES.includes(k)){const r=[...cfg[k]];r[+inp.dataset.i]=Math.min(1,Math.max(0,+inp.value));setCfg({[k]:r})}
   else setCfg({[k]:clamp(k,inp.value)},inp)});
  if(inp.type==='range')inp.addEventListener('change',()=>sendCfg());
 });
 syncUI();
}
// Localized view of a setting: label, help, option names and unit from the string tables.
const LM=k=>{const b=META[k],o={...b,l:t('s.'+k),h:t('s.'+k+'.h')};if(b.opts)o.opts=Object.fromEntries(Object.keys(b.opts).map(v=>[v,t('s.'+k+'.o.'+v)]));if(b.u&&b.u[0]==='@')o.u=t('unit.'+b.u.slice(1));return o};
function rowHTML(k){const m=LM(k);const d=Array.isArray(DEF[k])?DEF[k].join(', '):DEF[k];
 const tip=`<div class="why" id="why_${k}"></div><p class="tip">${m.h}　<span class="def">${t('set.default',{v:(m.bool?(d?t('st.on'):t('st.off')):m.opts?m.opts[d]:d)+(m.u?' '+m.u:'')})}</span></p>`;
 if(m.bool)return `<div class="row" id="row_${k}"><label class="toggle-row" style="margin:0" title="${m.h}"><input type="checkbox" data-k="${k}"><span class="toggle-ui"></span><span>${m.l}</span></label>${tip}</div>`;
 if(m.text)return `<div class="row field" id="row_${k}"><label for="f_${k}">${m.l}</label><input class="control" id="f_${k}" type="text" data-k="${k}">${tip}</div>`;
 if(m.box)return `<div class="row" id="row_${k}"><div class="top"><label title="${m.h}">${m.l}</label></div><div class="roi4">${['x0','y0','x1','y1'].map((n,j)=>`<input class="control" type="number" data-k="${k}" data-i="${j}" min="0" max="1" step="0.01" title="${n}">`).join('')}</div>${tip}</div>`;
 if(m.opts)return `<div class="row field" id="row_${k}"><label for="f_${k}">${m.l}</label><select class="control" id="f_${k}" data-k="${k}">${Object.entries(m.opts).map(([o,t])=>`<option value="${o}">${t}</option>`).join('')}</select>${tip}</div>`;
 return `<div class="row" id="row_${k}"><div class="field-label range-label"><label for="f_${k}">${m.l}</label><output id="o_${k}"></output></div><input class="range" id="f_${k}" type="range" data-k="${k}" min="${m.min}" max="${m.max}" step="${m.step}"><div class="top" style="margin-top:6px"><input class="control" type="number" data-k="${k}" min="${m.min}" max="${m.max}" step="${m.step}" style="width:84px"><span class="def">${m.u}</span></div>${tip}</div>`}
function applyDeps(){for(const m of MODULES){
  const gateOff=m.gate&&(META[m.gate].bool?!cfg[m.gate]:false);
  for(const k of m.keys){const r=$('row_'+k);if(!r)continue;const ru=ruleOf(m,k);
   const vis=!gateOff&&(!ru.show||ru.show(cfg))&&(ADV||!m.adv.includes(k)||r.classList.contains('changed'));
   const ok=!ru.need||ru.need[0](cfg);r.hidden=!vis;r.classList.toggle('dis',!ok);r.querySelectorAll('input,select').forEach(x=>x.disabled=!ok);
   $('why_'+k).textContent=ok?'':'⛔ '+t(ru.need[1])}
  const st=m.status(cfg),gn=$('gn_'+m.id);$('ms_'+m.id).textContent=st;const fs=$('fs_'+m.id);if(fs){fs.textContent=st;fs.parentElement.classList.toggle('off',[t('st.off'),t('st.browser_vad')].includes(st))}
  if(gn){const bad=m.gateNeed&&!m.gateNeed[0](cfg)&&cfg[m.gate]!=='none';gn.textContent=bad?'⚠ '+t(m.gateNeed[1]):gateOff?t('gate.off'):'';}
  const body=$('mod_'+m.id).querySelector('.mbody'),hid=m.keys.filter(k=>$('row_'+k)&&$('row_'+k).hidden).length;
  body.dataset.hidden=hid;}}
function syncUI(skip){if(cfg.asr_provider==='none')captionOff();
 document.querySelectorAll('#groups input,#groups select').forEach(inp=>{if(inp===skip)return;const k=inp.dataset.k;
  if(inp.type==='checkbox'){inp.checked=!!cfg[k];return}
  const v=BOXES.includes(k)?(+cfg[k][+inp.dataset.i]).toFixed(2):cfg[k];inp.value=v;
  if(inp.type==='range'){const o=$('o_'+k);if(o)o.textContent=LM(k).u?`${v} ${LM(k).u}`:v}});
 for(const k in DEF){const r=$('row_'+k);if(r)r.classList.toggle('changed',JSON.stringify(cfg[k])!==JSON.stringify(DEF[k]))}
 if(typeof applyDeps==='function'&&$('flow').childElementCount)applyDeps();
}
let sendTimer;
function setCfg(part,src,immediate){Object.assign(cfg,part);if(!cfg.reid_enabled||!cfg.reid_consent)clearReidDisplay();save();syncUI(src);
 if(src&&src.type==='range'){clearTimeout(sendTimer);sendTimer=setTimeout(sendCfg,250)}else sendCfg();
 if(['fps','adaptive_fps','fps_low','fps_high'].some(k=>k in part))restartFrames();draw();}
function serverCfg(){const o={};for(const k in DEF)if(!CLIENT.includes(k))o[k]=cfg[k];return o}
function sendCfg(){if(!cfg.reid_enabled||!cfg.reid_consent)clearReidDisplay();if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'config',config:serverCfg()}))}
document.querySelectorAll('#presets [data-p]').forEach(b=>b.onclick=()=>{const p=b.dataset.p;
 cfg=Object.assign(structuredClone(DEF),p==='reset'?{}:(PRESETS[p]||MOD_PRESETS[p]));save();syncUI();sendCfg();restartFrames();draw();toast(t('toast.preset',{p:b.textContent}))});
$('exp').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(cfg,null,2)],{type:'application/json'}));a.download='presence-config.json';a.click()};
$('impB').onclick=()=>$('imp').click();
$('imp').onchange=async e=>{try{const j=JSON.parse(await e.target.files[0].text());const n=structuredClone(DEF);
 const jm=migrate(j);for(const k in DEF)if(k in jm)n[k]=BOXES.includes(k)?jm[k].slice(0,4).map(x=>Math.min(1,Math.max(0,+x))):clamp(k,jm[k]);cfg=n;save();syncUI();sendCfg();restartFrames();draw();toast(t('toast.imported'))}
 catch(err){toast(t('toast.import_fail',{e:err.message}))}e.target.value=''};
const drawer=(o)=>{$('drawer').classList.toggle('is-open',o);$('scrim').classList.toggle('is-open',o);document.body.classList.toggle('has-drawer',o)};
$('openSet').onclick=()=>drawer(true);$('closeSet').onclick=$('scrim').onclick=()=>drawer(false);
