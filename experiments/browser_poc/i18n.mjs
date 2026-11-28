// Match PresenceKit's URL, browser language and saved preference.
const strings = {
  title: ['PresenceKit · Live Monitor', 'PresenceKit · 即時監看'],
  navigation: ['Navigation', '導覽'],
  pageNavigation: ['On this page', '頁內導覽'],
  live: ['Live', '即時'],
  monitor: ['Monitor', '監看'],
  playback: ['Recording playback', '錄音回放'],
  localRuntime: ['On-device processing', '本機運算'],
  language: ['Language', '語言'],
  theme: ['Theme', '主題'],
  auto: ['Auto', '自動'],
  light: ['Light', '淺色'],
  dark: ['Dark', '深色'],
  liveMonitor: ['Live Monitor', '即時監看'],
  nearPriority: ['Near-field priority · Fixed 2 fps video', '近場優先 · 影像固定 2 fps'],
  captureOff: ['Camera and microphone off', '鏡頭／麥克風關閉'],
  captureOn: ['Capturing', '擷取中'],
  start: ['Start', '開始'],
  stop: ['Stop', '停止'],
  videoAudio: ['Video and audio', '影像與音訊'],
  alignedTracks: ['Raw and output share the same time and level scales', '原音與輸出共用時間軸和刻度'],
  trackDisplay: ['Audio track display', '音軌顯示'],
  waveform: ['Waveform', '波形'],
  spectrogram: ['Spectrogram', '頻譜'],
  both: ['Both', '兩者'],
  startPrompt: ['Select Start to enable your camera and microphone', '按「開始」開啟鏡頭與麥克風'],
  permissionPrompt: ['Browser permission required', '需要瀏覽器授權'],
  loadingPermission: ['Loading models and requesting device access', '載入模型並等待裝置授權'],
  near: ['Near', '近場'],
  far: ['Far', '遠場'],
  trackNear: ['#{index} Near', '#{index} 近場'],
  trackFar: ['#{index} Far', '#{index} 遠場'],
  raw: ['Raw', '原音'],
  output: ['Output', '輸出'],
  rawWaveform: ['Raw waveform', '原音波形'],
  rawSpectrogram: ['Raw spectrogram', '原音頻譜'],
  outputWaveform: ['Output waveform', '輸出波形'],
  outputSpectrogram: ['Output spectrogram', '輸出頻譜'],
  fiveSecondsAgo: ['−5 s', '−5 秒'],
  now: ['Now', '現在'],
  playbackHint: ['Stop capture to play or download the output', '停止後可試聽與下載本次輸出'],
  outputLength: ['{seconds} s of output', '輸出 {seconds} 秒'],
  download: ['Download WAV', '下載 WAV'],
  liveDecision: ['Live decision', '即時判斷'],
  ready: ['Ready to start', '等待啟動'],
  standby: ['Standby', '待命'],
  speechDetection: ['Speech detection', '說話判定'],
  rawMinusOutput: ['Raw − output', '原音 − 輸出'],
  rmsDifference: ['RMS level difference over the last 240 ms', '最近 240 毫秒的 RMS 音量差'],
  faces: ['Detected faces', '偵測到的人臉'],
  liveListening: ['Live listening', '即時監聽'],
  headphones: ['Use headphones to prevent microphone feedback.', '請戴耳機，避免喇叭聲被麥克風收進去。'],
  listeningTrack: ['Listening track', '監聽音軌'],
  off: ['Off', '關閉'],
  decisionTimeline: ['Decision timeline', '判斷時間軸'],
  lastTenSeconds: ['Last 10 seconds', '最近 10 秒'],
  speech: ['Speech', '算說話'],
  speaking: ['Speaking', '說話'],
  quiet: ['Quiet', '安靜'],
  processing: ['Processing', '處理'],
  timelineDescription: ['Speech, attenuation, and recovery over the last 10 seconds', '最近 10 秒的說話、壓低與救回判斷'],
  attenuated: ['Attenuated', '壓低'],
  recovered: ['Raw audio recovered', '原音救回'],
  keptSpeech: ['Speech kept', '保留語音'],
  keptIsolation: ['Isolation kept', '維持隔離'],
  cutFar: ['Far-field attenuated', '遠場壓低'],
  cutEmpty: ['Attenuated, no person', '無人時壓低'],
  audioOnly: ['Audio-only decision', '僅音訊判斷'],
  stopped: ['Stopped', '已停止'],
  loadingModels: ['Loading PureSound and YuNet', '正在載入 PureSound 與 YuNet'],
  capturingLocally: ['Capturing on this device · 2 fps', '本機擷取中 · 2 fps'],
  captureOverload: ['Device could not keep up. Processed audio was saved.', '裝置跟不上即時處理，已保留完成的音訊'],
  captureLimit: ['Two-minute recording complete', '已完成兩分鐘試錄'],
  backgroundStop: ['Capture stopped when you switched tabs', '切換分頁，已停止擷取'],
  permissionDenied: ['Allow camera and microphone access in your browser, then try again.', '請在瀏覽器允許相機與麥克風存取，再試一次。'],
  devicesMissing: ['Connect a camera and microphone, then try again.', '請連接相機與麥克風，再試一次。'],
  devicesBusy: ['Camera or microphone unavailable. Check whether another app is using it.', '無法使用相機或麥克風，請確認是否被其他程式佔用。'],
  secureContext: ['Camera and microphone access requires HTTPS or localhost.', '相機與麥克風需要 HTTPS 或 localhost。'],
  captureError: ['Capture could not continue. Check your device and try again.', '無法繼續擷取，請確認裝置後再試。'],
};

const storageKey = 'presence-lang';
const languages = ['en', 'zh-TW'];
const originalOverride = new URLSearchParams(location.search).get('lang');
const languageHint = (navigator.languages ?? [navigator.language]).find(lang => /^(zh|en)(-|$)/i.test(lang));
const placeholders = new WeakMap();
let language = [originalOverride, localStorage.getItem(storageKey)].find(value => languages.includes(value))
  ?? (/^zh/i.test(languageHint) ? 'zh-TW' : 'en');
if (languages.includes(originalOverride)) localStorage.setItem(storageKey, language);

function translate(key, values = {}) {
  return strings[key][language === 'en' ? 0 : 1].replace(/\{(\w+)\}/g, (_, name) => values[name]);
}

export function localize(element, key, values) {
  element.dataset.i18n = key;
  placeholders.set(element, values);
  element.textContent = translate(key, values);
}

function render() {
  document.documentElement.lang = language === 'en' ? 'en' : 'zh-Hant-TW';
  for (const element of document.querySelectorAll('[data-i18n]')) {
    element.textContent = translate(element.dataset.i18n, placeholders.get(element));
  }
  for (const attr of ['aria-label', 'title']) {
    for (const element of document.querySelectorAll(`[data-i18n-${attr}]`)) {
      element.setAttribute(attr, translate(element.getAttribute(`data-i18n-${attr}`)));
    }
  }
  for (const button of document.querySelectorAll('[data-language]')) {
    button.setAttribute('aria-pressed', button.dataset.language === language);
  }
}

export function initLanguage() {
  render();
  for (const button of document.querySelectorAll('[data-language]')) {
    button.onclick = () => {
      language = button.dataset.language;
      localStorage.setItem(storageKey, language);
      const query = new URL(location.href);
      if (query.searchParams.has('lang')) {
        query.searchParams.set('lang', language);
        history.replaceState(history.state, '', query);
      }
      render();
    };
  }
}
