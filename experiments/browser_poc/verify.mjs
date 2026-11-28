import {writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const [base, output, upstream, ...fixtures] = process.argv.slice(2);
const {chromium} = await import(pathToFileURL(resolve(upstream, 'sdk/web/node_modules/playwright/index.mjs')));
const cameraFixture = process.env.PRESENCE_POC_CAMERA_FIXTURE;
const micFixture = process.env.PRESENCE_POC_MIC_FIXTURE;
const browser = await chromium.launch({args: [
  '--disable-background-timer-throttling', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
  ...(cameraFixture ? ['--use-file-for-fake-video-capture=' + resolve(cameraFixture)] : []),
  ...(micFixture ? ['--use-file-for-fake-audio-capture=' + resolve(micFixture)] : []),
]});
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}});
  const unexpected = [];
  page.on('pageerror', error => unexpected.push(error.message));
  page.on('websocket', socket => unexpected.push(socket.url()));
  const origin = new URL(base).origin;
  await page.context().route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) {
      unexpected.push(url.href);
      return route.abort();
    }
    const relative = url.pathname.slice(new URL(base).pathname.length);
    const match = /^fixtures\/([\w-]+)\/(fixture\.json|raw\.wav|\d+\.jpg)$/.exec(relative);
    if (match && fixtures.includes(match[1])) {
      return route.fulfill({path: resolve(output, 'fixtures', match[1], match[2])});
    }
    return route.continue();
  });
  console.log('Chromium', browser.version());
  const englishUrl = new URL(base);
  englishUrl.searchParams.set('lang', 'en');
  await page.goto(englishUrl.href);
  await page.waitForFunction(() => document.documentElement.lang === 'en' && document.title === 'PresenceKit · Live Monitor');
  await page.locator('[data-language=zh-TW]').click();
  const chosen = await page.evaluate(() => ({
    html: document.documentElement.lang, saved: localStorage.getItem('presence-lang'), query: new URL(location.href).searchParams.get('lang'),
  }));
  if (chosen.html !== 'zh-Hant-TW' || chosen.saved !== 'zh-TW' || chosen.query !== 'zh-TW') {
    throw Error('Language switch did not save Traditional Chinese');
  }
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#start').textContent === '開始');
  await page.locator('[data-language=en]').click();
  await page.screenshot({path: resolve(output, 'monitor-idle.png')});
  for (const name of fixtures) {
    const begin = Date.now();
    const result = await page.evaluate(async fixture => {
      const {replayFixture} = await import('./runtime.mjs');
      const data = await replayFixture('./fixtures/' + fixture + '/');
      for (const field of ['out', 'vi', 'speech', 'allowed']) {
        data[field] = new Uint8Array(data[field].buffer).toBase64();
      }
      return data;
    }, name);
    for (const field of ['out', 'vi', 'speech', 'allowed']) {
      await writeFile(resolve(output, name + '-web-' + field + '.bin'), Buffer.from(result[field], 'base64'));
      delete result[field];
    }
    await writeFile(resolve(output, name + '-web.json'), JSON.stringify(result));
    console.log(name, result.videoMs.length, result.audioMs.length, result.maxAudioBacklog, Date.now() - begin, 'ms');
  }
  await page.locator('#start').click();
  await page.waitForFunction(() => document.querySelector('#cam').classList.contains('is-live'));
  if (!(await page.locator('#status').textContent()).includes('Capturing on this device')) {
    throw Error('Live capture status did not appear in English');
  }
  await page.locator('#camera').evaluate(video => { video.dataset.testCapture = video.srcObject.id; });
  await page.locator('[data-language=zh-TW]').click();
  if (!(await page.locator('#status').textContent()).includes('本機擷取中')) throw Error('Live status did not translate');
  await page.locator('[data-language=en]').click();
  if (!await page.locator('#camera').evaluate(video => video.srcObject?.id === video.dataset.testCapture)) {
    throw Error('Changing language interrupted live capture');
  }
  const videoReady = await page.locator('#camera').evaluate(video => video.videoWidth > 0 && !!video.srcObject);
  if (!videoReady) throw Error('Camera did not start');
  await page.waitForFunction(() => document.querySelector('#db-out').textContent.endsWith('dB'));
  if (cameraFixture) await page.waitForSelector('#people .person');
  if (micFixture) {
    await page.waitForFunction(() => ['#spec-raw', '#spec-out'].every(id => {
      const canvas = document.querySelector(id);
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      return pixels.some((value, i) => i % 4 === 3 && value > 100);
    }));
  }
  await page.locator('[data-view-choice=wave]').click();
  if (await page.locator('#spec-raw').isVisible()) throw Error('Wave view still shows spectra');
  await page.locator('[data-view-choice=spec]').click();
  if (await page.locator('#wave-out').isVisible()) throw Error('Spectrum view still shows waveforms');
  await page.locator('[data-view-choice=both]').click();
  if (!await page.locator('#wave-out').isVisible() || !await page.locator('#spec-out').isVisible()) {
    throw Error('Combined view is missing a signal plot');
  }
  for (const mode of ['raw', 'out', 'off']) {
    const button = page.locator('[data-monitor=' + mode + ']');
    await button.click();
    if (await button.getAttribute('aria-pressed') !== 'true') throw Error('Monitor switch failed');
  }
  await page.screenshot({path: resolve(output, 'monitor-live.png')});
  await page.locator('#camera').evaluate(video => new Promise((resolve, reject) => {
    const startingTime = video.currentTime;
    const timeout = setTimeout(() => reject(Error('Camera did not sustain capture')), 6000);
    const check = () => {
      if (video.currentTime < startingTime + 3) return;
      clearTimeout(timeout);
      video.removeEventListener('timeupdate', check);
      resolve();
    };
    video.addEventListener('timeupdate', check);
  }));
  await page.locator('#stop').click();
  const wavBytes = await page.locator('#download').evaluate(async a => (await (await fetch(a.href)).blob()).size);
  if (wavBytes < 44 + 2.5 * 16000 * 2 || await page.locator('#camera').evaluate(video => !!video.srcObject)) {
    throw Error('Live capture did not stop and produce a WAV');
  }
  if (!/^[\d.]+ s of output$/.test(await page.locator('#recording-note').textContent())) {
    throw Error('Recording length did not translate');
  }
  await page.locator('[data-language=zh-TW]').click();
  if (!/^輸出 [\d.]+ 秒$/.test(await page.locator('#recording-note').textContent())) {
    throw Error('Switching language lost the recording length');
  }
  await page.locator('[data-language=en]').click();
  await page.setViewportSize({width: 390, height: 844});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) {
    throw Error('Mobile layout overflows the viewport');
  }
  await page.screenshot({path: resolve(output, 'monitor-mobile.png'), fullPage: true});
  await page.evaluate(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Blocked', 'NotAllowedError'); };
  });
  await page.locator('#start').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Allow camera and microphone access'));
  await page.locator('[data-language=zh-TW]').click();
  if (!(await page.locator('#status').textContent()).startsWith('請在瀏覽器允許')) {
    throw Error('Permission error did not translate');
  }
  if (unexpected.length) throw Error('Unexpected browser activity: ' + unexpected.join(', '));
  console.log('Live UI: fake camera/microphone, captured WAV bytes:', wavBytes, 'WebSockets/external requests: 0');
} finally {
  await browser.close();
}
