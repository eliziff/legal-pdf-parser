import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const { chromium } = createRequire(`${process.cwd()}/package.json`)('playwright');
const [
  url,
  listPath,
  mode = '.7',
  lane = 'system',
  runTesseract = 'true',
  tessData = 'fast',
  tessPsm = 'AUTO',
  tessOutput = 'blocks',
  tessInput = 'canvas',
  tessWorkers = '1',
  tessRounds = '1',
] = process.argv.slice(2);
if (!url || !listPath) throw new Error('usage: node benchmark-browser-paired.mjs URL image-list.txt [mode] [system|shared|both|layout]');
const images = readFileSync(listPath, 'utf8').split(/\r?\n/).filter(Boolean);
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
page.on('console', message => process.stderr.write(`browser ${message.type()}: ${message.text()}\n`));
page.on('pageerror', error => process.stderr.write(`browser error: ${error.message}\n`));
const tessCoreRequests = [];
page.on('request', (request) => {
  if (request.url().includes('tesseract-core-')) tessCoreRequests.push(request.url());
});
process.stderr.write('browser ready\n');
await page.goto(url);
process.stderr.write('page loaded\n');
await page.waitForFunction(() => document.querySelector('#status')?.textContent.includes('Model ready'), null, { timeout: 120000 });
process.stderr.write('kraken ready\n');
await page.locator(`[name="mode"][value="${mode}"]`).check();
const needTesseract = runTesseract === 'true' || lane === 'layout';
if (needTesseract) await page.addScriptTag({ url: '/node_modules/tesseract.js/dist/tesseract.min.js' });
await page.evaluate(async ({ benchmarkLane, needTesseract, tessConfig }) => {
  window.benchmarkLane = benchmarkLane;
  window.tessConfig = tessConfig;
  window.tessOutput = tessConfig.output === 'blocks' ? { text: true, blocks: true } : { text: true };
  window.tessSource = async () => tessConfig.input === 'png'
    ? new Uint8Array(await document.querySelector('#file').files[0].arrayBuffer())
    : document.querySelector('#page');
  const psm = needTesseract ? Tesseract.PSM[tessConfig.psm] : null;
  if (needTesseract && !psm) throw new Error(`unknown Tesseract PSM: ${tessConfig.psm}`);
  const dataConfig = tessConfig.data === 'best-int'
    ? { path: '/dist/tessdata-best-int', gzip: true }
    : { path: '/dist/tessdata', gzip: false };
  window.tesseractWorkers = [];
  for (let i = 0; needTesseract && i < tessConfig.workers; i += 1) {
    const worker = await Tesseract.createWorker('eng', Tesseract.OEM.LSTM_ONLY, {
      workerPath: '/node_modules/tesseract.js/dist/worker.min.js',
      corePath: '/node_modules/tesseract.js-core',
      langPath: dataConfig.path,
      cachePath: `tess-${tessConfig.data}`,
      gzip: dataConfig.gzip,
    });
    await worker.setParameters({ tessedit_pageseg_mode: psm });
    window.tesseractWorkers.push(worker);
  }
  window.tesseractWorker = window.tesseractWorkers[0];
  if (window.tesseractWorkers.length > 1) {
    window.tesseractScheduler = Tesseract.createScheduler();
    window.tesseractWorkers.forEach((worker) => window.tesseractScheduler.addWorker(worker));
  }
  if (['shared','both'].includes(window.benchmarkLane)) window.tesseractLineWorker = await Tesseract.createWorker('eng', Tesseract.OEM.LSTM_ONLY, {
    workerPath: '/node_modules/tesseract.js/dist/worker.min.js',
    corePath: '/node_modules/tesseract.js-core',
    langPath: '/dist/tessdata',
    gzip: false,
  });
  if (window.tesseractLineWorker) await window.tesseractLineWorker.setParameters({ tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE });
  window.layoutRecognize = async () => {
    const page = document.querySelector('#page'), started = performance.now();
    const found = await window.tesseractWorker.recognize(page, {}, { text: false, layoutBlocks: true });
    const boxes = found.data.layoutBlocks.flatMap(b => b.paragraphs.flatMap(p => p.lines.map(l => l.bbox)));
    const lines = boxes.map(b => { const c = document.createElement('canvas'), x = Math.max(0, b.x0-10), y = Math.max(0, b.y0-6); c.width = Math.min(page.width, b.x1+11)-x; c.height = Math.min(page.height, b.y1+7)-y; c.getContext('2d').drawImage(page,x,y,c.width,c.height,0,0,c.width,c.height); return c; });
    return { seconds: (performance.now()-started)/1000, text: await window.krakenLiteRecognizeLines(lines) };
  };
}, {
  benchmarkLane: lane,
  needTesseract,
  tessConfig: {
    data: tessData,
    psm: tessPsm,
    output: tessOutput,
    input: tessInput,
    workers: Number(tessWorkers),
    rounds: Number(tessRounds),
  },
});
process.stderr.write('tesseract ready\n');
process.stderr.write(`tesseract core ${tessCoreRequests.at(-1) || 'not loaded'}\n`);

// One unscored page warms both persistent sessions under the same policy.
await page.setInputFiles('#file', images[0]);
await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
await page.evaluate(async () => {
  await window.krakenLiteRecognizePage();
  if (window.tesseractWorker) {
    const source = await window.tessSource();
    for (const worker of window.tesseractWorkers) {
      await worker.recognize(source, {}, window.tessOutput);
    }
  }
  if (window.tesseractLineWorker) {
    const lines = window.krakenLiteLineCanvases();
    await window.krakenLiteRecognizeLines(lines);
    await window.tesseractLineWorker.recognize(lines[0]);
  }
  if (window.benchmarkLane === 'layout') await window.layoutRecognize();
});
process.stderr.write('warmup complete\n');
await page.setInputFiles('#file', []);

if (Number(tessWorkers) > 1 && lane === 'system') {
  const buffered = [];
  await page.evaluate(() => { window.tessPoolSources = []; window.krakenPoolCanvases = []; });
  for (const image of images) {
    await page.setInputFiles('#file', image);
    await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
    process.stderr.write(`page ready ${image}\n`);
    await page.evaluate(async (includeTesseract) => {
      const source = document.querySelector('#page'), canvas = document.createElement('canvas');
      canvas.width = source.width; canvas.height = source.height;
      canvas.getContext('2d').drawImage(source, 0, 0);
      window.krakenPoolCanvases.push(canvas);
      if (includeTesseract) window.tessPoolSources.push(await window.tessSource());
    }, runTesseract === 'true');
    buffered.push({ image });
  }
  const krakenPooled = await page.evaluate(() => window.krakenLiteRecognizePages(window.krakenPoolCanvases));
  const pooled = runTesseract !== 'true' ? null : await page.evaluate(async () => {
    const runs = [];
    for (let round = 0; round < window.tessConfig.rounds; round += 1) {
      const started = performance.now();
      const results = await Promise.all(window.tessPoolSources.map((source) => (
        window.tesseractScheduler.addJob('recognize', source, {}, window.tessOutput)
      )));
      runs.push({ seconds: (performance.now() - started) / 1000, results });
    }
    const { seconds, results } = [...runs].sort((a, b) => a.seconds - b.seconds)[Math.floor(runs.length / 2)];
    return results.map((result) => ({
      seconds: seconds / results.length,
      text: result.data.text,
      blocks: (result.data.blocks || []).map(({ text, bbox }) => ({ text, bbox })),
    }));
  });
  buffered.forEach(({ image }, index) => {
    const kraken = {
      seconds: krakenPooled.seconds / buffered.length,
      layout_seconds: krakenPooled.layout_seconds / buffered.length,
      inference_seconds: krakenPooled.inference_seconds / buffered.length,
      text: krakenPooled.pages[index].text,
    };
    process.stdout.write(`${JSON.stringify({ image, kraken, tesseract: pooled?.[index] || null, shared: null, layout: null })}\n`);
  });
} else for (const image of images) {
  await page.setInputFiles('#file', image);
  await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
  process.stderr.write(`page ready ${image}\n`);
  const kraken = lane === 'shared' ? null : await page.evaluate(() => window.krakenLiteRecognizePage());
  const tesseract = lane === 'shared' || runTesseract !== 'true' ? null : await page.evaluate(async () => {
    const source = await window.tessSource();
    const started = performance.now();
    const result = await window.tesseractWorker.recognize(source, {}, window.tessOutput);
    return {
      seconds: (performance.now() - started) / 1000,
      text: result.data.text,
      blocks: (result.data.blocks || []).map(({ text, bbox }) => ({ text, bbox })),
    };
  });
  const shared = !['shared','both'].includes(lane) ? null : await page.evaluate(async runTess => {
    const lines = window.krakenLiteLineCanvases();
    let started = performance.now();
    const krakenText = await window.krakenLiteRecognizeLines(lines);
    const krakenSeconds = (performance.now() - started) / 1000;
    if (!runTess) return { kraken: { seconds: krakenSeconds, text: krakenText }, tesseract: null };
    started = performance.now(); const texts = [];
    for (const line of lines) texts.push((await window.tesseractLineWorker.recognize(line)).data.text.trim());
    return { kraken: { seconds: krakenSeconds, text: krakenText }, tesseract: { seconds: (performance.now() - started) / 1000, text: texts.join('\n') } };
  }, runTesseract === 'true');
  const layout = lane === 'layout' ? await page.evaluate(() => window.layoutRecognize()) : null;
  process.stdout.write(`${JSON.stringify({ image, kraken, tesseract, shared, layout })}\n`);
}
await page.evaluate(() => window.tesseractScheduler?.terminate() || window.tesseractWorker?.terminate());
await page.evaluate(() => window.tesseractLineWorker?.terminate());
await browser.close();
