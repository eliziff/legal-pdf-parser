import { readFileSync, statSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { createRequire } from 'node:module';

const { chromium } = createRequire(`${process.cwd()}/package.json`)('playwright');
const [url, listPath, workerArg = '2', roundArg = '1', schedule = 'round-robin'] = process.argv.slice(2);
if (!url || !listPath) throw new Error('usage: node benchmark-kraken-pool.mjs URL image-list.txt [workers]');
const images = readFileSync(listPath, 'utf8').split(/\r?\n/).filter(Boolean);
const workerCount = Number(workerArg);
const rounds = Number(roundArg);
const browser = await chromium.launch({ headless: true });
const pages = await Promise.all(Array.from({ length: workerCount }, async () => {
  const page = await browser.newPage();
  await page.goto(url);
  await page.waitForFunction(() => document.querySelector('#status')?.textContent.includes('Model ready'), null, { timeout: 120000 });
  await page.evaluate(() => { window.krakenPoolCanvases = []; });
  return page;
}));

await Promise.all(pages.map(async (page, index) => {
  await page.setInputFiles('#file', images[index % images.length]);
  await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
  await page.evaluate(() => window.krakenLiteRecognizePage());
  await page.setInputFiles('#file', []);
}));
process.stderr.write(`warmup complete: ${workerCount} workers\n`);

const assignments = pages.map(() => []);
if(schedule==='bytes'){
  const loads=pages.map(()=>0);
  images.map((image,index)=>({image,index,cost:statSync(image).size})).sort((a,b)=>b.cost-a.cost).forEach(item=>{
    const worker=loads.indexOf(Math.min(...loads));assignments[worker].push(item);loads[worker]+=item.cost;
  });
  process.stderr.write(`scheduled by input bytes: ${loads.join(',')}\n`);
}else images.forEach((image, index) => assignments[index % workerCount].push({ image, index }));
await Promise.all(pages.map(async (page, worker) => {
  let decoded = 0;
  for (const item of assignments[worker]) {
    await page.setInputFiles('#file', item.image);
    await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
    await page.evaluate(() => {
      const source = document.querySelector('#page'), canvas = document.createElement('canvas');
      canvas.width = source.width; canvas.height = source.height;
      canvas.getContext('2d').drawImage(source, 0, 0);
      window.krakenPoolCanvases.push(canvas);
    });
    decoded += 1;
    if (!(decoded % 10) || decoded === assignments[worker].length) process.stderr.write(`worker ${worker + 1}: decoded ${decoded}/${assignments[worker].length}\n`);
  }
}));

const runs = [];
for (let round = 0; round < rounds; round += 1) {
  const started = performance.now();
  const workerResults = await Promise.all(pages.map(page => page.evaluate(
    () => window.krakenLiteRecognizePages(window.krakenPoolCanvases),
  )));
  const seconds = (performance.now() - started) / 1000;
  runs.push({ seconds, workerResults });
  process.stderr.write(`round ${round + 1}/${rounds}: ${seconds.toFixed(3)}s\n`);
  process.stderr.write(`${workerResults.map((result, worker) =>
    `w${worker + 1}=${result.seconds.toFixed(3)}s/${result.pages.reduce((sum, page) => sum + page.line_count, 0)} lines/c${result.confidence_median?.toFixed(2) ?? '-'}/r${result.retried ?? 0}`).join(' ')}\n`);
}
const { seconds, workerResults } = [...runs].sort((a, b) => a.seconds - b.seconds)[Math.floor(runs.length / 2)];
const results = new Array(images.length);
workerResults.forEach((result, worker) => result.pages.forEach((value, offset) => {
  results[assignments[worker][offset].index] = value;
}));
results.forEach((result, index) => process.stdout.write(`${JSON.stringify({
  image: images[index],
  kraken: { seconds: seconds / images.length, text: result.text },
  tesseract: null,
  shared: null,
  layout: null,
})}\n`));
await browser.close();
