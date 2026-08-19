import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const { chromium } = createRequire(`${process.cwd()}/package.json`)('playwright');
const [url, listPath, configArg, roundsArg = '2', outputPath] = process.argv.slice(2);
if (!url || !listPath || !configArg || !outputPath) {
  throw new Error('usage: node tune-crosspage.mjs URL xml-list.txt batch:bucket,... [rounds] output.json');
}
const images = readFileSync(listPath, 'utf8').split(/\r?\n/).filter(Boolean).map(path => path.replace(/\.xml$/i, '.png'));
const configs = configArg.split(',').map(value => {
  const [batch, bucket] = value.split(':').map(Number);
  return { batch, bucket };
});
const rounds = Number(roundsArg);
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto(url);
await page.waitForFunction(() => document.querySelector('#status')?.textContent.includes('Model ready'), null, { timeout: 120000 });
await page.locator('[name="mode"][value=".7"]').check();
await page.evaluate(() => { window.tuningCanvases = []; });
for (const image of images) {
  await page.setInputFiles('#file', image);
  await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
  await page.evaluate(() => {
    const source = document.querySelector('#page'), canvas = document.createElement('canvas');
    canvas.width = source.width; canvas.height = source.height;
    canvas.getContext('2d').drawImage(source, 0, 0);
    window.tuningCanvases.push(canvas);
  });
}
await page.evaluate(async () => {
  window.tuningPrepared = await window.krakenLitePreparePages(window.tuningCanvases);
  await window.krakenLiteRecognizePreparedPages(window.tuningPrepared);
});
const runs = [];
let referenceTexts;
for (let round = 0; round < rounds; round += 1) {
  const order = [...configs.slice(round % configs.length), ...configs.slice(0, round % configs.length)];
  if (round % 2) order.reverse();
  for (const config of order) {
    const result = await page.evaluate(async value => {
      window.krakenLiteSetTuning(value);
      return window.krakenLiteRecognizePreparedPages(window.tuningPrepared);
    }, config);
    const texts = result.pages.map(item => item.text);
    referenceTexts ||= texts;
    const sameText = texts.length === referenceTexts.length && texts.every((text, index) => text === referenceTexts[index]);
    if (!sameText) throw new Error(`OCR output changed for batch ${config.batch}, bucket ${config.bucket}`);
    runs.push({ round: round + 1, ...config, seconds: result.inference_seconds, same_text: sameText });
    process.stderr.write(`round ${round + 1}: batch ${config.batch}, bucket ${config.bucket}, ${result.inference_seconds.toFixed(3)}s\n`);
  }
}
writeFileSync(outputPath, JSON.stringify({
  protocol: { pages: images.length, rounds, mode: '.7', url, timing: 'prepared line preprocessing + cross-page inference; layout excluded', all_outputs_identical: true },
  reference_texts: referenceTexts,
  runs,
}, null, 2));
await browser.close();
