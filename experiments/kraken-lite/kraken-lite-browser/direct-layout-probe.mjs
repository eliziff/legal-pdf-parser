import { createRequire } from 'node:module';

const { chromium } = createRequire(`${process.cwd()}/package.json`)('playwright');
const [url, image] = process.argv.slice(2);
if (!url || !image) throw new Error('usage: node direct-layout-probe.mjs URL image.png');

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto(url);
await page.setInputFiles('#file', image);
await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
const result = await page.evaluate(async () => {
  const canvas = document.querySelector('#page');
  const { TesseractLayout } = await import('/tesseract-layout.js');
  const layout = new TesseractLayout();
  const started = performance.now();
  const lines = await layout.findLines(canvas);
  layout.terminate();
  return { lines: lines.length, boxes: lines, seconds: (performance.now() - started) / 1000 };
});
process.stdout.write(`${JSON.stringify(result)}\n`);
await browser.close();
