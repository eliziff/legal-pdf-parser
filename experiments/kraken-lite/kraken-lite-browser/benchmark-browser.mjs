import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const { chromium } = createRequire(`${process.cwd()}/package.json`)('playwright');

const [url, listPath, mode = '.5'] = process.argv.slice(2);
if (!url || !listPath) throw new Error('usage: node benchmark-browser.mjs URL image-list.txt');
const images = readFileSync(listPath, 'utf8').split(/\r?\n/).filter(Boolean);
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto(url);
await page.waitForFunction(() => document.querySelector('#status')?.textContent.includes('Model ready'), null, { timeout: 120000 });
await page.locator(`[name="mode"][value="${mode}"]`).check();

for (const image of images) {
  await page.setInputFiles('#file', image);
  await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
  const start = performance.now();
  await page.click('#ocr');
  await page.waitForFunction(() => document.querySelector('#status')?.textContent.startsWith('Done in'), null, { timeout: 120000 });
  const text = await page.locator('#text').textContent();
  process.stdout.write(`${JSON.stringify({ image, seconds: (performance.now() - start) / 1000, text })}\n`);
}
await browser.close();
