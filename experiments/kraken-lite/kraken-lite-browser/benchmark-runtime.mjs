import { chromium } from 'playwright';

const [url, image, repetitions = '6'] = process.argv.slice(2);
if (!url || !image) throw new Error('usage: node benchmark-runtime.mjs URL image.png [repetitions]');
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto(url);
await page.waitForFunction(() => document.querySelector('#status')?.textContent.includes('Model ready'));
await page.setInputFiles('#file', image);
await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
await page.evaluate(() => window.krakenLiteRecognizePage());
const seconds = [];
for (let i = 0; i < Number(repetitions); i++) {
  seconds.push((await page.evaluate(() => window.krakenLiteRecognizePage())).seconds);
}
seconds.sort((a, b) => a - b);
console.log(JSON.stringify({ url, seconds, median: seconds[Math.floor(seconds.length / 2)] }));
await browser.close();
