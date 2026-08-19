import { createRequire } from 'node:module';

const { chromium } = createRequire(`${process.cwd()}/package.json`)('playwright');
const [url, repetitionsArg = '5'] = process.argv.slice(2);
if (!url) throw new Error('usage: node benchmark-startup.mjs URL [repetitions]');
const browser = await chromium.launch({ headless: true });
const milliseconds = [];
for (let run = 0; run < Number(repetitionsArg); run += 1) {
  const context = await browser.newContext();
  const page = await context.newPage();
  const started = performance.now();
  await page.goto(url);
  try {
    await page.waitForFunction(() => /Model (ready|failed)/.test(document.querySelector('#status')?.textContent || ''), null, { timeout: 120000 });
    const status = await page.locator('#status').textContent();
    if (!status.includes('Model ready')) throw new Error(status);
  } catch (error) {
    const status = await page.locator('#status').textContent().catch(() => '<missing>');
    throw new Error(`startup failed with status ${JSON.stringify(status)}`, { cause: error });
  }
  milliseconds.push(performance.now() - started);
  await context.close();
  process.stderr.write(`startup ${run + 1}/${repetitionsArg}: ${milliseconds.at(-1).toFixed(0)}ms\n`);
}
milliseconds.sort((a, b) => a - b);
console.log(JSON.stringify({ url, milliseconds, median: milliseconds[Math.floor(milliseconds.length / 2)] }));
await browser.close();
