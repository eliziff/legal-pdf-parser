import { chromium } from 'playwright';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on('console', message => console.error(`browser console: ${message.type()}: ${message.text()}`));
page.on('pageerror', error => console.error(`browser error: ${error.stack || error.message}`));
page.on('requestfailed', request => console.error(`request failed: ${request.url()} ${request.failure()?.errorText}`));
const requestedTarget = process.argv[2];
const target = requestedTarget?.startsWith('http')
  ? requestedTarget
  : requestedTarget
    ? pathToFileURL(requestedTarget).href
    : 'http://127.0.0.1:8771/';
await page.goto(target);
try {
  await page.waitForFunction(() => document.querySelector('#status')?.textContent.includes('Model ready'), null, { timeout: 15000 });
} catch (error) {
  console.error(`status: ${await page.locator('#status').textContent()}`);
  throw error;
}
const modes = await page.locator('[name="mode"]').evaluateAll(items => items.map(item => item.value));
if (JSON.stringify(modes) !== JSON.stringify(['1', '.85', '.7', '.62'])) throw new Error(`unexpected modes: ${modes}`);
await page.setInputFiles('#file', '../kraken-lite-native/courtlistener-scan-silver/4327746/page-003.png');
await page.waitForFunction(() => document.querySelector('#status')?.textContent === 'Page ready.');
await page.locator('#form').evaluate(form => form.requestSubmit());
await page.waitForFunction(() => document.querySelector('#status')?.textContent.startsWith('Done in'), null, { timeout: 120000 });
const characters = (await page.locator('#text').textContent()).length;
const output = await page.locator('#text').textContent();
const sha256 = createHash('sha256').update(output).digest('hex');
if (characters < 100) throw new Error(`unexpectedly short OCR result: ${characters}`);
await page.screenshot({ path: 'ui-final.png', fullPage: true });
console.log(JSON.stringify({ target, status: await page.locator('#status').textContent(), modes, characters, sha256 }));
await browser.close();
