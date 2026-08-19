import { chromium } from 'playwright';

const base = process.argv[2] || 'http://127.0.0.1:8786/ep-probe.html';
const browser = await chromium.launch({
  headless: !process.argv.includes('--headed'),
  args: process.argv.includes('--unsafe-webgpu') ? ['--enable-unsafe-webgpu', '--use-angle=swiftshader'] : [],
});
for (const ep of ['webgpu', 'webnn']) {
  const page = await browser.newPage();
  await page.goto(`${base}?ep=${ep}&fallback=1`);
  await page.waitForFunction(() => document.body.dataset.done === '1', null, { timeout: 120000 });
  console.log(await page.locator('body').textContent());
  await page.close();
}
await browser.close();
