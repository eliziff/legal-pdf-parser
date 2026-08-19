import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const { chromium }=createRequire(`${process.cwd()}/package.json`)('playwright');
let [target,png,pdf]=process.argv.slice(2);
if(!target||!png||!pdf)throw new Error('usage: node smoke-product.mjs URL_OR_HTML PNG TWO_PAGE_PDF');
if(!target.includes('://'))target=pathToFileURL(target).href;
const browser=await chromium.launch({headless:true}),page=await browser.newPage(),messages=[];page.on('console',message=>messages.push(message.text()));page.on('pageerror',error=>messages.push(error.message));
await page.goto(target);try{await page.waitForFunction(()=>/Model (ready|failed)/.test(document.querySelector('#status')?.textContent||''),null,{timeout:15000})}catch(error){throw new Error(`startup timed out at ${JSON.stringify(await page.locator('#status').textContent())}; console=${JSON.stringify(messages)}`,{cause:error})}
let status=await page.locator('#status').textContent();if(!status.includes('Model ready'))throw new Error(`${status}; console=${JSON.stringify(messages)}`);
await page.setInputFiles('#file',png);await page.waitForFunction(()=>document.querySelector('#status')?.textContent==='Page ready.');
const pngResult=await page.evaluate(()=>window.krakenLiteRecognizePage());if(pngResult.text.length<100)throw new Error(`PNG OCR returned only ${pngResult.text.length} characters`);
await page.setInputFiles('#file',pdf);await page.waitForFunction(()=>document.querySelector('#status')?.textContent==='Page ready.');await page.click('#all');await page.waitForFunction(()=>document.querySelector('#status')?.textContent.startsWith('Done · 2 pages'),null,{timeout:120000});
const pairs=await page.locator('#document .pair').count(),texts=await page.locator('#document .pair pre').allTextContents();if(pairs!==2||texts.some(text=>text.length<100))throw new Error(`PDF streaming failed: ${pairs} pairs, text lengths ${texts.map(text=>text.length)}`);
console.log(JSON.stringify({target,startup:status,png_chars:pngResult.text.length,png_seconds:pngResult.seconds,pdf_pairs:pairs,pdf_chars:texts.map(text=>text.length),status:await page.locator('#status').textContent()}));await browser.close();
