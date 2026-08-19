import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { createRequire } from 'node:module';

const { chromium }=createRequire(`${process.cwd()}/package.json`)('playwright');
const [url,listPath,roundArg='3',mode='1']=process.argv.slice(2),images=readFileSync(listPath,'utf8').split(/\r?\n/).filter(Boolean),rounds=Number(roundArg);
if(!url||!listPath)throw new Error('usage: node benchmark-kraken-worker-pool.mjs URL image-list.txt [rounds] [mode]');
const browser=await chromium.launch({headless:true}),page=await browser.newPage();await page.goto(url);
await page.waitForFunction(()=>document.querySelector('#status')?.textContent.includes('Model ready'),null,{timeout:120000});
await page.locator(`[name="mode"][value="${mode}"]`).check();
await page.evaluate(()=>{window.krakenPoolCanvases=[]});
for(let index=0;index<images.length;index++){
  await page.setInputFiles('#file',images[index]);await page.waitForFunction(()=>document.querySelector('#status')?.textContent==='Page ready.');
  await page.evaluate(()=>{const source=document.querySelector('#page'),canvas=document.createElement('canvas');canvas.width=source.width;canvas.height=source.height;canvas.getContext('2d').drawImage(source,0,0);window.krakenPoolCanvases.push(canvas)});
  if(!((index+1)%10)||index+1===images.length)process.stderr.write(`decoded ${index+1}/${images.length}\n`);
}
const poolSize=Math.max(1,Number(new URL(url).searchParams.get('pool'))||7);
await page.evaluate(count=>window.krakenLiteRecognizePagesPooled(window.krakenPoolCanvases.slice(0,count)),Math.min(poolSize,images.length));process.stderr.write(`worker pool warmup complete (${Math.min(poolSize,images.length)} pages)\n`);
const runs=[];for(let round=0;round<rounds;round++){const started=performance.now(),result=await page.evaluate(()=>window.krakenLiteRecognizePagesPooled(window.krakenPoolCanvases)),seconds=(performance.now()-started)/1000;runs.push({seconds,result});process.stderr.write(`round ${round+1}/${rounds}: ${seconds.toFixed(3)}s (layout wall ${result.layout_wall_seconds.toFixed(3)}s)\n`)}
const {seconds,result}=[...runs].sort((a,b)=>a.seconds-b.seconds)[Math.floor(runs.length/2)];result.pages.forEach((value,index)=>process.stdout.write(`${JSON.stringify({image:images[index],kraken:{seconds:seconds/images.length,text:value.text},tesseract:null,shared:null,layout:null})}\n`));await browser.close();
