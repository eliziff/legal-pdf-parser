import * as ort from 'onnxruntime-web/wasm';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist/build/pdf.mjs';
import { TesseractLayout } from './tesseract-layout.js';
import { orderLayoutLines } from './layout-order.js';
import { preprocessLine } from './line-preprocess.js';
import { recognitionWorkers } from './worker-policy.js';

const embedded = globalThis.KRAKEN_LITE_ASSETS;
const experimental = new URLSearchParams(location.search);
const bundledLayout = !embedded || Boolean(embedded.layoutCore);
const workerPoolSize = experimental.has('pool')?Number(experimental.get('pool')):bundledLayout?recognitionWorkers(navigator.hardwareConcurrency):0;
const fullRuntime = experimental.get('runtime') === 'full';
const runtimeStem = !experimental.has('runtime')||experimental.get('runtime') === 'relaxed' ? 'ort-wasm-relaxedsimd'
  : experimental.get('runtime') === 'qsimd' ? 'ort-wasm-qsimd' : 'ort-wasm-simd';
const debugRuntime = experimental.get('debug') === '1';
const requestedThreads = Number(experimental.get('threads')) || (workerPoolSize?1:4);
const threadCount = crossOriginIsolated ? Math.min(requestedThreads, navigator.hardwareConcurrency || 4) : 1;
const runtimeSuffix = ['ort-wasm-relaxedsimd', 'ort-wasm-qsimd'].includes(runtimeStem) || threadCount > 1 ? '-threaded' : '';
let batchSize = Number(experimental.get('batch')) || 32;
let bucketSize = Number(experimental.get('bucket')) || 24;
const yieldEachBatch = experimental.get('yield') === '1';
const sortedBatches = experimental.get('sort') === '1';
const inputPadding = experimental.has('pad') ? Number(experimental.get('pad')) : 16;
const scaleOverride = Number(experimental.get('scale')) || 0;
const quantModel = experimental.get('quant') !== '0';
const argmaxModel = experimental.get('argmax') === '1' || experimental.get('top1') === '1';
const quantVariant = experimental.get('qvariant');
const preprocessing = experimental.get('pre') || 'none';
const modelPath = experimental.get('model');
const graphOptimizationLevel = experimental.get('opt') || 'all';
const fixedWidths = (experimental.get('widths') || '').split(',').map(Number).filter(Boolean).sort((a,b)=>a-b);
const reuseBuffers = experimental.get('reuse') !== '0';
const atlasBatch = experimental.get('atlas') !== '0';
const layoutWorkerCount = Math.max(1,Number(experimental.get('layoutWorkers'))||(workerPoolSize?2:1));
const layoutCoreStem = experimental.get('layoutCore')||'layout-core-dpi';
const layoutScale = Number(experimental.get('layoutScale'))||1;
const trimLayout = experimental.get('layoutTrim')==='1';
const layoutPsm = Number(experimental.get('layoutPsm'))||3;
const layoutBinaryThreshold = Number(experimental.get('layoutBinary'))||0;
const retryThreshold = Number(experimental.get('retry')) || 0;
const retryScale = Number(experimental.get('retryScale')) || 1;
const collectConfidence = retryThreshold > 0 || experimental.get('confidence') === '1';
const decodeAsset = value => Uint8Array.from(atob(value), character => character.charCodeAt(0));
const assetUrl = (name, fallback, type = 'application/javascript') => embedded?.[name]
  ? `data:${type};base64,${embedded[name]}` : fallback;
const assetBytes = async (name, fallback) => embedded?.[name] ? decodeAsset(embedded[name]) : fetch(fallback).then(response => response.arrayBuffer());

GlobalWorkerOptions.workerSrc = assetUrl('pdfWorker', './dist/pdf.worker.min.mjs');
const runtimePaths = embedded ? {
  mjs: assetUrl(threadCount > 1 ? 'ortThreadedMjs' : 'ortMjs', '', 'application/javascript'),
  wasm: assetUrl(threadCount > 1 ? 'ortThreadedWasm' : 'ortWasm', '', 'application/wasm'),
} : fullRuntime ? new URL('./node_modules/onnxruntime-web/dist/', location.href).href : {
  mjs: new URL(`./dist/${runtimeStem}${runtimeSuffix}.mjs`, location.href).href,
  wasm: new URL(`./dist/${runtimeStem}${runtimeSuffix}.wasm`, location.href).href,
};
ort.env.wasm.wasmPaths = runtimePaths;
ort.env.wasm.numThreads = threadCount;
if (debugRuntime) ort.env.logLevel = 'verbose';

const $ = id => document.getElementById(id);
const pageCanvas = $('page'), overlay = $('overlay'), status = $('status'), text = $('text');
if (experimental.has('seg')) $('segmentation').value = experimental.get('seg');
const context = pageCanvas.getContext('2d'), overlayContext = overlay.getContext('2d');
let session, modelBytes, labels, pdf, imageBitmap, pageNumber = 1, crop, cropTemplate, dragStart;
let sessionPromise;
let poolReady=Promise.resolve(),poolQueue=[],idleWorkers=[],poolTaskId=0;
let tensorScratch = new Float32Array(), lengthsScratch = new BigInt64Array();
const fixedSessions = new Map();
const lineHeight = 48;
const batchCanvas=document.createElement('canvas'),batchContext=batchCanvas.getContext('2d',{willReadFrequently:true});
const hasTesseractLayout = !embedded || Boolean(embedded.layoutCore);
const layoutOptions={workerPath:assetUrl('layoutWorker','./tesseract-layout-worker.js'),corePath:assetUrl('layoutCore',`./dist/${layoutCoreStem}.mjs`),wasmPath:assetUrl('layoutWasm',`./dist/${layoutCoreStem}.wasm`,'application/wasm'),sourceResolution:Math.round(200*layoutScale),trim:trimLayout,psm:layoutPsm,binaryThreshold:layoutBinaryThreshold};
const layouts=hasTesseractLayout?Array.from({length:layoutWorkerCount},()=>new TesseractLayout(layoutOptions)):[];
let idleLayouts=[],layoutQueue=[];
idleLayouts.push(...layouts);
const layoutReady = Promise.resolve();
if (!hasTesseractLayout) {
  $('segmentation').querySelector('[value="tesseract"]').remove();
  $('segmentation').value = 'fast';
}

const ready = (async () => {
  const [codec, model] = await Promise.all([
    embedded ? JSON.parse(new TextDecoder().decode(decodeAsset(embedded.codec))) : fetch('./dist/codec.json').then(r => r.json()),
    assetBytes('model', modelPath || (quantVariant ? `./dist/quant-candidates-ort/${quantVariant}.ort` : quantModel
      ? argmaxModel ? './dist/ort-argmax/model.dynamic-u8.argmax.ort' : './dist/optimized-models-ort/lstm-channel-preprocessed.ort'
      : argmaxModel ? './dist/ort-argmax/model.argmax.ort' : './dist/ort-minimal-wasm/model.ort')),
    layoutReady,
  ]);
  modelBytes = model;
  labels = Object.fromEntries(Object.entries(codec).flatMap(([character, ids]) => ids.map(id => [id, character])));
  if(workerPoolSize)poolReady=initializeRecognitionPool(codec,model);
  else await ensureMainSession();
  await poolReady;
  status.textContent = 'Model ready. Select a PNG or PDF.';
  $('ocr').disabled = !$('file').files.length;
  $('all').disabled = !$('file').files.length;
})();
ready.catch(error => { status.textContent = `Model failed to load: ${error.message}`; });

function ensureMainSession(){
  return sessionPromise||=(ort.InferenceSession.create(modelBytes,{executionProviders:['wasm'],graphOptimizationLevel,...(debugRuntime&&{logSeverityLevel:0,logVerbosityLevel:1})}).then(value=>session=value));
}

function pumpRecognitionPool(){
  while(idleWorkers.length&&poolQueue.length){const slot=idleWorkers.pop(),task=poolQueue.shift();slot.task=task;slot.worker.postMessage({type:'recognize',id:task.id,bitmap:task.bitmap,lines:task.lines,scale:task.scale},[task.bitmap])}
}

function pumpLayoutPool(){
  while(idleLayouts.length&&layoutQueue.length){const worker=idleLayouts.pop(),task=layoutQueue.shift();worker.findLines(task.canvas).then(task.resolve,task.reject).finally(()=>{idleLayouts.push(worker);pumpLayoutPool()})}
}

async function findLayoutLines(canvas){
  await layoutReady;return new Promise((resolve,reject)=>{layoutQueue.push({canvas,resolve,reject});pumpLayoutPool()});
}

async function initializeRecognitionPool(codec,model){
  const workerUrl=assetUrl('recognitionWorker','./dist/recognition-worker.js');
  const slots=await Promise.all(Array.from({length:workerPoolSize},()=>new Promise((resolve,reject)=>{
    const worker=new Worker(workerUrl),slot={worker,task:null};
    worker.onmessage=({data})=>{if(data.type==='ready'){idleWorkers.push(slot);resolve(slot);pumpRecognitionPool();return}const task=slot.task;slot.task=null;if(data.type==='error'){if(task)task.reject(new Error(data.message));else reject(new Error(data.message));return}task.resolve(data.text);idleWorkers.push(slot);pumpRecognitionPool()};worker.onerror=event=>reject(new Error(event.message||`recognition worker failed at ${event.filename||'unknown source'}:${event.lineno||0}`));
    worker.postMessage({type:'init',runtimeMjs:runtimePaths.mjs,runtimeWasm:runtimePaths.wasm,model:model.slice(0),codec,batchSize,bucketSize,padding:inputPadding});
  })));
  return slots;
}

async function recognizeInPool(bitmap,lines,scale){
  await poolReady;return new Promise((resolve,reject)=>{poolQueue.push({id:++poolTaskId,bitmap,lines,scale,resolve,reject});pumpRecognitionPool()});
}

function setCanvasSize(width, height) {
  pageCanvas.width = overlay.width = width; pageCanvas.height = overlay.height = height;
}

async function renderPage(number) {
  if (pdf) {
    const page = await pdf.getPage(number), viewport = page.getViewport({ scale: 2 });
    setCanvasSize(Math.round(viewport.width), Math.round(viewport.height));
    await page.render({ canvasContext: context, viewport }).promise;
  } else if (imageBitmap) {
    setCanvasSize(imageBitmap.width, imageBitmap.height); context.drawImage(imageBitmap, 0, 0);
  }
  crop = cropTemplate ? {
    x: cropTemplate.x * pageCanvas.width, y: cropTemplate.y * pageCanvas.height,
    w: cropTemplate.w * pageCanvas.width, h: cropTemplate.h * pageCanvas.height
  } : null;
  drawCrop();
  $('page-label').textContent = `Page ${number}${pdf ? ` of ${pdf.numPages}` : ''}`;
  $('prev').disabled = number <= 1; $('next').disabled = !pdf || number >= pdf.numPages;
}

function point(event) {
  const box = overlay.getBoundingClientRect();
  return { x:(event.clientX-box.left)*overlay.width/box.width, y:(event.clientY-box.top)*overlay.height/box.height };
}
function drawCrop() {
  overlayContext.clearRect(0,0,overlay.width,overlay.height); if(!crop)return;
  overlayContext.fillStyle='rgb(180 35 24 / 18%)';overlayContext.strokeStyle='#b42318';overlayContext.lineWidth=3;
  overlayContext.fillRect(crop.x,crop.y,crop.w,crop.h);overlayContext.strokeRect(crop.x,crop.y,crop.w,crop.h);
}
overlay.addEventListener('pointerdown',e=>{dragStart=point(e);crop=null;overlay.setPointerCapture(e.pointerId)});
overlay.addEventListener('pointermove',e=>{if(!dragStart)return;const p=point(e);crop={x:Math.min(p.x,dragStart.x),y:Math.min(p.y,dragStart.y),w:Math.abs(p.x-dragStart.x),h:Math.abs(p.y-dragStart.y)};drawCrop()});
overlay.addEventListener('pointerup',()=>{dragStart=null;if(crop&&(crop.w<8||crop.h<8))crop=null;cropTemplate=crop?{x:crop.x/overlay.width,y:crop.y/overlay.height,w:crop.w/overlay.width,h:crop.h/overlay.height}:null;drawCrop();status.textContent=crop?'Crop selected for every page.':'Crop cleared.'});
$('clear').onclick=()=>{crop=cropTemplate=null;drawCrop()};

$('file').onchange = async () => {
  const file=$('file').files[0]; if(!file)return;$('ocr').disabled=true;$('all').disabled=true;status.textContent='Loading page…';await ready; pageNumber=1; pdf=null; imageBitmap=null; crop=cropTemplate=null;
  if(file.type==='application/pdf'||file.name.toLowerCase().endsWith('.pdf')) pdf=await getDocument({data:await file.arrayBuffer()}).promise;
  else imageBitmap=await createImageBitmap(file);
  await renderPage(1); $('ocr').disabled=false; $('all').disabled=false; text.textContent='Drag a crop or recognize the whole page.';status.textContent='Page ready.';
};
$('prev').onclick=async()=>{if(pageNumber>1)await renderPage(--pageNumber)};
$('next').onclick=async()=>{if(pdf&&pageNumber<pdf.numPages)await renderPage(++pageNumber)};

function sourceCanvas() {
  if(!crop)return pageCanvas; const canvas=document.createElement('canvas');canvas.width=Math.round(crop.w);canvas.height=Math.round(crop.h);
  canvas.getContext('2d').drawImage(pageCanvas,crop.x,crop.y,crop.w,crop.h,0,0,canvas.width,canvas.height);return canvas;
}

function inkThreshold(pixels) {
  const histogram=new Uint32Array(256);for(let i=0;i<pixels.length;i+=16)histogram[Math.round((pixels[i]+pixels[i+1]+pixels[i+2])/3)]++;
  let total=0,sum=0;for(let i=0;i<256;i++){total+=histogram[i];sum+=i*histogram[i]}let left=0,leftSum=0,best=0,threshold=0;
  for(let i=0;i<256;i++){left+=histogram[i];leftSum+=i*histogram[i];if(!left||left===total)continue;const between=(leftSum/left-(sum-leftSum)/(total-left))**2*left*(total-left);if(between>best){best=between;threshold=i}}
  return threshold>160?threshold:210;
}

function projectionLines(source,pixels,dark,x0=0,y0=0,width=source.width,height=source.height) {
  const sourceWidth=source.width;dark??=inkThreshold(pixels);const inkRows=new Uint16Array(height),low=Math.max(4,width/200),high=Math.max(4,width/60);for(let y=0;y<height;y++)for(let x=0;x<width;x+=2){const i=((y0+y)*sourceWidth+x0+x)*4;if((pixels[i]+pixels[i+1]+pixels[i+2])/3<dark)inkRows[y]++}
  const ranges=(threshold,from=0,to=height)=>{const out=[];let start=-1;for(let y=from;y<=to;y++){if(y<to&&inkRows[y]>=threshold&&start<0)start=y;if((y===to||inkRows[y]<threshold)&&start>=0){if(y-start>=4)out.push([start,y]);start=-1}}return out};
  let bands=ranges(low),sizes=bands.map(([a,b])=>b-a).filter(n=>n<100).sort((a,b)=>a-b),typical=sizes[Math.floor(sizes.length/2)]||40;
  bands=bands.flatMap(([a,b])=>b-a>typical*1.65?ranges(high,a,b):[[a,b]]);
  const lines=[];for(const [start,y] of bands){let left=width,right=0;for(let yy=Math.max(0,start-6);yy<Math.min(height,y+6);yy++)for(let x=0;x<width;x+=2){const i=((y0+yy)*sourceWidth+x0+x)*4;if((pixels[i]+pixels[i+1]+pixels[i+2])/3<dark){left=Math.min(left,x);right=Math.max(right,x)}}left=Math.max(0,left-12);right=Math.min(width,right+14);const c=document.createElement('canvas'),cropY=Math.max(0,start-6);c.width=Math.max(1,right-left);c.height=Math.min(height,y-start+12);c.dataset.y=y0+start;c.getContext('2d').drawImage(source,x0+left,y0+cropY,c.width,c.height,0,0,c.width,c.height);lines.push(c)}
  return lines;
}

function lineCanvases(source) {
  const {width,height}=source, pixels=source.getContext('2d').getImageData(0,0,width,height).data,dark=inkThreshold(pixels);let ruleX=0,ruleInk=0;
  for(let x=Math.floor(width*.55);x<width*.9;x+=2){let ink=0;for(let y=0;y<height;y+=2){const i=(y*width+x)*4;if((pixels[i]+pixels[i+1]+pixels[i+2])/3<160)ink++}if(ink>ruleInk){ruleInk=ink;ruleX=x}}
  if(ruleInk<height*.18){
    const rows=new Uint16Array(height);for(let y=0;y<height;y+=2)for(let x=0;x<width;x+=4){const i=(y*width+x)*4;if((pixels[i]+pixels[i+1]+pixels[i+2])/3<dark)rows[y]++}let top=0,bottom=(height-1)&~1;while(top<bottom&&rows[top]<4)top+=2;while(bottom>top&&rows[bottom]<4)bottom-=2;const activeHeight=bottom-top+1;
    let start=-1,bestStart=0,bestEnd=0;for(let x=Math.floor(width*.25);x<=Math.floor(width*.75);x++){let ink=0;for(let y=top;y<=bottom;y+=4){const i=(y*width+x)*4;if((pixels[i]+pixels[i+1]+pixels[i+2])/3<dark)ink++}if(ink<=activeHeight/80&&start<0)start=x;if((ink>activeHeight/80||x===Math.floor(width*.75))&&start>=0){if(x-start>bestEnd-bestStart){bestStart=start;bestEnd=x}start=-1}}
    if(bestEnd-bestStart>=width*.015){const split=Math.round((bestStart+bestEnd)/2);return [projectionLines(source,pixels,dark,0,0,split,height),projectionLines(source,pixels,dark,split,0,width-split,height)].flat()}
    return projectionLines(source,pixels,dark);
  }
  let top=height,bottom=0;for(let y=0;y<height;y++){const i=(y*width+ruleX)*4;if((pixels[i]+pixels[i+1]+pixels[i+2])/3<160){top=Math.min(top,y);bottom=Math.max(bottom,y)}}
  return [projectionLines(source,pixels,dark,0,0,width,top),projectionLines(source,pixels,dark,0,top,ruleX,bottom-top),projectionLines(source,pixels,dark,ruleX+1,top,width-ruleX-1,bottom-top),projectionLines(source,pixels,dark,0,bottom,width,height-bottom)].flat();
}

function withoutRunningHeader(lines, source) {
  const heights=lines.map(line=>line.height).sort((a,b)=>a-b), typical=heights[Math.floor(heights.length/2)]||1;
  const rule=lines.find(line=>Number(line.dataset.y||0)<source.height*.1&&line.width>source.width*.85&&line.height<typical*.55);
  return rule?lines.filter(line=>Number(line.dataset.y||0)>Number(rule.dataset.y)):lines;
}

async function tesseractLines(source) {
  let layoutSource=source;
  if(layoutScale!==1){layoutSource=document.createElement('canvas');layoutSource.width=Math.max(1,Math.round(source.width*layoutScale));layoutSource.height=Math.max(1,Math.round(source.height*layoutScale));layoutSource.getContext('2d').drawImage(source,0,0,layoutSource.width,layoutSource.height)}
  let boxes=await findLayoutLines(layoutSource);
  if(layoutScale!==1)boxes=boxes.map(box=>({x0:box.x0/layoutScale,y0:box.y0/layoutScale,x1:box.x1/layoutScale,y1:box.y1/layoutScale}));
  boxes = orderLayoutLines(boxes, source.width, source.height);
  if(atlasBatch)return boxes.map(box=>({source,x:Math.max(0,box.x0-10),y:Math.max(0,box.y0-6),width:Math.max(1,Math.min(source.width,box.x1+11)-Math.max(0,box.x0-10)),height:Math.max(1,Math.min(source.height,box.y1+7)-Math.max(0,box.y0-6))}));
  return boxes.map(box => {
    const x = Math.max(0, box.x0 - 10), y = Math.max(0, box.y0 - 6);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.min(source.width, box.x1 + 11) - x);
    canvas.height = Math.max(1, Math.min(source.height, box.y1 + 7) - y);
    canvas.getContext('2d').drawImage(source, x, y, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
    return canvas;
  });
}

function preparedWidth(line,scale){return Math.max(1,Math.round(line.width*lineHeight/line.height*scale))+inputPadding*2}

function drawPrepared(ctx,line,x,y,width){
  const drawn=width-inputPadding*2;
  if(line.source)ctx.drawImage(line.source,line.x,line.y,line.width,line.height,x+inputPadding,y,drawn,lineHeight);
  else ctx.drawImage(line,x+inputPadding,y,drawn,lineHeight);
}

function prepareAtlas(batch,max) {
  batchCanvas.width=max;batchCanvas.height=batch.length*lineHeight;batchContext.fillStyle='white';batchContext.fillRect(0,0,batchCanvas.width,batchCanvas.height);
  batch.forEach((item,n)=>drawPrepared(batchContext,item.line,0,n*lineHeight,item.width));
  const rgba=batchContext.getImageData(0,0,batchCanvas.width,batchCanvas.height).data,tensorLength=batch.length*lineHeight*max;
  if(reuseBuffers&&tensorScratch.length<tensorLength)tensorScratch=new Float32Array(tensorLength);
  const tensor=reuseBuffers?tensorScratch.subarray(0,tensorLength):new Float32Array(tensorLength);
  if(preprocessing==='none')for(let i=0;i<tensorLength;i++){const p=i*4;tensor[i]=1-(rgba[p]+rgba[p+1]+rgba[p+2])/(3*255)}
  else for(let n=0;n<batch.length;n++){const line=new Uint8ClampedArray(lineHeight*max*4);for(let y=0;y<lineHeight;y++){const start=(n*lineHeight+y)*max*4;line.set(rgba.subarray(start,start+max*4),y*max*4)}tensor.set(preprocessLine(line,max,lineHeight,preprocessing),n*lineHeight*max)}
  return tensor;
}

function prepare(line, scale) {
  const height=lineHeight;
  const width=Math.max(1,Math.round(line.width*height/line.height*scale)),padding=inputPadding,inputWidth=width+padding*2;
  const c=document.createElement('canvas');c.width=inputWidth;c.height=height;const ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,inputWidth,height);ctx.drawImage(line,padding,0,width,height);
  const rgba=ctx.getImageData(0,0,inputWidth,height).data,data=preprocessLine(rgba,inputWidth,height,preprocessing);
  return {data,width:inputWidth};
}

function decode(output,n,validSteps) {
  const d=output.dims,data=output.data,classes=d[1],steps=d[d.length-1],stride=classes*steps;let previous=0,result='',confidence=0,emissions=0;
  for(let t=0;t<Math.min(steps,validSteps);t++){let best=0,maximum=-Infinity;for(let c=0;c<classes;c++){const value=data[n*stride+c*steps+t];if(value>maximum){maximum=value;best=c}}if(best&&best!==previous){result+=labels[best]||'';if(collectConfidence){confidence+=maximum;emissions++}}previous=best}return collectConfidence?{text:result,confidence:emissions?confidence/emissions:0}:{text:result};
}

function decodeIds(output,n,validSteps) {
  const steps=output.dims.at(-1),stride=output.size/output.dims[0];let previous=0,result='';
  for(let t=0;t<Math.min(steps,validSteps);t++){const best=Number(output.data[n*stride+t]);if(best&&best!==previous)result+=labels[best]||'';previous=best}
  return {text:result};
}

function cleanModelText(value) {
  return value.replace(/[\u00ad\u00ac]\r?\n/g,'').replace(/[\u00ad\u00ac]/g,'');
}

async function inferenceSession(width) {
  const fixed=fixedWidths.find(value=>value>=width);
  if(!fixed)return {runner:session,width};
  if(!fixedSessions.has(fixed))fixedSessions.set(fixed,ort.InferenceSession.create(modelBytes,{
    executionProviders:['wasm'],graphOptimizationLevel,freeDimensionOverrides:{width:fixed}
  }));
  return {runner:await fixedSessions.get(fixed),width:fixed};
}

async function infer(lines,scale,allowRetry=true) {
  await ensureMainSession();
  const results=new Array(lines.length),items=lines.map((line,index)=>atlasBatch?{line,width:preparedWidth(line,scale),index}:{...prepare(line,scale),index}),groups=new Map();
  if(sortedBatches)groups.set(0,items.sort((a,b)=>a.width-b.width));else items.forEach(item=>{const key=Math.ceil(item.width/bucketSize);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(item)});let done=0;
  for(const group of groups.values())for(let offset=0;offset<group.length;offset+=batchSize){const batch=group.slice(offset,offset+batchSize),selected=await inferenceSession(Math.max(...batch.map(x=>x.width))),max=selected.width,tensorLength=batch.length*lineHeight*max;if(reuseBuffers&&lengthsScratch.length<batch.length)lengthsScratch=new BigInt64Array(batch.length);const lengths=reuseBuffers?lengthsScratch.subarray(0,batch.length):new BigInt64Array(batch.length),tensor=atlasBatch?prepareAtlas(batch,max):(reuseBuffers?(tensorScratch.length<tensorLength&&(tensorScratch=new Float32Array(tensorLength)),tensorScratch.subarray(0,tensorLength)):new Float32Array(tensorLength));if(!atlasBatch){tensor.fill(0);batch.forEach((item,n)=>{for(let y=0;y<lineHeight;y++)tensor.set(item.data.subarray(y*item.width,(y+1)*item.width),n*lineHeight*max+y*max)})}batch.forEach((item,n)=>{lengths[n]=BigInt(item.width)});
    const feeds={image:new ort.Tensor('float32',tensor,[batch.length,1,lineHeight,max]),sequence_lengths:new ort.Tensor('int64',lengths,[batch.length])};
    const out=await selected.runner.run(feeds), logits=out.logits,ids=out.class_ids,output=ids||logits||Object.values(out)[0], outputLengths=out.output_lengths;for(let n=0;n<batch.length;n++)results[batch[n].index]=(ids?decodeIds:decode)(output,n,outputLengths?Number(outputLengths.data[n]):output.dims.at(-1));done+=batch.length;status.textContent=`Recognized ${done} of ${lines.length} lines…`;if(yieldEachBatch)await new Promise(requestAnimationFrame)}
  if(allowRetry&&retryThreshold&&scale<retryScale){const uncertain=[];results.forEach((result,index)=>{if((result.confidence??Infinity)<retryThreshold)uncertain.push(index)});if(uncertain.length){const replacements=await infer(uncertain.map(index=>lines[index]),retryScale,false);uncertain.forEach((index,n)=>{results[index]={...replacements[n],retried:true}})}}
  return results;
}

async function segment(canvas) {
  const started=performance.now(),segmentation=$('segmentation').value;
  const lines=segmentation==='tesseract'?await tesseractLines(canvas):segmentation==='none'?[canvas]:withoutRunningHeader(lineCanvases(canvas),canvas);
  return {lines,seconds:(performance.now()-started)/1000};
}

async function recognize(canvas, timing) {
  await ready;const mode=scaleOverride||document.querySelector('[name="mode"]:checked').value;
  if(workerPoolSize&&$('segmentation').value==='tesseract'){const layoutStarted=performance.now(),{lines}=await segment(canvas);if(timing)timing.layout_seconds=(performance.now()-layoutStarted)/1000;const boxes=lines.map(({x,y,width,height})=>({x,y,width,height})),inferenceStarted=performance.now(),value=await recognizeInPool(await createImageBitmap(canvas),boxes,Number(mode));if(timing)timing.inference_seconds=(performance.now()-inferenceStarted)/1000;return value}
  const {lines,seconds}=await segment(canvas);
  if(timing)timing.layout_seconds=seconds;
  const inferenceStarted=performance.now();
  const results=await infer(lines,Number(mode));
  if(timing)timing.inference_seconds=(performance.now()-inferenceStarted)/1000;
  return cleanModelText(results.map(item=>item.text).join('\n'));
}

async function preparePages(canvases) {
  await ready;const started=performance.now(),segments=[],lines=[];
  for(const canvas of canvases){const value=await segment(canvas);segments.push({offset:lines.length,count:value.lines.length});lines.push(...value.lines)}
  return {lines,segments,layout_seconds:(performance.now()-started)/1000};
}

async function recognizePreparedPages(prepared) {
  const started=performance.now(),results=await infer(prepared.lines,scaleOverride||Number(document.querySelector('[name="mode"]:checked').value));
  const confidences=results.map(result=>result.confidence).filter(Number.isFinite).sort((a,b)=>a-b);
  return {inference_seconds:(performance.now()-started)/1000,retried:results.filter(result=>result.retried).length,confidence_median:confidences[Math.floor(confidences.length/2)],pages:prepared.segments.map(({offset,count})=>({line_count:count,text:cleanModelText(results.slice(offset,offset+count).map(item=>item.text).join('\n'))}))};
}

window.krakenLiteLineCanvases=()=>{const source=sourceCanvas();return withoutRunningHeader(lineCanvases(source),source)};
window.krakenLiteRecognizeLines=async lines=>cleanModelText((await infer(lines,scaleOverride||Number(document.querySelector('[name="mode"]:checked').value))).map(item=>item.text).join('\n'));
window.krakenLiteRecognizePage=async()=>{const started=performance.now(),timing={},value=await recognize(sourceCanvas(),timing);return {seconds:(performance.now()-started)/1000,...timing,text:value}};
window.krakenLitePreparePages=preparePages;
window.krakenLiteRecognizePreparedPages=recognizePreparedPages;
window.krakenLiteSetTuning=value=>{if(value.batch)batchSize=value.batch;if(value.bucket)bucketSize=value.bucket};
window.krakenLiteRecognizePages=async canvases=>{const started=performance.now(),prepared=await preparePages(canvases),recognized=await recognizePreparedPages(prepared);return {seconds:(performance.now()-started)/1000,layout_seconds:prepared.layout_seconds,...recognized}};
window.krakenLiteRecognizePagesPooled=async canvases=>{
  await ready;await poolReady;if(!workerPoolSize)throw new Error('Set ?pool=N to enable the recognition worker pool');
  const started=performance.now(),scale=scaleOverride||Number(document.querySelector('[name="mode"]:checked').value),counts=[];let layoutsDone=0,layout_wall_seconds=0;
  const tasks=canvases.map(async(canvas,index)=>{const {lines}=await segment(canvas);if(++layoutsDone===canvases.length)layout_wall_seconds=(performance.now()-started)/1000;counts[index]=lines.length;const boxes=lines.map(({x,y,width,height})=>({x,y,width,height}));return recognizeInPool(await createImageBitmap(canvas),boxes,scale)});
  const values=await Promise.all(tasks);return {seconds:(performance.now()-started)/1000,layout_wall_seconds,pages:values.map((text,index)=>({line_count:counts[index],text}))};
};

$('form').onsubmit=async event=>{event.preventDefault();const start=performance.now();text.textContent='';text.textContent=await recognize(sourceCanvas());status.textContent=`Done in ${((performance.now()-start)/1000).toFixed(2)} seconds · browser inference`;};
async function documentCanvas(number){
  if(!pdf)return sourceCanvas();const page=await pdf.getPage(number),viewport=page.getViewport({scale:2}),canvas=document.createElement('canvas');canvas.width=Math.round(viewport.width);canvas.height=Math.round(viewport.height);await page.render({canvasContext:canvas.getContext('2d'),viewport}).promise;
  if(!cropTemplate)return canvas;const selected=document.createElement('canvas'),area={x:cropTemplate.x*canvas.width,y:cropTemplate.y*canvas.height,w:cropTemplate.w*canvas.width,h:cropTemplate.h*canvas.height};selected.width=Math.round(area.w);selected.height=Math.round(area.h);selected.getContext('2d').drawImage(canvas,area.x,area.y,area.w,area.h,0,0,selected.width,selected.height);return selected;
}

$('all').onclick=async()=>{
  const target=$('document');target.textContent='';const count=pdf?.numPages||1,started=performance.now(),entries=Array.from({length:count},(_,index)=>{const pair=document.createElement('article');pair.className='pair';const img=document.createElement('img');img.alt=`Page ${index+1}`;const copy=document.createElement('div');copy.className='text';const pre=document.createElement('pre');pre.textContent='Waiting…';copy.append(pre);pair.append(img,copy);target.append(pair);return{img,pre}});let next=1,done=0;
  const runner=async()=>{while(next<=count){const number=next++,canvas=await documentCanvas(number),entry=entries[number-1];entry.img.src=canvas.toDataURL('image/jpeg',.72);entry.pre.textContent='Recognizing…';try{entry.pre.textContent=await recognize(canvas)}catch(error){entry.pre.textContent=`OCR failed: ${error.message}`}done++;status.textContent=`Recognized ${done} of ${count} pages · ${((performance.now()-started)/1000).toFixed(1)} seconds`;await new Promise(requestAnimationFrame)}};
  await Promise.all(Array.from({length:Math.min(count,workerPoolSize||1)},runner));status.textContent=`Done · ${count} pages in ${((performance.now()-started)/1000).toFixed(2)} seconds`;
};
