import * as ort from 'onnxruntime-web/wasm';

let session, labels, batchSize, bucketSize, padding;
const height=48,canvas=new OffscreenCanvas(1,1),context=canvas.getContext('2d',{willReadFrequently:true});

function widthOf(line,scale){return Math.max(1,Math.round(line.width*height/line.height*scale))+padding*2}

function decode(output,n,validSteps){
  const dims=output.dims,data=output.data,classes=dims[1],steps=dims.at(-1),stride=classes*steps;let previous=0,text='';
  for(let t=0;t<Math.min(steps,validSteps);t++){let best=0,maximum=-Infinity;for(let c=0;c<classes;c++){const value=data[n*stride+c*steps+t];if(value>maximum){maximum=value;best=c}}if(best&&best!==previous)text+=labels[best]||'';previous=best}
  return text;
}

async function recognize(bitmap,lines,scale){
  const results=new Array(lines.length),groups=new Map();
  lines.forEach((line,index)=>{const width=widthOf(line,scale),key=Math.ceil(width/bucketSize);if(!groups.has(key))groups.set(key,[]);groups.get(key).push({line,index,width})});
  for(const group of groups.values())for(let offset=0;offset<group.length;offset+=batchSize){
    const batch=group.slice(offset,offset+batchSize),max=Math.max(...batch.map(item=>item.width));canvas.width=max;canvas.height=batch.length*height;context.fillStyle='white';context.fillRect(0,0,canvas.width,canvas.height);
    batch.forEach((item,n)=>context.drawImage(bitmap,item.line.x,item.line.y,item.line.width,item.line.height,padding,n*height,item.width-padding*2,height));
    const rgba=context.getImageData(0,0,canvas.width,canvas.height).data,tensor=new Float32Array(batch.length*height*max),lengths=new BigInt64Array(batch.length);
    for(let i=0;i<tensor.length;i++){const p=i*4;tensor[i]=1-(rgba[p]+rgba[p+1]+rgba[p+2])/(3*255)}
    batch.forEach((item,n)=>{lengths[n]=BigInt(item.width)});
    const out=await session.run({image:new ort.Tensor('float32',tensor,[batch.length,1,height,max]),sequence_lengths:new ort.Tensor('int64',lengths,[batch.length])}),output=out.logits||Object.values(out)[0],valid=out.output_lengths;
    batch.forEach((item,n)=>{results[item.index]=decode(output,n,valid?Number(valid.data[n]):output.dims.at(-1))});
  }
  bitmap.close();return results.join('\n').replace(/[\u00ad\u00ac]\r?\n/g,'').replace(/[\u00ad\u00ac]/g,'');
}

self.onmessage=async({data})=>{
  try{
    if(data.type==='init'){
      ort.env.wasm.wasmPaths={mjs:data.runtimeMjs,wasm:data.runtimeWasm};ort.env.wasm.numThreads=1;
      labels=Object.fromEntries(Object.entries(data.codec).flatMap(([character,ids])=>ids.map(id=>[id,character])));batchSize=data.batchSize;bucketSize=data.bucketSize;padding=data.padding;
      session=await ort.InferenceSession.create(data.model,{executionProviders:['wasm'],graphOptimizationLevel:'all'});self.postMessage({type:'ready'});return;
    }
    self.postMessage({type:'result',id:data.id,text:await recognize(data.bitmap,data.lines,data.scale)});
  }catch(error){self.postMessage({type:'error',id:data.id,message:error.message,stack:error.stack})}
};
