export class TesseractLayout {
  constructor({ workerPath = './tesseract-layout-worker.js', corePath = './dist/layout-core.mjs', wasmPath = './dist/layout-core.wasm', sourceResolution = 200, trim = false, psm = 3, binaryThreshold = 0 } = {}) {
    this.worker = new Worker(workerPath);
    this.corePath = new URL(corePath, location.href).href;
    this.wasmPath = new URL(wasmPath, location.href).href;
    this.sourceResolution = sourceResolution;
    this.trim = trim;
    this.psm = psm;
    this.binaryThreshold = binaryThreshold;
    this.nextId = 0;
    this.pending = new Map();
    this.worker.onmessage = ({ data }) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      this.pending.delete(data.id);
      data.error ? pending.reject(new Error(data.error)) : pending.resolve(data.lines.map(line=>({x0:line.x0+pending.x,y0:line.y0+pending.y,x1:line.x1+pending.x,y1:line.y1+pending.y})));
    };
    this.worker.onerror = error => {
      for (const pending of this.pending.values()) pending.reject(new Error(error.message || 'layout worker failed'));
      this.pending.clear();
    };
  }

  async findLines(canvas) {
    let { width, height } = canvas;
    let x=0,y=0,cropWidth=width,cropHeight=height;
    if(this.trim){
      const divisor=8,sampleWidth=Math.ceil(width/divisor),sampleHeight=Math.ceil(height/divisor),sample=typeof OffscreenCanvas==='function'?new OffscreenCanvas(sampleWidth,sampleHeight):document.createElement('canvas');sample.width=sampleWidth;sample.height=sampleHeight;const sampleContext=sample.getContext('2d',{willReadFrequently:true});sampleContext.drawImage(canvas,0,0,sampleWidth,sampleHeight);const preview=sampleContext.getImageData(0,0,sampleWidth,sampleHeight).data;
      let left=sampleWidth,top=sampleHeight,right=0,bottom=0;
      for(let yy=0;yy<sampleHeight;yy++)for(let xx=0;xx<sampleWidth;xx++){const offset=(yy*sampleWidth+xx)*4;if(preview[offset]+preview[offset+1]+preview[offset+2]<720){left=Math.min(left,xx);right=Math.max(right,xx);top=Math.min(top,yy);bottom=Math.max(bottom,yy)}}
      if(right>left&&bottom>top){const margin=48;x=Math.max(0,left*divisor-margin);y=Math.max(0,top*divisor-margin);const cropRight=Math.min(width,(right+1)*divisor+margin),cropBottom=Math.min(height,(bottom+1)*divisor+margin);cropWidth=cropRight-x;cropHeight=cropBottom-y;
        if(cropWidth*cropHeight>=width*height*.95){x=0;y=0;cropWidth=width;cropHeight=height}
      }
    }
    const pixels=canvas.getContext('2d').getImageData(x,y,cropWidth,cropHeight).data;width=cropWidth;height=cropHeight;
    const id = ++this.nextId;
    const result = new Promise((resolve, reject) => this.pending.set(id, { resolve, reject, x, y }));
    this.worker.postMessage({ id, pixels: pixels.buffer, width, height, corePath: this.corePath, wasmPath: this.wasmPath, sourceResolution: this.sourceResolution, psm: this.psm, binaryThreshold: this.binaryThreshold }, [pixels.buffer]);
    return result;
  }

  terminate() {
    this.worker.terminate();
  }
}
