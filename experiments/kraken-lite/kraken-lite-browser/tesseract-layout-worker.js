let ready;

async function start(corePath, wasmPath) {
  const createCore = (await import(corePath)).default;
  const core = await createCore({ locateFile: () => wasmPath });
  return { core, api: core._kl_create() };
}

self.onmessage = async ({ data: { id, pixels, width, height, corePath, wasmPath, sourceResolution = 200, psm = 3, binaryThreshold = 0 } }) => {
  try {
    ready ||= start(corePath, wasmPath);
    const { api, core } = await ready;
    core._kl_set_psm?.(api,psm);
    const rgba=new Uint8Array(pixels),useGray=corePath.includes('gray'),input=useGray?new Uint8Array(width*height):rgba;
    if(useGray)for(let index=0,offset=0;index<input.length;index++,offset+=4)input[index]=(rgba[offset]*77+rgba[offset+1]*150+rgba[offset+2]*29)>>8;
    const image = core._malloc(input.byteLength);
    core.HEAPU8.set(input, image);
    let capacity = 128;
    let boxes = core._malloc(capacity * 16);
    const useBinary=Boolean(binaryThreshold&&core._kl_lines_binary),find=useGray?core._kl_lines_gray:useBinary?core._kl_lines_binary:(core._kl_lines_dpi||core._kl_lines);
    const call=()=>useGray?find(api,image,width,height,boxes,capacity):useBinary?find(api,image,width,height,boxes,capacity,sourceResolution,binaryThreshold):core._kl_lines_dpi?find(api,image,width,height,boxes,capacity,sourceResolution):find(api,image,width,height,boxes,capacity);
    let count = call();
    if (count < 0) {
      capacity = -count;
      core._free(boxes);
      boxes = core._malloc(capacity * 16);
      count = call();
    }
    const lines = Array.from({ length: count }, (_, index) => {
      const offset = boxes / 4 + index * 4;
      const [x0, y0, x1, y1] = core.HEAP32.subarray(offset, offset + 4);
      return { x0, y0, x1, y1 };
    });
    core._free(boxes);
    core._free(image);
    self.postMessage({ id, lines });
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
