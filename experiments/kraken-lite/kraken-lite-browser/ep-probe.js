import * as ort from 'onnxruntime-web/all';

const params = new URLSearchParams(location.search);
const ep = params.get('ep') || 'webgpu';
const fallback = params.get('fallback') === '1';
const result = { ep, navigator_gpu: Boolean(navigator.gpu), navigator_ml: Boolean(navigator.ml) };
ort.env.wasm.wasmPaths = new URL('./node_modules/onnxruntime-web/dist/', location.href).href;

try {
  const model = await fetch('./dist/model.onnx').then(response => response.arrayBuffer());
  const started = performance.now();
  const session = await ort.InferenceSession.create(model, {
    executionProviders: fallback ? [ep, 'wasm'] : [ep],
    graphOptimizationLevel: 'all',
  });
  result.init_seconds = (performance.now() - started) / 1000;
  const width = 128;
  const runStarted = performance.now();
  const output = await session.run({
    image: new ort.Tensor('float32', new Float32Array(48 * width), [1, 1, 48, width]),
    sequence_lengths: new ort.Tensor('int64', new BigInt64Array([BigInt(width)]), [1]),
  });
  result.run_seconds = (performance.now() - runStarted) / 1000;
  result.outputs = Object.keys(output);
  result.fallback = fallback;
  result.ok = true;
} catch (error) {
  result.ok = false;
  result.error = String(error?.message || error);
}

document.body.textContent = JSON.stringify(result);
document.body.dataset.done = '1';
