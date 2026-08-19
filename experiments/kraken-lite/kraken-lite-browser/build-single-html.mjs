import { readFileSync, writeFileSync } from 'node:fs';

const root = import.meta.dirname;
const heavy = process.argv.includes('--heavy');
const base64 = path => readFileSync(`${root}/${path}`).toString('base64');
const assets = {
  codec: base64('dist/codec.json'),
  model: base64('dist/optimized-models-ort/lstm-channel-preprocessed.ort'),
  ortMjs: base64('dist/ort-wasm-relaxedsimd-threaded.mjs'),
  ortWasm: base64('dist/ort-wasm-relaxedsimd-threaded.wasm'),
  ortThreadedMjs: base64('dist/ort-wasm-relaxedsimd-threaded.mjs'),
  ortThreadedWasm: base64('dist/ort-wasm-relaxedsimd-threaded.wasm'),
  pdfWorker: base64('dist/pdf.worker.min.mjs'),
};
if (heavy) {
  assets.recognitionWorker = base64('dist/recognition-worker.js');
  assets.layoutWorker = base64('tesseract-layout-worker.js');
  assets.layoutCore = base64('dist/layout-core-dpi.mjs');
  assets.layoutWasm = base64('dist/layout-core-dpi.wasm');
}

const app = readFileSync(`${root}/dist/app.js`, 'utf8').replaceAll('</script', '<\\/script');
const bootstrap = `<script>globalThis.KRAKEN_LITE_ASSETS=${JSON.stringify(assets)}</script><script type="module">${app}</script>`;
const html = readFileSync(`${root}/index.html`, 'utf8').replace('<script type="module" src="./dist/app.js"></script>', () => bootstrap);
const output = `${root}/dist/kraken-lite${heavy ? '' : '-lean'}.html`;
writeFileSync(output, html);
process.stdout.write(`${output}\n`);
