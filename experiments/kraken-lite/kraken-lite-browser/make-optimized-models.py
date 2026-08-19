from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
sys.path[:0] = [str(ROOT / '.runtime-tools'), str(EXPERIMENT / 'kraken-lite-spin-deps'), str(EXPERIMENT / 'kraken-lite-native')]

import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.tools.convert_onnx_models_to_ort import OptimizationStyle, convert_onnx_models_to_ort

import ocr

SOURCE = ROOT / 'dist' / 'model.onnx'
OUTPUT = ROOT / 'dist' / 'optimized-models'
ORT_OUTPUT = ROOT / 'dist' / 'optimized-models-ort'
SPLIT = EXPERIMENT / 'kraken-lite-native' / 'benchmark-splits' / 'benchmark-153.lst'


def line_samples(limit: int = 48):
    for xml in SPLIT.read_text(encoding='utf-8-sig').splitlines()[:12]:
        page = Image.open(Path(xml).with_suffix('.png')).convert('L')
        for box in ocr.tesseract_line_boxes(page):
            line = page.crop((box.left, box.top, box.right, box.bottom))
            if line.height < 8 or line.width < 20:
                continue
            width = max(1, round(line.width * 48 / line.height))
            canvas = Image.new('L', (width + 24, 48), 255)
            canvas.paste(line.resize((width, 48), Image.Resampling.BILINEAR), (12, 0))
            pixels = 1 - np.asarray(canvas, dtype=np.float32) / 255
            yield {'image': pixels[None, None], 'sequence_lengths': np.array([canvas.width], dtype=np.int64)}
            limit -= 1
            if not limit:
                return


class Lines(CalibrationDataReader):
    def __init__(self):
        self.rewind()

    def get_next(self):
        return next(self.samples, None)

    def rewind(self):
        self.samples = iter(line_samples())


def main():
    OUTPUT.mkdir(exist_ok=True)
    preprocessed = OUTPUT / 'model.preprocessed.onnx'
    quant_pre_process(SOURCE, preprocessed, auto_merge=True)
    quantize_dynamic(preprocessed, OUTPUT / 'lstm-channel-preprocessed.onnx', weight_type=QuantType.QInt8,
                     per_channel=True, op_types_to_quantize=['LSTM'])
    static_conv = OUTPUT / 'conv-static.onnx'
    quantize_static(preprocessed, static_conv, Lines(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=['Conv'])
    quantize_dynamic(static_conv, OUTPUT / 'conv-static-lstm-channel.onnx', weight_type=QuantType.QInt8,
                     per_channel=True, op_types_to_quantize=['LSTM'])
    ORT_OUTPUT.mkdir(exist_ok=True)
    convert_onnx_models_to_ort(OUTPUT, ORT_OUTPUT, [OptimizationStyle.Fixed], enable_type_reduction=True)


if __name__ == '__main__':
    main()
