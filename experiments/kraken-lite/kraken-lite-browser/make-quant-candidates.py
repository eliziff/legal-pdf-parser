from collections import Counter
from pathlib import Path
import json

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT = Path(__file__).parent
SOURCE = ROOT / "dist" / "model.onnx"
OUTPUT = ROOT / "dist" / "quant-candidates"
LSTM_LAYERS = [f"/nn/L_{index}/layer/LSTM" for index in range(2, 6)]

# Dynamic quantization is ORT's recommended path for recurrent models. LSTM
# weights are always signed INT8 in ORT 1.22; weight_type controls the final
# MatMul. Selective variants test whether leaving that small projection in FP32
# buys accuracy more cheaply than it costs throughput/bytes.
CANDIDATES = {
    "s8-tensor": dict(weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM", "MatMul"]),
    "u8-tensor": dict(weight_type=QuantType.QUInt8, op_types_to_quantize=["LSTM", "MatMul"]),
    "s8-tensor-rr": dict(weight_type=QuantType.QInt8, reduce_range=True, op_types_to_quantize=["LSTM", "MatMul"]),
    "s8-channel": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["LSTM", "MatMul"]),
    "s8-channel-rr": dict(weight_type=QuantType.QInt8, per_channel=True, reduce_range=True, op_types_to_quantize=["LSTM", "MatMul"]),
    "u8-channel": dict(weight_type=QuantType.QUInt8, per_channel=True, op_types_to_quantize=["LSTM", "MatMul"]),
    "lstm-tensor": dict(weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM"]),
    "lstm-tensor-rr": dict(weight_type=QuantType.QInt8, reduce_range=True, op_types_to_quantize=["LSTM"]),
    "lstm-channel": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["LSTM"]),
    "lstm-channel-rr": dict(weight_type=QuantType.QInt8, per_channel=True, reduce_range=True, op_types_to_quantize=["LSTM"]),
    "lstm-l5-tensor": dict(weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM"], nodes_to_quantize=LSTM_LAYERS[-1:]),
    "lstm-l45-tensor": dict(weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM"], nodes_to_quantize=LSTM_LAYERS[-2:]),
    "lstm-l345-tensor": dict(weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM"], nodes_to_quantize=LSTM_LAYERS[-3:]),
    "lstm-l5-channel": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["LSTM"], nodes_to_quantize=LSTM_LAYERS[-1:]),
    "lstm-l45-channel": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["LSTM"], nodes_to_quantize=LSTM_LAYERS[-2:]),
}


def audit(path: Path) -> dict:
    model = onnx.load(path, load_external_data=False)
    return {
        "bytes": path.stat().st_size,
        "operators": dict(sorted(Counter(f"{node.domain}:{node.op_type}" for node in model.graph.node).items())),
        "initializer_types": dict(sorted(Counter(onnx.TensorProto.DataType.Name(value.data_type) for value in model.graph.initializer).items())),
    }


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    receipt = {}
    for name, options in CANDIDATES.items():
        target = OUTPUT / f"{name}.onnx"
        quantize_dynamic(SOURCE, target, **options)
        receipt[name] = {"options": {key: str(value) for key, value in options.items()}, **audit(target)}
        print(f"{name}: {target.stat().st_size:,} bytes")
    (OUTPUT / "models.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
