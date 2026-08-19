from pathlib import Path

import onnx
from onnx import TensorProto, helper


def add_argmax(source: Path, target: Path) -> None:
    model = onnx.load(source)
    logits = model.graph.output[0]
    shape = logits.type.tensor_type.shape.dim
    output = helper.make_tensor_value_info(
        "class_ids",
        TensorProto.INT64,
        [shape[0].dim_param or shape[0].dim_value, 1, shape[-1].dim_param or shape[-1].dim_value],
    )
    model.graph.node.append(
        helper.make_node("ArgMax", [logits.name], [output.name], axis=1, keepdims=0, name="KrakenArgMax")
    )
    model.graph.output.remove(logits)
    model.graph.output.insert(0, output)
    onnx.checker.check_model(model)
    onnx.save(model, target)


if __name__ == "__main__":
    dist = Path(__file__).parent / "dist"
    add_argmax(dist / "model.onnx", dist / "model.argmax.onnx")
    add_argmax(dist / "model.dynamic-u8.onnx", dist / "model.dynamic-u8.argmax.onnx")
