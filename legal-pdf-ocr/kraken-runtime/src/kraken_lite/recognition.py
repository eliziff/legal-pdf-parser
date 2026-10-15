from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .geometry import Rectification
from .model_pack import ModelPack, Session
from .types import Character


@dataclass(slots=True)
class LabelSpan:
    label: int
    start: int
    end: int  # inclusive, matching Kraken
    confidence: float


@dataclass(slots=True)
class DecodedSpan:
    text: str
    start: int
    end: int  # inclusive network timestep, matching Kraken
    confidence: float


@dataclass(slots=True)
class PreparedLine:
    tensor: np.ndarray
    source_width: int
    content_width: int
    input_width: int
    left_padding: int
    horizontal_scale: float


@dataclass(slots=True)
class RecognitionResult:
    text: str
    confidence: float
    characters: list[Character]


class Codec:
    """Kraken-compatible prefix-free mapping between labels and graphemes."""

    def __init__(self, c2l: Mapping[str, Sequence[int]]) -> None:
        self.c2l = {
            str(text): tuple(int(value) for value in labels)
            for text, labels in c2l.items()
        }
        if not self.c2l:
            raise ValueError("Recognition codec is empty")
        self.l2c: dict[tuple[int, ...], str] = {}
        for text, labels in self.c2l.items():
            if not labels:
                raise ValueError(f"Codec entry {text!r} has no labels")
            if labels in self.l2c:
                raise ValueError(f"Non-singular codec label sequence: {labels}")
            self.l2c[labels] = text
        self._label_sequences = sorted(self.l2c, key=len, reverse=True)
        for first in self._label_sequences:
            for second in self._label_sequences:
                if (
                    first != second
                    and len(first) <= len(second)
                    and second[: len(first)] == first
                ):
                    raise ValueError("Codec is not prefix-free")

    @classmethod
    def from_pack(cls, pack: ModelPack) -> "Codec":
        codec_spec = pack.manifest.get("codec", {})
        if "mapping" in codec_spec:
            mapping = codec_spec["mapping"]
        elif "file" in codec_spec:
            codec_path = pack.root / codec_spec["file"]
            mapping = json.loads(codec_path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Recognition manifest has no codec mapping or file")
        return cls(mapping)

    def decode(self, spans: Sequence[LabelSpan]) -> list[DecodedSpan]:
        labels = tuple(span.label for span in spans)
        output: list[DecodedSpan] = []
        index = 0
        while index < len(spans):
            match: tuple[int, ...] | None = None
            for candidate in self._label_sequences:
                if labels[index : index + len(candidate)] == candidate:
                    match = candidate
                    break
            if match is None:
                index += 1
                continue
            selected = spans[index : index + len(match)]
            output.append(
                DecodedSpan(
                    text=self.l2c[match],
                    start=selected[0].start,
                    end=selected[-1].end,
                    confidence=float(
                        np.mean([span.confidence for span in selected])
                    ),
                )
            )
            index += len(match)
        return output


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(values - maximum)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def greedy_ctc(probabilities: np.ndarray, *, blank_label: int = 0) -> list[LabelSpan]:
    """Best-path CTC decoder retaining confidence and timestep geometry."""

    if probabilities.ndim != 2:
        raise ValueError("CTC probabilities must have shape classes x timesteps")
    confidences = probabilities.max(axis=0)
    labels = probabilities.argmax(axis=0)
    output: list[LabelSpan] = []
    for label, grouped in groupby(
        zip(range(len(labels)), labels.tolist(), confidences.tolist(), strict=True),
        key=lambda item: item[1],
    ):
        group = list(grouped)
        if int(label) == int(blank_label):
            continue
        output.append(
            LabelSpan(
                label=int(label),
                start=int(group[0][0]),
                end=int(group[-1][0]),
                confidence=float(max(item[2] for item in group)),
            )
        )
    return output


def _fixed_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    source_width, source_height = image.size
    if height <= 0 and width <= 0:
        return image
    if height <= 0:
        height = max(1, int(source_height * width / source_width))
    elif width <= 0:
        width = max(1, int(source_width * height / source_height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def prepare_line(image: Image.Image, manifest: Mapping[str, Any]) -> PreparedLine:
    spec = manifest.get("input", {})
    channels = int(spec.get("channels", 1))
    height = int(spec.get("height", 0))
    width = int(spec.get("width", 0))
    if channels not in (1, 3):
        raise ValueError(f"Recognition input channels must be 1 or 3, got {channels}")
    converted = image.convert("L" if channels == 1 else "RGB")
    resized = _fixed_resize(converted, height, width)
    source_width = image.width
    horizontal_scale = resized.width / max(1, source_width)

    padding = spec.get("padding", 16)
    if isinstance(padding, int):
        left = right = int(padding)
    elif len(padding) == 2:
        left, right = (int(value) for value in padding)
    else:
        raise ValueError("Recognition input.padding must be an integer or pair")

    array = np.asarray(resized, dtype=np.float32) / 255.0
    if channels == 1:
        array = array[..., None]
    array = np.pad(
        array,
        ((0, 0), (left, right), (0, 0)),
        mode="constant",
        constant_values=1.0,
    )
    if bool(spec.get("invert", True)):
        maximum = float(array.max()) if array.size else 1.0
        array = maximum - array
    tensor = np.transpose(array, (2, 0, 1))[None].astype(np.float32, copy=False)
    return PreparedLine(
        tensor=tensor,
        source_width=source_width,
        content_width=resized.width,
        input_width=array.shape[1],
        left_padding=left,
        horizontal_scale=horizontal_scale,
    )


def _canonical_probabilities_batch(
    output: np.ndarray, manifest: Mapping[str, Any]
) -> np.ndarray:
    """Converts model output to ``batch x classes x timesteps``."""

    values = np.asarray(output)
    if values.ndim < 2:
        raise ValueError(f"Unexpected recognition output shape {values.shape}")
    if values.ndim == 2:
        values = values[None, ...]

    output_spec = manifest.get("output", {})
    class_axis = int(output_spec.get("class_axis", 1))
    if class_axis < 0:
        class_axis += values.ndim
    if class_axis <= 0 or class_axis >= values.ndim:
        raise ValueError(f"Invalid class axis for output shape {values.shape}")
    values = np.moveaxis(values, class_axis, 1)

    # Kraken recognizers commonly emit N x C x 1 x T. Remove singleton spatial
    # axes while retaining batch, class, and the final time axis.
    for axis in range(values.ndim - 2, 1, -1):
        if values.shape[axis] == 1:
            values = np.squeeze(values, axis=axis)
    if values.ndim != 3:
        raise ValueError(
            "Recognition output did not reduce to batch x classes x timesteps: "
            f"{values.shape}"
        )
    activation = output_spec.get("activation", "logits")
    if activation == "logits":
        values = _softmax(values.astype(np.float32), axis=1)
    elif activation not in {"probabilities", "softmax"}:
        raise ValueError(f"Unsupported recognition activation {activation}")
    return values.astype(np.float32, copy=False)


def _canonical_probabilities(
    output: np.ndarray, manifest: Mapping[str, Any]
) -> np.ndarray:
    values = _canonical_probabilities_batch(output, manifest)
    if values.shape[0] != 1:
        raise ValueError(f"Expected a single recognition sample, got {values.shape[0]}")
    return values[0]


class Recognizer:
    """Runs a compiled Kraken recognizer and maps CTC spans to page polygons."""

    def __init__(
        self,
        session: Session,
        manifest: Mapping[str, Any],
        codec: Codec,
    ) -> None:
        if manifest.get("kind") != "recognition":
            raise ValueError("Recognizer requires a recognition manifest")
        self.session = session
        self.manifest = dict(manifest)
        self.codec = codec

    @classmethod
    def from_pack(
        cls,
        path: str,
        *,
        device: str = "auto",
        providers: list[str] | None = None,
        intra_threads: int = 1,
        allow_cpu_fallback: bool = True,
        tensorrt_fp16: bool = False,
        tensorrt_cache: str | None = None,
        cuda_tf32: bool = True,
    ) -> "Recognizer":
        pack = ModelPack.load(path)
        if pack.kind != "recognition":
            raise ValueError(f"Expected a recognition pack, got {pack.kind}")
        return cls(
            pack.open_session(
                device=device,
                providers=providers,
                intra_threads=intra_threads,
                allow_cpu_fallback=allow_cpu_fallback,
                tensorrt_fp16=tensorrt_fp16,
                tensorrt_cache=tensorrt_cache,
                cuda_tf32=cuda_tf32,
            ),
            pack.manifest,
            Codec.from_pack(pack),
        )

    @property
    def default_batch_size(self) -> int:
        info = getattr(self.session, "execution_info", None)
        if info is not None and info.accelerated:
            return int(self.manifest.get("recognition", {}).get("accelerated_batch_size", 32))
        return int(self.manifest.get("recognition", {}).get("cpu_batch_size", 4))

    def _decode(
        self,
        probabilities: np.ndarray,
        prepared: PreparedLine,
        rectification: Rectification,
    ) -> RecognitionResult:
        output_length = probabilities.shape[1]
        output_spec = self.manifest.get("output", {})
        labels = greedy_ctc(
            probabilities,
            blank_label=int(output_spec.get("blank_label", 0)),
        )
        decoded = self.codec.decode(labels)
        characters: list[Character] = []
        for span in decoded:
            # Kraken's decoder stores inclusive CTC cuts. Keep that convention
            # so confidence spans and page-space character geometry remain
            # comparable with full Kraken output.
            input_start = span.start / max(1, output_length) * prepared.input_width
            input_end = span.end / max(1, output_length) * prepared.input_width
            content_start = np.clip(
                input_start - prepared.left_padding, 0.0, prepared.content_width
            )
            content_end = np.clip(
                input_end - prepared.left_padding, content_start, prepared.content_width
            )
            source_start = float(
                content_start / max(prepared.horizontal_scale, 1e-8)
            )
            source_end = float(content_end / max(prepared.horizontal_scale, 1e-8))
            polygon = rectification.project_interval(source_start, source_end)
            # PytorchCodec emits one record per Unicode code point when a
            # single label maps to a multi-codepoint string. They share the
            # same cut and confidence; reproduce that geometry contract.
            for codepoint in span.text:
                characters.append(
                    Character(
                        text=codepoint,
                        confidence=span.confidence,
                        start=span.start,
                        end=span.end,
                        polygon=list(polygon),
                    )
                )
        confidence = (
            float(np.mean([character.confidence for character in characters]))
            if characters
            else 0.0
        )
        return RecognitionResult(
            text="".join(character.text for character in characters),
            confidence=confidence,
            characters=characters,
        )

    def recognize_many(
        self,
        rectifications: Sequence[Rectification],
        *,
        batch_size: int | None = None,
        width_bucket: int = 128,
    ) -> list[RecognitionResult]:
        """Recognizes lines in width-bucketed batches.

        This is the path that materially uses CUDA-class hardware: page
        segmentation is one graph invocation, while recognition may contain
        hundreds of line images. Bucketing limits wasted padded pixels and the
        exported sequence-length input prevents padded columns from becoming
        text.
        """

        if not rectifications:
            return []
        if batch_size is None or batch_size <= 0:
            batch_size = self.default_batch_size
        batch_size = max(1, int(batch_size))
        if not bool(self.manifest.get("model", {}).get("dynamic_batch", False)):
            batch_size = 1
        width_bucket = max(1, int(width_bucket))

        prepared = [prepare_line(item.image, self.manifest) for item in rectifications]
        groups: dict[int, list[int]] = {}
        for index, item in enumerate(prepared):
            key = (item.input_width + width_bucket - 1) // width_bucket
            groups.setdefault(key, []).append(index)

        results: list[RecognitionResult | None] = [None] * len(rectifications)
        model_spec = self.manifest.get("model", {})
        input_name = str(model_spec.get("input", "image"))
        output_name = str(model_spec.get("output", "logits"))
        lengths_input = model_spec.get("lengths_input")
        lengths_output = model_spec.get("lengths_output")

        for key in sorted(groups):
            indices = groups[key]
            for offset in range(0, len(indices), batch_size):
                batch_indices = indices[offset : offset + batch_size]
                maximum_width = max(prepared[index].input_width for index in batch_indices)
                first = prepared[batch_indices[0]].tensor
                tensor = np.zeros(
                    (len(batch_indices), first.shape[1], first.shape[2], maximum_width),
                    dtype=np.float32,
                )
                input_lengths = np.empty(len(batch_indices), dtype=np.int64)
                for local_index, source_index in enumerate(batch_indices):
                    item = prepared[source_index]
                    tensor[local_index, :, :, : item.input_width] = item.tensor[0]
                    input_lengths[local_index] = item.input_width

                inputs: dict[str, np.ndarray] = {input_name: tensor}
                if lengths_input:
                    inputs[str(lengths_input)] = input_lengths
                outputs = self.session.run(inputs)
                if output_name not in outputs:
                    raise RuntimeError(
                        f"Recognizer did not return {output_name!r}; got {list(outputs)}"
                    )
                probabilities = _canonical_probabilities_batch(
                    outputs[output_name], self.manifest
                )
                if probabilities.shape[0] != len(batch_indices):
                    raise RuntimeError(
                        "Recognition graph changed the batch dimension: "
                        f"expected {len(batch_indices)}, got {probabilities.shape[0]}"
                    )

                if lengths_output and str(lengths_output) in outputs:
                    output_lengths = np.asarray(outputs[str(lengths_output)]).reshape(-1)
                    if len(output_lengths) != len(batch_indices):
                        raise RuntimeError("Recognition output lengths do not match batch size")
                else:
                    # Compatibility fallback for older compiled packs. New packs
                    # export sequence lengths explicitly.
                    total = probabilities.shape[2]
                    output_lengths = np.maximum(
                        1,
                        np.floor(total * input_lengths / max(1, maximum_width)).astype(
                            np.int64
                        ),
                    )

                for local_index, source_index in enumerate(batch_indices):
                    length = int(
                        np.clip(output_lengths[local_index], 1, probabilities.shape[2])
                    )
                    results[source_index] = self._decode(
                        probabilities[local_index, :, :length],
                        prepared[source_index],
                        rectifications[source_index],
                    )

        if any(result is None for result in results):  # pragma: no cover - invariant
            raise RuntimeError("Recognizer did not produce every requested result")
        return [result for result in results if result is not None]

    def recognize(self, rectification: Rectification) -> RecognitionResult:
        return self.recognize_many([rectification], batch_size=1)[0]
