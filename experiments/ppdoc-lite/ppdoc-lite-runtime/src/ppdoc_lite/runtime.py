from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PACK_FORMAT = "legalpdf.ppdoc-lite-model/1"
SKIP_ORDER_LABELS = {
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ModelPack:
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ModelPack":
        root = Path(path).resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing model manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != PACK_FORMAT:
            raise ValueError(f"Unsupported model-pack format: {manifest.get('format')!r}")
        if not str(manifest.get("variant_id") or manifest.get("profile") or "").strip():
            raise ValueError("Model manifest has no variant_id")
        if not isinstance(manifest.get("labels"), list) or not manifest["labels"]:
            raise ValueError("Model manifest has no labels")
        model = manifest.get("model")
        if not isinstance(model, dict) or not model.get("file"):
            raise ValueError("Model manifest has no model.file")
        pack = cls(root=root, manifest=manifest)
        expected = str(model.get("sha256") or "").lower()
        if not pack.model_path.is_file():
            raise FileNotFoundError(f"Missing model graph: {pack.model_path}")
        if expected:
            actual = sha256_file(pack.model_path)
            if actual != expected:
                raise ValueError(
                    f"Model hash mismatch for {pack.model_path}: expected {expected}, got {actual}"
                )
        for member in model.get("files") or []:
            if not isinstance(member, dict) or not member.get("file"):
                raise ValueError("Model manifest contains an invalid model.files entry")
            member_path = pack._member_path(str(member["file"]))
            if not member_path.is_file():
                raise FileNotFoundError(f"Missing model artifact: {member_path}")
            member_hash = str(member.get("sha256") or "").lower()
            if member_hash and sha256_file(member_path) != member_hash:
                raise ValueError(f"Model hash mismatch for {member_path}")
        return pack

    def _member_path(self, value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError("Model path must be relative to its pack")
        resolved = (self.root / relative).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Model path escapes its pack: {value!r}") from exc
        return resolved

    @property
    def model_path(self) -> Path:
        return self._member_path(str(self.manifest["model"]["file"]))

    @property
    def profile(self) -> str:
        """Compatibility alias for older callers."""
        return self.variant_id

    @property
    def variant_id(self) -> str:
        return str(self.manifest.get("variant_id") or self.manifest.get("profile"))

    @property
    def labels(self) -> list[str]:
        return [str(value) for value in self.manifest["labels"]]

    @property
    def output_names(self) -> dict[str, str]:
        outputs = self.manifest["model"].get("outputs") or {}
        return {str(key): str(value) for key, value in outputs.items() if key != "contract"}

    @property
    def output_contract(self) -> str:
        outputs = self.manifest["model"].get("outputs") or {}
        return str(outputs.get("contract") or "decoded_boxes")

    @property
    def model_bytes(self) -> int:
        return int(self.manifest["model"].get("bytes") or self.model_path.stat().st_size)


def load_rgb(path: str | Path, backend: str) -> np.ndarray:
    """Decode an image without imposing an image library on the core runtime."""

    if backend == "opencv":
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install the images-opencv extra to decode with OpenCV") from exc
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if backend == "pillow":
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install the images-pillow extra to decode with Pillow") from exc
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    raise ValueError(f"Unsupported image backend: {backend!r}")


def prepare_rgb(
    image: np.ndarray,
    target_size: Sequence[int] = (800, 800),
    *,
    backend: str = "opencv",
    scale: float = 1.0 / 255.0,
    mean: Sequence[float] = (0.0, 0.0, 0.0),
    std: Sequence[float] = (1.0, 1.0, 1.0),
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """Prepare an RGB uint8 array using the preprocessing pinned by a model pack."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {image.shape}")
    height, width = image.shape[:2]
    target_height, target_width = (int(target_size[0]), int(target_size[1]))
    if backend == "opencv":
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install the images-opencv extra to resize with OpenCV") from exc
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    elif backend == "pillow":
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install the images-pillow extra to resize with Pillow") from exc
        resized = np.asarray(
            Image.fromarray(image).resize((target_width, target_height), Image.Resampling.BICUBIC)
        )
    else:
        raise ValueError(f"Unsupported image backend: {backend!r}")
    tensor = resized.astype(np.float32)
    tensor *= float(scale)
    tensor -= np.asarray(mean, dtype=np.float32)
    tensor /= np.asarray(std, dtype=np.float32)
    tensor = tensor.transpose((2, 0, 1))
    return (
        {
            "im_shape": np.asarray([target_height, target_width], dtype=np.float32),
            "image": tensor,
            "scale_factor": np.asarray(
                [target_height / height, target_width / width], dtype=np.float32
            ),
        },
        (width, height),
    )


def prepare_image(
    path: str | Path,
    target_size: Sequence[int] = (800, 800),
    *,
    backend: str = "opencv",
    scale: float = 1.0 / 255.0,
    mean: Sequence[float] = (0.0, 0.0, 0.0),
    std: Sequence[float] = (1.0, 1.0, 1.0),
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    return prepare_rgb(
        load_rgb(path, backend),
        target_size,
        backend=backend,
        scale=scale,
        mean=mean,
        std=std,
    )


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _nms(boxes: np.ndarray, iou_same: float = 0.6, iou_diff: float = 0.98) -> np.ndarray:
    remaining = list(np.argsort(boxes[:, 1])[::-1])
    selected: list[int] = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [
            index
            for index in remaining
            if _iou(boxes[current, 2:6], boxes[index, 2:6])
            < (iou_same if boxes[current, 0] == boxes[index, 0] else iou_diff)
        ]
    return boxes[selected]


def _overlap_of_smaller(left: Sequence[int], right: Sequence[int]) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    smaller = min(left_area, right_area)
    return intersection / smaller if smaller > 0 else 0.0


def _filter_overlaps(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # PaddleX 3.6's rectangular layout filter deliberately suppresses the
    # generic reference class in favour of reference_content.
    boxes = [box for box in boxes if box["label"] != "reference"]
    dropped: set[int] = set()
    visual = {"image", "table", "seal", "chart"}
    for left_index, left in enumerate(boxes):
        x0, y0, x1, y1 = left["coordinate"]
        if x1 - x0 < 6 or y1 - y0 < 6:
            dropped.add(left_index)
        for right_index in range(left_index + 1, len(boxes)):
            if left_index in dropped or right_index in dropped:
                continue
            right = boxes[right_index]
            overlap = _overlap_of_smaller(left["coordinate"], right["coordinate"])
            if overlap <= 0.7:
                continue
            if "inline_formula" in {left["label"], right["label"]}:
                if overlap > 0.5:
                    if left["label"] == "inline_formula":
                        dropped.add(left_index)
                    if right["label"] == "inline_formula":
                        dropped.add(right_index)
                continue
            labels = {left["label"], right["label"]}
            if labels & visual and len(labels) > 1:
                if "table" not in labels or labels <= visual:
                    continue
            left_area = (left["coordinate"][2] - left["coordinate"][0]) * (
                left["coordinate"][3] - left["coordinate"][1]
            )
            right_area = (right["coordinate"][2] - right["coordinate"][0]) * (
                right["coordinate"][3] - right["coordinate"][1]
            )
            dropped.add(right_index if left_area >= right_area else left_index)
    return [box for index, box in enumerate(boxes) if index not in dropped]


def postprocess_boxes(
    raw_boxes: np.ndarray,
    *,
    labels: Sequence[str],
    image_size: tuple[int, int],
    threshold: float,
    layout_nms: bool = False,
    filter_overlap_boxes: bool = True,
) -> list[dict[str, Any]]:
    boxes = np.asarray(raw_boxes, dtype=np.float32).copy()
    if boxes.size == 0:
        return []
    boxes = boxes.reshape((-1, boxes.shape[-1]))
    boxes[:, 2:6] = np.round(boxes[:, 2:6])
    boxes = boxes[(boxes[:, 1] > float(threshold)) & (boxes[:, 0] > -1)]
    if not len(boxes):
        return []
    if layout_nms:
        boxes = _nms(boxes)

    width, height = image_size
    if len(boxes) > 1:
        area_threshold = 0.82 if width > height else 0.93
        image_label = labels.index("image") if "image" in labels else None
        filtered = []
        for box in boxes:
            if image_label is None or int(box[0]) != image_label:
                filtered.append(box)
                continue
            x0, y0, x1, y1 = box[2:6]
            area = (min(width, x1) - max(0, x0)) * (min(height, y1) - max(0, y0))
            if area <= area_threshold * width * height:
                filtered.append(box)
        if filtered:
            boxes = np.asarray(filtered)

    if boxes.shape[1] >= 7:
        boxes = boxes[np.argsort(boxes[:, 6])]
    rows: list[dict[str, Any]] = []
    for box in boxes:
        class_id = int(box[0])
        if class_id < 0 or class_id >= len(labels):
            continue
        x0 = int(max(0, box[2]))
        y0 = int(max(0, box[3]))
        x1 = int(min(width, box[4]))
        y1 = int(min(height, box[5]))
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append(
            {
                "cls_id": class_id,
                "label": str(labels[class_id]),
                "score": float(box[1]),
                "coordinate": [x0, y0, x1, y1],
                "order": len(rows) + 1,
            }
        )
    if filter_overlap_boxes:
        rows = _filter_overlaps(rows)
    order = 1
    for row in rows:
        if row["label"] in SKIP_ORDER_LABELS:
            row["order"] = None
        else:
            row["order"] = order
            order += 1
    return rows


def normalized_detections(boxes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "score": float(box["score"]),
            "label_id": int(box["cls_id"]),
            "label": str(box["label"]),
            "box": [float(value) for value in box["coordinate"]],
            "raw": dict(box),
        }
        for box in boxes
    ]


def decode_rtdetr_raw(
    boxes: np.ndarray,
    logits: np.ndarray,
    feeds: Mapping[str, np.ndarray],
    top_k: int,
) -> np.ndarray:
    centers = boxes[..., :2]
    half_size = boxes[..., 2:] * np.float32(0.5)
    xyxy = np.concatenate((centers - half_size, centers + half_size), axis=-1)
    original_shape = np.floor(
        feeds["im_shape"] / feeds["scale_factor"] + np.float32(0.5)
    )
    xyxy *= np.tile(original_shape[:, ::-1], (1, 2))[:, None, :]

    scores = np.float32(1.0) / (np.float32(1.0) + np.exp(-logits))
    flat = scores.reshape(scores.shape[0], -1)
    flat_indices = np.argsort(-flat, axis=1, kind="stable")[:, :top_k]
    selected_scores = np.take_along_axis(flat, flat_indices, axis=1)
    labels = flat_indices % logits.shape[-1]
    queries = flat_indices // logits.shape[-1]
    selected_boxes = xyxy[np.arange(xyxy.shape[0])[:, None], queries]
    order = np.full(selected_scores.shape, -1, dtype=np.float32)
    return np.concatenate(
        (
            labels[..., None].astype(np.float32),
            selected_scores[..., None],
            selected_boxes.astype(np.float32),
            order[..., None],
        ),
        axis=-1,
    )


def decode_ppyoloe_raw(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    score_threshold: float,
    nms_threshold: float,
    nms_top_k: int,
    keep_top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply PaddleDetection's class-wise MultiClassNMS to raw PP-YOLOE outputs."""

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected PP-YOLOE boxes [B,Q,4], got {boxes.shape}")
    if scores.ndim != 3 or scores.shape[0] != boxes.shape[0] or scores.shape[2] != boxes.shape[1]:
        raise ValueError(
            f"Expected PP-YOLOE scores [B,C,Q] matching boxes, got {scores.shape}"
        )
    if nms_top_k < 1 or keep_top_k < 1:
        raise ValueError("PP-YOLOE NMS top-k values must be positive")

    def iou(left: np.ndarray, right: np.ndarray) -> float:
        width = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])))
        height = max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])))
        intersection = width * height
        left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
        right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
        union = left_area + right_area - intersection
        return intersection / union if union > 0.0 else 0.0

    batches: list[np.ndarray] = []
    counts: list[int] = []
    for batch in range(boxes.shape[0]):
        candidates: list[tuple[float, int, int, np.ndarray]] = []
        for label in range(scores.shape[1]):
            per_class = []
            for query in range(boxes.shape[1]):
                score = float(scores[batch, label, query])
                box = boxes[batch, query]
                if (
                    np.isfinite(score)
                    and score > score_threshold
                    and np.isfinite(box).all()
                    and box[2] > box[0]
                    and box[3] > box[1]
                ):
                    per_class.append((score, label, query, box))
            per_class.sort(key=lambda row: (-row[0], row[2]))
            candidates.extend(per_class[:nms_top_k])
        candidates.sort(key=lambda row: (-row[0], row[1], row[2]))

        selected_by_class: list[list[np.ndarray]] = [[] for _ in range(scores.shape[1])]
        selected: list[list[float]] = []
        for score, label, _query, box in candidates:
            if all(iou(box, prior) <= nms_threshold for prior in selected_by_class[label]):
                selected_by_class[label].append(box)
                selected.append(
                    [float(label), score, *(float(value) for value in box), -1.0]
                )
                if len(selected) == keep_top_k:
                    break
        rows = np.asarray(selected, dtype=np.float32).reshape((-1, 7))
        batches.append(rows)
        counts.append(len(rows))
    flattened = np.concatenate(batches, axis=0) if batches else np.empty((0, 7), dtype=np.float32)
    return flattened, np.asarray(counts, dtype=np.int32)


def _provider_name(device: str, available: Sequence[str]) -> str:
    choices = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "directml": "DmlExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
    }
    if device != "auto":
        try:
            provider = choices[device]
        except KeyError as exc:
            raise ValueError(f"Unsupported device: {device!r}") from exc
        if provider not in available:
            raise RuntimeError(f"{provider} is unavailable; installed providers: {list(available)}")
        return provider
    for provider in (
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ):
        if provider in available:
            return provider
    raise RuntimeError("ONNX Runtime exposes no usable execution provider")


class PPDocLite:
    def __init__(
        self,
        pack: ModelPack,
        *,
        device: str = "auto",
        providers: Sequence[str] | None = None,
        threads: int = 0,
        inter_threads: int = 1,
        strict_device: bool = False,
        image_backend: str = "opencv",
        graph_optimization: str = "all",
        execution_mode: str = "sequential",
        allow_spinning: bool = True,
        cpu_mem_arena: bool = True,
        disable_prepacking: bool = False,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required. Install the runtime-cpu, runtime-cuda, "
                "or runtime-directml extra."
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(0, int(threads))
        options.inter_op_num_threads = max(1, int(inter_threads))
        modes = {
            "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
            "parallel": ort.ExecutionMode.ORT_PARALLEL,
        }
        optimizations = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        try:
            options.execution_mode = modes[execution_mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported execution mode: {execution_mode!r}") from exc
        try:
            options.graph_optimization_level = optimizations[graph_optimization]
        except KeyError as exc:
            raise ValueError(f"Unsupported graph optimization: {graph_optimization!r}") from exc
        options.enable_cpu_mem_arena = bool(cpu_mem_arena)
        options.add_session_config_entry(
            "session.intra_op.allow_spinning", "1" if allow_spinning else "0"
        )
        if disable_prepacking:
            options.add_session_config_entry("session.disable_prepacking", "1")
        available = tuple(ort.get_available_providers())
        configured = list(providers) if providers else [_provider_name(device, available)]
        if not strict_device and "CPUExecutionProvider" in available and "CPUExecutionProvider" not in configured:
            configured.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(
            str(pack.model_path), sess_options=options, providers=configured
        )
        self._input_names = tuple(value.name for value in self._session.get_inputs())
        self.pack = pack
        self.providers = tuple(self._session.get_providers())
        self._outputs = pack.output_names
        self._output_contract = pack.output_contract
        input_config = pack.manifest.get("input", {})
        self._target_size = tuple(input_config.get("target_size", [800, 800]))
        self._scale = float(input_config.get("scale", 1.0 / 255.0))
        self._mean = tuple(input_config.get("mean", [0.0, 0.0, 0.0]))
        self._std = tuple(input_config.get("std", [1.0, 1.0, 1.0]))
        self.image_backend = image_backend

    def _raw_predictions(self, feeds: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self._output_contract == "decoded_boxes":
            box_name = self._outputs.get("boxes", "fetch_name_0")
            count_name = self._outputs.get("counts")
            names = [box_name, *([count_name] if count_name else [])]
            values = self._session.run(
                names, {name: feeds[name] for name in self._input_names}
            )
            boxes = np.asarray(values[0])
            if count_name:
                counts = np.asarray(values[1]).reshape(-1)
            else:
                batch = int(feeds["image"].shape[0])
                fixed = int(self.pack.manifest["model"].get("detections_per_image") or boxes.shape[0] // batch)
                counts = np.full((batch,), fixed, dtype=np.int32)
            return boxes.reshape((-1, boxes.shape[-1])), counts
        if self._output_contract == "ppdoc_rect_parts":
            names = [self._outputs[key] for key in ("classes", "scores", "coordinates")]
            classes, scores, coordinates = self._session.run(
                names, {name: feeds[name] for name in self._input_names}
            )
            boxes = np.concatenate((classes, scores, coordinates), axis=-1)
            counts = np.full((boxes.shape[0],), boxes.shape[1], dtype=np.int32)
            return boxes.reshape((-1, boxes.shape[-1])), counts
        if self._output_contract == "rtdetr_raw":
            names = [self._outputs[key] for key in ("boxes", "logits")]
            boxes, logits = self._session.run(
                names, {name: feeds[name] for name in self._input_names}
            )
            top_k = int(
                self.pack.manifest["model"].get("detections_per_image") or 300
            )
            decoded = decode_rtdetr_raw(boxes, logits, feeds, top_k)
            counts = np.full((decoded.shape[0],), decoded.shape[1], dtype=np.int32)
            return decoded.reshape((-1, decoded.shape[-1])), counts
        if self._output_contract == "ppyoloe_raw":
            names = [self._outputs[key] for key in ("boxes", "scores")]
            boxes, scores = self._session.run(
                names, {name: feeds[name] for name in self._input_names}
            )
            model_nms = self.pack.manifest.get("postprocess", {}).get("model_nms", {})
            return decode_ppyoloe_raw(
                boxes,
                scores,
                score_threshold=float(model_nms.get("score_threshold", 0.01)),
                nms_threshold=float(model_nms.get("nms_threshold", 0.7)),
                nms_top_k=int(model_nms.get("nms_top_k", 1_000)),
                keep_top_k=int(
                    model_nms.get("keep_top_k")
                    or self.pack.manifest["model"].get("detections_per_image")
                    or 300
                ),
            )
        raise ValueError(f"Unsupported output contract: {self._output_contract!r}")

    def infer_tensors(
        self,
        feeds: Mapping[str, np.ndarray],
        image_sizes: Sequence[tuple[int, int]],
        *,
        image_ids: Sequence[str | None] | None = None,
        threshold: float = 0.10,
        layout_nms: bool = False,
        filter_overlap_boxes: bool = True,
    ) -> list[dict[str, Any]]:
        """Infer from caller-preprocessed tensors without an image-library dependency.

        ``feeds`` follows the PaddleDetection contract: batched ``image`` NCHW
        float32 data plus ``im_shape`` and ``scale_factor`` arrays. Image sizes
        are original ``(width, height)`` pairs used to clip decoded boxes.
        """

        required = {"im_shape", "image", "scale_factor"}
        missing = required.difference(feeds)
        if missing:
            raise ValueError(f"Missing model inputs: {sorted(missing)}")
        image = np.asarray(feeds["image"])
        if image.ndim != 4:
            raise ValueError(f"Expected batched NCHW image input, got {image.shape}")
        batch_size = int(image.shape[0])
        if len(image_sizes) != batch_size:
            raise ValueError(
                f"Got {len(image_sizes)} image sizes for a batch of {batch_size}"
            )
        identifiers = list(image_ids) if image_ids is not None else [None] * batch_size
        if len(identifiers) != batch_size:
            raise ValueError(
                f"Got {len(identifiers)} image IDs for a batch of {batch_size}"
            )

        boxes, counts = self._raw_predictions(feeds)
        flat_counts = np.asarray(counts).reshape(-1)
        if len(flat_counts) != batch_size:
            raise ValueError(
                f"Model returned {len(flat_counts)} counts for a batch of {batch_size}"
            )
        results: list[dict[str, Any]] = []
        offset = 0
        for identifier, image_size, count_value in zip(
            identifiers, image_sizes, flat_counts, strict=True
        ):
            count = int(count_value)
            raw = boxes[offset : offset + count]
            offset += count
            paddle_boxes = postprocess_boxes(
                raw,
                labels=self.pack.labels,
                image_size=image_size,
                threshold=threshold,
                layout_nms=layout_nms,
                filter_overlap_boxes=filter_overlap_boxes,
            )
            results.append(
                {
                    "image": identifier,
                    "image_size": list(image_size),
                    "detections": normalized_detections(paddle_boxes),
                    "raw_payloads": [{"input_path": identifier, "boxes": paddle_boxes}],
                }
            )
        return results

    def infer_rgb(
        self,
        images: Sequence[np.ndarray],
        *,
        image_ids: Sequence[str | None] | None = None,
        threshold: float = 0.10,
        layout_nms: bool = False,
        filter_overlap_boxes: bool = True,
    ) -> list[dict[str, Any]]:
        """Infer from caller-owned RGB arrays using the selected resize backend."""

        if not images:
            return []
        prepared = [
            prepare_rgb(
                image,
                self._target_size,
                backend=self.image_backend,
                scale=self._scale,
                mean=self._mean,
                std=self._std,
            )
            for image in images
        ]
        feeds = {
            name: np.stack([row[0][name] for row in prepared], axis=0)
            for name in ("im_shape", "image", "scale_factor")
        }
        return self.infer_tensors(
            feeds,
            [row[1] for row in prepared],
            image_ids=image_ids,
            threshold=threshold,
            layout_nms=layout_nms,
            filter_overlap_boxes=filter_overlap_boxes,
        )

    def infer(
        self,
        paths: Sequence[str | Path],
        *,
        threshold: float = 0.10,
        layout_nms: bool = False,
        filter_overlap_boxes: bool = True,
    ) -> list[dict[str, Any]]:
        if not paths:
            return []
        identifiers = [str(path) for path in paths]
        return self.infer_rgb(
            [load_rgb(path, self.image_backend) for path in paths],
            image_ids=identifiers,
            threshold=threshold,
            layout_nms=layout_nms,
            filter_overlap_boxes=filter_overlap_boxes,
        )
