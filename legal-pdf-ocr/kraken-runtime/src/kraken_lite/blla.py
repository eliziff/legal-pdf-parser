# This module is a dependency-light reimplementation of algorithms used by
# Kraken's BLLA inference pipeline (Apache-2.0). See README.md, Attribution.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from .geometry import (
    close_polygon,
    point_in_polygon,
    polygonize_baseline,
    polyline_point_at_fraction,
    reading_order,
    resample_polyline,
    simplify_polyline,
    skeleton_main_paths,
    zhang_suen_thinning,
)
from .model_pack import ModelPack, Session
from .types import BaselineLine, Point, Region, Segmentation


@dataclass(slots=True)
class PreparedPage:
    tensor: np.ndarray
    padded_gray: np.ndarray
    gray: np.ndarray
    padding: tuple[int, int, int, int]
    source_size: tuple[int, int]


def _fixed_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    source_width, source_height = image.size
    if height <= 0 and width <= 0:
        return image
    if height <= 0:
        # Match Kraken's ``pil_fixed_resize`` exactly: proportional dimensions
        # are truncated with ``int`` rather than rounded. A one-pixel mismatch
        # changes the BLLA heatmap-to-page scale and therefore every baseline
        # and polygon coordinate downstream.
        height = max(1, int(source_height * width / source_width))
    elif width <= 0:
        width = max(1, int(source_width * height / source_height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def prepare_blla_page(image: Image.Image, manifest: Mapping[str, Any]) -> PreparedPage:
    spec = manifest.get("input", {})
    channels = int(spec.get("channels", 1))
    height = int(spec.get("height", 0))
    width = int(spec.get("width", 0))
    if channels not in (1, 3):
        raise ValueError(f"BLLA input channels must be 1 or 3, got {channels}")
    converted = image.convert("L" if channels == 1 else "RGB")
    resized = _fixed_resize(converted, height, width)

    padding_value = spec.get("padding", [0, 0, 0, 0])
    if isinstance(padding_value, int):
        left = right = top = bottom = int(padding_value)
    elif len(padding_value) == 2:
        left = right = int(padding_value[0])
        top = bottom = int(padding_value[1])
    elif len(padding_value) == 4:
        left, right, top, bottom = (int(value) for value in padding_value)
    else:
        raise ValueError("input.padding must be an int, pair, or four-tuple")

    gray = np.asarray(resized.convert("L"), dtype=np.uint8)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if channels == 1:
        array = array[..., None]
    if any((left, right, top, bottom)):
        array = np.pad(
            array,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=1.0,
        )
        padded_gray = np.pad(
            gray,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=255,
        )
    else:
        padded_gray = gray
    if bool(spec.get("invert", True)):
        # Kraken's runtime uses ``tensor.max() - tensor``. Do not assume a
        # white point of 1.0: tightly cropped or unusually dark pages can have
        # a lower maximum, and changing it alters the model input distribution.
        maximum = float(array.max()) if array.size else 1.0
        array = maximum - array
    tensor = np.transpose(array, (2, 0, 1))[None].astype(np.float32, copy=False)
    return PreparedPage(
        tensor=tensor,
        padded_gray=padded_gray,
        gray=gray,
        padding=(left, right, top, bottom),
        source_size=image.size,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp = np.exp(values[~positive])
    result[~positive] = exp / (1.0 + exp)
    return result


def _resize_heatmap(heatmap: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.stack(
        [
            cv2.resize(channel, (width, height), interpolation=cv2.INTER_NEAREST)
            for channel in heatmap
        ],
        axis=0,
    )


def _sato_like_ridge(probability: np.ndarray) -> np.ndarray:
    """Multi-scale Hessian ridge response approximating skimage.filters.sato.

    BLLA emits a thick baseline probability ribbon. Kraken runs Sato filtering
    before skeletonization; this implementation preserves that model contract
    without importing SciPy or scikit-image.
    """

    image = np.asarray(probability, dtype=np.float32)
    responses: list[np.ndarray] = []
    minimum_side = min(image.shape)
    for sigma in (1.0, 3.0, 5.0, 7.0, 9.0):
        if sigma * 4 >= minimum_side:
            continue
        blurred = cv2.GaussianBlur(
            image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT
        )
        dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * sigma * sigma
        dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * sigma * sigma
        dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * sigma * sigma
        trace = dxx + dyy
        discriminant = np.sqrt(np.maximum(0.0, (dxx - dyy) ** 2 + 4.0 * dxy**2))
        first = 0.5 * (trace - discriminant)
        second = 0.5 * (trace + discriminant)
        # Bright ridges have a large negative curvature across the ridge and
        # little curvature along it. Suppress blob-like responses.
        across = np.minimum(first, second)
        along = np.maximum(first, second)
        response = np.maximum(0.0, -across) * np.exp(
            -np.square(along / (across - 1e-6)) / 2.0
        )
        responses.append(response)
    if not responses:
        return image
    ridge = np.max(np.stack(responses, axis=0), axis=0)
    scale = float(np.percentile(ridge, 99.5))
    if scale <= 1e-8:
        return image
    return np.clip(ridge / scale, 0.0, 1.0)


def _endpoint_score(channel: np.ndarray, point: Point, radius: int = 10) -> float:
    x, y = (int(round(value)) for value in point)
    y0 = max(0, y - radius)
    y1 = min(channel.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(channel.shape[1], x + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(channel[y0:y1, x0:x1].max(initial=0.0))


def vectorize_blla_baselines(
    maps: np.ndarray,
    *,
    threshold: float = 0.17,
    min_length: float = 5.0,
    text_direction: str = "horizontal",
) -> list[list[Point]]:
    if maps.shape[0] != 3:
        raise ValueError("BLLA baseline vectorization expects start/end/baseline maps")
    start_map, end_map, baseline_map = maps
    ridge = _sato_like_ridge(baseline_map)
    binary = ridge > float(threshold)
    # Retain extremely confident model pixels even where the finite-difference
    # ridge response is weak at short line ends.
    binary |= baseline_map > max(0.65, threshold * 2.5)
    binary = cv2.morphologyEx(
        binary.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    skeleton = zhang_suen_thinning(binary)
    paths = skeleton_main_paths(skeleton, min_length=min_length)
    oriented: list[list[Point]] = []
    for path in paths:
        path = simplify_polyline(path, epsilon=2.5)
        if len(path) < 2:
            continue
        left = path[0]
        right = path[-1]
        left_start = _endpoint_score(start_map, left)
        left_end = _endpoint_score(end_map, left)
        right_start = _endpoint_score(start_map, right)
        right_end = _endpoint_score(end_map, right)
        if left_start - left_end > 0.05 and right_end - right_start > 0.05:
            pass
        elif left_end - left_start > 0.05 and right_start - right_end > 0.05:
            path = list(reversed(path))
        elif text_direction == "horizontal" and path[0][0] > path[-1][0]:
            path = list(reversed(path))
        elif text_direction == "vertical" and path[0][1] > path[-1][1]:
            path = list(reversed(path))
        length = sum(
            math.dist(first, second) for first, second in zip(path[:-1], path[1:], strict=True)
        )
        if length >= min_length:
            oriented.append(path)
    return oriented


def vectorize_regions(
    probability: np.ndarray, *, threshold: float = 0.5, simplify: float = 4.0
) -> list[list[Point]]:
    binary = (np.asarray(probability) > threshold).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[Point]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 8.0:
            continue
        approximation = cv2.approxPolyDP(contour, simplify, True).reshape(-1, 2)
        if len(approximation) >= 3:
            polygons.append(close_polygon([(float(x), float(y)) for x, y in approximation]))
    return polygons


def _clip_points(points: Sequence[Point], width: int, height: int) -> list[Point]:
    return [
        (
            float(np.clip(x, 0, width - 1)),
            float(np.clip(y, 0, height - 1)),
        )
        for x, y in points
    ]


def _scale_points(points: Sequence[Point], sx: float, sy: float) -> list[Point]:
    return [(float(x * sx), float(y * sy)) for x, y in points]


class BLLASegmenter:
    """Stock-BLLA-compatible segmentation and geometry runtime."""

    def __init__(
        self,
        session: Session,
        manifest: Mapping[str, Any],
    ) -> None:
        if manifest.get("kind") != "blla-segmentation":
            raise ValueError("BLLASegmenter requires a blla-segmentation manifest")
        self.session = session
        self.manifest = dict(manifest)

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
    ) -> "BLLASegmenter":
        pack = ModelPack.load(path)
        if pack.kind != "blla-segmentation":
            raise ValueError(f"Expected a BLLA pack, got {pack.kind}")
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
        )

    @property
    def default_batch_size(self) -> int:
        execution = getattr(self.session, "execution_info", None)
        segmentation = self.manifest.get("segmentation", {})
        if execution is not None and execution.accelerated:
            return int(segmentation.get("accelerated_batch_size", 4))
        return int(segmentation.get("cpu_batch_size", 1))

    def segment(
        self,
        image: Image.Image,
        *,
        text_direction: str = "horizontal-lr",
        raise_on_error: bool = False,
        batch_size: int | None = None,
    ) -> Segmentation:
        return self.segment_many(
            [image],
            text_direction=text_direction,
            raise_on_error=raise_on_error,
            batch_size=1 if batch_size is None else batch_size,
        )[0]

    def segment_many(
        self,
        images: Sequence[Image.Image],
        *,
        text_direction: str = "horizontal-lr",
        raise_on_error: bool = False,
        batch_size: int | None = None,
        width_bucket: int | None = None,
    ) -> list[Segmentation]:
        """Runs stock BLLA in page batches without changing decoded geometry.

        Exact tensor shapes are batched by default, which is common for pages
        rasterized from one PDF and preserves single-page convolution boundary
        semantics. A positive ``width_bucket`` explicitly opts into right-side
        zero padding for higher throughput; it remains a benchmark-only mode
        until corpus parity confirms the model's right-edge behaviour.
        """

        if not images:
            return []
        prepared = [prepare_blla_page(image, self.manifest) for image in images]
        segmentation_spec = self.manifest.get("segmentation", {})
        model_spec = self.manifest.get("model", {})
        dynamic_batch = bool(model_spec.get("dynamic_batch", False))
        if batch_size is None or batch_size <= 0:
            batch_size = self.default_batch_size
        if not dynamic_batch:
            batch_size = 1
        batch_size = max(1, int(batch_size))
        if width_bucket is None:
            width_bucket = int(segmentation_spec.get("width_bucket", 0))
        width_bucket = max(0, int(width_bucket))
        maximum_pixels = int(
            segmentation_spec.get("maximum_batch_pixels", 80_000_000)
        )

        groups: dict[tuple[int, int, int], list[int]] = {}
        for index, item in enumerate(prepared):
            _, channels, height, width = item.tensor.shape
            grouped_width = (
                ((width + width_bucket - 1) // width_bucket) * width_bucket
                if width_bucket > 0
                else width
            )
            groups.setdefault((channels, height, grouped_width), []).append(index)

        input_name = str(model_spec.get("input", "image"))
        output_name = str(model_spec.get("output", "heatmap"))
        results: list[Segmentation | None] = [None] * len(images)
        for channels, height, grouped_width in sorted(groups):
            indices = groups[(channels, height, grouped_width)]
            pixel_limited = max(
                1,
                maximum_pixels // max(1, channels * height * grouped_width),
            )
            effective_batch = min(batch_size, pixel_limited)
            for offset in range(0, len(indices), effective_batch):
                batch_indices = indices[offset : offset + effective_batch]
                maximum_width = max(prepared[index].tensor.shape[3] for index in batch_indices)
                maximum_height = max(prepared[index].tensor.shape[2] for index in batch_indices)
                tensor = np.zeros(
                    (len(batch_indices), channels, maximum_height, maximum_width),
                    dtype=np.float32,
                )
                for local_index, source_index in enumerate(batch_indices):
                    value = prepared[source_index].tensor[0]
                    tensor[
                        local_index,
                        :,
                        : value.shape[1],
                        : value.shape[2],
                    ] = value
                outputs = self.session.run({input_name: tensor})
                if output_name not in outputs:
                    raise RuntimeError(
                        f"BLLA graph did not return {output_name!r}; got {list(outputs)}"
                    )
                heatmaps = np.asarray(outputs[output_name])
                if heatmaps.ndim == 3:
                    heatmaps = heatmaps[None, ...]
                if heatmaps.ndim != 4 or heatmaps.shape[0] != len(batch_indices):
                    raise RuntimeError(
                        "Unexpected BLLA batch output shape "
                        f"{heatmaps.shape}; expected batch {len(batch_indices)}"
                    )
                for local_index, source_index in enumerate(batch_indices):
                    item = prepared[source_index]
                    # Match Kraken's order: resize logits first, then sigmoid.
                    page_logits = _resize_heatmap(
                        heatmaps[local_index],
                        (maximum_height, maximum_width),
                    )
                    own_height, own_width = item.tensor.shape[2:]
                    page_logits = page_logits[:, :own_height, :own_width]
                    results[source_index] = self._decode_prepared(
                        item,
                        page_logits,
                        text_direction=text_direction,
                        raise_on_error=raise_on_error,
                        execution_batch_size=len(batch_indices),
                    )

        if any(result is None for result in results):  # pragma: no cover - invariant
            raise RuntimeError("BLLA page scheduler did not return every page")
        return [result for result in results if result is not None]

    def _decode_prepared(
        self,
        prepared: PreparedPage,
        heatmap: np.ndarray,
        *,
        text_direction: str,
        raise_on_error: bool,
        execution_batch_size: int,
    ) -> Segmentation:
        if heatmap.ndim != 3:
            raise RuntimeError(f"Unexpected BLLA output shape {heatmap.shape}")
        activation = self.manifest.get("output", {}).get("activation", "sigmoid")
        if activation == "sigmoid":
            heatmap = _sigmoid(heatmap)
        elif activation != "none":
            raise ValueError(f"Unsupported BLLA output activation {activation!r}")

        left, right, top, bottom = prepared.padding
        y1 = heatmap.shape[1] - bottom if bottom else heatmap.shape[1]
        x1 = heatmap.shape[2] - right if right else heatmap.shape[2]
        heatmap = heatmap[:, top:y1, left:x1]
        gray = prepared.padded_gray[top:y1, left:x1]

        segmentation_spec = self.manifest.get("segmentation", {})
        class_mapping = segmentation_spec.get("class_mapping", {})
        aux_mapping = class_mapping.get("aux", {})
        baseline_mapping = class_mapping.get("baselines", {})
        region_mapping = class_mapping.get("regions", {})
        start_index = int(aux_mapping.get("_start_separator", 0))
        end_index = int(aux_mapping.get("_end_separator", 1))
        if not baseline_mapping:
            raise ValueError("BLLA manifest has no baseline class mapping")

        model_regions: dict[str, list[Region]] = {}
        region_counter = 0
        for region_type, channel in region_mapping.items():
            polygons = vectorize_regions(
                heatmap[int(channel)],
                threshold=float(segmentation_spec.get("region_threshold", 0.5)),
            )
            values: list[Region] = []
            for polygon in polygons:
                region_counter += 1
                values.append(
                    Region(
                        id=f"region-{region_counter:06d}",
                        type=str(region_type),
                        boundary=polygon,
                        tags={"type": str(region_type)},
                    )
                )
            model_regions[str(region_type)] = values

        all_regions = [value for values in model_regions.values() for value in values]
        raw_lines: list[tuple[str, list[Point]]] = []
        direction = "horizontal" if text_direction.startswith("horizontal") else "vertical"
        for baseline_type, channel in baseline_mapping.items():
            maps = heatmap[[start_index, end_index, int(channel)]]
            paths = vectorize_blla_baselines(
                maps,
                threshold=float(segmentation_spec.get("baseline_threshold", 0.17)),
                min_length=float(segmentation_spec.get("minimum_baseline_length", 5.0)),
                text_direction=direction,
            )
            raw_lines.extend((str(baseline_type), path) for path in paths)

        topline = segmentation_spec.get("topline", False)
        lines_model: list[BaselineLine] = []
        for index, (baseline_type, baseline) in enumerate(raw_lines):
            containing_region = next(
                (
                    region
                    for region in all_regions
                    if point_in_polygon(
                        polyline_point_at_fraction(baseline, 0.5),
                        region.boundary,
                    )
                ),
                None,
            )
            others = [candidate for _, candidate in raw_lines if candidate is not baseline]
            try:
                boundary = polygonize_baseline(
                    gray,
                    baseline,
                    other_baselines=others,
                    region=containing_region,
                    maximum_distance=float(
                        segmentation_spec.get("maximum_polygon_distance", 72.0)
                    ),
                    topline=topline,
                )
            except Exception:
                if raise_on_error:
                    raise
                # polygonize_baseline has its own geometric fallback; this is
                # only a final fail-closed guard for malformed model output.
                from .geometry import fallback_boundary

                boundary = fallback_boundary(baseline)
            lines_model.append(
                BaselineLine(
                    id=f"line-{index + 1:06d}",
                    type=baseline_type,
                    baseline=baseline,
                    boundary=_clip_points(boundary, gray.shape[1], gray.shape[0]),
                    regions=[containing_region.id] if containing_region else [],
                    tags={"type": baseline_type},
                )
            )

        model_order = reading_order(
            lines_model,
            all_regions,
            text_direction=text_direction[-2:],
        )
        lines_model = [lines_model[index] for index in model_order]

        source_width, source_height = prepared.source_size
        sx = source_width / max(1, gray.shape[1])
        sy = source_height / max(1, gray.shape[0])
        scaled_regions: dict[str, list[Region]] = {}
        for kind, values in model_regions.items():
            scaled_regions[kind] = [
                Region(
                    id=value.id,
                    type=value.type,
                    boundary=_scale_points(value.boundary, sx, sy),
                    tags=value.tags,
                )
                for value in values
            ]
        scaled_lines = [
            BaselineLine(
                id=line.id,
                type=line.type,
                baseline=_scale_points(line.baseline, sx, sy),
                boundary=_scale_points(line.boundary, sx, sy),
                regions=line.regions,
                tags=line.tags,
            )
            for line in lines_model
        ]

        return Segmentation(
            width=source_width,
            height=source_height,
            lines=scaled_lines,
            regions=scaled_regions,
            text_direction=text_direction,  # type: ignore[arg-type]
            script_detection=len(baseline_mapping) > 1,
            line_orders=[],
            model={
                "id": self.manifest.get("id", "stock-blla"),
                "source": self.manifest.get("source", {}),
                "execution": (
                    self.session.execution_info.to_dict()
                    if getattr(self.session, "execution_info", None) is not None
                    else None
                ),
                "batchSize": int(execution_batch_size),
            },
        )
