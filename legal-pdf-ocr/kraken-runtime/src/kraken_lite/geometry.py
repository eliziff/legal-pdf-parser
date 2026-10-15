# This module is a dependency-light reimplementation of algorithms used by
# Kraken's BLLA inference pipeline (Apache-2.0). See README.md, Attribution.

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from .types import BaselineLine, Point, Polygon, Polyline, Region

_EPS = np.finfo(np.float64).eps
_NEIGHBOURS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _as_array(points: Sequence[Point], *, dtype=np.float64) -> np.ndarray:
    array = np.asarray(points, dtype=dtype)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Expected an Nx2 sequence of points")
    return array


def close_polygon(points: Sequence[Point]) -> Polygon:
    result = [(float(x), float(y)) for x, y in points]
    if not result:
        return result
    if result[0] != result[-1]:
        result.append(result[0])
    return result


def polygon_area(points: Sequence[Point]) -> float:
    polygon = _as_array(points)
    if len(polygon) < 3:
        return 0.0
    if np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_bbox(points: Sequence[Point]) -> tuple[float, float, float, float]:
    polygon = _as_array(points)
    return (
        float(polygon[:, 0].min()),
        float(polygon[:, 1].min()),
        float(polygon[:, 0].max()),
        float(polygon[:, 1].max()),
    )


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting containment test that treats boundary points as contained."""

    x, y = float(point[0]), float(point[1])
    pts = _as_array(polygon)
    if len(pts) < 3:
        return False
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        edge = np.array([xj - xi, yj - yi])
        rel = np.array([x - xi, y - yi])
        cross = abs(float(edge[0] * rel[1] - edge[1] * rel[0]))
        if cross <= 1e-8 and min(xi, xj) - 1e-8 <= x <= max(xi, xj) + 1e-8 and min(
            yi, yj
        ) - 1e-8 <= y <= max(yi, yj) + 1e-8:
            return True
        if (yi > y) != (yj > y):
            intercept = (xj - xi) * (y - yi) / ((yj - yi) + _EPS) + xi
            if x < intercept:
                inside = not inside
        j = i
    return inside


def ray_polygon_intersection(
    origin: np.ndarray, direction: np.ndarray, polygon: Sequence[Point]
) -> float | None:
    """Returns the nearest non-negative distance along a normalized ray."""

    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= _EPS:
        return None
    direction = direction / norm
    pts = _as_array(polygon)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    best = math.inf
    for p1, p2 in zip(pts, np.roll(pts, -1, axis=0), strict=True):
        edge = p2 - p1
        denom = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(denom) <= 1e-10:
            continue
        diff = p1 - origin
        t = (diff[0] * edge[1] - diff[1] * edge[0]) / denom
        u = (diff[0] * direction[1] - diff[1] * direction[0]) / denom
        if t >= -1e-7 and -1e-7 <= u <= 1.0 + 1e-7:
            best = min(best, max(0.0, float(t)))
    return None if math.isinf(best) else best


def _ray_page_intersection(
    origin: np.ndarray, direction: np.ndarray, width: int, height: int
) -> float:
    x, y = origin
    dx, dy = direction
    candidates: list[float] = []
    if abs(dx) > _EPS:
        for boundary_x in (0.0, float(width - 1)):
            t = (boundary_x - x) / dx
            yy = y + t * dy
            if t >= 0 and 0 <= yy <= height - 1:
                candidates.append(float(t))
    if abs(dy) > _EPS:
        for boundary_y in (0.0, float(height - 1)):
            t = (boundary_y - y) / dy
            xx = x + t * dx
            if t >= 0 and 0 <= xx <= width - 1:
                candidates.append(float(t))
    return min(candidates) if candidates else 0.0


def polyline_cumulative_lengths(points: Sequence[Point]) -> np.ndarray:
    line = _as_array(points)
    if len(line) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(line) == 1:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    )




def polyline_point_at_fraction(points: Sequence[Point], fraction: float = 0.5) -> Point:
    line = _as_array(points)
    if len(line) == 0:
        raise ValueError("Polyline is empty")
    if len(line) == 1:
        return (float(line[0, 0]), float(line[0, 1]))
    cumulative = polyline_cumulative_lengths(points)
    target = float(np.clip(fraction, 0.0, 1.0)) * float(cumulative[-1])
    x = float(np.interp(target, cumulative, line[:, 0]))
    y = float(np.interp(target, cumulative, line[:, 1]))
    return (x, y)


def resample_polyline(
    points: Sequence[Point],
    *,
    spacing: float = 1.0,
    normalize_direction: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resamples a polyline by arc length and returns points, tangents, normals, distances."""

    line = _as_array(points)
    if len(line) < 2:
        raise ValueError("A baseline needs at least two points")
    cumulative = polyline_cumulative_lengths(points)
    total = float(cumulative[-1])
    if total < 1e-6:
        raise ValueError("Baseline has zero length")
    count = max(2, int(math.ceil(total / max(spacing, 0.25))) + 1)
    distances = np.linspace(0.0, total, count)
    xs = np.interp(distances, cumulative, line[:, 0])
    ys = np.interp(distances, cumulative, line[:, 1])
    samples = np.column_stack([xs, ys])
    tangents = np.gradient(samples, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norms[tangent_norms <= _EPS] = 1.0
    tangents = tangents / tangent_norms

    if normalize_direction:
        # Geometry-only callers are direction-insensitive and benefit from a
        # stable canonical orientation. Rectification disables this so the
        # output x-axis follows BLLA's start/end orientation (including RTL).
        mean = tangents.mean(axis=0)
        if abs(mean[0]) >= abs(mean[1]):
            if mean[0] < 0:
                samples = samples[::-1]
                tangents = -tangents[::-1]
                distances = total - distances[::-1]
        elif mean[1] < 0:
            samples = samples[::-1]
            tangents = -tangents[::-1]
            distances = total - distances[::-1]

    # For a left-to-right horizontal line this normal points downward.
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return samples, tangents, normals, distances


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= _EPS:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        delta = points - start
        distances = np.abs(vector[0] * delta[:, 1] - vector[1] * delta[:, 0]) / length
    index = int(np.argmax(distances))
    maximum = float(distances[index])
    if maximum > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def simplify_polyline(points: Sequence[Point], epsilon: float = 2.0) -> Polyline:
    simplified = _rdp(_as_array(points), max(0.0, float(epsilon)))
    return [(float(x), float(y)) for x, y in simplified]


def zhang_suen_thinning(binary: np.ndarray, *, max_iterations: int = 200) -> np.ndarray:
    """Dependency-free Zhang-Suen thinning for BLLA baseline heatmaps."""

    image = (np.asarray(binary) > 0).astype(np.uint8)
    if image.ndim != 2:
        raise ValueError("Thinning expects a two-dimensional image")
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0

    for _ in range(max_iterations):
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            common = (
                (image == 1)
                & (neighbours >= 2)
                & (neighbours <= 6)
                & (transitions == 1)
            )
            if phase == 0:
                remove = common & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                remove = common & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            if np.any(remove):
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image.astype(bool)


def _component_graph(component: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    coordinates = np.argwhere(component)
    lookup = {tuple(point): index for index, point in enumerate(coordinates.tolist())}
    return coordinates, lookup


def _dijkstra_farthest(
    coordinates: np.ndarray,
    lookup: dict[tuple[int, int], int],
    start: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    count = len(coordinates)
    distances = np.full(count, np.inf, dtype=np.float64)
    previous = np.full(count, -1, dtype=np.int64)
    distances[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        distance, index = heapq.heappop(queue)
        if distance != distances[index]:
            continue
        y, x = coordinates[index]
        for dy, dx in _NEIGHBOURS:
            neighbour = lookup.get((int(y + dy), int(x + dx)))
            if neighbour is None:
                continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            candidate = distance + step
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = index
                heapq.heappush(queue, (candidate, neighbour))
    finite = np.where(np.isfinite(distances))[0]
    farthest = int(finite[np.argmax(distances[finite])])
    return farthest, distances, previous


def skeleton_main_paths(
    skeleton: np.ndarray, *, min_length: float = 5.0, min_component_pixels: int = 5
) -> list[Polyline]:
    """Extracts the geodesic diameter of each connected skeleton component."""

    count, labels = cv2.connectedComponents((skeleton > 0).astype(np.uint8), connectivity=8)
    paths: list[Polyline] = []
    for label_index in range(1, count):
        component = labels == label_index
        if int(component.sum()) < min_component_pixels:
            continue
        coordinates, lookup = _component_graph(component)
        if len(coordinates) < 2:
            continue
        degrees = []
        for y, x in coordinates:
            degree = sum((int(y + dy), int(x + dx)) in lookup for dy, dx in _NEIGHBOURS)
            degrees.append(degree)
        endpoints = np.where(np.asarray(degrees) == 1)[0]
        start = int(endpoints[0]) if len(endpoints) else 0
        first, _, _ = _dijkstra_farthest(coordinates, lookup, start)
        second, distances, previous = _dijkstra_farthest(coordinates, lookup, first)
        if float(distances[second]) < min_length:
            continue
        indices = [second]
        cursor = second
        while cursor != first and previous[cursor] >= 0:
            cursor = int(previous[cursor])
            indices.append(cursor)
        indices.reverse()
        # np.argwhere is y,x; all public geometry is x,y.
        path = [(float(coordinates[index, 1]), float(coordinates[index, 0])) for index in indices]
        paths.append(path)
    return paths


def sample_map(
    gray: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    *,
    border_value: float = 255.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    map_x = points[None, :, 0] + offsets[:, None] * normals[None, :, 0]
    map_y = points[None, :, 1] + offsets[:, None] * normals[None, :, 1]
    sampled = cv2.remap(
        gray.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )
    return sampled, map_x, map_y


def _nearest_region_for_line(line: Sequence[Point], regions: Sequence[Region]) -> Region | None:
    samples, _, _, _ = resample_polyline(line, spacing=max(1.0, polyline_cumulative_lengths(line)[-1] / 2))
    midpoint = tuple(samples[len(samples) // 2])
    for region in regions:
        if point_in_polygon(midpoint, region.boundary):
            return region
    return None


def _corridor_limits(
    points: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    *,
    page_size: tuple[int, int],
    region: Region | None,
    other_baselines: Sequence[Sequence[Point]],
    maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = page_size
    top = np.full(len(points), float(maximum), dtype=np.float64)
    bottom = np.full(len(points), float(maximum), dtype=np.float64)
    polygon = region.boundary if region is not None else None

    for index, (point, normal) in enumerate(zip(points, normals, strict=True)):
        page_top = _ray_page_intersection(point, -normal, width, height)
        page_bottom = _ray_page_intersection(point, normal, width, height)
        top[index] = min(top[index], max(1.0, page_top))
        bottom[index] = min(bottom[index], max(1.0, page_bottom))
        if polygon is not None:
            region_top = ray_polygon_intersection(point, -normal, polygon)
            region_bottom = ray_polygon_intersection(point, normal, polygon)
            if region_top is not None and region_top > 0.5:
                top[index] = min(top[index], region_top)
            if region_bottom is not None and region_bottom > 0.5:
                bottom[index] = min(bottom[index], region_bottom)

    # Adjacent baselines constrain each side to slightly less than the midpoint.
    # The tangent gate prevents unrelated lines in distant columns from narrowing
    # the corridor. This used to allocate and scan one delta matrix per point;
    # vectorizing in bounded chunks is materially faster on ordinary laptop CPUs
    # while keeping memory bounded on very long lines.
    tangent_window = max(18.0, maximum * 0.5)
    point_chunk = 512
    for other in other_baselines:
        try:
            other_points, _, _, _ = resample_polyline(other, spacing=4.0)
        except ValueError:
            continue
        if not len(other_points):
            continue
        for start in range(0, len(points), point_chunk):
            stop = min(len(points), start + point_chunk)
            point_block = points[start:stop]
            tangent_block = tangents[start:stop]
            normal_block = normals[start:stop]
            delta_x = other_points[None, :, 0] - point_block[:, None, 0]
            delta_y = other_points[None, :, 1] - point_block[:, None, 1]
            along = (
                delta_x * tangent_block[:, None, 0]
                + delta_y * tangent_block[:, None, 1]
            )
            signed = (
                delta_x * normal_block[:, None, 0]
                + delta_y * normal_block[:, None, 1]
            )
            nearby = np.abs(along) <= tangent_window

            negative = np.where(
                (signed < -1.0) & nearby, -signed, np.inf
            ).min(axis=1)
            positive = np.where(
                (signed > 1.0) & nearby, signed, np.inf
            ).min(axis=1)

            top_block = top[start:stop]
            bottom_block = bottom[start:stop]
            valid_negative = np.isfinite(negative)
            valid_positive = np.isfinite(positive)
            top_block[valid_negative] = np.minimum(
                top_block[valid_negative],
                np.maximum(2.0, negative[valid_negative] * 0.46),
            )
            bottom_block[valid_positive] = np.minimum(
                bottom_block[valid_positive],
                np.maximum(2.0, positive[valid_positive] * 0.46),
            )

    return np.clip(top, 2.0, maximum), np.clip(bottom, 2.0, maximum)


def _trace_seam(cost: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Traces a minimum-cost seam with one-row transition smoothness.

    Columns remain sequential, but every row transition in a column is
    vectorized. The geometry result is identical to the scalar recurrence and
    avoids millions of Python-level row iterations on a multi-page document.
    """

    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        raise ValueError("Empty seam search")
    finite_cost = np.where(valid, cost, np.inf)
    dp = np.full_like(finite_cost, np.inf, dtype=np.float64)
    # int16 also safely records the rare corridor-collapse jump; int8 could
    # overflow when a wide vertical search band collapses.
    back = np.zeros((rows, columns), dtype=np.int16)
    dp[:, 0] = finite_cost[:, 0]
    if not np.isfinite(dp[:, 0]).any():
        dp[:, 0] = cost[:, 0]
    smoothness = 0.35
    deltas = np.asarray([-1, 0, 1], dtype=np.int16)
    infinity = np.asarray([np.inf], dtype=np.float64)

    for column in range(1, columns):
        previous = dp[:, column - 1]
        candidates = np.stack(
            [
                np.concatenate([infinity, previous[:-1] + smoothness]),
                previous,
                np.concatenate([previous[1:] + smoothness, infinity]),
            ],
            axis=0,
        )
        choice = np.argmin(candidates, axis=0)
        best = np.take_along_axis(
            candidates, choice[None, :], axis=0
        )[0]
        allowed = np.isfinite(finite_cost[:, column]) & np.isfinite(best)
        dp[allowed, column] = finite_cost[allowed, column] + best[allowed]
        back[allowed, column] = deltas[choice[allowed]]

        if not np.isfinite(dp[:, column]).any():
            # Local corridor collapse: bridge to the cheapest current row from
            # the cheapest reachable row in the prior column.
            row = int(np.argmin(cost[:, column]))
            previous_row = int(np.argmin(previous))
            dp[row, column] = cost[row, column] + previous[previous_row]
            back[row, column] = previous_row - row

    seam = np.zeros(columns, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[:, -1]))
    for column in range(columns - 1, 0, -1):
        seam[column - 1] = seam[column] + int(back[seam[column], column])
        seam[column - 1] = int(np.clip(seam[column - 1], 0, rows - 1))
    return seam


def fallback_boundary(
    baseline: Sequence[Point], *, top: float = 24.0, bottom: float = 7.0
) -> Polygon:
    points, _, normals, _ = resample_polyline(baseline, spacing=4.0)
    upper = points - normals * float(top)
    lower = points + normals * float(bottom)
    polygon = np.vstack([upper, lower[::-1]])
    return close_polygon(simplify_polyline(polygon.tolist(), epsilon=1.5))


def polygonize_baseline(
    gray: np.ndarray,
    baseline: Sequence[Point],
    *,
    other_baselines: Sequence[Sequence[Point]] = (),
    region: Region | None = None,
    maximum_distance: float = 72.0,
    topline: bool | None = False,
) -> Polygon:
    """Computes a whitespace seam polygon around a BLLA baseline.

    This is the dependency-light counterpart to Kraken's seam-carved polygonal
    environment. It keeps adjacent baselines and region/page boundaries as hard
    corridor constraints, traces low-ink seams, and emits source-page geometry.
    """

    height, width = gray.shape
    points, tangents, normals, _ = resample_polyline(baseline, spacing=2.0)
    top_limit, bottom_limit = _corridor_limits(
        points,
        tangents,
        normals,
        page_size=(width, height),
        region=region,
        other_baselines=other_baselines,
        maximum=maximum_distance,
    )

    max_top = max(4, int(math.ceil(float(np.percentile(top_limit, 90)))))
    max_bottom = max(4, int(math.ceil(float(np.percentile(bottom_limit, 90)))))
    offsets = np.arange(-max_top, max_bottom + 1, dtype=np.float64)
    strip, map_x, map_y = sample_map(gray, points, normals, offsets)
    ink = 1.0 - np.clip(strip / 255.0, 0.0, 1.0)
    ink = cv2.GaussianBlur(ink.astype(np.float32), (3, 3), 0.8)
    gradient = np.abs(cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)) / 1020.0
    profile = np.mean(ink, axis=1)
    baseline_row = max_top

    threshold = max(0.015, float(np.percentile(profile, 75)) * 0.25)
    active = np.where(profile > threshold)[0]
    if active.size:
        top_ink = int(active.min())
        bottom_ink = int(active.max())
    else:
        top_ink = max(0, baseline_row - 16)
        bottom_ink = min(len(offsets) - 1, baseline_row + 3)

    if topline is True:
        # A topline has most ink below it.
        top_preferred = max(1, top_ink - 2)
        bottom_preferred = min(len(offsets) - 2, bottom_ink + 3)
    elif topline is None:
        top_preferred = max(1, top_ink - 2)
        bottom_preferred = min(len(offsets) - 2, bottom_ink + 2)
    else:
        # Stock BLLA is a conventional bottom-baseline model.
        top_preferred = max(1, top_ink - 3)
        bottom_preferred = min(len(offsets) - 2, max(baseline_row + 2, bottom_ink + 2))

    rows = np.arange(len(offsets), dtype=np.float64)[:, None]
    base_cost = 3.0 * ink + 0.8 * gradient

    top_cost = base_cost + 0.035 * np.square(rows - top_preferred)
    bottom_cost = base_cost + 0.035 * np.square(rows - bottom_preferred)
    row_indices = np.arange(len(offsets), dtype=np.int32)[:, None]
    top_band = 12
    bottom_band = 10

    top_minimum = np.searchsorted(offsets, -top_limit, side="left")
    top_minimum = np.maximum(top_minimum, top_preferred - top_band)
    top_maximum = np.full(
        len(points),
        min(baseline_row - 1, top_preferred + top_band),
        dtype=np.int32,
    )
    top_valid = (
        (row_indices >= top_minimum[None, :])
        & (row_indices <= top_maximum[None, :])
        & (top_minimum[None, :] <= top_maximum[None, :])
    )

    bottom_minimum = np.full(
        len(points),
        max(baseline_row + 1, bottom_preferred - bottom_band),
        dtype=np.int32,
    )
    bottom_maximum = np.searchsorted(
        offsets, bottom_limit, side="right"
    ) - 1
    bottom_maximum = np.minimum(
        bottom_maximum, bottom_preferred + bottom_band
    )
    bottom_valid = (
        (row_indices >= bottom_minimum[None, :])
        & (row_indices <= bottom_maximum[None, :])
        & (bottom_minimum[None, :] <= bottom_maximum[None, :])
    )

    try:
        top_rows = _trace_seam(top_cost, top_valid)
        bottom_rows = _trace_seam(bottom_cost, bottom_valid)
    except ValueError:
        return fallback_boundary(baseline)

    upper = np.column_stack(
        [map_x[top_rows, np.arange(len(points))], map_y[top_rows, np.arange(len(points))]]
    )
    lower = np.column_stack(
        [
            map_x[bottom_rows, np.arange(len(points))],
            map_y[bottom_rows, np.arange(len(points))],
        ]
    )
    polygon = np.vstack([upper, lower[::-1]])
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    simplified = close_polygon(simplify_polyline(polygon.tolist(), epsilon=2.0))
    midpoint = tuple(points[len(points) // 2])
    if polygon_area(simplified) < 10.0 or not point_in_polygon(midpoint, simplified):
        return fallback_boundary(baseline)
    return simplified


@dataclass(slots=True)
class Rectification:
    image: Image.Image
    source_x: np.ndarray
    source_y: np.ndarray
    baseline_row: int
    source_length: float

    @property
    def width(self) -> int:
        return int(self.source_x.shape[1])

    @property
    def height(self) -> int:
        return int(self.source_x.shape[0])

    def project_interval(self, start_x: float, end_x: float) -> Polygon:
        left = int(np.clip(math.floor(start_x), 0, self.width - 1))
        right = int(np.clip(math.ceil(end_x), left, self.width - 1))
        return [
            (float(self.source_x[0, left]), float(self.source_y[0, left])),
            (float(self.source_x[0, right]), float(self.source_y[0, right])),
            (float(self.source_x[-1, right]), float(self.source_y[-1, right])),
            (float(self.source_x[-1, left]), float(self.source_y[-1, left])),
        ]


def rectify_line(
    image: Image.Image,
    baseline: Sequence[Point],
    boundary: Sequence[Point],
    *,
    minimum_height: int = 12,
) -> Rectification:
    """Straightens a curved BLLA line while retaining a dense page-coordinate map."""

    if image.mode == "1":
        source_image = image.convert("L")
    elif image.mode in {"L", "RGB"}:
        source_image = image
    else:
        source_image = image.convert("RGB")
    source = np.asarray(source_image, dtype=np.float32)
    points, tangents, normals, distances = resample_polyline(
        baseline, spacing=1.0, normalize_direction=False
    )
    # Preserve BLLA's along-line orientation while keeping the rectified
    # image's cross-line axis visually upright: down for horizontal lines and
    # right for vertical lines. This is essential for RTL character geometry.
    mean_tangent = tangents.mean(axis=0)
    if abs(mean_tangent[0]) >= abs(mean_tangent[1]):
        if float(normals[:, 1].mean()) < 0:
            normals = -normals
    elif float(normals[:, 0].mean()) < 0:
        normals = -normals
    top = np.zeros(len(points), dtype=np.float64)
    bottom = np.zeros(len(points), dtype=np.float64)
    for index, (point, normal) in enumerate(zip(points, normals, strict=True)):
        top_value = ray_polygon_intersection(point, -normal, boundary)
        bottom_value = ray_polygon_intersection(point, normal, boundary)
        top[index] = max(1.0, float(top_value) if top_value is not None else 8.0)
        bottom[index] = max(1.0, float(bottom_value) if bottom_value is not None else 4.0)

    target_top = max(2, int(math.ceil(float(np.percentile(top, 85)))))
    target_bottom = max(2, int(math.ceil(float(np.percentile(bottom, 85)))))
    output_height = max(minimum_height, target_top + target_bottom + 1)
    target_top = min(target_top, output_height - 2)
    target_bottom = output_height - target_top - 1

    source_x = np.empty((output_height, len(points)), dtype=np.float32)
    source_y = np.empty_like(source_x)
    for row in range(output_height):
        if row <= target_top:
            fraction = (target_top - row) / max(1, target_top)
            offsets = -fraction * top
        else:
            fraction = (row - target_top) / max(1, target_bottom)
            offsets = fraction * bottom
        source_x[row] = points[:, 0] + offsets * normals[:, 0]
        source_y[row] = points[:, 1] + offsets * normals[:, 1]

    border_value: float | tuple[float, ...]
    if source.ndim == 2:
        border_value = 255.0
    else:
        border_value = tuple(255.0 for _ in range(source.shape[2]))
    rectified = cv2.remap(
        source,
        source_x,
        source_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return Rectification(
        image=Image.fromarray(np.clip(rectified, 0, 255).astype(np.uint8)),
        source_x=source_x,
        source_y=source_y,
        baseline_row=target_top,
        source_length=float(distances[-1]),
    )


def _bbox_for_line(line: BaselineLine) -> tuple[float, float, float, float]:
    return polygon_bbox(line.boundary if line.boundary else line.baseline)


def _partial_order(
    boxes: Sequence[tuple[float, float, float, float]], text_direction: str
) -> np.ndarray:
    count = len(boxes)
    order = np.zeros((count, count), dtype=bool)

    def overlaps_x(first, second) -> bool:
        return first[0] < second[2] and first[2] > second[0]

    def above(first, second) -> bool:
        return first[1] < second[1]

    def left_of(first, second) -> bool:
        return first[2] < second[0]

    def separates(candidate, first, second) -> bool:
        if candidate is first or candidate is second:
            return False
        if candidate[3] < min(first[1], second[1]):
            return False
        if candidate[1] > max(first[3], second[3]):
            return False
        return candidate[0] < first[2] and candidate[2] > second[0]

    for i, first in enumerate(boxes):
        for j, second in enumerate(boxes):
            if i == j:
                continue
            if overlaps_x(first, second):
                if above(first, second):
                    order[i, j] = True
            elif not any(separates(candidate, first, second) for candidate in boxes):
                if text_direction == "rl":
                    if not left_of(first, second):
                        order[i, j] = True
                elif left_of(first, second):
                    order[i, j] = True
    return order


def _topological_order(
    relation: np.ndarray, boxes: Sequence[tuple[float, float, float, float]]
) -> list[int]:
    count = len(boxes)
    indegree = relation.sum(axis=0).astype(np.int64)
    remaining = set(range(count))
    result: list[int] = []
    while remaining:
        available = [index for index in remaining if indegree[index] == 0]
        if not available:
            # A contradictory heuristic edge must not make ordering nondeterministic.
            available = list(remaining)
        chosen = min(available, key=lambda index: (boxes[index][1], boxes[index][0], index))
        result.append(chosen)
        remaining.remove(chosen)
        for target in np.where(relation[chosen])[0]:
            indegree[target] = max(0, indegree[target] - 1)
    return result


def reading_order(
    lines: Sequence[BaselineLine],
    regions: Sequence[Region] = (),
    *,
    text_direction: str = "lr",
) -> list[int]:
    """Region-aware implementation of Kraken's polygonal reading-order heuristic."""

    if not lines:
        return []
    line_region: dict[int, int | None] = {}
    for line_index, line in enumerate(lines):
        midpoint = polyline_point_at_fraction(line.baseline, 0.5)
        line_region[line_index] = next(
            (
                region_index
                for region_index, region in enumerate(regions)
                if point_in_polygon(midpoint, region.boundary)
            ),
            None,
        )

    item_boxes: list[tuple[float, float, float, float]] = []
    item_payload: list[tuple[str, int]] = []
    for region_index, region in enumerate(regions):
        contained = [idx for idx, value in line_region.items() if value == region_index]
        if contained:
            item_boxes.append(polygon_bbox(region.boundary))
            item_payload.append(("region", region_index))
    for line_index, region_index in line_region.items():
        if region_index is None:
            item_boxes.append(_bbox_for_line(lines[line_index]))
            item_payload.append(("line", line_index))

    global_order = _topological_order(_partial_order(item_boxes, text_direction), item_boxes)
    result: list[int] = []
    for item_index in global_order:
        kind, source_index = item_payload[item_index]
        if kind == "line":
            result.append(source_index)
            continue
        contained = [idx for idx, value in line_region.items() if value == source_index]
        boxes = [_bbox_for_line(lines[idx]) for idx in contained]
        local = _topological_order(_partial_order(boxes, text_direction), boxes)
        result.extend(contained[index] for index in local)

    # Region polygons may overlap. Ensure every line appears exactly once.
    result.extend(index for index in range(len(lines)) if index not in result)
    return result
