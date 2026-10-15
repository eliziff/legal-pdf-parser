from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

Point: TypeAlias = tuple[float, float]
Polygon: TypeAlias = list[Point]
Polyline: TypeAlias = list[Point]


def _point_list(points: list[Point] | tuple[Point, ...]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


@dataclass(slots=True)
class Region:
    id: str
    boundary: Polygon
    type: str = "default"
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "boundary": _point_list(self.boundary),
            "tags": self.tags,
        }


@dataclass(slots=True)
class BaselineLine:
    id: str
    baseline: Polyline
    boundary: Polygon
    type: str = "default"
    regions: list[str] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "baseline": _point_list(self.baseline),
            "boundary": _point_list(self.boundary),
            "regions": list(self.regions),
            "tags": self.tags,
        }


@dataclass(slots=True)
class Segmentation:
    width: int
    height: int
    lines: list[BaselineLine]
    regions: dict[str, list[Region]] = field(default_factory=dict)
    text_direction: Literal[
        "horizontal-lr", "horizontal-rl", "vertical-lr", "vertical-rl"
    ] = "horizontal-lr"
    script_detection: bool = False
    line_orders: list[list[int]] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "type": "baselines",
            "textDirection": self.text_direction,
            "scriptDetection": self.script_detection,
            "lines": [line.to_dict() for line in self.lines],
            "regions": {
                kind: [region.to_dict() for region in values]
                for kind, values in self.regions.items()
            },
            "lineOrders": [list(order) for order in self.line_orders],
            "model": self.model,
        }


@dataclass(slots=True)
class Character:
    text: str
    confidence: float
    start: int
    end: int
    polygon: Polygon

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": float(self.confidence),
            "start": int(self.start),
            "end": int(self.end),
            "polygon": _point_list(self.polygon),
        }


@dataclass(slots=True)
class OCRLine:
    id: str
    text: str
    confidence: float
    baseline: Polyline
    boundary: Polygon
    characters: list[Character]
    type: str = "default"
    regions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "confidence": float(self.confidence),
            "type": self.type,
            "baseline": _point_list(self.baseline),
            "boundary": _point_list(self.boundary),
            "regions": list(self.regions),
            "characters": [char.to_dict() for char in self.characters],
        }


@dataclass(slots=True)
class OCRPage:
    width: int
    height: int
    lines: list[OCRLine]
    regions: dict[str, list[Region]] = field(default_factory=dict)
    text_direction: str = "horizontal-lr"
    models: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "textDirection": self.text_direction,
            "text": self.text,
            "lines": [line.to_dict() for line in self.lines],
            "regions": {
                kind: [region.to_dict() for region in values]
                for kind, values in self.regions.items()
            },
            "models": self.models,
        }
