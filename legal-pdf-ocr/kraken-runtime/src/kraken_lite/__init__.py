"""Dependency-light inference runtime for Kraken-trained OCR models."""

from .blla import BLLASegmenter
from .engine import KrakenLiteEngine
from .model_pack import ModelPack
from .recognition import Recognizer
from .types import BaselineLine, Character, OCRLine, OCRPage, Region, Segmentation

__all__ = [
    "BLLASegmenter",
    "BaselineLine",
    "Character",
    "KrakenLiteEngine",
    "ModelPack",
    "OCRLine",
    "OCRPage",
    "Recognizer",
    "Region",
    "Segmentation",
]

__version__ = "0.1.0"
