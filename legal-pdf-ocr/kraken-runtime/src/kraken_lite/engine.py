from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from .blla import BLLASegmenter
from .geometry import Rectification, rectify_line
from .recognition import RecognitionResult, Recognizer
from .types import OCRLine, OCRPage, Segmentation


class KrakenLiteEngine:
    """End-to-end stock-BLLA geometry plus compiled Kraken recognition.

    Neural work is scheduled in two independent queues:

    * page-shape buckets for the stock BLLA graph; and
    * line-width buckets for the recognition graph.

    The geometry stages remain page-local and deterministic. This lets a CUDA
    provider process pages and lines from an entire document in useful batches
    without changing baseline, polygon, rectification, or character-coordinate
    semantics.
    """

    def __init__(
        self,
        segmenter: BLLASegmenter,
        recognizer: Recognizer | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.recognizer = recognizer

    def segment(
        self,
        image: Image.Image,
        *,
        text_direction: str = "horizontal-lr",
        raise_on_error: bool = False,
        batch_size: int | None = None,
    ) -> Segmentation:
        return self.segmenter.segment(
            image,
            text_direction=text_direction,
            raise_on_error=raise_on_error,
            batch_size=batch_size,
        )

    def segment_many(
        self,
        images: Sequence[Image.Image],
        *,
        text_direction: str = "horizontal-lr",
        raise_on_error: bool = False,
        batch_size: int | None = None,
    ) -> list[Segmentation]:
        """Segments pages in shape-compatible BLLA batches.

        Older model packs that do not declare a dynamic batch dimension are
        automatically kept at batch one by :class:`BLLASegmenter`.
        """

        return self.segmenter.segment_many(
            images,
            text_direction=text_direction,
            raise_on_error=raise_on_error,
            batch_size=batch_size,
        )

    def recognize(
        self,
        image: Image.Image,
        *,
        text_direction: str = "horizontal-lr",
        raise_on_error: bool = False,
        batch_size: int | None = None,
        page_batch_size: int | None = None,
    ) -> OCRPage:
        return self.recognize_many(
            [image],
            text_direction=text_direction,
            raise_on_error=raise_on_error,
            batch_size=batch_size,
            page_batch_size=1 if page_batch_size is None else page_batch_size,
        )[0]

    def recognize_many(
        self,
        images: Sequence[Image.Image],
        *,
        text_direction: str = "horizontal-lr",
        raise_on_error: bool = False,
        batch_size: int | None = None,
        page_batch_size: int | None = None,
    ) -> list[OCRPage]:
        """Runs a document through batched BLLA and recognition inference.

        Rectified lines from every page are flattened into one recognition
        scheduling queue. ``Recognizer`` then width-buckets them across page
        boundaries, which is essential for keeping CUDA-class hardware busy on
        typical documents with many relatively small text lines. Results are
        restored to their original page and line positions before serialization.

        A failed recognition batch is retried line-by-line unless
        ``raise_on_error`` is set. Geometry is preserved even when recognition
        fails for an individual line.
        """

        if self.recognizer is None:
            raise RuntimeError("No recognition model is configured")
        if not images:
            return []

        pages = list(images)
        segmentations = self.segment_many(
            pages,
            text_direction=text_direction,
            raise_on_error=raise_on_error,
            batch_size=page_batch_size,
        )
        if len(segmentations) != len(pages):  # pragma: no cover - invariant
            raise RuntimeError("Segmenter did not return one result per page")

        return self.recognize_segmentations(
            pages, segmentations, raise_on_error=raise_on_error, batch_size=batch_size
        )

    def recognize_segmentations(
        self,
        images: Sequence[Image.Image],
        segmentations: Sequence[Segmentation],
        *,
        raise_on_error: bool = False,
        batch_size: int | None = None,
    ) -> list[OCRPage]:
        """Recognize declared lines on full original pages without segmentation."""
        if self.recognizer is None:
            raise RuntimeError("No recognition model is configured")
        pages = list(images)
        if len(pages) != len(segmentations):
            raise ValueError("One supplied segmentation is required per image")
        for image, segmentation in zip(pages, segmentations, strict=True):
            if image.size != (segmentation.width, segmentation.height):
                raise ValueError("Supplied segmentation dimensions differ from image")

        # Keep the page/line address beside every rectification so batched
        # recognition can freely reorder by width and still restore the exact
        # document structure afterwards.
        addresses: list[tuple[int, int]] = []
        rectifications: list[Rectification] = []
        failures: set[tuple[int, int]] = set()
        for page_index, (image, segmentation) in enumerate(
            zip(pages, segmentations, strict=True)
        ):
            for line_index, line in enumerate(segmentation.lines):
                address = (page_index, line_index)
                try:
                    rectification = rectify_line(
                        image,
                        line.baseline,
                        line.boundary,
                    )
                except Exception:
                    if raise_on_error:
                        raise
                    failures.add(address)
                    continue
                addresses.append(address)
                rectifications.append(rectification)

        recognized: dict[tuple[int, int], RecognitionResult] = {}
        if rectifications:
            try:
                results = self.recognizer.recognize_many(
                    rectifications,
                    batch_size=batch_size,
                )
                recognized.update(zip(addresses, results, strict=True))
            except Exception:
                if raise_on_error:
                    raise
                # Preserve geometry even if a provider-specific batch fails.
                # Retry individual lines so one malformed input or transient
                # accelerator failure does not erase the rest of the document.
                for address, rectification in zip(
                    addresses, rectifications, strict=True
                ):
                    try:
                        recognized[address] = self.recognizer.recognize(
                            rectification
                        )
                    except Exception:
                        failures.add(address)

        recognition_execution = getattr(
            self.recognizer.session, "execution_info", None
        )
        recognition_batch_size = (
            int(batch_size)
            if batch_size is not None and batch_size > 0
            else self.recognizer.default_batch_size
        )

        output_pages: list[OCRPage] = []
        for page_index, segmentation in enumerate(segmentations):
            output_lines: list[OCRLine] = []
            for line_index, line in enumerate(segmentation.lines):
                result = recognized.get((page_index, line_index))
                if result is None:
                    output_lines.append(
                        OCRLine(
                            id=line.id,
                            text="",
                            confidence=0.0,
                            baseline=line.baseline,
                            boundary=line.boundary,
                            characters=[],
                            type=line.type,
                            regions=line.regions,
                        )
                    )
                    continue
                output_lines.append(
                    OCRLine(
                        id=line.id,
                        text=result.text,
                        confidence=result.confidence,
                        baseline=line.baseline,
                        boundary=line.boundary,
                        characters=result.characters,
                        type=line.type,
                        regions=line.regions,
                    )
                )

            output_pages.append(
                OCRPage(
                    width=segmentation.width,
                    height=segmentation.height,
                    lines=output_lines,
                    regions=segmentation.regions,
                    text_direction=segmentation.text_direction,
                    models={
                        "segmentation": segmentation.model,
                        "recognition": {
                            "id": self.recognizer.manifest.get(
                                "id", "recognizer"
                            ),
                            "source": self.recognizer.manifest.get(
                                "source", {}
                            ),
                            "execution": (
                                recognition_execution.to_dict()
                                if recognition_execution is not None
                                else None
                            ),
                            "batchSize": recognition_batch_size,
                            "documentQueue": True,
                        },
                    },
                )
            )
        return output_pages
