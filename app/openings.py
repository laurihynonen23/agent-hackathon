from __future__ import annotations

import numpy as np
from PIL import Image

from .classify import classify_pages
from .types import ClassificationResult, DocumentPage, FacadeModel, Opening


def _page_for_role(
    pages: list[DocumentPage],
    classifications: list[ClassificationResult],
    role: str,
) -> DocumentPage | None:
    best = None
    best_score = -1.0
    pages_by_id = {page.page_id: page for page in pages}
    for classification in classifications:
        if classification.role != role:
            continue
        if classification.score > best_score:
            best = pages_by_id.get(classification.page_id)
            best_score = classification.score
    return best


def _dedupe_boxes(candidates: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    deduped: list[tuple[float, float, float, float]] = []
    for candidate in sorted(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True):
        x0, y0, x1, y1 = candidate
        keep = True
        for existing in deduped:
            ex0, ey0, ex1, ey1 = existing
            inter_x0 = max(x0, ex0)
            inter_y0 = max(y0, ey0)
            inter_x1 = min(x1, ex1)
            inter_y1 = min(y1, ey1)
            if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                continue
            inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
            area_candidate = (x1 - x0) * (y1 - y0)
            area_existing = (ex1 - ex0) * (ey1 - ey0)
            union = area_candidate + area_existing - inter
            if union > 0 and inter / union > 0.65:
                keep = False
                break
        if keep:
            deduped.append(candidate)
    return deduped


def _raster_boxes_for_facade(page: DocumentPage, facade: FacadeModel) -> list[tuple[float, float, float, float]]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return []

    image = cv2.imread(str(page.render_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []

    scale_x = image.shape[1] / page.width_pt
    scale_y = image.shape[0] / page.height_pt
    x0 = max(0, int(facade.cluster_box.x0 * scale_x))
    y0 = max(0, int(facade.cluster_box.y0 * scale_y))
    x1 = min(image.shape[1], int(facade.cluster_box.x1 * scale_x))
    y1 = min(image.shape[0], int(facade.cluster_box.y1 * scale_y))
    if x1 <= x0 or y1 <= y0:
        return []

    crop = image[y0:y1, x0:x1]
    _, thresh = cv2.threshold(crop, 235, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[float, float, float, float]] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 15 or bh < 15:
            continue
        if bw > crop.shape[1] * 0.72 or bh > crop.shape[0] * 0.82:
            continue
        boxes.append(
            (
                (x0 + bx) / scale_x,
                (y0 + by) / scale_y,
                (x0 + bx + bw) / scale_x,
                (y0 + by + bh) / scale_y,
            )
        )
    return boxes


def extract_openings(
    pages: list[DocumentPage],
    classifications: list[ClassificationResult],
    facades: list[FacadeModel],
) -> tuple[list[Opening], list[str]]:
    warnings: list[str] = []
    if not classifications:
        classifications = classify_pages(pages)
    elevation_page = _page_for_role(pages, classifications, "elevations")
    if elevation_page is None:
        return [], ["No elevation page available for opening detection."]

    openings: list[Opening] = []
    for facade in facades:
        if facade.cluster_box.area <= 0 or facade.width_m <= 0:
            continue
        px_per_m = facade.cluster_box.width / facade.width_m
        if px_per_m <= 0:
            continue

        candidates = []
        for primitive in elevation_page.vector_primitives:
            box = primitive.bbox
            if not box.intersects(facade.cluster_box):
                continue
            if box.area < 60.0:
                continue
            if box.width < 8.0 or box.height < 8.0:
                continue
            if box.width > facade.cluster_box.width * 0.72:
                continue
            if box.height > facade.cluster_box.height * 0.82:
                continue
            area_ratio = box.area / max(facade.cluster_box.area, 1.0)
            if area_ratio < 0.002 or area_ratio > 0.22:
                continue
            if box.cy < facade.cluster_box.y0 + 5.0:
                continue
            candidates.append((box.x0, box.y0, box.x1, box.y1))

        if not candidates:
            candidates.extend(_raster_boxes_for_facade(elevation_page, facade))

        deduped_boxes = _dedupe_boxes(candidates)
        for box in deduped_boxes[:20]:
            x0, y0, x1, y1 = box
            width_m = (x1 - x0) / px_per_m
            height_m = (y1 - y0) / px_per_m
            area_m2 = width_m * height_m
            if width_m < 0.2 or height_m < 0.2:
                continue
            if width_m > 4.2 or height_m > 3.5:
                continue
            if area_m2 < 0.15 or area_m2 > 8.5:
                continue
            bottom_gap = abs(facade.cluster_box.y1 - y1)
            opening_type = "door" if bottom_gap < facade.cluster_box.height * 0.08 and height_m > 1.7 else "window"
            confidence = 0.62 if opening_type == "window" else 0.68
            openings.append(
                Opening(
                    facade_name=facade.name,
                    opening_type=opening_type,
                    bbox_page=facade.cluster_box.__class__(x0=x0, y0=y0, x1=x1, y1=y1),
                    width_m=width_m,
                    height_m=height_m,
                    area_m2=area_m2,
                    source_page_id=elevation_page.page_id,
                    detection_mode="vector_bbox",
                    confidence=confidence,
                )
            )

    if not openings:
        warnings.append("No openings were confidently detected from elevations; opening subtraction may be understated.")

    return openings, warnings
