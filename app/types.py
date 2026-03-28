from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


SheetRole = Literal["plan", "section", "elevations", "unknown"]


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return self.x0 + self.width / 2.0

    @property
    def cy(self) -> float:
        return self.y0 + self.height / 2.0

    def expand(self, padding: float) -> "BBox":
        return BBox(
            x0=self.x0 - padding,
            y0=self.y0 - padding,
            x1=self.x1 + padding,
            y1=self.y1 + padding,
        )

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x1 < other.x0
            or self.x0 > other.x1
            or self.y1 < other.y0
            or self.y0 > other.y1
        )

    def intersection_area(self, other: "BBox") -> float:
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0)

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union

    @classmethod
    def from_points(cls, points: list[tuple[float, float]]) -> "BBox":
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return cls(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    @classmethod
    def union(cls, boxes: list["BBox"]) -> "BBox | None":
        if not boxes:
            return None
        return cls(
            x0=min(box.x0 for box in boxes),
            y0=min(box.y0 for box in boxes),
            x1=max(box.x1 for box in boxes),
            y1=max(box.y1 for box in boxes),
        )


class TextSpan(BaseModel):
    text: str
    bbox: BBox
    block_no: int = 0
    line_no: int = 0
    word_no: int = 0


class VectorPrimitive(BaseModel):
    primitive_type: str
    bbox: BBox
    stroke_width: float = 0.0
    fill: bool = False
    item_count: int = 0


class DocumentPage(BaseModel):
    page_id: str
    file_name: str
    file_path: Path
    page_number: int
    width_pt: float
    height_pt: float
    render_path: Path
    raw_text: str
    text_spans: list[TextSpan] = Field(default_factory=list)
    vector_primitives: list[VectorPrimitive] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    page_id: str
    file_name: str
    role: SheetRole
    score: float
    reasons: list[str] = Field(default_factory=list)
    subview_boxes: dict[str, BBox] = Field(default_factory=dict)


class Measurement(BaseModel):
    kind: str
    value_m: float
    raw_text: str
    source_page_id: str
    bbox: BBox
    confidence: float
    orientation: Literal["horizontal", "vertical", "unknown"] = "unknown"
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class MeasurementStore(BaseModel):
    measurements: list[Measurement] = Field(default_factory=list)


class Point2D(BaseModel):
    x: float
    y: float


class FootprintModel(BaseModel):
    width_m: float
    depth_m: float
    polygon: list[Point2D]
    perimeter_m: float
    source_page_id: str
    source_strategy: str
    confidence: float
    plan_bbox: BBox | None = None


class FacadeModel(BaseModel):
    name: str
    width_m: float
    total_height_m: float
    eave_height_m: float | None = None
    ridge_height_m: float | None = None
    area_gross_m2: float
    cluster_box: BBox
    source_page_id: str
    confidence: float
    is_gable: bool = False
    assigned_dimension_kind: str = ""


class Opening(BaseModel):
    facade_name: str
    opening_type: Literal["window", "door", "unknown"] = "unknown"
    bbox_page: BBox
    width_m: float
    height_m: float
    area_m2: float
    source_page_id: str
    detection_mode: str
    confidence: float


class CladdingRegion(BaseModel):
    facade_name: str
    material_code: str
    area_m2: float
    source_page_id: str
    method: str
    confidence: float
    bbox_page: BBox | None = None


class MaterialSpec(BaseModel):
    code: str
    description: str
    nominal_cover_mm: int | None = None
    source_page_id: str
    source_text: str


class MaterialQuantity(BaseModel):
    area_m2: float
    linear_m_nominal_cover: float
    assumed_nominal_cover_mm: int


class ConfidenceScores(BaseModel):
    overall: float
    geometry: float
    openings: float
    materials: float


class AiSettings(BaseModel):
    mode: Literal["off", "auto", "require"] = "auto"
    model: str | None = None
    base_url: str | None = None


class AiDecision(BaseModel):
    decision_type: str
    used: bool = False
    provider: str | None = None
    model: str | None = None
    selected: Any = None
    confidence: float | None = None
    rationale: str = ""
    fallback_used: bool = False
    fallback_reason: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class ValidationSummary(BaseModel):
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    cross_checks: dict[str, float] = Field(default_factory=dict)
    confidence: ConfidenceScores


class Results(BaseModel):
    perimeter_exterior_m: float
    gross_outer_wall_area_m2: float
    openings_area_m2: float
    net_cladding_area_m2: float
    cladding_by_type: dict[str, MaterialQuantity]
    assumptions: list[str]
    warnings: list[str]
    confidence: ConfidenceScores


class PipelineArtifacts(BaseModel):
    pages: list[DocumentPage]
    classifications: list[ClassificationResult]
    measurements: MeasurementStore
    footprint: FootprintModel
    facades: list[FacadeModel]
    openings: list[Opening]
    cladding_regions: list[CladdingRegion]
    material_specs: dict[str, MaterialSpec]
    validation: ValidationSummary
    results: Results
    ai_decisions: list[AiDecision] = Field(default_factory=list)
