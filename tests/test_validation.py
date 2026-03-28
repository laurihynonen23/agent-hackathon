from app.types import BBox, CladdingRegion, ConfidenceScores, FacadeModel, FootprintModel, MeasurementStore, Opening, Point2D
from app.validate import validate_takeoff


def test_validate_takeoff_confidence_bounds():
    footprint = FootprintModel(
        width_m=9.0,
        depth_m=12.0,
        polygon=[Point2D(x=0, y=0), Point2D(x=9, y=0), Point2D(x=9, y=12), Point2D(x=0, y=12)],
        perimeter_m=42.0,
        source_page_id="plan",
        source_strategy="test",
        confidence=0.8,
        plan_bbox=BBox(x0=0, y0=0, x1=1, y1=1),
    )
    facades = [
        FacadeModel(
            name="A",
            width_m=12,
            total_height_m=4,
            eave_height_m=4,
            ridge_height_m=None,
            area_gross_m2=48,
            cluster_box=BBox(x0=0, y0=0, x1=10, y1=10),
            source_page_id="elev",
            confidence=0.8,
            is_gable=False,
            assigned_dimension_kind="long",
        )
    ]
    openings = [
        Opening(
            facade_name="A",
            bbox_page=BBox(x0=1, y0=1, x1=2, y1=2),
            width_m=1.0,
            height_m=1.2,
            area_m2=1.2,
            source_page_id="elev",
            detection_mode="vector_bbox",
            confidence=0.7,
        )
    ]
    regions = [
        CladdingRegion(
            facade_name="A",
            material_code="3a",
            area_m2=46.8,
            source_page_id="elev",
            method="single_label_assignment",
            confidence=0.8,
            bbox_page=BBox(x0=0, y0=0, x1=10, y1=10),
        )
    ]
    summary = validate_takeoff(footprint, facades, openings, regions, MeasurementStore(measurements=[]), classifications=[])
    assert 0.0 <= summary.confidence.overall <= 1.0
    assert 0.0 <= summary.confidence.geometry <= 1.0

