from app.ui import STAGE_ORDER, build_index_html


def test_ui_html_exposes_transparent_process_copy():
    html = build_index_html()
    assert "Local PDF takeoff with the process fully exposed." in html
    assert "Process Ledger" in html
    assert "No black box" in html
    assert "AI resolver" in html


def test_ui_stage_order_matches_pipeline_expectations():
    assert STAGE_ORDER == ["ingest", "classify", "extract", "geometry", "openings", "materials", "validate", "report"]
