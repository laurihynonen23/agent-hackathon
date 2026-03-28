import json
from pathlib import Path

import pytest

from app.planner import run_pipeline
from app.run import main


@pytest.mark.integration
def test_cli_smoke(tmp_path: Path):
    pytest.importorskip("fitz")
    input_dir = Path("tests/fixtures/sample_drawings")
    output_dir = tmp_path / "out"
    exit_code = main(["--input", str(input_dir), "--output", str(output_dir)])
    assert exit_code == 0

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert results["perimeter_exterior_m"] == pytest.approx(59.43, abs=0.01)
    assert results["gross_outer_wall_area_m2"] == pytest.approx(238.53, abs=0.15)
    assert "confidence" in results
    assert list(results["cladding_by_type"]) == ["3a"]
    assert results["cladding_by_type"]["3a"]["linear_m_nominal_cover"] == pytest.approx(1509.0, abs=2.0)

    debug_dir = output_dir / "debug"
    overlays_dir = output_dir / "overlays"
    assert (debug_dir / "classification.json").exists()
    assert (debug_dir / "measurements.json").exists()
    assert (debug_dir / "geometry.json").exists()
    assert (overlays_dir / "plan_overlay.png").exists()
    assert (output_dir / "report.md").exists()


@pytest.mark.integration
def test_pipeline_emits_progress_events(tmp_path: Path):
    pytest.importorskip("fitz")
    input_dir = Path("tests/fixtures/sample_drawings")
    output_dir = tmp_path / "out_progress"
    events = []
    artifacts = run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        progress_callback=events.append,
    )
    assert artifacts.results.perimeter_exterior_m == pytest.approx(59.43, abs=0.01)
    assert artifacts.results.gross_outer_wall_area_m2 == pytest.approx(238.53, abs=0.15)
    stages = [event["stage"] for event in events if event["status"] == "completed"]
    assert "ingest" in stages
    assert "classify" in stages
    assert "report" in stages
