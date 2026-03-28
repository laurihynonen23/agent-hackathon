from pathlib import Path

from app.ai import CandidateRegion, HybridAiResolver
from app.types import AiSettings, BBox, DocumentPage


def test_ai_resolver_records_fallback_when_endpoint_is_unavailable():
    resolver = HybridAiResolver(AiSettings(mode="auto"))
    selected = resolver.choose_wall_height(
        candidate_values_m=[2.846, 4.015, 6.317],
        evidence={"counts": {"2.846": 1, "4.015": 2, "6.317": 1}},
    )
    assert selected is None
    assert resolver.decisions
    assert resolver.decisions[0].decision_type == "wall_height"
    assert resolver.decisions[0].fallback_used is True


def test_ai_resolver_uses_mocked_plan_choice(monkeypatch):
    resolver = HybridAiResolver(AiSettings(mode="auto"))
    page = DocumentPage(
        page_id="plan_p1",
        file_name="plan.pdf",
        file_path=Path("plan.pdf"),
        page_number=1,
        width_pt=1000,
        height_pt=800,
        render_path=Path("tests/fixtures/sample_drawings/ARK 02 Pohjakuva 1111 (2).pdf"),
        raw_text="POHJA PARVEN POHJA",
        text_spans=[],
        vector_primitives=[],
    )
    candidates = [
        CandidateRegion("main_top_1", "POHJA 1:50", BBox(x0=0, y0=0, x1=1000, y1=400), {"view_kind": "main"}),
        CandidateRegion("loft_bottom_1", "PARVEN POHJA 1:50", BBox(x0=0, y0=400, x1=1000, y1=800), {"view_kind": "loft"}),
    ]

    monkeypatch.setattr(
        resolver,
        "_chat_json",
        lambda *args, **kwargs: {
            "selected_candidate_id": "main_top_1",
            "confidence": 0.92,
            "rationale": "Ground-floor plan should drive exterior footprint.",
            "_provider": "mock",
            "_model": "mock-model",
            "_raw": {"ok": True},
        },
    )

    selected = resolver.choose_plan_region(page, candidates)
    assert selected == "main_top_1"
    assert resolver.decisions[-1].used is True
    assert resolver.decisions[-1].selected == "main_top_1"
