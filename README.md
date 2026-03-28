# First Mate Local-First Estimator

Hackathon-grade but deterministic-first estimator for house exterior quantities from architectural PDF drawings.

## What it does

- ingests all PDFs in an input folder
- classifies pages as `plan`, `section`, `elevations`, or `unknown`
- extracts embedded text and vector geometry with PyMuPDF
- reconstructs a rectangular exterior footprint from overall plan dimensions
- estimates facade widths and heights from elevations plus section cross-checks
- detects likely windows and doors from elevation vector boxes
- parses cladding labels such as `3a` and `3b`
- writes JSON results, a Markdown report, annotated overlays, and debug JSON

The pipeline is local-first. It does not require remote LLM access. OCR is an optional fallback only.
An optional AI resolver can be enabled in `auto` or `require` mode. It tries a local Ollama endpoint first and can also use a configured OpenAI-compatible endpoint, but the estimator still runs without any model runtime.
If `OPENAI_API_KEY` is present, the resolver now defaults to OpenAI with `gpt-4o` unless you override the model.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Optional OCR fallback:

- install Tesseract locally
- keep `--ocr auto` or use `--ocr require`

## Usage

```bash
python -m app.run --input ./data --output ./out
```

Optional hybrid AI mode:

```bash
python -m app.run \
  --input ./data \
  --output ./out \
  --ai auto
```

With OpenAI:

```bash
export OPENAI_API_KEY=...
python -m app.run \
  --input ./data \
  --output ./out \
  --ai auto
```

Optional model override:

```bash
python -m app.run \
  --input ./data \
  --output ./out \
  --ai require \
  --ai-model llava
```

Example with the included sample fixtures:

```bash
python -m app.run \
  --input ./tests/fixtures/sample_drawings \
  --output ./out
```

## Local UI

Launch the local drag-and-drop UI:

```bash
python -m app.ui --open-browser
```

Then open `http://127.0.0.1:8765` if the browser does not open automatically.

The UI:

- accepts multiple PDF uploads by drag and drop
- can run the bundled sample drawing set with one click
- exposes an `AI resolver` mode and optional model override
- streams the pipeline stages as they execute
- shows final metrics, warnings, assumptions, facade tables, overlays, and report preview
- keeps all uploaded files and outputs local under `.ui_runs/<job_id>/`

## Outputs

- `out/results.json`
- `out/report.md`
- `out/overlays/plan_overlay.png`
- `out/overlays/section_overlay.png`
- `out/overlays/elevations_overlay.png`
- `out/debug/*.json`
- `out/debug/ai_decisions.json`

UI runs create the same outputs under `.ui_runs/<job_id>/output/`.

## Result schema

```json
{
  "perimeter_exterior_m": 0.0,
  "gross_outer_wall_area_m2": 0.0,
  "openings_area_m2": 0.0,
  "net_cladding_area_m2": 0.0,
  "cladding_by_type": {},
  "assumptions": [],
  "warnings": [],
  "confidence": {
    "overall": 0.0,
    "geometry": 0.0,
    "openings": 0.0,
    "materials": 0.0
  }
}
```

## Notes on the current heuristic model

- footprint reconstruction is intentionally conservative and currently assumes an orthogonal exterior footprint from overall plan dimension chains
- facade gross areas are calibrated from elevation cluster widths and heights
- gable end area is modeled as `width * eave_height + 0.5 * width * (ridge_height - eave_height)`
- material split is deterministic but approximate when multiple local cladding labels appear on a facade
- the report and debug JSON are intended to make every assumption auditable

## Optional AI resolver

When `--ai auto` or `--ai require` is enabled, the agent can use an optional model runtime to resolve ambiguous decisions while keeping the math deterministic.

Current AI decision points:

- choose the correct plan subview when a page contains both a main floor and loft/secondary view
- choose the wall-height marker from competing section/elevation level candidates
- choose the dominant cladding code when legend and facade labeling are ambiguous

The AI layer never computes quantities directly. It only selects among deterministic candidates and records its decision in `debug/ai_decisions.json`, the Markdown report, and the UI.

## Tests

```bash
pytest
```
