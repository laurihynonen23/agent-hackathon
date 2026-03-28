# Demo Notes: First Mate Local-First Estimator Agent

## One-line pitch

This is a local-first estimator agent for house drawing PDFs that uses deterministic PDF extraction and geometry for the math, plus an optional bounded AI resolver for ambiguous semantic decisions.

## What to emphasize first

Say this early:

1. The biggest design decision was to avoid a pure LLM blueprint-reader.
2. The agent is hybrid: deterministic for extraction and math, AI only for ambiguity resolution.
3. Every stage leaves an audit trail: debug JSON, overlays, AI decisions, assumptions, warnings, and confidence.

That framing hits the rubric directly:

- logic and critical thinking
- agentic workflow understanding
- reliability and safe failure
- clarity of decomposition

## 90-second demo structure

### 1. Problem framing: 15 seconds

"The challenge is to estimate perimeter, wall area, and cladding quantities from architectural PDFs. The risky way is to ask an LLM to interpret the drawings directly. I chose a hybrid design: deterministic extraction and geometry for the quantities, and AI only where the drawings are semantically ambiguous."

### 2. Show the trigger: 10 seconds

Open the UI and say:

"The trigger is simple: drop in a drawing set or run the bundled sample. That starts a local pipeline, no cloud dependency required."

### 3. Show the live process ledger: 20 seconds

Point at the stage list and say:

"The agent runs a fixed but inspectable workflow: ingest, classify, extract, geometry, openings, materials, validate, and report. Inside that flow, AI can choose things like the correct plan subview, the right wall-height marker, or the dominant cladding code. This is intentionally not a black box. Every stage emits intermediate state and metrics."

### 4. Show one or two artifacts: 20 seconds

Open overlays or debug JSON and say:

"Here you can see exactly what was measured. The plan overlay shows the footprint source and dimension evidence. The elevation overlay shows facade bounds, openings, and material labeling assumptions."

### 5. Show the result and confidence: 15 seconds

"The final output is not just a number. It also includes AI decisions, assumptions, warnings, and confidence split into geometry, openings, and materials. That matters because drawings are messy and a reliable estimator should expose uncertainty rather than invent certainty."

### 6. Close with design tradeoff: 10 seconds

"So the core idea is: deterministic extraction first, bounded AI for ambiguity resolution second, and explicit validation throughout."

## 3-minute demo structure

### Part 1: Trigger and pipeline

"The user drops PDFs into the UI. The UI stores them locally and starts a pipeline job. The planner is the agentic layer: it decides the stage order, invokes the optional AI resolver where useful, and collects intermediate artifacts."

Reference:

- [app/ui.py](/Users/laurihynonen/Aalto_Hackathon/app/ui.py)
- [app/planner.py](/Users/laurihynonen/Aalto_Hackathon/app/planner.py)

### Part 2: Ingestion

"We ingest all PDFs in the folder, render every page to PNG, extract embedded text, and extract vector primitives using PyMuPDF. OCR is only a fallback."

Reference:

- [app/ingest.py](/Users/laurihynonen/Aalto_Hackathon/app/ingest.py)

Key point:

"That is why the system is local, fast, and explainable. It is using the PDF's actual structure, not guessing from pixels first."

### Part 3: Sheet classification

"We classify each page into plan, section, elevations, or unknown using text cues like POHJA, LEIKKAUS, and JULKISIVU, plus simple layout heuristics like multiple facade labels."

Reference:

- [app/classify.py](/Users/laurihynonen/Aalto_Hackathon/app/classify.py)

Key point:

"This is simple on purpose. Under hackathon time constraints, transparent heuristics beat overcomplicated models."

### Part 4: Measurement extraction

"We normalize scales, dimensions, level markers, and material sizes into a measurement store. Everything becomes typed data with provenance, confidence, and a bounding box."

Reference:

- [app/extract_dimensions.py](/Users/laurihynonen/Aalto_Hackathon/app/extract_dimensions.py)
- [app/types.py](/Users/laurihynonen/Aalto_Hackathon/app/types.py)

Key point:

"This measurement store is the agent's working memory. It is explicit and inspectable, not hidden."

### Part 5: Geometry reconstruction

"Geometry is reconstructed from plan dimensions. For the sample set, the key fix was summing the outer dimension chains instead of accidentally using inner spans. That gives the correct perimeter. The optional AI resolver can choose which plan subview is the real footprint source when a page contains both `POHJA` and `PARVEN POHJA`. Facade areas then use plan-derived widths and repeated wall-height markers from section and elevation sheets."

Reference:

- [app/geometry.py](/Users/laurihynonen/Aalto_Hackathon/app/geometry.py)

Key point:

"This is where most of the real estimator logic lives: turn drawing evidence into a footprint and facade model."

### Part 6: Openings and materials

"Openings are detected from elevation vector boxes first, with raster fallback. Materials are parsed from the legend and section notes. The optional AI resolver can choose the dominant cladding code when the legend and facade labeling disagree. For the sample set, the section note drives the board takeoff."

Reference:

- [app/openings.py](/Users/laurihynonen/Aalto_Hackathon/app/openings.py)
- [app/materials.py](/Users/laurihynonen/Aalto_Hackathon/app/materials.py)

Key point:

"This is a pragmatic tradeoff: when exact facade segmentation is weak, we fall back to the most defensible material signal and say so explicitly."

### Part 7: Validation and reporting

"We cross-check the geometry against section level markers, evaluate opening coverage, evaluate material coverage, compute confidence, and then write results, report, debug JSON, AI decisions, and overlays."

Reference:

- [app/validate.py](/Users/laurihynonen/Aalto_Hackathon/app/validate.py)
- [app/report.py](/Users/laurihynonen/Aalto_Hackathon/app/report.py)

Key point:

"The system does not hide uncertainty. If evidence is weak, confidence drops and warnings go up."

## How to explain the agentic workflow

Use this exact structure if they ask "where is the agent":

### Trigger

- CLI run or UI upload

### Planning

- `run_pipeline()` is the planner
- it sequences the stages and emits progress events
- it invokes the optional AI resolver only for ambiguous choices

### Tool calls / actions

- PyMuPDF for PDF render/text/vector extraction
- regex parsing for dimensions and materials
- geometry reconstruction from dimension chains
- vector-first opening detection with raster fallback
- optional model calls for ambiguous decisions
- PIL overlays and JSON/Markdown report writing

### State / memory

- `DocumentPage`, `MeasurementStore`, `FootprintModel`, `FacadeModel`, `Opening`, `CladdingRegion`, `AiDecision`
- all explicit Pydantic models
- persisted debug artifacts in `out/debug/*.json`
- UI job state in memory for the current run

### Error handling

- fail early on missing input or no plan sheet
- OCR only when needed
- warnings instead of fabricated certainty
- confidence scores instead of binary success/failure

### Why this counts as agentic

"Because the system is not just a linear parser. It is a staged orchestrator that chooses which evidence to trust, uses fallbacks, validates cross-stage consistency, and produces an auditable state trace."

Important nuance:

"It is an agentic system, but the LLM is bounded. AI chooses among candidates; deterministic code still performs the actual measurement math."

## The strongest design choices to defend

### 1. Deterministic first, not LLM first

"The problem contains geometry, measurements, legends, and repeated notation. That is a bad fit for free-form LLM interpretation and a good fit for deterministic extraction."

### 2. Explicit state instead of hidden chain-of-thought

"Every important intermediate representation is saved as structured JSON. That makes debugging and judging easy."

### 3. Reliability over flash

"The system prefers a transparent approximation with warnings over a confident hallucination."

### 4. Smart simplicity under six-hour constraints

"The classification is heuristic, the planning is rule-based, and the fallbacks are simple. That is not a weakness here. It is evidence of good engineering judgment under time pressure."

## What to say about limitations

Do not hide them. Frame them as conscious scope choices.

"This version is strongest on text-bearing architectural PDFs with clear dimensions and legends. The weakest area is fine-grained material region segmentation and perfect opening subtraction on every facade style. That is why we expose assumptions and confidence rather than pretending the system is CAD-perfect."

## Likely judge questions and good answers

### "Why not just use GPT-4 or another LLM to read the drawings?"

"Because the challenge is mostly geometry and structured notation. Deterministic extraction is more reliable, more local-first, cheaper, and easier to debug. So I only use AI where the drawings become semantically ambiguous, not for the arithmetic itself."

### "What makes this an agent rather than a script?"

"The orchestrator manages a staged workflow, chooses fallback paths, keeps explicit working state, invokes AI only for ambiguous choices, validates consistency across stages, and surfaces confidence and warnings. It behaves like a constrained specialist agent rather than a one-shot script."

### "What is the memory?"

"The memory is the typed state: pages, measurements, geometry, openings, cladding regions, validation results, plus persisted debug JSON. I intentionally made memory explicit rather than implicit."

### "How does it fail?"

"It fails safely. If the plan is missing, it errors clearly. If a section or elevations sheet is weak, it still produces output but adds warnings and lowers confidence."

### "What is the most brittle part?"

"Openings and facade material segmentation are the least deterministic parts, because drawings vary in how windows and cladding regions are represented. That is why the validation and assumptions layer matters."

### "What was the hardest bug you fixed?"

"The key one was perimeter undercounting because inner plan spans were being used instead of the outer dimension chain. Fixing that made the geometry align with the reference house."

### "If you had two more days, what would you improve?"

"I would add stronger facade-region segmentation, better opening calibration from elevation scales, and a configurable procurement layer for waste factor and board lengths."

### "Why is the UI useful instead of just showing the final number?"

"Because the judging rubric rewards reasoning under constraints. The UI makes the reasoning visible: stage progression, intermediate metrics, overlays, assumptions, warnings, and confidence."

## A good final sentence

"The core value of the project is not just that it outputs a number. It is that it shows how it got there, how sure it is, and where it is making assumptions."

## Commands to remember

```bash
python -m app.run --input ./tests/fixtures/sample_drawings --output ./out
python -m app.ui --host 127.0.0.1 --port 8765
pytest -q tests
```
