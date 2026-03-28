from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .extract_dimensions import measurement_summary
from .planner import run_pipeline
from .types import PipelineArtifacts


APP_ROOT = Path(__file__).resolve().parent.parent
UI_RUNS_DIR = APP_ROOT / ".ui_runs"
SAMPLE_INPUT_DIR = APP_ROOT / "tests" / "fixtures" / "sample_drawings"
STAGE_ORDER = ["ingest", "classify", "extract", "geometry", "openings", "materials", "validate", "report"]
STAGE_LABELS = {
    "ingest": "Ingest",
    "classify": "Classify",
    "extract": "Extract",
    "geometry": "Reconstruct",
    "openings": "Openings",
    "materials": "Materials",
    "validate": "Validate",
    "report": "Report",
}


@dataclass
class JobState:
    job_id: str
    input_dir: Path
    output_dir: Path
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    events: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


JOB_REGISTRY: dict[str, JobState] = {}
JOB_REGISTRY_LOCK = threading.Lock()


def build_index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>First Mate Transparent Estimator</title>
  <style>
    :root {
      --bg: #f5efe2;
      --bg-deep: #efe6d5;
      --ink: #18161a;
      --muted: #5d5a5f;
      --line: rgba(24, 22, 26, 0.12);
      --blue: #0f5f8a;
      --orange: #d26a1b;
      --green: #2b7a43;
      --red: #a83f33;
      --card: rgba(255, 252, 247, 0.82);
      --shadow: 0 18px 45px rgba(58, 39, 11, 0.12);
      --radius: 22px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 95, 138, 0.12), transparent 25%),
        radial-gradient(circle at bottom right, rgba(210, 106, 27, 0.15), transparent 20%),
        linear-gradient(180deg, var(--bg), var(--bg-deep));
      min-height: 100vh;
      overflow-x: hidden;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(15, 95, 138, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(15, 95, 138, 0.05) 1px, transparent 1px);
      background-size: 24px 24px;
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.4), transparent 85%);
    }
    .shell {
      width: min(100%, 1460px);
      margin: 0 auto;
      padding: 28px 24px 56px;
      position: relative;
      z-index: 1;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 22px;
    }
    .panel {
      background: var(--card);
      border: 1px solid rgba(24, 22, 26, 0.08);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
      backdrop-filter: blur(14px);
    }
    .hero-main {
      padding: 30px 32px 28px;
      position: relative;
      overflow: hidden;
    }
    .hero-main::after {
      content: "";
      position: absolute;
      right: -60px;
      top: -80px;
      width: 240px;
      height: 240px;
      background: radial-gradient(circle, rgba(210, 106, 27, 0.22), transparent 70%);
      border-radius: 50%;
    }
    .hero-kicker {
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--blue);
      font-size: 12px;
      margin-bottom: 10px;
    }
    .hero h1 {
      margin: 0 0 12px;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      font-size: clamp(2.4rem, 4vw, 4.4rem);
      line-height: 0.95;
      max-width: 12ch;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 1.02rem;
      line-height: 1.6;
      max-width: 58ch;
    }
    .hero-side {
      padding: 26px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .badge {
      border: 1px solid rgba(24, 22, 26, 0.12);
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 0.83rem;
      background: rgba(255,255,255,0.55);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(320px, 390px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .controls {
      padding: 22px;
      position: sticky;
      top: 18px;
      align-self: start;
    }
    .hero-main, .hero-side, .controls, .workspace, .panel, .block, .status-strip, .stage-card, .metric, .overlay-card {
      min-width: 0;
    }
    .section-title {
      margin: 0 0 10px;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      font-size: 1.35rem;
    }
    .muted {
      color: var(--muted);
      font-size: 0.95rem;
      line-height: 1.6;
    }
    .dropzone {
      margin-top: 16px;
      border: 2px dashed rgba(15, 95, 138, 0.35);
      border-radius: 18px;
      padding: 18px;
      background: rgba(15, 95, 138, 0.04);
      transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
      cursor: pointer;
    }
    .dropzone.dragover {
      border-color: var(--orange);
      background: rgba(210, 106, 27, 0.09);
      transform: translateY(-1px);
    }
    .dropzone strong {
      display: block;
      font-size: 1rem;
      margin-bottom: 6px;
    }
    .file-list {
      margin: 12px 0 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 180px;
      overflow: auto;
    }
    .file-list li {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.8rem;
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(24, 22, 26, 0.08);
    }
    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }
    label {
      display: block;
      font-size: 0.85rem;
      margin-bottom: 6px;
      color: var(--muted);
    }
    select, input[type="number"] {
      width: 100%;
      padding: 12px 13px;
      border-radius: 14px;
      border: 1px solid rgba(24, 22, 26, 0.12);
      background: rgba(255,255,255,0.92);
      font-size: 0.95rem;
      color: var(--ink);
    }
    .button-row {
      display: flex;
      gap: 10px;
      margin-top: 18px;
      flex-wrap: wrap;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 13px 18px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: wait; transform: none; }
    .primary-btn { background: var(--ink); color: white; }
    .secondary-btn { background: rgba(255,255,255,0.8); color: var(--ink); border: 1px solid rgba(24,22,26,0.12); }
    .workspace {
      display: grid;
      gap: 18px;
      min-width: 0;
    }
    .status-strip {
      padding: 18px 22px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .status-meta {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.8rem;
      color: var(--muted);
    }
    .status-pill {
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(15, 95, 138, 0.08);
      color: var(--blue);
    }
    .stage-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      padding: 0 22px 22px;
    }
    .stage-card {
      border: 1px solid rgba(24,22,26,0.08);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.7);
      min-height: 96px;
    }
    .stage-card h4 {
      margin: 0 0 8px;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .stage-state {
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.78rem;
      margin-bottom: 8px;
      color: var(--muted);
    }
    .stage-card.pending { opacity: 0.58; }
    .stage-card.running { border-color: rgba(15, 95, 138, 0.3); box-shadow: inset 0 0 0 1px rgba(15, 95, 138, 0.15); }
    .stage-card.completed { border-color: rgba(43, 122, 67, 0.35); }
    .stage-card.failed { border-color: rgba(168, 63, 51, 0.38); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    .metric {
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255,255,255,0.75);
      border: 1px solid rgba(24,22,26,0.08);
    }
    .metric-label {
      font-size: 0.82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .metric-value {
      margin-top: 10px;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      font-size: 2rem;
      line-height: 1;
    }
    .content-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
      gap: 18px;
    }
    .block {
      padding: 22px;
    }
    .block-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 14px;
    }
    .event-feed {
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 620px;
      overflow: auto;
      padding-right: 4px;
    }
    .event {
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(24,22,26,0.08);
      border-left: 4px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
    }
    .event.running { border-left-color: var(--blue); }
    .event.completed { border-left-color: var(--green); }
    .event.failed { border-left-color: var(--red); }
    .event header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 8px;
      font-size: 0.84rem;
    }
    .event-title {
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .event-time {
      color: var(--muted);
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
    }
    .event-summary {
      margin: 0 0 10px;
      color: var(--ink);
      line-height: 1.5;
    }
    .event-details {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .event-detail {
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(24, 22, 26, 0.04);
      border: 1px solid rgba(24,22,26,0.06);
      min-width: 0;
    }
    .event-detail-key {
      display: block;
      margin-bottom: 6px;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .event-detail-value {
      margin: 0;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.74rem;
      line-height: 1.45;
      color: var(--ink);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .confidence-strip {
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }
    .confidence-card {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(24,22,26,0.08);
      min-width: 0;
      min-height: 88px;
    }
    .confidence-label {
      font-size: 0.74rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      line-height: 1.25;
    }
    .confidence-value {
      margin-top: 8px;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 1rem;
      font-weight: 700;
      color: var(--ink);
      line-height: 1.1;
    }
    .artifact-links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .artifact-links a {
      text-decoration: none;
      color: var(--ink);
      border: 1px solid rgba(24,22,26,0.1);
      border-radius: 999px;
      padding: 9px 12px;
      font-size: 0.84rem;
      background: rgba(255,255,255,0.78);
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid rgba(24,22,26,0.08);
      background: rgba(255,255,255,0.72);
      min-width: 0;
      max-width: 100%;
    }
    .chip-key {
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.72rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .chip-value {
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--ink);
    }
    table {
      width: 100%;
      table-layout: fixed;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid rgba(24,22,26,0.08);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 0.74rem;
    }
    .list {
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.6;
    }
    .overlay-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .overlay-card {
      display: block;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(24,22,26,0.08);
      background: rgba(255,255,255,0.75);
      text-decoration: none;
      color: inherit;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
      cursor: pointer;
    }
    .overlay-card:hover {
      transform: translateY(-2px);
      border-color: rgba(15, 95, 138, 0.24);
      box-shadow: 0 14px 28px rgba(58, 39, 11, 0.08);
    }
    .overlay-card img {
      width: 100%;
      display: block;
      background: white;
    }
    .overlay-caption {
      padding: 12px 14px;
      font-size: 0.86rem;
    }
    .overlay-hint {
      display: block;
      margin-top: 4px;
      font-size: 0.74rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    pre {
      margin: 0;
      padding: 16px;
      background: #1d1d1f;
      color: #f8f0e3;
      border-radius: 18px;
      overflow-x: hidden;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.8rem;
      line-height: 1.55;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .empty-state {
      padding: 24px;
      border-radius: 18px;
      background: rgba(255,255,255,0.6);
      border: 1px dashed rgba(24,22,26,0.14);
      color: var(--muted);
      line-height: 1.6;
    }
    .report-preview {
      max-height: 520px;
      overflow: auto;
    }
    .split-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .status-ok { color: var(--green); }
    .status-bad { color: var(--red); }
    @media (max-width: 1280px) {
      .layout { grid-template-columns: minmax(300px, 340px) minmax(0, 1fr); }
      .metrics, .stage-grid, .overlay-grid, .confidence-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 1200px) {
      .hero, .layout, .content-grid { grid-template-columns: 1fr; }
      .controls { position: static; }
      .metrics, .stage-grid, .overlay-grid, .confidence-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      .shell { padding: 16px; }
      .metrics, .stage-grid, .overlay-grid, .controls-grid, .split-grid, .confidence-strip { grid-template-columns: 1fr; }
      .button-row { flex-direction: column; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel hero-main">
        <div class="hero-kicker">First Mate • Transparent Workbench</div>
        <h1>Local PDF takeoff with the process fully exposed.</h1>
        <p>
          Upload architectural drawing PDFs, run the estimator locally, and inspect every stage:
          sheet classification, measurement extraction, geometry reconstruction, openings, cladding split,
          validation, and output generation. No black box. The evidence stays visible.
        </p>
      </div>
      <aside class="panel hero-side">
        <div>
          <div class="section-title">Why this feels transparent</div>
          <p class="muted">The UI streams the actual pipeline steps, shows intermediate metrics, exposes warnings early, and keeps the raw artifacts downloadable.</p>
        </div>
        <div class="badge-row">
          <span class="badge">Drag & drop PDFs</span>
          <span class="badge">Local-first</span>
          <span class="badge">Deterministic geometry</span>
          <span class="badge">Overlay previews</span>
          <span class="badge">Debug JSON</span>
        </div>
      </aside>
    </section>

    <section class="layout">
      <aside class="panel controls">
        <div class="section-title">New Run</div>
        <p class="muted">Drop one or more PDFs. The app will ingest all uploaded sheets together as one drawing set.</p>

        <input id="fileInput" type="file" accept=".pdf,application/pdf" multiple hidden>
        <div id="dropzone" class="dropzone" tabindex="0">
          <strong>Drop PDFs here</strong>
          <span class="muted">or click to browse local files</span>
          <ul id="fileList" class="file-list"></ul>
        </div>

        <div class="controls-grid">
          <div>
            <label for="ocrMode">OCR policy</label>
            <select id="ocrMode">
              <option value="auto">auto</option>
              <option value="off">off</option>
              <option value="require">require</option>
            </select>
          </div>
          <div>
            <label for="renderDpi">Render DPI</label>
            <input id="renderDpi" type="number" min="96" max="400" step="4" value="200">
          </div>
          <div>
            <label for="aiMode">AI resolver</label>
            <select id="aiMode">
              <option value="auto">auto</option>
              <option value="off">off</option>
              <option value="require">require</option>
            </select>
          </div>
          <div>
            <label for="aiModel">AI model</label>
            <input id="aiModel" type="text" placeholder="gpt-4o by default with OpenAI">
          </div>
        </div>

        <div class="button-row">
          <button id="runUploads" class="primary-btn">Analyze Uploaded PDFs</button>
          <button id="runSample" class="secondary-btn">Run Bundled Sample Set</button>
        </div>

        <div style="margin-top:18px">
          <div class="section-title" style="font-size:1.08rem">Execution style</div>
          <ul class="list">
            <li>Files are stored under a local `.ui_runs` folder.</li>
            <li>Each job exposes raw warnings, assumptions, and stage metrics.</li>
            <li>Final results never hide uncertainty behind a single number.</li>
          </ul>
        </div>
      </aside>

      <main class="workspace">
        <section class="panel">
          <div class="status-strip">
            <div>
              <div class="section-title" style="margin-bottom:6px">Agent Process</div>
              <div class="muted">Live execution stages and evidence from the current run.</div>
            </div>
            <div>
              <span id="runStatusPill" class="status-pill">Idle</span>
            </div>
            <div id="statusMeta" class="status-meta"></div>
          </div>
          <div id="stageGrid" class="stage-grid"></div>
        </section>

        <section class="content-grid">
          <div class="panel block">
            <div class="block-header">
              <div class="section-title" style="font-size:1.18rem">Process Ledger</div>
              <span class="muted">Newest stage activity first</span>
            </div>
            <div id="eventFeed" class="event-feed">
              <div class="empty-state">Start a run to see the stage-by-stage ledger. Each entry will include the pipeline step, summary, timestamp, and compact metrics.</div>
            </div>
          </div>

          <div class="panel block">
            <div class="block-header">
              <div class="section-title" style="font-size:1.18rem">Final Numbers</div>
              <span class="muted">Result summary</span>
            </div>
            <div id="metricsGrid" class="metrics">
              <div class="metric"><div class="metric-label">Perimeter</div><div class="metric-value">--</div></div>
              <div class="metric"><div class="metric-label">Gross Wall Area</div><div class="metric-value">--</div></div>
              <div class="metric"><div class="metric-label">Openings</div><div class="metric-value">--</div></div>
              <div class="metric"><div class="metric-label">Net Cladding</div><div class="metric-value">--</div></div>
            </div>
            <div id="confidenceStrip" class="confidence-strip"></div>
            <div id="artifactLinks" class="artifact-links" style="margin-top:16px"></div>
          </div>
        </section>

        <section class="content-grid">
          <div class="panel block">
            <div class="block-header">
              <div class="section-title" style="font-size:1.18rem">Result Breakdown</div>
              <span class="muted">Structured output and evidence</span>
            </div>
            <div id="resultBreakdown" class="empty-state">No run results yet.</div>
          </div>

          <div class="panel block">
            <div class="block-header">
              <div class="section-title" style="font-size:1.18rem">Warnings & Assumptions</div>
              <span class="muted">Audit trail</span>
            </div>
            <div id="assumptionWarningBlock" class="empty-state">Warnings and assumptions will appear here after a run.</div>
          </div>
        </section>

        <section class="panel block">
          <div class="block-header">
            <div class="section-title" style="font-size:1.18rem">Overlay Preview</div>
            <span class="muted">Annotated evidence images</span>
          </div>
          <div id="overlayGrid" class="overlay-grid">
            <div class="empty-state">Overlays will appear after a successful run.</div>
          </div>
        </section>

        <section class="content-grid">
          <div class="panel block">
            <div class="block-header">
              <div class="section-title" style="font-size:1.18rem">Report Preview</div>
              <span class="muted">Generated Markdown</span>
            </div>
            <div id="reportPreview" class="report-preview empty-state">No report yet.</div>
          </div>

          <div class="panel block">
            <div class="block-header">
              <div class="section-title" style="font-size:1.18rem">Debug Snapshot</div>
              <span class="muted">Compact raw JSON</span>
            </div>
            <pre id="debugPreview">{}</pre>
          </div>
        </section>
      </main>
    </section>
  </div>

  <script>
    const stageOrder = ["ingest", "classify", "extract", "geometry", "openings", "materials", "validate", "report"];
    const stageLabels = {
      ingest: "Ingest",
      classify: "Classify",
      extract: "Extract",
      geometry: "Reconstruct",
      openings: "Openings",
      materials: "Materials",
      validate: "Validate",
      report: "Report"
    };
    let selectedFiles = [];
    let pollTimer = null;
    let currentJobId = null;

    const fileInput = document.getElementById("fileInput");
    const fileList = document.getElementById("fileList");
    const dropzone = document.getElementById("dropzone");
    const runUploads = document.getElementById("runUploads");
    const runSample = document.getElementById("runSample");
    const statusPill = document.getElementById("runStatusPill");
    const statusMeta = document.getElementById("statusMeta");

    function escapeHtml(text) {
      return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setFiles(files) {
      selectedFiles = Array.from(files).filter((file) => file.name.toLowerCase().endsWith(".pdf"));
      renderFileList();
    }

    function renderFileList() {
      if (!selectedFiles.length) {
        fileList.innerHTML = "";
        return;
      }
      fileList.innerHTML = selectedFiles.map((file) => {
        const size = (file.size / 1024 / 1024).toFixed(2) + " MB";
        return `<li><span>${escapeHtml(file.name)}</span><span>${size}</span></li>`;
      }).join("");
    }

    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });
    fileInput.addEventListener("change", (event) => setFiles(event.target.files));

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      setFiles(event.dataTransfer.files);
    });

    function setBusy(busy) {
      runUploads.disabled = busy;
      runSample.disabled = busy;
      fileInput.disabled = busy;
    }

    function renderStageGrid(events) {
      const latestByStage = {};
      for (const event of events) {
        latestByStage[event.stage] = event;
      }
      const html = stageOrder.map((stage) => {
        const event = latestByStage[stage];
        const status = event ? event.status : "pending";
        const summary = event ? escapeHtml(event.summary) : "Waiting for this stage.";
        return `
          <div class="stage-card ${status}">
            <h4>${stageLabels[stage]}</h4>
            <div class="stage-state">${status}</div>
            <div>${summary}</div>
          </div>
        `;
      }).join("");
      document.getElementById("stageGrid").innerHTML = html;
    }

    function renderStatus(job) {
      statusPill.textContent = job.status;
      statusPill.className = "status-pill";
      if (job.status === "completed") {
        statusPill.style.color = "var(--green)";
        statusPill.style.background = "rgba(43, 122, 67, 0.08)";
      } else if (job.status === "failed") {
        statusPill.style.color = "var(--red)";
        statusPill.style.background = "rgba(168, 63, 51, 0.08)";
      } else if (job.status === "running") {
        statusPill.style.color = "var(--blue)";
        statusPill.style.background = "rgba(15, 95, 138, 0.08)";
      } else {
        statusPill.style.color = "var(--muted)";
        statusPill.style.background = "rgba(24, 22, 26, 0.06)";
      }
      statusMeta.innerHTML = `
        <span>job: ${escapeHtml(job.job_id ?? "--")}</span>
        <span>created: ${escapeHtml(job.created_at ?? "--")}</span>
        <span>input: ${escapeHtml(job.input_dir ?? "--")}</span>
      `;
    }

    function renderEvents(events) {
      const feed = document.getElementById("eventFeed");
      if (!events.length) {
        feed.innerHTML = `<div class="empty-state">No events yet.</div>`;
        return;
      }
      const html = [...events].reverse().map((event) => {
        const details = event.details || {};
        const detailEntries = Object.entries(details).slice(0, 6).map(([key, value]) => {
          const formatted = typeof value === "object"
            ? JSON.stringify(value, null, 2)
            : String(value);
          const clipped = formatted.length > 700 ? formatted.slice(0, 700) + "\\n..." : formatted;
          return `
            <div class="event-detail">
              <span class="event-detail-key">${escapeHtml(key)}</span>
              <div class="event-detail-value">${escapeHtml(clipped)}</div>
            </div>
          `;
        }).join("");
        return `
          <article class="event ${escapeHtml(event.status)}">
            <header>
              <span class="event-title">${escapeHtml(stageLabels[event.stage] || event.stage)}</span>
              <span class="event-time">${escapeHtml(event.timestamp)}</span>
            </header>
            <p class="event-summary">${escapeHtml(event.summary)}</p>
            ${detailEntries ? `<div class="event-details">${detailEntries}</div>` : ""}
          </article>
        `;
      }).join("");
      feed.innerHTML = html;
    }

    function metricCard(label, value) {
      return `<div class="metric"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div></div>`;
    }

    function confidenceCard(label, value) {
      return `
        <div class="confidence-card">
          <div class="confidence-label">${escapeHtml(label)}</div>
          <div class="confidence-value">${escapeHtml(value)}</div>
        </div>
      `;
    }

    function renderMetrics(results) {
      const grid = document.getElementById("metricsGrid");
      if (!results) {
      grid.innerHTML = [
        metricCard("Perimeter", "--"),
        metricCard("Gross Wall Area", "--"),
        metricCard("Openings", "--"),
        metricCard("Net Cladding", "--"),
        metricCard("Board Takeoff", "--")
      ].join("");
      renderConfidence(null);
      return;
    }
      const quantities = Object.entries(results.cladding_by_type || {});
      const primaryQuantity = quantities.sort((left, right) => right[1].linear_m_nominal_cover - left[1].linear_m_nominal_cover)[0];
      const takeoffValue = primaryQuantity
        ? `${primaryQuantity[1].linear_m_nominal_cover.toFixed(0)} lm · ${primaryQuantity[0]}`
        : "--";
      grid.innerHTML = [
        metricCard("Perimeter", results.perimeter_exterior_m.toFixed(2) + " m"),
        metricCard("Gross Wall Area", results.gross_outer_wall_area_m2.toFixed(2) + " m²"),
        metricCard("Openings", results.openings_area_m2.toFixed(2) + " m²"),
        metricCard("Net Cladding", results.net_cladding_area_m2.toFixed(2) + " m²"),
        metricCard("Board Takeoff", takeoffValue)
      ].join("");
      renderConfidence(results.confidence || null);
    }

    function renderConfidence(confidence) {
      const strip = document.getElementById("confidenceStrip");
      if (!confidence) {
        strip.innerHTML = [
          confidenceCard("Overall Confidence", "--"),
          confidenceCard("Geometry", "--"),
          confidenceCard("Openings", "--"),
          confidenceCard("Materials", "--")
        ].join("");
        return;
      }
      strip.innerHTML = [
        confidenceCard("Overall Confidence", confidence.overall.toFixed(2)),
        confidenceCard("Geometry", confidence.geometry.toFixed(2)),
        confidenceCard("Openings", confidence.openings.toFixed(2)),
        confidenceCard("Materials", confidence.materials.toFixed(2))
      ].join("");
    }

    function renderArtifactLinks(summary) {
      const el = document.getElementById("artifactLinks");
      if (!summary || !summary.files) {
        el.innerHTML = "";
        return;
      }
      const links = [];
      links.push(`<a href="${summary.files.results_json}" target="_blank" rel="noreferrer">Download results.json</a>`);
      links.push(`<a href="${summary.files.report_md}" target="_blank" rel="noreferrer">Open report.md</a>`);
      links.push(`<a href="${summary.files.debug_dir}" target="_blank" rel="noreferrer">Open debug folder listing</a>`);
      el.innerHTML = links.join("");
    }

    function renderResultBreakdown(summary) {
      const block = document.getElementById("resultBreakdown");
      if (!summary) {
        block.className = "empty-state";
        block.innerHTML = "No run results yet.";
        return;
      }
      const classificationRows = summary.classifications.map((item) => `
        <tr>
          <td>${escapeHtml(item.file_name)}</td>
          <td>${escapeHtml(item.role)}</td>
          <td>${escapeHtml(item.score.toFixed(2))}</td>
          <td>${escapeHtml((item.reasons || []).join("; "))}</td>
        </tr>
      `).join("");
      const materialRows = Object.entries(summary.results.cladding_by_type || {}).map(([code, payload]) => `
        <tr>
          <td>${escapeHtml(code)}</td>
          <td>${escapeHtml(payload.area_m2.toFixed(2))}</td>
          <td>${escapeHtml(payload.linear_m_nominal_cover.toFixed(2))}</td>
          <td>${escapeHtml(String(payload.assumed_nominal_cover_mm))}</td>
        </tr>
      `).join("") || `<tr><td colspan="4">No material quantities available.</td></tr>`;
      const aiDecisionRows = (summary.ai_decisions || []).map((decision) => `
        <tr>
          <td>${escapeHtml(decision.decision_type)}</td>
          <td>${escapeHtml(decision.used ? "ai" : "fallback")}</td>
          <td>${escapeHtml(decision.selected == null ? "--" : String(decision.selected))}</td>
          <td>${escapeHtml(decision.confidence == null ? "--" : decision.confidence.toFixed(2))}</td>
          <td>${escapeHtml(decision.rationale || decision.fallback_reason || "")}</td>
        </tr>
      `).join("") || `<tr><td colspan="5">No AI decisions recorded for this run.</td></tr>`;

      const facadeRows = summary.facades.map((facade) => `
        <tr>
          <td>${escapeHtml(facade.name)}</td>
          <td>${escapeHtml(facade.width_m.toFixed(2))}</td>
          <td>${escapeHtml(facade.total_height_m.toFixed(2))}</td>
          <td>${escapeHtml(facade.area_gross_m2.toFixed(2))}</td>
        </tr>
      `).join("");
      const measurementChips = Object.entries(summary.measurement_counts || {}).map(([key, value]) => `
        <span class="chip">
          <span class="chip-key">${escapeHtml(key)}</span>
          <span class="chip-value">${escapeHtml(String(value))}</span>
        </span>
      `).join("");
      block.className = "";
      block.innerHTML = `
        <div class="muted" style="margin-bottom:8px">Measurement counts</div>
        <div class="chip-row">${measurementChips}</div>
        <div style="margin-bottom:18px">
          <table>
            <thead><tr><th>AI Decision</th><th>Mode</th><th>Selected</th><th>Confidence</th><th>Rationale</th></tr></thead>
            <tbody>${aiDecisionRows}</tbody>
          </table>
        </div>
        <div style="margin-bottom:18px">
          <table>
            <thead><tr><th>Sheet</th><th>Role</th><th>Score</th><th>Reasons</th></tr></thead>
            <tbody>${classificationRows}</tbody>
          </table>
        </div>
        <div style="margin-bottom:18px">
          <table>
            <thead><tr><th>Facade</th><th>Width (m)</th><th>Height (m)</th><th>Gross Area (m²)</th></tr></thead>
            <tbody>${facadeRows}</tbody>
          </table>
        </div>
        <div>
          <table>
            <thead><tr><th>Material</th><th>Area (m²)</th><th>Linear m</th><th>Nominal cover</th></tr></thead>
            <tbody>${materialRows}</tbody>
          </table>
        </div>
      `;
    }

    function renderAssumptionsWarnings(summary) {
      const block = document.getElementById("assumptionWarningBlock");
      if (!summary) {
        block.className = "empty-state";
        block.innerHTML = "Warnings and assumptions will appear here after a run.";
        return;
      }
      const assumptions = (summary.results.assumptions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>None</li>";
      const warnings = (summary.results.warnings || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>None</li>";
      block.className = "";
      block.innerHTML = `
        <div class="split-grid">
          <div>
            <div class="section-title" style="font-size:1rem; margin-bottom:8px">Assumptions</div>
            <ul class="list">${assumptions}</ul>
          </div>
          <div>
            <div class="section-title" style="font-size:1rem; margin-bottom:8px">Warnings</div>
            <ul class="list">${warnings}</ul>
          </div>
        </div>
      `;
    }

    function renderOverlays(summary) {
      const grid = document.getElementById("overlayGrid");
      if (!summary || !summary.files || !summary.files.overlays) {
        grid.innerHTML = `<div class="empty-state">Overlays will appear after a successful run.</div>`;
        return;
      }
      const cards = Object.entries(summary.files.overlays).map(([label, url]) => `
        <a class="overlay-card" href="${url}" target="_blank" rel="noreferrer">
          <img src="${url}" alt="${escapeHtml(label)} overlay">
          <div class="overlay-caption">
            ${escapeHtml(label)}
            <span class="overlay-hint">Open full image</span>
          </div>
        </a>
      `).join("");
      grid.innerHTML = cards;
    }

    function renderReport(summary) {
      const report = document.getElementById("reportPreview");
      if (!summary || !summary.report_markdown) {
        report.className = "report-preview empty-state";
        report.textContent = "No report yet.";
        return;
      }
      report.className = "report-preview";
      report.innerHTML = `<pre>${escapeHtml(summary.report_markdown)}</pre>`;
    }

    function renderDebug(summary) {
      const debug = document.getElementById("debugPreview");
      if (!summary) {
        debug.textContent = "{}";
        return;
      }
      const snapshot = {
        confidence: summary.results.confidence,
        validation: summary.validation,
        openings_preview: summary.openings.slice(0, 8),
        materials_preview: summary.cladding_regions.slice(0, 8)
      };
      debug.textContent = JSON.stringify(snapshot, null, 2);
    }

    function renderJob(job) {
      renderStatus(job);
      renderStageGrid(job.events || []);
      renderEvents(job.events || []);
      if (job.summary) {
        renderMetrics(job.summary.results);
        renderArtifactLinks(job.summary);
        renderResultBreakdown(job.summary);
        renderAssumptionsWarnings(job.summary);
        renderOverlays(job.summary);
        renderReport(job.summary);
        renderDebug(job.summary);
      } else {
        renderMetrics(null);
        renderArtifactLinks(null);
      }
      if (job.status === "failed" && job.error) {
        document.getElementById("eventFeed").innerHTML = `
          <article class="event failed">
            <header><span class="event-title">Job failed</span></header>
            <p class="event-summary">${escapeHtml(job.error)}</p>
          </article>
        ` + document.getElementById("eventFeed").innerHTML;
      }
    }

    async function createJob(useSample) {
      const formData = new FormData();
      formData.append("ocr_mode", document.getElementById("ocrMode").value);
      formData.append("render_dpi", document.getElementById("renderDpi").value);
      formData.append("ai_mode", document.getElementById("aiMode").value);
      const aiModelValue = document.getElementById("aiModel").value.trim();
      if (aiModelValue) formData.append("ai_model", aiModelValue);
      if (useSample) {
        formData.append("use_sample", "1");
      } else {
        if (!selectedFiles.length) {
          alert("Upload at least one PDF or run the bundled sample set.");
          return;
        }
        for (const file of selectedFiles) {
          formData.append("files", file, file.name);
        }
      }
      setBusy(true);
      const response = await fetch("/api/jobs", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) {
        setBusy(false);
        alert(payload.error || "Job creation failed");
        return;
      }
      currentJobId = payload.job_id;
      pollJob();
      pollTimer = window.setInterval(pollJob, 1000);
    }

    async function pollJob() {
      if (!currentJobId) return;
      const response = await fetch(`/api/jobs/${currentJobId}`);
      const payload = await response.json();
      renderJob(payload);
      if (payload.status === "completed" || payload.status === "failed") {
        clearInterval(pollTimer);
        pollTimer = null;
        setBusy(false);
      }
    }

    runUploads.addEventListener("click", () => createJob(false));
    runSample.addEventListener("click", () => createJob(true));
    renderStageGrid([]);
  </script>
</body>
</html>"""


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")


def _read_markdown(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _make_artifact_url(job_id: str, relative_path: str) -> str:
    return f"/api/jobs/{job_id}/artifact/{relative_path}"


def build_job_summary(job_id: str, artifacts: PipelineArtifacts, output_dir: Path) -> dict[str, Any]:
    overlay_names = ["plan_overlay.png", "section_overlay.png", "elevations_overlay.png"]
    overlays = {
        name.replace("_", " ").replace(".png", "").title(): _make_artifact_url(job_id, f"overlays/{name}")
        for name in overlay_names
        if (output_dir / "overlays" / name).exists()
    }
    return {
        "job_id": job_id,
        "results": artifacts.results.model_dump(mode="json"),
        "classifications": [item.model_dump(mode="json") for item in artifacts.classifications],
        "measurement_counts": measurement_summary(artifacts.measurements),
        "facades": [item.model_dump(mode="json") for item in artifacts.facades],
        "openings": [item.model_dump(mode="json") for item in artifacts.openings],
        "cladding_regions": [item.model_dump(mode="json") for item in artifacts.cladding_regions],
        "material_specs": {code: spec.model_dump(mode="json") for code, spec in artifacts.material_specs.items()},
        "ai_decisions": [item.model_dump(mode="json") for item in artifacts.ai_decisions],
        "validation": artifacts.validation.model_dump(mode="json"),
        "report_markdown": _read_markdown(output_dir / "report.md"),
        "files": {
            "results_json": _make_artifact_url(job_id, "results.json"),
            "report_md": _make_artifact_url(job_id, "report.md"),
            "debug_dir": _make_artifact_url(job_id, "debug/classification.json"),
            "overlays": overlays,
        },
    }


def _safe_relative_resolve(base: Path, relative_path: str) -> Path:
    candidate = (base / relative_path).resolve()
    candidate.relative_to(base.resolve())
    return candidate


def _parse_multipart_form(content_type: str, body: bytes) -> tuple[dict[str, str], list[dict[str, Any]]]:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=default).parsebytes(header + body)
    fields: dict[str, str] = {}
    files: list[dict[str, Any]] = []
    if not message.is_multipart():
        return fields, files

    for part in message.iter_parts():
        params_list = part.get_params(header="content-disposition", failobj=[], unquote=True) or []
        params = {key: value for key, value in params_list[1:] if key}
        name = params.get("name")
        filename = params.get("filename")
        payload = part.get_payload(decode=True) or b""
        if filename:
            files.append(
                {
                    "field_name": name,
                    "filename": Path(filename).name,
                    "content_type": part.get_content_type(),
                    "content": payload,
                }
            )
        elif name:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    return fields, files


def _register_job(job: JobState) -> None:
    with JOB_REGISTRY_LOCK:
        JOB_REGISTRY[job.job_id] = job


def _get_job(job_id: str) -> JobState | None:
    with JOB_REGISTRY_LOCK:
        return JOB_REGISTRY.get(job_id)


def _record_job_event(job: JobState, event: dict[str, Any]) -> None:
    with job.lock:
        job.events.append(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                **event,
            }
        )


def _run_job(job: JobState, render_dpi: int, ocr_mode: str, ai_mode: str, ai_model: str | None) -> None:
    with job.lock:
        job.status = "running"
    _record_job_event(job, {"stage": "ingest", "status": "running", "summary": "Job started.", "details": {}})

    try:
        artifacts = run_pipeline(
            input_dir=job.input_dir,
            output_dir=job.output_dir,
            render_dpi=render_dpi,
            ocr_mode=ocr_mode,
            ai_mode=ai_mode,
            ai_model=ai_model,
            report_format="md",
            progress_callback=lambda event: _record_job_event(job, event),
        )
        summary = build_job_summary(job.job_id, artifacts, job.output_dir)
        with job.lock:
            job.summary = summary
            job.status = "completed"
        _record_job_event(job, {"stage": "report", "status": "completed", "summary": "Job finished successfully.", "details": {}})
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.error = str(exc)
        _record_job_event(job, {"stage": "report", "status": "failed", "summary": str(exc), "details": {}})


def _create_job_from_request(
    fields: dict[str, str],
    uploads: list[dict[str, Any]],
) -> JobState:
    job_id = uuid.uuid4().hex[:12]
    run_dir = UI_RUNS_DIR / job_id
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_sample = fields.get("use_sample") == "1"
    if use_sample:
        if not SAMPLE_INPUT_DIR.exists():
            raise FileNotFoundError("Bundled sample drawings are not available.")
        for sample in SAMPLE_INPUT_DIR.glob("*.pdf"):
            shutil.copy2(sample, input_dir / sample.name)
    else:
        pdf_uploads = [upload for upload in uploads if upload["filename"].lower().endswith(".pdf")]
        if not pdf_uploads:
            raise ValueError("No PDF files were uploaded.")
        for upload in pdf_uploads:
            (input_dir / upload["filename"]).write_bytes(upload["content"])

    job = JobState(job_id=job_id, input_dir=input_dir, output_dir=output_dir)
    _register_job(job)

    render_dpi = int(fields.get("render_dpi", "200"))
    ocr_mode = fields.get("ocr_mode", "auto")
    ai_mode = fields.get("ai_mode", "auto")
    ai_model = fields.get("ai_model") or None
    threading.Thread(target=_run_job, args=(job, render_dpi, ocr_mode, ai_mode, ai_model), daemon=True).start()
    return job


class EstimatorUIHandler(BaseHTTPRequestHandler):
    server_version = "FirstMateUI/0.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send(HTTPStatus.OK, build_index_html().encode("utf-8"), "text/html; charset=utf-8")
            return

        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 3:
                job_id = parts[2]
                job = _get_job(job_id)
                if job is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown job id"})
                    return

                if len(parts) == 3:
                    with job.lock:
                        payload = {
                            "job_id": job.job_id,
                            "status": job.status,
                            "created_at": job.created_at,
                            "input_dir": str(job.input_dir),
                            "output_dir": str(job.output_dir),
                            "events": job.events,
                            "summary": job.summary,
                            "error": job.error,
                        }
                    self._send_json(HTTPStatus.OK, payload)
                    return

                if len(parts) >= 5 and parts[3] == "artifact":
                    relative_path = unquote("/".join(parts[4:]))
                    try:
                        candidate = _safe_relative_resolve(job.output_dir, relative_path)
                    except Exception:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid artifact path"})
                        return
                    if not candidate.exists() or not candidate.is_file():
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Artifact not found"})
                        return
                    body = candidate.read_bytes()
                    content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                    self._send(HTTPStatus.OK, body, content_type)
                    return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Expected multipart form data"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        fields, uploads = _parse_multipart_form(content_type, body)
        try:
            job = _create_job_from_request(fields, uploads)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "job_id": job.job_id,
                "status": job.status,
                "created_at": job.created_at,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def run_ui_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), EstimatorUIHandler)
    url = f"http://{host}:{port}"
    print(f"First Mate Transparent UI running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the transparent local estimator UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local server to")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local server")
    parser.add_argument("--open-browser", action="store_true", help="Open the UI in a browser after launch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_ui_server(host=args.host, port=args.port, open_browser=args.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
