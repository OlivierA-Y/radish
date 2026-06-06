"""
Live pipeline monitor.
Discovers runs from <results-dir>/runs/<timestamp>/ directories.
Legacy flat llm_log*.jsonl files at the results root are also shown.

Usage:
    python monitor.py
    python monitor.py --results-dir results/ucsf-pdgm --port 8765
Then open http://localhost:8765 in a browser.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_SUBJECT_RE = re.compile(r"^/api/subject/([^?]+)")

ROOT = Path(__file__).parent

_RESULTS_DIR: Path = ROOT / "results" / "ucsf-pdgm"
_META_CSV:    Path | None = None
_TOTAL:       int = 0          # fallback when no header present


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _load_ground_truth() -> dict[str, int]:
    sys.path.insert(0, str(ROOT))
    try:
        from src.data.data_loader import _parse_ucsf_idh
    except ImportError:
        return {}
    gt: dict[str, int] = {}
    if _META_CSV is None or not _META_CSV.exists():
        return gt
    with open(_META_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row.get("ID", "").strip()
            label = _parse_ucsf_idh(row.get("IDH", ""))
            if sid and label is not None:
                gt[sid] = label
    return gt


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _read_log_meta(log_path: Path) -> tuple[dict | None, int, str]:
    """Read header entry, count predictions, and extract model name."""
    header = None
    n = 0
    model = ""
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "run_header":
                        header = entry
                        if not model:
                            model = entry.get("model", "")
                    else:
                        n += 1
                        if not model:
                            model = entry.get("model_version") or entry.get("model", "")
                except Exception:
                    pass
    except OSError:
        pass
    return header, n, model


def _read_log(log_path: Path) -> list[dict]:
    """Return prediction entries only (skip run_header)."""
    entries = []
    if not log_path.exists():
        return entries
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") != "run_header":
                    entries.append(entry)
            except json.JSONDecodeError:
                pass
    return entries


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _format_label(ts: str) -> str:
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d  %H:%M:%S")
    except ValueError:
        return ts


def _discover_runs() -> list[dict]:
    """
    Returns [{id, label, n, total_subjects, model, done}] newest-first.

    New-style: results-dir/runs/<timestamp>/llm_log.jsonl
    Legacy:    results-dir/llm_log*.jsonl
    """
    runs: list[dict] = []

    # New-style runs
    runs_subdir = _RESULTS_DIR / "runs"
    if runs_subdir.exists():
        folders = sorted(
            [d for d in runs_subdir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        for folder in folders:
            log = folder / "llm_log.jsonl"
            if not log.exists():
                continue
            run_id = folder.name
            header, n, model = _read_log_meta(log)
            total = (header.get("total_subjects") if header else 0) or 0
            runs.append({
                "id":             run_id,
                "label":          _format_label(run_id),
                "n":              n,
                "total_subjects": total,
                "model":          model,
                "done":           (folder / "predictions.csv").exists(),
            })

    # Legacy flat files
    for f in sorted(
        _RESULTS_DIR.glob("llm_log*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        stem = f.stem
        ts = stem[len("llm_log_"):] if stem != "llm_log" else "legacy"
        header, n, model = _read_log_meta(f)
        total = (header.get("total_subjects") if header else 0) or 0
        runs.append({
            "id":             ts,
            "label":          _format_label(ts) + "  ·  legacy",
            "n":              n,
            "total_subjects": total,
            "model":          model,
            "done":           (_RESULTS_DIR / "predictions.csv").exists(),
        })

    return runs


def _resolve_log(run_id: str | None) -> Path:
    """Map a run ID to its log file path."""
    if run_id:
        # New-style
        new = _RESULTS_DIR / "runs" / run_id / "llm_log.jsonl"
        if new.exists():
            return new
        # Legacy
        for p in [
            _RESULTS_DIR / f"llm_log_{run_id}.jsonl",
            _RESULTS_DIR / f"llm_log.jsonl",
        ]:
            if p.exists():
                return p

    # Default: most recent new-style, then legacy
    runs_subdir = _RESULTS_DIR / "runs"
    if runs_subdir.exists():
        for folder in sorted(
            [d for d in runs_subdir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        ):
            log = folder / "llm_log.jsonl"
            if log.exists():
                return log
    files = sorted(
        _RESULTS_DIR.glob("llm_log*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else _RESULTS_DIR / "llm_log.jsonl"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats(log_path: Path) -> dict:
    gt    = _load_ground_truth()
    preds = _read_log(log_path)
    n     = len(preds)

    header, _, _ = _read_log_meta(log_path)
    total = (header.get("total_subjects") if header else 0) or _TOTAL

    run_dir  = log_path.parent          # works for both new-style and legacy
    feat_dir = _RESULTS_DIR / "features"

    n_mutant = sum(1 for p in preds if "mutant" in p.get("parsed", {}).get("idh_status", "").lower())
    correct  = sum(
        1 for p in preds
        if p.get("subject_id") in gt
        and (1 if "mutant" in p["parsed"].get("idh_status", "").lower() else 0) == gt[p["subject_id"]]
    )
    labelled = sum(1 for p in preds if p.get("subject_id") in gt)
    phantoms = sum(1 for p in preds if p.get("phantom_citations"))

    conf = {"high": 0, "medium": 0, "low": 0}
    for p in preds:
        c = p.get("parsed", {}).get("confidence", "").lower()
        if c in conf:
            conf[c] += 1

    avg_lat = round(sum(p.get("latency_s", 0) for p in preds) / n, 2) if n else 0

    recent = []
    for p in reversed(preds):
        parsed = p.get("parsed", {})
        sid    = p.get("subject_id", "")
        pred_l = 1 if "mutant" in parsed.get("idh_status", "").lower() else 0
        gt_l   = gt.get(sid)
        recent.append({
            "subject_id":    sid,
            "idh_status":    parsed.get("idh_status", "?"),
            "confidence":    parsed.get("confidence", "?"),
            "decisive":      (parsed.get("decisive_feature") or "")[:40],
            "phantoms":      p.get("phantom_citations") or [],
            "latency_s":     round(p.get("latency_s", 0), 2),
            "correct":       (pred_l == gt_l) if gt_l is not None else None,
            "model_version": p.get("model_version", ""),
        })

    final_met = None
    met_path  = run_dir / "metrics.json"
    if met_path.exists():
        try:
            final_met = json.loads(met_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "n":           n,
        "n_features":  len(list(feat_dir.glob("*.json"))) if feat_dir.exists() else 0,
        "total":       total,
        "n_mutant":    n_mutant,
        "n_wildtype":  n - n_mutant,
        "labelled":    labelled,
        "correct":     correct,
        "accuracy":    round(correct / labelled, 4) if labelled else None,
        "avg_latency": avg_lat,
        "conf":        conf,
        "phantoms":    phantoms,
        "recent":      recent,
        "done":        (run_dir / "predictions.csv").exists(),
        "final":       final_met,
    }


def subject_detail(subject_id: str, log_path: Path) -> dict | None:
    gt = _load_ground_truth()
    if not log_path.exists():
        return None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("subject_id") == subject_id:
                parsed = entry.get("parsed", {})
                pred_l = 1 if "mutant" in parsed.get("idh_status", "").lower() else 0
                gt_l   = gt.get(subject_id)
                entry["ground_truth"] = gt_l
                entry["correct"]      = (pred_l == gt_l) if gt_l is not None else None
                return entry
    return None


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline Monitor</title>
<style>
  :root {
    --bg:    #0f1117;
    --card:  #1a1d27;
    --border:#2a2d3a;
    --text:  #e2e8f0;
    --muted: #64748b;
    --green: #22c55e;
    --red:   #ef4444;
    --blue:  #3b82f6;
    --amber: #f59e0b;
    --purple:#a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif;
         font-size: 14px; line-height: 1.5; padding: 24px; }
  h1   { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: .85rem; margin-bottom: 20px; }
  .header-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
  .pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--green);
           box-shadow: 0 0 0 0 rgba(34,197,94,.7);
           animation: pulse 1.5s infinite; display: inline-block; margin-right: 6px; }
  .pulse.done { background: var(--blue); animation: none; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(34,197,94,.7)} 70%{box-shadow:0 0 0 8px rgba(34,197,94,0)} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0)} }

  /* Run selector */
  .run-bar { display:flex; align-items:flex-start; gap:6px; flex-wrap:wrap;
             padding-bottom:16px; margin-bottom:20px; border-bottom:1px solid var(--border); }
  .run-bar-label { font-size:.72rem; color:var(--muted); text-transform:uppercase;
                   letter-spacing:.07em; padding-top:9px; flex-shrink:0; }
  .run-tab { background:var(--card); border:1px solid var(--border); border-radius:8px;
             color:var(--muted); font-size:.8rem; font-weight:500; padding:7px 14px 6px;
             cursor:pointer; transition:all .15s; white-space:nowrap; text-align:left; line-height:1.3; }
  .run-tab:hover { border-color:var(--blue); color:var(--text); }
  .run-tab.active { background:rgba(59,130,246,.15); border-color:var(--blue); color:var(--text); }
  .run-tab .rt-ts    { display:block; font-size:.8rem; }
  .run-tab .rt-meta  { display:block; font-size:.68rem; color:var(--muted); margin-top:1px; }
  .run-tab.active .rt-meta { color:rgba(147,197,253,.7); }
  .no-runs { color:var(--muted); font-size:.85rem; padding:8px 0; }

  /* Progress */
  .progress-wrap { background: var(--border); border-radius: 999px; height: 10px; margin-bottom: 20px; overflow: hidden; }
  .progress-bar  { height: 100%; background: linear-gradient(90deg,var(--blue),var(--purple));
                   border-radius: 999px; transition: width .4s ease; }
  .progress-label{ font-size:.8rem; color:var(--muted); text-align:right; margin-top:4px; margin-bottom:20px; }

  /* Cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card  { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .card .val  { font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
  .card .lbl  { font-size: .75rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing:.05em; }
  .green { color: var(--green); }
  .red   { color: var(--red); }
  .blue  { color: var(--blue); }
  .amber { color: var(--amber); }

  /* Conf bar */
  .conf-bar { display: flex; gap: 4px; margin: 8px 0 4px; height: 6px; border-radius: 3px; overflow: hidden; }
  .conf-bar span { display: block; }
  .conf-label { font-size: .7rem; color: var(--muted); }

  /* Final metrics */
  .final-box { background: var(--card); border: 1px solid var(--green); border-radius: 10px;
               padding: 16px; margin-bottom: 24px; }
  .final-box h2 { font-size: 1rem; margin-bottom: 12px; color: var(--green); }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
  .metric-item  { background: var(--bg); border-radius: 6px; padding: 8px 12px; }
  .metric-item .mv { font-size: 1.2rem; font-weight: 600; }
  .metric-item .mk { font-size: .7rem; color: var(--muted); text-transform: uppercase; }

  /* Table */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: .82rem; }
  thead th { padding: 8px 12px; text-align: left; color: var(--muted); font-weight: 600;
             border-bottom: 1px solid var(--border); white-space: nowrap; }
  tbody tr { border-bottom: 1px solid var(--border); transition: background .1s; }
  tbody tr:hover { background: var(--card); }
  tbody td { padding: 7px 12px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .73rem; font-weight: 600; }
  .badge-mut { background: rgba(168,85,247,.2); color: var(--purple); }
  .badge-wt  { background: rgba(59,130,246,.2); color: var(--blue); }
  .badge-h   { background: rgba(34,197,94,.2);  color: var(--green); }
  .badge-m   { background: rgba(245,158,11,.2); color: var(--amber); }
  .badge-l   { background: rgba(239,68,68,.2);  color: var(--red); }
  .tick { color: var(--green); font-weight: 700; }
  .cross{ color: var(--red);   font-weight: 700; }
  .phantom { font-size:.7rem; color: var(--red); }
  .updated { font-size: .75rem; color: var(--muted); }
  h2.section { font-size: .9rem; color: var(--muted); text-transform: uppercase;
               letter-spacing: .06em; margin-bottom: 12px; }
  tbody tr { cursor: pointer; }

  /* Modal */
  .overlay { position:fixed; inset:0; background:rgba(0,0,0,.6); z-index:100;
             display:flex; align-items:center; justify-content:center;
             opacity:0; pointer-events:none; transition:opacity .15s; }
  .overlay.open { opacity:1; pointer-events:all; }
  .modal { background:var(--card); border:1px solid var(--border); border-radius:14px;
           width:min(860px,95vw); max-height:90vh; display:flex; flex-direction:column;
           box-shadow:0 24px 60px rgba(0,0,0,.5); }
  .modal-head { display:flex; align-items:center; justify-content:space-between;
                padding:16px 20px 12px; border-bottom:1px solid var(--border); flex-shrink:0; }
  .modal-head h3 { font-size:1rem; font-weight:700; }
  .modal-head .meta { font-size:.75rem; color:var(--muted); margin-top:2px; }
  .close-btn { background:none; border:none; color:var(--muted); font-size:1.3rem;
               cursor:pointer; padding:4px 8px; border-radius:6px; line-height:1; }
  .close-btn:hover { background:var(--border); color:var(--text); }
  .tabs { display:flex; gap:2px; padding:10px 20px 0; border-bottom:1px solid var(--border);
          flex-shrink:0; }
  .tab { background:none; border:none; color:var(--muted); font-size:.83rem; font-weight:500;
         padding:6px 14px; cursor:pointer; border-radius:6px 6px 0 0;
         border-bottom:2px solid transparent; margin-bottom:-1px; }
  .tab:hover  { color:var(--text); background:var(--border); }
  .tab.active { color:var(--text); border-bottom-color:var(--blue); }
  .tab-body   { overflow-y:auto; padding:20px; flex:1; }
  .tab-pane   { display:none; }
  .tab-pane.active { display:block; }

  /* Structured output */
  .row-group { margin-bottom:16px; }
  .row-group label { display:block; font-size:.7rem; color:var(--muted); text-transform:uppercase;
                     letter-spacing:.06em; margin-bottom:4px; }
  .row-group .val { font-size:.9rem; }
  .chip-list { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { background:var(--border); border-radius:6px; padding:3px 10px; font-size:.78rem;
          font-family:monospace; }
  .chip.phantom { background:rgba(239,68,68,.15); color:var(--red); }
  .text-block { font-size:.85rem; line-height:1.6; color:var(--text); }
  pre.code-block { background:var(--bg); border:1px solid var(--border); border-radius:8px;
                   padding:14px; font-size:.75rem; line-height:1.5; white-space:pre-wrap;
                   word-break:break-word; overflow-x:auto; font-family:'Cascadia Code',
                   'Fira Code','Consolas',monospace; color:#a5d6ff; }
  .section-label { font-size:.7rem; color:var(--muted); text-transform:uppercase;
                   letter-spacing:.06em; margin:12px 0 6px; }
  .section-label:first-child { margin-top:0; }
</style>
</head>
<body>
<div class="header-row">
  <div>
    <h1>Pipeline Monitor</h1>
    <div class="subtitle">zero-shot IDH prediction · radish</div>
  </div>
  <div><span class="pulse" id="dot"></span><span class="updated" id="updated">Connecting…</span></div>
</div>

<div class="run-bar" id="run-bar">
  <span class="run-bar-label">Run</span>
  <span class="no-runs" id="no-runs-msg">No runs found</span>
</div>

<div class="progress-wrap"><div class="progress-bar" id="pbar" style="width:0%"></div></div>
<div class="progress-label" id="plabel">0 predictions</div>

<div class="cards">
  <div class="card"><div class="val green"  id="accuracy">--</div>   <div class="lbl">Accuracy</div></div>
  <div class="card"><div class="val blue"   id="npred">0</div>       <div class="lbl">Predictions</div></div>
  <div class="card"><div class="val purple" id="nmut">--</div>       <div class="lbl">IDH-Mutant</div></div>
  <div class="card"><div class="val blue"   id="nwt">--</div>        <div class="lbl">IDH-Wildtype</div></div>
  <div class="card"><div class="val amber"  id="latency">--</div>    <div class="lbl">Avg Latency (s)</div></div>
  <div class="card"><div class="val red"    id="phantoms">--</div>   <div class="lbl">Phantom Cites</div></div>
</div>

<div id="final-section" style="display:none"></div>

<h2 class="section">Confidence distribution</h2>
<div class="conf-bar" id="conf-bar"></div>
<div class="conf-label" id="conf-label"></div>

<br>
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
  <h2 class="section" style="margin:0">All predictions (<span id="pred-count">0</span>, newest first)</h2>
  <input id="search" type="text" placeholder="Filter by subject ID or status…"
    style="background:var(--card);border:1px solid var(--border);border-radius:6px;
           color:var(--text);padding:6px 12px;font-size:.82rem;width:280px;outline:none;">
</div>
<div class="table-wrap">
<table>
  <thead><tr>
    <th>Subject</th><th>GT</th><th>Prediction</th><th>✓</th>
    <th>Confidence</th><th>Decisive Feature</th><th>Latency</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
</div>

<!-- Detail modal -->
<div class="overlay" id="overlay" role="dialog" aria-modal="true">
  <div class="modal">
    <div class="modal-head">
      <div>
        <h3 id="m-title">Subject detail</h3>
        <div class="meta" id="m-meta"></div>
      </div>
      <button class="close-btn" id="close-btn" aria-label="Close">&times;</button>
    </div>
    <div class="tabs" id="tabs">
      <button class="tab active" type="button" data-tab="output">Structured Output</button>
      <button class="tab"        type="button" data-tab="prompt">User Prompt</button>
      <button class="tab"        type="button" data-tab="raw">Raw Response</button>
    </div>
    <div class="tab-body">
      <div class="tab-pane active" id="tab-output"></div>
      <div class="tab-pane"        id="tab-prompt"></div>
      <div class="tab-pane"        id="tab-raw"></div>
    </div>
  </div>
</div>

<script>
let _currentRun = null;

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function fmt(v, digits=1) { return v == null ? '--' : (v*100).toFixed(digits)+'%'; }
function badge(status) {
  const m = status && status.toLowerCase().includes('mutant');
  return `<span class="badge ${m?'badge-mut':'badge-wt'}">${esc(status)||'?'}</span>`;
}
function confBadge(c) {
  const cls = c==='high'?'badge-h':c==='medium'?'badge-m':'badge-l';
  return `<span class="badge ${cls}">${esc(c)}</span>`;
}
function gtBadge(v) {
  if (v===null||v===undefined) return '<span style="color:var(--muted)">?</span>';
  return v===1?'<span class="badge badge-mut">mutant</span>':'<span class="badge badge-wt">wildtype</span>';
}

// ── Run selector ──────────────────────────────────────────────────────────
async function loadRuns() {
  try {
    const r = await fetch('/api/runs');
    const runs = await r.json();
    const bar   = document.getElementById('run-bar');
    const noMsg = document.getElementById('no-runs-msg');

    if (!runs.length) { noMsg.style.display=''; return; }
    noMsg.style.display = 'none';

    const existing = [...bar.querySelectorAll('.run-tab')].map(b => b.dataset.run);
    const incoming = runs.map(r => r.id);
    if (JSON.stringify(existing) === JSON.stringify(incoming)) return;

    bar.querySelectorAll('.run-tab').forEach(b => b.remove());
    runs.forEach((run, i) => {
      const isActive = _currentRun ? _currentRun === run.id : i === 0;
      const btn = document.createElement('button');
      btn.className = 'run-tab' + (isActive ? ' active' : '');
      btn.dataset.run = run.id;

      const nStr = run.total_subjects
        ? `${run.n} / ${run.total_subjects}`
        : `${run.n} pred`;
      const modelStr = run.model || '';
      btn.innerHTML =
        `<span class="rt-ts">${esc(run.label)}</span>`+
        `<span class="rt-meta">${esc(modelStr)}${modelStr?' · ':''}${esc(nStr)}</span>`;

      btn.addEventListener('click', () => {
        bar.querySelectorAll('.run-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        _currentRun = run.id;
        refresh();
      });
      bar.appendChild(btn);
    });
    if (!_currentRun) _currentRun = runs[0].id;
  } catch(e) { /* ignore */ }
}

// ── Dashboard update ──────────────────────────────────────────────────────
function update(d) {
  const total = d.total || 0;
  const pct   = (total && d.n) ? d.n / total * 100 : 0;
  document.getElementById('pbar').style.width = pct + '%';
  document.getElementById('plabel').textContent = total
    ? `${d.n} / ${total} predictions · ${d.n_features} features cached`
    : `${d.n} predictions · ${d.n_features} features cached`;

  document.getElementById('accuracy').textContent = fmt(d.accuracy);
  document.getElementById('npred').textContent    = d.n;
  document.getElementById('nmut').textContent     = d.n_mutant + (d.n ? ` (${(d.n_mutant/d.n*100).toFixed(0)}%)` : '');
  document.getElementById('nwt').textContent      = d.n_wildtype + (d.n ? ` (${(d.n_wildtype/d.n*100).toFixed(0)}%)` : '');
  document.getElementById('latency').textContent  = d.avg_latency;
  document.getElementById('phantoms').textContent = d.phantoms;

  const c = d.conf, ct = c.high + c.medium + c.low || 1;
  document.getElementById('conf-bar').innerHTML =
    `<span style="width:${c.high/ct*100}%;background:var(--green)"></span>`+
    `<span style="width:${c.medium/ct*100}%;background:var(--amber)"></span>`+
    `<span style="width:${c.low/ct*100}%;background:var(--red)"></span>`;
  document.getElementById('conf-label').textContent =
    `High ${c.high}  ·  Medium ${c.medium}  ·  Low ${c.low}`;

  if (d.final) {
    const f = d.final;
    const keys   = ['accuracy','sensitivity','specificity','ppv','npv','f1','balanced_accuracy','auc_roc'];
    const labels = ['Accuracy','Sensitivity','Specificity','PPV','NPV','F1','Balanced Acc','AUC-ROC'];
    const items  = keys.map((k,i) => f[k]!=null
      ?`<div class="metric-item"><div class="mv">${(f[k]*100).toFixed(1)}%</div><div class="mk">${labels[i]}</div></div>`
      :'').join('');
    document.getElementById('final-section').style.display='block';
    document.getElementById('final-section').innerHTML =
      `<div class="final-box"><h2>&#10003; Run complete &mdash; Final metrics (N=${f.n_total})</h2>`+
      `<div class="metrics-grid">${items}</div></div>`;
  }

  document.getElementById('dot').className = 'pulse' + (d.done ? ' done' : '');
  document.getElementById('updated').textContent =
    (d.done ? 'Run complete · ' : 'Live · ') + new Date().toLocaleTimeString();

  window._allRows = d.recent;
  document.getElementById('pred-count').textContent = d.recent.length;
  renderRows(window._allRows);

  // Update run tab label for the current run
  const activeTab = document.querySelector('.run-tab.active');
  if (activeTab && d.total) {
    const meta = activeTab.querySelector('.rt-meta');
    if (meta) {
      const model = activeTab.dataset.model || '';
      meta.textContent = (model ? model + ' · ' : '') + `${d.n} / ${d.total}`;
    }
  }
}

// ── Modal ─────────────────────────────────────────────────────────────────
const overlay  = document.getElementById('overlay');
const closeBtn = document.getElementById('close-btn');

function closeModal() { overlay.classList.remove('open'); }
closeBtn.addEventListener('click', closeModal);
overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

document.getElementById('tabs').addEventListener('click', e => {
  const btn = e.target.closest('[data-tab]');
  if (!btn) return;
  e.stopPropagation();
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  const pane = document.getElementById('tab-' + btn.dataset.tab);
  if (pane) pane.classList.add('active');
});

function chips(arr, cls='') {
  if (!arr || !arr.length) return '<span style="color:var(--muted)">none</span>';
  return arr.map(x => `<span class="chip ${cls}">${esc(x)}</span>`).join('');
}

function renderOutput(d) {
  const p = d.parsed || {};
  const gt = d.ground_truth;
  const pred_l = (p.idh_status||'').toLowerCase().includes('mutant') ? 1 : 0;
  const correct = gt != null ? (pred_l === gt) : null;
  const tickHtml = correct===true  ? `<span class="tick"> &#10003; correct</span>`
                 : correct===false ? `<span class="cross"> &#10007; incorrect (GT: ${gt===1?'mutant':'wildtype'})</span>`
                 : '';
  return `
    <div class="row-group"><label>Prediction</label>
      <div class="val">${badge(p.idh_status)} &nbsp; ${confBadge(p.confidence)} ${tickHtml}</div>
    </div>
    <div class="row-group"><label>Reasoning</label>
      <div class="text-block">${esc(p.reasoning||'—')}</div>
    </div>
    <div class="row-group"><label>Decisive Feature</label>
      <div class="chip-list">${chips([p.decisive_feature].filter(Boolean))}</div>
    </div>
    <div class="row-group"><label>Supporting Features</label>
      <div class="chip-list">${chips(p.supporting_features)}</div>
    </div>
    <div class="row-group"><label>Contradicting Features</label>
      <div class="chip-list">${chips(p.contradicting_features)}</div>
    </div>
    <div class="row-group"><label>Findings</label>
      <div class="text-block">${esc(p.findings||'—')}</div>
    </div>
    <div class="row-group"><label>Impression</label>
      <div class="text-block">${esc(p.impression||'—')}</div>
    </div>
    ${(d.phantom_citations||[]).length
      ?`<div class="row-group"><label>Phantom Citations</label>
        <div class="chip-list">${chips(d.phantom_citations,'phantom')}</div></div>`:''}`;
}

function _activateTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab===name));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id==='tab-'+name));
}

async function openDetail(sid) {
  _activateTab('output');
  document.getElementById('m-title').textContent  = sid;
  document.getElementById('m-meta').textContent   = 'Loading…';
  document.getElementById('tab-output').innerHTML = '<div style="color:var(--muted);padding:20px 0">Loading…</div>';
  document.getElementById('tab-prompt').innerHTML = '';
  document.getElementById('tab-raw').innerHTML    = '';
  overlay.classList.add('open');

  try {
    const qs = _currentRun ? `?run=${encodeURIComponent(_currentRun)}` : '';
    const r = await fetch(`/api/subject/${encodeURIComponent(sid)}${qs}`);
    if (!r.ok) { document.getElementById('m-meta').textContent = 'Not found'; return; }
    const d = await r.json();

    const mv  = d.model_version || d.model || '';
    const ts  = d.timestamp ? new Date(d.timestamp).toLocaleString() : '';
    const lat = d.latency_s  ? `${d.latency_s.toFixed(2)}s` : '';
    document.getElementById('m-meta').textContent =
      [mv, ts, lat && `latency ${lat}`].filter(Boolean).join('  ·  ');

    document.getElementById('tab-output').innerHTML = renderOutput(d);
    document.getElementById('tab-prompt').innerHTML =
      `<div class="section-label">System Prompt</div>`+
      `<pre class="code-block">${esc(d.system_prompt||'')}</pre>`+
      `<div class="section-label">User Prompt</div>`+
      `<pre class="code-block">${esc(d.user_prompt||'')}</pre>`;
    document.getElementById('tab-raw').innerHTML =
      `<pre class="code-block">${esc(d.raw_response||'(empty)')}</pre>`;
  } catch(e) {
    document.getElementById('m-meta').textContent = 'Error: ' + e.message;
  }
}

// ── Row rendering + search ────────────────────────────────────────────────
function rowHtml(r) {
  const tick = r.correct===true  ? '<span class="tick">&#10003;</span>'
             : r.correct===false ? '<span class="cross">&#10007;</span>'
             : '<span style="color:var(--muted)">&#8212;</span>';
  const ph = r.phantoms && r.phantoms.length
           ? `<br><span class="phantom">&#9888; ${r.phantoms.length} phantom</span>` : '';
  return `<tr data-sid="${esc(r.subject_id)}" title="Click to view full model I/O">
    <td style="font-family:monospace">${esc(r.subject_id)}</td>
    <td>${gtBadge(r.correct===null?null:(r.correct?(r.idh_status.includes('mutant')?1:0):(r.idh_status.includes('mutant')?0:1)))}</td>
    <td>${badge(r.idh_status)}${ph}</td>
    <td>${tick}</td>
    <td>${confBadge(r.confidence)}</td>
    <td style="color:var(--muted);font-size:.78rem">${esc(r.decisive)||'&mdash;'}</td>
    <td style="color:var(--muted)">${r.latency_s}s</td>
  </tr>`;
}

function renderRows(rows) {
  const q = (document.getElementById('search').value || '').toLowerCase();
  const filtered = q ? rows.filter(r =>
    r.subject_id.toLowerCase().includes(q) ||
    (r.idh_status||'').toLowerCase().includes(q) ||
    (r.confidence||'').toLowerCase().includes(q) ||
    (r.decisive||'').toLowerCase().includes(q)
  ) : rows;
  document.getElementById('tbody').innerHTML = filtered.map(rowHtml).join('');
  document.querySelectorAll('#tbody tr[data-sid]').forEach(tr => {
    tr.addEventListener('click', () => openDetail(tr.dataset.sid));
  });
}

document.getElementById('search').addEventListener('input', () => {
  if (window._allRows) renderRows(window._allRows);
});

// ── Polling ───────────────────────────────────────────────────────────────
async function refresh() {
  await loadRuns();
  try {
    const qs = _currentRun ? `?run=${encodeURIComponent(_currentRun)}` : '';
    const r = await fetch('/api/stats' + qs);
    const d = await r.json();
    update(d);
    setTimeout(refresh, d.done ? 10000 : 3000);
  } catch(e) {
    document.getElementById('updated').textContent = 'Reconnecting…';
    setTimeout(refresh, 3000);
  }
}
refresh();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _run_param(self) -> str | None:
        qs = parse_qs(urlparse(self.path).query)
        vals = qs.get("run")
        return vals[0] if vals else None

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/runs":
            self._json(_discover_runs())

        elif path == "/api/stats":
            log_path = _resolve_log(self._run_param())
            self._json(stats(log_path))

        elif m := _SUBJECT_RE.match(path):
            log_path = _resolve_log(self._run_param())
            data = subject_detail(m.group(1), log_path)
            if data:
                self._json(data)
            else:
                self._json({"error": "not found"}, status=404)

        elif path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live pipeline monitor")
    p.add_argument("--results-dir", type=str,
                   default=str(ROOT / "results" / "ucsf-pdgm"),
                   help="Directory containing runs/ subfolder and/or legacy llm_log*.jsonl files")
    p.add_argument("--meta-csv", type=str, default=None,
                   help="Metadata CSV for ground-truth labels (optional)")
    p.add_argument("--total", type=int, default=0,
                   help="Fallback total subject count when log has no header (0 = unknown)")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    _RESULTS_DIR = Path(args.results_dir)
    _TOTAL       = args.total

    default_meta = ROOT / "data" / "UCSF-PDGM" / "UCSF-PDGM-metadata_v2.csv"
    if args.meta_csv:
        _META_CSV = Path(args.meta_csv)
    elif default_meta.exists():
        _META_CSV = default_meta

    url = f"http://localhost:{args.port}"
    print(f"Monitor -> {url}  (Ctrl+C to stop)")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    HTTPServer(("", args.port), _Handler).serve_forever()
