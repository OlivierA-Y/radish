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
import math
import re
import sqlite3
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
# SQL query
# ---------------------------------------------------------------------------

def _build_sqlite(entries: list[dict], gt: dict[str, int]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE predictions (
            subject_id    TEXT,
            idh_status    TEXT,
            latency_s     REAL,
            correct       INTEGER,
            model_version TEXT
        )
    """)
    for p in entries:
        parsed = p.get("parsed", {})
        sid    = p.get("subject_id", "")
        pred_l = 1 if "mutant" in parsed.get("idh_status", "").lower() else 0
        gt_l   = gt.get(sid)
        correct = (1 if pred_l == gt_l else 0) if gt_l is not None else None
        conn.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?)",
            (
                sid,
                parsed.get("idh_status", ""),
                round(p.get("latency_s", 0), 2),
                correct,
                p.get("model_version", ""),
            ),
        )
    conn.commit()
    return conn


def query_predictions(sql: str, log_path: Path) -> tuple[list[dict], str | None]:
    """Execute a SELECT query against predictions. Returns (rows, error)."""
    if not re.match(r"(?i)^\s*select\b", sql.strip()):
        return [], "Only SELECT queries are allowed."
    gt      = _load_ground_truth()
    entries = _read_log(log_path)
    conn    = _build_sqlite(entries, gt)
    try:
        cur  = conn.execute(sql.strip())
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return rows, None
    except sqlite3.Error as exc:
        return [], str(exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

_WEB_DIR = ROOT / "web"


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def _sanitize(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _sanitize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_sanitize(x) for x in v]
    return v


class _Handler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        body = json.dumps(_sanitize(data), ensure_ascii=False, default=str).encode()
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
            body = (_WEB_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/query":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._json({"error": "Invalid JSON"}, status=400)
                return
            sql = (data.get("sql") or "").strip()
            if not sql:
                self._json({"error": "Missing 'sql' field"}, status=400)
                return
            log_path = _resolve_log(data.get("run"))
            rows, error = query_predictions(sql, log_path)
            if error:
                self._json({"error": error}, status=400)
            else:
                cols = list(rows[0].keys()) if rows else []
                self._json({"rows": rows, "columns": cols})
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
