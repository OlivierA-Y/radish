"""
Generate a formal clinical report for a subject from an llm_log.jsonl entry.

The log entry already contains the MRI narrative, the IDH prediction, and
structured rationale (findings, impression, decisive/supporting features).
This script passes all of that context to an LLM and asks it to produce a
polished radiology-style report.

Usage:
    python report.py SUBJECT_ID LOG_PATH
    python report.py UCSF-PDGM-004 results/ucsf-pdgm/llm_log_20260605_094038.jsonl
    python report.py UCSF-PDGM-004 results/ucsf-pdgm/runs/20260611_222743/llm_log.jsonl
    python report.py UCSF-PDGM-004 results/ucsf-pdgm/llm_log.jsonl --output report.txt
    python report.py UCSF-PDGM-004 results/ucsf-pdgm/llm_log.jsonl --model anthropic/claude-opus-4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _find_entry(log_path: Path, subject_id: str) -> Optional[dict]:
    """Return the last log entry whose subject_id matches (skips run_header)."""
    found: Optional[dict] = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "run_header":
                continue
            if entry.get("subject_id") == subject_id:
                found = entry   # keep last match in case of duplicates
    return found


def _extract_narrative(user_prompt: str) -> str:
    """Extract the JSON features block from the user prompt."""
    start = user_prompt.find("{")
    end   = user_prompt.rfind("}")
    if start != -1 and end != -1:
        return user_prompt[start:end + 1].strip()
    return user_prompt.strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_REPORT_SYSTEM_PROMPT = """\
You are a senior neuroradiologist writing a formal clinical report for a brain \
glioma case. You have access to MRI-derived quantitative imaging features and an \
AI-generated IDH mutation status prediction with supporting rationale. Produce a \
concise, professional radiology report that integrates the imaging findings with \
the AI analysis. Use standard radiology report language and structure."""

_REPORT_USER_TEMPLATE = """\
Write a formal radiology report for the case below.

## Subject: {subject_id}

## MRI Feature Report
{narrative}

## AI Prediction
- IDH Status: {idh_status}
- Reasoning:  {reasoning}

## Format
Produce the report in exactly these five sections (use the exact headings):

CLINICAL INDICATION

TECHNIQUE

FINDINGS

IMPRESSION

AI ANALYSIS NOTE

Rules:
- FINDINGS: cite key quantitative values from the MRI feature report \
(volumes in cm³, mean signal intensities). One paragraph.
- IMPRESSION: state the IDH prediction in one sentence, then give a brief \
differential in one sentence.
- AI ANALYSIS NOTE: one sentence disclosing AI assistance.
- Do not invent information not present in the feature report.
- Do not output JSON or markdown code fences."""


def _build_prompts(entry: dict) -> tuple[str, str]:
    parsed = entry.get("parsed", {})
    user_prompt = _REPORT_USER_TEMPLATE.format(
        subject_id=entry.get("subject_id", "UNKNOWN"),
        narrative=_extract_narrative(entry.get("user_prompt", "")),
        idh_status=parsed.get("idh_status", "unknown"),
        reasoning=parsed.get("reasoning", ""),
    )
    return _REPORT_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _get_client(model: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from src.pipeline.llm_predictor import (
        _load_env_key, _resolve_model, OPENROUTER_BASE_URL,
    )
    from openai import OpenAI

    model_id = _resolve_model(model)
    key = _load_env_key("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to the environment or a .env file."
        )
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
        default_headers={
            "HTTP-Referer": "https://github.com/idh-llm-pipeline",
            "X-Title":      "IDH-LLM-Pipeline",
        },
    )
    return client, model_id


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    subject_id: str,
    log_path: Path,
    model: str = "anthropic/claude-opus-4",
    output: Optional[Path] = None,
) -> str:
    entry = _find_entry(log_path, subject_id)
    if entry is None:
        raise ValueError(
            f"Subject '{subject_id}' not found in {log_path}.\n"
            "Check the subject ID with: python monitor.py"
        )

    sys_p, usr_p = _build_prompts(entry)
    client, model_id = _get_client(model)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user",   "content": usr_p},
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    body = (response.choices[0].message.content or "").strip()

    header = (
        f"{'=' * 60}\n"
        f"  CLINICAL AI REPORT — IDH MUTATION STATUS ANALYSIS\n"
        f"{'=' * 60}\n"
        f"  Subject:      {subject_id}\n"
        f"  Source log:   {log_path.name}\n"
        f"  Original run: {entry.get('timestamp', 'unknown')}\n"
        f"  Report model: {model_id}\n"
        f"  Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 60}\n\n"
    )
    full_report = header + body + "\n"

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(full_report, encoding="utf-8")
        print(f"Report written to: {output}", file=sys.stderr)
    else:
        print(full_report)

    return full_report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a clinical report for a subject from an llm_log.jsonl entry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("subject_id", help="Subject ID (e.g. UCSF-PDGM-004)")
    p.add_argument("log_path",   type=Path, help="Path to llm_log.jsonl")
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write report to file instead of stdout",
    )
    p.add_argument(
        "--model", default="anthropic/claude-opus-4",
        help="OpenRouter model ID or alias (default: anthropic/claude-opus-4)",
    )
    args = p.parse_args()

    if not args.log_path.exists():
        p.error(f"Log file not found: {args.log_path}")

    try:
        generate_report(
            subject_id=args.subject_id,
            log_path=args.log_path,
            model=args.model,
            output=args.output,
        )
    except (ValueError, EnvironmentError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
