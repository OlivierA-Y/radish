"""
Serialization: convert extracted features → structured JSON for LLM consumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def to_json(
    subject_id: str,
    features: Dict,
    clinical: Optional[Dict] = None,
    indent: int = 2,
) -> str:
    payload = {
        "subject_id":    subject_id,
        "clinical_data": _format_clinical(clinical),
        "semantic_visual": features.get("semantic_visual", {}),
        "quantitative":    features.get("quantitative", {}),
    }
    return json.dumps(_round_floats(payload), indent=indent)


def save_json(json_str: str, path: str | Path) -> None:
    Path(path).write_text(json_str, encoding="utf-8")


def _format_clinical(clinical: Optional[Dict]) -> Dict:
    """Return only age and gender with standardized keys."""
    if not clinical:
        return {}
    result = {}
    for k, v in clinical.items():
        lower = k.lower().strip()
        if lower in ("sex", "gender"):
            result["gender"] = v
        elif lower in ("age at mri", "age"):
            result["age (years)"] = v
    return result


def _round_floats(obj: Any, decimals: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj
