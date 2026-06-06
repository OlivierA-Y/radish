"""
Serialization: convert extracted features → structured JSON + human-readable
text narrative for LLM consumption.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def _round_floats(obj: Any, decimals: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj


def to_json(
    subject_id: str,
    features: Dict,
    clinical: Optional[Dict] = None,
    indent: int = 2,
) -> str:
    payload = {
        "subject_id": subject_id,
        "imaging_features": _round_floats(features),
        "clinical": _filter_clinical(clinical),
    }
    return json.dumps(payload, indent=indent)


def save_json(json_str: str, path: str | Path) -> None:
    Path(path).write_text(json_str, encoding="utf-8")


# ---------------------------------------------------------------------------
# Text narrative (what gets inserted into the LLM prompt)
# ---------------------------------------------------------------------------

def _filter_clinical(clinical: Optional[Dict]) -> Dict:
    """Return only sex and age from a clinical dict, with normalized keys."""
    if not clinical:
        return {}
    result = {}
    for k, v in clinical.items():
        lower = k.lower().strip()
        if lower == "sex":
            result["sex"] = v
        elif lower in ("age at mri", "age"):
            result["age"] = v
    return result


def _fmt_stats(stats: Dict[str, float]) -> str:
    parts = []
    if "mean" in stats:
        parts.append(f"mean={stats['mean']:.3f}")
    if "std" in stats:
        parts.append(f"std={stats['std']:.3f}")
    if "p5" in stats and "p95" in stats:
        parts.append(f"range=[{stats['p5']:.3f}–{stats['p95']:.3f}]")
    return ", ".join(parts)


def to_narrative(
    subject_id: str,
    features: Dict,
    clinical: Optional[Dict] = None,
) -> str:
    """
    Convert feature dict to a radiology-style plain-text report for the LLM.
    """
    lines = [
        f"=== MRI Feature Report — Subject {subject_id} ===",
        "",
        "## Tumour Compartment Volumes & Signal Statistics",
    ]

    label_stats = features.get("label_stats", {})
    label_names = {
        "wt":  "Whole Tumour",
        "et":  "Enhancing Tumour (ET)",
        "ncr": "Necrotic Core (NCR)",
        "ed":  "Peritumoral Oedema (ED)",
    }
    for key, display in label_names.items():
        info = label_stats.get(key)
        if not info:
            continue
        vol = info.get("volume_mm3", 0)
        lines.append(f"\n### {display}")
        lines.append(f"  Volume: {vol:.1f} mm³")
        for mod in ("t1", "t1ce", "t2", "flair"):
            if mod in info:
                lines.append(f"  {mod.upper()}: {_fmt_stats(info[mod])}")

    er = label_stats.get("enhancement_ratio_global")
    if er is not None:
        lines.append(f"\n  Enhancement ratio (T1-CE / T1): {er:.3f}")

    # Atlas region involvement
    atlas_regions = features.get("atlas_regions", {})
    if atlas_regions:
        lines.append("\n## Anatomical Region Involvement (Atlas Mapping)")
        for atlas_name, region_list in atlas_regions.items():
            if not region_list:
                continue
            lines.append(f"\n### {atlas_name.replace('_', ' ').title()} Atlas")
            for r in region_list[:10]:   # cap at top-10 for prompt length
                name = r.get("region_name", "Unknown")
                vox  = r.get("tumor_voxels", 0)
                er_r = r.get("enhancement_ratio")
                er_str = f"  enhancement_ratio={er_r:.3f}" if er_r else ""
                lines.append(f"  - {name}: {vox} tumour voxels{er_str}")

    # Clinical metadata (sex and age only)
    clin_filtered = _filter_clinical(clinical)
    if clin_filtered:
        lines.append("\n## Clinical Information")
        for k, v in clin_filtered.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)
