"""
Feature extraction: per-region, per-modality signal statistics.
Also computes enhancement ratios, sphericity, boundary sharpness, and semantic visual features.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .atlas_mapping import AtlasMap, AtlasRegion

logger = logging.getLogger(__name__)

MODALITIES = ("t1", "t1ce", "t2", "flair")

_DEEP_GRAY_KEYWORDS = ("thalamus", "caudate", "putamen", "pallidum", "amygdala", "accumbens")


def _stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0, "skewness": 0.0}
    return {
        "mean":     float(values.mean()),
        "std":      float(values.std()),
        "median":   float(np.median(values)),
        "p5":       float(np.percentile(values, 5)),
        "p95":      float(np.percentile(values, 95)),
        "skewness": float(_skewness(values)),
    }


def _skewness(v: np.ndarray) -> float:
    mu, sigma = v.mean(), v.std()
    if sigma < 1e-8:
        return 0.0
    return float(((v - mu) ** 3).mean() / sigma ** 3)


# ---------------------------------------------------------------------------
# Per-label (NCR/ED/ET) statistics — preserved for backward-compat with tests
# ---------------------------------------------------------------------------

def extract_label_features(
    images: Dict[str, np.ndarray],
    seg: np.ndarray,
    voxel_size_mm: tuple = (1.0, 1.0, 1.0),
) -> Dict:
    """Global stats per BraTS label (NCR=1, ED=2, ET=4) and whole-tumour."""
    label_map = {"ncr": 1, "ed": 2, "et": 4}
    vox_vol = float(np.prod(voxel_size_mm))
    result: Dict = {}

    for label_name, label_idx in label_map.items():
        mask = (seg == label_idx)
        result[label_name] = {"volume_mm3": float(mask.sum() * vox_vol)}
        for mod in MODALITIES:
            if mod not in images:
                continue
            result[label_name][mod] = _stats(images[mod][mask].astype(np.float64))

    wt_mask = np.isin(seg, [1, 2, 4])
    result["wt"] = {"volume_mm3": float(wt_mask.sum() * vox_vol)}
    for mod in MODALITIES:
        if mod not in images:
            continue
        result["wt"][mod] = _stats(images[mod][wt_mask].astype(np.float64))

    if "t1" in images and "t1ce" in images:
        t1_mean   = float(images["t1"][wt_mask].mean())   if wt_mask.any() else 0.0
        t1ce_mean = float(images["t1ce"][wt_mask].mean()) if wt_mask.any() else 0.0
        result["enhancement_ratio_global"] = t1ce_mean / (abs(t1_mean) + 1e-8)

    return result


# ---------------------------------------------------------------------------
# Per-region statistics — preserved for backward-compat
# ---------------------------------------------------------------------------

def extract_region_features(
    images: Dict[str, np.ndarray],
    seg: np.ndarray,
    region: AtlasRegion,
    seg_labels: tuple = (1, 2, 4),
) -> Dict:
    tumor_mask  = np.isin(seg, list(seg_labels))
    region_mask = region.voxel_mask.astype(bool)

    if region_mask.shape != tumor_mask.shape:
        return {}
    intersection = tumor_mask & region_mask
    if not intersection.any():
        return {}

    result: Dict = {
        "region_name":  region.name,
        "region_index": region.index,
        "tumor_voxels": int(intersection.sum()),
    }
    for mod in MODALITIES:
        if mod not in images:
            continue
        result[mod] = _stats(images[mod][intersection].astype(np.float64))

    if "t1ce" in images and "t1" in images:
        t1_vals   = images["t1"][intersection].astype(np.float64)
        t1ce_vals = images["t1ce"][intersection].astype(np.float64)
        result["enhancement_ratio"] = float(t1ce_vals.mean() / (abs(t1_vals.mean()) + 1e-8))

    if "flair" in images and "t2" in images:
        flair_vals = images["flair"][intersection].astype(np.float64)
        t2_vals    = images["t2"][intersection].astype(np.float64)
        result["flair_t2_ratio"] = float(flair_vals.mean() / (abs(t2_vals.mean()) + 1e-8))

    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dilate1(mask: np.ndarray) -> np.ndarray:
    """Dilate boolean mask by 1 voxel in each cardinal direction."""
    out = mask.copy()
    out[1:]        |= mask[:-1]
    out[:-1]       |= mask[1:]
    out[:, 1:]     |= mask[:, :-1]
    out[:, :-1]    |= mask[:, 1:]
    out[:, :, 1:]  |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _surface_area_mm2(mask: np.ndarray, voxel_size_mm: tuple) -> float:
    sx, sy, sz = voxel_size_mm
    area = 0.0
    for axis, face in enumerate([sy * sz, sx * sz, sx * sy]):
        slc1 = [slice(None)] * 3; slc1[axis] = slice(None, -1); s1 = tuple(slc1)
        slc2 = [slice(None)] * 3; slc2[axis] = slice(1, None);  s2 = tuple(slc2)
        area += float((mask[s1] & ~mask[s2]).sum() + (~mask[s1] & mask[s2]).sum()) * face
    return area


def _sphericity(mask: np.ndarray, voxel_size_mm: tuple) -> float:
    n = int(mask.sum())
    if n == 0:
        return 0.0
    vol  = n * float(np.prod(voxel_size_mm))
    area = _surface_area_mm2(mask, voxel_size_mm)
    if area < 1e-8:
        return 1.0
    return min(1.0, float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area))


def _boundary_sharpness(image: np.ndarray, mask: np.ndarray, voxel_size_mm: tuple) -> float:
    """Mean intensity gradient (per mm) across the tumor boundary."""
    total, count = 0.0, 0
    for axis, spacing in enumerate(voxel_size_mm):
        slc_a = [slice(None)] * 3; slc_a[axis] = slice(None, -1); sa = tuple(slc_a)
        slc_b = [slice(None)] * 3; slc_b[axis] = slice(1, None);  sb = tuple(slc_b)
        cross = mask[sa] ^ mask[sb]   # voxel pairs straddling the boundary
        if cross.any():
            diff = np.abs(image[sa][cross].astype(float) - image[sb][cross].astype(float)) / spacing
            total += float(diff.mean())
            count += 1
    return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Full feature set — returns the target JSON structure
# ---------------------------------------------------------------------------

def extract_all_features(
    images: Dict[str, np.ndarray],
    seg: np.ndarray,
    atlases: Dict[str, AtlasMap],
    voxel_size_mm: tuple = (1.0, 1.0, 1.0),
    max_regions_per_atlas: int = 20,
) -> Dict:
    """
    Extract all features and return them in the target JSON schema structure
    (semantic_visual + quantitative), ready for serialization and LLM consumption.
    """
    vox_vol = float(np.prod(voxel_size_mm))

    # ── Masks ────────────────────────────────────────────────────────────────
    ncr_mask  = (seg == 1)
    ed_mask   = (seg == 2)
    et_mask   = (seg == 4)
    wt_mask   = ncr_mask | ed_mask | et_mask
    core_mask = ncr_mask | et_mask   # NCR + ET = tumor core

    # ── Volumes (mL) ─────────────────────────────────────────────────────────
    ncr_vol = float(ncr_mask.sum()) * vox_vol / 1000
    ed_vol  = float(ed_mask.sum())  * vox_vol / 1000
    et_vol  = float(et_mask.sum())  * vox_vol / 1000
    wt_vol  = float(wt_mask.sum())  * vox_vol / 1000
    safe_wt = wt_vol if wt_vol > 1e-8 else 1e-8

    # ── Fractions ────────────────────────────────────────────────────────────
    enh_ratio = 0.0
    if "t1" in images and "t1ce" in images and et_mask.any():
        t1_et   = float(images["t1"][et_mask].mean())
        t1ce_et = float(images["t1ce"][et_mask].mean())
        enh_ratio = t1ce_et / (abs(t1_et) + 1e-8)

    # ── Semantic / visual features ───────────────────────────────────────────
    flair_core_mean = (
        float(images["flair"][core_mask].mean()) if "flair" in images and core_mask.any() else 0.0
    )
    flair_ed_mean = (
        float(images["flair"][ed_mask].mean()) if "flair" in images and ed_mask.any() else 0.0
    )
    flair_core_suppressed  = bool(flair_core_mean < flair_ed_mean)
    flair_rim_hyperintense = bool(flair_ed_mean > 0.5)

    hollowness = ncr_vol / safe_wt

    rim_core_adj = 0.0
    if core_mask.any() and ed_mask.any():
        adjacent     = _dilate1(core_mask) & ed_mask
        rim_core_adj = float(adjacent.sum()) / float(ed_mask.sum())

    # Mass effect: tumor CoM vs image midline
    midline_shift_mm    = 0.0
    dominant_hemisphere = "N/A"
    if wt_mask.any():
        cx         = float(np.argwhere(wt_mask)[:, 0].mean())
        midline_x  = seg.shape[0] / 2.0
        midline_shift_mm    = round(abs(cx - midline_x) * float(voxel_size_mm[0]), 2)
        dominant_hemisphere = "left" if cx < midline_x else "right"

    # ── Atlas features ───────────────────────────────────────────────────────
    location_features: List[Dict] = []
    deep_gray_nuclei: Dict = {}
    frontal_left = frontal_right = False

    for atlas_name, atlas in atlases.items():
        region_count = 0
        for region in atlas.regions:
            if region.voxel_mask.shape != seg.shape:
                continue
            region_mask = region.voxel_mask.astype(bool)
            overlap = int((wt_mask & region_mask).sum())
            if overlap < 5:
                continue
            region_total = int(region_mask.sum())
            frac         = round(overlap / max(region_total, 1), 4)
            rname        = region.name
            location_features.append({
                "region":           rname,
                "atlas":            atlas_name,
                "overlap_voxels":   overlap,
                "overlap_fraction": frac,
            })

            rname_lower = rname.lower()
            for kw in _DEEP_GRAY_KEYWORDS:
                if kw in rname_lower:
                    deep_gray_nuclei[rname] = {"overlap_voxels": overlap, "overlap_fraction": frac}
                    break

            if "frontal" in rname_lower:
                if "left"  in rname_lower: frontal_left  = True
                elif "right" in rname_lower: frontal_right = True
                else:
                    frontal_left = frontal_right = True

            region_count += 1
            if region_count >= max_regions_per_atlas:
                break

    location_features.sort(key=lambda x: x["overlap_voxels"], reverse=True)
    location_features = location_features[:10]

    # ── Sphericity ───────────────────────────────────────────────────────────
    sphericity = {
        name: round(_sphericity(mask, voxel_size_mm), 4)
        for name, mask in [("WT", wt_mask), ("ET", et_mask), ("NCR", ncr_mask)]
    }

    # ── Boundary sharpness ───────────────────────────────────────────────────
    sharpness: Dict = {}
    if wt_mask.any():
        for mod in MODALITIES:
            if mod in images:
                sharpness[mod] = round(_boundary_sharpness(images[mod], wt_mask, voxel_size_mm), 4)

    return {
        "semantic_visual": {
            "flair: tumor core suppressed": flair_core_suppressed,
            "flair: rim hyperintense":      flair_rim_hyperintense,
            "hollowness":                   round(hollowness, 4),
            "rim-core adjacency":           round(rim_core_adj, 4),
            "deep gray nuclei involvement": deep_gray_nuclei,
            "bilateral frontal involvement": frontal_left and frontal_right,
            "location features":            location_features,
            "mass effect metrics": {
                "midline_shift_mm":    midline_shift_mm,
                "dominant_hemisphere": dominant_hemisphere,
            },
        },
        "quantitative": {
            "volumes (ml)": {
                "NCR": round(ncr_vol, 3),
                "ED":  round(ed_vol,  3),
                "ET":  round(et_vol,  3),
                "WT":  round(wt_vol,  3),
            },
            "fractions": {
                "ET/WT":             round(et_vol  / safe_wt, 4),
                "NCR/WT":            round(ncr_vol / safe_wt, 4),
                "ED/WT":             round(ed_vol  / safe_wt, 4),
                "enhancement_ratio": round(enh_ratio, 4),
            },
            "boundary sharpness metrics (intensity/mm)": sharpness,
            "sphericity metrics (0-1)": sphericity,
        },
    }
