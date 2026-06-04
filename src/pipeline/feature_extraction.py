"""
Feature extraction: per-region, per-modality signal statistics.
Also computes enhancement ratios (T1-CE / T1) and FLAIR/T2 ratios.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .atlas_mapping import AtlasMap, AtlasRegion

logger = logging.getLogger(__name__)

MODALITIES = ("t1", "t1ce", "t2", "flair")


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
# Per-region statistics
# ---------------------------------------------------------------------------

def extract_region_features(
    images: Dict[str, np.ndarray],
    seg: np.ndarray,
    region: AtlasRegion,
    seg_labels: tuple = (1, 2, 4),
) -> Dict:
    """
    Compute signal statistics for all modalities within the intersection of
    a given atlas region and the tumor segmentation mask.
    """
    tumor_mask = np.isin(seg, list(seg_labels))
    region_mask = region.voxel_mask.astype(bool)

    # guard shape mismatch
    if region_mask.shape != tumor_mask.shape:
        return {}

    intersection = tumor_mask & region_mask

    if not intersection.any():
        return {}

    result: Dict = {
        "region_name":       region.name,
        "region_index":      region.index,
        "tumor_voxels":      int(intersection.sum()),
    }

    for mod in MODALITIES:
        if mod not in images:
            continue
        vol = images[mod]
        vals = vol[intersection].astype(np.float64)
        result[mod] = _stats(vals)

    # Enhancement ratio: T1-CE / T1 (proxy for BBB breakdown)
    if "t1ce" in images and "t1" in images:
        t1_vals  = images["t1"][intersection].astype(np.float64)
        t1ce_vals = images["t1ce"][intersection].astype(np.float64)
        denom = np.abs(t1_vals.mean()) + 1e-8
        result["enhancement_ratio"] = float(t1ce_vals.mean() / denom)

    # FLAIR/T2 ratio (edema characterisation)
    if "flair" in images and "t2" in images:
        flair_vals = images["flair"][intersection].astype(np.float64)
        t2_vals    = images["t2"][intersection].astype(np.float64)
        denom = np.abs(t2_vals.mean()) + 1e-8
        result["flair_t2_ratio"] = float(flair_vals.mean() / denom)

    return result


# ---------------------------------------------------------------------------
# Per-label (NCR/ED/ET) statistics — global, no atlas
# ---------------------------------------------------------------------------

def extract_label_features(
    images: Dict[str, np.ndarray],
    seg: np.ndarray,
    voxel_size_mm: tuple = (1.0, 1.0, 1.0),
) -> Dict:
    """
    Global statistics per BraTS label (NCR=1, ED=2, ET=4) and whole-tumour.
    """
    label_map = {
        "ncr": 1,   # necrotic core
        "ed":  2,   # peritumoral edema
        "et":  4,   # enhancing tumour
    }
    vox_vol = float(np.prod(voxel_size_mm))
    result: Dict = {}

    for label_name, label_idx in label_map.items():
        mask = (seg == label_idx)
        result[label_name] = {"volume_mm3": float(mask.sum() * vox_vol)}
        for mod in MODALITIES:
            if mod not in images:
                continue
            vals = images[mod][mask].astype(np.float64)
            result[label_name][mod] = _stats(vals)

    # Whole tumour
    wt_mask = np.isin(seg, [1, 2, 4])
    result["wt"] = {"volume_mm3": float(wt_mask.sum() * vox_vol)}
    for mod in MODALITIES:
        if mod not in images:
            continue
        vals = images[mod][wt_mask].astype(np.float64)
        result["wt"][mod] = _stats(vals)

    # Enhancement ratio for the whole tumour
    if "t1" in images and "t1ce" in images:
        t1_mean   = float(images["t1"][wt_mask].mean())  if wt_mask.any() else 0.0
        t1ce_mean = float(images["t1ce"][wt_mask].mean()) if wt_mask.any() else 0.0
        result["enhancement_ratio_global"] = t1ce_mean / (abs(t1_mean) + 1e-8)

    return result


# ---------------------------------------------------------------------------
# Full feature set
# ---------------------------------------------------------------------------

def extract_all_features(
    images: Dict[str, np.ndarray],
    seg: np.ndarray,
    atlases: Dict[str, AtlasMap],
    voxel_size_mm: tuple = (1.0, 1.0, 1.0),
    max_regions_per_atlas: int = 20,
) -> Dict:
    """
    Combine label-level and atlas-region features into a single dict
    ready for serialisation.
    """
    features: Dict = {
        "label_stats": extract_label_features(images, seg, voxel_size_mm),
        "atlas_regions": {},
    }

    for atlas_name, atlas in atlases.items():
        region_features: List[Dict] = []
        for region in atlas.regions:
            rf = extract_region_features(images, seg, region)
            if rf:
                region_features.append(rf)
            if len(region_features) >= max_regions_per_atlas:
                break
        features["atlas_regions"][atlas_name] = region_features
        logger.debug(
            "Atlas %s: %d regions with tumour overlap",
            atlas_name,
            len(region_features),
        )

    return features
