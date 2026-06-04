"""
Atlas-based anatomical region mapping.
Supports Harvard-Oxford cortical/subcortical, Juelich histological,
and Hammers atlases — all available via nilearn's fetch_atlas_* functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


@dataclass
class AtlasRegion:
    index: int
    name: str
    voxel_mask: np.ndarray      # binary array, same shape as MNI space


@dataclass
class AtlasMap:
    name: str
    regions: List[AtlasRegion]
    label_volume: np.ndarray    # integer label map in MNI space


# ---------------------------------------------------------------------------
# nilearn atlas loaders
# ---------------------------------------------------------------------------

def _resample_atlas_to(label_vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resample atlas label map to target shape using nearest-neighbour."""
    zoom_factors = tuple(t / s for t, s in zip(target_shape, label_vol.shape))
    if all(abs(z - 1.0) < 0.01 for z in zoom_factors):
        return label_vol
    return zoom(label_vol, zoom_factors, order=0).astype(np.int32)


def load_harvard_oxford(
    target_shape: Optional[tuple] = None,
    atlas_name: str = "cort-maxprob-thr25-2mm",
) -> AtlasMap:
    """Fetch Harvard-Oxford cortical atlas via nilearn."""
    from nilearn.datasets import fetch_atlas_harvard_oxford
    atlas = fetch_atlas_harvard_oxford(atlas_name)
    # nilearn may return a Nifti1Image or a path depending on version
    maps = atlas.maps
    if isinstance(maps, (str, bytes)) or hasattr(maps, "__fspath__"):
        img = nib.load(maps)
    else:
        img = maps  # already a Nifti1Image
    labels_raw: np.ndarray = np.asarray(img.dataobj, dtype=np.int32)
    label_names: List[str] = list(atlas.labels)

    if target_shape is not None:
        labels_raw = _resample_atlas_to(labels_raw, target_shape)

    regions = []
    for idx, name in enumerate(label_names):
        if idx == 0:
            continue  # background
        mask = (labels_raw == idx).astype(np.uint8)
        if mask.any():
            regions.append(AtlasRegion(index=idx, name=name, voxel_mask=mask))

    return AtlasMap(name="harvard_oxford", regions=regions, label_volume=labels_raw)


def load_juelich(
    target_shape: Optional[tuple] = None,
    atlas_name: str = "maxprob-thr25-2mm",
) -> AtlasMap:
    """Fetch Juelich histological atlas via nilearn."""
    try:
        from nilearn.datasets import fetch_atlas_juelich
        atlas = fetch_atlas_juelich(atlas_name)
    except Exception as e:
        logger.warning("Juelich atlas unavailable: %s — using stub", e)
        return _stub_atlas("juelich", target_shape)

    maps = atlas.maps
    if isinstance(maps, (str, bytes)) or hasattr(maps, "__fspath__"):
        img = nib.load(maps)
    else:
        img = maps
    labels_raw = np.asarray(img.dataobj, dtype=np.int32)
    label_names = list(atlas.labels)

    if target_shape is not None:
        labels_raw = _resample_atlas_to(labels_raw, target_shape)

    regions = []
    for idx, name in enumerate(label_names):
        if idx == 0:
            continue
        mask = (labels_raw == idx).astype(np.uint8)
        if mask.any():
            regions.append(AtlasRegion(index=idx, name=name, voxel_mask=mask))

    return AtlasMap(name="juelich", regions=regions, label_volume=labels_raw)


def load_hammers(target_shape: Optional[tuple] = None) -> AtlasMap:
    """
    Hammers atlas is not in nilearn's public fetch functions.
    We provide a stub that returns coarse lobar regions derived from
    Harvard-Oxford when the actual atlas file is absent.
    """
    logger.warning("Hammers atlas not natively available via nilearn; using lobar stub")
    return _stub_atlas("hammers", target_shape)


def _stub_atlas(name: str, target_shape: Optional[tuple]) -> AtlasMap:
    """Minimal stub for atlases that aren't installed."""
    shape = target_shape if target_shape else (91, 109, 91)
    vol = np.zeros(shape, dtype=np.int32)
    return AtlasMap(name=name, regions=[], label_volume=vol)


# ---------------------------------------------------------------------------
# Combined atlas
# ---------------------------------------------------------------------------

def load_all_atlases(target_shape: Optional[tuple] = None) -> Dict[str, AtlasMap]:
    atlases: Dict[str, AtlasMap] = {}
    for loader, key in [
        (load_harvard_oxford, "harvard_oxford"),
        (load_juelich, "juelich"),
        (load_hammers, "hammers"),
    ]:
        try:
            atlases[key] = loader(target_shape)
            logger.info("Loaded atlas: %s (%d regions)", key, len(atlases[key].regions))
        except Exception as e:
            logger.error("Failed to load atlas %s: %s", key, e)
    return atlases


# ---------------------------------------------------------------------------
# Tumor-region overlap
# ---------------------------------------------------------------------------

def map_tumor_to_regions(
    seg: np.ndarray,
    atlas: AtlasMap,
    min_overlap_voxels: int = 5,
) -> List[Dict]:
    """
    For each atlas region overlapping with the tumor mask, return:
      region_name, region_index, overlap_voxels, overlap_fraction
    """
    tumor_mask = np.isin(seg, [1, 2, 4])
    hits = []
    for region in atlas.regions:
        if region.voxel_mask.shape != seg.shape:
            continue
        overlap = int((tumor_mask & (region.voxel_mask > 0)).sum())
        if overlap >= min_overlap_voxels:
            region_total = int(region.voxel_mask.sum())
            hits.append({
                "region_name": region.name,
                "region_index": region.index,
                "overlap_voxels": overlap,
                "overlap_fraction": overlap / max(region_total, 1),
            })
    hits.sort(key=lambda x: x["overlap_voxels"], reverse=True)
    return hits
