"""
MRI preprocessing: loading, intensity normalization, skull stripping.
Handles 4-modality input: T1, T1-CE, T2, FLAIR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter

logger = logging.getLogger(__name__)

MODALITIES = ("t1", "t1ce", "t2", "flair")


@dataclass
class MRISubject:
    subject_id: str
    images: Dict[str, np.ndarray]        # modality -> 3-D float32 array
    affine: np.ndarray                   # voxel-to-world matrix (first loaded modality)
    header: nib.Nifti1Header
    voxel_size_mm: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 1.0, 1.0))
    brain_mask: Optional[np.ndarray] = None
    idh_label: Optional[int] = None      # 1 = mutant, 0 = wildtype, None = unknown


def load_subject(
    subject_id: str,
    modality_paths: Dict[str, Path],
    idh_label: Optional[int] = None,
) -> MRISubject:
    """Load multi-modal NIfTI volumes for one subject."""
    images: Dict[str, np.ndarray] = {}
    affine = header = voxel_size = None

    for mod in MODALITIES:
        path = modality_paths.get(mod)
        if path is None:
            logger.debug("Subject %s: modality %s not provided", subject_id, mod)
            continue
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        images[mod] = data
        if affine is None:
            affine = img.affine
            header = img.header
            voxel_size = tuple(float(v) for v in img.header.get_zooms()[:3])

    if not images:
        raise ValueError(f"No valid MRI files loaded for subject {subject_id}")

    return MRISubject(
        subject_id=subject_id,
        images=images,
        affine=affine,
        header=header,
        voxel_size_mm=voxel_size,
        idh_label=idh_label,
    )


def normalize_intensity(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "zscore",
) -> np.ndarray:
    """
    Intensity normalization inside a brain mask.
    method: 'zscore' | 'minmax' | 'percentile'
    """
    vol = volume.astype(np.float32)
    roi = vol[mask > 0] if mask is not None and mask.any() else vol[vol > 0]

    if method == "zscore":
        mu, sigma = roi.mean(), roi.std()
        if sigma < 1e-8:
            return vol
        out = (vol - mu) / sigma
        if mask is not None:
            out[mask == 0] = 0.0
        return out

    if method == "minmax":
        lo, hi = roi.min(), roi.max()
        if hi - lo < 1e-8:
            return vol
        out = (vol - lo) / (hi - lo)
        if mask is not None:
            out[mask == 0] = 0.0
        return out

    if method == "percentile":
        lo, hi = np.percentile(roi, 1), np.percentile(roi, 99)
        out = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)
        if mask is not None:
            out[mask == 0] = 0.0
        return out

    raise ValueError(f"Unknown normalization method: {method}")


def simple_skull_strip(volume: np.ndarray, threshold_percentile: float = 15.0) -> np.ndarray:
    """
    Otsu-style brain mask via intensity threshold + morphological cleanup.
    Used when no dedicated skull-stripping tool is available.
    Returns a binary mask (uint8).
    """
    thresh = np.percentile(volume[volume > 0], threshold_percentile)
    mask = (volume > thresh).astype(np.uint8)
    # Fill internal holes slice-by-slice
    for z in range(mask.shape[2]):
        mask[:, :, z] = binary_fill_holes(mask[:, :, z]).astype(np.uint8)
    # Remove small spurious blobs
    mask = binary_opening(mask, iterations=3).astype(np.uint8)
    return mask


def preprocess_subject(
    subject: MRISubject,
    norm_method: str = "percentile",
    smooth_fwhm_mm: float = 0.0,
) -> MRISubject:
    """
    Full preprocessing pass:
    1. Skull-strip (using T1 if available, otherwise first modality)
    2. Intensity normalize each modality within the brain mask
    3. Optional Gaussian smoothing
    Returns the same subject with images replaced by preprocessed versions.
    """
    ref_mod = "t1" if "t1" in subject.images else next(iter(subject.images))
    brain_mask = simple_skull_strip(subject.images[ref_mod])
    subject.brain_mask = brain_mask

    sigma = smooth_fwhm_mm / (2.35 * np.mean(subject.voxel_size_mm)) if smooth_fwhm_mm > 0 else 0.0

    processed: Dict[str, np.ndarray] = {}
    for mod, vol in subject.images.items():
        v = normalize_intensity(vol, mask=brain_mask, method=norm_method)
        if sigma > 0:
            v = gaussian_filter(v.astype(np.float64), sigma=sigma).astype(np.float32)
        processed[mod] = v

    subject.images = processed
    return subject
