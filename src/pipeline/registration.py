"""
Brain registration to MNI152 standard space using ANTsPy.
Falls back to affine-only alignment when ANTs is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# MNI152 1mm isotropic template shape
MNI_SHAPE = (182, 218, 182)
MNI_VOXEL_SIZE = (1.0, 1.0, 1.0)


def _try_ants_import():
    try:
        import ants
        return ants
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# ANTs-based registration
# ---------------------------------------------------------------------------

def register_to_mni_ants(
    images: Dict[str, np.ndarray],
    affine: np.ndarray,
    template_path: Optional[str] = None,
    transform_type: str = "SyN",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Register all modalities to MNI152 using ANTs SyN (or Rigid/Affine).
    The T1 modality drives the registration; the computed transforms are
    then applied to all other modalities and to the segmentation mask.

    Returns (registered_images, mni_affine).
    """
    ants = _try_ants_import()
    if ants is None:
        raise ImportError("antspyx is required for ANTs registration")

    # Build ANTs image from the T1 (or first available modality)
    ref_mod = "t1" if "t1" in images else next(iter(images))
    vol = images[ref_mod].astype(np.float32)
    moving = ants.from_numpy(vol)

    # Load or create template
    if template_path and Path(template_path).exists():
        fixed = ants.image_read(template_path)
    else:
        logger.warning("MNI template not found; using synthetic 1mm isotropic reference")
        fixed = ants.from_numpy(np.zeros(MNI_SHAPE, dtype=np.float32))

    logger.info("Running ANTs %s registration for subject…", transform_type)
    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform_type,
        verbose=False,
    )
    fwdtransforms = result["fwdtransforms"]

    registered: Dict[str, np.ndarray] = {}
    for mod, vol in images.items():
        ants_vol = ants.from_numpy(vol.astype(np.float32))
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=ants_vol,
            transformlist=fwdtransforms,
            interpolator="linear",
        )
        registered[mod] = warped.numpy()

    # Return a simple diagonal affine for MNI space
    mni_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    return registered, mni_affine


def apply_transform_to_seg_ants(
    seg: np.ndarray,
    fwdtransforms: list,
    fixed_template,
) -> np.ndarray:
    """Apply previously computed ANTs transforms to a segmentation (nearest-neighbour)."""
    ants = _try_ants_import()
    if ants is None:
        raise ImportError("antspyx required")
    seg_ants = ants.from_numpy(seg.astype(np.float32))
    warped = ants.apply_transforms(
        fixed=fixed_template,
        moving=seg_ants,
        transformlist=fwdtransforms,
        interpolator="nearestNeighbor",
    )
    return warped.numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Affine-only fallback (resampling via scipy)
# ---------------------------------------------------------------------------

def resample_to_shape(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """Resample a volume to a target shape using zoom."""
    from scipy.ndimage import zoom
    zoom_factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume, zoom_factors, order=order).astype(np.float32)


def register_to_mni_affine(
    images: Dict[str, np.ndarray],
    affine: np.ndarray,
    target_shape: Tuple[int, int, int] = MNI_SHAPE,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Simple affine fallback: just resample each modality to MNI voxel grid.
    No nonlinear warping — suitable for testing or when ANTs is unavailable.
    """
    logger.warning("Using affine-only resampling (ANTs not available)")
    registered = {
        mod: resample_to_shape(vol, target_shape)
        for mod, vol in images.items()
    }
    mni_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    return registered, mni_affine


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def register_subject(
    images: Dict[str, np.ndarray],
    affine: np.ndarray,
    template_path: Optional[str] = None,
    method: str = "auto",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Register images to MNI152 space.
    method: 'ants' | 'affine' | 'auto' (try ANTs, fall back to affine)
    """
    if method == "ants":
        return register_to_mni_ants(images, affine, template_path)
    if method == "affine":
        return register_to_mni_affine(images, affine)

    # auto
    if _try_ants_import() is not None:
        try:
            return register_to_mni_ants(images, affine, template_path)
        except Exception as e:
            logger.warning("ANTs registration failed (%s); falling back to affine", e)
    return register_to_mni_affine(images, affine)
