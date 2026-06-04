"""
Tumor segmentation module.

Primary path: BrainSegFounder (nnU-Net-based, BraTS-trained).
Fallback:     threshold + connected-component heuristic for synthetic/test data.

Segmentation labels follow BraTS convention:
  1 = necrotic core (NCR)
  2 = peritumoral edema (ED)
  4 = enhancing tumor (ET)
  Combined whole-tumor (WT) = labels {1, 2, 4}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import label as cc_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TumorSegmenter:
    def segment(
        self,
        images: Dict[str, np.ndarray],
        affine: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# BrainSegFounder (MONAI / nnU-Net wrapper)
# ---------------------------------------------------------------------------

class BrainSegFounderSegmenter(TumorSegmenter):
    """
    Calls BrainSegFounder via MONAI's sliding-window inference.
    Expects a MONAI-compatible checkpoint downloaded from the model zoo.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._model = None

    def _load_model(self):
        try:
            import torch
            from monai.networks.nets import SwinUNETR

            model = SwinUNETR(
                img_size=(128, 128, 128),
                in_channels=4,
                out_channels=4,
                feature_size=48,
                use_checkpoint=True,
            )
            state = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state.get("state_dict", state))
            model.eval()
            self._model = model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load BrainSegFounder: {e}") from e

    def segment(
        self,
        images: Dict[str, np.ndarray],
        affine: np.ndarray,
    ) -> np.ndarray:
        import torch
        from monai.inferers import sliding_window_inference

        if self._model is None:
            self._load_model()

        channel_order = ["t1", "t1ce", "t2", "flair"]
        channels = []
        for mod in channel_order:
            vol = images.get(mod, np.zeros_like(next(iter(images.values()))))
            channels.append(vol)

        tensor = torch.tensor(np.stack(channels)[np.newaxis], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = sliding_window_inference(
                tensor,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=self._model,
                overlap=0.5,
            )
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # Remap class indices to BraTS labels: 0→0, 1→1(NCR), 2→2(ED), 3→4(ET)
        remap = np.array([0, 1, 2, 4], dtype=np.uint8)
        return remap[pred]


# ---------------------------------------------------------------------------
# Threshold-based fallback (no deep-learning dependency)
# ---------------------------------------------------------------------------

class ThresholdSegmenter(TumorSegmenter):
    """
    Simple heuristic segmenter for synthetic/test data.
    Identifies hyper-intense voxels in FLAIR/T2 as edema and hyper-intense
    voxels in T1-CE as enhancing tumor.
    """

    def __init__(
        self,
        flair_threshold_std: float = 2.0,
        t1ce_threshold_std: float = 2.5,
        min_component_voxels: int = 50,
    ):
        self.flair_thr = flair_threshold_std
        self.t1ce_thr = t1ce_threshold_std
        self.min_vox = min_component_voxels

    def _largest_component(self, binary: np.ndarray) -> np.ndarray:
        labeled, n = cc_label(binary)
        if n == 0:
            return binary
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        largest = np.argmax(sizes) + 1
        return (labeled == largest).astype(np.uint8)

    def segment(
        self,
        images: Dict[str, np.ndarray],
        affine: np.ndarray,
    ) -> np.ndarray:
        shape = next(iter(images.values())).shape
        seg = np.zeros(shape, dtype=np.uint8)

        # Edema: hyper-intense in FLAIR
        if "flair" in images:
            flair = images["flair"]
            brain_vox = flair[flair != 0]
            mu, sigma = brain_vox.mean(), brain_vox.std()
            ed_mask = (flair > mu + self.flair_thr * sigma).astype(np.uint8)
            seg[ed_mask > 0] = 2  # edema

        # Enhancing tumor: hyper-intense in T1-CE
        if "t1ce" in images:
            t1ce = images["t1ce"]
            brain_vox = t1ce[t1ce != 0]
            mu, sigma = brain_vox.mean(), brain_vox.std()
            et_mask = (t1ce > mu + self.t1ce_thr * sigma).astype(np.uint8)
            seg[et_mask > 0] = 4  # enhancing tumor

        # Necrotic core: low-intensity inside the ET region
        if "t1" in images and seg.any():
            t1 = images["t1"]
            et_region = seg == 4
            if et_region.any():
                et_vals = t1[et_region]
                ncr_thresh = np.percentile(et_vals, 20)
                ncr_mask = et_region & (t1 < ncr_thresh)
                seg[ncr_mask] = 1

        return seg


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_segmenter(
    method: str = "threshold",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> TumorSegmenter:
    if method == "brainsegfounder":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path required for BrainSegFounder")
        return BrainSegFounderSegmenter(checkpoint_path, device)
    if method == "threshold":
        return ThresholdSegmenter()
    raise ValueError(f"Unknown segmentation method: {method}")


def compute_tumor_volumes(
    seg: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
) -> Dict[str, float]:
    """Compute per-label and total volumes in mm³."""
    vox_vol = float(np.prod(voxel_size_mm))
    return {
        "ncr_mm3": float((seg == 1).sum() * vox_vol),
        "ed_mm3":  float((seg == 2).sum() * vox_vol),
        "et_mm3":  float((seg == 4).sum() * vox_vol),
        "wt_mm3":  float(np.isin(seg, [1, 2, 4]).sum() * vox_vol),
    }
