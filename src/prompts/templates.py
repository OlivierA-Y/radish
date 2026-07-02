"""
LLM prompt templates for zero-shot IDH mutation prediction.
Mirrors the prompt structure described in the paper.
"""

import json
from typing import Optional


SYSTEM_PROMPT = (
    "You are an experienced brain radiologist with medical expertise in "
    "neuro-oncology, radiogenomics, and IDH genotyping."
)

USER_PROMPT_TEMPLATE = (
    "You are an experienced radiologist entasked with discriminating a brain "
    "glioma as either 'IDH mutant' or 'IDH wildtype'. You are presented a json "
    "file encapsulating semantic (visual) attributes and quantitative metrics "
    "about a brain tumor (glioma) - extracted from 3D multiparametric MRI "
    "sequences (FLAIR, T1-contrast enhanced, and T2-weighted) and co-registered "
    "3D segmentation map of tumor subregions. Note that we do not have "
    "information on the necrosis component of the tumor. Provide a compact "
    "response with compact reasoning."
)


def build_prompt(features_dict: dict, clinical_dict: Optional[dict] = None) -> tuple:
    """Return (system_prompt, user_prompt) with features and optional clinical data."""
    payload: dict = {}
    if clinical_dict:
        clinical_data = {}
        for k, v in clinical_dict.items():
            lower = k.lower().strip()
            if lower in ("sex", "gender"):
                clinical_data["gender"] = v
            elif lower in ("age at mri", "age"):
                clinical_data["age (years)"] = v
        if clinical_data:
            payload["clinical_data"] = clinical_data
    payload.update(features_dict)
    user_prompt = USER_PROMPT_TEMPLATE + f"\n\nHere is the json data:\n{json.dumps(payload, indent=4)}"
    return SYSTEM_PROMPT, user_prompt
