"""
Intervention layer — modify feature JSONs for faithfulness evaluation.

Supported operations (Change 2):
  - null a single feature value (keep schema intact)
  - null all values in a feature group
  - keep only specified features (null the rest)
  - substitute a feature's value

All functions return a deep copy; the original is never mutated.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

def build_vocab_map(features: Dict) -> Dict[str, List[str]]:
    """
    Map each feature identifier to its path in the features dict.

    Identifiers use dot notation:
      label_stats compartments:  "wt.volume_mm3", "wt.t1.mean", "wt.enhancement_ratio"
      atlas regions:             "atlas.<atlas>.<Region_Name>.<key>.<stat>"

    Returns {identifier: path_list} where path_list is suitable for
    _get_by_path / _set_by_path.
    """
    result: Dict[str, List[str]] = {}

    label_stats = features.get("label_stats", {})
    for compartment, stats in label_stats.items():
        if not isinstance(stats, dict):
            # Direct scalar value at the compartment level (e.g. enhancement_ratio_global)
            result[compartment] = ["label_stats", compartment]
            continue
        for key, val in stats.items():
            if isinstance(val, dict):
                for stat_name in val:
                    ident = f"{compartment}.{key}.{stat_name}"
                    result[ident] = ["label_stats", compartment, key, stat_name]
            else:
                ident = f"{compartment}.{key}"
                result[ident] = ["label_stats", compartment, key]

    atlas_regions = features.get("atlas_regions", {})
    for atlas_name, regions in atlas_regions.items():
        if not isinstance(regions, dict):
            continue
        for region_name, region_stats in regions.items():
            if not isinstance(region_stats, dict):
                continue
            clean = region_name.replace(" ", "_")
            for key, val in region_stats.items():
                if isinstance(val, dict):
                    for stat_name in val:
                        ident = f"atlas.{atlas_name}.{clean}.{key}.{stat_name}"
                        result[ident] = [
                            "atlas_regions", atlas_name, region_name, key, stat_name,
                        ]
                else:
                    ident = f"atlas.{atlas_name}.{clean}.{key}"
                    result[ident] = ["atlas_regions", atlas_name, region_name, key]

    return result


def extract_feature_vocabulary(
    features: Dict,
    clinical: Optional[Dict] = None,
) -> List[str]:
    """Return sorted list of all feature identifier strings, including clinical.age/sex."""
    vocab = list(build_vocab_map(features).keys())
    if clinical:
        keys = {k.lower().strip() for k in clinical}
        if keys & {"age", "age at mri"}:
            vocab.append("clinical.age")
        if "sex" in keys:
            vocab.append("clinical.sex")
    return sorted(vocab)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_by_path(obj: Any, path: List[str]) -> Any:
    for key in path:
        obj = obj[key]
    return obj


def _set_by_path(obj: Any, path: List[str], value: Any) -> None:
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def _null_all_leaves(obj: Any) -> None:
    """Recursively set every non-dict leaf to None (in-place)."""
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if isinstance(obj[k], dict):
                _null_all_leaves(obj[k])
            else:
                obj[k] = None


# ---------------------------------------------------------------------------
# Intervention functions (all return deep copies)
# ---------------------------------------------------------------------------

def null_by_path(features: Dict, path: List[str]) -> Dict:
    """Return a deep copy with the value at *path* set to None."""
    modified = copy.deepcopy(features)
    try:
        _set_by_path(modified, path, None)
    except (KeyError, TypeError, IndexError):
        pass
    return modified


def null_group_by_path(features: Dict, group_path: List[str]) -> Dict:
    """Return a deep copy with all leaves under *group_path* set to None."""
    modified = copy.deepcopy(features)
    try:
        subtree = _get_by_path(modified, group_path)
        _null_all_leaves(subtree)
    except (KeyError, TypeError, IndexError):
        pass
    return modified


def keep_only_by_identifiers(
    features: Dict,
    identifiers: List[str],
    vocab_map: Dict[str, List[str]],
) -> Dict:
    """
    Return a deep copy with ALL leaves nulled EXCEPT those in *identifiers*.
    Useful for the Sufficiency faithfulness test.
    """
    modified = copy.deepcopy(features)
    _null_all_leaves(modified)
    keep_set = set(identifiers)
    for ident in keep_set:
        if ident in vocab_map:
            path = vocab_map[ident]
            try:
                val = _get_by_path(features, path)
                _set_by_path(modified, path, val)
            except (KeyError, TypeError, IndexError):
                pass
    return modified


def remove_by_identifiers(
    features: Dict,
    identifiers: List[str],
    vocab_map: Dict[str, List[str]],
) -> Dict:
    """
    Return a deep copy with the specified *identifiers* set to None.
    Useful for the Comprehensiveness faithfulness test.
    """
    modified = copy.deepcopy(features)
    for ident in identifiers:
        if ident in vocab_map:
            path = vocab_map[ident]
            try:
                _set_by_path(modified, path, None)
            except (KeyError, TypeError, IndexError):
                pass
    return modified


def substitute_by_identifier(
    features: Dict,
    identifier: str,
    new_value: Any,
    vocab_map: Dict[str, List[str]],
) -> Dict:
    """Return a deep copy with *identifier*'s value replaced by *new_value*."""
    if identifier not in vocab_map:
        return copy.deepcopy(features)
    modified = copy.deepcopy(features)
    path = vocab_map[identifier]
    try:
        _set_by_path(modified, path, new_value)
    except (KeyError, TypeError, IndexError):
        pass
    return modified


def substitute_by_path(
    features: Dict,
    path: List[str],
    new_value: Any,
) -> Dict:
    """Return a deep copy with the value at *path* replaced by *new_value*."""
    modified = copy.deepcopy(features)
    try:
        _set_by_path(modified, path, new_value)
    except (KeyError, TypeError, IndexError):
        pass
    return modified
