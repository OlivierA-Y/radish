"""
Tests for the UCSF-PDGM loader.
All tests use synthetic fixtures — no real NIfTI files required.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import load_ucsf_pdgm, _parse_ucsf_idh, _ucsf_dir_id


# ── IDH label parsing ─────────────────────────────────────────────────────────

def test_parse_wildtype():
    assert _parse_ucsf_idh("wildtype") == 0

def test_parse_mutated_nos():
    assert _parse_ucsf_idh("mutated (NOS)") == 1

def test_parse_mutated_nos_lowercase():
    assert _parse_ucsf_idh("mutated (nos)") == 1

def test_parse_mutant_alias():
    assert _parse_ucsf_idh("mutant") == 1

def test_parse_mutated_alias():
    assert _parse_ucsf_idh("mutated") == 1

def test_parse_unknown_returns_none():
    assert _parse_ucsf_idh("") is None
    assert _parse_ucsf_idh("unknown") is None
    assert _parse_ucsf_idh("indeterminate") is None

def test_parse_strips_whitespace():
    assert _parse_ucsf_idh("  wildtype  ") == 0
    assert _parse_ucsf_idh("  mutated (NOS) ") == 1


# ── ID mapping ────────────────────────────────────────────────────────────────

def test_ucsf_dir_id_zero_pads():
    assert _ucsf_dir_id("UCSF-PDGM-004") == "UCSF-PDGM-0004"
    assert _ucsf_dir_id("UCSF-PDGM-021") == "UCSF-PDGM-0021"
    assert _ucsf_dir_id("UCSF-PDGM-100") == "UCSF-PDGM-0100"

def test_ucsf_dir_id_already_padded():
    assert _ucsf_dir_id("UCSF-PDGM-0004") == "UCSF-PDGM-0004"


# ── Fixture factory ───────────────────────────────────────────────────────────

def _make_ucsf_tree(tmp_path: Path, subjects: list[dict]) -> Path:
    """
    Build a UCSF-PDGM directory tree mirroring the actual downloaded layout.
      tmp_path/
        UCSF-PDGM-metadata_v2.csv          (3-digit IDs)
        UCSF-PDGM-v3/
          UCSF-PDGM-0004_nifti/            (4-digit + _nifti)
            UCSF-PDGM-0004_T1c.nii.gz
            UCSF-PDGM-0004_T2.nii.gz
            UCSF-PDGM-0004_FLAIR.nii.gz
            UCSF-PDGM-0004_tumor_segmentation.nii.gz
    subjects: list of dicts with keys: id (3-digit), idh, make_files (bool)
    """
    subjects_dir = tmp_path / "UCSF-PDGM-v3"
    subjects_dir.mkdir()

    fieldnames = ["ID", "Sex", "Age at MRI", "WHO CNS Grade",
                  "Final pathologic diagnosis (WHO 2021)",
                  "MGMT status", "MGMT index", "1p/19q", "IDH",
                  "1-dead 0-alive", "OS", "EOR", "BraTS21 ID"]
    with open(tmp_path / "UCSF-PDGM-metadata_v2.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in subjects:
            writer.writerow({
                "ID": s["id"],
                "Sex": "M",
                "Age at MRI": "50",
                "WHO CNS Grade": "4",
                "Final pathologic diagnosis (WHO 2021)": "Glioblastoma",
                "MGMT status": "negative",
                "IDH": s["idh"],
            })

    for s in subjects:
        if not s.get("make_files", True):
            continue
        sid   = s["id"]
        fsid  = _ucsf_dir_id(sid)           # 4-digit zero-padded ID
        subj_dir = subjects_dir / f"{fsid}_nifti"
        subj_dir.mkdir()
        for suffix in ("_T1c.nii.gz", "_T2.nii.gz", "_FLAIR.nii.gz",
                       "_tumor_segmentation.nii.gz"):
            (subj_dir / f"{fsid}{suffix}").write_bytes(b"")

    return tmp_path


# ── load_ucsf_pdgm ────────────────────────────────────────────────────────────

def test_loads_correct_subject_count(tmp_path):
    _make_ucsf_tree(tmp_path, [
        {"id": "UCSF-PDGM-004", "idh": "wildtype"},
        {"id": "UCSF-PDGM-021", "idh": "mutated (NOS)"},
    ])
    records = load_ucsf_pdgm(tmp_path)
    assert len(records) == 2


def test_idh_labels_parsed_correctly(tmp_path):
    _make_ucsf_tree(tmp_path, [
        {"id": "UCSF-PDGM-004", "idh": "wildtype"},
        {"id": "UCSF-PDGM-021", "idh": "mutated (NOS)"},
    ])
    records = load_ucsf_pdgm(tmp_path)
    by_id = {r.subject_id: r for r in records}
    assert by_id["UCSF-PDGM-004"].idh_label == 0
    assert by_id["UCSF-PDGM-021"].idh_label == 1


def test_modalities_found(tmp_path):
    _make_ucsf_tree(tmp_path, [{"id": "UCSF-PDGM-004", "idh": "wildtype"}])
    records = load_ucsf_pdgm(tmp_path)
    assert len(records) == 1
    mods = records[0].modality_paths
    # UCSF-PDGM only provides T1c — non-contrast T1 is absent from this dataset
    for mod in ("t1ce", "t2", "flair"):
        assert mod in mods, f"Missing modality: {mod}"
        assert mods[mod].exists()
    assert "t1" not in mods, "Non-contrast T1 should not be present in UCSF-PDGM"


def test_seg_path_resolved(tmp_path):
    _make_ucsf_tree(tmp_path, [{"id": "UCSF-PDGM-004", "idh": "wildtype"}])
    records = load_ucsf_pdgm(tmp_path)
    assert records[0].seg_path is not None
    assert records[0].seg_path.exists()


def test_clinical_fields_preserved(tmp_path):
    _make_ucsf_tree(tmp_path, [{"id": "UCSF-PDGM-004", "idh": "wildtype"}])
    records = load_ucsf_pdgm(tmp_path)
    clin = records[0].clinical
    assert "Sex" in clin
    assert "Age at MRI" in clin
    assert "WHO CNS Grade" in clin
    # Raw IDH string must be stripped from clinical (it's stored as idh_label)
    assert "IDH" not in clin
    assert "ID" not in clin


def test_dataset_tag(tmp_path):
    _make_ucsf_tree(tmp_path, [{"id": "UCSF-PDGM-004", "idh": "wildtype"}])
    records = load_ucsf_pdgm(tmp_path)
    assert records[0].dataset == "UCSF-PDGM"


def test_subject_missing_files_skipped(tmp_path):
    _make_ucsf_tree(tmp_path, [
        {"id": "UCSF-PDGM-004", "idh": "wildtype",       "make_files": True},
        {"id": "UCSF-PDGM-005", "idh": "wildtype",       "make_files": False},
    ])
    records = load_ucsf_pdgm(tmp_path)
    # Only the subject with files should be returned
    assert len(records) == 1
    assert records[0].subject_id == "UCSF-PDGM-004"


def test_missing_csv_raises(tmp_path):
    (tmp_path / "UCSF-PDGM-v3").mkdir()
    with pytest.raises(FileNotFoundError, match="metadata CSV not found"):
        load_ucsf_pdgm(tmp_path)


def test_custom_metadata_csv_path(tmp_path):
    root = _make_ucsf_tree(tmp_path, [{"id": "UCSF-PDGM-004", "idh": "wildtype"}])
    alt_csv = tmp_path / "alt_metadata.csv"
    (tmp_path / "UCSF-PDGM-metadata_v2.csv").rename(alt_csv)
    records = load_ucsf_pdgm(root, metadata_csv=alt_csv)
    assert len(records) == 1


def test_unknown_idh_value_gives_none_label(tmp_path):
    _make_ucsf_tree(tmp_path, [{"id": "UCSF-PDGM-004", "idh": "indeterminate"}])
    records = load_ucsf_pdgm(tmp_path)
    assert records[0].idh_label is None


def test_partial_download_logs_warning(tmp_path, caplog):
    import logging
    _make_ucsf_tree(tmp_path, [
        {"id": "UCSF-PDGM-004", "idh": "wildtype", "make_files": True},
        {"id": "UCSF-PDGM-005", "idh": "wildtype", "make_files": False},
    ])
    with caplog.at_level(logging.WARNING, logger="src.data.data_loader"):
        load_ucsf_pdgm(tmp_path)
    assert any("download" in m.lower() for m in caplog.messages)


# ── Live data smoke test (skipped if dataset not present) ─────────────────────

LIVE_ROOT = Path(__file__).parent.parent / "data" / "UCSF-PDGM"

@pytest.mark.skipif(
    not (LIVE_ROOT / "UCSF-PDGM-metadata_v2.csv").exists(),
    reason="UCSF-PDGM dataset not downloaded",
)
def test_live_load_counts_and_labels():
    records = load_ucsf_pdgm(LIVE_ROOT)
    assert len(records) > 0, "No subjects loaded from live data"

    labelled   = [r for r in records if r.idh_label is not None]
    mutants    = [r for r in labelled if r.idh_label == 1]
    wildtypes  = [r for r in labelled if r.idh_label == 0]

    print(f"\nLoaded {len(records)} subjects  "
          f"({len(mutants)} mutant, {len(wildtypes)} wildtype, "
          f"{len(records) - len(labelled)} unlabelled)")

    assert len(labelled) > 0, "No IDH-labelled subjects found"
    assert len(mutants)   > 0, "No IDH-mutant subjects found"
    assert len(wildtypes) > 0, "No IDH-wildtype subjects found"


@pytest.mark.skipif(
    not (LIVE_ROOT / "UCSF-PDGM-metadata_v2.csv").exists(),
    reason="UCSF-PDGM dataset not downloaded",
)
def test_live_modality_paths_exist():
    records = load_ucsf_pdgm(LIVE_ROOT)
    missing_mods = []
    for r in records:
        for mod, path in r.modality_paths.items():
            if not path.exists():
                missing_mods.append(f"{r.subject_id}/{mod}: {path}")
    assert not missing_mods, f"Missing modality files:\n" + "\n".join(missing_mods[:10])


@pytest.mark.skipif(
    not (LIVE_ROOT / "UCSF-PDGM-metadata_v2.csv").exists(),
    reason="UCSF-PDGM dataset not downloaded",
)
def test_live_seg_paths_exist():
    records = load_ucsf_pdgm(LIVE_ROOT)
    with_seg    = [r for r in records if r.seg_path is not None]
    missing_seg = [r for r in with_seg if not r.seg_path.exists()]
    assert not missing_seg, \
        f"{len(missing_seg)} records have seg_path set but file is missing"
    print(f"\n{len(with_seg)}/{len(records)} subjects have segmentation files")
