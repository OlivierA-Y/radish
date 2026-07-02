"""
Cross-dataset evaluation script.
Runs the pipeline across multiple datasets and produces a combined results table.

Usage:
  python evaluate_datasets.py --config configs/datasets.json --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_datasets")
sys.path.insert(0, str(Path(__file__).parent))


EXAMPLE_CONFIG = {
    "datasets": [
        {
            "name": "TCGA-LGG",
            "mode": "tcga",
            "data_root": "data/TCGA-LGG",
            "clinical_csv": "data/TCGA-LGG/clinical.csv",
        },
        {
            "name": "TCGA-GBM",
            "mode": "tcga",
            "data_root": "data/TCGA-GBM",
            "clinical_csv": "data/TCGA-GBM/clinical.csv",
        },
        {
            "name": "UPENN-GBM",
            "mode": "manifest",
            "manifest":   "data/UPENN-GBM/manifest.csv",
            "data_root":  "data/UPENN-GBM",
        },
    ]
}


def _load_subjects_for_dataset(ds_cfg: dict) -> list:
    from src.data.data_loader import load_tcga, load_brats, load_from_manifest

    mode = ds_cfg.get("mode", "tcga")
    if mode == "tcga":
        records = load_tcga(
            ds_cfg["data_root"],
            clinical_csv=ds_cfg.get("clinical_csv"),
            dataset_tag=ds_cfg["name"],
        )
    elif mode == "brats":
        records = load_brats(ds_cfg["data_root"], dataset_tag=ds_cfg["name"])
    elif mode == "manifest":
        records = load_from_manifest(
            ds_cfg["manifest"], ds_cfg["data_root"], dataset_tag=ds_cfg["name"]
        )
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

    return [
        {
            "subject_id":      r.subject_id,
            "modality_paths":  r.modality_paths,
            "idh_label":       r.idh_label,
            "clinical":        r.clinical,
            "seg_path":        r.seg_path,
            "dataset":         r.dataset,
        }
        for r in records
    ]


def run_evaluation(args: argparse.Namespace) -> None:
    from src.pipeline.preprocessing import load_subject, preprocess_subject
    from src.pipeline.segmentation import get_segmenter
    from src.pipeline.registration import register_subject
    from src.pipeline.atlas_mapping import load_all_atlases
    from src.pipeline.feature_extraction import extract_all_features
    from src.pipeline.serialization import to_json, save_json
    from src.pipeline.llm_predictor import LLMPredictor
    from src.pipeline.evaluation import (
        evaluate_predictions, save_metrics, save_predictions_csv, save_reasoning_traces,
    )

    # Load dataset config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    else:
        logger.warning("No config found; using built-in example config")
        config = EXAMPLE_CONFIG

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segmenter  = get_segmenter("threshold")
    atlases    = {}
    try:
        atlases = load_all_atlases()
    except Exception as e:
        logger.warning("Atlas load failed: %s", e)

    predictor  = LLMPredictor(model=args.model)
    all_metrics = []

    for ds_cfg in config["datasets"]:
        ds_name = ds_cfg["name"]
        logger.info("=== Dataset: %s ===", ds_name)
        ds_out = out_dir / ds_name
        ds_out.mkdir(exist_ok=True)
        feat_dir = ds_out / "features"
        feat_dir.mkdir(exist_ok=True)

        try:
            subjects = _load_subjects_for_dataset(ds_cfg)
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", ds_name, e)
            continue

        predictions = []
        for subj in subjects:
            sid = subj["subject_id"]
            try:
                subject = load_subject(sid, subj["modality_paths"], subj["idh_label"])
                subject = preprocess_subject(subject)
                seg_path = subj.get("seg_path")
                if seg_path and Path(str(seg_path)).exists():
                    import nibabel as nib
                    import numpy as np
                    seg = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.uint8)
                else:
                    seg = segmenter.segment(subject.images, subject.affine)

                images, affine = register_subject(
                    subject.images, subject.affine, method="auto"
                )
                features  = extract_all_features(images, seg, atlases, subject.voxel_size_mm)
                json_str  = to_json(sid, features, clinical=subj.get("clinical", {}))
                save_json(json_str, feat_dir / f"{sid}.json")

                pred = predictor.predict(
                    subject_id=sid,
                    ground_truth=subj["idh_label"],
                    features_dict=features,
                )
                predictions.append(pred)
                logger.info("  %s → %s", sid, pred.idh_status)

            except Exception as e:
                logger.error("  Error processing %s: %s", sid, e)

        if predictions:
            save_predictions_csv(predictions, ds_out / "predictions.csv")
            save_reasoning_traces(predictions, ds_out / "reasoning_traces.txt")
            labelled = [p for p in predictions if p.ground_truth is not None]
            if labelled:
                metrics = evaluate_predictions(labelled, model=args.model, dataset=ds_name)
                save_metrics(metrics, ds_out / "metrics.json")
                all_metrics.append(asdict(metrics))
                print(f"\n{metrics.summary()}")

    # Combined summary table
    if all_metrics:
        summary_path = out_dir / "summary_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n=== Summary saved to {summary_path} ===")
        _print_table(all_metrics)


def _print_table(metrics_list: list) -> None:
    header = f"{'Dataset':<20} {'N':>5} {'Acc':>6} {'Sens':>6} {'Spec':>6} {'AUC':>6}"
    print("\n" + header)
    print("-" * len(header))
    for m in metrics_list:
        auc = f"{m['auc_roc']:.3f}" if m.get("auc_roc") else "  —  "
        print(
            f"{m['dataset']:<20} {m['n_total']:>5} "
            f"{m['accuracy']:>6.3f} {m['sensitivity']:>6.3f} "
            f"{m['specificity']:>6.3f} {auc:>6}"
        )


def main():
    p = argparse.ArgumentParser(description="Cross-dataset IDH evaluation")
    p.add_argument("--config",     type=str, default=None)
    p.add_argument("--model",      type=str, default="anthropic/claude-opus-4",
                   help="OpenRouter model ID or alias")
    p.add_argument("--output_dir", type=str, default="results/cross_dataset/")
    args = p.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
