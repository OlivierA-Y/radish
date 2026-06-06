"""
Main pipeline runner — zero-shot IDH mutation prediction.

Usage:
  # Synthetic data, dry run (no API key needed)
  python main.py --mode synthetic --n_mutant 10 --n_wildtype 10 --dry_run

  # OpenAI
  python main.py --mode synthetic --provider openai --model gpt-4o

  # OpenRouter — any model, one key
  python main.py --mode synthetic --provider openrouter --model anthropic/claude-opus-4
  python main.py --mode synthetic --model anthropic/claude-opus-4   # auto-detected

  # Real data
  python main.py --mode tcga   --data_root data/TCGA-LGG/  --model gpt-4o
  python main.py --mode ucsf   --data_root data/UCSF-PDGM/ --model gpt-4o
  python main.py --mode manifest --manifest data/my_cohort.csv --data_root data/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zero-Shot IDH Mutation Prediction Pipeline")
    p.add_argument("--mode", choices=["synthetic", "tcga", "brats", "ucsf", "manifest"],
                   default="synthetic", help="Data source")
    p.add_argument("--data_root", type=str, default="data/",
                   help="Root directory for real data")
    p.add_argument("--manifest", type=str, default=None,
                   help="CSV manifest (for --mode manifest)")
    p.add_argument("--model", type=str, default="gpt-4o",
                   help="Model ID — OpenAI (gpt-4o) or OpenRouter (anthropic/claude-opus-4, …)")
    p.add_argument("--provider", type=str, default=None,
                   choices=["openai", "openrouter"],
                   help="API provider. Auto-detected from model name if omitted "
                        "(a '/' in the model ID → openrouter).")
    p.add_argument("--n_mutant",   type=int, default=10, help="# synthetic mutant subjects")
    p.add_argument("--n_wildtype", type=int, default=10, help="# synthetic wildtype subjects")
    p.add_argument("--output_dir", type=str, default="results/",
                   help="Where to write predictions, metrics, and JSON features")
    p.add_argument("--seg_method", choices=["threshold", "brainsegfounder"],
                   default="threshold", help="Segmentation approach")
    p.add_argument("--seg_checkpoint", type=str, default=None,
                   help="Path to BrainSegFounder checkpoint (only for --seg_method brainsegfounder)")
    p.add_argument("--reg_method", choices=["auto", "ants", "affine"],
                   default="affine", help="Registration approach")
    p.add_argument("--no_llm", action="store_true",
                   help="Skip LLM calls (feature extraction and serialization only)")
    p.add_argument("--dry_run", action="store_true",
                   help="Use mock LLM responses (no API key needed)")
    p.add_argument("--only", choices=["mutant", "wildtype"], default=None,
                   help="Restrict to subjects with a known label (mutant=1, wildtype=0)")
    p.add_argument("--seed", type=int, default=42)
    return p


def _mock_predict(subject_id: str, narrative: str, ground_truth=None):
    """Deterministic fake LLM response for --dry_run testing."""
    import hashlib
    from src.pipeline.llm_predictor import IDHPrediction

    h = int(hashlib.md5(subject_id.encode()).hexdigest(), 16)
    status = "IDH-mutant" if h % 2 == 0 else "IDH-wildtype"
    return IDHPrediction(
        subject_id=subject_id,
        model="mock",
        idh_status=status,
        confidence="medium",
        reasoning="Mock prediction for dry-run testing.",
        raw_response='{"idh_status":"' + status + '","confidence":"medium","reasoning":"Mock."}',
        latency_s=0.001,
        ground_truth=ground_truth,
        decisive_feature="wt.volume_mm3",
        supporting_features=["wt.t1.mean"],
        contradicting_features=[],
        findings="Mock findings paragraph.",
        impression="Mock impression sentence.",
        phantom_citations=[],
        model_version="mock-v1",
    )


def run_pipeline(args: argparse.Namespace) -> None:
    from src.data.synthetic import generate_dataset
    from src.data.data_loader import load_tcga, load_brats, load_ucsf_pdgm, load_from_manifest
    from src.pipeline.preprocessing import preprocess_subject, load_subject, MRISubject
    from src.pipeline.segmentation import get_segmenter, compute_tumor_volumes
    from src.pipeline.registration import register_subject
    from src.pipeline.atlas_mapping import load_all_atlases
    from src.pipeline.feature_extraction import extract_all_features
    from src.pipeline.serialization import to_narrative, to_json, save_json
    from src.pipeline.llm_predictor import LLMPredictor
    from src.pipeline.evaluation import evaluate_predictions, save_metrics, save_predictions_csv, save_reasoning_traces

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load subjects ──────────────────────────────────────────────────
    logger.info("Loading subjects (mode=%s)…", args.mode)
    if args.mode == "synthetic":
        raw_subjects = generate_dataset(
            n_mutant=args.n_mutant,
            n_wildtype=args.n_wildtype,
            seed=args.seed,
        )
    elif args.mode == "tcga":
        records = load_tcga(args.data_root)
        raw_subjects = [
            {
                "subject_id": r.subject_id,
                "modality_paths": r.modality_paths,
                "idh_label": r.idh_label,
                "clinical": r.clinical,
                "seg_path": r.seg_path,
            }
            for r in records
        ]
    elif args.mode == "brats":
        records = load_brats(args.data_root)
        raw_subjects = [
            {
                "subject_id": r.subject_id,
                "modality_paths": r.modality_paths,
                "idh_label": r.idh_label,
                "clinical": r.clinical,
                "seg_path": r.seg_path,
            }
            for r in records
        ]
    elif args.mode == "ucsf":
        records = load_ucsf_pdgm(args.data_root)
        raw_subjects = [
            {
                "subject_id": r.subject_id,
                "modality_paths": r.modality_paths,
                "idh_label": r.idh_label,
                "clinical": r.clinical,
                "seg_path": r.seg_path,
            }
            for r in records
        ]
    elif args.mode == "manifest":
        if not args.manifest:
            raise ValueError("--manifest CSV required for manifest mode")
        records = load_from_manifest(args.manifest, args.data_root)
        raw_subjects = [
            {
                "subject_id": r.subject_id,
                "modality_paths": r.modality_paths,
                "idh_label": r.idh_label,
                "clinical": r.clinical,
                "seg_path": r.seg_path,
            }
            for r in records
        ]
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logger.info("Loaded %d subjects", len(raw_subjects))

    # ── 2. Apply --only filter ────────────────────────────────────────────
    if args.only:
        _target = 1 if args.only == "mutant" else 0
        raw_subjects = [s for s in raw_subjects if s.get("idh_label") == _target]
        logger.info("--only %s: %d subjects retained", args.only, len(raw_subjects))

    # ── 3. Initialise pipeline components ────────────────────────────────
    segmenter = get_segmenter(
        method=args.seg_method,
        checkpoint_path=args.seg_checkpoint,
    )

    logger.info("Loading atlases (this may download data on first run)…")
    # Use a small target shape for synthetic data; real data uses MNI dims
    target_shape = (64, 64, 64) if args.mode == "synthetic" else None
    atlases = {}
    try:
        atlases = load_all_atlases(target_shape=target_shape)
    except Exception as e:
        logger.warning("Atlas loading partially failed: %s — continuing without atlases", e)

    # Create per-run directory and predictor
    predictor = None
    run_dir: Path | None = None
    if not args.no_llm:
        run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir / "runs" / run_ts
        run_dir.mkdir(parents=True, exist_ok=True)
        run_log = run_dir / "llm_log.jsonl"
        with open(run_log, "w", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "type":             "run_header",
                "total_subjects":   len(raw_subjects),
                "model":            args.model,
                "timestamp":        datetime.now().isoformat(),
            }) + "\n")
        predictor = LLMPredictor(
            model=args.model,
            provider=args.provider,
            log_path=run_log,
        )
        logger.info("Run directory: %s", run_dir)

    # ── 4. Process each subject ───────────────────────────────────────────

    all_predictions = []
    features_dir = out_dir / "features"
    features_dir.mkdir(exist_ok=True)

    for subj_data in raw_subjects:
        sid = subj_data["subject_id"]
        cache_path = features_dir / f"{sid}.json"
        logger.info("Processing %s…", sid)

        try:
            # A+B) Feature extraction — load from cache if available
            idh_label = subj_data.get("idh_label")
            clinical  = subj_data.get("clinical", {}) if args.mode != "synthetic" else {}

            if cache_path.exists():
                cached   = json.loads(cache_path.read_text(encoding="utf-8"))
                features = cached["imaging_features"]
                clinical = cached.get("clinical") or clinical
                logger.info("Loaded cached features for %s (skipping MRI processing)", sid)
            else:
                if args.mode == "synthetic":
                    images     = subj_data["images"]
                    seg        = subj_data["seg"]
                    affine     = subj_data["affine"]
                    voxel_size = subj_data["voxel_size"]
                else:
                    subject = load_subject(
                        sid,
                        subj_data["modality_paths"],
                        idh_label=idh_label,
                    )
                    subject = preprocess_subject(subject)
                    images     = subject.images
                    affine     = subject.affine
                    voxel_size = subject.voxel_size_mm

                    # Segmentation
                    if subj_data.get("seg_path") and Path(subj_data["seg_path"]).exists():
                        import nibabel as nib
                        import numpy as np
                        seg = np.asarray(
                            nib.load(str(subj_data["seg_path"])).dataobj, dtype=np.uint8
                        )
                    else:
                        seg = segmenter.segment(images, affine)

                    # Registration (skip for synthetic; already in voxel space)
                    if args.reg_method != "affine" or args.mode != "synthetic":
                        images, affine = register_subject(
                            images, affine, method=args.reg_method
                        )
                        # Resample seg to match registered image space (nearest-neighbour)
                        from src.pipeline.registration import resample_to_shape
                        import numpy as _np
                        target_shape = next(iter(images.values())).shape
                        if seg.shape != target_shape:
                            seg = resample_to_shape(seg, target_shape, order=0).astype(_np.uint8)

                features = extract_all_features(
                    images=images,
                    seg=seg,
                    atlases=atlases,
                    voxel_size_mm=voxel_size,
                )

                # C) Persist features
                json_str = to_json(sid, features, clinical=clinical)
                save_json(json_str, cache_path)

            # Always regenerate narrative (cheap, prompt-template may change)
            narrative = to_narrative(sid, features, clinical=clinical)

            # D) LLM prediction
            if args.no_llm:
                logger.info("Skipping LLM for %s (--no_llm)", sid)
                continue

            if args.dry_run:
                pred = _mock_predict(sid, narrative, ground_truth=idh_label)
            else:
                pred = predictor.predict(
                    subject_id=sid,
                    narrative=narrative,
                    ground_truth=idh_label,
                    features_dict=features,
                    clinical_dict=clinical,
                )

            all_predictions.append(pred)
            status_str = f"→ {pred.idh_status} [{pred.confidence}]"
            if pred.correct is not None:
                status_str += f"  ({'✓' if pred.correct else '✗'})"
            logger.info("%s  %s", sid, status_str)

        except Exception as e:
            logger.error("Failed to process %s: %s", sid, e, exc_info=True)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    if all_predictions:
        artifacts_dir = run_dir if run_dir else out_dir
        save_predictions_csv(all_predictions, artifacts_dir / "predictions.csv")
        save_reasoning_traces(all_predictions, artifacts_dir / "reasoning_traces.txt")
        logger.info("Saved predictions to %s", artifacts_dir / "predictions.csv")
        logger.info("Saved reasoning traces to %s", artifacts_dir / "reasoning_traces.txt")

        labelled = [p for p in all_predictions if p.ground_truth is not None]
        if labelled:
            metrics = evaluate_predictions(
                labelled,
                model=args.model if not args.dry_run else "mock",
                dataset=args.mode,
            )
            save_metrics(metrics, artifacts_dir / "metrics.json")
            print("\n" + metrics.summary())
        else:
            logger.warning("No labelled predictions — skipping metrics")
    else:
        logger.info("No predictions made (--no_llm or no subjects processed)")

    logger.info("Pipeline complete. Results in: %s", run_dir or out_dir)


def main():
    parser = _build_arg_parser()
    args   = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
