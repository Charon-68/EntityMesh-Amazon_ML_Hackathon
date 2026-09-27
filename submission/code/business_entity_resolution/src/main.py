"""
main.py — End-to-end pipeline orchestrator for Business Entity Resolution.

Ties together all pipeline stages into a single CLI entrypoint:

    Stage 1  split     — Train/val split (split_data.py)
    Stage 2  block     — Candidate generation / blocking (blocking.py)
    Stage 3  train     — Feature extraction + LightGBM training (train_model.py)
    Stage 4  predict   — Test-set inference + submission TSVs (predict.py)
    Stage 5  validate  — Run utils/validate_submission.py

Each stage is independently runnable via --steps. Run all stages:

    python submission/code/business_entity_resolution/src/main.py

Run specific stages only:

    python submission/code/business_entity_resolution/src/main.py --steps split block
    python submission/code/business_entity_resolution/src/main.py --steps train predict

Additional options are forwarded to the relevant sub-scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).resolve().parents[4]   # …/Amazon/
SRC_DIR = Path(__file__).resolve().parent

SPLIT_SCRIPT  = SRC_DIR / "split_data.py"
BLOCKING_MODE = "both"   # runs recall gate on train then generates test candidates
TRAIN_SCRIPT  = SRC_DIR / "train_model.py"
PREDICT_SCRIPT = SRC_DIR / "predict.py"
VALIDATOR     = ROOT / "utils" / "validate_submission.py"
OUTPUT_DIR    = ROOT / "submission" / "output"
TEST_DIR      = ROOT / "dataset" / "test"

PYTHON = sys.executable

# All valid stage names in execution order
ALL_STAGES = ["split", "block", "train", "predict", "validate"]


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    pass


def _run(cmd: List[str], stage: str) -> None:
    """Run a subprocess command, raising PipelineError on non-zero exit."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n{'─' * 68}")
    print(f"  STAGE [{stage.upper()}]  →  {cmd_str}")
    print(f"{'─' * 68}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise PipelineError(
            f"Stage '{stage}' failed with exit code {result.returncode}.\n"
            f"Command: {cmd_str}"
        )
    print(f"\n  ✅ Stage [{stage}] completed in {elapsed:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def stage_split(args: argparse.Namespace) -> None:
    """Stage 1: Create train/val split and data-profile report."""
    _run([PYTHON, str(SPLIT_SCRIPT)], "split")


def stage_block(args: argparse.Namespace) -> None:
    """Stage 2: Blocking — train recall gate + test candidate generation."""
    cmd = [
        PYTHON, str(SRC_DIR / "blocking.py"),
        "--mode", "both",
        "--k",    str(args.k),
    ]
    if args.no_recall_gate:
        cmd.append("--no-recall-gate")
    _run(cmd, "block")


def stage_train(args: argparse.Namespace) -> None:
    """Stage 3: Feature extraction + LightGBM training + threshold sweep."""
    _run([PYTHON, str(TRAIN_SCRIPT)], "train")


def stage_predict(args: argparse.Namespace) -> None:
    """Stage 4: Test-set inference → matching_results.tsv + candidate_pairs.tsv."""
    cmd = [PYTHON, str(PREDICT_SCRIPT), "--k", str(args.k)]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    if args.skip_blocking:
        cmd.append("--skip-blocking")
    # Suppress internal validation — we run it explicitly in stage_validate
    cmd.append("--skip-validation")
    _run(cmd, "predict")


def stage_validate(args: argparse.Namespace) -> None:
    """Stage 5: Official submission validation."""
    matching_path  = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    cmd = [
        PYTHON, str(VALIDATOR),
        "--matching",   str(matching_path),
        "--candidate",  str(candidate_path),
        "--test-dir",   str(TEST_DIR),
    ]
    _run(cmd, "validate")


# ---------------------------------------------------------------------------
# Pipeline pre-flight checks
# ---------------------------------------------------------------------------

def _preflight(stages: List[str]) -> None:
    """Warn if upstream artefacts are missing for the selected stages."""
    data_dir = SRC_DIR.parent / "data"
    models_dir = SRC_DIR.parent / "models"

    checks = {
        "block": [
            (data_dir / "train_ids.txt",
             "Run --steps split first to create train/val IDs."),
        ],
        "train": [
            (OUTPUT_DIR / "candidate_pairs_train.tsv",
             "Run --steps block first to generate train candidate pairs."),
        ],
        "predict": [
            (models_dir / "matcher.joblib",
             "Run --steps train first to produce the model."),
        ],
        "validate": [
            (OUTPUT_DIR / "matching_results.tsv",
             "Run --steps predict first to generate submission TSVs."),
        ],
    }

    warned = False
    for stage in stages:
        for path, hint in checks.get(stage, []):
            if not path.exists():
                print(f"  ⚠️  Pre-flight warning [{stage}]: {path.name} not found. {hint}")
                warned = True
    if warned:
        print()


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end Business Entity Resolution pipeline.\n\n"
            "Stages (run in order): split → block → train → predict → validate\n"
            "Use --steps to run only specific stages."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
        metavar="STAGE",
        help=(
            f"Which stages to run. Choices: {ALL_STAGES}. "
            "Default: run all stages in order."
        ),
    )
    parser.add_argument(
        "--k", type=int, default=20,
        help="Top-k for blocking (default: 20)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the trained model's decision threshold for prediction.",
    )
    parser.add_argument(
        "--skip-blocking", action="store_true",
        help="In predict stage: skip blocking if candidate_pairs.tsv already exists.",
    )
    parser.add_argument(
        "--no-recall-gate", action="store_true",
        help="In block stage: skip the recall gate check (faster).",
    )
    args = parser.parse_args()

    # Preserve requested stage order while deduplicating
    seen_stages: set = set()
    stages: List[str] = []
    for s in args.steps:
        if s not in seen_stages:
            stages.append(s)
            seen_stages.add(s)

    divider = "═" * 68
    print(f"\n{divider}")
    print("  BUSINESS ENTITY RESOLUTION — PIPELINE ORCHESTRATOR")
    print(divider)
    print(f"  Stages to run : {' → '.join(stages)}")
    print(f"  Top-k         : {args.k}")
    if args.threshold:
        print(f"  Threshold     : {args.threshold}")
    print(divider, flush=True)

    _preflight(stages)

    stage_fns = {
        "split":    stage_split,
        "block":    stage_block,
        "train":    stage_train,
        "predict":  stage_predict,
        "validate": stage_validate,
    }

    t_pipeline = time.perf_counter()
    failed_at: Optional[str] = None

    for stage in stages:
        try:
            stage_fns[stage](args)
        except PipelineError as exc:
            print(f"\n{'═' * 68}")
            print(f"  ❌ PIPELINE FAILED at stage [{stage}]")
            print(f"  {exc}")
            print(f"{'═' * 68}\n")
            failed_at = stage
            break

    elapsed_total = time.perf_counter() - t_pipeline

    print(f"\n{divider}")
    if failed_at:
        print(f"  PIPELINE ABORTED at [{failed_at}]  "
              f"(total elapsed: {elapsed_total:.1f}s)")
    else:
        print(f"  ✅ ALL STAGES COMPLETE  (total elapsed: {elapsed_total:.1f}s)")
        print()
        print(f"  Output files:")
        print(f"    submission/output/candidate_pairs.tsv")
        print(f"    submission/output/matching_results.tsv")
        print()
        print(f"  Upload matching_results.tsv to the leaderboard portal.")
        print(f"  Zip the submission/ folder for the final package submission.")
    print(divider + "\n")

    sys.exit(1 if failed_at else 0)


if __name__ == "__main__":
    main()
