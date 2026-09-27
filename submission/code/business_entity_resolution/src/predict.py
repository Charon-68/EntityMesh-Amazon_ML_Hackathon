"""
predict.py — Test-set inference: blocking → features → LightGBM → output TSVs.

Loads the trained model from models/matcher.joblib, runs the full inference
pipeline on the test set, and writes:

    submission/output/candidate_pairs.tsv
    submission/output/matching_results.tsv

Both files are then validated with utils/validate_submission.py (exit 0 = PASS).

Usage
-----
From project root (student_resource/):

    python submission/code/business_entity_resolution/src/predict.py

    # Optional overrides:
    python submission/code/business_entity_resolution/src/predict.py \\
        --threshold 0.45 \\
        --k 25
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gc
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).resolve().parents[4]   # …/Amazon/
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from normalize  import normalize_name, normalize_address  # noqa: E402
from features   import (                                   # noqa: E402
    FEATURE_NAMES,
    extract_pair_features,
    build_lookup_fast,
)
from blocking   import (                                   # noqa: E402
    CountryIndex,
    build_country_indices,
    _run_blocking,
    write_candidate_tsv,
    K_INIT,
)

TEST_DIR    = ROOT / "dataset" / "test"
OUTPUT_DIR  = ROOT / "submission" / "output"
MODELS_DIR  = Path(__file__).resolve().parents[1] / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_BATCH = 200_000   # pairs per feature-extraction chunk


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_source(path: Path, label: str) -> pd.DataFrame:
    print(f"  Loading {label} ({path.name}) …", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(
        path, sep="\t", dtype=str,
        keep_default_na=False, encoding="utf-8",
    )
    df["norm_name"]    = df["business_name"].apply(normalize_name)
    df["norm_address"] = df["business_address"].apply(normalize_address)
    df["country"]      = df["country"].str.strip()
    print(f"    → {len(df):,} rows  ({time.perf_counter()-t0:.1f}s)", flush=True)
    return df


# ---------------------------------------------------------------------------
# 2. Candidate TSV → flat pairs iterator (memory-efficient)
# ---------------------------------------------------------------------------

def iter_candidate_pairs(candidate_tsv: Path):
    """Yield (source1_entity_id, candidate_entity_id) from candidate_pairs.tsv."""
    with open(candidate_tsv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw   = row.get("candidate_entity_ids", "").strip()
            cands = [c.strip() for c in raw.split(",") if c.strip()] if raw else []
            for cid in cands:
                yield s1_id, cid


def count_candidate_pairs(candidate_tsv: Path) -> int:
    """Fast count of total pairs without loading everything into memory."""
    total = 0
    with open(candidate_tsv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            raw = row.get("candidate_entity_ids", "").strip()
            total += len([c for c in raw.split(",") if c.strip()]) if raw else 0
    return total


# ---------------------------------------------------------------------------
# 3. Feature extraction + inference (batched, streaming)
# ---------------------------------------------------------------------------

def infer_matches(
    candidate_tsv: Path,
    s1_lookup: Dict[str, Tuple[str, str]],
    pool_lookup: Dict[str, Tuple[str, str]],
    model,
    threshold: float,
    all_s1_ids: List[str],
) -> Dict[str, List[str]]:
    """Stream candidate pairs, extract features, score, apply threshold.

    Parameters
    ----------
    candidate_tsv  : path to candidate_pairs.tsv
    s1_lookup      : entity_id → (norm_name, norm_address) for S1
    pool_lookup    : entity_id → (norm_name, norm_address) for S2+S3
    model          : loaded LightGBM dict (keys: model, threshold, feature_names)
    threshold      : decision threshold (overrides model's saved threshold if provided)
    all_s1_ids     : ordered list of ALL test S1 IDs (to ensure every gets a row)

    Returns
    -------
    dict  source1_entity_id → list[matched_entity_id]  (positives only, deduped)
    """
    lgbm       = model["model"]
    feat_names = model.get("feature_names", FEATURE_NAMES)

    # Initialise predictions dict — every S1 entity must have an entry (even singletons)
    matches: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

    # --- Stream pairs in chunks ---
    buf_s1:   List[str]         = []
    buf_cand: List[str]         = []
    buf_feat: List[np.ndarray]  = []

    total_pairs     = count_candidate_pairs(candidate_tsv)
    processed_pairs = 0
    t0              = time.perf_counter()

    def _flush_buffer() -> None:
        if not buf_feat:
            return
        X     = np.vstack(buf_feat).astype(np.float32)
        probs = lgbm.predict_proba(X)[:, 1]
        for s1_id, cid, prob in zip(buf_s1, buf_cand, probs):
            if prob >= threshold:
                matches[s1_id].append(cid)
        buf_s1.clear()
        buf_cand.clear()
        buf_feat.clear()

    for s1_id, cid in iter_candidate_pairs(candidate_tsv):
        s1e  = s1_lookup.get(s1_id)
        ce   = pool_lookup.get(cid)

        if s1e is None or ce is None:
            processed_pairs += 1
            continue

        feat = extract_pair_features(
            s1e[0], s1e[1],
            ce[0],  ce[1],
            cid,
        )
        buf_s1.append(s1_id)
        buf_cand.append(cid)
        buf_feat.append(feat)
        processed_pairs += 1

        if len(buf_feat) >= FEAT_BATCH:
            _flush_buffer()

        if processed_pairs % (FEAT_BATCH * 5) == 0 or processed_pairs == total_pairs:
            elapsed = time.perf_counter() - t0
            pct     = 100 * processed_pairs / total_pairs if total_pairs else 0
            rate    = processed_pairs / elapsed if elapsed > 0 else 0
            print(
                f"    {processed_pairs:>10,}/{total_pairs:,} pairs  "
                f"({pct:.1f}%)  {rate:,.0f} pairs/s",
                flush=True,
            )

    _flush_buffer()   # flush remainder

    # Deduplicate matches preserving order
    for s1_id in matches:
        seen: Set[str] = set()
        deduped = []
        for cid in matches[s1_id]:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        matches[s1_id] = deduped

    n_with_match = sum(1 for v in matches.values() if v)
    n_singleton  = len(matches) - n_with_match
    print(
        f"\n  Inference complete: {n_with_match:,} entities with matches, "
        f"{n_singleton:,} singletons (predicted empty)",
        flush=True,
    )
    return matches


# ---------------------------------------------------------------------------
# 4. Write matching_results.tsv
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    """Return path relative to ROOT if possible, else absolute string."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_matching_tsv(
    matches: Dict[str, List[str]],
    s1_ids_ordered: List[str],
    path: Path,
) -> None:
    """Write matching_results.tsv — one row per S1 entity, ordered."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in s1_ids_ordered:
            matched = matches.get(s1_id, [])
            writer.writerow([s1_id, ",".join(matched)])
    print(f"  Written {len(s1_ids_ordered):,} rows → {_rel(path)}", flush=True)


# ---------------------------------------------------------------------------
# 5. Run validate_submission.py programmatically
# ---------------------------------------------------------------------------

def run_validator(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
) -> int:
    """Run utils/validate_submission.py and return its exit code."""
    validator = ROOT / "utils" / "validate_submission.py"
    cmd = [
        sys.executable, str(validator),
        "--matching",   str(matching_path),
        "--candidate",  str(candidate_path),
        "--test-dir",   str(test_dir),
    ]
    print(f"\n  Running validator: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test-set inference → submission TSVs.")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold (default: use value from models/matcher.joblib)",
    )
    parser.add_argument(
        "--k", type=int, default=K_INIT,
        help=f"Top-k for blocking (default: {K_INIT})",
    )
    parser.add_argument(
        "--skip-blocking", action="store_true",
        help="Skip blocking if submission/output/candidate_pairs.tsv already exists",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip running validate_submission.py at the end",
    )
    args = parser.parse_args()

    divider = "═" * 68
    print(f"\n{divider}")
    print("  PREDICT — Test-set inference")
    print(divider)

    # ---------------------------------------------------------------- model
    model_path = MODELS_DIR / "matcher.joblib"
    if not model_path.exists():
        print(
            f"  ❌ Model not found: {model_path}\n"
            "  Run train_model.py first (or main.py --steps train).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n  Loading model from {model_path.relative_to(ROOT)} …")
    model_bundle = joblib.load(model_path)
    saved_threshold = model_bundle.get("threshold", 0.5)
    threshold = args.threshold if args.threshold is not None else saved_threshold
    print(f"  Saved threshold  : {saved_threshold:.2f}")
    print(f"  Using threshold  : {threshold:.2f}")
    print(f"  Best iteration   : {model_bundle.get('best_iteration', 'N/A')}")

    # ---------------------------------------------------------------- data
    print(f"\n{divider}")
    print("  Loading test sources …")
    print(divider)
    test_s1 = load_source(TEST_DIR / "test_source1.tsv", "Test Source 1")
    test_s2 = load_source(TEST_DIR / "test_source2.tsv", "Test Source 2")
    test_s3 = load_source(TEST_DIR / "test_source3.tsv", "Test Source 3")

    s1_ids_ordered = test_s1["entity_id"].tolist()
    s1_lookup      = build_lookup_fast(test_s1)

    pool_df     = pd.concat([test_s2, test_s3], ignore_index=True)
    pool_lookup = build_lookup_fast(pool_df)
    del test_s2, test_s3
    gc.collect()

    # ---------------------------------------------------------------- blocking
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    if args.skip_blocking and candidate_path.exists():
        print(f"\n  Skipping blocking — using existing {candidate_path.name}")
    else:
        print(f"\n{divider}")
        print("  Running blocking on test set …")
        print(divider)
        country_indices = build_country_indices(pool_df)
        global_index    = CountryIndex("__global__", pool_df)

        candidates = _run_blocking(
            test_s1, pool_df, country_indices, global_index, k=args.k
        )
        write_candidate_tsv(candidates, s1_ids_ordered, candidate_path)
        del candidates, country_indices, global_index
        gc.collect()

    # ---------------------------------------------------------------- inference
    print(f"\n{divider}")
    print(f"  Feature extraction + LightGBM inference (threshold={threshold:.2f}) …")
    print(divider)

    matches = infer_matches(
        candidate_tsv  = candidate_path,
        s1_lookup      = s1_lookup,
        pool_lookup    = pool_lookup,
        model          = model_bundle,
        threshold      = threshold,
        all_s1_ids     = s1_ids_ordered,
    )

    # ---------------------------------------------------------------- write output
    print(f"\n{divider}")
    print("  Writing output TSVs …")
    print(divider)

    matching_path = OUTPUT_DIR / "matching_results.tsv"
    write_matching_tsv(matches, s1_ids_ordered, matching_path)

    # Also sync to ROOT output/ directory
    ROOT_OUTPUT = ROOT / "output"
    ROOT_OUTPUT.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(matching_path, ROOT_OUTPUT / "matching_results.tsv")
    shutil.copy2(candidate_path, ROOT_OUTPUT / "candidate_pairs.tsv")
    print(f"  Synced outputs → output/ matching_results.tsv & candidate_pairs.tsv", flush=True)

    # Stats
    n_matched   = sum(1 for v in matches.values() if v)
    n_singleton = len(matches) - n_matched
    avg_matches = (
        sum(len(v) for v in matches.values() if v) / n_matched if n_matched else 0.0
    )
    print(f"\n  Output summary:")
    print(f"    Total S1 entities   : {len(matches):,}")
    print(f"    With ≥1 match       : {n_matched:,}  ({100*n_matched/len(matches):.1f}%)")
    print(f"    Singletons (empty)  : {n_singleton:,}  ({100*n_singleton/len(matches):.1f}%)")
    print(f"    Avg matches (non-∅) : {avg_matches:.2f}")

    # ---------------------------------------------------------------- validate
    if not args.skip_validation:
        print(f"\n{divider}")
        print("  VALIDATION")
        print(divider)
        rc = run_validator(matching_path, candidate_path, TEST_DIR)
        if rc == 0:
            print("\n  ✅ PASS — submission files are valid and ready to submit.")
        else:
            print("\n  ❌ FAIL — fix the issues above before submitting.")
            sys.exit(rc)
    else:
        print("\n  Validation skipped.")

    print(f"\n{divider}")
    print("  PREDICT DONE.")
    print(f"  matching_results.tsv  → {matching_path.relative_to(ROOT)}")
    print(f"  candidate_pairs.tsv   → {candidate_path.relative_to(ROOT)}")
    print(divider + "\n")


if __name__ == "__main__":
    main()
