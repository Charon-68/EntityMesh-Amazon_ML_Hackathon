"""
split_data.py — Data profiling & leak-free train/val splitter.

Usage
-----
Run from the project root (student_resource/):

    python submission/code/business_entity_resolution/src/split_data.py

Outputs (written to submission/code/business_entity_resolution/data/):
    train_ids.txt        — Source-1 entity IDs assigned to the training fold
    val_ids.txt          — Source-1 entity IDs assigned to the validation fold
    val_ground_truth.tsv — Ground-truth rows for the validation entities
    train_ground_truth.tsv — Ground-truth rows for the training entities

The split is done **strictly by Source-1 entity ID** (80 % train / 20 % val)
to prevent any form of label leakage. Source-2 and Source-3 records are
shared across both folds (blocking is done on the full S2/S3 pool at
inference time).
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]   # …/Amazon/
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR  = ROOT / "dataset" / "test"

DATA_OUT  = Path(__file__).resolve().parents[1] / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VAL_FRAC    = 0.20


# ---------------------------------------------------------------------------
# 1. Loaders
# ---------------------------------------------------------------------------

def load_tsv(path: Path, label: str) -> pd.DataFrame:
    """Load a TSV with UTF-8 encoding; print size and missing-value summary."""
    print(f"\n{'─' * 60}")
    print(f"  Loading: {path.name}  [{label}]")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    missing = df.replace("", pd.NA).isna().sum()
    if missing.any():
        print("  Missing values:")
        for col, cnt in missing[missing > 0].items():
            pct = 100 * cnt / len(df)
            print(f"    {col:<25s} {cnt:>8,}  ({pct:.2f} %)")
    else:
        print("  Missing values: none")
    return df


def load_ground_truth(path: Path) -> Dict[str, List[str]]:
    """Load ground-truth TSV → {source1_entity_id: [matched_ids]}."""
    gt: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw   = row.get("matched_entity_ids", "").strip()
            matched = [m.strip() for m in raw.split(",") if m.strip()] if raw else []
            gt[s1_id] = matched
    return gt


# ---------------------------------------------------------------------------
# 2. Ground-truth profiling
# ---------------------------------------------------------------------------

def profile_ground_truth(gt: Dict[str, List[str]]) -> None:
    """Print detailed match-distribution statistics."""
    print(f"\n{'═' * 60}")
    print("  GROUND-TRUTH DISTRIBUTION")
    print(f"{'═' * 60}")

    total = len(gt)
    singletons = sum(1 for v in gt.values() if not v)
    non_singletons = total - singletons

    match_counts = [len(v) for v in gt.values() if v]   # only non-singletons
    total_pairs  = sum(match_counts)

    s2_matches = sum(
        sum(1 for m in v if m.startswith("S2-"))
        for v in gt.values()
    )
    s3_matches = sum(
        sum(1 for m in v if m.startswith("S3-"))
        for v in gt.values()
    )

    print(f"  Total S1 entities            : {total:>10,}")
    print(f"  Singletons (no match)        : {singletons:>10,}  ({100*singletons/total:.2f} %)")
    print(f"  Entities with ≥1 match       : {non_singletons:>10,}  ({100*non_singletons/total:.2f} %)")
    if match_counts:
        print(f"  Total matched pairs          : {total_pairs:>10,}")
        print(f"    ↳ S2 matches               : {s2_matches:>10,}")
        print(f"    ↳ S3 matches               : {s3_matches:>10,}")
        print(f"  Min matches per entity       : {min(match_counts):>10,}")
        print(f"  Max matches per entity       : {max(match_counts):>10,}")
        mean_mc = sum(match_counts) / len(match_counts)
        print(f"  Mean matches (non-singleton) : {mean_mc:>10.2f}")

        # Bucket distribution
        buckets = {1: 0, "2-5": 0, "6-10": 0, ">10": 0}
        for c in match_counts:
            if c == 1:
                buckets[1] += 1
            elif c <= 5:
                buckets["2-5"] += 1
            elif c <= 10:
                buckets["6-10"] += 1
            else:
                buckets[">10"] += 1
        print("\n  Match-count distribution (non-singletons):")
        for label, cnt in buckets.items():
            bar = "█" * min(40, int(40 * cnt / non_singletons)) if non_singletons else ""
            print(f"    {str(label):<8s} {cnt:>8,}  {bar}")


# ---------------------------------------------------------------------------
# 3. Split
# ---------------------------------------------------------------------------

def stratified_split(
    gt: Dict[str, List[str]],
    val_frac: float = VAL_FRAC,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str]]:
    """Return (train_ids, val_ids) with stratified singleton preservation.

    Strategy
    --------
    Shuffle S1 entities, then split while preserving the singleton ratio
    in both folds. This avoids a degenerate fold where all singletons land
    in one split.
    """
    rng = random.Random(seed)

    singletons     = [s1 for s1, v in gt.items() if not v]
    non_singletons = [s1 for s1, v in gt.items() if v]

    rng.shuffle(singletons)
    rng.shuffle(non_singletons)

    n_val_singletons     = round(len(singletons) * val_frac)
    n_val_non_singletons = round(len(non_singletons) * val_frac)

    val_ids   = singletons[:n_val_singletons] + non_singletons[:n_val_non_singletons]
    train_ids = singletons[n_val_singletons:] + non_singletons[n_val_non_singletons:]

    rng.shuffle(val_ids)
    rng.shuffle(train_ids)

    return train_ids, val_ids


# ---------------------------------------------------------------------------
# 4. Save split artefacts
# ---------------------------------------------------------------------------

def save_id_list(ids: List[str], path: Path) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"  Saved {len(ids):,} IDs → {path.relative_to(ROOT)}")


def save_split_gt(
    gt: Dict[str, List[str]],
    ids: List[str],
    path: Path,
) -> None:
    """Write a ground-truth TSV for a subset of S1 entity IDs."""
    id_set = set(ids)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in ids:
            if s1_id in gt:
                writer.writerow([s1_id, ",".join(gt[s1_id])])
    print(f"  Saved GT slice ({len(ids):,} rows) → {path.relative_to(ROOT)}")


def print_split_summary(
    train_ids: List[str],
    val_ids:   List[str],
    gt: Dict[str, List[str]],
) -> None:
    def frac_singletons(ids: List[str]) -> float:
        s = sum(1 for i in ids if not gt.get(i))
        return 100 * s / len(ids) if ids else 0.0

    print(f"\n{'═' * 60}")
    print("  SPLIT SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Train entities : {len(train_ids):>10,}  (singletons: {frac_singletons(train_ids):.2f} %)")
    print(f"  Val   entities : {len(val_ids):>10,}  (singletons: {frac_singletons(val_ids):.2f} %)")
    print(f"  Val fraction   : {100*len(val_ids)/(len(train_ids)+len(val_ids)):.2f} %")
    print(f"\n  Note: Source-2 and Source-3 records are shared.")
    print(f"  The split is strictly over Source-1 IDs — no pair-level leakage.")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "═" * 60)
    print("  DATA PROFILING & VALIDATION SPLITTER")
    print("═" * 60)

    # -- Load all 7 TSVs -----------------------------------------------------
    train_s1 = load_tsv(TRAIN_DIR / "train_source1.tsv", "Train Source 1")
    train_s2 = load_tsv(TRAIN_DIR / "train_source2.tsv", "Train Source 2")
    train_s3 = load_tsv(TRAIN_DIR / "train_source3.tsv", "Train Source 3")
    test_s1  = load_tsv(TEST_DIR  / "test_source1.tsv",  "Test  Source 1")
    test_s2  = load_tsv(TEST_DIR  / "test_source2.tsv",  "Test  Source 2")
    test_s3  = load_tsv(TEST_DIR  / "test_source3.tsv",  "Test  Source 3")

    # Ground truth counts as the 7th file
    print(f"\n{'─' * 60}")
    print(f"  Loading: train_ground_truth.tsv  [Ground Truth]")
    gt = load_ground_truth(TRAIN_DIR / "train_ground_truth.tsv")
    print(f"  Shape  : {len(gt):,} rows × 2 columns")

    # -- Country distributions -----------------------------------------------
    print(f"\n{'═' * 60}")
    print("  COUNTRY DISTRIBUTIONS")
    print(f"{'═' * 60}")
    for label, df in [
        ("train_source1", train_s1),
        ("train_source2", train_s2),
        ("train_source3", train_s3),
        ("test_source1",  test_s1),
        ("test_source2",  test_s2),
        ("test_source3",  test_s3),
    ]:
        counts = df["country"].value_counts().to_dict()
        counts_str = "  ".join(f"{k}: {v:,}" for k, v in sorted(counts.items()))
        print(f"  {label:<18s}  {counts_str}")

    # -- Ground truth profiling ----------------------------------------------
    profile_ground_truth(gt)

    # -- Split ---------------------------------------------------------------
    print(f"\n{'═' * 60}")
    print("  CREATING TRAIN / VAL SPLIT  (80 / 20, stratified by singleton ratio)")
    print(f"{'═' * 60}")

    train_ids, val_ids = stratified_split(gt, val_frac=VAL_FRAC, seed=RANDOM_SEED)
    print_split_summary(train_ids, val_ids, gt)

    # -- Save artefacts ------------------------------------------------------
    print(f"\n  Writing split artefacts to: {DATA_OUT.relative_to(ROOT)}/")
    save_id_list(train_ids, DATA_OUT / "train_ids.txt")
    save_id_list(val_ids,   DATA_OUT / "val_ids.txt")
    save_split_gt(gt, train_ids, DATA_OUT / "train_ground_truth.tsv")
    save_split_gt(gt, val_ids,   DATA_OUT / "val_ground_truth.tsv")

    print(f"\n{'═' * 60}")
    print("  DONE. All artefacts written successfully.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
