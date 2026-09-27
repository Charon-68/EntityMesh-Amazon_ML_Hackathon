"""
train_model.py — LightGBM classifier training + F0.5 threshold sweep.

Pipeline
--------
1. Load train candidate pairs (from blocking stage).
2. Build ground-truth labels from train_ground_truth.tsv.
3. Extract pairwise features via features.py.
4. Train LightGBM with logloss objective.
5. Sweep probability thresholds [0.30 … 0.95] on the validation set.
6. Select the threshold that maximises macro-averaged F0.5.
7. Save model + threshold artefacts to models/.

Usage
-----
From project root (student_resource/):

    python submission/code/business_entity_resolution/src/train_model.py

Outputs:
    models/matcher.joblib            LightGBM model + metadata
    models/threshold_sweep.tsv       Sweep table  (threshold, precision, recall, f05)
    models/training_report.txt       Summary report
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).resolve().parents[4]
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from normalize   import normalize_name, normalize_address  # noqa: E402
from features    import (                                   # noqa: E402
    FEATURE_NAMES,
    extract_pair_features,
    build_lookup_fast,
    candidate_tsv_to_pairs,
)
from metrics     import compute_f05                        # noqa: E402

TRAIN_DIR   = ROOT / "dataset" / "train"
DATA_DIR    = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR  = ROOT / "submission" / "output"
MODELS_DIR  = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

LGBM_PARAMS: Dict = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       127,
    "max_depth":        -1,
    "learning_rate":    0.05,
    "n_estimators":     1000,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}

EARLY_STOPPING_ROUNDS = 50
THRESHOLD_RANGE       = np.arange(0.30, 0.96, 0.05)
BATCH_FEAT_SIZE       = 200_000   # pairs per feature-extraction chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source_with_norm(path: Path, label: str) -> pd.DataFrame:
    print(f"  Loading {label} …", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    df["norm_name"]    = df["business_name"].apply(normalize_name)
    df["norm_address"] = df["business_address"].apply(normalize_address)
    print(f"    → {len(df):,} rows  ({time.perf_counter()-t0:.1f}s)", flush=True)
    return df


def load_gt_sets(path: Path) -> Dict[str, Set[str]]:
    gt: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw   = row.get("matched_entity_ids", "").strip()
            gt[s1_id] = {m.strip() for m in raw.split(",") if m.strip()} if raw else set()
    return gt


def load_id_set(path: Path) -> Set[str]:
    return set(path.read_text(encoding="utf-8").splitlines())


# ---------------------------------------------------------------------------
# Feature extraction (chunked for RAM efficiency)
# ---------------------------------------------------------------------------

def extract_features_chunked(
    pairs_df: pd.DataFrame,
    s1_lookup: Dict[str, Tuple[str, str]],
    pool_lookup: Dict[str, Tuple[str, str]],
    chunk_size: int = BATCH_FEAT_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for all pairs in chunks.

    Returns
    -------
    X            : float32 array (n_valid, n_features)
    y            : int8 array   (n_valid,)  – empty if pairs_df has no 'label'
    valid_indices: int64 array  (n_valid,)  – positional row-indices in pairs_df
                   that had successful lookups; use to re-align pairs_df with X.
    """
    Xs: List[np.ndarray]  = []
    ys: List[np.ndarray]  = []
    valid_idx: List[int]  = []
    has_label = "label" in pairs_df.columns
    n_total   = len(pairs_df)

    for start in range(0, n_total, chunk_size):
        chunk = pairs_df.iloc[start : start + chunk_size]
        rows  = []
        lbls  = []
        chunk_valid: List[int] = []

        for pos, (_, row) in enumerate(chunk.iterrows()):
            s1_id   = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]
            s1e     = s1_lookup.get(s1_id)
            ce      = pool_lookup.get(cand_id)

            if s1e is None or ce is None:
                continue

            feat = extract_pair_features(
                s1e[0], s1e[1],
                ce[0],  ce[1],
                cand_id,
            )
            rows.append(feat)
            chunk_valid.append(start + pos)
            if has_label:
                lbls.append(int(row["label"]))

        if rows:
            Xs.append(np.vstack(rows))
            valid_idx.extend(chunk_valid)
            if has_label:
                ys.append(np.array(lbls, dtype=np.int8))

        if (start // chunk_size) % 5 == 0 or start + chunk_size >= n_total:
            pct = min(100, 100 * (start + chunk_size) / n_total)
            print(f"    Feature extraction: {pct:.0f}%  ({start+len(chunk):,}/{n_total:,} pairs)", flush=True)

    X = np.vstack(Xs) if Xs else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.concatenate(ys) if ys else np.array([], dtype=np.int8)
    return X, y, np.array(valid_idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(
    model: lgb.LGBMClassifier,
    X_val: np.ndarray,
    pairs_val: pd.DataFrame,
    gt_val: Dict[str, Set[str]],
    gt_full: Dict[str, Set[str]],   # all S1 entities (for singletons)
    val_ids: Set[str],
) -> Tuple[float, pd.DataFrame]:
    """Sweep thresholds and return (best_threshold, sweep_table_df)."""
    print("\n  Computing val probabilities …", flush=True)
    probs = model.predict_proba(X_val)[:, 1]

    # Map pair indices to probabilities
    pair_idx_col = list(range(len(pairs_val)))  # already filtered to valid pairs
    # Rebuild per-entity predictions for each threshold
    s1_ids_val = pairs_val["source1_entity_id"].values
    cand_ids_val = pairs_val["candidate_entity_id"].values

    # Build full GT for val entities (including singletons)
    gt_eval: Dict[str, List[str]] = {
        s1_id: list(gt_full.get(s1_id, set()))
        for s1_id in val_ids
    }

    rows = []
    best_f05     = -1.0
    best_thresh  = 0.50

    print(f"\n  {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'Macro F0.5':>12}", flush=True)
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}", flush=True)

    for thresh in THRESHOLD_RANGE:
        # Build predictions dict for this threshold
        preds: Dict[str, List[str]] = defaultdict(list)
        for s1_id in val_ids:
            preds[s1_id] = []   # ensure every val entity has a row (handles singletons)

        for i, (s1_id, cand_id) in enumerate(zip(s1_ids_val, cand_ids_val)):
            if probs[i] >= thresh:
                preds[s1_id].append(cand_id)

        result = compute_f05(dict(preds), gt_eval)

        f05 = result["macro_f05"]
        p   = result["macro_precision"]
        r   = result["macro_recall"]

        marker = " ◀ best" if f05 > best_f05 else ""
        print(f"  {thresh:>10.2f}  {p:>10.4f}  {r:>10.4f}  {f05:>12.6f}{marker}", flush=True)

        rows.append({"threshold": round(float(thresh), 2), "precision": p, "recall": r, "macro_f05": f05})

        if f05 > best_f05:
            best_f05    = f05
            best_thresh = float(thresh)

    sweep_df = pd.DataFrame(rows)
    print(f"\n  ★ Best threshold: {best_thresh:.2f}  (macro F0.5 = {best_f05:.6f})", flush=True)
    return best_thresh, sweep_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    divider = "═" * 68
    print(f"\n{divider}")
    print("  FEATURE ENGINEERING & LGBM TRAINING")
    print(divider)

    # ------------------------------------------------------------------ load
    train_s1 = load_source_with_norm(TRAIN_DIR / "train_source1.tsv", "train_source1")
    train_s2 = load_source_with_norm(TRAIN_DIR / "train_source2.tsv", "train_source2")
    train_s3 = load_source_with_norm(TRAIN_DIR / "train_source3.tsv", "train_source3")

    s1_lookup   = build_lookup_fast(train_s1)
    pool_df     = pd.concat([train_s2, train_s3], ignore_index=True)
    pool_lookup = build_lookup_fast(pool_df)
    del train_s2, train_s3, pool_df
    gc.collect()

    gt_full    = load_gt_sets(TRAIN_DIR / "train_ground_truth.tsv")
    train_ids  = load_id_set(DATA_DIR / "train_ids.txt")
    val_ids    = load_id_set(DATA_DIR / "val_ids.txt")

    # Ground-truth sets split by fold
    gt_train   = {k: v for k, v in gt_full.items() if k in train_ids}
    gt_val     = {k: v for k, v in gt_full.items() if k in val_ids}

    # ---------------------------------------------------------- train pairs
    cand_train_path = OUTPUT_DIR / "candidate_pairs_train.tsv"
    if not cand_train_path.exists():
        raise FileNotFoundError(
            f"Train candidate pairs not found: {cand_train_path}\n"
            "Run blocking.py --mode train first."
        )

    print(f"\n  Loading train candidate pairs from {cand_train_path.name} …")
    pairs_train_raw = candidate_tsv_to_pairs(cand_train_path, gt=gt_train)
    pairs_train     = pairs_train_raw[pairs_train_raw["source1_entity_id"].isin(train_ids)]
    print(f"  Train pairs : {len(pairs_train):,}  "
          f"(positives: {pairs_train['label'].sum():,}, "
          f"negatives: {(pairs_train['label']==0).sum():,})")

    # ---------------------------------------------------------- val pairs
    print(f"\n  Loading val candidate pairs …")
    pairs_val_raw = candidate_tsv_to_pairs(cand_train_path, gt=gt_val)
    pairs_val     = pairs_val_raw[pairs_val_raw["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    print(f"  Val pairs   : {len(pairs_val):,}  "
          f"(positives: {pairs_val['label'].sum():,})")

    # ------------------------------------------------------- feature extraction
    print(f"\n{divider}")
    print("  EXTRACTING FEATURES — Train split")
    print(divider)
    X_train, y_train, _train_valid_idx = extract_features_chunked(pairs_train, s1_lookup, pool_lookup)
    print(f"  X_train shape: {X_train.shape}  positives: {y_train.sum():,}")

    print(f"\n{divider}")
    print("  EXTRACTING FEATURES — Val split")
    print(divider)
    X_val, y_val, val_valid_idx = extract_features_chunked(pairs_val, s1_lookup, pool_lookup)
    # Align pairs_val to exactly the rows that had successful lookups.
    # val_valid_idx contains positional indices into pairs_val (reset_index above
    # ensures iloc positions match logical positions).
    pairs_val_valid = pairs_val.iloc[val_valid_idx].reset_index(drop=True)
    n_skipped_val = len(pairs_val) - len(pairs_val_valid)
    if n_skipped_val > 0:
        print(f"  ⚠  Val pairs skipped (entity not in lookup): {n_skipped_val:,}")
    print(f"  X_val shape  : {X_val.shape}  positives: {y_val.sum():,}")

    # ----------------------------------------------------------------- train
    print(f"\n{divider}")
    print("  TRAINING LightGBM CLASSIFIER")
    print(divider)

    # Class weight to handle imbalance (many more negatives)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  Positives: {n_pos:,}  Negatives: {n_neg:,}  scale_pos_weight: {scale_pos_weight:.2f}")

    params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}

    model = lgb.LGBMClassifier(**params)

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Training done in {elapsed:.1f}s  |  best iteration: {model.best_iteration_}")

    # -------------------------------------------------------- feature importance
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: -x[1],
    )
    print("\n  Feature importances (top 10):")
    for fname, imp in importances[:10]:
        bar = "█" * int(40 * imp / max(i for _, i in importances))
        print(f"    {fname:<30s} {imp:>6}  {bar}")

    # ------------------------------------------------------- threshold sweep
    print(f"\n{divider}")
    print("  THRESHOLD SWEEP ON VALIDATION SET")
    print(divider)

    # pairs_val_valid is already aligned row-for-row with X_val (same valid_idx filter).
    best_thresh, sweep_df = threshold_sweep(
        model, X_val, pairs_val_valid,
        gt_val, gt_full, val_ids,
    )

    # ----------------------------------------------------------------- save
    print(f"\n{divider}")
    print("  SAVING ARTEFACTS")
    print(divider)

    model_path = MODELS_DIR / "matcher.joblib"
    joblib.dump({
        "model":          model,
        "threshold":      best_thresh,
        "feature_names":  FEATURE_NAMES,
        "lgbm_params":    params,
        "best_iteration": model.best_iteration_,
    }, model_path)
    print(f"  Model saved → {model_path.relative_to(ROOT)}")

    sweep_path = MODELS_DIR / "threshold_sweep.tsv"
    sweep_df.to_csv(sweep_path, sep="\t", index=False)
    print(f"  Sweep table → {sweep_path.relative_to(ROOT)}")

    # Training report
    report_lines = [
        "=" * 68,
        "  TRAINING REPORT",
        "=" * 68,
        f"  Train pairs          : {len(X_train):,}",
        f"  Val   pairs          : {len(X_val):,}",
        f"  Positive rate (train): {n_pos/len(y_train)*100:.2f}%",
        f"  LightGBM params      : {json.dumps(params, indent=4)}",
        f"  Best LGB iteration   : {model.best_iteration_}",
        "",
        "  Feature importances (all):",
    ]
    for fname, imp in importances:
        report_lines.append(f"    {fname:<30s} {imp:>6}")
    report_lines += [
        "",
        f"  Best threshold  : {best_thresh:.2f}",
        f"  Best macro F0.5 : {sweep_df.loc[sweep_df['macro_f05'].idxmax(), 'macro_f05']:.6f}",
        "=" * 68,
    ]

    report_path = MODELS_DIR / "training_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Report     → {report_path.relative_to(ROOT)}")

    print(f"\n{divider}")
    print(f"  DONE.  Best threshold = {best_thresh:.2f}")
    print(divider + "\n")


if __name__ == "__main__":
    main()
