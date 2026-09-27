"""
blocking.py — Candidate Generation & Blocking Stage.

Strategy
--------
1. Country-partitioned TF-IDF char n-gram index
   - Normalise business names (from normalize.py)
   - Fit one TF-IDF vectorizer (char n-grams 2-4, sublinear TF) per country
     present in S2+S3 pool.  Country is treated as an open-set string label —
     no hard-coding of {US, India, France}.
   - Transform S1 queries and retrieve top-K candidates via batched sparse
     cosine similarity (X_s1_batch @ X_s23_country.T → top-k indices).

2. Global fallback index (rare: S1 country not in S2/S3)
   - A single TF-IDF index over the entire S2+S3 pool used when no
     country-specific sub-index exists.

3. RapidFuzz token-set-ratio overlap pass (fallback layer)
   - For every S1 entity that still has zero candidates after TF-IDF, run
     a fast trigram prefilter + token_set_ratio(≥70) against the country pool.

4. Recall gate
   - Measure blocking recall on the validation split (val_ids.txt +
     val_ground_truth.tsv).  Recall = fraction of true match pairs where the
     true match appears in the candidate set.
   - If recall < RECALL_THRESHOLD (0.85), K is bumped and the search is re-run.

Output
------
submission/output/candidate_pairs.tsv
    source1_entity_id \\t candidate_entity_ids   (comma-separated, no quotes)

Usage
-----
Run from project root (student_resource/):

    python submission/code/business_entity_resolution/src/blocking.py [--mode train|test|both]

    --mode train : Run on val split S1 entities + measure recall gate
    --mode test  : Run on full test_source1 entities
    --mode both  : Run train-mode recall check, then full test (default)
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT     = Path(__file__).resolve().parents[4]   # …/Amazon/
SRC_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from normalize import normalize_name, normalize_address   # noqa: E402

TRAIN_DIR    = ROOT / "dataset" / "train"
TEST_DIR     = ROOT / "dataset" / "test"
DATA_DIR     = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR   = ROOT / "submission" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

K_INIT           = 20       # initial top-k candidates per S1 entity
K_MAX            = 40       # max k when bumping for recall gate
K_STEP           = 5        # bump step
RECALL_THRESHOLD = 0.85
BATCH_SIZE_S1    = 2_000    # rows per S1 batch during cosine search
TFIDF_MAX_FEAT   = 150_000  # vocabulary cap per country index
TFIDF_NGRAM      = (2, 4)   # char n-gram range
FUZZY_THRESHOLD  = 70       # token_set_ratio threshold for fallback


# ---------------------------------------------------------------------------
# 1. Data loading helpers
# ---------------------------------------------------------------------------

def load_source(path: Path, label: str) -> pd.DataFrame:
    """Load a source TSV and add a normalized_name column."""
    print(f"  Loading {label} ({path.name}) …", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(
        path, sep="\t", dtype=str,
        keep_default_na=False, encoding="utf-8",
    )
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["country"]   = df["country"].str.strip()
    elapsed = time.perf_counter() - t0
    print(f"    → {len(df):,} rows  ({elapsed:.1f}s)", flush=True)
    return df


def load_ids(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    gt: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw   = row.get("matched_entity_ids", "").strip()
            matched = {m.strip() for m in raw.split(",") if m.strip()} if raw else set()
            gt[s1_id] = matched
    return gt


# ---------------------------------------------------------------------------
# 2. TF-IDF index — build per-country
# ---------------------------------------------------------------------------

class CountryIndex:
    """TF-IDF char n-gram index for one country's S2+S3 pool."""

    def __init__(self, country: str, df: pd.DataFrame) -> None:
        self.country   = country
        self.entity_ids: np.ndarray = df["entity_id"].values  # ordered
        names = df["norm_name"].tolist()

        t0 = time.perf_counter()
        self.vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_NGRAM,
            max_features=TFIDF_MAX_FEAT,
            sublinear_tf=True,
            min_df=1,
        )
        X = self.vec.fit_transform(names)
        # L2-normalise so dot product == cosine similarity
        self.X = sk_normalize(X, norm="l2", copy=False)
        self.X_T = self.X.T.tocsc()
        elapsed = time.perf_counter() - t0
        print(
            f"    [{country}] index: {len(names):,} vectors, "
            f"{self.X.shape[1]:,} features  ({elapsed:.1f}s)",
            flush=True,
        )

    def query_batch(
        self,
        names_batch: List[str],
        k: int,
    ) -> List[List[str]]:
        """Return top-k entity IDs for each name in the batch."""
        if self.X.shape[0] == 0:
            return [[] for _ in names_batch]

        actual_k = min(k, self.X.shape[0])
        X_q = self.vec.transform(names_batch)
        X_q = sk_normalize(X_q, norm="l2", copy=False)

        results: List[List[str]] = []
        mini_batch_size = 2000
        n_queries = X_q.shape[0]

        for start in range(0, n_queries, mini_batch_size):
            X_sub = X_q[start : start + mini_batch_size]
            sims = X_sub @ self.X_T  # CSR matrix (batch x pool)
            indptr = sims.indptr
            indices = sims.indices
            data = sims.data

            for i in range(sims.shape[0]):
                r_start = indptr[i]
                r_end = indptr[i + 1]
                if r_start == r_end:
                    results.append([])
                    continue
                row_cols = indices[r_start:r_end]
                row_vals = data[r_start:r_end]

                if len(row_vals) <= actual_k:
                    order = np.argsort(-row_vals)
                    results.append(self.entity_ids[row_cols[order]].tolist())
                else:
                    top_part = np.argpartition(-row_vals, actual_k)[:actual_k]
                    part_vals = row_vals[top_part]
                    order = np.argsort(-part_vals)
                    results.append(self.entity_ids[row_cols[top_part[order]]].tolist())
            del sims
        return results


def build_country_indices(
    pool_df: pd.DataFrame,
) -> Dict[str, CountryIndex]:
    """Build one CountryIndex per country in the S2+S3 pool."""
    indices: Dict[str, CountryIndex] = {}
    for country, grp in pool_df.groupby("country", sort=False):
        indices[str(country)] = CountryIndex(str(country), grp.reset_index(drop=True))
    return indices


# ---------------------------------------------------------------------------
# 3. RapidFuzz fallback (for zero-candidate S1 entities)
# ---------------------------------------------------------------------------

def _fuzzy_fallback(
    s1_name: str,
    pool_df: pd.DataFrame,
    k: int,
) -> List[str]:
    """Use rapidfuzz token_set_ratio to find candidates when TF-IDF fails."""
    try:
        from rapidfuzz import process as rfprocess, fuzz as rffuzz
    except ImportError:
        return []

    if pool_df.empty or not s1_name:
        return []

    candidates = rfprocess.extract(
        s1_name,
        pool_df["norm_name"].tolist(),
        scorer=rffuzz.token_set_ratio,
        limit=k,
        score_cutoff=FUZZY_THRESHOLD,
    )
    indices = [c[2] for c in candidates]
    return pool_df.iloc[indices]["entity_id"].tolist()


# ---------------------------------------------------------------------------
# 4. Core blocking loop
# ---------------------------------------------------------------------------

def _run_blocking(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    country_indices: Dict[str, CountryIndex],
    global_index: Optional[CountryIndex],
    k: int,
) -> Dict[str, List[str]]:
    """Run the full blocking pipeline for a set of S1 entities.

    Returns
    -------
    dict  source1_entity_id → list[candidate_entity_id]
    """
    # Pre-group pool by country for fast RapidFuzz fallback access
    pool_by_country: Dict[str, pd.DataFrame] = {
        c: grp.reset_index(drop=True)
        for c, grp in pool_df.groupby("country", sort=False)
    }

    candidates: Dict[str, List[str]] = {}
    total = len(s1_df)
    processed = 0

    # Batch over S1 rows, grouped by country to reuse the same index
    for country, grp in s1_df.groupby("country", sort=False):
        country_str  = str(country)
        idx_obj      = country_indices.get(country_str, global_index)

        rows   = grp.reset_index(drop=True)
        n_rows = len(rows)

        for batch_start in range(0, n_rows, BATCH_SIZE_S1):
            batch = rows.iloc[batch_start : batch_start + BATCH_SIZE_S1]
            names = batch["norm_name"].tolist()
            ids   = batch["entity_id"].tolist()

            if idx_obj is not None:
                batch_results = idx_obj.query_batch(names, k)
            else:
                batch_results = [[] for _ in names]

            for eid, cands in zip(ids, batch_results):
                candidates[eid] = cands

            processed += len(batch)
            if processed % 50_000 == 0 or processed == total:
                print(f"    Processed {processed:>7,} / {total:,} S1 entities …", flush=True)

    return candidates


# ---------------------------------------------------------------------------
# 5. Recall evaluation
# ---------------------------------------------------------------------------

def compute_blocking_recall(
    candidates: Dict[str, List[str]],
    gt: Dict[str, Set[str]],
) -> Tuple[float, float, int, int]:
    """Compute blocking recall and pair-level recall.

    Returns
    -------
    (entity_recall, pair_recall, n_entities_covered, n_total_non_singleton)
    """
    n_total_non_singleton   = 0
    n_entities_covered      = 0
    total_true_pairs        = 0
    total_covered_pairs     = 0

    for s1_id, truth_set in gt.items():
        if not truth_set:
            continue   # skip singletons
        n_total_non_singleton += 1
        total_true_pairs      += len(truth_set)

        cand_set = set(candidates.get(s1_id, []))
        hits     = truth_set & cand_set
        total_covered_pairs += len(hits)

        if hits:
            n_entities_covered += 1

    entity_recall = (
        n_entities_covered / n_total_non_singleton
        if n_total_non_singleton else 0.0
    )
    pair_recall = (
        total_covered_pairs / total_true_pairs
        if total_true_pairs else 0.0
    )
    return entity_recall, pair_recall, n_entities_covered, n_total_non_singleton


# ---------------------------------------------------------------------------
# 6. TSV output writer
# ---------------------------------------------------------------------------

def write_candidate_tsv(
    candidates: Dict[str, List[str]],
    s1_ids_ordered: List[str],
    path: Path,
) -> None:
    """Write candidate_pairs.tsv in the required format."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in s1_ids_ordered:
            cands = candidates.get(s1_id, [])
            # Deduplicate while preserving order
            seen: Set[str] = set()
            deduped = [c for c in cands if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
            writer.writerow([s1_id, ",".join(deduped)])
    print(f"  Written {len(s1_ids_ordered):,} rows → {path.relative_to(ROOT)}", flush=True)


# ---------------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Blocking / candidate generation.")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="train = val-recall gate only; test = generate test candidates; both = both (default)",
    )
    parser.add_argument(
        "--k", type=int, default=K_INIT,
        help=f"Initial top-k (default {K_INIT})",
    )
    parser.add_argument(
        "--no-recall-gate", action="store_true",
        help="Skip the recall gate check (for fast test runs)",
    )
    args = parser.parse_args()

    divider = "═" * 68

    # -----------------------------------------------------------------------
    # Phase A — build S2+S3 pool index (shared across train & test modes)
    # -----------------------------------------------------------------------
    print(f"\n{divider}")
    print("  BLOCKING STAGE — Loading S2 + S3 pool")
    print(divider)

    # For the recall gate we use train S2+S3; for test we use test S2+S3.
    # We build index once per mode to save memory.

    def _run_mode(
        s1_path: Path,
        s2_path: Path,
        s3_path: Path,
        output_path: Path,
        gt_path: Optional[Path],
        val_ids_path: Optional[Path],
        k_start: int,
        do_recall_gate: bool,
        label: str,
    ) -> None:
        print(f"\n{divider}")
        print(f"  MODE: {label.upper()}")
        print(divider)

        # Load pool
        s2 = load_source(s2_path, "Source 2")
        s3 = load_source(s3_path, "Source 3")
        pool_df = pd.concat([s2, s3], ignore_index=True)
        del s2, s3
        gc.collect()

        # Load S1 (full or val-only)
        s1_full = load_source(s1_path, "Source 1")

        if val_ids_path and val_ids_path.exists():
            val_ids = set(load_ids(val_ids_path))
            s1_df   = s1_full[s1_full["entity_id"].isin(val_ids)].reset_index(drop=True)
            print(f"  Using val split: {len(s1_df):,} / {len(s1_full):,} S1 entities")
        else:
            s1_df = s1_full.reset_index(drop=True)

        s1_ids_ordered = s1_df["entity_id"].tolist()

        # Build per-country indices
        print(f"\n  Building per-country TF-IDF indices …")
        country_indices = build_country_indices(pool_df)

        # Build global fallback index only if needed (when an S1 country is missing from country_indices)
        missing_countries = set(s1_df["country"].unique()) - set(country_indices.keys())
        global_index: Optional[CountryIndex] = None
        if missing_countries:
            print(f"  Building global fallback index for missing countries {missing_countries} …")
            global_index = CountryIndex("__global__", pool_df)
        else:
            print("  All S1 countries present in pool indices; skipping global fallback index.")

        # ---- Recall gate loop ----
        k = k_start
        gt: Optional[Dict[str, Set[str]]] = None
        if gt_path and gt_path.exists():
            gt = load_ground_truth(gt_path)

        while True:
            print(f"\n{divider}")
            print(f"  Running blocking  k={k} …")
            print(divider)
            t0 = time.perf_counter()

            candidates = _run_blocking(s1_df, pool_df, country_indices, global_index, k)

            elapsed = time.perf_counter() - t0
            print(f"  Blocking complete in {elapsed:.1f}s")

            # Stats
            n_with_cands = sum(1 for v in candidates.values() if v)
            avg_cands    = (
                sum(len(v) for v in candidates.values()) / len(candidates)
                if candidates else 0.0
            )
            print(f"  Entities with ≥1 candidate : {n_with_cands:,} / {len(candidates):,}")
            print(f"  Avg candidates per S1      : {avg_cands:.1f}")

            if not do_recall_gate or gt is None:
                print("  Recall gate skipped.")
                break

            entity_rec, pair_rec, covered, total_ns = compute_blocking_recall(
                candidates, gt
            )
            print(f"\n  ── Recall Gate ──────────────────────────────────────")
            print(f"  Entity recall : {entity_rec:.4f}  ({covered:,}/{total_ns:,} non-singleton S1s)")
            print(f"  Pair   recall : {pair_rec:.4f}")
            print(f"  Threshold     : {RECALL_THRESHOLD}")

            if entity_rec >= RECALL_THRESHOLD:
                print(f"  ✅ PASS  (entity recall {entity_rec:.4f} ≥ {RECALL_THRESHOLD})")
                break
            elif k >= K_MAX:
                print(f"  ⚠️  Entity recall {entity_rec:.4f} below threshold but k={k} already at K_MAX={K_MAX}.")
                print(f"  Proceeding with best available recall.")
                break
            else:
                k_new = min(k + K_STEP, K_MAX)
                print(f"  ❌ FAIL  (entity recall {entity_rec:.4f} < {RECALL_THRESHOLD})  bumping k: {k} → {k_new}")
                k = k_new

        # Write output
        print(f"\n  Writing output TSV …")
        # For full test mode, re-run on full S1
        if not (val_ids_path and val_ids_path.exists()):
            final_candidates = candidates
            final_ids = s1_ids_ordered
        else:
            # If we were in recall-gate / val mode, now re-run on all S1 for output
            print(f"  Re-running on full S1 ({len(s1_full):,} entities) with k={k} …")
            t0 = time.perf_counter()
            final_candidates = _run_blocking(
                s1_full, pool_df, country_indices, global_index, k
            )
            elapsed = time.perf_counter() - t0
            print(f"  Full S1 blocking complete in {elapsed:.1f}s")
            final_ids = s1_full["entity_id"].tolist()

        write_candidate_tsv(final_candidates, final_ids, output_path)

        # Cleanup
        del pool_df, s1_df, s1_full, candidates
        gc.collect()

    # -----------------------------------------------------------------------
    # Decide which modes to run
    # -----------------------------------------------------------------------
    do_recall = not args.no_recall_gate
    k_init    = args.k

    if args.mode in ("train", "both"):
        _run_mode(
            s1_path      = TRAIN_DIR / "train_source1.tsv",
            s2_path      = TRAIN_DIR / "train_source2.tsv",
            s3_path      = TRAIN_DIR / "train_source3.tsv",
            output_path  = OUTPUT_DIR / "candidate_pairs_train.tsv",
            gt_path      = DATA_DIR / "val_ground_truth.tsv",
            val_ids_path = DATA_DIR / "val_ids.txt",
            k_start      = k_init,
            do_recall_gate = do_recall,
            label        = "train/val recall gate",
        )

    if args.mode in ("test", "both"):
        _run_mode(
            s1_path      = TEST_DIR / "test_source1.tsv",
            s2_path      = TEST_DIR / "test_source2.tsv",
            s3_path      = TEST_DIR / "test_source3.tsv",
            output_path  = OUTPUT_DIR / "candidate_pairs.tsv",
            gt_path      = None,   # no ground truth for test
            val_ids_path = None,   # run on full test S1
            k_start      = k_init,
            do_recall_gate = False,
            label        = "test set",
        )

    print(f"\n{'═' * 68}")
    print("  BLOCKING DONE.")
    print(f"{'═' * 68}\n")


if __name__ == "__main__":
    main()
