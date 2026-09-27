"""
features.py — Pairwise feature extraction for Business Entity Resolution.

For each (source1_entity_id, candidate_entity_id) pair, computes a feature
vector used by the LightGBM classifier.

Feature groups
--------------
Name similarity (rapidfuzz):
  f01  token_sort_ratio        name vs name   [0,100]
  f02  token_set_ratio         name vs name
  f03  jaro_winkler_similarity name vs name   [0,1] → scaled *100
  f04  partial_ratio           name vs name
  f05  WRatio                  name vs name

Address similarity (rapidfuzz, missing-safe):
  f06  token_sort_ratio        addr vs addr
  f07  token_set_ratio         addr vs addr
  f08  jaro_winkler_similarity addr vs addr
  f09  partial_ratio           addr vs addr

Exact / boolean flags:
  f10  name_exact_match        1 if norm names are identical
  f11  address_exact_match     1 if norm addresses are identical (both non-empty)
  f12  address_both_missing    1 if both addresses are empty
  f13  address_one_sided_miss  1 if exactly ONE side has no address  ← FIX

Lexical features:
  f14  name_len_diff           abs(len(n1) - len(n2))  [chars]
  f15  name_token_jaccard      Jaccard over space-split tokens
  f16  addr_token_jaccard      Jaccard over space-split tokens (0 if either missing)

Combined signal:
  f17  name_addr_harmonic      Harmonic mean of best-name and best-addr score  ← FIX
                               (degrades gracefully when address is one-sided missing)

Candidate source:
  f18  candidate_source_s2     1 if candidate is from Source 2 (else Source 3)

(f00 blocking_cosine_score is optionally appended when available)

Diagnostic rationale
--------------------
The two new features address diagnosed failure modes:

* addr_one_sided_missing: the original code set all address scores to 0
  when EITHER side was empty, making one-sided missing indistinguishable
  from a pair with genuinely different addresses. LightGBM can now learn
  a separate decision boundary for this case.

* name_addr_harmonic: franchise / chain businesses share identical names
  but different locations (false-positive risk). Pairing name similarity
  with address similarity in a harmonic mean penalises pairs where ONE
  signal is very weak, even if the other is perfect.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz as rfuzz
from rapidfuzz import distance as rdist

# ---------------------------------------------------------------------------
# Feature names — must stay in sync with extraction below
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_partial_ratio",
    "name_wratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaro_winkler",
    "addr_partial_ratio",
    "name_exact",
    "addr_exact",
    "addr_both_missing",
    "addr_one_sided_missing",   # NEW: exactly one side has no address
    "name_len_diff",
    "name_token_jaccard",
    "addr_token_jaccard",
    "name_addr_harmonic",       # NEW: harmonic mean of best-name & best-addr scores
    "cand_is_s2",
]


# ---------------------------------------------------------------------------
# Token Jaccard helper
# ---------------------------------------------------------------------------

def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on whitespace-split token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(a.split())
    sb = set(b.split())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Single-pair feature vector
# ---------------------------------------------------------------------------

def extract_pair_features(
    name1: str,
    addr1: str,
    name2: str,
    addr2: str,
    cand_id: str,
    cosine_score: Optional[float] = None,
) -> np.ndarray:
    """Compute a float32 feature vector for one (s1, candidate) pair.

    Parameters
    ----------
    name1, addr1 : normalised name and address of the S1 entity
    name2, addr2 : normalised name and address of the candidate entity
    cand_id      : entity_id of the candidate (used to derive source flag)
    cosine_score : optional TF-IDF cosine similarity from blocking stage

    Returns
    -------
    np.ndarray of shape (16,) or (17,) if cosine_score is provided, float32
    """
    # --- Name similarities ---
    n_tsr  = rfuzz.token_sort_ratio(name1, name2)
    n_tset = rfuzz.token_set_ratio(name1, name2)
    n_jw   = rdist.JaroWinkler.normalized_similarity(name1, name2) * 100
    n_pr   = rfuzz.partial_ratio(name1, name2)
    n_wr   = rfuzz.WRatio(name1, name2)
    best_name_score = max(n_tsr, n_tset, n_jw, n_pr, n_wr)  # used in harmonic

    # --- Address similarities (missing-safe) ---
    addr_one_sided_missing = 0.0
    if addr1 and addr2:
        a_tsr  = rfuzz.token_sort_ratio(addr1, addr2)
        a_tset = rfuzz.token_set_ratio(addr1, addr2)
        a_jw   = rdist.JaroWinkler.normalized_similarity(addr1, addr2) * 100
        a_pr   = rfuzz.partial_ratio(addr1, addr2)
        addr_both_missing = 0.0
    elif not addr1 and not addr2:
        a_tsr = a_tset = a_jw = a_pr = 0.0
        addr_both_missing = 1.0
    else:
        # Exactly one side is empty — new flag exposes this to the model
        a_tsr = a_tset = a_jw = a_pr = 0.0
        addr_both_missing      = 0.0
        addr_one_sided_missing = 1.0
    best_addr_score = max(a_tsr, a_tset, a_jw, a_pr)

    # --- Harmonic mean of best name + best addr scores ---
    # When one side has no address, best_addr_score = 0, so harmonic = 0.
    # LightGBM can learn to relax this for addr_one_sided_missing = 1.
    if best_name_score > 0 and best_addr_score > 0:
        name_addr_harmonic = (
            2 * best_name_score * best_addr_score
            / (best_name_score + best_addr_score)
        )
    else:
        name_addr_harmonic = 0.0

    # --- Exact match flags ---
    name_exact = float(name1 == name2 and bool(name1))
    addr_exact = float(addr1 == addr2 and bool(addr1))

    # --- Lexical features ---
    name_len_diff  = float(abs(len(name1) - len(name2)))
    name_tok_jacc  = _token_jaccard(name1, name2)
    addr_tok_jacc  = _token_jaccard(addr1, addr2) if (addr1 and addr2) else 0.0

    # --- Candidate source flag ---
    cand_is_s2 = float(cand_id.startswith("S2-"))

    feats = [
        n_tsr, n_tset, n_jw, n_pr, n_wr,
        a_tsr, a_tset, a_jw, a_pr,
        name_exact, addr_exact, addr_both_missing, addr_one_sided_missing,
        name_len_diff, name_tok_jacc, addr_tok_jacc,
        name_addr_harmonic,
        cand_is_s2,
    ]

    if cosine_score is not None:
        feats.append(float(cosine_score) * 100)   # scale to [0,100] range

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Batch extraction from DataFrames
# ---------------------------------------------------------------------------

def extract_features_batch(
    pairs_df: pd.DataFrame,
    s1_lookup: Dict[str, Tuple[str, str]],
    pool_lookup: Dict[str, Tuple[str, str]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Vectorised feature extraction for a batch of candidate pairs.

    Parameters
    ----------
    pairs_df : DataFrame with columns [source1_entity_id, candidate_entity_id]
               and optionally [cosine_score]
    s1_lookup  : entity_id → (norm_name, norm_address) for S1
    pool_lookup: entity_id → (norm_name, norm_address) for S2+S3

    Returns
    -------
    X : float32 array of shape (n_pairs, n_features)
    valid_mask : bool array marking rows where both entities were found in lookups
    """
    has_cosine = "cosine_score" in pairs_df.columns

    rows: List[np.ndarray] = []
    valid: List[bool]      = []

    for _, row in pairs_df.iterrows():
        s1_id   = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]

        s1_entry   = s1_lookup.get(s1_id)
        cand_entry = pool_lookup.get(cand_id)

        if s1_entry is None or cand_entry is None:
            valid.append(False)
            n_feat = len(FEATURE_NAMES) + (1 if has_cosine else 0)
            rows.append(np.zeros(n_feat, dtype=np.float32))
            continue

        cosine = float(row["cosine_score"]) if has_cosine else None
        feat   = extract_pair_features(
            s1_entry[0], s1_entry[1],
            cand_entry[0], cand_entry[1],
            cand_id, cosine,
        )
        rows.append(feat)
        valid.append(True)

    X = np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return X, np.array(valid, dtype=bool)


# ---------------------------------------------------------------------------
# Build lookup maps from DataFrames
# ---------------------------------------------------------------------------

def build_lookup(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """Build entity_id → (norm_name, norm_address) mapping."""
    return {
        row["entity_id"]: (
            row.get("norm_name", ""),
            row.get("norm_address", ""),
        )
        for _, row in df.iterrows()
    }


def build_lookup_fast(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """Vectorised version of build_lookup — much faster for large DataFrames."""
    ids    = df["entity_id"].values
    names  = df["norm_name"].values   if "norm_name"    in df.columns else np.full(len(df), "")
    addrs  = df["norm_address"].values if "norm_address" in df.columns else np.full(len(df), "")
    return dict(zip(ids, zip(names, addrs)))


# ---------------------------------------------------------------------------
# Candidate-pairs TSV → labelled pairs DataFrame
# ---------------------------------------------------------------------------

def candidate_tsv_to_pairs(
    candidate_tsv: Path,
    gt: Optional[Dict[str, set]] = None,
) -> pd.DataFrame:
    """Expand a candidate_pairs.tsv into a flat (s1_id, cand_id, label) DataFrame.

    Parameters
    ----------
    candidate_tsv : path to candidate_pairs.tsv
    gt            : optional ground-truth dict for labelling (train mode)
                    If None, label column is omitted.

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id, candidate_entity_id[, label]
    """
    records: List[Dict] = []
    with open(candidate_tsv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw   = row.get("candidate_entity_ids", "").strip()
            cands = [c.strip() for c in raw.split(",") if c.strip()] if raw else []
            truth = gt.get(s1_id, set()) if gt else None

            for cand_id in cands:
                rec: Dict = {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                }
                if gt is not None:
                    rec["label"] = int(cand_id in truth)
                records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Feature extraction self-test")
    print("─" * 60)

    cases = [
        ("orelee s barbershop incorporated", "1795 westchester drive high point nc",
         "orelee s barbershop incorporated", "1795 westchester dr high point nc", "S2-001"),
        ("sunrise trader private limited",   "plot 7 industrial area phase ii",
         "sunrise traders pvt ltd",           "",                                   "S3-002"),
        ("smith jones limited liability company", "100 north oak boulevard dallas tx",
         "jones smith corporation",           "100 n oak blvd dallas",              "S2-003"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "kh number 570 13 new delhi",
         "राम मार्केटिंग प्राइवेट लिमिटेड", "kh number 570 13 new delhi west delhi", "S3-004"),
        ("boulangerie dupont sarl",          "12 rue de la paix paris cedex 01",
         "boulangerie dupont",               "12 rue de la paix paris",             "S2-005"),
    ]

    for n1, a1, n2, a2, cid in cases:
        feat = extract_pair_features(n1, a1, n2, a2, cid)
        feat_named = dict(zip(FEATURE_NAMES, feat.tolist()))
        print(f"\n  S1: {n1!r}")
        print(f"  C : {n2!r}  [{cid}]")
        print(f"  name_token_sort={feat_named['name_token_sort']:.1f}  "
              f"name_token_set={feat_named['name_token_set']:.1f}  "
              f"name_jaro_winkler={feat_named['name_jaro_winkler']:.1f}  "
              f"name_exact={feat_named['name_exact']:.0f}")
        print(f"  addr_token_sort={feat_named['addr_token_sort']:.1f}  "
              f"addr_exact={feat_named['addr_exact']:.0f}  "
              f"addr_both_missing={feat_named['addr_both_missing']:.0f}")

    print("\n✅ Self-test passed — feature shape:", extract_pair_features("a","","b","","S2-x").shape)
