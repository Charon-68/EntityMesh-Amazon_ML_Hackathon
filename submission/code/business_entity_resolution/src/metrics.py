"""
metrics.py — F_0.5 evaluation metric for Business Entity Resolution.

The metric is macro-averaged F_0.5 across all Source-1 entities.

Scoring rules (from the challenge README):
  - For each S1 entity, compute per-entity precision, recall, and F_0.5.
  - Singletons (ground truth = empty set):
      * Predict empty  → F_0.5 = 1.0
      * Predict anything → F_0.5 = 0.0
  - Final score = mean of per-entity F_0.5 scores.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional


# ---------------------------------------------------------------------------
# Core F_0.5 helpers
# ---------------------------------------------------------------------------

def _f05_single(
    pred_set: Set[str],
    truth_set: Set[str],
) -> float:
    """Compute F_0.5 for a single Source-1 entity.

    Parameters
    ----------
    pred_set:
        Set of predicted matched entity IDs (S2-/S3-).
    truth_set:
        Set of ground-truth matched entity IDs (S2-/S3-).
        An empty set means this entity is a singleton.

    Returns
    -------
    float
        Per-entity F_0.5 score in [0, 1].
    """
    # --- Singleton special cases -------------------------------------------
    if not truth_set:
        # Ground truth has no matches → singleton entity.
        return 1.0 if not pred_set else 0.0

    # --- Standard P / R / F_0.5 -------------------------------------------
    if not pred_set:
        # Predicted nothing but ground truth is non-empty → Recall = 0
        return 0.0

    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set)
    recall = tp / len(truth_set)

    if precision == 0 and recall == 0:
        return 0.0

    # F_β with β = 0.5  →  (1 + β²) * P * R / (β² * P + R)
    #                   →  1.25 * P * R / (0.25 * P + R)
    beta_sq = 0.25  # β² = 0.5² = 0.25
    f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return f05


def compute_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Compute macro-averaged F_0.5 score.

    Parameters
    ----------
    predictions:
        Mapping  source1_entity_id → list of predicted matched IDs.
        Entities with no predicted matches should map to an empty list.
    ground_truth:
        Mapping  source1_entity_id → list of ground-truth matched IDs.
        Singletons (entities with no match) map to an empty list.

    Returns
    -------
    dict with keys:
        "macro_f05"  – final leaderboard score
        "macro_precision"
        "macro_recall"
        "n_entities"
        "n_singletons"        – GT singletons
        "n_correct_singletons"– GT singletons correctly predicted empty
        "per_entity"          – dict[entity_id, f05_score] (full breakdown)
    """
    per_entity: Dict[str, float] = {}
    per_entity_p: Dict[str, float] = {}
    per_entity_r: Dict[str, float] = {}

    n_singletons = 0
    n_correct_singletons = 0

    for s1_id, truth_list in ground_truth.items():
        truth_set = set(truth_list) if truth_list else set()
        pred_list = predictions.get(s1_id, [])
        pred_set = set(pred_list) if pred_list else set()

        if not truth_set:
            n_singletons += 1
            if not pred_set:
                n_correct_singletons += 1

        per_entity[s1_id] = _f05_single(pred_set, truth_set)

        # Precision / recall for summary (0 for singletons)
        if truth_set:
            tp = len(pred_set & truth_set)
            per_entity_p[s1_id] = tp / len(pred_set) if pred_set else 0.0
            per_entity_r[s1_id] = tp / len(truth_set)
        else:
            per_entity_p[s1_id] = 1.0 if not pred_set else 0.0
            per_entity_r[s1_id] = 1.0 if not pred_set else 0.0

    n = len(per_entity)
    macro_f05 = sum(per_entity.values()) / n if n else 0.0
    macro_p = sum(per_entity_p.values()) / n if n else 0.0
    macro_r = sum(per_entity_r.values()) / n if n else 0.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "n_entities": n,
        "n_singletons": n_singletons,
        "n_correct_singletons": n_correct_singletons,
        "per_entity": per_entity,
    }


# ---------------------------------------------------------------------------
# I/O helpers — load predictions / GT from TSV files
# ---------------------------------------------------------------------------

def load_ground_truth_tsv(path: str) -> Dict[str, List[str]]:
    """Load train_ground_truth.tsv → dict[source1_entity_id, matched_ids].

    Parameters
    ----------
    path: str
        Path to a ground-truth TSV with columns
        ``source1_entity_id`` and ``matched_entity_ids``.

    Returns
    -------
    dict
        Empty list for singletons (no match in the truth file).
    """
    import csv

    gt: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw = row.get("matched_entity_ids", "").strip()
            matched = [m.strip() for m in raw.split(",") if m.strip()] if raw else []
            gt[s1_id] = matched
    return gt


def load_predictions_tsv(path: str) -> Dict[str, List[str]]:
    """Load a matching_results.tsv → dict[source1_entity_id, matched_ids].

    Same column schema as the ground truth file.
    """
    return load_ground_truth_tsv(path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse, json

    parser = argparse.ArgumentParser(description="Evaluate F_0.5 score locally.")
    parser.add_argument("--pred",   required=True, help="Path to matching_results.tsv (predictions)")
    parser.add_argument("--truth",  required=True, help="Path to ground-truth TSV")
    parser.add_argument("--json",   action="store_true", help="Print full JSON result")
    args = parser.parse_args()

    gt   = load_ground_truth_tsv(args.truth)
    pred = load_predictions_tsv(args.pred)

    result = compute_f05(pred, gt)

    if args.json:
        out = {k: v for k, v in result.items() if k != "per_entity"}
        print(json.dumps(out, indent=2))
    else:
        print(f"  Macro F_0.5  : {result['macro_f05']:.6f}")
        print(f"  Macro Prec   : {result['macro_precision']:.6f}")
        print(f"  Macro Recall : {result['macro_recall']:.6f}")
        print(f"  Entities     : {result['n_entities']}")
        print(f"  GT Singletons: {result['n_singletons']} "
              f"({result['n_correct_singletons']} correctly predicted empty)")


if __name__ == "__main__":
    _cli()
