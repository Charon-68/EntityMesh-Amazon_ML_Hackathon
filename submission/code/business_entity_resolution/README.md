# Business Entity Resolution — Reproduction Guide

## Overview

This package contains the full pipeline for the **Amazon ML Challenge 2026 —
Business Entity Resolution** task. Given business records from 3 independent
sources with noisy names and addresses, the pipeline determines which records
refer to the same real-world business entity.

**Evaluation metric:** Macro-averaged F₀.₅ (precision-weighted)  
**Model:** Country-partitioned TF-IDF blocking + LightGBM classifier

---

## Environment Setup

All commands are run from the **project root** (`student_resource/`).

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 2. Install all dependencies (exact pinned versions)
pip install -r submission/code/business_entity_resolution/requirements.txt
```

---

## End-to-End Reproduction (Single Command)

Run the complete pipeline — split → block → train → predict → validate:

```bash
python submission/code/business_entity_resolution/src/main.py
```

This executes all 5 stages in order and exits with code 0 on success.

---

## Step-by-Step Reproduction

Run individual stages when you need to iterate on a specific component:

### Stage 1 — Train/Val Split & Data Profiling

```bash
python submission/code/business_entity_resolution/src/split_data.py
```

**Output:**
- `submission/code/business_entity_resolution/data/train_ids.txt` — 1,765,457 S1 IDs
- `submission/code/business_entity_resolution/data/val_ids.txt` — 441,364 S1 IDs
- `submission/code/business_entity_resolution/data/train_ground_truth.tsv`
- `submission/code/business_entity_resolution/data/val_ground_truth.tsv`
- Dataset profile: row counts, missing values, country distributions, GT statistics

### Stage 2 — Blocking / Candidate Generation

```bash
# Full run: recall gate on train val split, then test candidates (default)
python submission/code/business_entity_resolution/src/blocking.py --mode both

# Only run the recall gate (check blocking quality on val split)
python submission/code/business_entity_resolution/src/blocking.py --mode train

# Only generate test candidates (skip recall gate)
python submission/code/business_entity_resolution/src/blocking.py --mode test --no-recall-gate

# Tune top-k (default 20, max 40):
python submission/code/business_entity_resolution/src/blocking.py --mode both --k 25
```

**Output:**
- `submission/output/candidate_pairs_train.tsv` — train/val candidates with recall gate
- `submission/output/candidate_pairs.tsv` — test set candidates

**Recall gate:** Automatically bumps k (by 5, up to 40) until entity recall ≥ 0.85.

### Stage 3 — Feature Extraction & LightGBM Training

```bash
python submission/code/business_entity_resolution/src/train_model.py
```

**Requires:** `submission/output/candidate_pairs_train.tsv` (from Stage 2)

**Output:**
- `submission/code/business_entity_resolution/models/matcher.joblib` — model + best threshold
- `submission/code/business_entity_resolution/models/threshold_sweep.tsv` — F₀.₅ by threshold
- `submission/code/business_entity_resolution/models/training_report.txt` — feature importances

### Stage 4 — Test Inference

```bash
# Use threshold from trained model automatically
python submission/code/business_entity_resolution/src/predict.py

# Override threshold (e.g., from sweep table analysis)
python submission/code/business_entity_resolution/src/predict.py --threshold 0.42

# Skip re-running blocking (reuse existing candidate_pairs.tsv)
python submission/code/business_entity_resolution/src/predict.py --skip-blocking
```

**Requires:** `models/matcher.joblib` (Stage 3) + `candidate_pairs.tsv` (Stage 2)

**Output:**
- `submission/output/matching_results.tsv` — final entity matches (**leaderboard file**)
- `submission/output/candidate_pairs.tsv` — blocking candidates (overwritten if blocking runs)

### Stage 5 — Validate Submission

```bash
python utils/validate_submission.py \
    --matching   submission/output/matching_results.tsv \
    --candidate  submission/output/candidate_pairs.tsv \
    --test-dir   dataset/test
```

Exit code **0 = PASS**, exit code **1 = FAIL** (issues listed on stdout).

---

## Local F₀.₅ Evaluation on Validation Split

```bash
python submission/code/business_entity_resolution/src/metrics.py \
    --pred  submission/output/matching_results_val.tsv \
    --truth submission/code/business_entity_resolution/data/val_ground_truth.tsv
```

---

## Source Layout

```
submission/
├── Documentation_template.md          Methodology write-up
├── output/
│   ├── matching_results.tsv           Final matches (leaderboard upload)
│   └── candidate_pairs.tsv            Blocking candidates
└── code/business_entity_resolution/
    ├── README.md                      This file
    ├── requirements.txt               Pinned dependencies
    ├── models/
    │   ├── matcher.joblib             Trained model + threshold
    │   ├── threshold_sweep.tsv        F₀.₅ at each threshold
    │   └── training_report.txt        Training summary
    ├── data/
    │   ├── train_ids.txt / val_ids.txt
    │   └── train/val ground truth TSVs
    └── src/
        ├── main.py          CLI orchestrator (entry point)
        ├── split_data.py    Train/val splitter + data profiler
        ├── normalize.py     Unicode-safe text normaliser
        ├── blocking.py      Country-partitioned TF-IDF blocking
        ├── features.py      18-feature pairwise extraction
        ├── metrics.py       Macro-F₀.₅ with singleton handling
        ├── train_model.py   LightGBM training + threshold sweep
        └── predict.py       Streaming test inference → TSVs
```

---

## System Requirements

- Python 3.9+
- RAM: ≥ 16 GB recommended (loading ~1.3 GB TSVs + sparse TF-IDF matrices)
- Disk: ≥ 5 GB free (dataset + model artefacts)
- Runtime: ~45–90 min for full pipeline (blocking dominates)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: candidate_pairs_train.tsv` | Run Stage 2 (`blocking.py --mode train`) first |
| `FileNotFoundError: matcher.joblib` | Run Stage 3 (`train_model.py`) first |
| `MemoryError` during blocking | Reduce `BATCH_SIZE_S1` in `blocking.py` (default: 2000) |
| Validator: `FAIL — duplicate source1_entity_id` | Re-run `predict.py`; do not concatenate output files manually |
| Validator: `FAIL — S1-xxx not in test set` | Ensure you're pointing `--test-dir` at `dataset/test/` |
