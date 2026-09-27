# PROJECT_CONTEXT.md

**Read this file before starting any significant work.**

## Project
Amazon ML Challenge 2026 — Business Entity Resolution.
Time budget: ~6 hours total, single sprint. No time for a second architecture.

## Goal
For every Source 1 entity in the test set, predict all matching Source 2 / Source 3
entities. Zero, one, or many matches per S1 entity. Every S1 test entity must appear
in the output, even with an empty match list (singleton).

## Evaluation
- Macro-averaged F0.5 per Source 1 entity, then averaged across all entities.
- F0.5 = (1.25 × P × R) / (0.25 × P + R) — **precision weighted 2x over recall.**
- A correctly-predicted singleton (empty list, no true matches) scores 1.0.
- A false merge on a true singleton scores 0.0. Do not force matches when unsure.

## Data
- `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` (no ground truth)
- All files tab-separated (`sep="\t"`). Addresses and ID lists contain commas — do not
  split on comma for the file itself, only for `matched_entity_ids` / `candidate_entity_ids`.
- Columns: `entity_id` (prefix S1-/S2-/S3- indicates source), `business_name`,
  `business_address`, `country`.
- Train countries: US, India only. **Test adds France, unseen in training.**
  `country` must be treated as an open string set — no hardcoded {US, India} logic,
  no country-specific branching that would silently break/skip France.

## Constraints
- No external data, APIs, geocoding, or entity-resolution services. Challenge data only.
- Final model must be MIT/Apache 2.0 licensed and ≤8B parameters.
- Output is TSV, exact column names, no quoting on comma-joined ID lists.
- No duplicate entity IDs within a list; no duplicate `source1_entity_id` rows.
- `matched_entity_ids` / `candidate_entity_ids` must only reference S2-/S3- IDs that
  exist in the test set. Never reference S1 IDs (no self-matches).
- Every match in `matching_results.tsv` must also appear in `candidate_pairs.tsv`
  for the same `source1_entity_id` (candidates are a superset of final matches).

## Current pipeline (fixed for this sprint — see ARCHITECTURE.md)
1. Data loading
2. Minimal normalization (name + address)
3. Blocking (TF-IDF char n-gram + cosine top-k, single method)
4. Candidate pair generation → `candidate_pairs.tsv`
5. Pairwise similarity features
6. Binary classifier (LightGBM or LogisticRegression)
7. Threshold tuning on validation split (optimize F0.5, not accuracy/F1)
8. Aggregation → `matching_results.tsv`
9. `utils/validate_submission.py`

## Current status
_(agent updates this section after each task — one line, e.g. "Stage 3 complete, blocking recall 0.91 on val")_

## Non-goals for this sprint
No embeddings, no FAISS, no ensembles, no reranker, no multi-stage blocking. See
DECISIONS.md and AGENTS.md — do not reopen these without explicit instruction.
