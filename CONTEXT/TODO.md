# TODO.md

One task at a time. Pattern per task: **run → inspect → understand → commit**, then
update the checkbox and the "Current status" line in `PROJECT_CONTEXT.md`.

Budget is ~6 hours total — rough time-box per task noted. If a task overruns its
box by more than ~50%, stop, report status, and ask whether to simplify or move on.

- [ ] **T1 — Inspect data** (10 min)
  Load all 7 TSVs. Report shape, dtypes, missing value counts, sample rows,
  match-count distribution in `train_ground_truth.tsv` (how many singletons? how
  many 1:many?). No modeling yet. Do not write pipeline code in this task.

- [ ] **T2 — F0.5 scorer** (15 min)
  Implement the macro-averaged F0.5 scorer exactly per the PS formula, as a
  standalone, testable function. Include a couple of hand-checked examples
  (including a singleton case and a partial-match case) as a sanity check.

- [ ] **T3 — Normalization** (20 min)
  Implement per ARCHITECTURE.md stage 2. Apply to all sources. Spot-check ~10
  before/after examples across both US and India records.

- [ ] **T4 — Train/val split** (10 min)
  Split train S1 entities (not raw pairs) into train/val, carry their ground truth
  along. Report split sizes and match-count distribution in each half.

- [ ] **T5 — Baseline blocking** (30 min)
  Implement per ARCHITECTURE.md stage 3. Run on the val split.

- [ ] **T6 — Blocking recall evaluation** (15 min)
  Measure recall: % of true matches (from val ground truth) present in the
  candidate set. **Report this number explicitly before starting T7.** If below
  ~0.85, increase k or add the key-block safety net before moving on.

- [ ] **T7 — Pair features** (30 min)
  Implement per ARCHITECTURE.md stage 5. Build the labeled pair dataset (positives
  from ground truth, negatives from blocking output).

- [ ] **T8 — Basic matcher** (30 min)
  Train the classifier per stage 6. Report train/val loss or a quick metric, not
  full evaluation yet (that's T9).

- [ ] **T9 — Thresholding** (20 min)
  Sweep threshold on val predictions, optimize macro F0.5. Report P/R/F0.5 at the
  chosen threshold and at 2-3 neighboring thresholds for comparison.

- [ ] **T10 — Submission** (30 min)
  Run the full pipeline on the real test set. Generate `matching_results.tsv` and
  `candidate_pairs.tsv`. Run `utils/validate_submission.py`. Fix any validator
  errors before proceeding.

- [ ] **T10.5 — First leaderboard upload** (checkpoint, not a coding task)
  Upload `matching_results.tsv` to the Portal now, before further iteration. This
  is the floor — don't lose it to a later bug.

- [ ] **T11 — One improvement pass** (remaining time, cap ~45 min)
  Pick exactly ONE, based on T6/T9 numbers:
  - Blocking recall was the bottleneck → increase k / add address-based blocking
  - Precision was weak → raise threshold or add a stricter feature
  - Otherwise → re-tune threshold only
  Log the choice and reasoning in `DECISIONS.md` before starting.

- [ ] **T12 — Package submission** (20 min)
  Fill `Documentation_template.md`, assemble the zip per the PS structure
  (`output/`, `code/business_entity_resolution/`, doc), verify `README.md` in the
  package actually reproduces the outputs end-to-end.

- [ ] **T13 — Final leaderboard upload + submit zip**
