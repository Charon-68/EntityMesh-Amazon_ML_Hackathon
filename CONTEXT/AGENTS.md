# AGENTS.md

Rules for how the coding agent works in this repository. This is the constitution —
follow it over any instinct toward "improving" things mid-task.

## General
- Read `PROJECT_CONTEXT.md` before modifying code. Read `ARCHITECTURE.md` before
  changing pipeline structure — and don't change pipeline structure this sprint
  without being explicitly asked.
- Do not introduce components outside the current task, however easy they'd be to add.
- Prefer the simplest implementation that works. This is a 6-hour sprint, not a
  research project — boring and working beats clever and half-finished.
- Do not add dependencies without stating why in your task summary.
- Do not change existing behavior without being asked, and without a quick sanity
  check (rerun the smallest relevant test/script) before reporting done.

## ML-specific
- Never use test labels during development — there are none; if you find yourself
  needing ground truth for the test set to validate something, stop and flag it,
  don't fabricate or infer it.
- Keep train/validation separation explicit and visible in code (named variables,
  not implicit slicing). Never let validation rows leak into training features
  (e.g. no fitting TF-IDF/vectorizers on validation+train combined in a way that
  uses validation label info).
- Optimize macro F0.5, not accuracy, not plain F1. Always report precision, recall,
  and F0.5 together — never F0.5 alone.
- Treat `country` as an open string field. Never write code that special-cases or
  filters on `{"US", "India"}` — test includes France. If you're about to hardcode
  a country list, stop and use a data-driven approach instead.
- Blocking recall (fraction of true matches surviving into the candidate set) must
  be measured and reported before moving to feature/model work. Do not proceed to
  the classifier stage without this number.

## Pipeline
- Each stage (normalize, block, featurize, classify, threshold, aggregate) should
  have a clear input/output contract — a function or script with defined inputs
  and outputs, not a notebook-style script that mutates global state.
- Avoid hidden global state between stages. Intermediate outputs (normalized data,
  candidate pairs, feature tables) should be inspectable — write them to disk or
  return them explicitly, don't just chain in-memory transformations invisibly.
- `candidate_pairs.tsv` must reflect exactly what was fed to the classifier at
  inference time — not an earlier, looser blocking pass.

## Code
- Add a minimal test or sanity check for new non-trivial logic (e.g. the F0.5
  scorer, the TSV writer, the normalizer). Given the time budget, "test" can mean
  a quick assertion script, not a full test suite.
- Run relevant checks after modifications before reporting a task done.
- Always run `utils/validate_submission.py` before saying a submission is ready.

## Reporting (end of every task)
1. Summarize what changed, in plain language.
2. List files changed/created.
3. Report any checks/tests run and their results (numbers, not "looks good").
4. Report assumptions made.
5. Report unresolved issues or risks, especially anything affecting blocking
   recall or precision on singletons.

## Hard stop
If a task would take the pipeline outside what's listed in `ARCHITECTURE.md`
("Non-goals" section), stop and ask instead of proceeding. Given the time budget,
an unapproved detour is expensive — flag it, don't just build it.
