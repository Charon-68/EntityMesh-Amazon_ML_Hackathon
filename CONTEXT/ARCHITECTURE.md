# ARCHITECTURE.md

The pipeline for this sprint. Deliberately boring — this is the first and only
version given the time budget. Do not deviate without updating this file and
logging a decision in `DECISIONS.md`.

## Pipeline stages

```
raw TSVs
  → [1] load
  → [2] normalize (name, address)
  → [3] block (TF-IDF char n-gram + cosine top-k, single method + optional key block)
  → [4] candidate_pairs.tsv
  → [5] pairwise features (string similarity only)
  → [6] classifier (LightGBM or LogisticRegression)
  → [7] threshold tuning (maximize F0.5 on validation split)
  → [8] aggregate per S1, singleton handling
  → [9] matching_results.tsv
  → [10] validate_submission.py
```

## Stage details

**1. Load**
`pandas.read_csv(..., sep="\t")` for all 7 files. No cleaning here — just load and
report shape/dtypes/missing counts.

**2. Normalize**
- Lowercase, strip punctuation, collapse whitespace.
- Legal suffix mapping built from observed data (`ltd`↔`limited`, `pvt`↔`private`,
  `corp`↔`corporation`, `&`↔`and`, `inc`, `co`).
- Address abbreviation mapping (`rd`↔`road`, `st`↔`street`, etc.), same rule:
  derived from what's actually in the data, not an external source.
- Applied identically regardless of `country` value — no branching on country.

**3. Blocking — single method**
- Fit `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))` on normalized
  `business_name` across S1+S2+S3 combined.
- For each S1 record, retrieve top-k (start k=15) nearest S2 candidates and top-k
  nearest S3 candidates by cosine similarity (`sklearn.neighbors.NearestNeighbors`).
- Optional cheap safety net: union in same-country + same-first-3-normalized-chars
  matches. Only add this if stage 3 recall is measured and found lacking — don't
  add speculatively.
- **Must report blocking recall on the validation split before proceeding to stage 5.**

**4. candidate_pairs.tsv**
Direct output of stage 3, in final submission format. This is what stage 6 actually
scores at inference — if stage 3's output set changes later, this file and stage 6's
input must change together.

**5. Pairwise features**
Minimal set:
- `rapidfuzz.fuzz.token_sort_ratio` on normalized name
- Jaro-Winkler on normalized name (`rapidfuzz.distance.JaroWinkler`)
- Same two on normalized address
- TF-IDF cosine similarity (reuse vectors from stage 3)
- `country` exact-match flag (binary, still just a flag — not a branch)

Explicitly out of scope this sprint: sentence embeddings, semantic similarity models,
learned representations of any kind.

**6. Classifier**
LightGBM (preferred) or `LogisticRegression` as fallback if LightGBM setup is
friction. Binary target: 1 if pair is in ground truth, 0 otherwise. Negatives
sampled from the blocking output itself (non-matching candidates), not randomly
from the full cross product.

**7. Threshold tuning**
Sweep on validation predictions, pick threshold maximizing macro F0.5. Do not use
0.5 by default — report the chosen value and the F0.5/P/R at that value.

**8. Aggregate**
Per S1 entity: all candidates with predicted probability ≥ threshold, deduped,
S2/S3 IDs only. Empty list allowed and expected for singletons.

**9/10. Output + validate**
Write both TSVs, run `utils/validate_submission.py` before calling anything done.

## Non-goals for this sprint (explicitly rejected — see DECISIONS.md)
- Sentence-transformer / embedding-based retrieval or similarity
- FAISS or any ANN library beyond sklearn's NearestNeighbors
- Multi-stage or learned blocking
- Cross-encoder reranking
- Self-training / pseudo-labeling
- Model ensembles
- LLM-based matching or feature extraction

If there's time left after a working submission exists, propose additions as a new
entry in `DECISIONS.md` first — don't silently add them.
