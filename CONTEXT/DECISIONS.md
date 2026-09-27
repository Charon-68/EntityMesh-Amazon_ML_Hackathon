# DECISIONS.md

Log every non-trivial choice here, so the agent (and teammates) don't reopen a
settled question mid-sprint. Newest entries at the bottom. Keep each entry short.

---

## Decision 000
**Date:** 2026-09-27
**Decision:** Fixed pipeline per ARCHITECTURE.md — TF-IDF char n-gram blocking,
string-similarity features only, LightGBM/LogReg classifier, no embeddings, no
ANN libraries beyond sklearn, no ensembles.
**Reason:** 6-hour time budget. First working version must be simple enough to
finish and submit. A partially-built sophisticated pipeline scores zero.
**Alternatives considered:** Sentence-transformer embedding retrieval, FAISS ANN,
cross-encoder reranker.
**Why rejected (for now):** Too much setup/tuning time relative to payoff at this
time budget. Revisit only if T10.5 (first submission) is done with time to spare —
log a new decision before adding.

---

## Decision 001
**Date:** _(fill in)_
**Decision:**
**Reason:**
**Alternatives considered:**
**Why rejected:**

---

<!--
Template for new entries — copy this block:

## Decision 00X
**Date:**
**Decision:**
**Reason:**
**Alternatives considered:**
**Why rejected:**
-->
