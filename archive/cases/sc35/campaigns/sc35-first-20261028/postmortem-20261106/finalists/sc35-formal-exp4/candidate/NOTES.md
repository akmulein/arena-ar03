# NOTES — formal-exp4 (SC-35 deterministic OOXML/text checks expansion, round 4)

Branch of immutable candidate `sc35-formal-exp3` (snapshot
`/workspace/work/formal-exp3-base`, itself a branch of `sc35-formal-exp2`).
This candidate is deterministic-only: `solution.json =
{"harness": "sc35-content", "parameters": {}}`; `solve.py` builds
`extra_checks.build_review(...)` and passes it as `extra_review` to
`review_document(...)` with `client=None`. No model is ever called.

Changed vs base (`diff -r -q -x '*.pyc' -x '__pycache__'` — verified):
- `runtime/sc35/extra/extra_checks.py` (all code changes)
- `NOTES.md` (this file)

Everything else is byte-identical to `formal-exp3-base`.

## What was added (each a general rule, never a per-case lookup)

The last two deterministic-reachable gold FN events (both in the
«Дополнительный материал испытания…» zone, category formulas, ГОСТ 19.106-78
п. 2.4.1) are now exact TPs with zero FP. Each uses the frozen
`rule_categories` for its category, `basis_kind=gost_requirement`, and valid
Unicode-codepoint spans taken verbatim from the canonical projection records
(`_text_span` of the target paragraph: the c0434 anchor kind `object` and the
c0648 anchor kind `text_object` match the gold).

### 1. `_formula_number_position` → `ESPD-19.106-78-2.4.1-formula_number_position`

Per п. 2.4.1 the formula number stands in round brackets at the right side of
the line, at formula level. **Condition:** among the document's display
formulas (an `m:oMath` paragraph whose canonical text contains `=`/`≈` and a
trailing arabic number `(N)` or `(N.M)`), those with an explicit `w:jc` of
`left`/`right`/`end` are collected. **Gate:** the check fires only on a
`jc=left` numbered display formula *when at least one other* numbered display
formula is `jc=right`/`end`. This is deliberately a *relative* inconsistency
detector: a document whose numbered formulas are all aligned one way (e.g.
every numbered formula left-aligned as a style) cannot be proven defective
without rendering, so it never fires; only the deviation is flagged.
**Anchor:** the full span of the `jc=left` formula paragraph.

- rule_id `ESPD-19.106-78-2.4.1-formula_number_position`, category `formulas`.
- TP on train: **+1** — c0434, `x = 2 (2)`, span (35406,35415) == gold exactly
  (its counterpart `x = 1 (1)` at (35396,35405) is `jc=right`).
- FP on train: **0** — across all 113 only c0434 has a left-aligned numbered
  display formula; c0424/c0428/c0648 have right-aligned zone formulas only and
  never fire (verified by a full-corpus scan of every aligned numbered display
  formula).
- Remaining public-split FP risk: **medium**. It would fire exactly when a real
  document (a) right-aligns at least one numbered display formula AND (b)
  contains a second numbered display formula explicitly left-aligned for layout
  reasons (e.g. a deliberately inset equation carrying a manual number). GOST
  still calls that number position wrong, but an author could legitimately mix
  alignment, so the contrast gate is the maximum provable without rendering.
  Documents aligning all numbered formulas uniformly never fire.

### 2. `_formula_part_prefix` → `ESPD-19.106-78-2.4.1-formula_part_prefix`

Per п. 2.4.1, in a part-numbered convention the part number precedes the
formula number with a dot. **Condition — all three signals must coexist:**
(1) an explicit declarative sentence assigns the document/fragment to a part N
(whitelisted patterns: `относится/относятся к (части) N`, `принадлежит (к)
части N`, `входит в (состав) части N`, `является частью N`, `части N
документа`); (2) at least one aligned display formula already carries a
part-prefixed number `(N.M)` whose prefix equals N from (1); (3) another aligned
display formula carries a bare `(K)` number. Every such bare `(K)` number
should carry the prefix; the defect is anchored at the un-prefixed formula.
"Aligned display formula" = an `m:oMath` paragraph with explicit `w:jc` in
`right`/`end`/`center` whose canonical text contains `=`/`≈` and a trailing
arabic number — this excludes the tab-composed main example equation
`Pусп.=…,(1)` (`jc=None`), which is not a display-formula number and is never
flagged. A document with only bare or only prefixed numbers never fires.

- rule_id `ESPD-19.106-78-2.4.1-formula_part_prefix`, category `formulas`.
- TP on train: **+1** — c0648, `x = 5 (1)`, span (35557,35566) == gold exactly
  (part 6 declared in «…относится к части 6 документа…»; `x = 6 (6.2)` at
  (35567,35578) carries prefix 6; the bare `x = 5 (1)` is the defect). The main
  body `Pусп.=…,(1)` is `jc=None` and excluded, so it produces no second
  finding.
- FP on train: **0** — of all 113 docs only c0648 has an explicit part-context
  narrative AND a part-prefixed aligned formula AND a bare aligned formula;
  c0424/c0428/c0434/c0648 are the only docs with aligned zone formulas, and the
  first three have no prefixed `(N.M)` and/or no part narrative (verified by a
  full-corpus scan).
- Remaining public-split FP risk: **low–medium**. It fires only when all three
  signals coexist in one document: a declared part N + a formula already
  numbered `N.M` + an additional bare-numbered aligned formula. It would fire
  wrongly on a real multi-part document whose per-part section uses `(N.M)`
  numbering *while an un-partitioned leading section deliberately keeps bare
  `(K)` formula numbers* and the text also contains an explicit «…относится к
  части N…»/«части N документа» sentence — i.e. a genuinely mixed numbering
  scheme plus an explicit part declaration (arguably itself inconsistent). The
  narrative-phrase whitelist exists to avoid the common innocent «в части 2 …»
  clause-reference usage.

## Before / after (frozen metric `sc35-span-overlap-v1`, full 113 train)

Command: `uv run --with lxml==6.1.3 --with jsonschema==4.26.0 \
analysis/exp_eval.py <cand>` — builds `build_review` + `review_document
(client=None)`, validates every review against the frozen schema (jsonschema
Draft202012Validator, raises on the first invalid one — ran clean), scores with
the frozen metric runtime.

| metric | formal-exp3-base | formal-exp4 | Δ |
|---|---|---|---|
| score_percent (macro F1) | 90.141608 | **90.279935** | +0.1383 |
| micro precision | 1.0 | 1.0 | — |
| micro recall % | 86.343154 | 86.967071 | +0.6239 |
| intersection / predicted chars | 2491 / 2491 | 2509 / 2509 | +18 / +18 |
| gold chars | 2885 | 2885 | — |

Per-category F1 (before → after), `predicted_chars == intersection_chars`
(P=100.0) in every category:

| category | base F1 | exp4 F1 | events tp/fp/fn |
|---|---|---|---|
| annotation_contents | 1.0000 | 1.0000 | 7/0/0 |
| appendices | 1.0000 | 1.0000 | 6/0/0 |
| figures | 0.6243 | 0.6243 | 5/0/3 |
| formulas | 0.9441 | **0.9607** | 10/0/3 (was 8/0/5) |
| headings | 1.0000 | 1.0000 | 17/0/0 |
| metadata | 1.0000 | 1.0000 | 16/0/0 |
| notes_footnotes | 1.0000 | 1.0000 | 5/0/0 |
| page_layout | 1.0000 | 1.0000 | 5/0/0 |
| paragraphs_lists | 1.0000 | 1.0000 | 7/0/0 |
| references_terms | 0.2486 | 0.2486 | 1/0/5 |
| sections_content | 1.0000 | 1.0000 | 21/0/0 |
| tables | 1.0000 | 1.0000 | 8/0/0 |

Added findings across all 113 (full review pipeline): exactly **2**, both exact
rule_id AND span matches of gold, zero removed, zero false positives
(`predicted == intersection` in every category; formulas micro intersection
grew by exactly 9+9=18 chars):
- c0434 `…-2.4.1-formula_number_position` span (35406,35415)
- c0648 `…-2.4.1-formula_part_prefix` span (35557,35566)

All 113 reviews schema-valid (validator raised nowhere).

## What remains unattacked (unchanged from exp3)

- formulas: c0451 `symbol_missing`, c0464 `symbol_newline` (anchors collapse
  to single `\n` characters), c0476 `parameter_dimension` (unit semantics) —
  semantic/render, not deterministic. **No remaining deterministic-reachable
  formula gold event.**
- figures c0511/c0520/c0526 — element-order heuristics (high FP, no
  `w:drawing` in the corpus).
- references_terms c0286/c0289/c0299/c0306/c0313 — semantic defects anchored
  at the synthetic «Контрольный фрагмент N» test label (test-fit; not
  attacked).

## Files

- Changed: `runtime/sc35/extra/extra_checks.py` (two new check functions plus
  `_PART_CONTEXT_PATTERNS` / `_PART_DISPLAY_ALIGN` constants, both checks
  registered in `build_review`), `NOTES.md`.
- Everything else byte-identical to `formal-exp3-base` (verified by recursive
  diff — single differing code file).
