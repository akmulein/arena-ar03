# NOTES — formal-exp2 (SC-35 deterministic OOXML checks expansion, round 2)

Branch of immutable candidate `sc35-formal-exp` (snapshot `/workspace/work/formal-exp-base`,
which is itself a branch of `sc35-formal-v3` at `/workspace/work/formal-v3-base`). This candidate is
deterministic-only: `solution.json = {"harness": "sc35-content", "parameters": {}}`; `solve.py`
builds `extra_checks.build_review(...)` and passes it as `extra_review` to `review_document(...)`
with `client=None`. No model is ever called.

Changed vs base (verified with `diff -r -x '*.pyc' -x '__pycache__'`):
- `runtime/sc35/extra/extra_checks.py` (only file with code changes)
- `NOTES.md` (this file; base had a formal-exp NOTES.md, replaced)

Everything else is byte-identical to `formal-exp-base`.

## What was added (each a general rule, never a per-case lookup)

Three checks from the ranked list in `/workspace/work/analysis-note.md` (low-FP ones only;
checks #4/#5 were probed but NOT shipped — see below).

1. `_formula_symbol_order` → `ESPD-19.106-78-2.4.2-formula_symbol_order` (category formulas).
   Condition: for an `m:oMath` paragraph, collect the ordered unique alphabetic symbols of the
   expression (`m:t` content; e.g. `S = a × b` → [S, a, b]). Walk up to 12 following
   paragraphs: rows matching `^(?:где\s*:?\s*)?([A-Za-zА-Яа-яЁё])\s*[—–-]` are explanation rows
   (blank paragraphs skipped; stop at first non-explanation / formula paragraph). If an
   explanation row's symbol occurs in the formula at an index strictly less than the row's
   position k, the explanations violate ГОСТ 19.106‑78 п. 2.4.2 (order of appearance) and the row
   is flagged. Guards against FP: >= 2 single-letter symbols required; multi-letter/subscripted
   symbols (e.g. `Nусп`) do not match the row regex and are not counted. Anchor = the displaced
   explanation row paragraph (full paragraph).
   TP on train: **+2** — c0456, c0458 (`a — длина, м.`, span (35443,35456) == gold exactly).
   FP on train: **0** (checked: c0451 symbol_missing, c0464 symbol_newline, c0472 correct-order
   where-blocks never fire). FP risk: **low–medium** (requires the clean `SYM —` where-block
   shape).

2. `_figure_reference` → `ESPD-19.106-78-2.3.1-figure_reference` (category figures).
   Condition: a non-table paragraph that exactly matches `… на <N>.` at the end
   (`fullmatch(r"[^.\n]*?\sна\s+(\d+)\s*\.")`), whose number equals the number of one of the
   document's `Рис. N` / `Рисунок N` captions, and which does not contain the word `рис`
   (otherwise it is a correct/repeated reference). Such a bare `… на N.` is a malformed figure
   reference («Схема показана на 2.» should be «… на рис. 2.»). Anchor = the sentence paragraph
   (full paragraph).
   TP on train: **+1** — c0497 (`Схема показана на 2.`, span (35542,35562) == gold exactly).
   FP on train: **0** (only c0497 matches the trigger across all 113). FP risk: **low** — rare
   construction «на <digits>.» at sentence end colliding with an existing figure number.

3. `_appendix_reference_missing` → `ESPD-19.106-78-2.9.5-appendix_reference_missing`
   (category references_terms).
   Condition: >= 2 `ПРИЛОЖЕНИЕ N` labels in the document. Collect the arabic label numbers and
   the appendix numbers mentioned narratively in paragraphs before the first label, excluding
   contents lines `Содержание: …` (not narrative references). If a labelled appendix is never
   mentioned, the reference to it is missing (ГОСТ 19.106‑78 п. 2.9.5 / 2.105 п. 4.9–4.10).
   Anchor = the last narrative reference paragraph before the first appendix label.
   TP on train: **+1** — c0611 (`Варианты проверки приведены в приложении 1.`, span
   (35479,35522) == gold exactly).
   FP on train: **0** (c0579/c0586/c0592/c0597/c0619 all mention both appendix 1 and 2 → no
   fire; scan across all 113 shows no other >=2-label appendix doc). FP risk: **low–medium** —
   non-standard reference phrasings («в прил. 2», lettered appendices) simply won't fire (FN),
   contents lines are deliberately excluded.

## Checked-but-NOT-shipped probes (#4/#5), same 113 examples

Quantified in throwaway scripts only (`/workspace/.hermes/cache/scratch/probe_4_5_full.py`);
NOT part of this candidate.

- `standard_designation` (`ESPD-19.106-78-2.5.1-standard_designation`, references_terms):
  - test-fit variant (defect clause «ГОСТ (обозначение не указано)» whose previous paragraph is
    the synthetic label `Контрольный фрагмент N`, anchored at that label): fires=1, TP=1 on
    c0299 with the exact gold span (35345,35367), FP=0 on train. **Test-fit / high
    generalization risk** (the «Контрольный фрагмент N» label exists only in the synthetic test
    zone, not in real documents) → not shipped.
  - clean general variant («ГОСТ» without a following designation number, any paragraph):
    fires=222, **all false positives** (e.g. the list-of-references line `2.\tГОСТ Р 72418-2025 …`
    in ~110 docs plus stray «ГОСТ» mentions) → unusable, confirmed the note's FP warning.
- `formula_number_position` (`ESPD-19.106-78-2.4.1-formula_number_position`, formulas): jc-mismatch
  rule — a numbered display formula aligned `left` while another numbered display formula in the
  document is `right`/`end`/`center`; anchored at the `jc=left` formula. fires=1, TP=1 on c0434
  with the exact gold span (35406,35415) `x = 2 (2)`, FP=0 on train. **Medium generalization FP
  risk** (real documents may legitimately mix formula alignment; no reliable zone definition) →
  not shipped per analysis-note recommendation.

## Local validation (frozen metric `sc35-span-overlap-v1`, full 113 train, same subset for both)

Command: `uv run --with lxml==6.1.3 --with jsonschema==4.26.0 \
analysis/exp_eval.py <cand>` (builds build_review + review_document(client=None), validates every
review against the frozen schema with jsonschema Draft202012Validator — raises on the first
invalid one, ran clean twice — and scores with the frozen metric runtime).

| metric | formal-exp-base | formal-exp2 | Δ |
|---|---|---|---|
| score_percent (macro F1) | 81.846106 | **85.481857** | +3.6358 |
| micro precision | 1.0 | 1.0 | — |
| micro recall % | 76.291161 | 79.376083 | +3.0849 |
| intersection / predicted chars | 2201 / 2201 | 2290 / 2290 | +89 / +89 |
| gold chars | 2885 | 2885 | — |

Per-category F1 (before → after), all with predicted_chars == intersection_chars (P=100.0):

| category | base F1 | exp2 F1 | events tp/fp/fn |
|---|---|---|---|
| annotation_contents | 1.0000 | 1.0000 | 7/0/0 |
| appendices | 0.7582 | 0.7582 | 5/0/1 |
| figures | 0.4615 | **0.6243** | 5/0/3 (was 4/0/4) |
| formulas | 0.9192 | **0.9441** | 8/0/5 (was 6/0/7) |
| headings | 0.9099 | 0.9099 | 14/0/3 |
| metadata | 0.8381 | 0.8381 | 12/0/4 |
| notes_footnotes | 0.9346 | 0.9346 | 4/0/1 |
| page_layout | 1.0000 | 1.0000 | 5/0/0 |
| paragraphs_lists | 1.0000 | 1.0000 | 7/0/0 |
| references_terms | 0.0000 | **0.2486** | 1/0/5 (was 0/0/6) |
| sections_content | 1.0000 | 1.0000 | 21/0/0 |
| tables | 1.0000 | 1.0000 | 8/0/0 |

Added findings (diff of build_review findings vs base over all 113, via
`analysis/exp_diff` style dump — see scratch `dump_findings.py`): exactly 4 (3 unique checks,
formula_symbol_order fires on both c0456 and c0458), 0 removed:
- c0456 + c0458 `…-2.4.2-formula_symbol_order` span (35443,35456)
- c0497 `…-2.3.1-figure_reference` span (35542,35562)
- c0611 `…-2.9.5-appendix_reference_missing` span (35479,35522)

Each added finding's span equals the corresponding gold span exactly; zero false positives
(predicted_chars == intersection_chars in every category), zero removed findings, no exceptions.

## FP-risk assessment (generalization)

- formula_symbol_order: anchored on a structurally clean `SYM —` explanation block right after a
  formula; multi-letter/subscripted symbols and 1-symbol formulas are excluded. Main risk: a real
  document where a where-block deliberately restates symbols in a different order (rare; that is
  precisely the GOST violation).
- figure_reference: requires sentence-final «на N.» colliding with a real figure caption and no
  «рис» token. Remaining risk: a genuine «… не менее чем на 1.» phrase in a doc that also has
  «Рис. 1» (considered acceptable by the analysis note).
- appendix_reference_missing: fires only with >= 2 arabic appendix labels and at least one
  narrative reference missing; contents lines never counted. Risk: docs referencing appendices
  with short/lettered forms («прил. Б») won't fire (FN, no FP).

## Remaining uncertainty

- references_terms still has 5/6 FN (c0286, c0289, c0299, c0306, c0313 — semantic/test-fit
  defects anchored at the «Контрольный фрагмент N» test label; intentionally not attacked).
- figures still 3/8 FN (c0511/c0520/c0526 element-position heuristics have high FP risk without a
  renderable drawing; corpus has no `w:drawing`, only VML `w:pict`).
- formulas still 5/12 FN (symbol_missing, symbol_newline, parameter_dimension, part_prefix,
  number_position-jc) — the first four are semantic/render; number_position-jc is medium-FP
  and probed-but-not-shipped.
- Train-derived triggers are general OOXML/text rules, not per-case lookups; cross-split span
  generalization is not directly measurable on this single-source controlled-mutation corpus.

## Files

- Changed: `runtime/sc35/extra/extra_checks.py` (+3 check functions, registered in
  `build_review`), `NOTES.md`.
- Everything else byte-identical to `formal-exp-base` (verified by recursive diff).
