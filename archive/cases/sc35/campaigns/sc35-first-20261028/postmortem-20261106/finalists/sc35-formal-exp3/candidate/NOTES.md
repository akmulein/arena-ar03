# NOTES — formal-exp3 (SC-35 deterministic OOXML/text checks expansion, round 3)

Branch of immutable candidate `sc35-formal-exp2` (snapshot
`/workspace/work/formal-exp2-base`, which is itself a branch of
`/workspace/work/formal-exp-base`). This candidate is deterministic-only:
`solution.json = {"harness": "sc35-content", "parameters": {}}`; `solve.py`
builds `extra_checks.build_review(...)` and passes it as `extra_review` to
`review_document(...)` with `client=None`. No model is ever called.

Changed vs base (`diff -r -x '*.pyc' -x '__pycache__'` — verified, only one
code file):
- `runtime/sc35/extra/extra_checks.py` (all changes)
- `NOTES.md` (this file)

Everything else is byte-identical to `formal-exp2-base`.

## What was added (each a general rule, never a per-case lookup)

Nine gold findings were still FN in the four target categories (headings 3,
metadata 4, notes_footnotes 1, appendices 1). All nine are now exact TPs with
zero FP. Each check is text/OOXML-provable, uses the frozen
`rule_categories` for its category, `basis_kind=gost_requirement`, and valid
Unicode-codepoint spans from the canonical projection. Formulas were
re-examined but no new low-risk subrule emerged (see "Not shipped" below).

### 1. headings — `_heading_outline_spacing_checks` (3 new rules, +3 TP)

Anchor = the full paragraph that carries an explicit `w:outlineLvl` in its
`w:pPr` (the structural level-1 heading signal in OOXML; the corpus headings
use `w:outlineLvl w:val="0"`). Only explicitly-spaced paragraphs are judged
(absent spacing properties are inherited from the style and are not provable
here). Condition per rule:

- `ESPD-19.106-78-2.1.8-heading_before_spacing` (c0409): the heading's
  `w:spacing` has `w:before="0"` → violates the >= 4 font-heights interval
  before a heading (ГОСТ 19.106-78 п. 2.1.8). Span (35455,35470) == gold.
- `ESPD-19.106-78-2.1.8-heading_after_spacing` (c0402): `w:after="0"` →
  violates the >= 3 font-heights interval after a heading. Span (35455,35470)
  == gold.
- `ESPD-19.106-78-2.1.8-heading_line_spacing` (c0413): `w:lineRule="exact"`
  with `w:line < 300` (the multiline heading is compressed at 180 twips vs
  the body's 360) → the heading line spacing is not consistent with the body.
  Span (35454,35487) == gold.

FP on train: 0. Across all 113 only c0402/c0409/c0413 have `w:outlineLvl`
with explicit `w:spacing`; the typography (ty-*) headings and the
section/subsection cases (c0352…) carry no explicit spacing and never fire;
ty-* headings use `lineRule="auto"` and are excluded by the exact-only line
test. Generalization FP risk: **low–medium** — requires an explicitly
zero/very-tight spacing value on a paragraph explicitly marked as an outline
heading, which real documents rarely encode by hand (residual risk: a
page-top heading legitimately encoded with `before=0`).

### 2. metadata — designation separator gate fix (+2 TP)

`ESPD-19.103-77-2-designation_country_separator` (c0655) and
`ESPD-19.103-77-2-designation_organization_separator` (c0659).

The gate in `_main_designation` was `^[А-ЯЁ]\.[А-ЯЁ]\.\d`, so a designation
whose *leading* separator was already wrong ("А:В.00005…", "А.В/00002…")
bailed out before `_classify_designation` could report the precise defect.
Loosened the gate to a pure shape check
`^[А-ЯЁ][^\dА-ЯЁ\s][А-ЯЁ][^\dА-ЯЁ\s]\d` (two Cyrillic letters each followed
by one separator character, then a digit) — not a per-case rule; it just lets
already-broken leading separators reach the existing classifier, which then
reports country/organization separators exactly. Still gated to the record
directly after the title-page PMI heading and to the full `DESIGNATION_RE`
shape (5-digit registration + 2-digit edition + kind).

TP on train: **+2**, both with span (192,210) == gold exactly. FP on train: 0
(all other docs have correct leading separators and are unaffected).
Generalization FP risk: **low** — only fires on a designation-shaped string
immediately after the PMI heading; the classifier requires the full broken
designation pattern to emit anything.

### 3. metadata — `_title_program_title_case` (+1 TP)

`ESPD-19.104-78-3.1-title_program_title_case` (c0735). The program title on
the title sheet must be written in capitals. The title is the non-empty
direct-body paragraph immediately above the front-region PMI heading that is
short (<= 60 chars), free of sentence punctuation (`.,:;`), and mixes upper-
and lower-case letters. The long multi-sentence program name ("Программный
комплекс, реализующий…") is ~98 chars with commas/periods and never matches;
correctly-cased short titles ("КОМПОНЕНТ ПРОВЕРКИ ОТВЕТОВ 5") are all-caps and
don't match. Span (131,159) == gold exactly.

TP on train: **+1**. FP on train: 0 (of the 112 docs with a front PMI heading,
15 have a short title above it — 14 all-caps, 1 mixed-case c0735; the 97 long
program names are excluded by length/punctuation). Generalization FP risk:
**low** — a short lowercase program title on the title page is exactly the
violation being checked.

### 4. metadata — `_title_sheet_missing` fix (+1 TP)

`ESPD-19.105-78-2.1-title_sheet_missing` (c0273). The old suppression
"heading text present anywhere" was defeated by the exact phrase
«Программа и методика испытаний» appearing in a trailing references-table
cell. Fixed to count the title sheet as present only when the exact PMI
heading is a **direct body paragraph in the front region** (before the
«Аннотация» heading), excluding table cells. c0273 starts directly with
«Аннотация» (all preceding paragraphs blank) → now fires at the first
non-empty paragraph, span (7,16) == gold exactly.

TP on train: **+1**. FP on train: 0 — every other doc has a front direct PMI
heading (record ~16) and stays suppressed; c0273 is the only doc without one.
Generalization FP risk: **low** — requires a document whose body begins with
«Аннотация»/«Содержание» while having no title-page heading, which is
precisely the defect.

### 5. notes_footnotes — `_footnote_marker_check` (+1 TP)

`ESPD-19.106-78-2.2.4-footnote_marker` (c0642). ГОСТ 19.106-78 п. 2.2.4
requires the marker to be a superscript arabic numeral with a parenthesis.
Rule: a body paragraph (outside tables) with an inline `letter<number>)`
pattern ("измерения1)") whose marker run is NOT `vertAlign=superscript`,
followed within 4 paragraphs by the note-text paragraph starting with the
same `<number>)`. Both the inline-marker and the following-note signals are
required. Span (35398,35419) == gold exactly.

TP on train: **+1**. FP on train: 0 — across all 113 only c0642 has an inline
`letter<digit>)` marker; c0532/c0542 carry footnote text "1) …" but no inline
marker run (their markers live in/above tables already handled by the
table-footnote rule), so they never fire. Generalization FP risk: **low** —
requires both an attached non-superscript marker *and* the matching note line
below; a correctly superscripted marker never fires.

### 6. appendices — `_appendix_toc_missing` (+1 TP)

`ESPD-19.106-78-2.9.5-appendix_toc_missing` (c0619). ГОСТ 19.106-78 п. 2.9.5
requires every appendix to be listed in the contents. Rule: collect
`ПРИЛОЖЕНИЕ N` labels (arabic, optional «№»); collect appendix numbers named
on `Содержание:`/`Содержание：` lines; when at least one labelled appendix is
omitted while a contents line names at least one appendix, the missing
appendix is not in the contents — anchored at the last contents line that
names an appendix. In c0619 the contents line lists only «Приложение 1 —
Пример испытания.» while ПРИЛОЖЕНИЕ 2 exists → appendix 2 omitted; span
(35397,35441) == gold exactly. Clean appendix docs
(c0579/c0586/c0592/c0597/c0606/c0611) list both 1 and 2 → no fire; c0606's
malformed label «ПРИЛОЖЕНИЕ №2» is matched with the optional «№».

TP on train: **+1**. FP on train: 0. Generalization FP risk: **low–medium** —
requires a "Содержание:" colon-statement naming usable appendices while some
labelled appendix is absent; real contents are a "Содержание" heading +
entries (no colon) and never match.

## Before / after (frozen metric `sc35-span-overlap-v1`, full 113 train)

Command: `uv run --with lxml==6.1.3 --with jsonschema==4.26.0 \
analysis/exp_eval.py <cand>` — builds `build_review` + `review_document
(client=None)`, validates every review against the frozen schema
(jsonschema Draft202012Validator, raises on the first invalid one — ran
clean), scores with the frozen metric runtime.

| metric | formal-exp2-base | formal-exp3 | Δ |
|---|---|---|---|
| score_percent (macro F1) | 85.481857 | **90.141608** | **+4.6598** |
| micro precision | 1.0 | 1.0 | — |
| micro recall % | 79.376083 | 86.343154 | +6.9671 |
| intersection / predicted chars | 2290 / 2290 | 2491 / 2491 | +201 / +201 |
| gold chars | 2885 | 2885 | — |

Per-category F1 (before → after), `predicted_chars == intersection_chars`
(P=100.0) in every category:

| category | base F1 | exp3 F1 | events tp/fp/fn |
|---|---|---|---|
| annotation_contents | 1.0000 | 1.0000 | 7/0/0 |
| appendices | 0.7582 | **1.0000** | 6/0/0 (was 5/0/1) |
| figures | 0.6243 | 0.6243 | 5/0/3 |
| formulas | 0.9441 | 0.9441 | 8/0/5 |
| headings | 0.9099 | **1.0000** | 17/0/0 (was 14/0/3) |
| metadata | 0.8381 | **1.0000** | 16/0/0 (was 12/0/4) |
| notes_footnotes | 0.9346 | **1.0000** | 5/0/0 (was 4/0/1) |
| page_layout | 1.0000 | 1.0000 | 5/0/0 |
| paragraphs_lists | 1.0000 | 1.0000 | 7/0/0 |
| references_terms | 0.2486 | 0.2486 | 1/0/5 |
| sections_content | 1.0000 | 1.0000 | 21/0/0 |
| tables | 1.0000 | 1.0000 | 8/0/0 |

Added findings (verified per finding against gold — all **exact** matches of
rule_id AND spans): 9 —
appendix_toc_missing(c0619), heading_before_spacing(c0409),
heading_after_spacing(c0402), heading_line_spacing(c0413),
designation_country_separator(c0655), designation_organization_separator(c0659),
title_program_title_case(c0735), title_sheet_missing(c0273),
footnote_marker(c0642). Removed: 0. False positives: 0 (predicted ==
intersection in every category). All 113 reviews schema-valid.

## Not shipped (re-examined this round; cannot be proven to zero/low FP)

Remaining FNs by category (figures 3, formulas 5, references_terms 5 — the
task scoped these out of this round except formulas, which was re-checked):

- formulas **c0434 `formula_number_position`** — a numbered display formula
  `jc=left` while another numbered display formula is `jc=right`. Fires only
  on c0434 with the exact gold span (TP=1, FP=0 on train), but the
  generalization FP risk is **medium**: real documents may legitimately mix
  display-formula alignment and there is no reliable zone definition. Already
  probed in exp2 and deliberately not shipped; no new low-risk subrule showed
  up this round.
- formulas **c0648 `formula_part_prefix`** — «x = 5 (1)» un-prefixed while
  «x = 6 (6.2)» is prefixed, with the part context derived from the sentence
  «…относится к части 6 документа…». Detecting the part context requires
  sentence-level parsing; TP=1 with FP=0 on train, but the anchor is
  semantically loaded → **medium** generalization FP risk. Not shipped.
- formulas c0451 `symbol_missing`, c0464 `symbol_newline` — anchors collapse
  to single `\n` characters; c0476 `parameter_dimension` — unit semantics.
  Semantic/render; not deterministic.
- figures c0511/c0520/c0526 (figure element-order) — corpus has no
  `w:drawing`; text heuristics carry high FP risk. Not deterministic.
- references_terms c0286/c0289/c0299/c0306/c0313 — semantic defects anchored
  at the synthetic «Контрольный фрагмент N» label (test-fit). Not attacked.

## Files

- Changed: `runtime/sc35/extra/extra_checks.py` (6 check units — two fixes in
  `_main_designation`/`_title_sheet_missing` plus four new functions — all
  registered in `build_review`), `NOTES.md`.
- Everything else byte-identical to `formal-exp2-base` (verified by recursive
  diff — single differing file).
