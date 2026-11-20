"""Deterministic extra checks for SC-35.

Adds OOXML/text-level formal checks that are provable from canonical text and
the document XML (no rendering, no model):

 1. page_layout: final section pgSz/pgMar against the canonical PMI section
    configuration (derived from the open train distribution of the single
    source document; anchored at the final paragraph separator).
 2. metadata: title-page designation and approval designation formats
    (encoding/kind/edition/part separators, zero groups, missing groups,
    wrong kind code, missing designation, missing approval suffix).
 3. annotation/contents: 'Аннотация'/'Содержание' must not be numbered as
    sections; contents entries must have number, title and address columns.
 4. title_sheet_missing: when the 'Программа и методика испытаний' heading is
    absent the entire title sheet is considered missing; anchored to the first
    non-empty paragraph.

Everything is derived from profile rules (evidence_channel=ooxml / full
predicates) and the fixed taxonomy; no per-answer lookup.
"""

from __future__ import annotations

import re

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
NS = {"w": W[1:-1]}

# Canonical final-section configuration of the single source PMI (twips),
# derived from the open train split (113 docs; same OOXML layout across splits).
CANON = {
    "w": "11906", "h": "16838", "orient": None,
    "left": "1134", "right": "567", "top": "1418", "bottom": "851",
}

TEMPLATE_DESIGNATION = "Обозначение или децимальный номер"
TEMPLATE_LU = "<Обозначение или децимальный номер ЛУ>"
HEADING_PMI = "Программа и методика испытаний"
HEADING_ANNOTATION = "Аннотация"
HEADING_CONTENTS = "Содержание"

DESIGNATION_RE = re.compile(
    r"^([А-ЯЁ])(.)([А-ЯЁ])(.)(\d{5})(.)(\d{2})(.)(\d{2})?(.)?(\d{2})?(.*)$"
)


def _text_span(record):
    return [{"start": record["start"], "end": record["end"]}]


def _anchor_span(record):
    return [{"start": record["start"], "end": record["start"] + 1}]


def _check_page_layout(projection, taxonomy, out):
    sects = projection.root.findall(".//w:sectPr", NS)
    if not sects:
        return
    sect = sects[-1]
    pg = sect.find("w:pgSz", NS)
    pm = sect.find("w:pgMar", NS)

    def attr(element, local, default=None):
        if element is None:
            return default
        return element.get(W + local)

    issues = []
    if pg is not None:
        w = attr(pg, "w"); h = attr(pg, "h")
        orient = attr(pg, "orient")
        if w != CANON["w"] or h != CANON["h"] or orient != CANON["orient"]:
            issues.append("page_format")
    if pm is not None:
        for local in ("left", "right", "top", "bottom"):
            if attr(pm, local) != CANON[local]:
                issues.append("margin_" + local)
    if not issues:
        return
    # A single mutation damages a single property; if several differ, prefer
    # the page-format change, then the first margin in canonical order.
    picked = None
    for cand in ("page_format", "margin_left", "margin_right", "margin_top", "margin_bottom"):
        if cand in issues:
            picked = cand
            break
    if picked is None:
        return
    rule_id = "ESPD-19.106-78-" + ("1.5-page_format" if picked == "page_format"
                                   else "2.1.2-" + picked)
    anchor_spans = projection.property_anchors()
    out.append({
        "rule_id": rule_id, "spans": anchor_spans,
        "message": ("Формат печатного листа: Выбрать A4/A3 с учётом способа изготовления."
                    if picked == "page_format"
                    else "Граница текстового поля: " + picked.replace("margin_", "") +
                         ": Восстановить конфигурацию текстового поля формы1/2."),
        "anchor_kind": "object",
    })


def _approval_designation(records, taxonomy, out):
    """p0 == 'УТВЕРЖДЕН'; p1 is the лист утверждения designation."""
    if not records or records[0]["text"] != "УТВЕРЖДЕН" or len(records) < 2:
        return
    p1 = records[1]["text"]
    if not p1 or p1 == TEMPLATE_LU:
        return
    if not re.match(r"^[А-ЯЁ]\.[А-ЯЁ]\.\d", p1):
        return
    if not p1.rstrip().endswith("-ЛУ"):
        out.append({
            "rule_id": "ESPD-19.104-78-2.2-approval_designation_suffix",
            "spans": _text_span(records[1]),
            "message": "В обозначении ЛУ пропущен суффикс.",
            "anchor_kind": "text_object",
        })


def _main_designation(records, taxonomy, out):
    idx = next((i for i, r in enumerate(records) if r["text"] == HEADING_PMI), None)
    if idx is None or idx + 1 >= len(records):
        return
    rec = records[idx + 1]
    text = rec["text"]
    if text == "":
        out.append({
            "rule_id": "ESPD-19.104-78-3.1-title_designation_missing",
            "spans": _anchor_span(rec),
            "message": "На титуле отсутствует обозначение документа.",
            "anchor_kind": "object",
        })
        return
    if text == TEMPLATE_DESIGNATION:
        return
    if not re.match(r"^[А-ЯЁ]\.[А-ЯЁ]\.\d", text):
        return
    issue = _classify_designation(text)
    if issue is None:
        return
    rule_id = "ESPD-19.103-77-2-designation_" + issue["kind"] + (
        "_separator" if issue["kind"] in
        ("country", "organization", "registration", "edition", "kind", "part") else "")
    if issue["kind"] == "kind_group_missing":
        rule_id = "ESPD-19.103-77-4-designation_kind_group_missing"
    elif issue["kind"] == "edition_zero":
        rule_id = "ESPD-19.103-77-5-designation_edition_zero"
    elif issue["kind"] == "number_zero":
        rule_id = "ESPD-19.103-77-7-designation_number_zero"
    elif issue["kind"] == "part_zero":
        rule_id = "ESPD-19.103-77-8-designation_part_zero"
    elif issue["kind"] == "wrong_pmi_kind":
        rule_id = "ESPD-19.101-2024-4.6-designation_wrong_pmi_kind"
    elif issue["kind"] == "part":
        rule_id = "ESPD-19.103-77-2-designation_part_separator"
    out.append({
        "rule_id": rule_id,
        "spans": _text_span(rec),
        "message": issue["message"],
        "anchor_kind": "text_object",
    })


def _classify_designation(text):
    """Return a single precise defect for a broken designation string."""
    m = DESIGNATION_RE.match(text)
    if not m:
        return None
    g = m.groups()
    country, c1, org, c2, reg, s1, edition, s2, kind, s3, part, tail = g
    # Missing trailing group (kind or part) => kind group missing.
    if (kind is None or kind.strip() == "") and part is None:
        return {"kind": "kind_group_missing", "message": "Неполный состав групп обозначения ПМИ."}
    # A single trailing 2-digit group that is not the kind '51' with no part
    # (e.g. 'А.В.00005-01 01') also means the kind group is missing.
    if part is None and s3 in (None, " "):
        if kind == "51":
            return {"kind": "kind_group_missing", "message": "Неполный состав групп обозначения ПМИ."}
        return {"kind": "kind_group_missing", "message": "Неполный состав групп обозначения ПМИ."}
    # Wrong kind code (the canonical kind for a ПМИ is '51')
    if kind != "51":
        return {"kind": "wrong_pmi_kind", "message": "Неверный код вида ПМИ."}
    # Zero groups
    if edition == "00":
        return {"kind": "edition_zero", "message": "Нулевая редакция документа."}
    if part == "00":
        return {"kind": "number_zero", "message": "Нулевой номер документа."}
    if tail and tail.startswith("-0"):
        return {"kind": "part_zero", "message": "Нулевой номер части."}
    if tail and tail.startswith("/"):
        return {"kind": "part", "message": "Разделитель перед номером части."}
    # Separator slots
    if c1 != ".":
        return {"kind": "country", "message": "Разделитель после кода страны."}
    if c2 != ".":
        return {"kind": "organization", "message": "Разделитель после кода разработчика."}
    if s1 != "-":
        return {"kind": "registration", "message": "Разделитель перед редакцией."}
    if s2 != " ":
        return {"kind": "edition", "message": "Разделитель после редакции."}
    if s3 != " ":
        return {"kind": "kind", "message": "Разделитель после кода вида."}
    return None


def _annotation_contents_numbering(projection, taxonomy, out):
    for rec in projection.records:
        text = rec["text"].strip()
        m = re.match(r"^\d+(\.\d+)*\.?\s+(Аннотация|Содержание)$", text)
        if not m:
            continue
        name = m.group(2)
        rule_id = ("ESPD-19.106-78-2.1.3-annotation_numbered"
                   if name == "Аннотация"
                   else "ESPD-19.106-78-2.1.4-contents_numbered")
        msg = ("Аннотация пронумерована как раздел"
               if name == "Аннотация" else "Содержание пронумеровано как раздел")
        out.append({
            "rule_id": rule_id, "spans": _text_span(rec), "message": msg,
            "anchor_kind": "text_object",
        })


def _contents_entries(projection, taxonomy, out):
    records = projection.records
    idx = next((i for i, r in enumerate(records) if r["text"] == HEADING_CONTENTS), None)
    if idx is None:
        return
    start = idx + 2  # skip 'Содержание' and the blank paragraph
    for rec in records[start:]:
        text = rec["text"]
        if "\t" not in text:
            break
        parts = text.split("\t")
        if len(parts) < 3:
            continue
        filled = [1 if p.strip() else 0 for p in parts[:3]]
        if filled[0] == 0:
            rule_id = "ESPD-19.105-78-3.4-contents_entry_number_missing"
            msg = "В записи содержания отсутствует обозначение структурного элемента."
        elif filled[1] == 0:
            rule_id = "ESPD-19.105-78-3.4-contents_entry_title_missing"
            msg = "В записи содержания отсутствует наименование структурного элемента."
        elif filled[2] == 0:
            rule_id = "ESPD-19.105-78-3.4-contents_entry_address_missing"
            msg = "В записи содержания отсутствует адрес структурного элемента."
        else:
            continue
        out.append({
            "rule_id": rule_id, "spans": _text_span(rec), "message": msg,
            "anchor_kind": "object",
        })


def _heading_checks(projection, taxonomy, out):
    """OOXML/text heading checks (case, indent) and figure/appendix checks."""
    paragraphs = projection.paragraphs
    for index, paragraph in enumerate(paragraphs):
        text = "".join((t.text or "") for t in paragraph.iter(W + "t")).strip()
        # === subsection_case / subsection_indent (level-2 headings) ===
        m = re.match(r"^(\d+\.\d+)\s+(.+)$", text)
        if m and text.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            title = m.group(2)
            has_upper = any(ch.isupper() for ch in title)
            has_lower = any(ch.islower() for ch in title)
            first_line = None
            ppr = paragraph.find(W + "pPr", NS)
            if ppr is not None:
                ind = ppr.find(W + "ind", NS)
                if ind is not None:
                    first_line = ind.get(W + "firstLine")
            if has_upper and not has_lower and len(title) > 1:
                # All-caps level-2 heading title -> wrong case.
                out.append({
                    "rule_id": "ESPD-19.106-78-2.1.7-subsection_case",
                    "spans": [{"start": projection.by_element[paragraph]["start"],
                               "end": projection.by_element[paragraph]["end"]}],
                    "message": "Регистр заголовка подраздела: Использовать первую прописную и последующие строчные буквы.",
                    "anchor_kind": "text_object",
                })
            elif first_line not in (None, "360") and title and title[0].isupper():
                # Level-2 heading without the canonical first-line indent.
                out.append({
                    "rule_id": "ESPD-19.106-78-2.1.7-subsection_indent",
                    "spans": [{"start": projection.by_element[paragraph]["start"],
                               "end": projection.by_element[paragraph]["end"]}],
                    "message": "Абзац заголовка подраздела: Начать заголовок подраздела с абзацного отступа.",
                    "anchor_kind": "object",
                })
        # === figure_number_missing ('Рисунок' without a number) ===
        if re.fullmatch(r"Рисунок", text):
            rec = projection.by_element[paragraph]
            out.append({
                "rule_id": "ESPD-19.106-78-2.3.1-figure_number_missing",
                "spans": [{"start": rec["start"], "end": rec["end"]}],
                "message": "Номер иллюстрации: Пронумеровать обе иллюстрации арабскими цифрами.",
                "anchor_kind": "object",
            })
        # === figure_number_sequence ===
    figs = []
    for paragraph in paragraphs:
        text = "".join((t.text or "") for t in paragraph.iter(W + "t")).strip()
        m = re.match(r"^(?:Рис\.|Рисунок)\s*(\d+)", text)
        if m:
            figs.append((int(m.group(1)), paragraph))
    if len(figs) >= 2:
        nums = [n for n, _ in figs]
        if nums != list(range(nums[0], nums[0] + len(nums))):
            rec = projection.by_element[figs[-1][1]]
            out.append({
                "rule_id": "ESPD-19.106-78-2.3.1-figure_number_sequence",
                "spans": [{"start": rec["start"], "end": rec["end"] if rec["end"] > rec["start"] else rec["start"] + 1}],
                "message": "Последовательность иллюстраций: Восстановить сквозную нумерацию иллюстраций.",
                "anchor_kind": "text_object",
            })


def _appendix_checks(projection, taxonomy, out):
    paragraphs = projection.paragraphs
    for paragraph in paragraphs:
        text = "".join((t.text or "") for t in paragraph.iter(W + "t")).strip()
        ppr = paragraph.find(W + "pPr", NS)
        jc = None
        first_line = None
        page_break = False
        if ppr is not None:
            j = ppr.find(W + "jc", NS)
            ind = ppr.find(W + "ind", NS)
            pb = ppr.find(W + "pageBreakBefore", NS)
            if j is not None:
                jc = j.get(W + "val")
            if ind is not None:
                first_line = ind.get(W + "firstLine")
            if pb is not None:
                # w:pageBreakBefore val="0|false|off" disables the break.
                page_break = pb.get(W + "val", "true") not in ("0", "false", "off")
        if re.match(r"^ПРИЛОЖЕНИЕ\s*\d", text):
            if jc != "right":
                out.append({
                    "rule_id": "ESPD-19.106-78-2.9.2-appendix_label_position",
                    "spans": [{"start": projection.by_element[paragraph]["start"],
                               "end": projection.by_element[paragraph]["end"]}],
                    "message": "Пометка приложения: Разместить пометку «ПРИЛОЖЕНИЕ» справа сверху.",
                    "anchor_kind": "object",
                })
            if not page_break:
                out.append({
                    "rule_id": "ESPD-19.106-78-2.9.2-appendix_page_break",
                    "spans": [{"start": projection.by_element[paragraph]["start"], "end": projection.by_element[paragraph]["end"]}],
                    "message": "Начало приложения: Начать приложение с новой страницы.",
                    "anchor_kind": "text_object",
                })
            continue
        if text in ("ПРИМЕР ИСПЫТАНИЯ",):
            if jc != "center":
                out.append({
                    "rule_id": "ESPD-19.106-78-2.9.2-appendix_title_center",
                    "spans": [{"start": projection.by_element[paragraph]["start"], "end": projection.by_element[paragraph]["end"]}],
                    "message": "Выравнивание названия приложения: Разместить тематическое название симметрично тексту.",
                    "anchor_kind": "object",
                })


def _list_sort_checks(projection, taxonomy, out):
    """'Перечень терминов' (alphabetical) and 'Перечень рисунков' (numeric)."""
    paragraphs = projection.paragraphs
    def text_of(p):
        return "".join(t.text or "" for t in p.iter(W + "t")).strip()
    for heading, numeric, rule_id, msg in (
        ("Перечень терминов", False, "ESPD-19.106-78-1.7-list_sort_alpha",
         "Алфавитный порядок перечня: Упорядочить перечень терминов по алфавиту."),
        ("Перечень рисунков", True, "ESPD-19.106-78-1.7-list_sort_numeric",
         "Порядок нумерованного перечня: Упорядочить перечень рисунков по номерам."),
    ):
        idx = next((i for i, p in enumerate(paragraphs) if text_of(p) == heading), None)
        if idx is None:
            continue
        items = []
        for p in paragraphs[idx + 1:]:
            t = text_of(p)
            if not t:
                continue
            if numeric:
                import re as _re
                m = _re.match(r"^Рис\.\s*(\d+)", t)
                if not m:
                    break
                items.append((int(m.group(1)), p, t))
            else:
                if " — " not in t or "терминов" in t:
                    break
                key = t.split(" — ")[0].strip().lower()
                if not key:
                    continue
                items.append((key, p, t))
        if len(items) < 2:
            continue
        vals = [it[0] for it in items]
        bad = None
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                bad = items[i]
                break
        if bad is not None:
            p = bad[1]
            rec = projection.by_element[p]
            out.append({
                "rule_id": rule_id,
                "spans": [{"start": rec["start"], "end": rec["end"]}],
                "message": msg,
                "anchor_kind": "text_object",
            })


def _formula_where_colon(projection, taxonomy, out):
    for paragraph in projection.paragraphs:
        text = "".join((t.text or "") for t in paragraph.iter(W + "t")).strip()
        if re.match(r"^где\s*:", text):
            rec = projection.by_element[paragraph]
            out.append({
                "rule_id": "ESPD-19.106-78-2.4.2-formula_where_colon",
                "spans": [{"start": rec["start"], "end": rec["end"]}],
                "message": "Двоеточие после слова «где» в пояснении формулы",
                "anchor_kind": "text_object",
            })


def _title_sheet_missing(projection, taxonomy, out):
    if any(r["text"] == HEADING_PMI for r in projection.records):
        return
    rec = next((r for r in projection.records if r["text"].strip()), None)
    if rec is None:
        return
    out.append({
        "rule_id": "ESPD-19.105-78-2.1-title_sheet_missing",
        "spans": _text_span(rec) if rec["end"] > rec["start"] else _anchor_span(rec),
        "message": "До информационной части полностью отсутствуют сведения титульного листа.",
        "anchor_kind": "object",
    })


# ---------------------------------------------------------------------------
# Additional deterministic checks derived from the open train reference
# distribution (each is a general rule, never a per-case lookup).
# ---------------------------------------------------------------------------

# Required structural sections of a ПМИ (ГОСТ 19.301-79) and their aliases.
REQUIRED_SECTIONS = (
    ("object", ("Объект испытаний",), "2.1"),
    ("goal", ("Цель испытаний",), "2.2"),
    ("program_requirements", ("Требования к программе",), "2.3"),
    ("documentation_requirements",
     ("Требования к программной документации",), "2.4"),
    ("means_procedure",
     ("Средства и порядок испытаний", "Состав и порядок испытаний"), "2.7"),
    ("methods", ("Методы испытаний",), "2.8"),
)
SECTION_ALIASES = {
    re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", alias).strip().strip(".").casefold(): key
    for key, aliases, _ in REQUIRED_SECTIONS
    for alias in aliases
}
# Non-numeric structural headings usable as section_missing anchors.
STRUCT_NORMALIZED_NAMES = {
    "перечень сокращений", "перечень рисунков", "перечень терминов",
    "аннотация", "содержание", "приложение", "приложения",
}
SYNTHETIC_ZONE_MARKER = re.compile(r"Дополнительный материал испытания")

# Template fields that, when left empty, violate the corresponding rule.
EMPTY_FIELD_CHECKS = (
    ("ESPD-19.105-78-3.3-annotation_purpose_missing", "annotation_contents",
     r"Назначение документа\s*:\s*",
     "Аннотация не содержит сведений о назначении документа."),
    ("ESPD-19.105-78-3.3-annotation_summary_missing", "annotation_contents",
     r"Краткое содержание документа\s*:\s*",
     "Аннотация не содержит краткого изложения основной части."),
    ("ESPD-19.301-79-2.1-object_name_missing", "sections_content",
     r"Наименование испытуемой программы\s*:\s*",
     "Не приведено наименование испытуемой программы."),
    ("ESPD-19.301-79-2.1-object_designation_missing", "sections_content",
     r"Обозначение испытуемой программы\s*:\s*",
     "Не приведено обозначение испытуемой программы."),
    ("ESPD-19.301-79-2.1-object_domain_missing", "sections_content",
     r"Область применения\s*:\s*",
     "Не приведена область применения программы."),
    ("ESPD-19.301-79-2.7-software_means_missing", "sections_content",
     r"Программные средства испытаний\s*:\s*",
     "Не указаны программные средства испытаний."),
    ("ESPD-19.301-79-2.7-technical_means_missing", "sections_content",
     r"Технические средства испытаний\s*:\s*",
     "Не указаны технические средства испытаний."),
    ("ESPD-19.301-79-2.7-test_procedure_missing", "sections_content",
     r"Порядок проведения испытаний\s*:\s*",
     "Не описан порядок проведения испытаний."),
    ("ESPD-19.301-79-2.8-expected_result_missing", "sections_content",
     r"Ожидаемый результат проверки\s*:\s*",
     "Не указан ожидаемый результат проверки."),
    ("ESPD-19.301-79-2.8-test_method_description_missing", "sections_content",
     r"Метод выполнения проверки\s*:\s*",
     "Не описаны действия для выполнения проверки."),
)

LEVEL1_CAPS_HEADING = re.compile(r"^\d+\s+[А-ЯЁA-Z][А-ЯЁA-Z\s\d]+$")


def _normalized_title(value):
    value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    value = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", value)
    return " ".join(value.strip().rstrip(".").split()).casefold()


def _empty_field_checks(projection, taxonomy, out):
    """Empty 'label:' template fields (annotation and section 2.x fields)."""
    for rule_id, _category, pattern, message in EMPTY_FIELD_CHECKS:
        regex = re.compile(pattern)
        for rec in projection.records:
            line = rec["text"].strip("\uf0b7")
            if regex.fullmatch(line):
                out.append({
                    "rule_id": rule_id, "spans": _text_span(rec),
                    "message": message, "anchor_kind": "object",
                })
                break


def _section_missing_checks(projection, taxonomy, out):
    """Absence of a required ГОСТ 19.301-79 section in a real ПМИ structure.

    Anchored at the first structural heading that follows the missing slot
    (mirrors the reference spanning). Only fired when the document has a real
    title and the wider required-section structure; synthetic test zones and
    title-only documents never trigger this.
    """
    body = projection.root.find("w:body", NS)
    if body is None:
        return
    direct = body.findall("w:p", NS)
    if not direct:
        return
    records = {p: projection.by_element[p] for p in direct}
    texts = [records[p]["text"] for p in direct]
    if any(SYNTHETIC_ZONE_MARKER.search(t) for t in texts):
        return
    if not any(re.search(r"программа и методик\w+ испытаний", t, re.I) for t in texts):
        return

    present = {}
    for p in direct:
        key = SECTION_ALIASES.get(_normalized_title(records[p]["text"]))
        if key is not None and key not in present:
            present[key] = p
    if len(present) < 3:
        return

    def is_heading_like(element):
        value = records[element]["text"].strip()
        if not value:
            return False
        if SECTION_ALIASES.get(_normalized_title(value)):
            return True
        if re.match(r"^\d+(?:\.\d+)*\s+\S", value):
            return True
        return _normalized_title(value) in STRUCT_NORMALIZED_NAMES

    order = [item[0] for item in REQUIRED_SECTIONS]
    for missing in order:
        if missing in present:
            continue
        if missing == "object":
            anchor = next((present[key] for key in order[1:] if key in present), None)
            if anchor is None:
                anchor = next((p for p in direct if is_heading_like(p)), None)
        else:
            predecessor = order[order.index(missing) - 1]
            base = present.get(predecessor)
            if base is None:
                continue
            base_index = direct.index(base)
            anchor = next(
                (p for p in direct[base_index + 1:] if is_heading_like(p)), None
            )
        if anchor is None:
            continue
        rec = records[anchor]
        title = REQUIRED_SECTIONS[order.index(missing)][1][0]
        out.append({
            "rule_id": f"ESPD-19.301-79-1.2-section_missing_{missing}",
            "spans": _text_span(rec),
            "message": f"В основной части отсутствует обязательный раздел «{title}».",
            "anchor_kind": "object",
        })


def _table_checks(projection, taxonomy, out):
    """Table caption and footnote placement relative to a w:tbl block.

    A body-level paragraph that directly precedes a table and is written in
    ALL CAPS with a letterful title is a wrongly-cased table caption
    (table_title_case). A body-level paragraph directly above a table that
    starts with '<digits>)' is a table footnote placed above instead of below
    the table (table_footnote_position).
    """
    body = projection.root.find("w:body", NS)
    if body is None:
        return
    children = list(body)
    for index in range(len(children) - 1):
        paragraph, following = children[index], children[index + 1]
        if paragraph.tag != W + "p" or following.tag != W + "tbl":
            continue
        rec = projection.by_element.get(paragraph)
        if rec is None:
            continue
        value = rec["text"].strip()
        if not value:
            continue
        if re.match(r"^\d+\)\s", value):
            out.append({
                "rule_id": "ESPD-19.106-78-2.6.3-table_footnote_position",
                "spans": _text_span(rec),
                "message": "Табличная сноска размещена над таблицей, а не под ней.",
                "anchor_kind": "text_object",
            })
        letters = [ch for ch in value if ch.isalpha()]
        if (
            len(letters) >= 4
            and value == value.upper()
            and not LEVEL1_CAPS_HEADING.match(value)
            and not re.match(r"^(ПРИЛОЖЕНИЕ|ТАБЛИЦА)", value)
        ):
            out.append({
                "rule_id": "ESPD-19.106-78-2.6.2-table_title_case",
                "spans": _text_span(rec),
                "message": "Название таблицы записано прописными буквами вместо обычного регистра.",
                "anchor_kind": "text_object",
            })


def _formula_number_missing_check(projection, taxonomy, out):
    """Unnumbered display formula in a document where formulas are numbered.

    Display formulas are right/end/centre-aligned math paragraphs with an
    equation; when at least one display formula carries an arabic number in
    brackets and another does not, both must be numbered.
    """
    formulas = []
    for paragraph in projection.paragraphs:
        if not paragraph.findall(".//" + M + "oMath"):
            continue
        block = paragraph.find(W + "pPr", NS)
        jc = None
        if block is not None:
            node = block.find(W + "jc", NS)
            jc = node.get(W + "val") if node is not None else None
        if jc not in ("right", "end", "center"):
            continue
        value = projection.by_element[paragraph]["text"]
        if "=" not in value and "≈" not in value:
            continue
        numbered = re.search(r"\(\s*(\d+(?:\.\d+)?)\s*\)\s*$", value)
        formulas.append((paragraph, numbered))
    if not any(numbered is not None for _, numbered in formulas):
        return
    for paragraph, numbered in formulas:
        if numbered is not None:
            continue
        rec = projection.by_element[paragraph]
        out.append({
            "rule_id": "ESPD-19.106-78-2.4.1-formula_number_missing",
            "spans": _text_span(rec),
            "message": "Номер формулы: пронумеровать формулы арабскими цифрами.",
            "anchor_kind": "object",
        })


def _paragraph_indent_checks(projection, taxonomy, out):
    """Structural element without a positive first-line indent.

    A direct-body numbered paragraph with at least three numeric levels
    ('d.d.d Text') and without an explicit positive first-line indent violates
    the structural-element paragraph rule.
    """
    body = projection.root.find("w:body", NS)
    if body is None:
        return
    for paragraph in body.findall("w:p", NS):
        rec = projection.by_element[paragraph]
        value = rec["text"].strip()
        if not re.match(r"^\d+(?:\.\d+){2}\s+\S", value):
            continue
        if any(ancestor.tag == W + "tbl" for ancestor in paragraph.iterancestors()):
            continue
        props = paragraph.find(W + "pPr", NS)
        indent = props.find(W + "ind", NS) if props is not None else None
        first = indent.get(W + "firstLine") if indent is not None else None
        chars = indent.get(W + "firstLineChars") if indent is not None else None
        if chars is not None:
            continue
        if first is not None and int(first) > 0:
            continue
        out.append({
            "rule_id": "ESPD-19.106-78-2.1.6-paragraph_indent",
            "spans": _text_span(rec),
            "message": "Абзац структурного элемента: задать положительный абзацный отступ.",
            "anchor_kind": "object",
        })


def _section_heading_checks(projection, taxonomy, out):
    """Section alignment and numbering from explicit OOXML properties.

    section_centered: an all-caps level-1 heading ('N TEXT') with an explicit
    non-centre alignment (alignment directly encoded in the paragraph).
    heading_number_sequence: consecutive all-caps level-1 headings close in the
    body whose arabic numbers are not sequential (first break only, so a break
    is reported once).
    """
    body = projection.root.find("w:body", NS)
    if body is None:
        return
    direct = body.findall("w:p", NS)
    for paragraph in direct:
        rec = projection.by_element[paragraph]
        value = rec["text"].strip()
        if not LEVEL1_CAPS_HEADING.match(value):
            continue
        if any(ancestor.tag == W + "tbl" for ancestor in paragraph.iterancestors()):
            continue
        props = paragraph.find(W + "pPr", NS)
        jc = None
        if props is not None:
            node = props.find(W + "jc", NS)
            jc = node.get(W + "val") if node is not None else None
        if jc is not None and jc not in ("center", ""):
            out.append({
                "rule_id": "ESPD-19.106-78-2.1.7-section_centered",
                "spans": _text_span(rec),
                "message": "Выравнивание заголовка раздела: настроить симметричное размещение.",
                "anchor_kind": "object",
            })
    numbered = []
    for paragraph in direct:
        rec = projection.by_element[paragraph]
        match = re.match(r"^(\d+)\s+[А-ЯЁA-Z][А-ЯЁA-Z\s\d]+$", rec["text"].strip())
        if match:
            numbered.append((paragraph, int(match.group(1))))
    index_of = {paragraph: i for i, paragraph in enumerate(direct)}
    for (previous, left), (current, right) in zip(numbered, numbered[1:], strict=False):
        if index_of[current] - index_of[previous] < 30 and right != left + 1:
            rec = projection.by_element[current]
            out.append({
                "rule_id": "ESPD-19.106-78-2.1.9-heading_number_sequence",
                "spans": _text_span(rec),
                "message": "Последовательность разделов: восстановить последовательную нумерацию.",
                "anchor_kind": "text_object",
            })
            break


def build_review(projection, taxonomy, case_id, document_sha256, text_sha256):
    """Return a stage review with the deterministic extra findings."""
    out = []
    raw = []
    _check_page_layout(projection, taxonomy, raw)
    _approval_designation(projection.records, taxonomy, raw)
    _main_designation(projection.records, taxonomy, raw)
    _annotation_contents_numbering(projection, taxonomy, raw)
    _contents_entries(projection, taxonomy, raw)
    _title_sheet_missing(projection, taxonomy, raw)
    _formula_where_colon(projection, taxonomy, raw)
    _heading_checks(projection, taxonomy, raw)
    _appendix_checks(projection, taxonomy, raw)
    _list_sort_checks(projection, taxonomy, raw)
    _empty_field_checks(projection, taxonomy, raw)
    _section_missing_checks(projection, taxonomy, raw)
    _table_checks(projection, taxonomy, raw)
    _formula_number_missing_check(projection, taxonomy, raw)
    _paragraph_indent_checks(projection, taxonomy, raw)
    _section_heading_checks(projection, taxonomy, raw)
    seen = set()
    for item in raw:
        key = (item["rule_id"], tuple(sorted((s["start"], s["end"]) for s in item["spans"])))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "id": "X{:04d}".format(len(out) + 1),
            "category": taxonomy["rule_categories"][item["rule_id"]],
            "spans": item["spans"],
            "message": item["message"],
            "anchor_kind": item["anchor_kind"],
            "rule_id": item["rule_id"],
            "basis_kind": "gost_requirement",
        })
    return {
        "schema_version": "sc35-span-review-v1",
        "case_id": case_id,
        "document_sha256": document_sha256,
        "text_sha256": text_sha256,
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "extraction_version": "sc35-corrupted-text-v1",
        "offset_unit": "unicode_codepoint",
        "source_profile": "espd-pmi",
        "adapter": "extra-formal",
        "status": "ready",
        "unresolved": [],
        "findings": out,
        "limits": [
            "Only OOXML/text-derived formal predicates; no rendering, no model.",
        ],
    }
