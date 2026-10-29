"""Excel export of the model's proposals on the candidate pairs (step 08a), for people to read: one row per pair with the two uses side by side
(label, paper, definition with its page, meaning profile, measures), the signals, and the model's substitution test and reasons.

Everything in it is the model's PROPOSAL, not a decision: the workbook says so on its first sheet. Whoever reads it is no longer blind to the model's class
on these pairs, so should not be a blind annotator of the same pairs afterwards.
"""
import datetime as dt

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import pairs
from .proposals import PROMPT_VERSION

FONT = "Arial"
MAX_CELL = 32000           # Excel holds at most 32,767 characters per cell
SIDE = (("Label", 22), ("Name in the paper", 22), ("Paper", 40), ("Year", 7), ("Journal", 24), ("DOI", 22), ("Definition (author's)", 60), ("Definition page", 9),
        ("Meaning profile", 50), ("Measures", 40))
COLUMNS = [("Model class", 18), ("Retrieval stratum", 22), *[(f"A: {t}", w) for t, w in SIDE], *[(f"B: {t}", w) for t, w in SIDE],
           ("S_def", 8), ("S_op", 8), ("C_cit", 8), ("Substitution test (model)", 60), ("Reasons (model)", 70), ("Contrary evidence (model)", 50),
           ("Evidence needed (model)", 40), ("Pair ID", 38)]


def _text(value):
    if value is None:
        return None
    text = ILLEGAL_CHARACTERS_RE.sub("", str(value))
    return text if len(text) <= MAX_CELL else text[:MAX_CELL] + " [...]"


def _side(content, journals):
    paper = content.get("paper") or {}
    definitions = [m for m in content.get("mentions", []) if m.get("definition")]
    names = sorted({m.get("surface_form") for m in content.get("mentions", []) if m.get("surface_form")})
    profile = "; ".join(f"{a['kind'].replace('_', ' ')}: {a['value']}" for a in content.get("meaning_profile", []) if a.get("value"))
    measures = "; ".join(m.get("name") or "" for m in content.get("measures", []) if m.get("name"))
    return [content.get("label"), ", ".join(names), paper.get("title"), paper.get("year"), journals.get(str(paper.get("canonical_id"))), paper.get("doi"),
            " | ".join(m["definition"] for m in definitions) or None, ", ".join(str(m["page"]) for m in definitions if m.get("page")) or None, profile or None, measures or None]


def rows(con, signal_run_id, classes=pairs.POSITIVE):
    found = con.execute(
        "SELECT DISTINCT ON (q.pair_id) q.pair_id,q.decision_class,p.hint,p.s_def,p.s_op,p.c_cit,q.substitution_test,q.rationale,q.contrary_evidence,q.evidence_needed,d.content "
        "FROM pair_proposal q JOIN candidate_pair p USING (pair_id) JOIN pair_dossier d USING (pair_id) "
        "WHERE p.signal_run_id=%s AND q.prompt_version=%s ORDER BY q.pair_id,q.created_at DESC,q.proposal_id", (signal_run_id, PROMPT_VERSION)).fetchall()
    found = [r for r in found if r[1] in classes]
    ids = list({r[10][side]["paper"]["canonical_id"] for r in found for side in ("a", "b") if (r[10][side].get("paper") or {}).get("canonical_id")})
    journals = {str(cid): source for cid, source in con.execute("SELECT canonical_id,source_title FROM canonical_paper WHERE canonical_id = ANY(%s::uuid[])", (ids,))}
    out = []
    for pair_id, cls, hint, s_def, s_op, c_cit, substitution, rationale, contrary, needed, content in found:
        out.append([pairs.CLASS_LABELS.get(cls, cls), hint, *_side(content["a"], journals), *_side(content["b"], journals),
                    *(round(v, 3) if v is not None else None for v in (s_def, s_op, c_cit)), substitution, rationale, contrary, needed, str(pair_id)])
    out.sort(key=lambda r: (r[0], r[1], r[2] or "", r[12] or ""))
    return out


def export_xlsx(con, signal_run_id, path, classes=pairs.POSITIVE):
    if not classes or not set(classes) <= set(pairs.CLASSES):
        raise ValueError("Classes are: " + ", ".join(pairs.CLASSES))
    data = rows(con, signal_run_id, classes)
    if not data:
        raise ValueError("No model proposal of these classes on this run")
    wb = Workbook()
    readme = wb.active
    readme.title = "Read me"
    sheet = wb.create_sheet("Pairs")
    bold, plain = Font(name=FONT, bold=True), Font(name=FONT)
    for col, (title, width) in enumerate(COLUMNS, 1):
        cell = sheet.cell(row=1, column=col, value=title)
        cell.font, cell.fill = Font(name=FONT, bold=True, color="FFFFFF"), PatternFill("solid", fgColor="44546A")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.column_dimensions[get_column_letter(col)].width = width
    for r, values in enumerate(data, 2):
        for col, value in enumerate(values, 1):
            cell = sheet.cell(row=r, column=col, value=_text(value) if isinstance(value, str) else value)
            cell.font, cell.alignment = plain, Alignment(wrap_text=True, vertical="top")
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(data) + 1}"
    last = len(data) + 1
    strata = sorted({row[1] for row in data})
    lines = [("Model proposals on construct pairs: " + ", ".join(pairs.CLASS_LABELS[c] for c in classes), bold),
             ("These are PROPOSALS of a language model (prompt " + PROMPT_VERSION + "), not decisions. Nothing here is validated: people decide each pair with the codebook.", plain),
             ("Whoever reads this file is no longer blind to the model's class on these pairs and should not be a blind annotator of them.", plain),
             (f"Signal run {signal_run_id} · exported {dt.date.today().isoformat()} · one row per pair, use A and B side by side (sheet Pairs, filters on the header).", plain),
             ("S_def: closeness of the definitions and meaning profiles; S_op: closeness of the measures; C_cit: shared references. They only explain why the pair was retrieved.", plain),
             ("", plain), ("Pairs by model class and retrieval stratum (counted from sheet Pairs)", bold)]
    for i, (text, font) in enumerate(lines, 1):
        readme.cell(row=i, column=1, value=text).font = font
    top = len(lines) + 1
    readme.cell(row=top, column=1, value="Model class").font = bold
    for j, stratum in enumerate(strata, 2):
        readme.cell(row=top, column=j, value=stratum).font = bold
    readme.cell(row=top, column=len(strata) + 2, value="Total").font = bold
    for i, cls in enumerate(classes, top + 1):
        readme.cell(row=i, column=1, value=pairs.CLASS_LABELS[cls]).font = plain
        for j in range(2, len(strata) + 2):
            col = get_column_letter(j)
            readme.cell(row=i, column=j, value=f"=COUNTIFS(Pairs!$A$2:$A${last},$A{i},Pairs!$B$2:$B${last},{col}${top})").font = plain
        readme.cell(row=i, column=len(strata) + 2, value=f"=SUM(B{i}:{get_column_letter(len(strata) + 1)}{i})").font = bold
    readme.column_dimensions["A"].width = 26
    for j in range(2, len(strata) + 3):
        readme.column_dimensions[get_column_letter(j)].width = 22
    wb.calculation.fullCalcOnLoad = True       # openpyxl stores no computed values: Excel computes the summary when the file is opened
    wb.save(path)
    return {"path": str(path), "pairs": len(data), "by_class": {pairs.CLASS_LABELS[c]: sum(1 for r in data if r[0] == pairs.CLASS_LABELS[c]) for c in classes}}
