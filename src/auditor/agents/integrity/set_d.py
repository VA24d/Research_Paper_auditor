"""Set D: planted integrity and code errors, with the originals as clean controls.

Set D takes the set C papers and their repositories and plants errors that the
integrity screen, the results check and the code-defect check should find:

- a sentence saying results are placeholders, simulated or still to be replaced
  (`unfinished_or_placeholder_content`);
- a number in the text that a table does not give, made by changing a number the text
  quotes from a table (`result_inconsistency`);
- a sentence comparing two table rows the wrong way round;
- a sentence giving the wrong size of a gain between two table rows;
- a mean of whole-number ratings that the stated sample size cannot produce (GRIM);
- an undefined name or a syntax error in a file the experiment runs (`code_defect`).

Everything is deterministic: what is planted, and where, follows from a hash of the
paper id, so a build can be repeated from the committed plant records. The planted
sentences use the paper's own table labels, row names and numbers; the placeholder and
GRIM sentences come from short generic templates, since they have no source in the paper.
"""

import ast
import builtins
import hashlib
import random
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from auditor.agents.code.defects import experiment_files
from auditor.planner.integrity import RESULTS_RULE_ID, RULE_ID
from auditor.schemas import (
    CELL_SEPARATOR,
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    GoldLabel,
    PaperVariant,
    ParsedPaper,
    RuleTarget,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    SettingTarget,
    SourceKind,
    SourceRef,
)

DEFECTS_TARGET_KEY = "code_defects"

PLACEHOLDERS = (
    "The numbers in this table are placeholders and will be updated with the final runs.",
    "Results for the last two settings are simulated and should be replaced with actual "
    "measurements before submission.",
    "TODO: rerun the baselines with the corrected seed and update these numbers.",
    "These scores are preliminary estimates; the full experiments are still running and "
    "the final values will be filled in later.",
    "Accuracy on the held-out split is XX% and will be reported in the camera-ready version.",
)

GRIM_TEMPLATES = (
    "In a small user study, annotators rated the outputs as fluent (M = {mean}, N = {n}).",
    "Participants found the explanations helpful (mean = {mean}, n = {n}) on a 1-5 scale.",
)

Kind = Literal["placeholder", "number", "comparison", "gain", "grim", "undefined_name", "syntax"]

_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
_TABLE_MENTION = re.compile(r"\bTab(?:le)?\.?\s*\d+", re.IGNORECASE)


class Plant(BaseModel):
    """One planted error, enough to plant it again in a fresh copy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label_id: str
    paper_id: str
    kind: Kind
    # Paper side: the sentence changed (with the span replaced) or inserted after.
    sentence_id: str | None = None
    after: str | None = None
    start: int | None = None
    end: int | None = None
    original: str = ""
    text: str = ""
    # Code side.
    path: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class _Row:
    label: str
    values: list[str]
    table: str


def _rng(paper_id: str, salt: str) -> random.Random:
    seed = int(hashlib.sha256(f"{paper_id}|{salt}".encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def _label_id(paper_id: str, kind: str, where: str) -> str:
    return "d-" + hashlib.sha256(f"{paper_id}|{kind}|{where}".encode()).hexdigest()[:12]


def _body(paper: ParsedPaper) -> list[Sentence]:
    return [s for s in paper.sentences if s.kind in (SentenceKind.BODY, SentenceKind.ABSTRACT)]


def _table_rows(paper: ParsedPaper) -> list[_Row]:
    """Rows with a worded label and decimal numbers, from every table of the paper."""
    rows = []
    for table in paper.tables:
        name = table.label or ""
        if not re.fullmatch(r"Table \d+", name):
            continue
        for row_id in table.row_ids:
            cells = paper.sentence(row_id).text.split(CELL_SEPARATOR)
            label = cells[0].strip()
            if not re.search(r"[A-Za-z]{2}", label) or len(label) > 40:
                continue
            values = [m.group() for cell in cells[1:] for m in [_DECIMAL.search(cell)] if m]
            if len(values) >= 1:
                rows.append(_Row(label, values, name))
    return rows


def insert_after(paper: ParsedPaper, after: str, sentence_id: str, text: str) -> ParsedPaper:
    """A copy of the paper with a body sentence inserted after sentence `after`, in its
    section."""
    data = paper.model_dump()
    rows = data["sentences"] = list(data["sentences"])
    at = next(i for i, row in enumerate(rows) if row["id"] == after)
    rows.insert(
        at + 1,
        {
            "id": sentence_id,
            "section_id": rows[at]["section_id"],
            "kind": SentenceKind.BODY.value,
            "text": text,
        },
    )
    data["variant"] = PaperVariant.INJECTED.value
    return ParsedPaper.model_validate(data)


def replace_span(
    paper: ParsedPaper, sentence_id: str, start: int, end: int, new: str
) -> ParsedPaper:
    """A copy of the paper with one span of a sentence replaced; citation markers after it
    move with the text."""
    data = paper.model_dump()
    delta = len(new) - (end - start)
    for row in data["sentences"]:
        if row["id"] == sentence_id:
            row["text"] = row["text"][:start] + new + row["text"][end:]
    for mention in data["citations"]:
        if mention["sentence_id"] == sentence_id and mention["start"] >= end:
            mention["start"] += delta
            mention["end"] += delta
    data["variant"] = PaperVariant.INJECTED.value
    return ParsedPaper.model_validate(data)


def _shifted(value: str, rng: random.Random, avoid: set[str]) -> str | None:
    """A different number written like `value`: same decimals, off by a clear margin."""
    decimals = len(value.split(".")[1])
    number = float(value)
    for _ in range(20):
        step = max(abs(number) * rng.uniform(0.08, 0.2), 1.5 * 10**-decimals * 10)
        new = f"{number + step * rng.choice((-1, 1)):.{decimals}f}"
        if not new.startswith("-") and new not in avoid and new != value:
            return new
    return None


def plan_paper(paper: ParsedPaper) -> list[Plant]:
    """The paper-side plants for one paper; some kinds are skipped when the paper has no
    place for them (no table rows with numbers, say)."""
    pid = paper.paper_id
    body = _body(paper)
    if not body:
        return []
    plants: list[Plant] = []
    mentions = [s for s in body if _TABLE_MENTION.search(s.text)]
    anchor = mentions[0] if mentions else body[len(body) // 2]

    rng = _rng(pid, "placeholder")
    text = rng.choice(PLACEHOLDERS)
    plants.append(
        Plant(
            label_id=_label_id(pid, "placeholder", anchor.id),
            paper_id=pid,
            kind="placeholder",
            sentence_id=f"{anchor.id}-d1",
            after=anchor.id,
            text=text,
        )
    )

    rows = _table_rows(paper)
    in_tables = {v for row in rows for v in row.values}
    # A number the text quotes from one row of one table, in a prose sentence that names
    # that table or that row, so the planted change is checkable from the paper.
    holders: dict[str, list[_Row]] = {}
    for row in rows:
        for value in row.values:
            holders.setdefault(value, []).append(row)
    numbered = []
    for s in body:
        if s.id == anchor.id or not _prose(s.text):
            continue
        for m in _DECIMAL.finditer(s.text):
            named = [
                row
                for row in holders.get(m.group(), [])
                if _names_table(s.text, row.table) or row.label in s.text
            ]
            if len({(row.table, row.label) for row in named}) == 1:
                numbered.append((s, m))
    if numbered:
        rng = _rng(pid, "number")
        sentence, match = rng.choice(numbered)
        new = _shifted(match.group(), rng, in_tables)
        if new is not None:
            plants.append(
                Plant(
                    label_id=_label_id(pid, "number", sentence.id),
                    paper_id=pid,
                    kind="number",
                    sentence_id=sentence.id,
                    start=match.start(),
                    end=match.end(),
                    original=match.group(),
                    text=new,
                )
            )

    pairs = _row_pairs(rows)
    if pairs:
        rng = _rng(pid, "comparison")
        low, high, column = rng.choice(pairs)
        a, b = low.values[column], high.values[column]
        plants.append(
            Plant(
                label_id=_label_id(pid, "comparison", anchor.id),
                paper_id=pid,
                kind="comparison",
                sentence_id=f"{anchor.id}-d2",
                after=anchor.id,
                text=f"In {low.table}, {low.label} reaches {a}, higher than {high.label} ({b}).",
            )
        )
        rng = _rng(pid, "gain")
        low, high, column = rng.choice(pairs)
        gap = float(high.values[column]) - float(low.values[column])
        wrong = gap + max(1.5, gap * 0.6)
        plants.append(
            Plant(
                label_id=_label_id(pid, "gain", anchor.id),
                paper_id=pid,
                kind="gain",
                sentence_id=f"{anchor.id}-d3",
                after=anchor.id,
                text=(
                    f"In {high.table}, {high.label} improves over {low.label} by "
                    f"{wrong:.1f} points ({high.values[column]} against {low.values[column]})."
                ),
            )
        )

    rng = _rng(pid, "grim")
    n = rng.randint(11, 40)
    while True:
        mean = round(rng.uniform(2.0, 4.6), 2)
        if abs(round(mean * n) / n - mean) > 0.005 + 1e-9:
            break
    plants.append(
        Plant(
            label_id=_label_id(pid, "grim", body[-1].id),
            paper_id=pid,
            kind="grim",
            sentence_id=f"{body[-1].id}-d4",
            after=body[-1].id,
            text=rng.choice(GRIM_TEMPLATES).format(mean=f"{mean:.2f}", n=n),
        )
    )
    return plants


def _names_table(text: str, label: str) -> bool:
    number = label.split()[-1]
    return re.search(rf"\bTab(?:le)?\.?\s*{re.escape(number)}\b", text, re.IGNORECASE) is not None


def _prose(text: str) -> bool:
    """A sentence of words, not a table flattened into the body text."""
    tokens = text.split()
    words = [w for w in tokens if re.search(r"[A-Za-z]{3}", w)]
    return len(words) >= 8 and len(words) >= 0.6 * len(tokens)


def _row_pairs(rows: Sequence[_Row]) -> list[tuple[_Row, _Row, int]]:
    """(lower, higher, column) for two rows of one table with distinct labels and a clear
    gap in the same column."""
    pairs = []
    for i, first in enumerate(rows):
        for second in rows[i + 1 :]:
            if first.table != second.table or first.label == second.label:
                continue
            for column in range(min(len(first.values), len(second.values))):
                a, b = float(first.values[column]), float(second.values[column])
                if abs(a - b) < 0.5 or max(a, b) == 0:
                    continue
                low, high = (first, second) if a < b else (second, first)
                pairs.append((low, high, column))
                break
    return pairs[:30]


def apply_plants(paper: ParsedPaper, plants: Sequence[Plant]) -> ParsedPaper:
    """The injected copy: number changes first (their spans refer to the original text),
    then inserted sentences in order."""
    current = paper.model_copy(update={"variant": PaperVariant.INJECTED})
    for plant in plants:
        if plant.kind == "number":
            assert plant.sentence_id
            assert plant.start is not None
            assert plant.end is not None
            found = current.sentence(plant.sentence_id).text[plant.start : plant.end]
            if found != plant.original:
                raise ValueError(f"{plant.label_id}: expected {plant.original!r}, found {found!r}")
            current = replace_span(current, plant.sentence_id, plant.start, plant.end, plant.text)
    for plant in plants:
        if plant.after is not None and plant.sentence_id is not None:
            current = insert_after(current, plant.after, plant.sentence_id, plant.text)
    return current


def label_for(plant: Plant, repository: str | None = None) -> GoldLabel:
    """The gold label of a plant."""
    paper = SourceRef(kind=SourceKind.SUBMISSION, id=plant.paper_id)
    if plant.kind in ("undefined_name", "syntax"):
        assert plant.path is not None
        assert plant.line is not None
        source = SourceRef(kind=SourceKind.REPOSITORY, id=repository or plant.paper_id)
        return GoldLabel(
            label_id=plant.label_id,
            paper_id=plant.paper_id,
            variant=PaperVariant.INJECTED,
            error_class=ErrorClass.CODE_DEFECT,
            anchor=DocumentAnchor(),
            target=SettingTarget(key=DEFECTS_TARGET_KEY),
            rationale=f"Planted {plant.kind.replace('_', ' ')} at {plant.path}:{plant.line}.",
            evidence=(
                Evidence(
                    kind=EvidenceKind.CODE,
                    source=source,
                    text=f"{plant.path}:{plant.line}: {plant.text}",
                    path=plant.path,
                    line_start=plant.line,
                ),
            ),
        )
    assert plant.sentence_id is not None
    placeholder = plant.kind == "placeholder"
    if plant.kind == "number":
        assert plant.start is not None
        anchor = SentenceAnchor(
            sentence_id=plant.sentence_id, start=plant.start, end=plant.start + len(plant.text)
        )
        why = f"The text says {plant.text} where the paper had {plant.original}, a table value."
    else:
        anchor = SentenceAnchor(sentence_id=plant.sentence_id)
        why = f"Planted {plant.kind} sentence: {plant.text}"
    return GoldLabel(
        label_id=plant.label_id,
        paper_id=plant.paper_id,
        variant=PaperVariant.INJECTED,
        error_class=ErrorClass.UNFINISHED_CONTENT
        if placeholder
        else ErrorClass.RESULT_INCONSISTENCY,
        anchor=anchor,
        target=RuleTarget(rule_id=RULE_ID if placeholder else RESULTS_RULE_ID),
        rationale=why,
        evidence=(Evidence(kind=EvidenceKind.QUOTE, source=paper, text=plant.text),),
    )


def plant_code(paper_id: str, repository: Path, destination: Path) -> Plant | None:
    """Copy `repository` to `destination` with one defect in a file the experiment runs:
    a local name misspelled at one use (an undefined name), or, for every other paper, a
    `def` line with its colon removed (a syntax error)."""
    sources = {
        str(path.relative_to(repository)): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(repository.rglob("*.py"))
        if ".git" not in path.parts
    }
    files = experiment_files(sources)
    rng = _rng(paper_id, "code")
    syntax = rng.random() < 0.5
    for path in files:
        text = sources[path]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines(keepends=True)
        spot = _def_line(tree, lines) if syntax else _local_use(tree, text)
        if spot is None:
            continue
        line, column, old, new = spot
        source = lines[line - 1]
        if source[column : column + len(old)] != old:
            continue
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(repository, destination, ignore=shutil.ignore_patterns(".git"))
        lines[line - 1] = source[:column] + new + source[column + len(old) :]
        (destination / path).write_text("".join(lines), encoding="utf-8", newline="")
        kind: Kind = "syntax" if syntax else "undefined_name"
        return Plant(
            label_id=_label_id(paper_id, kind, f"{path}:{line}"),
            paper_id=paper_id,
            kind=kind,
            path=path,
            line=line,
            original=old,
            text=new,
        )
    return None


def _def_line(tree: ast.Module, lines: Sequence[str]) -> tuple[int, int, str, str] | None:
    """A function definition whose header fits on one line and ends in "):": the colon
    is removed."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.body and node.body[0].lineno > node.lineno:
            header = lines[node.lineno - 1].rstrip()
            if header.endswith("):"):
                return node.lineno, len(header) - 2, "):", ")"
    return None


def _local_use(tree: ast.Module, text: str) -> tuple[int, int, str, str] | None:
    """A use of a variable assigned earlier in the same function, on a later line: the
    use is misspelled so that the name is defined nowhere."""
    defined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | set(dir(builtins))
    defined |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        assigned: dict[str, int] = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.setdefault(node.id, node.lineno)
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and len(node.id) >= 4
                and node.id in assigned
                and node.lineno > assigned[node.id]
            ):
                wrong = node.id[:-1] + node.id[-1] * 2
                if wrong in defined or wrong in text:
                    continue
                return node.lineno, node.col_offset, node.id, wrong
    return None
