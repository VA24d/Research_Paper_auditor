"""A ParsedPaper from a paper's LaTeX source, with its formulas kept (plan D7).

GROBID loses much of a PDF's mathematics, so the theory checker sees broken formulas on
real papers. When the source is at hand (an arXiv e-print, or the authors' own files),
this parser reads it instead:

- `\\input` and `\\include` are followed inside the source folder, and comments are removed;
- the title and abstract become metadata and abstract sentences;
- `\\section`, `\\subsection` and `\\subsubsection` start sections;
- each paragraph is one sentence, with its mathematics left as LaTeX, as in SPOT;
- theorem-like environments (theorem, lemma, proof, ...) keep their name as a prefix, so
  "Lemma 3." reads as it does in the PDF;
- tables become `Table`s, one row per `\\\\`, cells joined by `CELL_SEPARATOR`, with their
  captions; figure captions are kept as captions.

The bibliography is left out; references are Part 2's and come from the PDF.
"""

import gzip
import io
import re
import tarfile
from pathlib import Path

from auditor.schemas import (
    CELL_SEPARATOR,
    PaperMetadata,
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceKind,
    Table,
)

PARSER_VERSION = "1"
MAX_INPUT_DEPTH = 10

_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_SECTION = re.compile(r"\\(section|subsection|subsubsection)\*?\s*\{")
_ENV_BLOCK = r"\\begin\{{({names})\*?\}}(.*?)\\end\{{\1\*?\}}"
_THEOREM_LIKE = (
    "theorem|lemma|proposition|corollary|definition|proof|claim|remark|assumption|"
    "conjecture|example|fact|observation"
)
_THEOREM_ENV = re.compile(r"\\begin\{(" + _THEOREM_LIKE + r")\*?\}(\[[^\]]*\])?", re.IGNORECASE)
_END_THEOREM = re.compile(r"\\end\{(?:" + _THEOREM_LIKE + r")\*?\}", re.IGNORECASE)
_DROP_COMMANDS = re.compile(
    r"\\(?:label|vspace|hspace|vskip|newpage|clearpage|centering)\*?(?:\{[^}]*\})?"
)
_FORMAT = re.compile(r"\\(?:textbf|textit|emph|underline|mathbf|text|textrm|textsc)\s*\{([^{}]*)\}")
_RULES = re.compile(
    r"\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\}|cmidrule(?:\([^)]*\))?\{[^}]*\})"
)


# Limits for an unpacked arXiv e-print.
MAX_SOURCE_BYTES = 50_000_000
MAX_SOURCE_FILES = 2_000


class LatexError(ValueError):
    """The source has no document body, or an archive could not be unpacked."""


def unpack_eprint(content: bytes, destination: Path) -> Path:
    """Unpack an arXiv e-print (a gzipped tar of the source, or one gzipped .tex file)
    into `destination`, an empty folder, and return its main .tex file. Members that
    would leave the folder, links and devices are refused (tarfile's "data" filter)."""
    destination.mkdir(parents=True, exist_ok=True)
    try:
        raw = gzip.decompress(content)
    except OSError:
        raw = content
    if len(raw) > MAX_SOURCE_BYTES:
        raise LatexError("the e-print is larger than the limit")
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            members = archive.getmembers()
            if len(members) > MAX_SOURCE_FILES:
                raise LatexError("the e-print has too many files")
            if sum(m.size for m in members) > MAX_SOURCE_BYTES:
                raise LatexError("the e-print is larger than the limit")
            archive.extractall(destination, filter="data")
    except tarfile.ReadError:
        (destination / "main.tex").write_bytes(raw)
    return find_main(destination)


def find_main(folder: Path) -> Path:
    """The .tex file in `folder` that holds the document (\\documentclass and
    \\begin{document}), preferring the largest when several do."""
    found = []
    for path in sorted(folder.rglob("*.tex")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\\documentclass" in text and "\\begin{document}" in text:
            found.append((len(text), path))
    if not found:
        raise LatexError(f"no main .tex file in {folder}")
    return max(found)[1]


def read_source(main: Path) -> str:
    """The main file with its inputs inlined and comments removed. An input that would
    leave the source folder, or is missing, is skipped."""
    root = main.parent.resolve()

    def expand(path: Path, depth: int) -> str:
        text = _COMMENT.sub("", path.read_text(encoding="utf-8", errors="replace"))
        if depth >= MAX_INPUT_DEPTH:
            return text

        def include(match: re.Match[str]) -> str:
            name = match.group(1).strip()
            target = (path.parent / name).resolve()
            if target.suffix != ".tex":
                target = target.with_name(target.name + ".tex")
            if not target.is_relative_to(root) or not target.is_file():
                return ""
            return expand(target, depth + 1)

        return _INPUT.sub(include, text)

    return expand(main.resolve(), 0)


def _braced(text: str, start: int) -> tuple[str, int]:
    """The contents of the brace group opening at `start`, and the index after it."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return text[start + 1 :], len(text)


def _command_argument(text: str, command: str) -> str:
    match = re.search(rf"\\{command}\s*(?:\[[^\]]*\])?\s*\{{", text)
    if match is None:
        return ""
    return _braced(text, match.end() - 1)[0]


def _clean(text: str) -> str:
    text = _DROP_COMMANDS.sub("", text)
    for _ in range(3):
        text = _FORMAT.sub(r"\1", text)
    text = text.replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def _table_rows(block: str) -> list[str]:
    match = re.search(r"\\begin\{tabular[x*]?\}", block)
    if match is None:
        return []
    # Skip the column specification.
    rest = block[match.end() :]
    if rest.lstrip().startswith("{"):
        offset = len(rest) - len(rest.lstrip())
        _, after = _braced(rest, offset)
        rest = rest[after:]
    body = rest.split("\\end{tabular", 1)[0]
    body = _RULES.sub("", body)
    rows = []
    for raw in re.split(r"\\\\(?:\[[^\]]*\])?", body):
        cells = [_clean(re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", c)) for c in raw.split("&")]
        cells = [c.strip("{} ") for c in cells]
        if any(cells):
            rows.append(CELL_SEPARATOR.join(cells))
    return rows


class _Builder:
    """Sentences, sections and tables, in reading order."""

    def __init__(self) -> None:
        self.sentences: list[Sentence] = []
        self.sections: list[Section] = [Section(id="sec0", title="")]
        self.tables: list[Table] = []

    def add(self, text: str, kind: SentenceKind = SentenceKind.BODY) -> str:
        sid = f"s{len(self.sentences)}"
        self.sentences.append(
            Sentence(id=sid, section_id=self.sections[-1].id, kind=kind, text=text)
        )
        return sid

    def float_block(self, match: re.Match[str]) -> str:
        block = match.group(2)
        caption = _clean(_command_argument(block, "caption"))
        if match.group(1).startswith("table"):
            label = f"Table {len(self.tables) + 1}"
            if caption:
                self.add(f"{label}: {caption}", SentenceKind.CAPTION)
            row_ids = [self.add(row, SentenceKind.TABLE) for row in _table_rows(block)]
            self.tables.append(
                Table(id=f"tab{len(self.tables) + 1}", label=label, row_ids=tuple(row_ids))
            )
        elif caption:
            self.add(f"Figure: {caption}", SentenceKind.CAPTION)
        return "\n\n"

    def paragraphs(self, text: str) -> None:
        """One section's text: floats first, then a sentence per paragraph."""
        text = re.sub(
            _ENV_BLOCK.format(names="table|figure"), self.float_block, text, flags=re.DOTALL
        )
        text = re.sub(r"\\maketitle|\\appendix", "", text)
        text = _THEOREM_ENV.sub(_theorem_head, text)
        text = _END_THEOREM.sub("\n\n", text)
        for paragraph in re.split(r"\n\s*\n", text):
            cleaned = _clean(paragraph)
            if cleaned and re.search(r"\w", cleaned):
                self.add(cleaned)


def _theorem_head(match: re.Match[str]) -> str:
    name = match.group(1).capitalize()
    note = f" {match.group(2)}" if match.group(2) else ""
    return f"\n\n{name}{note}. "


def parse_latex(main: Path, paper_id: str | None = None) -> ParsedPaper:
    source = read_source(main)
    if "\\begin{document}" not in source:
        raise LatexError(f"{main} has no \\begin{{document}}")
    preamble, body = source.split("\\begin{document}", 1)
    body = body.split("\\end{document}", 1)[0]
    title = _clean(_command_argument(preamble, "title") or _command_argument(body, "title"))
    body = re.split(r"\\bibliography\{|\\begin\{thebibliography\}", body)[0]
    built = _Builder()
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.DOTALL)
    if abstract is not None:
        for paragraph in re.split(r"\n\s*\n", abstract.group(1)):
            if _clean(paragraph):
                built.add(_clean(paragraph), SentenceKind.ABSTRACT)
        body = body[: abstract.start()] + body[abstract.end() :]
    position = 0
    for match in _SECTION.finditer(body):
        built.paragraphs(body[position : match.start()])
        title_text, after = _braced(body, match.end() - 1)
        built.sections.append(Section(id=f"sec{len(built.sections)}", title=_clean(title_text)))
        position = after
    built.paragraphs(body[position:])
    if not built.sentences:
        raise LatexError(f"{main} has no text")
    used = {s.section_id for s in built.sentences}
    return ParsedPaper(
        paper_id=paper_id or main.parent.name,
        parser=ParserInfo(name="latex", version=PARSER_VERSION),
        metadata=PaperMetadata(title=title or None),
        sections=tuple(s for s in built.sections if s.id in used),
        sentences=tuple(built.sentences),
        tables=tuple(built.tables),
    )
