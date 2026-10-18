from pathlib import Path

import pytest

from auditor.parser.latex import LatexError, parse_latex
from auditor.schemas import CELL_SEPARATOR, SentenceKind

MAIN = r"""
\documentclass{article}
\title{Sums of \emph{Integers}}
\begin{document}
\maketitle
\begin{abstract}
We study $S(n)$. % a comment that must go
\end{abstract}
\section{Main result}
\input{body}
\begin{table}[t]
\centering
\caption{Accuracy of the methods.}
\begin{tabular}{lcc}
\toprule
Method & A & B \\
\midrule
Base & 70.1 & \textbf{80.2} \\
Ours & 71.5 & 79.0 \\
\bottomrule
\end{tabular}
\end{table}
\section{Proofs}
\begin{lemma}[Gauss]\label{lem:sum}
For every $n \geq 1$ we have $S(n) = \frac{n(n+1)}{2}$.
\end{lemma}
\begin{proof}
By induction on $n$.
\end{proof}
\bibliography{refs}
\end{document}
"""

BODY = r"""Let $S(n) = \sum_{i=1}^{n} i$ denote the sum.

It satisfies \[ S(n+1) = S(n) + n + 1 . \]
"""


def write(tmp_path: Path) -> Path:
    (tmp_path / "body.tex").write_text(BODY)
    main = tmp_path / "main.tex"
    main.write_text(MAIN)
    return main


def test_the_source_becomes_sections_paragraphs_and_tables_with_formulas_kept(
    tmp_path: Path,
) -> None:
    paper = parse_latex(write(tmp_path), paper_id="p")
    assert paper.metadata.title == "Sums of Integers"
    texts = [s.text for s in paper.sentences]
    assert texts[0] == "We study $S(n)$."  # the abstract, without its comment
    assert paper.sentences[0].kind is SentenceKind.ABSTRACT
    assert r"Let $S(n) = \sum_{i=1}^{n} i$ denote the sum." in texts  # from the \input
    assert any(r"\[ S(n+1) = S(n) + n + 1 . \]" in t for t in texts)
    assert any(t.startswith("Lemma [Gauss]. For every $n \\geq 1$") for t in texts)
    assert any(t.startswith("Proof. By induction") for t in texts)
    assert [s.title for s in paper.sections if s.title] == ["Main result", "Proofs"]
    (table,) = paper.tables
    rows = [paper.sentence(i).text for i in table.row_ids]
    assert rows == [
        CELL_SEPARATOR.join(["Method", "A", "B"]),
        CELL_SEPARATOR.join(["Base", "70.1", "80.2"]),
        CELL_SEPARATOR.join(["Ours", "71.5", "79.0"]),
    ]
    assert "Table 1: Accuracy of the methods." in texts


def test_an_input_outside_the_source_folder_is_not_read(tmp_path: Path) -> None:
    source = tmp_path / "paper"
    source.mkdir()
    (tmp_path / "secret.tex").write_text("SECRET TEXT")
    (source / "main.tex").write_text(
        "\\begin{document}\nIntro text here.\n\n\\input{../secret}\n\\end{document}\n"
    )
    paper = parse_latex(source / "main.tex")
    assert all("SECRET" not in s.text for s in paper.sentences)
    assert paper.paper_id == "paper"


def test_a_file_without_a_document_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "macros.tex"
    path.write_text("\\newcommand{\\R}{\\mathbb{R}}\n")
    with pytest.raises(LatexError):
        parse_latex(path)


def _tar(files: dict[str, bytes]) -> bytes:
    import gzip
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(buffer.getvalue())


def test_an_eprint_is_unpacked_and_its_main_file_found(tmp_path: Path) -> None:
    from auditor.parser.latex import unpack_eprint

    content = _tar(
        {
            "macros.tex": b"\\newcommand{\\R}{\\mathbb{R}}",
            "paper.tex": MAIN.encode(),
            "body.tex": BODY.encode(),
        }
    )
    main = unpack_eprint(content, tmp_path / "src")
    assert main.name == "paper.tex"
    assert parse_latex(main).tables


def test_an_eprint_member_outside_its_folder_is_refused(tmp_path: Path) -> None:
    import tarfile

    from auditor.parser.latex import unpack_eprint

    content = _tar({"../evil.tex": b"x", "paper.tex": MAIN.encode()})
    with pytest.raises(tarfile.OutsideDestinationError):
        unpack_eprint(content, tmp_path / "src")
    assert not (tmp_path / "evil.tex").exists()


def test_a_single_gzipped_file_is_an_eprint_too(tmp_path: Path) -> None:
    import gzip

    from auditor.parser.latex import unpack_eprint

    simple = b"\\documentclass{article}\\begin{document}\nHello $x$.\n\\end{document}"
    main = unpack_eprint(gzip.compress(simple), tmp_path / "src")
    assert parse_latex(main).sentences[0].text == "Hello $x$."
