"""Converts GROBID TEI XML into a ParsedPaper.

Sentences come from GROBID's `<s>` elements, in document order: abstract, body sections,
figure captions and footnotes, then back-matter sections such as appendices. Citation
markers are `<ref type="bibr">` elements inside sentences; adjacent markers such as
`[6,` + `3]` are merged into one mention.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field

from auditor.schemas import (
    CELL_SEPARATOR,
    BibEntry,
    Box,
    CitationMention,
    Layout,
    Link,
    LinkMethod,
    LinkSource,
    PageLayout,
    PaperMetadata,
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceKind,
    Table,
)

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

# New-style ids (2310.06825) and old-style ids (hep-th/9901001), without the version.
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})(?:v\d+)?", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>\"'()\[\]{}]+")
# GROBID sometimes keeps the year label in front of a title: "2022a. PromptChainer: ...",
# "b. AI chains: ...", or with the last author's surname too: "Le. 2022a. Finetuned ...".
_TITLE_YEAR_LABEL = re.compile(
    r"^(?:(?:[^\W\d_][\w'-]*\.\s+)?(?P<year>(?:19|20)\d{2})[a-z]?|[a-z])\.\s+"
)
_MARKER_GAP = re.compile(r"^[\s,;]*$")
# GROBID leaves citation markers in table cells as plain text ("ConvS2S [9]"). A whole-number
# bracket right after a name (capitalised, with a digit, or ending in a parenthesis as in
# "Petrov et al. (2006) [29]") is taken as a marker; the numeric linker then accepts it only
# if it agrees with the paper's numbering. "range [1, 5]" and "[0.1, 0.5]" do not match.
_TABLE_MARKER = re.compile(
    r"(?:\b[A-Z][\w+'.-]*|\b[\w+'-]*\d[\w+'-]*|\))\s?"
    r"(?P<marker>\[\d{1,3}(?:\s*[,\u2013-]\s*\d{1,3})*\])"
)


class TeiError(ValueError):
    """The TEI document is not usable."""


@dataclass
class _Ref:
    start: int
    end: int
    target: str | None


@dataclass
class _Builder:
    sections: list[Section] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    citations: list[CitationMention] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)

    def add_table(self, figure: ET.Element) -> None:
        """Each row becomes a TABLE sentence; empty cells are kept so columns line up.
        Citation markers in cells are kept, so a cited result in a table gets checked."""
        table = figure.find(f"{TEI}table")
        if table is None:
            return
        boxes = parse_coords(table.get("coords") or figure.get("coords"))
        row_ids = []
        for row in table.findall(f"{TEI}row"):
            texts: list[str] = []
            refs: list[_Ref] = []
            urls: list[str] = []
            for cell in row.findall(f"{TEI}cell"):
                offset = len(CELL_SEPARATOR.join(texts)) + (len(CELL_SEPARATOR) if texts else 0)
                cell_text, cell_refs, cell_urls = _sentence_content(cell)
                cell_text, cell_refs = _collapsed(cell_text, cell_refs)
                if not cell_refs:
                    cell_refs = [
                        _Ref(m.start("marker"), m.end("marker"), None)
                        for m in _TABLE_MARKER.finditer(cell_text)
                    ]
                texts.append(cell_text)
                refs += [_Ref(r.start + offset, r.end + offset, r.target) for r in cell_refs]
                urls += cell_urls
            if not any(texts):
                continue
            sentence_id = f"s{len(self.sentences)}"
            row_text = CELL_SEPARATOR.join(texts)
            self.sentences.append(
                Sentence(id=sentence_id, kind=SentenceKind.TABLE, text=row_text, boxes=boxes)
            )
            self._add_mentions(sentence_id, row_text, refs, urls, boxes)
            row_ids.append(sentence_id)
        if row_ids:
            label = _text(figure.find(f"{TEI}head")).rstrip(" :") or None
            self.tables.append(
                Table(id=f"t{len(self.tables)}", label=label, row_ids=tuple(row_ids))
            )

    def add_section(self, title: str, number: str | None) -> str:
        section_id = f"sec{len(self.sections)}"
        self.sections.append(Section(id=section_id, title=title, number=number))
        return section_id

    def add_sentence(self, element: ET.Element, kind: SentenceKind, section_id: str | None) -> None:
        text, refs, urls = _sentence_content(element)
        if not text:
            return
        sentence_id = f"s{len(self.sentences)}"
        boxes = parse_coords(element.get("coords"))
        self.sentences.append(
            Sentence(id=sentence_id, section_id=section_id, kind=kind, text=text, boxes=boxes)
        )
        self._add_mentions(sentence_id, text, refs, urls, boxes)

    def _add_mentions(
        self,
        sentence_id: str,
        text: str,
        refs: list[_Ref],
        urls: list[str],
        boxes: tuple[Box, ...],
    ) -> None:
        """Citation mentions and links of a sentence that was just added."""
        for ref in _merge_adjacent(refs, text):
            marker = text[ref.start : ref.end]
            targets = tuple(dict.fromkeys(t for t in (ref.target or "").split() if t))
            self.citations.append(
                CitationMention(
                    id=f"c{len(self.citations)}",
                    sentence_id=sentence_id,
                    start=ref.start,
                    end=ref.end,
                    marker=marker,
                    bib_ids=targets,
                    link_method=LinkMethod.PARSER if targets else LinkMethod.NONE,
                )
            )
        page = boxes[0].page if boxes else None
        for url in dict.fromkeys([*urls, *(m.group(0) for m in _URL.finditer(text))]):
            self.links.append(
                Link(
                    url=_clean_url(url),
                    page=page,
                    sentence_id=sentence_id,
                    source=LinkSource.TEXT,
                )
            )


def parse_coords(value: str | None) -> tuple[Box, ...]:
    """Parse GROBID's `page,x,y,w,h;...` coordinates, skipping malformed parts."""
    if not value:
        return ()
    boxes = []
    for part in value.split(";"):
        fields = part.split(",")
        if len(fields) != 5:
            continue
        try:
            page = int(fields[0])
            x, y, width, height = (float(f) for f in fields[1:])
        except ValueError:
            continue
        if page >= 1 and width >= 0 and height >= 0:
            boxes.append(Box(page=page, x=x, y=y, width=width, height=height))
    return tuple(boxes)


def normalise_arxiv_id(value: str) -> str | None:
    """`arXiv:2310.06825v1 [cs.CL]` -> `2310.06825`."""
    match = _ARXIV_ID.search(value)
    return match.group(1) if match else None


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:")


def _local(tag: str) -> str:
    return tag.removeprefix(TEI)


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _sentence_content(element: ET.Element) -> tuple[str, list[_Ref], list[str]]:
    """Text of a sentence with the character spans of its citation markers."""
    pieces: list[str] = []
    length = 0
    refs: list[_Ref] = []
    urls: list[str] = []

    def add(piece: str | None) -> None:
        nonlocal length
        if piece:
            # Same-length replacement keeps every recorded offset valid.
            piece = piece.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            pieces.append(piece)
            length += len(piece)

    add(element.text)
    for child in element:
        start = length
        add("".join(child.itertext()))
        if _local(child.tag) == "ref":
            ref_type = child.get("type")
            target = child.get("target")
            if ref_type == "bibr":
                refs.append(_Ref(start, length, target.replace("#", "") if target else None))
            elif ref_type == "url" and target:
                urls.append(target)
        add(child.tail)

    raw = "".join(pieces)
    stripped = raw.strip()
    if not stripped:
        return "", [], urls
    offset = len(raw) - len(raw.lstrip())
    kept = []
    for ref in refs:
        start, end = ref.start - offset, ref.end - offset
        while start < end and stripped[start].isspace():
            start += 1
        while end > start and stripped[end - 1].isspace():
            end -= 1
        if start < end and start >= 0 and end <= len(stripped):
            kept.append(_Ref(start, end, ref.target))
    return stripped, kept, urls


def _collapsed(text: str, refs: list[_Ref]) -> tuple[str, list[_Ref]]:
    """The text with each run of whitespace made one space, as _text does, and the
    marker spans moved to match."""
    out: list[str] = []
    position: list[int] = []
    for char in text:
        position.append(len(out))
        if not char.isspace():
            out.append(char)
        elif out and out[-1] != " ":
            out.append(" ")
    position.append(len(out))
    # The input is already stripped, so no space is left at either end.
    return "".join(out), [_Ref(position[r.start], position[r.end], r.target) for r in refs]


def _merge_adjacent(refs: list[_Ref], text: str) -> list[_Ref]:
    """Join markers split by GROBID, e.g. `[6,` and `3]`, when both are linked or both not."""
    merged: list[_Ref] = []
    for ref in sorted(refs, key=lambda r: r.start):
        if merged:
            last = merged[-1]
            same_status = (last.target is None) == (ref.target is None)
            if same_status and _MARKER_GAP.match(text[last.end : ref.start]):
                targets = " ".join(t for t in (last.target, ref.target) if t)
                merged[-1] = _Ref(last.start, ref.end, targets or None)
                continue
        merged.append(ref)
    return merged


def _sentences_in(container: ET.Element) -> list[ET.Element]:
    """Sentences of every paragraph in the container. A paragraph without `<s>` counts as
    one sentence, and a container without paragraphs counts as one sentence itself."""
    paragraphs = list(container.iter(f"{TEI}p"))
    if not paragraphs:
        return [container]
    return [s for paragraph in paragraphs for s in _sentences_in_paragraph(paragraph)]


def _collect_div(
    builder: _Builder, div: ET.Element, kind: SentenceKind, fallback_title: str | None
) -> None:
    head = div.find(f"{TEI}head")
    section_id = None
    # A back-matter wrapper without its own heading only becomes a section when it holds
    # text directly; otherwise its nested divs carry the real headings.
    holds_text = div.find(f"{TEI}p") is not None
    if head is not None or (fallback_title and holds_text):
        title = _text(head) or (fallback_title or "")
        section_id = builder.add_section(title, head.get("n") if head is not None else None)
    for child in div:
        _collect_child(builder, child, kind, section_id)


def _collect_child(
    builder: _Builder, child: ET.Element, kind: SentenceKind, section_id: str | None
) -> None:
    """A paragraph, nested division, figure or footnote, wherever it appears. GROBID puts
    appendix tables and figures inside back-matter divisions, not at the top of the body."""
    tag = _local(child.tag)
    if tag == "p":
        for sentence in _sentences_in_paragraph(child):
            builder.add_sentence(sentence, kind, section_id)
    elif tag == "div":
        _collect_div(builder, child, kind, None)
    elif tag == "figure":
        description = child.find(f"{TEI}figDesc")
        if description is not None:
            for sentence in _sentences_in(description):
                builder.add_sentence(sentence, SentenceKind.CAPTION, section_id)
        if child.get("type") == "table":
            builder.add_table(child)
    elif tag == "note" and child.get("place") == "foot":
        for sentence in _sentences_in(child):
            builder.add_sentence(sentence, SentenceKind.FOOTNOTE, section_id)


def _sentences_in_paragraph(paragraph: ET.Element) -> list[ET.Element]:
    sentences = paragraph.findall(f"{TEI}s")
    return sentences if sentences else [paragraph]


def _person(element: ET.Element) -> str:
    names = [_text(n) for n in element.findall(f"{TEI}forename")]
    names.append(_text(element.find(f"{TEI}surname")))
    return " ".join(n for n in names if n)


def _authors(parent: ET.Element | None) -> tuple[str, ...]:
    if parent is None:
        return ()
    people = (_person(p) for p in parent.findall(f"{TEI}author/{TEI}persName"))
    return tuple(name for name in people if name)


def _year(element: ET.Element | None) -> int | None:
    if element is None:
        return None
    for date in element.iter(f"{TEI}date"):
        when = date.get("when") or ""
        if len(when) >= 4 and when[:4].isdigit():
            return int(when[:4])
    return None


def _idno(element: ET.Element, kind: str) -> str | None:
    for idno in element.iter(f"{TEI}idno"):
        if (idno.get("type") or "").lower() == kind.lower():
            value = _text(idno)
            if kind.lower() == "arxiv":
                return normalise_arxiv_id(value)
            return value or None
    return None


def _bib_entry(element: ET.Element) -> BibEntry:
    bib_id = element.get(XML_ID)
    if not bib_id:
        raise TeiError("biblStruct without xml:id")
    analytic = element.find(f"{TEI}analytic")
    monogr = element.find(f"{TEI}monogr")
    title = _text(analytic.find(f"{TEI}title")) if analytic is not None else ""
    venue = ""
    if monogr is not None:
        monogr_title = _text(monogr.find(f"{TEI}title"))
        if title:
            venue = monogr_title
        else:
            title = monogr_title
        if not venue:
            venue = _text(monogr.find(f"{TEI}meeting"))
    pointer = element.find(f".//{TEI}ptr")
    raw = None
    for note in element.findall(f"{TEI}note"):
        if note.get("type") == "raw_reference":
            raw = _text(note) or None
    year = _year(monogr) or _year(element)
    label = _TITLE_YEAR_LABEL.match(title)
    if label is not None:
        title = title[label.end() :]
        if year is None and label["year"]:
            year = int(label["year"])
    return BibEntry(
        id=bib_id,
        raw=raw,
        title=title or None,
        authors=_authors(analytic) or _authors(monogr),
        year=year,
        venue=venue or None,
        doi=_idno(element, "DOI"),
        arxiv_id=_idno(element, "arXiv"),
        url=pointer.get("target") if pointer is not None else None,
        boxes=parse_coords(element.get("coords")),
    )


def _metadata(header: ET.Element | None) -> PaperMetadata:
    if header is None:
        return PaperMetadata()
    title = _text(header.find(f".//{TEI}titleStmt/{TEI}title"))
    source = header.find(f".//{TEI}sourceDesc/{TEI}biblStruct")
    return PaperMetadata(
        title=title or None,
        authors=_authors(source.find(f"{TEI}analytic")) if source is not None else (),
        year=_year(header.find(f".//{TEI}publicationStmt")),
        doi=_idno(source, "DOI") if source is not None else None,
        arxiv_id=_idno(source, "arXiv") if source is not None else None,
    )


def _grobid_version(header: ET.Element | None) -> str:
    if header is not None:
        for application in header.iter(f"{TEI}application"):
            if application.get("ident") == "GROBID" and application.get("version"):
                return application.get("version") or "unknown"
    return "unknown"


def page_sizes(root: ET.Element) -> Layout | None:
    """Page sizes from `<facsimile>`, used when no PDF layout is available."""
    surfaces = root.findall(f"{TEI}facsimile/{TEI}surface")
    pages = []
    for number, surface in enumerate(surfaces, start=1):
        try:
            width = float(surface.get("lrx", "")) - float(surface.get("ulx", "0"))
            height = float(surface.get("lry", "")) - float(surface.get("uly", "0"))
        except ValueError:
            return None
        if width <= 0 or height <= 0:
            return None
        pages.append(PageLayout(number=number, width=width, height=height))
    return Layout(pages=tuple(pages)) if pages else None


def _repair_url(link: Link, sentence_text: dict[str, str], annotations: list[Link]) -> Link:
    """Use the PDF's own link when GROBID cut a URL at a line break.

    "https://github.com/ google-research/vit" in the text becomes the annotation's
    "https://github.com/google-research/vit" if that full URL appears in the sentence once
    whitespace is removed.
    """
    if link.sentence_id is None:
        return link
    compact = re.sub(r"\s+", "", sentence_text[link.sentence_id])
    for annotation in annotations:
        longer = annotation.url != link.url and annotation.url.startswith(link.url)
        same_page = annotation.page is None or link.page is None or annotation.page == link.page
        if longer and same_page and annotation.url in compact:
            return link.model_copy(update={"url": annotation.url})
    return link


def tei_to_paper(
    tei: str | bytes,
    *,
    paper_id: str,
    pdf_sha256: str | None = None,
    layout: Layout | None = None,
    extra_links: Iterable[Link] = (),
) -> ParsedPaper:
    try:
        root = ET.fromstring(tei)
    except ET.ParseError as exc:
        raise TeiError(f"TEI is not well-formed XML: {exc}") from exc
    if _local(root.tag) != "TEI":
        raise TeiError(f"expected a TEI document, got <{_local(root.tag)}>")

    header = root.find(f"{TEI}teiHeader")
    builder = _Builder()

    abstract = root.find(f".//{TEI}profileDesc/{TEI}abstract")
    if abstract is not None:
        for sentence in _sentences_in(abstract):
            builder.add_sentence(sentence, SentenceKind.ABSTRACT, None)

    body = root.find(f"{TEI}text/{TEI}body")
    if body is not None:
        for child in body:
            _collect_child(builder, child, SentenceKind.BODY, None)

    back = root.find(f"{TEI}text/{TEI}back")
    bibliography: list[BibEntry] = []
    if back is not None:
        for div in back.findall(f"{TEI}div"):
            div_type = div.get("type") or ""
            if div_type == "references":
                continue
            fallback = div_type.replace("_", " ").title() or None
            _collect_div(builder, div, SentenceKind.BODY, fallback)
        for element in back.iter(f"{TEI}biblStruct"):
            bibliography.append(_bib_entry(element))

    if not builder.sentences:
        raise TeiError("GROBID found no text in the document")

    known = {entry.id for entry in bibliography}
    citations = [
        mention
        if set(mention.bib_ids) <= known
        else mention.model_copy(
            update={
                "bib_ids": tuple(b for b in mention.bib_ids if b in known),
                "link_method": LinkMethod.PARSER
                if set(mention.bib_ids) & known
                else LinkMethod.NONE,
            }
        )
        for mention in builder.citations
    ]

    annotations = list(extra_links)
    sentence_text = {sentence.id: sentence.text for sentence in builder.sentences}
    links: list[Link] = []
    for link in builder.links:
        repaired = _repair_url(link, sentence_text, annotations)
        if repaired not in links:
            links.append(repaired)
    # Annotations add links the text does not show, e.g. a hyperlinked word.
    in_text = {(link.url, link.page) for link in links}
    for annotation in annotations:
        if (annotation.url, annotation.page) not in in_text:
            links.append(annotation)
            in_text.add((annotation.url, annotation.page))

    return ParsedPaper(
        paper_id=paper_id,
        parser=ParserInfo(name="grobid", version=_grobid_version(header), pdf_sha256=pdf_sha256),
        metadata=_metadata(header),
        sections=tuple(builder.sections),
        sentences=tuple(builder.sentences),
        bibliography=tuple(bibliography),
        citations=tuple(citations),
        tables=tuple(builder.tables),
        links=tuple(links),
        layout=layout if layout is not None else page_sizes(root),
    )
