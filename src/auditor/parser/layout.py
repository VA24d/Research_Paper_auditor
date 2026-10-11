"""Page geometry, fonts and link annotations from the PDF itself, for the template checker."""

import re
from collections import Counter
from pathlib import Path

import pdfplumber

from auditor.schemas import FontUsage, Layout, Link, LinkSource, PageLayout

# Embedded fonts are named like "MCWOPG+NimbusRomNo9L-Medi"; the prefix is per-file noise.
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")


class LayoutError(RuntimeError):
    """The PDF could not be read for layout."""


def extract_layout(path: Path) -> tuple[Layout, list[Link]]:
    try:
        return _extract(path)
    except LayoutError:
        raise
    except Exception as exc:
        raise LayoutError(f"could not read layout from {path.name}: {exc}") from exc


def _extract(path: Path) -> tuple[Layout, list[Link]]:
    pages: list[PageLayout] = []
    fonts: Counter[tuple[str, float]] = Counter()
    links: list[Link] = []
    with pdfplumber.open(path) as pdf:
        if not pdf.pages:
            raise LayoutError(f"{path.name} has no pages")
        for page in pdf.pages:
            # Rotated text, such as arXiv's identifier stamp in the left margin, is not
            # part of the text block a venue's margin rules are about.
            chars = [
                c for c in page.chars if str(c.get("text", "")).strip() and c.get("upright", True)
            ]
            box: dict[str, float | None] = dict.fromkeys(("left", "top", "right", "bottom"))
            if chars:
                box = {
                    "left": round(min(float(c["x0"]) for c in chars), 2),
                    "top": round(min(float(c["top"]) for c in chars), 2),
                    "right": round(max(float(c["x1"]) for c in chars), 2),
                    "bottom": round(max(float(c["bottom"]) for c in chars), 2),
                }
            pages.append(
                PageLayout(
                    number=page.page_number,
                    width=round(float(page.width), 2),
                    height=round(float(page.height), 2),
                    text_left=box["left"],
                    text_top=box["top"],
                    text_right=box["right"],
                    text_bottom=box["bottom"],
                )
            )
            for char in chars:
                name = _SUBSET_PREFIX.sub("", str(char.get("fontname", "")))
                fonts[(name, round(float(char.get("size", 0.0)), 1))] += 1
            for link in page.hyperlinks:
                uri = str(link.get("uri") or "")
                if uri.startswith(("http://", "https://")):
                    links.append(Link(url=uri, page=page.page_number, source=LinkSource.ANNOTATION))
            # Release the page's cached objects; long papers otherwise keep every page alive.
            page.close()
    usage = tuple(
        FontUsage(name=name, size=size, chars=count)
        for (name, size), count in fonts.most_common()
        if size > 0
    )
    return Layout(pages=tuple(pages), fonts=usage), links
