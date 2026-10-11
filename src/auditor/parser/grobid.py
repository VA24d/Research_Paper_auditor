"""Client for a GROBID service (decision D12; `docker compose up -d` starts one locally)."""

from auditor.infra.http import HttpClient, HttpResponse

TEI_COORDINATES = ("s", "ref", "biblStruct", "head", "figure", "note", "formula")


class GrobidError(RuntimeError):
    """GROBID could not process a document."""


class GrobidClient:
    def __init__(self, http: HttpClient, base_url: str, *, timeout_s: float = 300.0) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def is_alive(self) -> bool:
        try:
            response = await self._http.get(
                f"{self._base_url}/api/isalive", use_cache=False, attempts=1
            )
        except Exception:
            return False
        return response.ok and response.text.strip() == "true"

    async def process_fulltext(self, pdf: bytes, filename: str) -> str:
        """Return the TEI XML for a PDF.

        Header and citation consolidation are off: they would make GROBID call Crossref
        itself, outside the shared HTTP client, and reference resolution belongs to the
        citation agent anyway.
        """
        response = await self._http.post(
            f"{self._base_url}/api/processFulltextDocument",
            data={
                "segmentSentences": "1",
                "includeRawCitations": "1",
                "consolidateHeader": "0",
                "consolidateCitations": "0",
                "teiCoordinates": list(TEI_COORDINATES),
            },
            files={"input": (filename, pdf, "application/pdf")},
            request_timeout_s=self._timeout_s,
        )
        return _tei_or_raise(response)


def _tei_or_raise(response: HttpResponse) -> str:
    if response.status_code == 200:
        return response.text
    if response.status_code == 204:
        raise GrobidError("GROBID extracted no content from the document")
    if response.status_code == 503:
        raise GrobidError("GROBID is busy (503) and did not recover after retries")
    detail = response.text.strip()[:500]
    raise GrobidError(f"GROBID returned HTTP {response.status_code}: {detail}")
