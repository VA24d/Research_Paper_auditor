"""The auditor's web UI (Streamlit): run a live audit on one paper and read its report.

Start it with `uv run auditor ui` (needs `uv sync --extra ui`). An audit runs
`auditor audit` in a child process, so it behaves exactly as on the command line; see
`auditor.ui.audit` for the limits that protect the shared credits.
"""

import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx
import streamlit as st

from auditor.config import Settings
from auditor.reporting import AuditReport
from auditor.reporting.report import NotChecked, ReportItem, same_text, short_repository
from auditor.schemas import EvidenceKind, SourceKind
from auditor.ui.audit import (
    GEMINI_KEY,
    MODELS,
    OPENAI_COMPATIBLE,
    AuditRequest,
    command,
    default_project,
    demo_papers,
    environment,
    model_choices,
    new_run_id,
    progress,
    report_of,
    reports,
    save_upload,
    spent,
    tail,
)

ROOT = Path.cwd()
SETTINGS = Settings()
# AUDITOR_DEMO=1: the public read-only demo (reports under AUDITOR_RUNS_DIR, no audits).
DEMO = os.environ.get("AUDITOR_DEMO") == "1"
# The demo's papers (bundled ParsedPaper JSON with their repositories), and where audits
# run from the demo go: a temporary folder, so one visitor's audit is not listed for others.
DEMO_PAPERS = ROOT / "demo" / "papers.jsonl"
DEMO_RUNS = Path(tempfile.gettempdir()) / "auditor-demo-runs"
UI_DIR = SETTINGS.data_dir / "ui"
LEDGER = UI_DIR / "ledger.jsonl"
DEMO_DATASET = ROOT / "configs" / "experiments" / "set_c.jsonl"
VENUES = sorted((ROOT / "configs" / "venues").glob("*.json"))
# Rough cost of one audit, from the set C runs (Gemini 3.8 Flash, citation agent left out).
# Flash-Lite: one test audit of a set C paper, 33 calls, $0.03.
COST_PER_AUDIT = {"gemini-3.8-flash": 0.08, "gemini-3.5-flash-lite": 0.03, "gemini-2.5-flash": 0.04}

_MD_SPECIAL = str.maketrans({c: f"\\{c}" for c in "\\`*_{}[]<>()#+-.!|~$"})


def md(text: str) -> str:
    """Text shown as is in Markdown (paper text can contain *, _, $ or #)."""
    return " ".join(text.split()).translate(_MD_SPECIAL)


def reachable(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code < 500
    except httpx.HTTPError:
        return False


# --- report ---------------------------------------------------------------------------


def show_item(number: int, item: ReportItem, *, reason: bool) -> None:
    tags = [
        t
        for t in (
            item.severity and f"{item.severity} severity",
            item.uncertain and "marked uncertain",
        )
        if t
    ]
    with st.expander(f"{number}. {item.title}: {item.place.description}"):
        st.markdown(f"**About:** {md(item.about)}" + (f"  ·  _{', '.join(tags)}_" if tags else ""))
        if item.place.text:
            st.markdown("**What the paper says**")
            st.markdown(f"> {md(item.place.text)}")
        st.markdown(f"**What the check found:** {md(item.summary)}")
        if item.explanation and item.explanation != item.summary:
            st.caption(item.explanation)
        for evidence in item.evidence:
            # A quote of the whole flagged sentence is already shown above.
            if evidence.source_kind is SourceKind.SUBMISSION and same_text(
                evidence.text, item.place.text or ""
            ):
                continue
            where = evidence.location or ""
            source = evidence.source_title or evidence.source
            if evidence.source_kind is SourceKind.REPOSITORY:
                source = short_repository(evidence.source)
            st.markdown(f"**Evidence** ({evidence.source_kind.value}) `{source}` {md(where)}")
            if evidence.kind is EvidenceKind.CODE:
                st.code(evidence.text, language="python")
            else:
                st.markdown(f"> {md(evidence.text)}")
        if reason:
            st.info(f"Why it is here: {item.verifier_reason}")
        if item.also_reported_as:
            st.caption("Also reported as " + ", ".join(item.also_reported_as))


def show_not_checked(items: tuple[NotChecked, ...], key: str) -> None:
    st.dataframe(
        [
            {
                "Check": i.kind,
                "Where": i.place.description,
                "About": i.about,
                "Outcome": i.outcome,
                "Reason": " ".join(i.reason.split()),
            }
            for i in items
        ],
        hide_index=True,
        key=f"{key}-not-checked",
        width="stretch",
    )


def show_report(path: Path, key: str) -> None:
    """The report at `path`; `key` keeps its widgets apart from another report on the page."""
    report = AuditReport.model_validate_json(path.read_text(encoding="utf-8"))
    st.subheader(report.title or report.paper_id)
    st.caption(f"Paper {report.paper_id} ({report.variant}) · run {report.run_id}")
    st.info(report.notice)
    outcomes = report.task_outcomes
    cols = st.columns(5)
    cols[0].metric("Verified flags", len(report.verified))
    cols[1].metric("For human review", len(report.review))
    cols[2].metric("Not checked", len(report.not_checked))
    cols[3].metric("Set aside", len(report.dropped))
    cols[4].metric("Checks completed", f"{outcomes.get('checked', 0)}/{report.tasks}")
    tabs = st.tabs(
        [
            f"Verified flags ({len(report.verified)})",
            f"Human review ({len(report.review)})",
            f"Not checked ({len(report.not_checked)})",
            f"Set aside ({len(report.dropped)})",
        ]
    )
    with tabs[0]:
        if not report.verified:
            st.write("No finding was verified.")
        for n, item in enumerate(report.verified, start=1):
            show_item(n, item, reason=False)
    with tabs[1]:
        if not report.review:
            st.write("Nothing needs review.")
        for n, item in enumerate(report.review, start=1):
            show_item(n, item, reason=True)
    with tabs[2]:
        if report.not_checked:
            st.caption(
                "These checks could not be completed, so a missing flag here says nothing "
                "about the paper."
            )
            show_not_checked(report.not_checked, key)
        else:
            st.write("Every check was completed.")
    with tabs[3]:
        if not report.dropped:
            st.write("The verifier set nothing aside.")
        for n, item in enumerate(report.dropped, start=1):
            show_item(n, item, reason=True)
    markdown = path.with_suffix(".md")
    buttons = st.columns(2)
    if markdown.is_file():
        buttons[0].download_button(
            "Download report.md",
            markdown.read_bytes(),
            "report.md",
            "text/markdown",
            key=f"{key}-md",
        )
    buttons[1].download_button(
        "Download report.json",
        path.read_bytes(),
        "report.json",
        "application/json",
        key=f"{key}-json",
    )


# --- running an audit -----------------------------------------------------------------


def run_audit(request: AuditRequest) -> Path | None:
    """Runs the audit with live progress; returns its report, or None when it failed."""
    run_id = new_run_id()
    run_dir = (request.runs_dir or SETTINGS.runs_dir) / run_id
    process = subprocess.Popen(
        command(request, run_id),
        cwd=ROOT,
        env=environment(request),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    feed: queue.Queue[str] = queue.Queue()

    def read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            feed.put(line)

    threading.Thread(target=read, daemon=True).start()
    started = time.monotonic()
    with st.status(f"Auditing ({run_id})...", expanded=True) as status:
        numbers = st.empty()
        log = st.empty()
        while True:
            while not feed.empty():
                lines.append(feed.get())
            now = progress(run_dir, SETTINGS)
            numbers.markdown(
                f"**{int(time.monotonic() - started)} s** · {now.calls} model calls · "
                f"${now.cost_usd:.3f} so far"
            )
            log.code(tail(lines) or "starting...", language=None)
            if process.poll() is not None and feed.empty():
                break
            time.sleep(1.0)
        report = report_of(run_dir)
        if process.returncode == 0 and report is not None:
            status.update(
                label=f"Done in {int(time.monotonic() - started)} s, ${now.cost_usd:.3f}",
                state="complete",
                expanded=False,
            )
        elif report is not None:
            status.update(
                label="Finished with incomplete checks (a limit was reached)", state="error"
            )
        else:
            status.update(label="The audit failed; the log above says why", state="error")
    return report


# --- page -----------------------------------------------------------------------------

st.set_page_config(page_title="Research Paper Auditor", page_icon="🔬", layout="wide")

col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("Research Paper Auditor")
    st.caption(
        "Evidence-gated factual auditing of scientific submissions: citations, code repositories, "
        "hyperparameters and mathematical derivations. Advisory only — a human reviewer decides."
    )
with col_badge:
    st.markdown(
        """
        <div style="background-color: #f0edf6; border: 1px solid #7030A0; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="font-weight: 800; font-size: 0.82rem; color: #47247B;">HOOD RESEARCH DEPARTMENT</div>
            <div style="font-weight: 700; font-size: 0.72rem; color: #7030A0;">SERC & CVIT LABS</div>
            <div style="font-size: 0.68rem; color: #555;">IIIT Hyderabad</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Metric overview banner
m1, m2, m3, m4 = st.columns(4)
m1.metric("Macro-F1", "0.636", "13× over baselines")
m2.metric("Clean Control Precision", "100%", "0 False Alarms / 72")
m3.metric("Evidence Grounding", "100%", "Dual Verbatim Quotes")
m4.metric("Avg Audit Cost", "₹7.4 ($0.08)", "7× cheaper than ReAct")
st.divider()



def show_saved_reports(key: str) -> None:
    found_reports = reports(SETTINGS.runs_dir)
    if not found_reports:
        st.write("No audit reports yet.")
        return
    names = [f"{p.parent.parent.name} / {p.parent.name}" for p in found_reports]
    pick = st.selectbox("Report", range(len(found_reports)), format_func=lambda i: names[i])
    show_report(found_reports[pick], key=key)


def run_own_model_audit() -> None:
    """The demo's live audit: the visitor's own Gemini key or OpenAI-compatible server."""
    st.write(
        "Bring your own model. The key stays in this session: it goes to the audit process "
        "and is not saved or logged. A set C paper takes a few minutes and about 13 model "
        "calls."
    )
    entries = demo_papers(DEMO_PAPERS)
    labels = [
        f"{Path(e.paper).stem} "
        f"({'injected copy' if Path(e.paper).parent.name == 'injected' else 'original'})"
        for e in entries
    ]
    paper: Path | None = None
    repository = commit = ""
    if entries:
        choice = st.selectbox("Paper", range(len(entries)), format_func=lambda i: labels[i])
        paper = Path(entries[choice].paper)
        repository = entries[choice].repository or ""
        commit = entries[choice].repository_commit or ""
        st.caption(f"Code: {repository or 'none'}. Injected copies carry planted errors.")
    upload = st.file_uploader("Or upload a parsed paper (ParsedPaper JSON)", type=["json"])
    if upload is not None:
        paper = save_upload(upload.name, upload.getvalue(), DEMO_RUNS / "uploads")
        repository = st.text_input("Its code repository (optional)", "")
        commit = ""

    source = st.radio(
        "Model",
        [GEMINI_KEY, OPENAI_COMPATIBLE],
        format_func=lambda s: {
            GEMINI_KEY: "Gemini API key",
            OPENAI_COMPATIBLE: "OpenAI-compatible server (OpenAI, OpenRouter, Ollama, vLLM, ...)",
        }[s],
        horizontal=True,
    )
    base_url: str | None = None
    if source == GEMINI_KEY:
        model = st.selectbox("Gemini model", MODELS)
        api_key = st.text_input(
            "Gemini API key", type="password", help="From Google AI Studio (aistudio.google.com)."
        )
        ready = bool(api_key)
    else:
        base_url = st.text_input("Base URL", "http://localhost:11434/v1")
        model = st.text_input("Model name", "qwen3:8b")
        api_key = st.text_input("API key (if the server needs one)", type="password")
        st.caption(
            "On the hosted demo, localhost is the server, not your computer. For a model on "
            "your machine, run the app locally (see demo/README.md in the repository) or "
            "expose your server with a tunnel. Small local models often fail the "
            "structured answers the checkers need."
        )
        ready = bool(base_url and model)
    max_cost = st.number_input(
        "Limit per audit ($, Gemini models only)", 0.05, 5.0, 0.50, 0.05, key="demo_cost"
    )
    if st.button("Run audit", type="primary", disabled=paper is None or not ready):
        assert paper is not None
        request = AuditRequest(
            paper=paper,
            repository=repository.strip() or None,
            commit=commit.strip() or None,
            model=model.strip(),
            source=source,
            api_key=api_key.strip() or None,
            base_url=base_url.strip() if base_url else None,
            max_cost_usd=float(max_cost),
            runs_dir=DEMO_RUNS,
        )
        st.session_state["demo_report"] = run_audit(request)
    last = st.session_state.get("demo_report")
    if last is not None and Path(last).is_file():
        show_report(Path(last), key="demo-live")


if DEMO:
    # Read-only demo for public hosting: no model settings, no project name, no new audits.
    with st.sidebar:
        st.header("Demo")
        st.write(
            "Saved reports the auditor wrote on papers from our paper-and-code test set. "
            "To run a new audit, bring your own Gemini key or an OpenAI-compatible server "
            "in the second tab."
        )
        st.write("Papers named *injected* are copies with errors we planted.")
        st.markdown("[Code](https://github.com/VA24d/Research_Paper_auditor)")
    saved, live = st.tabs(["Saved reports", "Run an audit with your own model"])
    with saved:
        show_saved_reports(key="demo")
    with live:
        run_own_model_audit()
    st.stop()

with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle(
        "Use the model",
        value=True,
        help="Off: free, but the checks that need a model end as 'not checked'.",
    )
    model = st.selectbox("Model", model_choices(SETTINGS), disabled=not use_llm)
    project = st.text_input("Vertex project", default_project(SETTINGS), disabled=not use_llm)
    skip_citation = st.checkbox(
        "Leave out the citation agent",
        value=True,
        help="The citation agent looks up every reference online: slower and costlier.",
    )
    max_cost = st.number_input("Limit per audit ($)", 0.05, 5.0, 0.50, 0.05, disabled=not use_llm)
    budget = st.number_input(
        "Limit for all UI audits ($)", 0.5, 50.0, 5.0, 0.5, disabled=not use_llm
    )
    used = spent(LEDGER)
    st.progress(min(used / budget, 1.0), text=f"UI audits so far: ${used:.2f} of ${budget:.2f}")
    # The method check ranks code with nomic embeddings. They are cached for the set C
    # papers; another paper needs an embedding server there, or falls back to BM25.
    embed_url = "http://localhost:1234"
    st.caption(
        "Code retrieval: nomic embeddings (cached for the set C papers; other papers use "
        "an embedding server at localhost:1234 if one runs, else BM25)"
    )

new, past = st.tabs(["New audit", "Past audits"])

with new:
    source = st.radio("Paper", ["Set C demo paper", "Upload a paper"], horizontal=True)
    paper: Path | None = None
    repository = commit = ""
    if source == "Set C demo paper":
        entries = demo_papers(DEMO_DATASET)
        if not entries:
            st.warning("No set C papers here; unpack the data archives (data-archive/README.md).")
        else:
            labels = [
                f"{Path(e.paper).stem} "
                f"({'injected copy' if Path(e.paper).parent.name == 'injected' else 'original'})"
                for e in entries
            ]
            choice = st.selectbox(
                "Demo paper", range(len(entries)), format_func=lambda i: labels[i]
            )
            entry = entries[choice]
            paper = Path(entry.paper)
            repository, commit = entry.repository or "", entry.repository_commit or ""
            st.caption("Injected copies carry planted errors from set C.")
    else:
        upload = st.file_uploader("ParsedPaper JSON, or a PDF (needs GROBID)", type=["json", "pdf"])
        if upload is not None:
            paper = save_upload(upload.name, upload.getvalue(), UI_DIR / "uploads")
            if paper.suffix == ".pdf" and not reachable(SETTINGS.grobid_url + "/api/isalive"):
                st.error(
                    "GROBID is not running. Start it with `docker compose up -d` (about 20 s)."
                )
                paper = None
    cols = st.columns([3, 2, 2])
    repository = cols[0].text_input("Code repository (optional)", repository)
    commit = cols[1].text_input("Commit (optional)", commit)
    venue_names = ["(none)"] + [v.stem for v in VENUES]
    venue_name = cols[2].selectbox("Venue rules", venue_names)
    venue = next((v for v in VENUES if v.stem == venue_name), None)

    if use_llm:
        estimate = COST_PER_AUDIT.get(model, 0.08)
        st.caption(f"About ${estimate:.2f} and a few minutes per paper; stops at ${max_cost:.2f}.")
    over = use_llm and used >= budget
    if over:
        st.error("The UI's budget is used up; raise the limit in the sidebar to continue.")
    if st.button("Run audit", type="primary", disabled=paper is None or over):
        assert paper is not None
        request = AuditRequest(
            paper=paper,
            repository=repository.strip() or None,
            commit=commit.strip() or None,
            venue=venue,
            use_llm=use_llm,
            model=model,
            project=project.strip() or default_project(SETTINGS),
            skip_citation=skip_citation,
            max_cost_usd=float(max_cost),
            ledger=LEDGER,
            ledger_limit_usd=float(budget),
            embed_url=embed_url,
        )
        st.session_state["last_report"] = run_audit(request)
    last = st.session_state.get("last_report")
    if last is not None and Path(last).is_file():
        show_report(Path(last), key="new")

with past:
    show_saved_reports(key="past")
