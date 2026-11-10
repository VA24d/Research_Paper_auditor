"""A review sheet for the D18 second review: one local HTML page per review file.

The page shows each sampled label with the paper's sentence, the label's rationale and its
evidence, and lets the reviewer mark agree or disagree with a note. "Download" saves the
filled file in the same format, for `scripts/set_c.py apply-review` (or the set B
equivalent). Progress is kept in the browser while the page is open.

    uv run python scripts/review_sheet.py datasets/set_c/review.jsonl -o data/p4/set_c-review.html
"""

# The page template below is HTML and JavaScript, whose lines are left long.
# ruff: noqa: E501

import argparse
import json
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label review</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e3e2dd;
        --ok:#1f7a4d; --bad:#b3261e; --code:#f3f2ee; }
@media (prefers-color-scheme: dark) { :root { --bg:#1b1b1a; --fg:#ecebe6; --muted:#a3a29c;
        --card:#242422; --line:#3a3936; --ok:#5cc28f; --bad:#f2827a; --code:#2d2c29; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif;
       margin:0 auto; max-width:900px; padding:16px; }
header { position:sticky; top:0; background:var(--bg); padding:8px 0 12px;
         border-bottom:1px solid var(--line); z-index:1; }
h1 { font-size:20px; margin:0 0 6px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:14px; margin:14px 0; }
.card.ok { border-left:4px solid var(--ok); } .card.bad { border-left:4px solid var(--bad); }
.meta { color:var(--muted); font-size:13px; }
blockquote { margin:8px 0; padding:6px 10px; border-left:3px solid var(--line); }
pre { background:var(--code); padding:8px; border-radius:6px; overflow-x:auto;
      white-space:pre-wrap; word-break:break-word; font-size:13px; }
label { margin-right:14px; cursor:pointer; }
textarea { width:100%; box-sizing:border-box; min-height:44px; background:var(--bg);
           color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:6px; }
input[type=text] { background:var(--bg); color:var(--fg); border:1px solid var(--line);
                   border-radius:6px; padding:5px 8px; }
button { padding:6px 12px; border-radius:6px; border:1px solid var(--line);
         background:var(--card); color:var(--fg); cursor:pointer; }
</style></head><body>
<header>
<h1>Label review (D18)</h1>
<div>Reviewer <input type="text" id="reviewer" placeholder="your name">
<button id="download">Download review.jsonl</button>
<span class="meta" id="progress"></span></div>
<div class="meta">Agree when the label is right: for an error label, the sentence and the
evidence really disagree; for a clean control ("none"), they really agree.</div>
</header>
<main id="items"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const rows = JSON.parse(document.getElementById("data").textContent);
const KEY = "review-sheet:" + __NAME__;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const reviewer = document.getElementById("reviewer");
reviewer.value = saved.__reviewer || "";
function store() {
  saved.__reviewer = reviewer.value;
  try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {}
  const done = rows.filter(r => saved[r.label_id]?.agree !== undefined).length;
  document.getElementById("progress").textContent = ` ${done} of ${rows.length} reviewed`;
}
reviewer.addEventListener("input", store);
const main = document.getElementById("items");
rows.forEach((r, n) => {
  const s = saved[r.label_id] || {};
  const ev = (r.evidence || []).map(e => {
    const where = e.path ? `${e.path}${e.line_start ? ":" + e.line_start : ""}` : (e.page ? "page " + e.page : "");
    return `<div class="meta">${esc(e.source.kind)} ${esc(e.source.url || e.source.id)} ${esc(where)}</div><pre>${esc(e.text)}</pre>`;
  }).join("");
  const card = document.createElement("section");
  card.className = "card" + (s.agree === true ? " ok" : s.agree === false ? " bad" : "");
  card.innerHTML = `<div class="meta">${n + 1}. ${esc(r.label_id)} &middot; ${esc(r.paper_id)} (${esc(r.variant)})</div>
    <div><strong>${esc(r.error_class)}</strong> &middot; ${esc(JSON.stringify(r.target))}</div>
    <blockquote>${esc(r.sentence || "(no sentence: a document-level label)")}</blockquote>
    <div><em>Rationale:</em> ${esc(r.rationale)}</div>${ev}
    <div><label><input type="radio" name="a${n}" value="yes" ${s.agree === true ? "checked" : ""}> Agree</label>
    <label><input type="radio" name="a${n}" value="no" ${s.agree === false ? "checked" : ""}> Disagree</label></div>
    <textarea placeholder="Note (needed when you disagree)">${esc(s.note || "")}</textarea>`;
  card.querySelectorAll("input[type=radio]").forEach(i => i.addEventListener("change", () => {
    saved[r.label_id] = {...(saved[r.label_id] || {}), agree: i.value === "yes"};
    card.className = "card " + (i.value === "yes" ? "ok" : "bad"); store();
  }));
  card.querySelector("textarea").addEventListener("input", e => {
    saved[r.label_id] = {...(saved[r.label_id] || {}), note: e.target.value}; store();
  });
  main.appendChild(card);
});
store();
document.getElementById("download").addEventListener("click", () => {
  if (!reviewer.value.trim()) { alert("Enter your name as reviewer first."); return; }
  const lines = rows.map(r => {
    const s = saved[r.label_id] || {};
    return JSON.stringify({...r, reviewer: s.agree === undefined ? null : reviewer.value.trim(),
                           agree: s.agree === undefined ? null : s.agree, note: s.note || ""});
  });
  const blob = new Blob([lines.join("\\n") + "\\n"], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "review.jsonl"; a.click();
});
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("review", type=Path, help="A review file from review-sample")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.review.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Inside <script>, "</" would end the element; JSON allows "<\\/" for it.
    data = json.dumps(rows).replace("</", "<\\/")
    page = PAGE.replace("__DATA__", data).replace("__NAME__", json.dumps(str(args.review)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"{len(rows)} labels in {args.output}; open it in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
