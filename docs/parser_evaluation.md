# Parser Evaluation (task P1.10)

How well PDF parsing works on real ML papers, measured on ten arXiv papers that cover
numeric and author-year citation styles and several venue templates.

## Method

- Papers: `datasets/parser_fixtures/arxiv_ids.txt` (fetched with `scripts/fetch_arxiv.py`;
  PDFs are not committed, decision D13).
- Pipeline: GROBID 0.9.1 (CRF image) with sentence segmentation and coordinates,
  pdfplumber for layout, then the fallback citation linker (`auditor.parser.linking`).
- Statistics: `uv run python scripts/parser_report.py data/papers`.
- Hand checks: every citation link added by the fallback linker (58 links in five papers)
  was compared against the paper's reference list. Settings extraction and repository
  detection were inspected on all ten papers.

## Results

| Paper | Pages | Sentences | References | With title / year | Markers | Linked (parser + heuristic) | Settings | Repository |
|---|---|---|---|---|---|---|---|---|
| 1512.03385 (ResNet) | 12 | 454 | 50 | 96% / 100% | 118 | 98% (116 + 0) | 19 | - |
| 1706.03762 (Transformer) | 15 | 228 | 40 | 98% / 100% | 45 | 100% (45 + 0) | 3 | tensorflow/tensor2tensor |
| 1810.04805 (BERT) | 16 | 363 | 55 | 98% / 98% | 78 | 96% (63 + 12) | 28 | google-research/bert |
| 2005.11401 (RAG) | 19 | 295 | 69 | 100% / 99% | 97 | 100% (97 + 0) | 1 | huggingface/transformers |
| 2010.11929 (ViT) | 22 | 382 | 58 | 98% / 100% | 78 | 96% (65 + 10) | 24 | google-research/vision_transformer |
| 2106.09685 (LoRA) | 26 | 474 | 63 | 100% / 94% | 97 | 88% (80 + 5) | 11 | microsoft/LoRA |
| 2201.11903 (Chain-of-Thought) | 43 | 1148 | 84 | 99% / 90% | 112 | 88% (84 + 14) | 0 | - |
| 2210.03629 (ReAct) | 33 | 562 | 40 | 98% / 98% | 53 | 94% (34 + 16) | 5 | - |
| 2305.18290 (DPO) | 27 | 473 | 53 | 100% / 92% | 56 | 100% (56 + 0) | 4 | - |
| 2310.06825 (Mistral 7B) | 9 | 132 | 29 | 100% / 100% | 27 | 100% (27 + 0) | 0 | mistralai/mistral-src |

Overall, 724 of 761 citation markers (95.1%) are linked to a bibliography entry: 667 by
GROBID and 57 by the fallback linker. All 58 fallback links checked by hand point at the
right entry. Parsing takes about 5 to 15 seconds per paper on CPU, and GROBID's output is
cached, so later runs take about a second.

Repository detection found the paper's own repository for all six papers that announce one
in the text, and returned nothing for the four that do not.

## Problems found and fixed during this evaluation

| Problem | Example | Fix |
|---|---|---|
| A year letter matched the wrong work | BERT: "Peters et al., 2018a" linked to the 2018b paper, because GROBID lost the 2018a entry's year | A lettered year must appear with that letter in the entry's raw reference text |
| Author-year markers GROBID leaves unresolved | ReAct: "(Shridhar et al., 2020b)", "(Yao et al., 2020; Huang et al., 2022a)", "(Fever, Thorne et al., 2018" | Per-part linking; a marker links only if every part resolves; surname taken before "et al." |
| A marker citing two works linked only one | Chain-of-Thought: "(Wu et al., 2022a,b)" | "2022a,b" expands to both works |
| URL cut at a line break | ViT: "https://github.com/ google-research/vision_transformer" | Repaired from the PDF's link annotation |
| Margin box polluted by arXiv's side stamp | Every arXiv PDF, page 1 | Rotated text is left out of the text box |
| Cited tools taken as the paper's repository | DPO picked `kingoflolz/mesh-transformer-jax` | Repository only from sentences that announce code; no fallback to the first link |
| Table rows read as settings | LoRA: "Batch Size 8 8 32 4 ..." | Sentences that look like flattened table rows are skipped |
| Other systems' settings | LoRA: "GPT-3 175B fine-tuned with Adam" | Optimizer names only from first-person sentences |
| Non-training "steps" | ReAct: "We set 7 and 5 steps for HotpotQA" | Bare "N steps" needs a training context |

## Known limitations

- **Tables.** Hyperparameter tables are not parsed into settings yet. Mistral 7B, for
  example, gives its architecture only in a table, so no settings are extracted. Adding table
  extraction (planned with Docling) is the main gap for the code checks.
- **Unresolved markers.** About 5% of markers stay unlinked, mostly author-year markers
  whose reference entry GROBID parsed badly. They still become citation tasks with the marker
  text (decision D6), so they are reported as unverifiable rather than silently skipped.
- **Statistics, not a gold set.** The table counts what was extracted and linked. Apart from
  the fallback links, link correctness was not hand-checked for every marker. A gold sample
  per paper is still to be annotated for the final write-up.
- **Repository detection** can still pick a repository the paper only mentions next to
  words like "our code". The chosen repository is written to the run manifest, and `--repo`
  overrides it.
