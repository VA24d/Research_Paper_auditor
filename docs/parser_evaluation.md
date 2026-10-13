# Parser Evaluation (task P1.10)

How well PDF parsing works on real ML papers, measured on ten arXiv papers that cover
numeric and author-year citation styles and several venue templates.

## Method

- Papers: `datasets/parser_fixtures/arxiv_ids.txt`, which pins each arXiv version and the
  PDF's SHA-256 (fetched and checked with `scripts/fetch_arxiv.py`; PDFs are not
  committed, decision D13).
- Pipeline: GROBID 0.9.1 (CRF image) with sentence segmentation and coordinates,
  pdfplumber for layout, then the fallback citation linker (`auditor.parser.linking`).
- Statistics: `uv run python scripts/parser_report.py data/papers`.
- Hand checks: every citation link added by the fallback linker (98 links in seven papers)
  was compared against the paper's reference list. Settings extraction and repository
  detection were inspected on all ten papers.
- Gold sample: 100 citation markers, 10 per paper, drawn at random with a fixed seed and
  annotated against the reference list printed in the PDF
  (`datasets/parser_fixtures/citation_gold.jsonl`, scored with
  `uv run python scripts/citation_sample.py score data/papers
  datasets/parser_fixtures/citation_gold.jsonl`). Answers name the reference by first-author
  surname, printed year and the start of the title, so the file stays valid when the
  parser's entry ids change. A link counts as correct only when the linked entries are
  exactly the expected ones.

## Results

| Paper | Pages | Sentences | References | With title / year | Markers | Linked (parser + heuristic) | Settings | Repository |
|---|---|---|---|---|---|---|---|---|
| 1512.03385 (ResNet) | 12 | 529 | 50 | 96% / 100% | 132 | 98% (116 + 14) | 27 | - |
| 1706.03762 (Transformer) | 15 | 271 | 40 | 98% / 100% | 63 | 100% (45 + 18) | 3 | tensorflow/tensor2tensor |
| 1810.04805 (BERT) | 16 | 413 | 55 | 98% / 100% | 78 | 96% (63 + 12) | 36 | google-research/bert |
| 2005.11401 (RAG) | 19 | 342 | 69 | 100% / 99% | 101 | 96% (97 + 0) | 6 | huggingface/transformers |
| 2010.11929 (ViT) | 22 | 474 | 58 | 98% / 100% | 78 | 96% (65 + 10) | 33 | google-research/vision_transformer |
| 2106.09685 (LoRA) | 26 | 574 | 63 | 100% / 94% | 97 | 88% (80 + 5) | 22 | microsoft/LoRA |
| 2201.11903 (Chain-of-Thought) | 43 | 1263 | 84 | 99% / 95% | 112 | 94% (84 + 21) | 7 | - |
| 2210.03629 (ReAct) | 33 | 635 | 40 | 98% / 98% | 53 | 98% (34 + 18) | 6 | - |
| 2305.18290 (DPO) | 27 | 539 | 53 | 100% / 92% | 56 | 100% (56 + 0) | 6 | - |
| 2310.06825 (Mistral 7B) | 9 | 158 | 29 | 100% / 100% | 27 | 100% (27 + 0) | 9 | mistralai/mistral-src |

Sentence counts include table rows. Settings counts include those read from hyperparameter
tables without an LLM and the evaluation setup (datasets, splits, metrics); the LLM table
reader (see the limitations) can add more. Every dataset, split and metric extracted from
the ten papers was checked by hand.

Overall, 765 of 797 citation markers (96.0%) are linked to a bibliography entry: 667 by
GROBID and 98 by the fallback linker. All 98 fallback links checked by hand point at the
right entry. Parsing takes about 5 to 15 seconds per paper on CPU, and GROBID's output is
cached, so later runs take about a second.

On the gold sample, the parser as first annotated linked 99 of the 100 markers, all
correctly (precision 1.00, recall 0.99). The one miss was "Wiegreffe and Marasovic, 2021"
in Chain-of-Thought, which shares first author and year with "Wiegreffe et al., 2021"; it
led to the co-author rule below, and with that rule the sample scores 100 of 100. The
second number is measured on the sample that prompted the fix, so 0.99 is the fairer
estimate.

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
| Same first author and year, different works | Chain-of-Thought: "(Roy and Roth, 2015)" and "(Roy et al., 2015)" | The marker's author count ("X", "X and Y", "X et al.") picks between them |
| Settings only in tables | Mistral 7B gives its architecture only in Table 1 | Table rows kept as table sentences; settings read from rows and columns with a known key |
| Author and year label left in a title | Chain-of-Thought: "Le. 2022a. Finetuned language models are zero-shot learners" | The label is removed, and gives the entry its year when GROBID found none (five entries gained a year) |
| Citations in tables lost | Transformer: "ConvS2S [9] \| 25.16", a score attributed to a cited model | Markers in table cells, which GROBID leaves as plain text, become citation mentions (36 in three papers) |
| No dataset, split or metric settings | "We use an 80/20 train/test split and report macro-F1" gave no task | Patterns for split ratios, held-out fractions, metrics and dataset names, own setup only |
| Two numbers read as one | "16 32" parsed as 1632 | Spaces join digits only before a three-digit group |
| Appendix tables, captions and footnotes dropped | ViT kept 5 of its 8 tables, DPO 0 of 2 | Figures and notes are read inside sections and back matter too; every table GROBID detects is now kept |
| Powers of ten not read | ViT's appendix: "8 • 10 -4", "10 -3" | Any multiplication sign, and a bare power with a flattened superscript |

## Findings in the papers themselves

- **A wrong reference number in RAG.** "who train RoBERTa [35]" points at reference [35],
  which in the published PDF is a neural machine translation paper by Liu et al., not
  RoBERTa. The parser links the marker to what the PDF says; catching that the citation
  does not support the claim is the citation agent's job, which makes this a useful real
  test case for it.
- **Reference entries split by GROBID.** In RAG and DPO, GROBID cut a few reference entries
  in two, so the bibliography has a few extra fragment entries. The markers that cite those
  works still linked to the right entry.

## Known limitations

- **Tables.** Rule-based table reading needs rows GROBID kept intact and a key from a fixed
  list of names (`auditor.planner.tables.ALIASES`). When a repository is known and an LLM
  is configured, tables are also given to the LLM table reader, which keeps only values that
  appear verbatim in the row. ViT's appendix Table 3, whose header cells GROBID merged, is
  an example only the LLM reader can use. Tables GROBID does not detect at all are still
  missed.
- **Unresolved markers.** About 4% of markers stay unlinked, mostly author-year markers
  whose reference entry GROBID parsed badly. They still become citation tasks with the marker
  text (decision D6), so they are reported as unverifiable rather than silently skipped.
- **One annotation pass, spot-checked.** The gold sample was annotated once. A second
  check of 20 markers (two per paper, 29 references) against the PDFs found every expected
  reference correct, but two answers were written down wrongly: a year taken from an arXiv
  id (1902 for a 2019 paper) and a title with the end of the author list attached. Checking
  the whole file for the same causes found seven more such years. All nine fields are
  corrected; the scores did not change. 100 markers still give only a
  coarse estimate: a 95% interval for 99 of 100 runs from about 0.95 to 1.00.
- **Repository detection** can still pick a repository the paper only mentions next to
  words like "our code". The chosen repository is written to the run manifest, and `--repo`
  overrides it.
