# Venue Profiles

Each file is a `VenueProfile` (`auditor.schemas.venue`) for the submission (review)
version of a paper at one venue. Pass one to the pipeline with
`uv run auditor audit paper.pdf --venue configs/venues/neurips-2025.json`. The planner makes
one template task per rule, and the template checker (`auditor.agents.template`) runs the
check named in the rule's `check` field. `scripts/template_eval.py` reports how the checks
do on the hand-built fixture cases.

Every limit below was checked against the venue's call for papers, author guide or official
style files in October 2026. Each rule's `description` ends with its source. A limit that
could not be confirmed in an official source is left out of the profile.

| File | Page limit (main text) | Body font | Paper | Required section | Not allowed at submission |
|---|---|---|---|---|---|
| `neurips-2025.json` | 9 | 10 pt | US Letter | NeurIPS Paper Checklist | Acknowledgments |
| `iclr-2026.json` | 9 (10 for camera-ready) | 10 pt | US Letter | none | none verified |
| `icml-2025.json` | 8 | 10 pt | US Letter | Impact Statement | Acknowledgements |
| `acl-2025.json` | 8 (long) | 11 pt | A4 | Limitations | Acknowledgements |
| `acl-2025-short.json` | 4 (short) | 11 pt | A4 | Limitations | Acknowledgements |
| `emnlp-2025.json` | 8 (long) | 11 pt | A4 | Limitations | Acknowledgements |
| `emnlp-2025-short.json` | 4 (short) | 11 pt | A4 | Limitations | Acknowledgements |

All five venues review double-blind, so every profile has an `anonymous` rule. It needs
visual inspection and always ends `unverifiable`. Author names in the parsed header and links
to accounts on code hosts (`github.com/<owner>`) are attached to it as uncertain hints.

## Sources

- NeurIPS 2025: call for papers <https://neurips.cc/Conferences/2025/CallForPapers> (page
  limit, what does not count, no acknowledgments at submission, checklist required) and the
  style files <https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip>,
  `neurips_2025.tex` (10 pt type, US Letter, "NeurIPS Paper Checklist" heading).
- ICLR 2026: author guide <https://iclr.cc/Conferences/2026/AuthorGuide> ("At the time of
  submission, the main text should be 9 pages or fewer"; references, appendices, the ethics
  and reproducibility statements, the LLM-usage section and acknowledgements do not count)
  and the template <https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip>,
  `iclr2026_conference.tex` (10 pt type, US Letter).
- ICML 2025: call for papers <https://icml.cc/Conferences/2025/CallForPapers> (8 pages;
  references, impact statement and appendices unlimited; Impact Statement required) and the
  style files <https://media.icml.cc/Conferences/ICML2025/Styles/icml2025.zip>,
  `example_paper.tex` (10 pt Times, no acknowledgements in the blind version) and
  `icml2025.sty` (8.5 x 11 in paper).
- ACL 2025 and EMNLP 2025, both reviewed through ACL Rolling Review: the calls
  <https://2025.aclweb.org/calls/main_conference_papers/> and
  <https://2025.emnlp.org/calls/main_conference_papers/> (8 pages for long papers, 4 for
  short ones, double-blind), the ACL formatting guide
  <https://acl-org.github.io/ACLPUB/formatting.html> (A4, 11 pt text, Limitations and ethics
  sections and references outside the limit, no acknowledgements in the review version) and
  the ARR call <https://aclrollingreview.org/cfp> (a section titled "Limitations" is
  required, appendices go after the references). The ARR page read was the live 2026 version;
  the Limitations requirement dates from the December 2023 cycle, so it applied in 2025.

## Left out, and why

- **Margins and text area.** The official values are known (NeurIPS and ICLR: a 5.5 x 9 in
  text block with a 1.5 in left margin; ICML: 6.75 x 9 in, 0.75 in left margin; ACL: 2.5 cm
  on every side). The parser's text box for a page covers every upright character, though,
  and that includes the line numbers that review versions print in the margin (ACL's ruler,
  the NeurIPS and ICML line numbers) and the page numbers in the footer. A margin rule on
  that box would flag almost every compliant submission. The `margins` check is implemented
  and tested for use once the layout separates body text from margin furniture.
- **ICLR 2026 acknowledgements.** No official text says to omit them at submission beyond
  the general anonymity rule.
- **ICLR 2026 required sections.** The ethics and reproducibility statements are optional,
  and the LLM-usage section is required only when LLMs played a significant role, which a
  formatting check cannot know.
- **NeurIPS limitations section.** Encouraged by the checklist, not required.
- **Font family and line spacing.** No check reads them yet.

## Things to know when using these profiles

- The profiles describe submissions. arXiv versions of accepted papers are camera-ready:
  they name their authors, often add acknowledgements, may drop the NeurIPS checklist and
  have one more page (ICML, ACL, EMNLP) or ten pages (ICLR). Use a profile only on papers in
  the form it describes, or drop the rules that do not apply.
- `exclude_sections` lists the section titles (regular expressions, case-insensitive) that
  do not count toward the page limit. Sections numbered with a letter (A, B.2) after
  numbered sections count as appendices too, and so does everything after the first page
  of the reference list.
- Findings for `paper_size` and `margins` are anchored on the pages involved. Every other
  rule gives a document-level finding, so gold labels for a page limit, a font size or a
  section rule use a `document` anchor (decision D3).
