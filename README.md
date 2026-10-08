# Evidence-Gated Verification Agent for Research Paper Auditing

Group project for **CL3.410: Language Models and Agents (Monsoon 2026)**.

## Team: Hood Research Department

- Vijay Aravynthan (2023111007)
- Anurag Peddi (2023101090)
- Itikela Bhaskar (2023111032)
- Sudheera Y S (2023111002)

## Overview

An automated verification system that audits research conference submissions for factual correctness (specifically citations and data) before human review. The system decomposes submissions into checkable items, inspects cited full texts and code repositories, and enforces evidence-gated verification where flagged discrepancies must be backed by verbatim quotes.

## Documents

- [Project Proposal (Markdown)](./LMA_Group_Project_Proposal.md)
- [Project Proposal (PDF)](./LMA%20group%20project%20proposal.pdf)

## System Pipeline

1. **Submission & Venue Profile**: Paper text, bibliography, repository, and venue rules.
2. **Planner & State Tracker**: Extracts claims, citations, numbers, and code settings; tracks audit items.
3. **Specialists**:
   - **Citation Agent**: Reference lookup, full-text retrieval, and claim validity check.
   - **Code Agent**: Static repository search for settings, splits, and metrics.
   - **Template Checker**: Deterministic format rules check (no LLM call).
4. **Verifier Agent**: Tests suspected errors against source evidence; rejects flags lacking verbatim quotes.
5. **Output States**: Verified flag (quoted evidence), Needs human review, or Dropped.

