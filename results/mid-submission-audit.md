# Audit of the mid-submission report, 30 October 2026

Two read-only audits ran on `docs/mid-submission.pdf`: one checked its numbers against the repository, and one read it as a newcomer. Each main numbers finding was checked against its source before the report changed.

## Still unconfirmed

- **B0's judge.** B0's SPOT run has first-judge grades, but its files do not record which model graded them. The full system's first judge was Gemini 2.5 Flash. If B0's judge was another model, 57% against 29% is not like-for-like. No regrading was run, because it costs money.
- **Crop-table numbers.** 31 and 18 of 33 come from `docs/decisions.md`. No later rerun was found.
- **Macro-F1.** 0.68, 0.14 and 0.00 match `docs/p3-mid-results.md` but were not recomputed.

## Numbers audit

| # | Report said | Evidence | Now |
|---|---|---|---|
| 1 | Theory card: triage 8, pooled 7, two looks 6 of 14 | 8 and 7 are first-judge counts, 6 is both judges (`decisions.md:801-826`) | All first judge (8, 7, 7), both-judge figures listed (5, 5, 6) |
| 2 | Full system 5 of 14 "same judge" against B0's 4 of 14 | 5 needs both judges; B0 has first-judge grades only | 8 of 14 against 4 of 14, first judge; note that 5 remain with a second judge |
| 3 | Table reading 57, 43, 72, 96 to 98, 96%, "12 papers" | Mixed a 39-paper and a 12-paper benchmark (`decisions.md:1075-1093`) | 12-paper figures: 41, 20, 51, 98, 96% |
| 4 | Reversed 28 of 33, gain 21 of 33 | Before crop tables (`decisions.md:1100-1101`) | 31 of 33 and 18 of 33 |
| 5 | Citation linking "797 markers", under "Caught" | 765 of 797 linked; the 100-marker check is separate | 765 of 797, column renamed "Result" |
| 6 | Settings 60 of 60, no caveat | Planted only where the agent had read the value (`p3-mid-results.md:44-45`) | Caveat added; set D marked as planted from templates |
| 7 | 46% seen but not reported, as a current figure; mitigation already tried | From an earlier run that kept 10 of 57 (`decisions.md:435-445`) | Labelled as earlier run; mitigation names the 12 of 57 tried |
| 8 | "254 clean papers" | Real papers, not known clean | "254 real papers; its only hit there is a real error" |
| 9 | Template cases shown as set D | Separate hand-built papers | Row says "not set D" |
| 10 | Scope card: "too few claim labels" | B1 found almost nothing in either condition (`p2_audit_report.md:178-186`) | Reworded |
| 11 | Claim loop "32 timeouts", low effort credited alone | 17 to 32; effort and time limit changed together (`decisions.md:333`) | "17 to 32"; row names both changes |
| 12 | Retrieved chunks 14% beside method check 30% | Different subsets and code versions | Card says "earlier version" |
| 13 | RQ1 "than the same model in a single pass" | RQ1 is answered against B1 (`decisions.md:576-580`) | Names B1, with B0 as second reference |
| 14 | 23 and 152 unmatched findings could read as one number | Different runs | 152 labelled as settings check |

Checked and correct: set C and set D label counts, set A and set B figures, the RQ1 table, the $83 estimate, reading-model and retrieval counts, SPOT dataset counts, the three real-paper examples and the plain-word rewrites. B0's $3.39 was recomputed from its token log.

## Reader audit

| # | Finding | Outcome |
|---|---|---|
| 1 | RQ1 numbers split from their caveats | "Caveats on this comparison" heading; line under the score tiles |
| 2 | Hard wording in caveats | Rewritten; embedding-unload bullet shortened and kept, as it discloses a rerun |
| 3 | Baseline table mixed datasets and misleading costs | SPOT only; set B moved to a bullet; costs show paper counts |
| 4 | Results rows unclear | Header, counts and notes fixed; model moved to the group title |
| 5 | Bar colours had two meanings | One rule for both; bold marks the best variant; false-alarm bars blue |
| 6 | Takeaways went beyond small counts | Gaps and sample sizes stated |
| 7 | Undefined terms | Glossary moved before first use; pair and version defined; plain labels in cards |
| 8 | One thing with two names | "Findings" and "method errors" used throughout |
| 9 | Diagram unreadable, with code names, and unbuilt parts shown as working | Redrawn; unbuilt parts marked |
| 10 | Opening thin; no course name | Rewritten; CL3.410 in the header |
| 11 | Challenges inconsistent | Reworded; verifier date now matches the timeline |
| 12 | Card layout; no heading on continuation | Same height per row; "Ablations, continued" added; "Not yet measured" moved out of the grid |
| 13 | Gantt header numbers touching | Smaller type |
| 14 | Tight gap above the diagram | Gap added |
| 15 | Mixed voice in the Part table | Done column uses noun phrases, Still to do uses imperatives |
| 16 | "Real cases" set used in no result; vague set B wording | Row says "not scored in this report"; set B counts stated |
| 17 | "What is left" spread across sections | "Not yet measured" moved next to Challenges; Part table and timeline unchanged |

## Same errors elsewhere

`docs/p3-mid-results.md` still shows the mixed theory and table-reading figures and the old 28 and 21 of 33.
