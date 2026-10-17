from auditor.orchestration.synthesis import severity_of, synthesise
from auditor.schemas import (
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    PaperVariant,
    RuleTarget,
    SentenceAnchor,
    SettingTarget,
    Severity,
    SourceKind,
    SourceRef,
    Target,
    VerificationState,
    VerifiedFinding,
)

REPO = SourceRef(kind=SourceKind.REPOSITORY, id="https://github.com/o/r@abc")
PAPER = SourceRef(kind=SourceKind.SUBMISSION, id="paper")


def code(line: int) -> Evidence:
    return Evidence(
        kind=EvidenceKind.CODE, source=REPO, text="metric = 'auc'", path="cfg.py", line_start=line
    )


def quote(text: str) -> Evidence:
    return Evidence(kind=EvidenceKind.QUOTE, source=PAPER, text=text)


def decision(
    number: int,
    *,
    sentence: str = "s1",
    error_class: ErrorClass = ErrorClass.METRIC_MISMATCH,
    target: Target | None = None,
    summary: str = "The paper states accuracy; the repository has auc (cfg.py:3)",
    evidence: tuple[Evidence, ...] | None = None,
    state: VerificationState = VerificationState.NEEDS_HUMAN_REVIEW,
    uncertain: bool = False,
) -> VerifiedFinding:
    finding = CandidateFinding(
        finding_id=f"t{number}-f0",
        task_id=f"t{number}",
        paper_id="paper",
        variant=PaperVariant.ORIGINAL,
        error_class=error_class,
        anchor=SentenceAnchor(sentence_id=sentence),
        target=target or SettingTarget(key="eval_metric"),
        summary=summary,
        evidence=evidence if evidence is not None else (quote(f"text of {sentence}"), code(3)),
        uncertain=uncertain,
    )
    accepted = finding.evidence if state is VerificationState.VERIFIED_FLAG else ()
    return VerifiedFinding(finding=finding, state=state, reason="r", accepted_evidence=accepted)


def test_the_same_mismatch_stated_twice_is_one_finding_with_a_duplicate() -> None:
    first, second = synthesise([decision(1, sentence="s1"), decision(2, sentence="s9")])
    assert first.duplicate_of is None
    assert second.duplicate_of == first.finding.finding_id


def test_different_code_evidence_or_claims_are_not_duplicates() -> None:
    other_line = decision(2, evidence=(quote("q"), code(7)))
    other_claim = decision(3, summary="The paper states F1; the repository has auc (cfg.py:3)")
    out = synthesise([decision(1), other_line, other_claim])
    assert [d.duplicate_of for d in out] == [None, None, None]


def test_findings_without_repository_or_source_evidence_are_never_merged() -> None:
    bare = [decision(1, evidence=(quote("q"),)), decision(2, evidence=(quote("q"),))]
    assert [d.duplicate_of for d in synthesise(bare)] == [None, None]


def test_a_verified_flag_is_kept_over_its_duplicate_for_review() -> None:
    review = decision(1)
    flag = decision(2, state=VerificationState.VERIFIED_FLAG)
    first, second = synthesise([review, flag])
    assert first.finding.finding_id == "t2-f0"
    assert second.duplicate_of == "t2-f0"


def test_dropped_findings_are_not_grouped() -> None:
    dropped = decision(2, state=VerificationState.DROPPED)
    out = synthesise([decision(1), dropped])
    assert [d.duplicate_of for d in out] == [None, None]


def test_findings_are_ranked_by_state_then_severity_then_certainty() -> None:
    template = decision(
        1,
        error_class=ErrorClass.TEMPLATE_VIOLATION,
        target=RuleTarget(rule_id="margins"),
        evidence=(),
    )
    setting = decision(2, error_class=ErrorClass.SETTING_MISMATCH, evidence=(quote("a"), code(5)))
    unsure = decision(3, uncertain=True, evidence=(quote("b"), code(6)))
    leak = decision(4, error_class=ErrorClass.DATA_LEAKAGE, evidence=(quote("c"), code(8)))
    dropped = decision(5, error_class=ErrorClass.DATA_LEAKAGE, state=VerificationState.DROPPED)
    out = synthesise([template, setting, unsure, leak, dropped])
    assert [d.finding.task_id for d in out] == ["t4", "t3", "t2", "t1", "t5"]
    assert [d.severity for d in out] == [
        Severity.HIGH,
        Severity.HIGH,
        Severity.MEDIUM,
        Severity.LOW,
        Severity.HIGH,
    ]


def test_severity_follows_the_error_class_and_desk_reject_rules() -> None:
    def of(error_class: ErrorClass, target: Target | None = None) -> Severity:
        return severity_of(decision(1, error_class=error_class, target=target).finding)

    assert of(ErrorClass.METHOD_MISMATCH) is Severity.HIGH
    assert of(ErrorClass.REFERENCE_NOT_FOUND) is Severity.HIGH
    assert of(ErrorClass.SETTING_MISMATCH) is Severity.MEDIUM
    assert of(ErrorClass.REFERENCE_METADATA_MISMATCH) is Severity.LOW
    assert of(ErrorClass.TEMPLATE_VIOLATION, RuleTarget(rule_id="max_pages")) is Severity.HIGH
    assert of(ErrorClass.TEMPLATE_VIOLATION, RuleTarget(rule_id="margins")) is Severity.LOW
    assert all(severity_of(decision(1, error_class=c).finding) for c in ErrorClass)
