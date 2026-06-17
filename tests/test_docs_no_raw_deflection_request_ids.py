from __future__ import annotations

from pathlib import Path
import re
import subprocess


RAW_DEFLECTION_REQUEST_ID_RE = re.compile(r"content-ops-[0-9a-f]{32}")
LOCAL_HOME_PATH_RE = re.compile(r"/home/[A-Za-z0-9._-]+/[^\s`\"']+")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PLACEHOLDER_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
PROOF_REDACTION_FILES = (
    Path("docs/extraction/validation/content_ops_faq_deflection_portfolio_hosted_smoke_2026-05-30.md"),
    Path("docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md"),
    Path("docs/extraction/validation/fixtures/deflection_full_volume_live_proof_20260614/summary.json"),
)


def _raw_request_id_hits(paths: list[Path]) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        count = len(RAW_DEFLECTION_REQUEST_ID_RE.findall(text))
        if count:
            hits.append((str(path), count))
    return hits


def _proof_redaction_hits(paths: list[Path]) -> list[tuple[str, str, int]]:
    hits: list[tuple[str, str, int]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        local_home_paths = LOCAL_HOME_PATH_RE.findall(text)
        if local_home_paths:
            hits.append((str(path), "local_home_path", len(local_home_paths)))
        non_placeholder_emails = [
            value
            for value in EMAIL_RE.findall(text)
            if not value.lower().endswith(PLACEHOLDER_EMAIL_DOMAINS)
        ]
        if non_placeholder_emails:
            hits.append((str(path), "non_placeholder_email", len(non_placeholder_emails)))
    return hits


def _tracked_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        repo_root / raw.decode("utf-8")
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def test_raw_request_id_detector_catches_capability_shape(tmp_path: Path) -> None:
    bad_id = "content-ops-" + ("a" * 32)
    candidate = tmp_path / "proof.md"
    candidate.write_text(f"request_id={bad_id}\n", encoding="utf-8")

    assert _raw_request_id_hits([candidate]) == [(str(candidate), 1)]


def test_raw_request_id_detector_allows_redacted_and_non_capability_tokens(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "proof.md"
    candidate.write_text(
        "\n".join(
            (
                "content-ops-[redacted:a83f5c797c38]",
                "content-ops-deflection-request-id-redaction",
                "content-ops-123",
            )
        ),
        encoding="utf-8",
    )

    assert _raw_request_id_hits([candidate]) == []


def test_raw_request_id_detector_allows_fixture_shaped_checker_tokens(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "proof_checker_test.py"
    candidate.write_text(
        "content-ops-fixture-45c06a6950ec4677a214368d6e4dc44f",
        encoding="utf-8",
    )

    assert _raw_request_id_hits([candidate]) == []


def test_proof_redaction_detector_catches_local_paths_and_real_emails(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "proof.md"
    candidate.write_text(
        "\n".join(
            (
                "/home/alice/Desktop/Atlas/tmp/source.csv",
                "contact-email alice@gmail.com",
            )
        ),
        encoding="utf-8",
    )

    assert _proof_redaction_hits([candidate]) == [
        (str(candidate), "local_home_path", 1),
        (str(candidate), "non_placeholder_email", 1),
    ]


def test_proof_redaction_detector_allows_placeholders_and_relative_paths(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "proof.md"
    candidate.write_text(
        "\n".join(
            (
                "tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl",
                "contact-email ops@example.com",
            )
        ),
        encoding="utf-8",
    )

    assert _proof_redaction_hits([candidate]) == []


def test_committed_files_do_not_contain_raw_deflection_request_ids() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hits = _raw_request_id_hits(_tracked_files(repo_root))

    assert hits == []


def test_deflection_proof_artifacts_do_not_reintroduce_stale_identifiers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [repo_root / path for path in PROOF_REDACTION_FILES]

    assert _proof_redaction_hits(paths) == []
