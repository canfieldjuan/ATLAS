import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


SEVERITIES = frozenset({"BLOCKER", "MAJOR", "NIT"})
BLOCKING_SEVERITIES = frozenset({"BLOCKER", "MAJOR"})


@dataclass(frozen=True)
class BrandVoiceFinding:
    rule_id: str
    severity: str
    category: str
    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        return self.message


class BrandVoiceValidator:
    """
    A deterministic validator that checks text against a codified brand voice
    defined in a YAML configuration file.
    """

    def __init__(self, config_path: Path):
        """
        Initializes the validator by loading the brand voice configuration.

        Args:
            config_path: The path to the brand_voice.yml file.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Brand voice config not found at: {config_path}")
        with open(config_path, "r") as f:
            # An empty config file parses to None; coerce so callers of
            # self.config.get(...) never hit AttributeError.
            self.config = yaml.safe_load(f) or {}
        self._validate_config_shape()

    def _validate_config_shape(self) -> None:
        """
        Fail loud at load time on a malformed rule, naming the offender.

        A hand-edited rule missing 'pattern' or 'description', or carrying an
        invalid regex, would otherwise crash the whole gate with an opaque
        KeyError/re.error part-way through validate() -- silently disabling every
        rule after it. Surface it clearly at construction instead.
        """
        self._validate_vocabulary_use_shape()
        for section in ("tone_rules", "content_rules"):
            for index, rule in enumerate(self.config.get(section) or []):
                if not isinstance(rule, dict):
                    raise ValueError(
                        f"{section}[{index}] must be a mapping, got "
                        f"{type(rule).__name__}"
                    )
                required = {"id", "pattern", "description"}
                if section == "content_rules":
                    required.add("applies_to")
                missing = required - rule.keys()
                if missing:
                    rule_id = rule.get("id", "<unnamed>")
                    raise ValueError(
                        f"{section}[{index}] (id={rule_id!r}) missing required "
                        f"keys: {sorted(missing)}"
                    )
                if not str(rule["id"]).strip():
                    raise ValueError(f"{section}[{index}] has an empty rule id")
                if section == "tone_rules" and rule.get("fail_on_match", True) is not True:
                    rule_id = rule.get("id", "<unnamed>")
                    raise ValueError(
                        f"{section}[{index}] (id={rule_id!r}) must omit "
                        "fail_on_match or set it to true; tone rules only fail "
                        "on matches"
                    )
                self._normalize_severity(
                    rule.get("severity"),
                    default=self._default_severity(section),
                    section=section,
                    index=index,
                    rule_id=str(rule["id"]),
                )
                try:
                    re.compile(rule["pattern"])
                except re.error as exc:
                    rule_id = rule.get("id", "<unnamed>")
                    raise ValueError(
                        f"{section}[{index}] (id={rule_id!r}) has an invalid "
                        f"regex pattern: {exc}"
                    ) from exc

    def _vocabulary_config(self) -> dict:
        vocabulary = self.config.get("vocabulary")
        if vocabulary is None:
            return {}
        if not isinstance(vocabulary, dict):
            raise ValueError("vocabulary must be a mapping")
        return vocabulary

    def _vocabulary_use_entries(self) -> list[object]:
        vocabulary = self._vocabulary_config()
        if "use" not in vocabulary or vocabulary["use"] is None:
            return []
        use_entries = vocabulary["use"]
        if not isinstance(use_entries, list):
            raise ValueError("vocabulary.use must be a list of preferred-term mappings")
        return use_entries

    def _validate_vocabulary_use_shape(self) -> None:
        use_entries = self._vocabulary_use_entries()
        for index, entry in enumerate(use_entries):
            self._vocabulary_use_pair(entry, index)

    @staticmethod
    def _vocabulary_use_pair(entry: object, index: int) -> tuple[str, str]:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                f"vocabulary.use[{index}] must be a one-item mapping of "
                "preferred term to discouraged term"
            )
        preferred, discouraged = next(iter(entry.items()))
        if not isinstance(preferred, str) or not preferred.strip():
            raise ValueError(f"vocabulary.use[{index}] has an empty preferred term")
        if not isinstance(discouraged, str) or not discouraged.strip():
            raise ValueError(f"vocabulary.use[{index}] has an empty discouraged term")
        return preferred, discouraged

    def _vocabulary_use_pairs(self) -> list[tuple[str, str]]:
        use_entries = self._vocabulary_use_entries()
        return [
            self._vocabulary_use_pair(entry, index)
            for index, entry in enumerate(use_entries)
        ]

    @staticmethod
    def _default_severity(section: str) -> str:
        if section == "tone_rules":
            return "MAJOR"
        return "BLOCKER"

    @staticmethod
    def _normalize_severity(
        value: object,
        *,
        default: str,
        section: str,
        index: int,
        rule_id: str,
    ) -> str:
        if value is None:
            return default
        severity = str(value).upper()
        if severity not in SEVERITIES:
            allowed = ", ".join(sorted(SEVERITIES))
            raise ValueError(
                f"{section}[{index}] (id={rule_id!r}) has invalid severity "
                f"{value!r}; expected one of: {allowed}"
            )
        return severity

    def _rule_severity(self, rule: dict, *, section: str, default: str) -> str:
        return self._normalize_severity(
            rule.get("severity"),
            default=default,
            section=section,
            index=-1,
            rule_id=str(rule["id"]),
        )

    def validate(self, text: str, content_type: str) -> list[BrandVoiceFinding]:
        """
        Validates a piece of text against the loaded brand voice rules.

        Args:
            text: The content to validate.
            content_type: The type of content (e.g., 'landing_page', 'blog_post').

        Returns:
            A list of structured findings. An empty list means the content is
            on-brand.
        """
        if not isinstance(text, str):
            raise TypeError("text must be str")

        findings = []
        lower_text = text.lower()

        # 1. Forbidden vocabulary -- whole-word match. Unanchored substring
        #    matching false-positives on clean prose ('transform' in
        #    'Transformer', 'disrupt' in 'non-disruptive', 'leverage' in
        #    'deleverage'); \b...\b anchors to word boundaries (the hyphen in
        #    'non-disruptive' is itself a boundary, so 'disrupt' no longer fires).
        for word in self._vocabulary_config().get("avoid") or []:
            if re.search(r"\b" + re.escape(word) + r"\b", lower_text):
                findings.append(
                    BrandVoiceFinding(
                        rule_id=f"vocabulary.avoid.{word}",
                        severity="BLOCKER",
                        category="vocabulary",
                        message=f"Contains forbidden word: '{word}'",
                    )
                )

        for preferred, discouraged in self._vocabulary_use_pairs():
            if re.search(r"\b" + re.escape(discouraged) + r"\b", text, re.IGNORECASE):
                findings.append(
                    BrandVoiceFinding(
                        rule_id=f"vocabulary.use.{discouraged}",
                        severity="NIT",
                        category="vocabulary",
                        message=f"Prefer '{preferred}' over '{discouraged}'",
                        suggestion=f"Use '{preferred}' instead of '{discouraged}'",
                    )
                )

        # 2. Tone rules (regex-based).
        for rule in self.config.get("tone_rules") or []:
            if re.search(rule["pattern"], text, re.IGNORECASE):
                findings.append(
                    BrandVoiceFinding(
                        rule_id=str(rule["id"]),
                        severity=self._rule_severity(
                            rule, section="tone_rules", default="MAJOR"
                        ),
                        category="tone",
                        message=f"Tone violation: {rule['description']}",
                    )
                )

        # 3. Content-specific rules.
        for rule in self.config.get("content_rules") or []:
            if rule.get("applies_to") != content_type:
                continue
            pattern_found = re.search(rule["pattern"], lower_text, re.IGNORECASE)
            # Default True mirrors tone_rules: a rule that omits fail_on_match
            # BANS on match rather than silently inverting into a must-contain
            # check (which flagged clean text and passed banned text).
            fail_on_match = rule.get("fail_on_match", True)

            # Fails if the pattern is found (e.g., "don't use future tense").
            if fail_on_match and pattern_found:
                findings.append(
                    BrandVoiceFinding(
                        rule_id=str(rule["id"]),
                        severity=self._rule_severity(
                            rule, section="content_rules", default="BLOCKER"
                        ),
                        category="content",
                        message=f"Content rule violation: {rule['description']}",
                    )
                )

            # Fails if the pattern is NOT found (e.g., "must mention extensibility").
            elif not fail_on_match and not pattern_found:
                findings.append(
                    BrandVoiceFinding(
                        rule_id=str(rule["id"]),
                        severity=self._rule_severity(
                            rule, section="content_rules", default="BLOCKER"
                        ),
                        category="content",
                        message=f"Content rule violation: {rule['description']}",
                    )
                )

        return findings


def _print_findings(findings: list[BrandVoiceFinding]) -> None:
    for finding in findings:
        print(f"  - [{finding.severity}] {finding.rule_id}: {finding.message}")
        if finding.suggestion:
            print(f"    suggestion: {finding.suggestion}")


def _print_advisory_findings(
    findings: list[BrandVoiceFinding], file_path: Path, *, strict: bool
) -> None:
    status = "FAIL" if strict else "WARN"
    strict_note = " in strict mode" if strict else ""
    print(
        f"{status}: Found {len(findings)} advisory brand voice "
        f"findings{strict_note} in {file_path}:"
    )
    _print_findings(findings)


def _exit_code(
    blocking_findings: list[BrandVoiceFinding],
    advisory_findings: list[BrandVoiceFinding],
    *,
    strict: bool,
) -> int:
    if blocking_findings or (strict and advisory_findings):
        return 1
    return 0


def _result_status(
    blocking_findings: list[BrandVoiceFinding],
    advisory_findings: list[BrandVoiceFinding],
    *,
    strict: bool,
) -> str:
    if blocking_findings or (strict and advisory_findings):
        return "fail"
    if advisory_findings:
        return "warn"
    return "pass"


def _finding_payload(finding: BrandVoiceFinding) -> dict[str, str | None]:
    return {
        "rule_id": finding.rule_id,
        "severity": finding.severity,
        "category": finding.category,
        "message": finding.message,
        "suggestion": finding.suggestion,
    }


def _print_json_result(
    *,
    file_path: Path,
    content_type: str,
    strict: bool,
    blocking_findings: list[BrandVoiceFinding],
    advisory_findings: list[BrandVoiceFinding],
    exit_code: int,
) -> None:
    payload = {
        "ok": exit_code == 0,
        "status": _result_status(
            blocking_findings, advisory_findings, strict=strict
        ),
        "file": str(file_path),
        "content_type": content_type,
        "strict": strict,
        "summary": {
            "total": len(blocking_findings) + len(advisory_findings),
            "blocking": len(blocking_findings),
            "advisory": len(advisory_findings),
        },
        "findings": [
            _finding_payload(finding)
            for finding in [*blocking_findings, *advisory_findings]
        ],
    }
    print(json.dumps(payload, sort_keys=True))


def _print_json_error(
    *,
    file_path: Path,
    content_type: str,
    strict: bool,
    code: str,
    message: str,
) -> None:
    payload = {
        "ok": False,
        "status": "error",
        "file": str(file_path),
        "content_type": content_type,
        "strict": strict,
        "summary": {
            "total": 0,
            "blocking": 0,
            "advisory": 0,
        },
        "findings": [],
        "error": {
            "code": code,
            "message": message,
        },
    }
    print(json.dumps(payload, sort_keys=True))


def main():
    """
    Command-line interface for the BrandVoiceValidator.
    """
    parser = argparse.ArgumentParser(
        description="Validate a text file against the codified brand voice."
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the text or markdown file to validate.",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["landing_page", "blog_post", "release_notes", "tweet"],
        help="The type of content being validated.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "skills/brand/brand_voice.yml",
        help="Path to the brand_voice.yml configuration file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on advisory NIT findings instead of warning only.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for validation results.",
    )

    args = parser.parse_args()

    if not args.file.exists():
        message = f"File not found at {args.file}"
        if args.format == "json":
            _print_json_error(
                file_path=args.file,
                content_type=args.type,
                strict=args.strict,
                code="file_not_found",
                message=message,
            )
        else:
            print(f"Error: {message}")
        exit(1)

    with open(args.file, "r") as f:
        content_to_validate = f.read()

    try:
        validator = BrandVoiceValidator(config_path=args.config)
        findings = validator.validate(content_to_validate, args.type)
    except (ValueError, TypeError, yaml.YAMLError, FileNotFoundError) as exc:
        if args.format == "json":
            _print_json_error(
                file_path=args.file,
                content_type=args.type,
                strict=args.strict,
                code="invalid_config",
                message=str(exc),
            )
        else:
            print(f"Error: {exc}")
        exit(1)
    blocking_findings = [
        finding for finding in findings if finding.severity in BLOCKING_SEVERITIES
    ]
    advisory_findings = [
        finding for finding in findings if finding.severity not in BLOCKING_SEVERITIES
    ]
    exit_code = _exit_code(
        blocking_findings, advisory_findings, strict=args.strict
    )

    if args.format == "json":
        _print_json_result(
            file_path=args.file,
            content_type=args.type,
            strict=args.strict,
            blocking_findings=blocking_findings,
            advisory_findings=advisory_findings,
            exit_code=exit_code,
        )
        exit(exit_code)

    if blocking_findings:
        print(
            f"FAIL: Found {len(blocking_findings)} blocking brand voice "
            f"violations in {args.file}:"
        )
        _print_findings(blocking_findings)
        if advisory_findings:
            _print_advisory_findings(
                advisory_findings, args.file, strict=args.strict
            )
        exit(exit_code)
    elif advisory_findings:
        _print_advisory_findings(advisory_findings, args.file, strict=args.strict)
        exit(exit_code)
    else:
        print(f"PASS: {args.file} is on-brand.")
        exit(exit_code)


if __name__ == "__main__":
    main()
