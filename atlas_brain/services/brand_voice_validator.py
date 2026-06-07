import yaml
import re
import argparse
from pathlib import Path


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
        for section in ("tone_rules", "content_rules"):
            for index, rule in enumerate(self.config.get(section) or []):
                if not isinstance(rule, dict):
                    raise ValueError(
                        f"{section}[{index}] must be a mapping, got "
                        f"{type(rule).__name__}"
                    )
                missing = {"pattern", "description"} - rule.keys()
                if missing:
                    rule_id = rule.get("id", "<unnamed>")
                    raise ValueError(
                        f"{section}[{index}] (id={rule_id!r}) missing required "
                        f"keys: {sorted(missing)}"
                    )
                try:
                    re.compile(rule["pattern"])
                except re.error as exc:
                    rule_id = rule.get("id", "<unnamed>")
                    raise ValueError(
                        f"{section}[{index}] (id={rule_id!r}) has an invalid "
                        f"regex pattern: {exc}"
                    ) from exc

    def validate(self, text: str, content_type: str) -> list[str]:
        """
        Validates a piece of text against the loaded brand voice rules.

        Args:
            text: The content to validate.
            content_type: The type of content (e.g., 'landing_page', 'blog_post').

        Returns:
            A list of string descriptions of any violations found. An empty
            list means the content is on-brand.
        """
        if not isinstance(text, str):
            raise TypeError("text must be str")

        violations = []
        lower_text = text.lower()

        # 1. Forbidden vocabulary -- whole-word match. Unanchored substring
        #    matching false-positives on clean prose ('transform' in
        #    'Transformer', 'disrupt' in 'non-disruptive', 'leverage' in
        #    'deleverage'); \b...\b anchors to word boundaries (the hyphen in
        #    'non-disruptive' is itself a boundary, so 'disrupt' no longer fires).
        for word in (self.config.get("vocabulary") or {}).get("avoid") or []:
            if re.search(r"\b" + re.escape(word) + r"\b", lower_text):
                violations.append(f"Contains forbidden word: '{word}'")

        # 2. Tone rules (regex-based).
        for rule in self.config.get("tone_rules") or []:
            if re.search(rule["pattern"], text, re.IGNORECASE) and rule.get(
                "fail_on_match", True
            ):
                violations.append(f"Tone violation: {rule['description']}")

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
                violations.append(f"Content rule violation: {rule['description']}")

            # Fails if the pattern is NOT found (e.g., "must mention extensibility").
            elif not fail_on_match and not pattern_found:
                violations.append(f"Content rule violation: {rule['description']}")

        return violations


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

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found at {args.file}")
        exit(1)

    with open(args.file, "r") as f:
        content_to_validate = f.read()

    validator = BrandVoiceValidator(config_path=args.config)
    violations = validator.validate(content_to_validate, args.type)

    if violations:
        print(f"FAIL: Found {len(violations)} brand voice violations in {args.file}:")
        for violation in violations:
            print(f"  - {violation}")
        exit(1)
    else:
        print(f"PASS: {args.file} is on-brand.")
        exit(0)


if __name__ == "__main__":
    main()
