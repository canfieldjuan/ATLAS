"""
Math skill - handles basic arithmetic queries locally.

Uses safe tokenized evaluation (no eval/exec).
"""

import operator
import re

from .base import SkillResult

# Supported operators
_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}

# Token pattern: numbers (with decimals), operators, parentheses
_TOKEN_RE = re.compile(r"(\d+\.?\d*|[+\-*/()])")


def _safe_eval(expr: str) -> float:
    """
    Safely evaluate a basic arithmetic expression.

    Supports: +, -, *, /, parentheses, decimals.
    Raises ValueError on invalid input.
    """
    tokens = _TOKEN_RE.findall(expr)
    if not tokens:
        raise ValueError("No valid expression found")

    # Reconstruct and validate — only digits, operators, parens, dots, spaces
    cleaned = "".join(tokens)
    pos = 0

    def parse_expr() -> float:
        nonlocal pos
        result = parse_term()
        while pos < len(tokens) and tokens[pos] in ("+", "-"):
            op = tokens[pos]
            pos += 1
            right = parse_term()
            result = _OPS[op](result, right)
        return result

    def parse_term() -> float:
        nonlocal pos
        result = parse_factor()
        while pos < len(tokens) and tokens[pos] in ("*", "/"):
            op = tokens[pos]
            pos += 1
            right = parse_factor()
            if op == "/" and right == 0:
                raise ValueError("Division by zero")
            result = _OPS[op](result, right)
        return result

    def parse_factor() -> float:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
        token = tokens[pos]
        if token == "(":
            pos += 1
            result = parse_expr()
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            return result
        elif token == "-":
            pos += 1
            return -parse_factor()
        else:
            try:
                pos += 1
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid token: {token}")

    result = parse_expr()
    return result


def _format_number(n: float) -> str:
    """Format number for display (strip trailing zeros)."""
    if n == int(n):
        return str(int(n))
    return f"{n:.4f}".rstrip("0").rstrip(".")


class MathSkill:
    """Evaluates basic arithmetic expressions safely."""

    name = "math"
    description = "Calculates basic arithmetic expressions"
    patterns = [
        re.compile(r"(?:calculate|compute|solve)\s+(.+)"),
        re.compile(r"(?:what(?:'s| is)?\s+)?(\d+)\s*(?:percent|%)\s*(?:of)\s*(\d+\.?\d*)"),
        re.compile(r"(?:what(?:'s| is)?\s+)?(\d[\d\s\+\-\*\/\.\(\)]*\d)"),
    ]

    async def execute(self, query: str, match: re.Match) -> SkillResult:
        try:
            # Percent pattern
            if match.re == self.patterns[1]:
                pct = float(match.group(1))
                value = float(match.group(2))
                result = (pct / 100) * value
                return SkillResult(
                    success=True,
                    response_text=f"{_format_number(pct)}% of {_format_number(value)} is {_format_number(result)}.",
                    skill_name=self.name,
                )

            # General expression
            expr = match.group(1)
            result = _safe_eval(expr)
            return SkillResult(
                success=True,
                response_text=f"That's {_format_number(result)}.",
                skill_name=self.name,
            )

        except ValueError as e:
            return SkillResult(
                success=False,
                response_text="Sorry, I couldn't calculate that.",
                skill_name=self.name,
                error=str(e),
            )
