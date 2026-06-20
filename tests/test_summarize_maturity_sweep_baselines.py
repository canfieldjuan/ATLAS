from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_maturity_sweep_baselines.py"

SPEC = importlib.util.spec_from_file_location("summarize_maturity_sweep_baselines", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_summaries_orders_by_total_score_then_lane(tmp_path: Path) -> None:
    write_json(
        tmp_path / "baseline_beta.json",
        {
            "beta/a.py": {"score": 5, "counts": {"SWALLOWED_EXCEPT": 1}},
            "beta/b.py": {"score": 3, "counts": {"NO_TEST_FILE": 1}},
        },
    )
    write_json(
        tmp_path / "baseline_alpha.json",
        {"alpha/a.py": {"score": 8, "counts": {"NO_TEST_FILE": 1}}},
    )
    write_json(
        tmp_path / "baseline_gamma.json",
        {"gamma/a.py": {"score": 2, "counts": {"WEAK_CONTRACT": 2}}},
    )

    summaries = MOD.collect_summaries([str(tmp_path / "baseline_*.json")])

    assert [item.lane for item in summaries] == ["alpha", "beta", "gamma"]
    assert summaries[1].files == 2
    assert summaries[1].total_score == 8
    assert summaries[1].top_score == 5
    assert summaries[1].counts == {"NO_TEST_FILE": 1, "SWALLOWED_EXCEPT": 1}


def test_render_text_is_deterministic_and_limits_top_counts(tmp_path: Path) -> None:
    write_json(
        tmp_path / "baseline_lane.json",
        {
            "lane/a.py": {
                "score": 11,
                "counts": {"ZZZ": 2, "AAA": 2, "MID": 1},
            }
        },
    )

    output = MOD.render_text(
        MOD.collect_summaries([str(tmp_path / "baseline_*.json")]),
        top_counts=2,
    )

    assert output.splitlines() == [
        "lane files total_score top_score top_findings",
        "lane 1 11 11 AAA:2,ZZZ:2",
    ]


def test_json_output_uses_same_summary_shape(tmp_path: Path, capsys) -> None:
    write_json(
        tmp_path / "baseline_lane.json",
        {"lane/a.py": {"score": 4, "counts": {"NO_TEST_FILE": 1}}},
    )

    assert MOD.main([str(tmp_path / "baseline_*.json"), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "counts": {"NO_TEST_FILE": 1},
            "files": 1,
            "lane": "lane",
            "path": str(tmp_path / "baseline_lane.json"),
            "top_score": 4,
            "total_score": 4,
        }
    ]


def test_malformed_count_fails_closed(tmp_path: Path, capsys) -> None:
    write_json(
        tmp_path / "baseline_bad.json",
        {"bad/a.py": {"score": 4, "counts": {"NO_TEST_FILE": "one"}}},
    )

    assert MOD.main([str(tmp_path / "baseline_*.json")]) == 1

    assert "counts.NO_TEST_FILE must be an integer" in capsys.readouterr().err


def test_no_matching_baselines_fails_closed(tmp_path: Path, capsys) -> None:
    assert MOD.main([str(tmp_path / "missing_*.json")]) == 1

    assert "no baseline files matched" in capsys.readouterr().err
