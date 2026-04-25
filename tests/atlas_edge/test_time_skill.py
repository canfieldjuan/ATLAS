"""
Tests for edge TimeSkill formatting behavior.
"""

from datetime import datetime

from atlas_edge.skills.time_skill import TimeSkill


class TestTimeSkillFormatting:
    """Validate platform-independent formatting helpers."""

    def test_format_date_avoids_platform_specific_directives(self):
        now = datetime(2026, 4, 5, 9, 7)
        assert TimeSkill._format_date(now) == "Today is Sunday, April 5, 2026."

    def test_format_time_midnight(self):
        now = datetime(2026, 4, 5, 0, 7)
        assert TimeSkill._format_time(now) == "It's 12:07 AM."

    def test_format_time_afternoon(self):
        now = datetime(2026, 4, 5, 15, 45)
        assert TimeSkill._format_time(now) == "It's 3:45 PM."
