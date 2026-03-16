"""认知摩擦测试。"""

import pytest

from aurora.relation.friction import (
    FRICTION_THRESHOLD,
    compute_friction_score,
    should_create_conflict,
)


class TestCognitiveFriction:
    def test_no_overlap(self):
        score = compute_friction_score("用户喜欢苹果", "用户今天买了电脑")
        assert score == 0.0

    def test_high_overlap_no_negation(self):
        score = compute_friction_score("用户喜欢苹果", "用户喜欢橙子")
        assert 0.0 < score < 0.5

    def test_negation_creates_friction(self):
        score = compute_friction_score("用户喜欢苹果", "用户不喜欢苹果")
        assert score > 0.5

    def test_complete_contradiction(self):
        score = compute_friction_score("用户从不喝酒", "用户每天喝酒")
        assert score >= 0.9

    def test_conflict_with_low_confidence(self):
        create, score = should_create_conflict(
            "用户喜欢苹果",
            "用户不喜欢苹果",
            current_confidence=0.3,
        )
        assert not create

    def test_conflict_with_high_confidence(self):
        create, score = should_create_conflict(
            "用户从不喝酒",
            "用户每天喝酒",
            current_confidence=0.8,
        )
        assert create
        assert score > FRICTION_THRESHOLD
