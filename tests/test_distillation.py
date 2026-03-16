"""蒸馏管道测试。"""

import pytest

from aurora.relation.state import RelationalState


class TestRelationalState:
    def test_default_values(self):
        state = RelationalState()
        assert state.intimacy_level == 5
        assert state.current_vibe == "中性"
        assert state.interaction_rules == []
        assert state.last_distilled_at == 0.0

    def test_to_prompt_segment(self):
        state = RelationalState()
        state.intimacy_level = 8
        state.current_vibe = "高压"
        state.interaction_rules = ["用户讨厌废话"]

        segment = state.to_prompt_segment()
        assert "intimacy_level: 8" in segment
        assert 'current_vibe: "高压"' in segment
        assert "用户讨厌废话" in segment

    def test_apply_patch_intimacy(self):
        state = RelationalState()
        state.intimacy_level = 5
        state.apply_patch(intimacy_delta=3)
        assert state.intimacy_level == 8

        state.apply_patch(intimacy_delta=-10)
        assert state.intimacy_level == 0

    def test_apply_patch_vibe(self):
        state = RelationalState()
        state.apply_patch(vibe="轻松")
        assert state.current_vibe == "轻松"

    def test_apply_patch_rules(self):
        state = RelationalState()
        state.apply_patch(new_rules=["规则1", "规则2"])
        assert len(state.interaction_rules) == 2

        state.apply_patch(new_rules=["规则1", "规则3"])
        assert len(state.interaction_rules) == 3
        assert "规则1" in state.interaction_rules
        assert "规则3" in state.interaction_rules
