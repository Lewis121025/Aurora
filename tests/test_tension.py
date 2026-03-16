"""张力队列测试。"""

import pytest

from aurora.relation.tension import TensionQueue, TensionItem


class TestTensionQueue:
    def test_empty_queue(self):
        queue = TensionQueue()
        assert len(queue) == 0

    def test_push_and_peek(self):
        queue = TensionQueue()
        queue.push("话题1", 0.8, 24.0, "提示1", 1000.0)

        item = queue.peek(1000.0)
        assert item is not None
        assert item.topic == "话题1"
        assert item.urgency == 0.8

    def test_pop_returns_highest_urgency(self):
        queue = TensionQueue()
        queue.push("低优先级", 0.3, 24.0, "提示", 1000.0)
        queue.push("高优先级", 0.9, 24.0, "提示", 1000.0)

        item = queue.pop(1000.0)
        assert item.topic == "高优先级"

    def test_decay_over_time(self):
        queue = TensionQueue()
        queue.push("话题", 1.0, 1.0, "提示", 0.0)

        item = queue.peek(3600.0)
        assert item.current_urgency(3600.0) < 1.0
        assert item.current_urgency(3600.0) > 0.4

    def test_to_prompt_segment_empty(self):
        queue = TensionQueue()
        segment = queue.to_prompt_segment(1000.0)
        assert "无悬案" in segment

    def test_to_prompt_segment_with_items(self):
        queue = TensionQueue()
        queue.push("话题1", 0.8, 24.0, "提示1", 1000.0)

        segment = queue.to_prompt_segment(1000.0)
        assert "话题1" in segment
        assert "提示1" in segment
