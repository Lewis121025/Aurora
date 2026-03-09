"""
AURORA 时间索引模块
====================

将时间作为一等公民的索引与访问能力。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from aurora.core.models.plot import Plot


class TemporalMemoryMixin:
    """提供基于时间的一等公民索引与检索辅助方法。"""

    def _get_day_bucket(self, ts: float) -> int:
        """将时间戳转换为用于时间索引的日期 bucket。"""
        return int(ts // 86400)

    def _add_to_temporal_index(self, plot: Plot) -> None:
        """将 plot 添加到时间索引。"""
        day_bucket = self._get_day_bucket(plot.ts)

        if day_bucket not in self._temporal_index:
            self._temporal_index[day_bucket] = []

        self._temporal_index[day_bucket].append(plot.id)

        if not self._temporal_index_min_bucket or day_bucket < self._temporal_index_min_bucket:
            self._temporal_index_min_bucket = day_bucket
        if not self._temporal_index_max_bucket or day_bucket > self._temporal_index_max_bucket:
            self._temporal_index_max_bucket = day_bucket

    def _remove_from_temporal_index(self, plot: Plot) -> None:
        """从时间索引中移除 plot。"""
        day_bucket = self._get_day_bucket(plot.ts)
        if day_bucket in self._temporal_index:
            try:
                self._temporal_index[day_bucket].remove(plot.id)
                if not self._temporal_index[day_bucket]:
                    del self._temporal_index[day_bucket]
            except ValueError:
                pass

    def get_plots_in_time_range(
        self,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: int = 100,
    ) -> List[str]:
        """获取时间范围内的 plot ID。"""
        if not self._temporal_index:
            return []

        start_bucket = self._get_day_bucket(start_ts) if start_ts else self._temporal_index_min_bucket
        end_bucket = self._get_day_bucket(end_ts) if end_ts else self._temporal_index_max_bucket

        plot_ids: List[str] = []
        for bucket in range(start_bucket, end_bucket + 1):
            if bucket in self._temporal_index:
                plot_ids.extend(self._temporal_index[bucket])

        if start_ts is not None or end_ts is not None:
            filtered: List[Tuple[float, str]] = []
            for pid in plot_ids:
                plot = self.plots.get(pid)
                if plot is None:
                    continue
                if start_ts is not None and plot.ts < start_ts:
                    continue
                if end_ts is not None and plot.ts > end_ts:
                    continue
                filtered.append((plot.ts, pid))

            filtered.sort(key=lambda x: x[0])
            return [pid for _, pid in filtered[:limit]]

        plot_ids_with_ts = [(self.plots[pid].ts, pid) for pid in plot_ids if pid in self.plots]
        plot_ids_with_ts.sort(key=lambda x: x[0])
        return [pid for _, pid in plot_ids_with_ts[:limit]]

    def get_recent_plots(self, n: int = 10) -> List[str]:
        """获取 N 个最近的 plot ID。"""
        if not self._temporal_index:
            return []

        plot_ids: List[Tuple[float, str]] = []
        bucket = self._temporal_index_max_bucket

        while bucket >= self._temporal_index_min_bucket and len(plot_ids) < n * 2:
            if bucket in self._temporal_index:
                for pid in self._temporal_index[bucket]:
                    plot = self.plots.get(pid)
                    if plot:
                        plot_ids.append((plot.ts, pid))
            bucket -= 1

        plot_ids.sort(key=lambda x: -x[0])
        return [pid for _, pid in plot_ids[:n]]

    def get_earliest_plots(self, n: int = 10) -> List[str]:
        """获取 N 个最早的 plot ID。"""
        if not self._temporal_index:
            return []

        plot_ids: List[Tuple[float, str]] = []
        bucket = self._temporal_index_min_bucket

        while bucket <= self._temporal_index_max_bucket and len(plot_ids) < n * 2:
            if bucket in self._temporal_index:
                for pid in self._temporal_index[bucket]:
                    plot = self.plots.get(pid)
                    if plot:
                        plot_ids.append((plot.ts, pid))
            bucket += 1

        plot_ids.sort(key=lambda x: x[0])
        return [pid for _, pid in plot_ids[:n]]

    def get_temporal_statistics(self) -> Dict[str, Any]:
        """获取关于记忆时间分布的统计信息。"""
        if not self._temporal_index:
            return {
                "total_days": 0,
                "earliest_ts": None,
                "latest_ts": None,
                "avg_plots_per_day": 0.0,
                "most_active_day": None,
            }

        import datetime

        total_days = len(self._temporal_index)
        total_plots = sum(len(pids) for pids in self._temporal_index.values())

        most_active_bucket = max(self._temporal_index, key=lambda b: len(self._temporal_index[b]))
        most_active_count = len(self._temporal_index[most_active_bucket])
        most_active_date = datetime.datetime.fromtimestamp(most_active_bucket * 86400)

        earliest_ts = self._temporal_index_min_bucket * 86400 if self._temporal_index_min_bucket else None
        latest_ts = (self._temporal_index_max_bucket + 1) * 86400 - 1 if self._temporal_index_max_bucket else None

        return {
            "total_days": total_days,
            "earliest_ts": earliest_ts,
            "latest_ts": latest_ts,
            "avg_plots_per_day": total_plots / total_days if total_days > 0 else 0.0,
            "most_active_day": {
                "date": most_active_date.strftime("%Y-%m-%d"),
                "count": most_active_count,
            },
        }
