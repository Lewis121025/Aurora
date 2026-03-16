"""数据库迁移模块。

定义 SQLite 数据库 schema，创建所有必需的表和索引：
- turn_events: 转换事件日志
- phase_events: 相位转换日志
- fragments: 记忆片段
- traces: 记忆轨迹
- associations: 片段关联边
- threads: 记忆线程
- knots: 记忆结
- relation_moments: 关系时刻
- relation_formations: 关系形成记录
- orientation_state: 本体定向状态
- metabolic_state: 代谢状态
"""
from __future__ import annotations

import sqlite3


def apply_migrations(connection: sqlite3.Connection) -> None:
    """应用数据库迁移。

    创建所有必需的表和索引（如果不存在）。
    使用 WAL 模式以支持并发读取。

    Args:
        connection: SQLite 数据库连接。
    """
    with connection:
        # 转换事件表：记录用户和 Aurora 的交互
        connection.execute(
            "CREATE TABLE IF NOT EXISTS turn_events("
            "turn_id TEXT PRIMARY KEY, relation_id TEXT NOT NULL, session_id TEXT NOT NULL, "
            "speaker TEXT NOT NULL, text TEXT NOT NULL, created_at REAL NOT NULL)"
        )

        # 相位事件表：记录相位转换历史
        connection.execute(
            "CREATE TABLE IF NOT EXISTS phase_events("
            "transition_id TEXT PRIMARY KEY, from_phase TEXT NOT NULL, to_phase TEXT NOT NULL, "
            "reason TEXT NOT NULL, created_at REAL NOT NULL)"
        )

        # 片段表：记忆基本单位
        connection.execute(
            "CREATE TABLE IF NOT EXISTS fragments("
            "fragment_id TEXT PRIMARY KEY, relation_id TEXT NOT NULL, turn_id TEXT, surface TEXT NOT NULL, "
            "tags TEXT NOT NULL, vividness REAL NOT NULL, salience REAL NOT NULL, "
            "unresolvedness REAL NOT NULL, thread_ids TEXT NOT NULL, knot_ids TEXT NOT NULL, "
            "created_at REAL NOT NULL, last_touched_at REAL NOT NULL, activation_count INTEGER NOT NULL)"
        )

        # 轨迹表：片段的通道记录
        connection.execute(
            "CREATE TABLE IF NOT EXISTS traces("
            "trace_id TEXT PRIMARY KEY, relation_id TEXT NOT NULL, fragment_id TEXT NOT NULL, "
            "channel TEXT NOT NULL, intensity REAL NOT NULL, carry REAL NOT NULL, "
            "created_at REAL NOT NULL, last_touched_at REAL NOT NULL)"
        )

        # 关联表：片段间的边
        connection.execute(
            "CREATE TABLE IF NOT EXISTS associations("
            "edge_id TEXT PRIMARY KEY, src_fragment_id TEXT NOT NULL, dst_fragment_id TEXT NOT NULL, "
            "kind TEXT NOT NULL, weight REAL NOT NULL, evidence TEXT NOT NULL, "
            "created_at REAL NOT NULL, last_touched_at REAL NOT NULL)"
        )

        # 线程表：叙事线索
        connection.execute(
            "CREATE TABLE IF NOT EXISTS threads("
            "thread_id TEXT PRIMARY KEY, relation_id TEXT NOT NULL, fragment_ids TEXT NOT NULL, "
            "dominant_channels TEXT NOT NULL, tension REAL NOT NULL, coherence REAL NOT NULL, "
            "created_at REAL NOT NULL, last_rewoven_at REAL NOT NULL)"
        )

        # 记忆结表：未解决的张力结构
        connection.execute(
            "CREATE TABLE IF NOT EXISTS knots("
            "knot_id TEXT PRIMARY KEY, relation_id TEXT NOT NULL, fragment_ids TEXT NOT NULL, "
            "dominant_channels TEXT NOT NULL, intensity REAL NOT NULL, resolved INTEGER NOT NULL, "
            "created_at REAL NOT NULL, last_rewoven_at REAL NOT NULL)"
        )

        # 关系时刻表：记录每次交互的关系动态
        connection.execute(
            "CREATE TABLE IF NOT EXISTS relation_moments("
            "moment_id TEXT PRIMARY KEY, relation_id TEXT NOT NULL, user_turn_id TEXT NOT NULL, "
            "aurora_turn_id TEXT, user_channels TEXT NOT NULL, aurora_move TEXT NOT NULL, "
            "boundary_event INTEGER NOT NULL, repair_event INTEGER NOT NULL, summary TEXT NOT NULL, "
            "created_at REAL NOT NULL)"
        )

        # 关系形成表：记录关系的长期结构
        connection.execute(
            "CREATE TABLE IF NOT EXISTS relation_formations("
            "relation_id TEXT PRIMARY KEY, thread_ids TEXT NOT NULL, knot_ids TEXT NOT NULL, "
            "boundary_events INTEGER NOT NULL, repair_events INTEGER NOT NULL, "
            "resonance_events INTEGER NOT NULL, last_contact_at REAL NOT NULL)"
        )

        # 定向状态表：本体定向（单行）
        connection.execute(
            "CREATE TABLE IF NOT EXISTS orientation_state("
            "id INTEGER PRIMARY KEY CHECK(id = 1), self_evidence TEXT NOT NULL, "
            "world_evidence TEXT NOT NULL, relation_evidence TEXT NOT NULL, "
            "anchor_thread_ids TEXT NOT NULL, active_knot_ids TEXT NOT NULL, "
            "last_updated_at REAL NOT NULL)"
        )

        # 代谢状态表：代谢状态（单行）
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metabolic_state("
            "id INTEGER PRIMARY KEY CHECK(id = 1), phase TEXT NOT NULL, sleep_need REAL NOT NULL, "
            "active_relation_ids TEXT NOT NULL, active_knot_ids TEXT NOT NULL, "
            "pending_sleep_relation_ids TEXT NOT NULL, last_transition_at REAL NOT NULL)"
        )

        # 创建索引加速查询
        connection.execute("CREATE INDEX IF NOT EXISTS idx_fragments_relation ON fragments(relation_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_traces_fragment ON traces(fragment_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_traces_relation ON traces(relation_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_associations_src ON associations(src_fragment_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_associations_dst ON associations(dst_fragment_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_threads_relation ON threads(relation_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_knots_relation ON knots(relation_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_moments_relation ON relation_moments(relation_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_turns_relation ON turn_events(relation_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_phase_events_created ON phase_events(created_at)")
