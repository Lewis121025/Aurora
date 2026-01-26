#!/usr/bin/env python3
"""
Aurora Migration Script: SQLite -> PostgreSQL
==============================================

Migrates Aurora data from SQLite-based storage to PostgreSQL + pgvector.

This script handles:
1. Event log migration (SQLite -> PostgreSQL)
2. Document store migration (SQLite -> PostgreSQL)
3. State/snapshot migration (pickle -> JSON in PostgreSQL)
4. Vector index migration (in-memory/pickle -> pgvector)

Usage:
    # Dry run (preview changes)
    python -m aurora.scripts.migrate_to_postgres --dry-run
    
    # Full migration
    python -m aurora.scripts.migrate_to_postgres \
        --data-dir ./data \
        --postgres-dsn postgresql://user:pass@localhost/aurora
    
    # Migrate specific user
    python -m aurora.scripts.migrate_to_postgres \
        --data-dir ./data \
        --user-id user_123 \
        --postgres-dsn postgresql://...

Requirements:
    pip install asyncpg pgvector aiosqlite

Schema (run before migration):
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Event log
    CREATE TABLE aurora_events (
        seq SERIAL PRIMARY KEY,
        id TEXT UNIQUE NOT NULL,
        ts DOUBLE PRECISION NOT NULL,
        user_id TEXT NOT NULL,
        session_id TEXT,
        type TEXT NOT NULL,
        payload JSONB NOT NULL,
        migrated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX idx_aurora_events_user ON aurora_events (user_id);
    CREATE INDEX idx_aurora_events_user_seq ON aurora_events (user_id, seq);
    
    -- Document store
    CREATE TABLE aurora_docs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        user_id TEXT NOT NULL,
        ts DOUBLE PRECISION NOT NULL,
        body JSONB NOT NULL,
        migrated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX idx_aurora_docs_user ON aurora_docs (user_id);
    CREATE INDEX idx_aurora_docs_kind ON aurora_docs (kind);
    
    -- Vectors (pgvector)
    CREATE TABLE aurora_vectors (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        embedding vector(1024),
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX idx_aurora_vectors_hnsw ON aurora_vectors 
        USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
    CREATE INDEX idx_aurora_vectors_user ON aurora_vectors (user_id);
    
    -- State store
    CREATE TABLE aurora_states (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(user_id, key)
    );
    CREATE INDEX idx_aurora_states_user ON aurora_states (user_id);
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pickle
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class MigrationStats:
    """Statistics for a migration run."""
    events_migrated: int = 0
    docs_migrated: int = 0
    vectors_migrated: int = 0
    states_migrated: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class AuroraMigrator:
    """Handles migration from SQLite to PostgreSQL."""
    
    def __init__(
        self,
        data_dir: str,
        postgres_dsn: str,
        dim: int = 1024,
        dry_run: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.postgres_dsn = postgres_dsn
        self.dim = dim
        self.dry_run = dry_run
        
        self._pg_pool = None
    
    async def _ensure_postgres(self):
        """Create PostgreSQL connection pool."""
        if self._pg_pool is None:
            import asyncpg
            self._pg_pool = await asyncpg.create_pool(
                self.postgres_dsn,
                min_size=2,
                max_size=10,
            )
    
    async def _init_schema(self):
        """Initialize PostgreSQL schema."""
        await self._ensure_postgres()
        
        async with self._pg_pool.acquire() as conn:
            # Enable pgvector
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS aurora_events (
                    seq SERIAL PRIMARY KEY,
                    id TEXT UNIQUE NOT NULL,
                    ts DOUBLE PRECISION NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    type TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    migrated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_aurora_events_user 
                ON aurora_events (user_id)
            """)
            
            # Documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS aurora_docs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    ts DOUBLE PRECISION NOT NULL,
                    body JSONB NOT NULL,
                    migrated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_aurora_docs_user 
                ON aurora_docs (user_id)
            """)
            
            # Vectors table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS aurora_vectors (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    embedding vector({self.dim}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_aurora_vectors_hnsw 
                ON aurora_vectors 
                USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_aurora_vectors_user 
                ON aurora_vectors (user_id)
            """)
            
            # States table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS aurora_states (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, key)
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_aurora_states_user 
                ON aurora_states (user_id)
            """)
    
    async def migrate_user(self, user_id: str) -> MigrationStats:
        """Migrate all data for a single user."""
        stats = MigrationStats()
        user_dir = self.data_dir / f"user_{user_id}"
        
        if not user_dir.exists():
            logger.warning(f"User directory not found: {user_dir}")
            return stats
        
        logger.info(f"Migrating user: {user_id}")
        
        # Migrate events
        events_db = user_dir / "events.sqlite3"
        if events_db.exists():
            try:
                count = await self._migrate_events(user_id, events_db)
                stats.events_migrated = count
                logger.info(f"  Migrated {count} events")
            except Exception as e:
                stats.errors.append(f"Events migration failed: {e}")
                logger.error(f"  Events migration failed: {e}")
        
        # Migrate documents
        docs_db = user_dir / "docs.sqlite3"
        if docs_db.exists():
            try:
                count = await self._migrate_docs(user_id, docs_db)
                stats.docs_migrated = count
                logger.info(f"  Migrated {count} documents")
            except Exception as e:
                stats.errors.append(f"Docs migration failed: {e}")
                logger.error(f"  Docs migration failed: {e}")
        
        # Migrate snapshots (extract vectors and state)
        snapshots_dir = user_dir / "snapshots"
        if snapshots_dir.exists():
            try:
                vec_count, state_count = await self._migrate_snapshots(user_id, snapshots_dir)
                stats.vectors_migrated = vec_count
                stats.states_migrated = state_count
                logger.info(f"  Migrated {vec_count} vectors, {state_count} states from snapshots")
            except Exception as e:
                stats.errors.append(f"Snapshots migration failed: {e}")
                logger.error(f"  Snapshots migration failed: {e}")
        
        return stats
    
    async def _migrate_events(self, user_id: str, db_path: Path) -> int:
        """Migrate events from SQLite to PostgreSQL."""
        if self.dry_run:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        
        await self._ensure_postgres()
        
        # Read from SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("""
            SELECT id, ts, user_id, session_id, type, payload FROM events ORDER BY seq
        """)
        
        count = 0
        batch = []
        
        for row in cursor:
            event_id, ts, uid, session_id, event_type, payload_json = row
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                payload = {"raw": payload_json}
            
            batch.append((event_id, ts, uid, session_id, event_type, json.dumps(payload)))
            
            if len(batch) >= 100:
                await self._insert_events_batch(batch)
                count += len(batch)
                batch = []
        
        if batch:
            await self._insert_events_batch(batch)
            count += len(batch)
        
        conn.close()
        return count
    
    async def _insert_events_batch(self, batch: List[Tuple]) -> None:
        """Insert batch of events into PostgreSQL."""
        async with self._pg_pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO aurora_events (id, ts, user_id, session_id, type, payload)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
            """, batch)
    
    async def _migrate_docs(self, user_id: str, db_path: Path) -> int:
        """Migrate documents from SQLite to PostgreSQL."""
        if self.dry_run:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM docs")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        
        await self._ensure_postgres()
        
        # Read from SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("""
            SELECT id, kind, user_id, ts, body FROM docs
        """)
        
        count = 0
        batch = []
        
        for row in cursor:
            doc_id, kind, uid, ts, body_json = row
            try:
                body = json.loads(body_json)
            except json.JSONDecodeError:
                body = {"raw": body_json}
            
            batch.append((doc_id, kind, uid, ts, json.dumps(body)))
            
            if len(batch) >= 100:
                await self._insert_docs_batch(batch)
                count += len(batch)
                batch = []
        
        if batch:
            await self._insert_docs_batch(batch)
            count += len(batch)
        
        conn.close()
        return count
    
    async def _insert_docs_batch(self, batch: List[Tuple]) -> None:
        """Insert batch of documents into PostgreSQL."""
        async with self._pg_pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO aurora_docs (id, kind, user_id, ts, body)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    body = EXCLUDED.body
            """, batch)
    
    async def _migrate_snapshots(self, user_id: str, snapshots_dir: Path) -> Tuple[int, int]:
        """Migrate snapshots (extract vectors and state)."""
        vec_count = 0
        state_count = 0
        
        # Find latest snapshot
        snapshot_files = list(snapshots_dir.glob("snapshot_*.pkl"))
        if not snapshot_files:
            return vec_count, state_count
        
        # Get highest seq number
        latest_file = max(
            snapshot_files,
            key=lambda f: int(f.stem.split("_")[1])
        )
        
        if self.dry_run:
            logger.info(f"  Would migrate from: {latest_file}")
            return 0, 0
        
        await self._ensure_postgres()
        
        # Load pickle snapshot
        try:
            with open(latest_file, "rb") as f:
                snapshot = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load snapshot {latest_file}: {e}")
            return vec_count, state_count
        
        # Extract AuroraMemory state
        memory = snapshot.state if hasattr(snapshot, "state") else snapshot
        
        if hasattr(memory, "to_state_dict"):
            # Use new serialization method
            state_dict = memory.to_state_dict()
        else:
            # Legacy: manual extraction
            state_dict = self._extract_legacy_state(memory)
        
        # Migrate vectors from state dict
        vec_count = await self._migrate_vectors_from_state(user_id, state_dict)
        
        # Save state dict to PostgreSQL
        async with self._pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO aurora_states (user_id, key, value)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
            """, user_id, "memory_state", json.dumps(state_dict))
            state_count = 1
        
        return vec_count, state_count
    
    def _extract_legacy_state(self, memory) -> Dict[str, Any]:
        """Extract state from legacy AuroraMemory without to_state_dict."""
        state = {
            "version": 1,
            "plots": {},
            "stories": {},
            "themes": {},
        }
        
        if hasattr(memory, "plots"):
            for pid, plot in memory.plots.items():
                state["plots"][pid] = {
                    "id": plot.id,
                    "ts": plot.ts,
                    "text": plot.text,
                    "actors": list(plot.actors),
                    "embedding": plot.embedding.tolist() if hasattr(plot.embedding, "tolist") else [],
                    "story_id": plot.story_id,
                    "status": plot.status,
                }
        
        if hasattr(memory, "stories"):
            for sid, story in memory.stories.items():
                state["stories"][sid] = {
                    "id": story.id,
                    "created_ts": story.created_ts,
                    "updated_ts": story.updated_ts,
                    "plot_ids": story.plot_ids,
                    "centroid": story.centroid.tolist() if story.centroid is not None and hasattr(story.centroid, "tolist") else None,
                    "status": story.status,
                }
        
        if hasattr(memory, "themes"):
            for tid, theme in memory.themes.items():
                state["themes"][tid] = {
                    "id": theme.id,
                    "created_ts": theme.created_ts,
                    "updated_ts": theme.updated_ts,
                    "story_ids": theme.story_ids,
                    "prototype": theme.prototype.tolist() if theme.prototype is not None and hasattr(theme.prototype, "tolist") else None,
                }
        
        return state
    
    async def _migrate_vectors_from_state(self, user_id: str, state_dict: Dict) -> int:
        """Extract and migrate vectors from state dict to pgvector."""
        vectors = []
        
        # Extract plot vectors
        for pid, plot in state_dict.get("plots", {}).items():
            embedding = plot.get("embedding")
            if embedding:
                vectors.append((
                    pid,
                    user_id,
                    "plot",
                    embedding,
                    json.dumps({"story_id": plot.get("story_id")}),
                ))
        
        # Extract story vectors (centroids)
        for sid, story in state_dict.get("stories", {}).items():
            centroid = story.get("centroid")
            if centroid:
                vectors.append((
                    sid,
                    user_id,
                    "story",
                    centroid,
                    json.dumps({"plot_count": len(story.get("plot_ids", []))}),
                ))
        
        # Extract theme vectors (prototypes)
        for tid, theme in state_dict.get("themes", {}).items():
            prototype = theme.get("prototype")
            if prototype:
                vectors.append((
                    tid,
                    user_id,
                    "theme",
                    prototype,
                    json.dumps({"story_count": len(theme.get("story_ids", []))}),
                ))
        
        # Batch insert
        if vectors:
            async with self._pg_pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO aurora_vectors (id, user_id, kind, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """, vectors)
        
        return len(vectors)
    
    async def migrate_all(self) -> Dict[str, MigrationStats]:
        """Migrate all users."""
        if not self.dry_run:
            await self._init_schema()
        
        results = {}
        
        # Find all user directories
        for user_dir in self.data_dir.glob("user_*"):
            if user_dir.is_dir():
                user_id = user_dir.name.replace("user_", "")
                try:
                    stats = await self.migrate_user(user_id)
                    results[user_id] = stats
                except Exception as e:
                    logger.error(f"Failed to migrate user {user_id}: {e}")
                    results[user_id] = MigrationStats(errors=[str(e)])
        
        return results
    
    async def close(self):
        """Close connections."""
        if self._pg_pool:
            await self._pg_pool.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate Aurora data from SQLite to PostgreSQL"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Path to Aurora data directory",
    )
    parser.add_argument(
        "--postgres-dsn",
        required=True,
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--user-id",
        help="Migrate specific user (default: all users)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Vector dimension",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without migrating",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    migrator = AuroraMigrator(
        data_dir=args.data_dir,
        postgres_dsn=args.postgres_dsn,
        dim=args.dim,
        dry_run=args.dry_run,
    )
    
    try:
        if args.user_id:
            stats = await migrator.migrate_user(args.user_id)
            results = {args.user_id: stats}
        else:
            results = await migrator.migrate_all()
        
        # Print summary
        print("\n" + "=" * 50)
        print("Migration Summary" + (" (DRY RUN)" if args.dry_run else ""))
        print("=" * 50)
        
        total_events = 0
        total_docs = 0
        total_vectors = 0
        total_states = 0
        total_errors = 0
        
        for user_id, stats in results.items():
            print(f"\nUser: {user_id}")
            print(f"  Events:  {stats.events_migrated}")
            print(f"  Docs:    {stats.docs_migrated}")
            print(f"  Vectors: {stats.vectors_migrated}")
            print(f"  States:  {stats.states_migrated}")
            if stats.errors:
                print(f"  Errors:  {len(stats.errors)}")
                for err in stats.errors:
                    print(f"    - {err}")
            
            total_events += stats.events_migrated
            total_docs += stats.docs_migrated
            total_vectors += stats.vectors_migrated
            total_states += stats.states_migrated
            total_errors += len(stats.errors)
        
        print("\n" + "-" * 50)
        print(f"Total: {len(results)} users")
        print(f"  Events:  {total_events}")
        print(f"  Docs:    {total_docs}")
        print(f"  Vectors: {total_vectors}")
        print(f"  States:  {total_states}")
        print(f"  Errors:  {total_errors}")
        
    finally:
        await migrator.close()


if __name__ == "__main__":
    asyncio.run(main())
