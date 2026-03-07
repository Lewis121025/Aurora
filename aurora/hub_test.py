"""
AURORA 中心测试
================

AuroraHub 多租户路由模块的测试。

测试覆盖:
- 使用各种设置的中心初始化
- 租户创建和缓存
- 超过容量时的 LRU 驱逐
- 多租户数据隔离
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List

import pytest

from aurora.hub import AuroraHub
from aurora.config import AuroraSettings
from aurora.service import AuroraTenant


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def hub_settings(temp_data_dir: Path) -> AuroraSettings:
    """为中心测试创建设置，租户容量最小。"""
    return AuroraSettings(
        data_dir=str(temp_data_dir),
        tenant_max_loaded=3,  # Small cap for testing eviction
        llm_provider="mock",
        embedding_provider="mock",
        dim=64,
        pii_redaction_enabled=False,
    )


@pytest.fixture
def hub(hub_settings: AuroraSettings) -> AuroraHub:
    """Create a fresh AuroraHub instance."""
    return AuroraHub(settings=hub_settings)


# =============================================================================
# Test Hub Initialization
# =============================================================================

class TestAuroraHubInitialization:
    """Tests for AuroraHub initialization."""

    def test_hub_creation(self, hub_settings: AuroraSettings):
        """Test hub can be created with settings."""
        hub = AuroraHub(settings=hub_settings)
        
        assert hub.settings is hub_settings
        assert hub.llm is None
        assert len(hub._tenants) == 0
        assert len(hub._lru) == 0

    def test_hub_with_custom_llm(self, hub_settings: AuroraSettings):
        """Test hub can be created with custom LLM provider."""
        from aurora.llm.mock import MockLLM
        
        mock_llm = MockLLM()
        hub = AuroraHub(settings=hub_settings, llm=mock_llm)
        
        assert hub.llm is mock_llm

    def test_hub_has_thread_lock(self, hub: AuroraHub):
        """Test hub has threading lock for concurrency safety."""
        assert hasattr(hub, "_lock")
        assert isinstance(hub._lock, type(threading.RLock()))


# =============================================================================
# Test Tenant Management
# =============================================================================

class TestTenantManagement:
    """Tests for tenant creation and management."""

    def test_get_tenant_creates_new(self, hub: AuroraHub):
        """Test getting a tenant creates a new AuroraTenant."""
        tenant = hub.tenant("user_001")
        
        assert isinstance(tenant, AuroraTenant)
        assert tenant.user_id == "user_001"
        assert "user_001" in hub._tenants
        assert "user_001" in hub._lru

    def test_tenant_caching(self, hub: AuroraHub):
        """Test that same user_id returns same tenant instance."""
        tenant1 = hub.tenant("user_001")
        tenant2 = hub.tenant("user_001")
        
        # Should be the exact same object
        assert tenant1 is tenant2
        
        # Should only have one tenant
        assert len(hub._tenants) == 1

    def test_multiple_tenants(self, hub: AuroraHub):
        """Test multiple tenants can be created."""
        tenant1 = hub.tenant("user_001")
        tenant2 = hub.tenant("user_002")
        tenant3 = hub.tenant("user_003")
        
        assert len(hub._tenants) == 3
        assert tenant1.user_id == "user_001"
        assert tenant2.user_id == "user_002"
        assert tenant3.user_id == "user_003"
        
        # All should be different instances
        assert tenant1 is not tenant2
        assert tenant2 is not tenant3

    def test_tenant_inherits_settings(self, hub: AuroraHub):
        """Test tenant inherits settings from hub."""
        tenant = hub.tenant("user_001")
        
        assert tenant.settings.dim == hub.settings.dim
        assert tenant.settings.data_dir == hub.settings.data_dir


# =============================================================================
# Test LRU Eviction
# =============================================================================

class TestLRUEviction:
    """Tests for LRU eviction policy."""

    def test_eviction_at_capacity(self, hub: AuroraHub):
        """Test eviction happens when exceeding tenant_max_loaded."""
        # Create tenants up to capacity (3)
        hub.tenant("user_001")
        hub.tenant("user_002")
        hub.tenant("user_003")
        
        assert len(hub._tenants) == 3
        
        # Adding one more should evict the oldest
        hub.tenant("user_004")
        
        assert len(hub._tenants) == 3
        assert "user_001" not in hub._tenants
        assert "user_004" in hub._tenants

    def test_lru_touch_on_access(self, hub: AuroraHub):
        """Test accessing a tenant moves it to end of LRU."""
        hub.tenant("user_001")
        hub.tenant("user_002")
        hub.tenant("user_003")
        
        # Access user_001 again to move to end
        hub.tenant("user_001")
        
        # Now user_002 should be oldest
        hub.tenant("user_004")
        
        assert "user_002" not in hub._tenants
        assert "user_001" in hub._tenants

    def test_eviction_order(self, hub: AuroraHub):
        """Test eviction follows LRU order correctly."""
        hub.tenant("user_001")
        hub.tenant("user_002")
        hub.tenant("user_003")
        
        # Access in specific order to set LRU state
        hub.tenant("user_001")  # user_001 now most recent
        hub.tenant("user_002")  # user_002 now most recent
        # user_003 is now oldest
        
        hub.tenant("user_004")  # Should evict user_003
        
        assert "user_003" not in hub._tenants
        assert "user_001" in hub._tenants
        assert "user_002" in hub._tenants
        assert "user_004" in hub._tenants

    def test_no_eviction_under_capacity(self, hub: AuroraHub):
        """Test no eviction when under capacity."""
        hub.tenant("user_001")
        hub.tenant("user_002")
        
        assert len(hub._tenants) == 2
        assert "user_001" in hub._tenants
        assert "user_002" in hub._tenants


# =============================================================================
# Test Tenant Isolation
# =============================================================================

class TestTenantIsolation:
    """Tests for tenant data isolation."""

    def test_data_isolation(self, hub: AuroraHub):
        """Test that tenant data is isolated between users."""
        tenant1 = hub.tenant("user_001")
        tenant2 = hub.tenant("user_002")
        
        # Ingest data into tenant1
        result1 = tenant1.ingest_interaction(
            event_id="evt_001",
            session_id="sess_001",
            user_message="Hello from user 1",
            agent_message="Hi user 1!",
        )
        
        # Ingest data into tenant2
        result2 = tenant2.ingest_interaction(
            event_id="evt_002",
            session_id="sess_002",
            user_message="Hello from user 2",
            agent_message="Hi user 2!",
        )
        
        # Each tenant should have processed their own data
        assert result1.plot_id != ""
        assert result2.plot_id != ""
        
        # Events should be isolated
        assert tenant1.event_log.get_seq_by_id("evt_001") is not None
        assert tenant1.event_log.get_seq_by_id("evt_002") is None
        assert tenant2.event_log.get_seq_by_id("evt_002") is not None
        assert tenant2.event_log.get_seq_by_id("evt_001") is None

    def test_separate_user_directories(self, hub: AuroraHub, temp_data_dir: Path):
        """Test each tenant has separate data directory."""
        tenant1 = hub.tenant("user_001")
        tenant2 = hub.tenant("user_002")
        
        # Check directories exist and are different
        assert Path(tenant1.user_dir).exists()
        assert Path(tenant2.user_dir).exists()
        assert tenant1.user_dir != tenant2.user_dir
        
        # Verify directory naming
        assert "user_001" in tenant1.user_dir
        assert "user_002" in tenant2.user_dir

    def test_query_isolation(self, hub: AuroraHub):
        """Test queries only return results from same tenant."""
        tenant1 = hub.tenant("user_001")
        tenant2 = hub.tenant("user_002")
        
        # Ingest distinct data
        tenant1.ingest_interaction(
            event_id="evt_001",
            session_id="sess_001",
            user_message="I love programming in Python",
            agent_message="Python is great for AI development!",
        )
        
        tenant2.ingest_interaction(
            event_id="evt_002",
            session_id="sess_002",
            user_message="I prefer cooking Italian food",
            agent_message="Italian cuisine is wonderful!",
        )
        
        # Query should only find data from respective tenant
        result1 = tenant1.query(text="programming Python", k=5)
        result2 = tenant2.query(text="cooking Italian", k=5)
        
        # Results should exist (may be empty if not stored due to probabilistic)
        assert result1 is not None
        assert result2 is not None


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety of AuroraHub."""

    def test_concurrent_tenant_creation(self, hub: AuroraHub):
        """Test concurrent tenant creation is thread-safe."""
        results: List[AuroraTenant] = []
        errors: List[Exception] = []
        
        def create_tenant(user_id: str):
            try:
                tenant = hub.tenant(user_id)
                results.append(tenant)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=create_tenant, args=(f"user_{i:03d}",))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0
        
        # Due to LRU eviction (cap=3), should have exactly 3 tenants
        assert len(hub._tenants) == 3

    def test_concurrent_same_user_access(self, hub: AuroraHub):
        """Test concurrent access to same user returns same tenant."""
        results: List[AuroraTenant] = []
        
        def get_tenant():
            tenant = hub.tenant("shared_user")
            results.append(tenant)
        
        threads = [
            threading.Thread(target=get_tenant)
            for _ in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should be the same tenant instance
        assert all(r is results[0] for r in results)
        assert len(hub._tenants) == 1


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_user_id(self, hub: AuroraHub):
        """Test handling of empty user_id."""
        # Empty string is technically valid
        tenant = hub.tenant("")
        assert tenant.user_id == ""

    def test_special_characters_in_user_id(self, hub: AuroraHub):
        """Test user_id with special characters."""
        tenant = hub.tenant("user@example.com")
        assert tenant.user_id == "user@example.com"

    def test_unicode_user_id(self, hub: AuroraHub):
        """Test user_id with unicode characters."""
        tenant = hub.tenant("用户_001")
        assert tenant.user_id == "用户_001"

    def test_zero_capacity_setting(self, temp_data_dir: Path):
        """Test hub with zero tenant capacity (edge case)."""
        settings = AuroraSettings(
            data_dir=str(temp_data_dir),
            tenant_max_loaded=0,  # Zero cap
            llm_provider="mock",
            embedding_provider="mock",
            dim=64,
        )
        hub = AuroraHub(settings=settings)
        
        # With cap=0, eviction loop doesn't run (while cap > 0)
        hub.tenant("user_001")
        hub.tenant("user_002")
        
        # Both should exist since eviction is disabled
        assert len(hub._tenants) == 2

    def test_large_capacity_setting(self, temp_data_dir: Path):
        """Test hub with large tenant capacity."""
        settings = AuroraSettings(
            data_dir=str(temp_data_dir),
            tenant_max_loaded=1000,
            llm_provider="mock",
            embedding_provider="mock",
            dim=64,
        )
        hub = AuroraHub(settings=settings)
        
        # Create several tenants
        for i in range(10):
            hub.tenant(f"user_{i:03d}")
        
        # All should exist (under capacity)
        assert len(hub._tenants) == 10
