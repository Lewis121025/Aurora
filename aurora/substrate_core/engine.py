from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np

from aurora.core_math.contracts import (
    CollapseRequest, FeedbackEnvelope, HealthEnvelope,
    InputEnvelope, IntegrityEnvelope, ReleasedTrace, SubstrateInputResult,
    SubstrateWakeResult, WakeEnvelope,
)
from aurora.core_math.dynamics import (
    advance_latent_ou_twin, boundary_budget, prediction_error,
    update_metric, verbosity_budget,
)
from aurora.core_math.encoding import Encoder, HashingEncoder, SemanticEncoder
from aurora.core_math.state import (
    ArrivalState, LatentState, MetricState, SEALED_STATE_VERSION,
    SealedState, SealedStateHeader, Spark,
    isoformat_utc, normalize, parse_utc, seal_state, unseal_state, utc_now,
)
from aurora.core_math.wake import (
    advance_arrival, observe_internal_action,
    observe_user_contact, sample_next_wake,
)


@dataclass(frozen=True)
class AuroraSubstrateConfig:
    latent_dim: int = 512  # Updated to match bge-small-zh-v1.5
    metric_rank: int = 8
    rng_seed: int = 7
    sample_count: int = 10  # 扩大提取窗口，让统一方程自然筛选出工作、情景与化石记忆
    capacity: int = 4096  # 容量扩充：从 1024 扩展为 4096，为化石记忆提供足够的沉积空间


class AuroraSubstrateCore:
    def __init__(self, config: AuroraSubstrateConfig | None = None, encoder: Encoder | None = None) -> None:
        self.config = config or AuroraSubstrateConfig()
        if encoder is None:
            if self.config.latent_dim == 512:
                self.encoder = SemanticEncoder()
            else:
                self.encoder = HashingEncoder(dim=self.config.latent_dim, seed=self.config.rng_seed)
        else:
            self.encoder = encoder

    def boot(self, now: datetime | None = None) -> bytes:
        start = now or utc_now()
        rng = np.random.default_rng(self.config.rng_seed)
        
        # Initialize the three latent vectors
        core_vec = normalize(rng.normal(scale=0.05, size=self.config.latent_dim))
        surface_vec = core_vec.copy()
        user_model_vec = core_vec.copy()
        
        state = SealedState(
            header=SealedStateHeader(SEALED_STATE_VERSION, isoformat_utc(start), isoformat_utc(start)),
            latent=LatentState(core_vector=core_vec, surface_vector=surface_vec, user_model=user_model_vec),
            metric=MetricState.isotropic(self.config.latent_dim, self.config.metric_rank),
            sparks={},  # Empty topology graph
            arrival=ArrivalState(isoformat_utc(start), 0.0, 0.2, 0.08, 0.05),
            rng_state=rng.bit_generator.state,
            last_event_time=isoformat_utc(start),
            next_wake_at=None,
        )
        state.next_wake_at = sample_next_wake(state.arrival, start, rng)
        return seal_state(state)

    def _thermodynamics(self, state: SealedState, dt_hours: float) -> dict[str, float]:
        """
        核心物理引擎：计算图谱中每个节点的激活度
        """
        high_decay = float(np.exp(-0.02 * dt_hours))
        activations = {}
        
        # 只处理包含文本的节点
        valid_sparks = [s for s in state.sparks.values() if s.type in ("episodic", "fossil") and s.text]
        if not valid_sparks:
            return activations
            
        for spark in valid_sparks:
            if spark.energy > 0.1:
                spark.energy *= high_decay
            else:
                spark.energy = max(1e-5, spark.energy - (0.0005 * dt_hours))
                
            # The Abyss: Structural Forgetting
            # If an episodic spark gets extremely cold, it falls into the Abyss (loses temporal context).
            if spark.type == "episodic" and spark.energy < 0.05:
                if spark.prev_id or spark.next_id:
                    # Sever from the timeline, bridge the gap
                    prev_s = state.sparks.get(spark.prev_id) if spark.prev_id else None
                    next_s = state.sparks.get(spark.next_id) if spark.next_id else None
                    if prev_s: prev_s.next_id = spark.next_id
                    if next_s: next_s.prev_id = spark.prev_id
                    spark.prev_id = None
                    spark.next_id = None

        warped_latent = state.metric.matrix() @ state.latent.surface_vector
        vectors = np.stack([spark.vector for spark in valid_sparks])
        resonances = vectors @ warped_latent
        
        for i, spark in enumerate(valid_sparks):
            if resonances[i] > 0.05:
                spark.energy += 0.2 * float(resonances[i])
            activations[spark.spark_id] = spark.energy + max(0.0, float(resonances[i])) * 2.0

        return activations
        
    def _extract_episodic_subgraph(self, state: SealedState, activations: dict[str, float]) -> list[Spark]:
        """锚点定位与情景重放 (Anchor Search & Subgraph Traversal)"""
        if not activations:
            return []
            
        # 寻找 1~2 个绝对高光记忆锚点
        sorted_ids = sorted(activations.keys(), key=lambda k: activations[k], reverse=True)
        anchors = sorted_ids[:2]
        
        extracted = set()
        for anchor_id in anchors:
            # 向前向后提取相邻节点
            curr_id = anchor_id
            for _ in range(2):  # 后退2步
                if curr_id not in state.sparks:
                    break
                extracted.add(curr_id)
                curr_id = state.sparks[curr_id].prev_id
                if not curr_id:
                    break
                    
            curr_id = state.sparks[anchor_id].next_id
            for _ in range(2):  # 前进2步
                if not curr_id or curr_id not in state.sparks:
                    break
                extracted.add(curr_id)
                curr_id = state.sparks[curr_id].next_id
                
        # 提取完整节点并按物理时间排序，恢复叙事连贯性
        nodes = [state.sparks[nid] for nid in extracted]
        nodes.sort(key=lambda s: parse_utc(s.timestamp))
        return nodes

    def _reincarnate(
        self, state: SealedState, timestamp: str,
        text: str, vector: np.ndarray, error: float, source: str,
    ) -> Spark:
        """夺舍新生与拓扑图更新"""
        # 如果超出容量，需要删除最冷节点并缝合链表
        if len(state.sparks) >= self.config.capacity:
            coldest_spark = min(
                (s for s in state.sparks.values() if s.type != "concept"), 
                key=lambda s: s.energy
            )
            prev_s = state.sparks.get(coldest_spark.prev_id) if coldest_spark.prev_id else None
            next_s = state.sparks.get(coldest_spark.next_id) if coldest_spark.next_id else None
            
            if prev_s:
                prev_s.next_id = coldest_spark.next_id
            if next_s:
                next_s.prev_id = coldest_spark.prev_id
                
            del state.sparks[coldest_spark.spark_id]
            
        # 寻找图中的最新节点（时间链表尾部）
        latest_spark = None
        if state.sparks:
            # Concept nodes and Abyss nodes don't participate in the chronological tail
            temporal_sparks = [s for s in state.sparks.values() if s.type in ("episodic", "fossil") and (s.prev_id or s.next_id or len(state.sparks)==1)]
            if temporal_sparks:
                latest_spark = max(temporal_sparks, key=lambda s: parse_utc(s.timestamp))

        new_spark = Spark(
            spark_id=uuid4().hex,
            type="episodic",
            timestamp=timestamp,
            text=text,
            vector=vector,
            energy=1.0 + error * 5.0,
            source=source,
            prev_id=latest_spark.spark_id if latest_spark else None,
            next_id=None
        )
        
        if latest_spark:
            latest_spark.next_id = new_spark.spark_id
            
        # 概念引力 (Concept Attractors)
        # Any new episodic spark is automatically drawn to highly aligned concepts
        concept_sparks = [s for s in state.sparks.values() if s.type == "concept"]
        for concept in concept_sparks:
            # Semantic resonance
            sim = cosine(concept.vector, new_spark.vector)
            if sim > 0.85:
                new_spark.resonant_links.append(concept.spark_id)
                concept.resonant_links.append(new_spark.spark_id)
            
        state.sparks[new_spark.spark_id] = new_spark
        return new_spark

    def on_input(self, sealed_state: bytes, envelope: InputEnvelope) -> SubstrateInputResult:
        state, rng = self._load(sealed_state)
        when = parse_utc(envelope.timestamp)
        dt = max(0.0, (when - parse_utc(state.last_event_time)).total_seconds() / 3600.0)

        state.latent.core_vector, state.latent.surface_vector = advance_latent_ou_twin(
            state.latent.core_vector, state.latent.surface_vector, state.metric, dt, rng
        )
        advance_arrival(state.arrival, when)

        cue = self.encoder.encode(envelope.user_text)
        error = prediction_error(cue, state.latent.surface_vector)

        # 运行热力学法则：获得统一的激活度分布景观
        activations = self._thermodynamics(state, dt)

        # 意识上浮：基于情景重放提取连贯上下文
        resonated_sparks = self._extract_episodic_subgraph(state, activations)

        # 空间折叠
        state.metric = update_metric(state.metric, cue, error)
        state.latent.surface_vector = normalize(0.8 * state.latent.surface_vector + 0.2 * cue)

        # 夺舍新生，插入节点
        self._reincarnate(state, envelope.timestamp, envelope.user_text, cue, error, "user")

        boundary = boundary_budget(error, sum(s.energy for s in resonated_sparks) * 0.1, state.arrival.mutual_respect)
        verbosity = verbosity_budget(error, len(resonated_sparks))

        observe_user_contact(state.arrival, error)
        next_wake = sample_next_wake(state.arrival, when, rng)
        state.last_event_time = state.header.updated_at = isoformat_utc(when)
        state.next_wake_at = next_wake

        return SubstrateInputResult(
            sealed_state=self._save(state, rng),
            collapse_request=CollapseRequest(
                user_text=envelope.user_text,
                released_traces=[ReleasedTrace(text=s.text, source=s.source) for s in resonated_sparks],
                language=envelope.language,
                emit_reply=boundary < 0.96,
                boundary_budget=boundary,
                verbosity_budget=verbosity,
                mutual_respect=state.arrival.mutual_respect,
                media_refs=envelope.media_refs,
            ),
            event_id=uuid4().hex,
            next_wake_at=next_wake,
            health=self._health(state),
        )

    def on_feedback(self, sealed_state: bytes, envelope: FeedbackEnvelope) -> bytes:
        """行为反噬：处理发言、内部做梦与睡眠压缩"""
        state, rng = self._load(sealed_state)
        when = parse_utc(envelope.timestamp)

        if envelope.output_text:
            text = f"[Spoke]: {envelope.output_text}"
            source = "self"
            if envelope.is_compression:
                text = f"[Consolidated Fossil]: {envelope.output_text}"
                source = "compression"
            elif envelope.is_internal:
                text = f"[Dreamt]: {envelope.output_text}"
                source = "dream"
                
            cue = self.encoder.encode(text)
            
            # 对普通梦境和发言产生惊奇
            if not envelope.is_compression:
                state.metric = update_metric(state.metric, cue, error=0.05)
                state.latent.surface_vector = normalize(0.9 * state.latent.surface_vector + 0.1 * cue)
                
            new_spark = self._reincarnate(
                state, envelope.timestamp, text, cue,
                error=0.0, source=source,
            )
            if envelope.is_compression:
                new_spark.type = "fossil"
                # 压缩完成后清理被合并的节点
                if envelope.consumed_nodes:
                    for nid in envelope.consumed_nodes:
                        if nid in state.sparks:
                            node = state.sparks[nid]
                            # 缝合前后节点
                            prev_s = state.sparks.get(node.prev_id) if node.prev_id else None
                            next_s = state.sparks.get(node.next_id) if node.next_id else None
                            if prev_s: prev_s.next_id = node.next_id
                            if next_s: next_s.prev_id = node.prev_id
                            del state.sparks[nid]

        if envelope.is_internal or envelope.is_compression:
            state.arrival.internal_drive = max(0.0, state.arrival.internal_drive - 0.4)
        elif not envelope.output_text:
            state.arrival.internal_drive = min(8.0, state.arrival.internal_drive + 0.1)

        state.last_event_time = state.header.updated_at = isoformat_utc(when)
        return self._save(state, rng)

    def on_wake(self, sealed_state: bytes, envelope: WakeEnvelope) -> SubstrateWakeResult:
        state, rng = self._load(sealed_state)
        when = parse_utc(envelope.timestamp)
        dt = max(0.0, (when - parse_utc(state.last_event_time)).total_seconds() / 3600.0)

        state.latent.core_vector, state.latent.surface_vector = advance_latent_ou_twin(
            state.latent.core_vector, state.latent.surface_vector, state.metric, dt, rng
        )
        advance_arrival(state.arrival, when)

        activations = self._thermodynamics(state, dt)
        
        dream_req = None
        event_id = uuid4().hex
        
        # 判断是否触发睡眠有损压缩
        is_capacity_critical = len(state.sparks) > self.config.capacity * 0.9
        
        consumed_nodes = None
        if is_capacity_critical or (state.arrival.internal_drive > 0.8):
            # 寻找低能量的一段连续的情景记忆
            cold_sparks = sorted([s for s in state.sparks.values() if s.type == "episodic" and s.energy < 0.5], key=lambda s: parse_utc(s.timestamp))
            if len(cold_sparks) > 5:
                # 截取最冷的一段进行压缩
                target_sparks = cold_sparks[:10]
                consumed_nodes = [s.spark_id for s in target_sparks]
                dream_req = CollapseRequest(
                    user_text="<internal_dream_compression>",
                    released_traces=[ReleasedTrace(text=s.text, source=s.source) for s in target_sparks],
                    language="auto",
                    emit_reply=True,
                    boundary_budget=1.0,
                    verbosity_budget=1.0,
                    is_internal_dream_compression=True
                )
                observe_internal_action(state.arrival, mass=0.8)
                
        if not dream_req and state.arrival.internal_drive > 0.7:
            resonated_sparks = self._extract_episodic_subgraph(state, activations)
            if resonated_sparks:
                dream_req = CollapseRequest(
                    user_text="<internal_dream>",
                    released_traces=[ReleasedTrace(text=s.text, source=s.source) for s in resonated_sparks],
                    language="auto",
                    emit_reply=True,
                    boundary_budget=1.0,
                    verbosity_budget=1.0,
                    is_internal_dream=True,
                )
                observe_internal_action(state.arrival, mass=0.5)

        next_wake = sample_next_wake(state.arrival, when, rng)
        state.last_event_time = state.header.updated_at = isoformat_utc(when)
        state.next_wake_at = next_wake

        return SubstrateWakeResult(
            sealed_state=self._save(state, rng),
            next_wake_at=next_wake,
            health=self._health(state),
            dream_request=dream_req,
            event_id=event_id,
            consumed_nodes=consumed_nodes,
        )

    def _load(self, blob: bytes) -> tuple[SealedState, np.random.Generator]:
        s = unseal_state(blob)
        r = np.random.default_rng()
        r.bit_generator.state = s.rng_state
        return s, r

    def _save(self, state: SealedState, rng: np.random.Generator) -> bytes:
        state.rng_state = rng.bit_generator.state
        return seal_state(state)

    def _health(self, s: SealedState, err: str | None = None, ph: bool = True) -> HealthEnvelope:
        alive_count = sum(1 for spark in s.sparks.values() if spark.text)
        return HealthEnvelope(SEALED_STATE_VERSION, True, s.header.version, alive_count, s.next_wake_at, err, ph)

    def health_snapshot(self, sealed_state: bytes, last_error: str | None, provider_healthy: bool) -> HealthEnvelope:
        s, _ = self._load(sealed_state)
        return self._health(s, last_error, provider_healthy)

    def integrity_report(self, sealed_state: bytes, config_fingerprint: str) -> IntegrityEnvelope:
        s, _ = self._load(sealed_state)
        return build_integrity_report(
            sealed_state_version=s.header.version,
            config_fingerprint=config_fingerprint,
        )


def build_integrity_report(
    *,
    sealed_state_version: str = SEALED_STATE_VERSION,
    config_fingerprint: str,
    runtime_boundary: str = "process-opaque",
    substrate_transport: str = "in-process",
) -> IntegrityEnvelope:
    return IntegrityEnvelope(
        version=SEALED_STATE_VERSION,
        runtime_boundary=runtime_boundary,
        substrate_transport=substrate_transport,
        sealed_state_version=sealed_state_version,
        config_fingerprint=config_fingerprint,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
