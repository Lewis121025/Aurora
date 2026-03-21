"""Unified ingest-and-evolve memory field for Aurora."""

from __future__ import annotations

import json
import math
import random
import re
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


NEGATION_MARKERS = {
    "not",
    "never",
    "no",
    "none",
    "cannot",
    "can't",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "isnt",
    "isn't",
    "won't",
    "without",
}

TEMPORAL_MARKERS = {
    "today",
    "tomorrow",
    "yesterday",
    "recent",
    "recently",
    "now",
    "soon",
    "later",
    "week",
    "month",
    "year",
}

POSITIVE_AFFECT = {
    "love",
    "like",
    "prefer",
    "enjoy",
    "good",
    "great",
    "happy",
}

NEGATIVE_AFFECT = {
    "hate",
    "dislike",
    "bad",
    "sad",
    "angry",
    "upset",
}

RELATION_HINTS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "have",
    "has",
    "had",
    "want",
    "need",
    "plan",
    "prefer",
    "like",
    "love",
    "work",
    "live",
    "build",
    "develop",
    "promise",
    "remind",
    "sync",
}

GENERIC_RELATIONS = {"is", "are", "was", "were", "be", "have", "has", "had"}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "with",
    "by",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "my",
    "your",
    "our",
    "their",
}

GENERIC_TOKENS = STOPWORDS | TEMPORAL_MARKERS | NEGATION_MARKERS
POSITIVE_EDGE_KINDS = {"supports", "refers_to", "part_of", "precedes", "abstracts", "instantiates", "version_of"}
NEGATIVE_EDGE_KINDS = {"contradicts", "suppresses"}
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
DATE_RE = re.compile(r"\b\d{4}[-/]?\d{1,2}[-/]?\d{0,2}\b")
CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+")


@dataclass(frozen=True)
class AtomCore:
    atom_id: str
    kind: str
    payload: Dict[str, Any]
    anchor: Dict[str, Any]
    address: Dict[str, Tuple[str, ...]]
    birth_step: int
    epistemic_mode: str
    source: str


@dataclass
class AtomState:
    activation: float = 0.0
    stability: float = 0.1
    plasticity: float = 0.9
    confidence: float = 0.5
    utility: float = 0.3
    fidelity: float = 1.0
    novelty: float = 1.0
    access_count: int = 0
    recall_hits: int = 0
    evidence_hits: int = 0
    prediction_hits: int = 0
    suppression_hits: int = 0
    last_touched: int = 0


@dataclass
class Atom:
    core: AtomCore
    state: AtomState


@dataclass
class Edge:
    src: str
    dst: str
    kind: str
    weight: float
    confidence: float = 0.5
    created_step: int = 0
    last_touched: int = 0


@dataclass(frozen=True)
class RecallItem:
    atom_id: str
    kind: str
    score: float
    text: str
    payload: Dict[str, Any]
    reason: Dict[str, Any]
    birth_step: int
    source: str


@dataclass(frozen=True)
class RecallEdge:
    src: str
    dst: str
    kind: str
    weight: float
    confidence: float


@dataclass
class RecallResult:
    cue: str
    items: List[RecallItem]
    edges: List[RecallEdge]
    trace: Dict[str, Any]


@dataclass(frozen=True)
class EventIngestResult:
    event_id: str
    anchor_id: str
    atom_ids: List[str]


class AddressCodec:
    """Compile input text into sparse multi-head addresses."""

    head_weights = {
        "semantic": 1.0,
        "entity": 1.2,
        "relation": 1.0,
        "temporal": 0.5,
        "affect": 0.4,
        "context": 0.3,
        "source": 0.3,
    }

    def tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in WORD_RE.findall(text or "") if token]

    def encode(self, text: str, metadata: Optional[Dict[str, Any]] = None, *, kind: str) -> Dict[str, Tuple[str, ...]]:
        metadata = dict(metadata or {})
        tokens = self.tokenize(text)
        semantic: Set[str] = set()
        entity: Set[str] = set()
        relation: Set[str] = set()
        temporal: Set[str] = set()
        affect: Set[str] = set()
        context: Set[str] = set()
        source: Set[str] = set()

        lowered = (text or "").lower()
        has_negation = self._contains_negation(lowered)
        for marker in DATE_RE.findall(text or ""):
            temporal.add(marker)

        for token in tokens:
            if token in TEMPORAL_MARKERS:
                temporal.add(token)
            if token in POSITIVE_AFFECT:
                affect.add(f"negated_pos:{token}" if has_negation else f"pos:{token}")
            if token in NEGATIVE_AFFECT:
                affect.add(f"neg:{token}")
            if token in RELATION_HINTS:
                relation.add(token)
            if token not in STOPWORDS and len(token) >= 2:
                semantic.add(token)

        if re.search(r"\b(i|me|my)\b", lowered):
            entity.add("self")
        if re.search(r"\b(you|your)\b", lowered):
            entity.add("other:user")

        for raw in WORD_RE.findall(text or ""):
            if any(ch.isupper() for ch in raw):
                entity.add(raw.lower())

        speaker = str(metadata.get("speaker", "")).strip().lower()
        if speaker:
            context.add(f"speaker:{speaker}")
            if speaker == "user":
                entity.add("self")
            elif speaker == "assistant":
                entity.add("other:assistant")

        source_name = str(metadata.get("source", kind)).strip().lower()
        if source_name:
            source.add(source_name)
            context.add(f"source:{source_name}")

        context.add(f"kind:{kind}")
        if has_negation:
            affect.add("negation")

        return {
            "semantic": tuple(sorted(semantic)),
            "entity": tuple(sorted(entity)),
            "relation": tuple(sorted(relation)),
            "temporal": tuple(sorted(temporal)),
            "affect": tuple(sorted(affect)),
            "context": tuple(sorted(context)),
            "source": tuple(sorted(source)),
        }

    def clause_split(self, text: str) -> List[str]:
        major = [part.strip() for part in CLAUSE_SPLIT_RE.split(text or "") if part.strip()]
        if not major:
            return []
        clauses: List[str] = []
        for piece in major:
            if len(piece) <= 96:
                clauses.append(piece)
                continue
            parts = [part.strip() for part in re.split(r",+", piece) if part.strip()]
            clauses.extend(parts or [piece])
        return clauses

    def signature(self, text: str, address: Dict[str, Tuple[str, ...]]) -> Tuple[str, ...]:
        entities = [token for token in address.get("entity", ()) if token]
        relations = [token for token in address.get("relation", ()) if token and token not in GENERIC_RELATIONS]
        identity = [token for token in entities if token in {"self", "other:user", "other:assistant"}]
        entity_head = identity or entities
        if entity_head and relations:
            return tuple(sorted(set(entity_head[:1] + relations[:2])))
        if relations:
            return tuple(sorted(set(relations[:2])))
        informative = [
            token
            for token in address.get("semantic", ())
            if token not in GENERIC_TOKENS and len(token) >= 2
        ]
        informative = sorted(set(informative), key=lambda item: (-len(item), item))[:4]
        return tuple(informative)

    def polarity(self, text: str) -> int:
        return -1 if self._contains_negation(text.lower()) else 1

    @staticmethod
    def _contains_negation(text: str) -> bool:
        return any(marker in text for marker in NEGATION_MARKERS)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class MemoryKernel:
    """Unified full-ingest memory kernel."""

    def __init__(self, *, seed: int = 7, max_neighbors: int = 24):
        self.codec = AddressCodec()
        self.rand = random.Random(seed)
        self.max_neighbors = max_neighbors
        self.step = 0
        self.atoms: Dict[str, Atom] = {}
        self.out_edges: DefaultDict[str, Dict[Tuple[str, str], Edge]] = defaultdict(dict)
        self.in_edges: DefaultDict[str, Dict[Tuple[str, str], Edge]] = defaultdict(dict)
        self.address_index: DefaultDict[str, DefaultDict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.signature_index: DefaultDict[Tuple[str, ...], List[str]] = defaultdict(list)
        self.abstract_index: Dict[Tuple[str, ...], str] = {}
        self.event_members: DefaultDict[str, List[str]] = defaultdict(list)
        self.coactivation: DefaultDict[Tuple[str, str], float] = defaultdict(float)

    def ingest(self, text: str, metadata: Optional[Dict[str, Any]] = None, source: str = "dialogue", *, now_ts: float | None = None) -> EventIngestResult:
        metadata = dict(metadata or {})
        metadata["source"] = source
        self.step += 1
        event_id = str(metadata.get("event_id") or f"evt-{self.step}-{uuid.uuid4().hex[:6]}")
        timestamp = time.time() if now_ts is None else now_ts

        anchor_address = self.codec.encode(text, metadata, kind="anchor")
        anchor_id = self._new_atom_id("anchor")
        anchor_core = AtomCore(
            atom_id=anchor_id,
            kind="anchor",
            payload={"text": text, "event_id": event_id},
            anchor={"event_id": event_id, "source": source, "timestamp": timestamp},
            address=anchor_address,
            birth_step=self.step,
            epistemic_mode=str(metadata.get("epistemic_mode", "observed")),
            source=source,
        )
        anchor_state = AtomState(
            activation=0.35,
            stability=0.18,
            plasticity=0.92,
            confidence=0.95,
            utility=0.2,
            fidelity=1.0,
            novelty=self._estimate_novelty(anchor_address),
            access_count=1,
            evidence_hits=1,
            last_touched=self.step,
        )
        self._store_atom(anchor_core, anchor_state)
        self.event_members[event_id].append(anchor_id)

        fact_ids: List[str] = []
        previous_fact_id: str | None = None
        clauses = self.codec.clause_split(text) or [text]
        for clause_index, clause in enumerate(clauses):
            fact_id = self._ingest_clause(
                clause=clause,
                anchor_id=anchor_id,
                event_id=event_id,
                clause_index=clause_index,
                metadata=metadata,
            )
            fact_ids.append(fact_id)
            self.event_members[event_id].append(fact_id)
            if previous_fact_id is not None:
                self._add_edge(previous_fact_id, fact_id, "precedes", 0.45, confidence=0.75)
            previous_fact_id = fact_id

        self._record_coactivation([anchor_id] + fact_ids)
        if fact_ids:
            self._maybe_form_abstraction(fact_ids)
        return EventIngestResult(event_id=event_id, anchor_id=anchor_id, atom_ids=[anchor_id] + fact_ids)

    def retrieve(self, cue: str, top_k: int = 8, propagation_steps: int = 3) -> RecallResult:
        self.step += 1
        cue_address = self.codec.encode(cue, {"source": "probe"}, kind="probe")
        cue_polarity = self.codec.polarity(cue)
        seeds = self._candidate_atoms(cue_address, max_candidates=128)
        base = {atom_id: score for atom_id, score in seeds if score >= 0.05}
        if not base:
            return RecallResult(cue=cue, items=[], edges=[], trace={"activated": 0, "reason": "no_candidate"})

        active: Dict[str, float] = {}
        for atom_id, score in base.items():
            self._apply_lazy_decay(atom_id)
            active[atom_id] = clamp(score + 0.25 * self.atoms[atom_id].state.activation, 0.0, 1.0)

        for _ in range(propagation_steps):
            active = self._propagate(active, base)
            if not active:
                break

        ranked = sorted(
            active.items(),
            key=lambda item: item[1] * self._readout_bonus(item[0]) * self._cue_alignment_bonus(item[0], cue_polarity, cue_address),
            reverse=True,
        )[:top_k]
        atom_ids = [atom_id for atom_id, _ in ranked]
        self._record_coactivation(atom_ids)
        self._reconsolidate(atom_ids, ranked)
        self._maybe_form_abstraction(atom_ids)
        return RecallResult(
            cue=cue,
            items=[self._recall_item(atom_id, score, cue_address) for atom_id, score in ranked],
            edges=self._selected_edges(atom_ids),
            trace={"activated": len(active), "seed_count": len(base), "top_ids": atom_ids},
        )

    def current_state(self, top_k: int = 10) -> RecallResult:
        cue = "self current recent live work like prefer commitment tension goal"
        result = self.retrieve(cue, top_k=top_k, propagation_steps=4)
        winners: Dict[Tuple[str, ...], RecallItem] = {}
        passthrough: List[RecallItem] = []
        for item in result.items:
            if item.kind == "anchor":
                continue
            signature = tuple(str(token) for token in item.payload.get("signature", ()))
            if signature:
                existing = winners.get(signature)
                if existing is None or item.score > existing.score:
                    winners[signature] = item
                continue
            passthrough.append(item)
        deduped = list(winners.values()) + passthrough
        deduped.sort(key=lambda item: item.score, reverse=True)
        result.items = deduped[:top_k]
        result.edges = self._selected_edges([item.atom_id for item in result.items])
        return result

    def replay(self, budget: int = 8) -> List[Dict[str, Any]]:
        self.step += 1
        priorities: List[Tuple[str, float]] = []
        for atom_id in self.atoms:
            self._apply_lazy_decay(atom_id)
            atom = self.atoms[atom_id]
            conflict = self._conflict_pressure(atom_id)
            priority = (
                0.32 * atom.state.utility
                + 0.23 * atom.state.novelty
                + 0.20 * (1.0 - atom.state.confidence)
                + 0.15 * conflict
                + 0.10 * atom.state.fidelity
            )
            if atom.core.kind == "anchor":
                priority *= 0.65
            priorities.append((atom_id, priority))

        seeds = [atom_id for atom_id, _ in sorted(priorities, key=lambda item: item[1], reverse=True)[:budget]]
        traces: List[Dict[str, Any]] = []
        for seed in seeds:
            neighborhood = self._expand_neighborhood(seed, depth=2, max_nodes=14)
            if not neighborhood:
                continue
            for rank, atom_id in enumerate(neighborhood):
                atom = self.atoms[atom_id]
                atom.state.activation = clamp(atom.state.activation + 0.10 / (1 + rank), 0.0, 1.0)
                atom.state.stability = clamp(atom.state.stability + 0.01, 0.05, 1.0)
                atom.state.plasticity = clamp(atom.state.plasticity + 0.02 * self._conflict_pressure(atom_id), 0.05, 1.0)
                atom.state.last_touched = self.step
            self._record_coactivation(neighborhood)
            abstract_id = self._maybe_form_abstraction(neighborhood)
            self._compress_details(neighborhood)
            self._reconcile_neighborhood(neighborhood)
            traces.append({"seed": seed, "neighborhood": neighborhood, "abstract_id": abstract_id})
        return traces

    def save_json(self, path: str | Path) -> None:
        payload = {
            "step": self.step,
            "atoms": [
                {
                    "core": {
                        **asdict(atom.core),
                        "address": {head: list(tokens) for head, tokens in atom.core.address.items()},
                    },
                    "state": asdict(atom.state),
                }
                for atom in self.atoms.values()
            ],
            "edges": [asdict(edge) for edge in self._iter_edges()],
            "coactivation": [{"a": left, "b": right, "value": value} for (left, right), value in self.coactivation.items()],
            "abstract_index": [{"key": list(key), "atom_id": atom_id} for key, atom_id in self.abstract_index.items()],
            "event_members": [{"event_id": event_id, "members": members} for event_id, members in self.event_members.items()],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "MemoryKernel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return kernel_from_snapshot(payload)

    def _recall_item(self, atom_id: str, score: float, cue_address: Dict[str, Tuple[str, ...]]) -> RecallItem:
        atom = self.atoms[atom_id]
        return RecallItem(
            atom_id=atom_id,
            kind=atom.core.kind,
            score=round(score, 4),
            text=self._display_text(atom_id),
            payload=dict(atom.core.payload),
            reason=self._build_reason(atom_id, cue_address),
            birth_step=atom.core.birth_step,
            source=atom.core.source,
        )

    def _selected_edges(self, atom_ids: Sequence[str]) -> List[RecallEdge]:
        allowed = set(atom_ids)
        edges = [
            RecallEdge(src=edge.src, dst=edge.dst, kind=edge.kind, weight=round(edge.weight, 4), confidence=round(edge.confidence, 4))
            for edge in self._iter_edges()
            if edge.src in allowed and edge.dst in allowed
        ]
        edges.sort(key=lambda edge: (edge.kind not in NEGATIVE_EDGE_KINDS, -(abs(edge.weight) * edge.confidence), edge.src, edge.dst))
        return edges[:32]

    def _ingest_clause(
        self,
        *,
        clause: str,
        anchor_id: str,
        event_id: str,
        clause_index: int,
        metadata: Dict[str, Any],
    ) -> str:
        normalized_clause = self._normalize_clause_text(clause, metadata)
        address = self.codec.encode(normalized_clause, metadata, kind="fact")
        signature = self.codec.signature(normalized_clause, address)
        polarity = self.codec.polarity(normalized_clause)
        utility = self._estimate_utility(normalized_clause, address)
        confidence = self._initial_confidence(metadata)
        novelty = self._estimate_novelty(address)
        fact_id = self._new_atom_id("fact")
        core = AtomCore(
            atom_id=fact_id,
            kind="fact",
            payload={
                "text": normalized_clause,
                "signature": list(signature),
                "polarity": polarity,
                "event_id": event_id,
                "clause_index": clause_index,
            },
            anchor={"anchor_id": anchor_id, "event_id": event_id},
            address=address,
            birth_step=self.step,
            epistemic_mode=str(metadata.get("epistemic_mode", "asserted")),
            source=str(metadata.get("source", "dialogue")),
        )
        state = AtomState(
            activation=0.30,
            stability=0.12,
            plasticity=0.88,
            confidence=confidence,
            utility=utility,
            fidelity=1.0,
            novelty=novelty,
            access_count=1,
            evidence_hits=1,
            last_touched=self.step,
        )
        self._store_atom(core, state)
        self._add_edge(anchor_id, fact_id, "part_of", 0.95, confidence=0.95)
        self._link_fact(fact_id)
        self.signature_index[signature].append(fact_id)
        return fact_id

    def _normalize_clause_text(self, clause: str, metadata: Dict[str, Any]) -> str:
        speaker = str(metadata.get("speaker", "")).strip().lower()
        normalized = clause.strip()
        lowered = normalized.lower()
        if speaker == "assistant":
            for prefix in ("i will ", "i'll ", "i can "):
                if lowered.startswith(prefix):
                    action = normalized[len(prefix) :].strip(" ,.!?")
                    if action:
                        return f"Aurora commitment: {action}"
        return normalized

    def _link_fact(self, atom_id: str) -> None:
        atom = self.atoms[atom_id]
        signature = tuple(str(token) for token in atom.core.payload.get("signature", ()))
        polarity = int(atom.core.payload.get("polarity", 1))

        for other_id in self.signature_index.get(signature, []):
            if other_id == atom_id:
                continue
            other = self.atoms[other_id]
            other_polarity = int(other.core.payload.get("polarity", 1))
            if other_polarity == polarity:
                self._add_edge(other_id, atom_id, "supports", 0.60, confidence=0.75)
                self._add_edge(atom_id, other_id, "version_of", 0.45, confidence=0.60)
                self._reinforce_alignment(atom_id, other_id)
            else:
                self._add_edge(other_id, atom_id, "contradicts", 0.78, confidence=0.80)
                self._add_edge(atom_id, other_id, "contradicts", 0.78, confidence=0.80)
                self._add_edge(other_id, atom_id, "suppresses", 0.55, confidence=0.65)
                self._add_edge(atom_id, other_id, "suppresses", 0.55, confidence=0.65)
                self._local_conflict_update(atom_id, other_id)

        for other_id, score in self._candidate_atoms(atom.core.address, max_candidates=self.max_neighbors):
            if other_id == atom_id:
                continue
            if score < 0.16:
                break
            other = self.atoms[other_id]
            if other.core.kind == "abstract":
                self._add_edge(other_id, atom_id, "abstracts", clamp(0.45 + score / 2.0, 0.0, 0.90), confidence=0.70)
                self._add_edge(atom_id, other_id, "instantiates", clamp(0.35 + score / 2.0, 0.0, 0.85), confidence=0.70)
            elif tuple(str(token) for token in other.core.payload.get("signature", ())) != signature:
                self._add_edge(atom_id, other_id, "refers_to", clamp(score, 0.0, 0.72), confidence=0.55)
                self._add_edge(other_id, atom_id, "refers_to", clamp(score * 0.82, 0.0, 0.65), confidence=0.55)

    def _propagate(self, active: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
        next_scores: DefaultDict[str, float] = defaultdict(float)
        touched_negative_pairs: Set[Tuple[str, str]] = set()

        for atom_id, score in active.items():
            next_scores[atom_id] += score * 0.45 + base.get(atom_id, 0.0) * 0.55
            for edge in self.out_edges.get(atom_id, {}).values():
                if edge.kind in POSITIVE_EDGE_KINDS:
                    next_scores[edge.dst] += score * edge.weight * 0.36
                elif edge.kind in NEGATIVE_EDGE_KINDS:
                    next_scores[edge.dst] -= score * edge.weight * 0.44
                    left, right = sorted((atom_id, edge.dst))
                    touched_negative_pairs.add((left, right))

        cleaned = {atom_id: max(0.0, score) for atom_id, score in next_scores.items() if score > 0.015}
        if not cleaned:
            return {}

        for left, right in touched_negative_pairs:
            if left not in cleaned or right not in cleaned:
                continue
            if cleaned[left] >= cleaned[right]:
                cleaned[right] *= 0.30
            else:
                cleaned[left] *= 0.30

        for atom_id in list(cleaned):
            atom = self.atoms[atom_id]
            bonus = 0.50 + 0.22 * atom.state.stability + 0.20 * atom.state.confidence + 0.08 * atom.state.utility
            cleaned[atom_id] *= bonus

        max_score = max(cleaned.values())
        if max_score > 1.0:
            for atom_id in list(cleaned):
                cleaned[atom_id] /= max_score

        ranked = sorted(cleaned.items(), key=lambda item: item[1], reverse=True)[:128]
        return dict(ranked)

    def _reconsolidate(self, atom_ids: Sequence[str], ranked: Sequence[Tuple[str, float]]) -> None:
        scores = dict(ranked)
        for rank, atom_id in enumerate(atom_ids):
            self._apply_lazy_decay(atom_id)
            atom = self.atoms[atom_id]
            score = scores.get(atom_id, 0.0)
            atom.state.activation = clamp(atom.state.activation + 0.18 * score, 0.0, 1.0)
            atom.state.stability = clamp(atom.state.stability + 0.025 * score, 0.05, 1.0)
            atom.state.utility = clamp(atom.state.utility + 0.012 / (1 + rank), 0.0, 1.0)
            surprise = 1.0 - atom.state.confidence
            atom.state.plasticity = clamp(atom.state.plasticity + 0.04 * surprise - 0.02 * atom.state.stability, 0.05, 1.0)
            atom.state.recall_hits += 1
            atom.state.access_count += 1
            atom.state.last_touched = self.step

    def _maybe_form_abstraction(self, atom_ids: Sequence[str]) -> Optional[str]:
        fact_ids = [atom_id for atom_id in atom_ids if self.atoms.get(atom_id) and self.atoms[atom_id].core.kind == "fact"]
        if len(fact_ids) < 3:
            return None

        token_freq: Counter[str] = Counter()
        for atom_id in fact_ids:
            for token in self.atoms[atom_id].core.address.get("semantic", ()):
                if token not in GENERIC_TOKENS and len(token) >= 2:
                    token_freq[token] += 1
        common = [token for token, count in token_freq.items() if count >= 2 and token not in GENERIC_TOKENS]
        common = sorted(common, key=lambda token: (-token_freq[token], -len(token), token))[:4]
        if not common:
            return None

        cluster = [
            atom_id
            for atom_id in fact_ids
            if sum(token in self.atoms[atom_id].core.address.get("semantic", ()) for token in common) >= max(1, len(common) // 2)
        ]
        distinct_signatures = {
            tuple(str(token) for token in self.atoms[atom_id].core.payload.get("signature", ()))
            for atom_id in cluster
            if self.atoms[atom_id].core.payload.get("signature")
        }
        if len(cluster) < 3 or len(distinct_signatures) < 2:
            return None

        key = tuple(sorted(common))
        existing_id = self.abstract_index.get(key)
        if existing_id is not None:
            abstract = self.atoms[existing_id]
            abstract.state.stability = clamp(abstract.state.stability + 0.02, 0.05, 1.0)
            abstract.state.utility = clamp(abstract.state.utility + 0.02, 0.0, 1.0)
            abstract.state.activation = clamp(abstract.state.activation + 0.06, 0.0, 1.0)
            abstract.state.last_touched = self.step
            for atom_id in cluster:
                self._add_edge(existing_id, atom_id, "abstracts", 0.68, confidence=0.72)
                self._add_edge(atom_id, existing_id, "instantiates", 0.58, confidence=0.72)
            return existing_id

        label = " / ".join(common)
        address = {
            "semantic": tuple(sorted(common)),
            "entity": tuple(),
            "relation": tuple(sorted(token for token in common if token in RELATION_HINTS)),
            "temporal": tuple(),
            "affect": tuple(),
            "context": ("kind:abstract", "source:replay"),
            "source": ("replay",),
        }
        abstract_id = self._new_atom_id("abstract")
        core = AtomCore(
            atom_id=abstract_id,
            kind="abstract",
            payload={"label": label, "support": len(cluster), "diversity": len(distinct_signatures), "member_ids": list(cluster[:32])},
            anchor={"source": "replay", "step": self.step},
            address=address,
            birth_step=self.step,
            epistemic_mode="derived",
            source="replay",
        )
        state = AtomState(
            activation=0.22,
            stability=clamp(0.18 + 0.05 * len(cluster), 0.05, 0.90),
            plasticity=0.48,
            confidence=clamp(0.35 + 0.05 * len(distinct_signatures), 0.10, 0.88),
            utility=clamp(0.18 + 0.08 * len(cluster), 0.0, 1.0),
            fidelity=0.82,
            novelty=0.30,
            access_count=1,
            prediction_hits=max(0, len(cluster) - 1),
            last_touched=self.step,
        )
        self._store_atom(core, state)
        self.abstract_index[key] = abstract_id
        for atom_id in cluster:
            self._add_edge(abstract_id, atom_id, "abstracts", 0.68, confidence=0.72)
            self._add_edge(atom_id, abstract_id, "instantiates", 0.58, confidence=0.72)
        return abstract_id

    def _compress_details(self, atom_ids: Sequence[str]) -> None:
        for atom_id in atom_ids:
            atom = self.atoms[atom_id]
            if atom.core.kind == "fact":
                atom.state.fidelity = clamp(atom.state.fidelity * 0.995, 0.25, 1.0)

    def _reconcile_neighborhood(self, atom_ids: Sequence[str]) -> None:
        local = set(atom_ids)
        for atom_id in atom_ids:
            for edge in self.out_edges.get(atom_id, {}).values():
                if edge.kind not in NEGATIVE_EDGE_KINDS or edge.dst not in local:
                    continue
                left = self.atoms[atom_id]
                right = self.atoms[edge.dst]
                left_score = left.state.confidence * 0.55 + left.state.stability * 0.35 + self._epistemic_bonus(left.core.epistemic_mode) * 0.10
                right_score = right.state.confidence * 0.55 + right.state.stability * 0.35 + self._epistemic_bonus(right.core.epistemic_mode) * 0.10
                if left_score >= right_score:
                    right.state.activation *= 0.72
                    right.state.suppression_hits += 1
                    left.state.confidence = clamp(left.state.confidence + 0.01, 0.0, 1.0)
                else:
                    left.state.activation *= 0.72
                    left.state.suppression_hits += 1
                    right.state.confidence = clamp(right.state.confidence + 0.01, 0.0, 1.0)

    def _estimate_novelty(self, address: Dict[str, Tuple[str, ...]]) -> float:
        candidates = self._candidate_atoms(address, max_candidates=16)
        if not candidates:
            return 1.0
        return clamp(1.0 - max(score for _, score in candidates), 0.05, 1.0)

    def _estimate_utility(self, text: str, address: Dict[str, Tuple[str, ...]]) -> float:
        lowered = text.lower()
        score = 0.15
        if any(term in lowered for term in ("plan", "promise", "must", "goal", "should", "commitment")):
            score += 0.25
        if "self" in address.get("entity", ()):
            score += 0.20
        if any(token in address.get("semantic", ()) for token in ("like", "prefer", "need", "limit", "develop", "design", "aurora")):
            score += 0.18
        if address.get("temporal"):
            score += 0.08
        return clamp(score, 0.05, 0.95)

    @staticmethod
    def _initial_confidence(metadata: Dict[str, Any]) -> float:
        mode = str(metadata.get("epistemic_mode", "asserted")).lower()
        return {
            "observed": 0.82,
            "asserted": 0.62,
            "inferred": 0.45,
            "hypothesis": 0.30,
            "derived": 0.40,
        }.get(mode, 0.60)

    def _store_atom(self, core: AtomCore, state: AtomState) -> None:
        self.atoms[core.atom_id] = Atom(core=core, state=state)
        for head, tokens in core.address.items():
            for token in tokens:
                self.address_index[head][token].add(core.atom_id)

    def _candidate_atoms(self, address: Dict[str, Tuple[str, ...]], max_candidates: int = 64) -> List[Tuple[str, float]]:
        raw_scores: DefaultDict[str, float] = defaultdict(float)
        for head, tokens in address.items():
            if not tokens:
                continue
            head_weight = self.codec.head_weights.get(head, 1.0)
            for token in tokens:
                for atom_id in self.address_index[head].get(token, ()):
                    raw_scores[atom_id] += head_weight / math.sqrt(len(tokens) + 1.0)

        scored: List[Tuple[str, float]] = []
        for atom_id, value in raw_scores.items():
            address_map = self.atoms[atom_id].core.address
            denominator = 1.0 + sum(len(tokens) for tokens in address_map.values())
            scored.append((atom_id, value / math.sqrt(denominator)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:max_candidates]

    def _readout_bonus(self, atom_id: str) -> float:
        atom = self.atoms[atom_id]
        return 0.45 + 0.22 * atom.state.stability + 0.18 * atom.state.confidence + 0.15 * atom.state.utility

    def _cue_alignment_bonus(self, atom_id: str, cue_polarity: int, cue_address: Dict[str, Tuple[str, ...]]) -> float:
        atom = self.atoms[atom_id]
        if atom.core.kind != "fact":
            return 1.0
        atom_polarity = int(atom.core.payload.get("polarity", 1))
        cue_tokens = set(cue_address.get("semantic", ())) | set(cue_address.get("relation", ()))
        atom_tokens = set(atom.core.address.get("semantic", ())) | set(atom.core.address.get("relation", ()))
        overlap = cue_tokens.intersection(atom_tokens)
        if not overlap:
            return 1.0
        if cue_polarity == atom_polarity:
            return 1.10
        return 0.82

    def _record_coactivation(self, atom_ids: Sequence[str]) -> None:
        deduped = list(dict.fromkeys(atom_ids))
        for index in range(len(deduped)):
            for offset in range(index + 1, len(deduped)):
                left, right = sorted((deduped[index], deduped[offset]))
                self.coactivation[(left, right)] += 1.0

    def _apply_lazy_decay(self, atom_id: str) -> None:
        atom = self.atoms[atom_id]
        delta = max(0, self.step - atom.state.last_touched)
        if delta <= 0:
            return
        atom.state.activation *= math.exp(-0.22 * delta)
        atom.state.stability = clamp(atom.state.stability * math.exp(-0.003 * delta), 0.05, 1.0)
        atom.state.fidelity = clamp(atom.state.fidelity * math.exp(-0.010 * delta), 0.20, 1.0)
        atom.state.plasticity = clamp(atom.state.plasticity + 0.015 * delta, 0.05, 1.0)
        atom.state.last_touched = self.step

    def _add_edge(self, src: str, dst: str, kind: str, weight: float, *, confidence: float = 0.5) -> None:
        key = (dst, kind)
        existing = self.out_edges[src].get(key)
        if existing is None:
            edge = Edge(
                src=src,
                dst=dst,
                kind=kind,
                weight=clamp(weight, 0.01, 0.99),
                confidence=clamp(confidence, 0.01, 0.99),
                created_step=self.step,
                last_touched=self.step,
            )
            self.out_edges[src][key] = edge
            self.in_edges[dst][(src, kind)] = edge
            return
        existing.weight = clamp((existing.weight + weight) / 2.0 + 0.02, 0.01, 0.99)
        existing.confidence = clamp((existing.confidence + confidence) / 2.0, 0.01, 0.99)
        existing.last_touched = self.step

    def _iter_edges(self) -> Iterable[Edge]:
        for edges in self.out_edges.values():
            for edge in edges.values():
                yield edge

    def _reinforce_alignment(self, atom_a: str, atom_b: str) -> None:
        for atom_id in (atom_a, atom_b):
            atom = self.atoms[atom_id]
            atom.state.stability = clamp(atom.state.stability + 0.03, 0.05, 1.0)
            atom.state.confidence = clamp(atom.state.confidence + 0.02, 0.0, 1.0)
            atom.state.activation = clamp(atom.state.activation + 0.04, 0.0, 1.0)
            atom.state.last_touched = self.step

    def _local_conflict_update(self, atom_a: str, atom_b: str) -> None:
        left = self.atoms[atom_a]
        right = self.atoms[atom_b]
        left_score = left.state.confidence * 0.55 + left.state.stability * 0.35 + self._epistemic_bonus(left.core.epistemic_mode) * 0.10
        right_score = right.state.confidence * 0.55 + right.state.stability * 0.35 + self._epistemic_bonus(right.core.epistemic_mode) * 0.10
        if left_score >= right_score:
            right.state.activation *= 0.60
            right.state.suppression_hits += 1
            left.state.plasticity = clamp(left.state.plasticity + 0.02, 0.05, 1.0)
        else:
            left.state.activation *= 0.60
            left.state.suppression_hits += 1
            right.state.plasticity = clamp(right.state.plasticity + 0.02, 0.05, 1.0)
        left.state.confidence = clamp(left.state.confidence - 0.01, 0.0, 1.0)
        right.state.confidence = clamp(right.state.confidence - 0.01, 0.0, 1.0)

    def _conflict_pressure(self, atom_id: str) -> float:
        total = 0.0
        for edge in self.out_edges.get(atom_id, {}).values():
            if edge.kind in NEGATIVE_EDGE_KINDS:
                total += edge.weight
        return clamp(total, 0.0, 1.0)

    def _expand_neighborhood(self, seed: str, depth: int = 2, max_nodes: int = 12) -> List[str]:
        visited: Set[str] = set()
        order: List[str] = []
        queue: deque[Tuple[str, int]] = deque([(seed, 0)])
        while queue and len(order) < max_nodes:
            atom_id, distance = queue.popleft()
            if atom_id in visited:
                continue
            visited.add(atom_id)
            order.append(atom_id)
            if distance >= depth:
                continue
            candidate_edges = sorted(self.out_edges.get(atom_id, {}).values(), key=lambda edge: edge.weight, reverse=True)
            for edge in candidate_edges:
                if edge.kind in POSITIVE_EDGE_KINDS and edge.dst not in visited:
                    queue.append((edge.dst, distance + 1))
        return order

    @staticmethod
    def _epistemic_bonus(mode: str) -> float:
        return {
            "observed": 1.00,
            "asserted": 0.75,
            "derived": 0.55,
            "inferred": 0.45,
            "hypothesis": 0.25,
        }.get(mode, 0.50)

    def _display_text(self, atom_id: str) -> str:
        atom = self.atoms[atom_id]
        if atom.core.kind == "abstract":
            return f"[ABSTRACT] {atom.core.payload.get('label', '')}"
        return str(atom.core.payload.get("text", ""))

    def _build_reason(self, atom_id: str, cue_address: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
        atom = self.atoms[atom_id]
        overlap: Dict[str, List[str]] = {}
        for head, tokens in cue_address.items():
            common = sorted(set(tokens).intersection(atom.core.address.get(head, ())))
            if common:
                overlap[head] = common
        return {
            "overlap": overlap,
            "stability": round(atom.state.stability, 4),
            "confidence": round(atom.state.confidence, 4),
            "utility": round(atom.state.utility, 4),
            "fidelity": round(atom.state.fidelity, 4),
        }

    @staticmethod
    def _new_atom_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


def snapshot_from_kernel(kernel: MemoryKernel) -> Dict[str, Any]:
    return {
        "step": kernel.step,
        "atoms": [
            {
                "core": {
                    **asdict(atom.core),
                    "address": {head: list(tokens) for head, tokens in atom.core.address.items()},
                },
                "state": asdict(atom.state),
            }
            for atom in kernel.atoms.values()
        ],
        "edges": [asdict(edge) for edge in kernel._iter_edges()],
        "coactivation": [{"a": left, "b": right, "value": value} for (left, right), value in kernel.coactivation.items()],
        "abstract_index": [{"key": list(key), "atom_id": atom_id} for key, atom_id in kernel.abstract_index.items()],
        "event_members": [{"event_id": event_id, "members": members} for event_id, members in kernel.event_members.items()],
    }


def kernel_from_snapshot(payload: Dict[str, Any]) -> MemoryKernel:
    kernel = MemoryKernel()
    kernel.step = int(payload.get("step", 0))

    for item in payload.get("atoms", []):
        core_payload = dict(item["core"])
        core_payload["address"] = {head: tuple(tokens) for head, tokens in core_payload["address"].items()}
        core = AtomCore(**core_payload)
        state = AtomState(**item["state"])
        kernel._store_atom(core, state)
        signature = tuple(str(token) for token in core.payload.get("signature", ()))
        if core.kind == "fact" and signature:
            kernel.signature_index[signature].append(core.atom_id)

    for edge_payload in payload.get("edges", []):
        edge = Edge(**edge_payload)
        kernel.out_edges[edge.src][(edge.dst, edge.kind)] = edge
        kernel.in_edges[edge.dst][(edge.src, edge.kind)] = edge

    for row in payload.get("coactivation", []):
        kernel.coactivation[(row["a"], row["b"])] = float(row["value"])

    for row in payload.get("abstract_index", []):
        kernel.abstract_index[tuple(row["key"])] = row["atom_id"]

    for row in payload.get("event_members", []):
        kernel.event_members[row["event_id"]] = list(row["members"])

    return kernel


__all__ = [
    "AddressCodec",
    "Atom",
    "AtomCore",
    "AtomState",
    "Edge",
    "EventIngestResult",
    "MemoryKernel",
    "RecallEdge",
    "RecallItem",
    "RecallResult",
    "kernel_from_snapshot",
    "snapshot_from_kernel",
]
