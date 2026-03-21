from __future__ import annotations

from aurora.field_engine import MemoryKernel


def _fact_ids(kernel: MemoryKernel) -> list[str]:
    return [atom_id for atom_id, atom in kernel.atoms.items() if atom.core.kind == "fact"]


def test_ingest_writes_anchor_and_fact_atoms() -> None:
    kernel = MemoryKernel(seed=13)

    result = kernel.ingest(
        "I live in Hangzhou. I like tea.",
        metadata={"speaker": "user", "epistemic_mode": "asserted"},
    )

    assert result.anchor_id in kernel.atoms
    assert kernel.atoms[result.anchor_id].core.kind == "anchor"
    assert len(result.atom_ids) == 3
    assert len(_fact_ids(kernel)) == 2


def test_retrieve_reinforces_activation() -> None:
    kernel = MemoryKernel(seed=13)
    kernel.ingest("I like tea.", metadata={"speaker": "user"})
    fact_id = _fact_ids(kernel)[0]
    baseline = kernel.atoms[fact_id].state.activation

    result = kernel.retrieve("What do I like?", top_k=4)

    assert result.items
    assert any("tea" in item.text.lower() for item in result.items)
    assert kernel.atoms[fact_id].state.activation > baseline
    assert kernel.atoms[fact_id].state.recall_hits == 1


def test_correction_builds_negative_pressure_without_deleting_old_fact() -> None:
    kernel = MemoryKernel(seed=13)
    kernel.ingest("I live in Shanghai.", metadata={"speaker": "user"})
    kernel.ingest("I do not live in Shanghai.", metadata={"speaker": "user"})

    facts = [atom for atom in kernel.atoms.values() if atom.core.kind == "fact"]
    shanghai_facts = [atom for atom in facts if "shanghai" in str(atom.core.payload.get("text", "")).lower()]

    assert len(shanghai_facts) == 2
    negative_edges = [edge for edge in kernel._iter_edges() if edge.kind in {"contradicts", "suppresses"}]
    assert negative_edges


def test_replay_can_form_abstractions() -> None:
    kernel = MemoryKernel(seed=13)
    kernel.ingest("I build Aurora systems.", metadata={"speaker": "user"})
    kernel.ingest("I design Aurora workflows.", metadata={"speaker": "user"})
    kernel.ingest("I develop Aurora tooling.", metadata={"speaker": "user"})

    traces = kernel.replay(budget=6)

    assert traces
    assert any(atom.core.kind == "abstract" for atom in kernel.atoms.values())


def test_current_state_dedupes_same_signature() -> None:
    kernel = MemoryKernel(seed=13)
    kernel.ingest("I live in Shanghai.", metadata={"speaker": "user"})
    kernel.ingest("I live in Hangzhou.", metadata={"speaker": "user"})

    state = kernel.current_state(top_k=8)

    live_items = [item for item in state.items if "live" in item.text.lower()]
    assert len(live_items) == 1
