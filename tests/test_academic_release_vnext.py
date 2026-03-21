from __future__ import annotations

from fastapi.testclient import TestClient

from aurora import mcp as mcp_module
from aurora.api import build_app
from aurora.system import response_output_to_dict
from tests.conftest import QueueLLM, SystemFactory


def test_release_scenario_old_material_remains_while_active_winner_changes(system_factory: SystemFactory) -> None:
    system = system_factory()
    system.ingest("I like coffee.", metadata={"speaker": "user"})
    system.ingest("I do not like coffee.", metadata={"speaker": "user"})

    result = system.retrieve("What do I like?", top_k=6)
    stored_texts = [atom.core.payload.get("text", "") for atom in system.kernel.atoms.values()]

    assert any(text == "I like coffee" for text in stored_texts)
    assert any(text == "I do not like coffee" for text in stored_texts)
    assert any(edge.kind == "suppresses" for edge in result.edges)


def test_release_scenario_retrieval_is_stateful(system_factory: SystemFactory) -> None:
    system = system_factory()
    event = system.ingest("I live in Hangzhou.", metadata={"speaker": "user"})
    fact_id = next(atom_id for atom_id in event.atom_ids if atom_id.startswith("fact_"))
    baseline = system.kernel.atoms[fact_id].state.activation

    system.retrieve("Where do I live?", top_k=4)

    assert system.kernel.atoms[fact_id].state.activation > baseline


def test_release_scenario_respond_uses_short_term_session_and_reingests_reply(system_factory: SystemFactory) -> None:
    system = system_factory(llm=QueueLLM("I will remind you tomorrow.", repeat_last=True))
    system.ingest("I live in Hangzhou.", metadata={"speaker": "user"})

    response = system.respond("session-a", "Please remind me tomorrow.")
    payload = response_output_to_dict(response)
    current_state = system.current_state(top_k=8)

    assert payload["session_id"] == "session-a"
    assert payload["response_text"] == "I will remind you tomorrow."
    assert "ongoing_commitments:" in payload["memory_brief"]
    assert any(item.text.startswith("Aurora commitment:") for item in current_state.items)


def test_release_surface_http_and_mcp_contracts(system_factory: SystemFactory) -> None:
    system = system_factory(llm=QueueLLM("ok", repeat_last=True))
    client = TestClient(build_app(system))

    ingest_response = client.post("/ingest", json={"text": "I like tea.", "metadata": {"speaker": "user"}})
    retrieve_response = client.post("/retrieve", json={"cue": "What do I like?"})
    state_response = client.post("/current-state", json={})

    assert ingest_response.status_code == 200
    assert retrieve_response.status_code == 200
    assert state_response.status_code == 200

    mcp_module._system = system
    mcp_recall = mcp_module.aurora_retrieve("What do I like?")
    mcp_state = mcp_module.aurora_current_state()
    assert mcp_recall["items"]
    assert "items" in mcp_state
    mcp_module._system = None
