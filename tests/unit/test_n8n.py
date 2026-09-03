"""Reading an n8n workflow export into the architecture model."""

from __future__ import annotations

import json

import pytest

from hiphopsllm import load_n8n, n8n_to_spec
from hiphopsllm.architecture.model import Role
from hiphopsllm.io.n8n import classify


def _node(name, node_type, **kwargs):
    node = {"name": name, "type": node_type, "typeVersion": 1, "parameters": {}}
    node.update(kwargs)
    return node


@pytest.fixture
def gmail_agent():
    """A redacted copy of a real Gmail auto-reply workflow.

    Trigger -> agent, with a chat model, a Postgres memory and two Gmail tools
    attached to the agent by ``ai_*`` connections, plus a sticky note.
    """
    return {
        "name": "Mail agent",
        "nodes": [
            _node("Gmail Trigger", "n8n-nodes-base.gmailTrigger",
                  credentials={"gmailOAuth2": {"id": "c1", "name": "Mail account"}}),
            _node("AI Agent", "@n8n/n8n-nodes-langchain.agent",
                  parameters={"options": {"systemMessage": "reply to business mail only"}}),
            _node("Chat Model", "@n8n/n8n-nodes-langchain.lmChatOpenAi",
                  parameters={"model": {"value": "gpt-4.1"}},
                  credentials={"openAiApi": {"id": "c2", "name": "OpenAI account"}}),
            _node("Reply", "n8n-nodes-base.gmailTool",
                  parameters={"operation": "reply",
                              "message": "={{ $fromAI('Message', ``, 'string') }}"},
                  credentials={"gmailOAuth2": {"id": "c1", "name": "Mail account"}}),
            _node("Read thread", "n8n-nodes-base.gmailTool",
                  parameters={"operation": "get"},
                  credentials={"gmailOAuth2": {"id": "c1", "name": "Mail account"}}),
            _node("Memory", "@n8n/n8n-nodes-langchain.memoryPostgresChat",
                  credentials={"postgres": {"id": "c3", "name": "Postgres account"}}),
            _node("Sticky Note", "n8n-nodes-base.stickyNote",
                  parameters={"content": "## tools"}),
        ],
        "connections": {
            "Gmail Trigger": {"main": [[{"node": "AI Agent", "type": "main", "index": 0}]]},
            "Chat Model": {"ai_languageModel": [[{"node": "AI Agent"}]]},
            "Reply": {"ai_tool": [[{"node": "AI Agent"}]]},
            "Read thread": {"ai_tool": [[{"node": "AI Agent"}]]},
            "Memory": {"ai_memory": [[{"node": "AI Agent"}]]},
        },
        "meta": {"instanceId": "abc123"},
    }


class TestClassification:
    def test_sticky_note_is_not_a_component(self, gmail_agent):
        wf = load_n8n(gmail_agent)
        assert wf.blocks["Sticky Note"].kind == "excluded"
        assert "Sticky Note" not in wf.to_spec()["nodes"]

    def test_trigger_is_the_boundary(self, gmail_agent):
        assert load_n8n(gmail_agent).blocks["Gmail Trigger"].role is Role.SOURCE

    def test_model_sub_node_becomes_a_resource_of_the_agent(self, gmail_agent):
        wf = load_n8n(gmail_agent)
        assert wf.blocks["Chat Model"].kind == "resource"
        assert "Chat Model" not in wf.to_spec()["nodes"]
        assert wf.blocks["AI Agent"].resources["llm"] == "gpt-4.1"

    def test_a_shared_model_makes_a_common_cause_group(self, gmail_agent):
        gmail_agent["nodes"].append(
            _node("Second Agent", "@n8n/n8n-nodes-langchain.agent")
        )
        gmail_agent["connections"]["Chat Model"]["ai_languageModel"][0].append(
            {"node": "Second Agent"}
        )
        gmail_agent["connections"]["AI Agent"] = {
            "main": [[{"node": "Second Agent"}]]
        }
        wf = load_n8n(gmail_agent)
        groups = wf.system().common_cause_groups()
        assert ("llm", "gpt-4.1") in groups
        assert groups[("llm", "gpt-4.1")] == ["AI Agent", "Second Agent"]

    def test_unknown_type_falls_back_to_transform_and_says_so(self):
        rule = classify(_node("X", "n8n-nodes-base.somethingNobodyHasSeen"))
        assert rule.role is Role.TRANSFORM
        assert "role_overrides" in rule.why

    def test_role_override_wins_and_is_recorded(self, gmail_agent):
        wf = load_n8n(gmail_agent, role_overrides={"Memory": "transform"})
        assert wf.blocks["Memory"].role is Role.TRANSFORM
        assert any("role_overrides" in note for note in wf.blocks["Memory"].notes)


class TestConnections:
    def test_tool_connection_is_reversed_into_an_invocation_and_a_loop(self, gmail_agent):
        wf = load_n8n(gmail_agent)
        edges = {(src, dst) for src, dst, _, _ in wf.edges}
        assert ("AI Agent", "Reply") in edges       # the agent calls the tool
        assert ("Reply", "AI Agent") in edges       # the observation comes back

    def test_tool_feedback_can_be_turned_off(self, gmail_agent):
        wf = load_n8n(gmail_agent, tool_feedback=False)
        edges = {(src, dst) for src, dst, _, _ in wf.edges}
        assert ("AI Agent", "Reply") in edges
        assert ("Reply", "AI Agent") not in edges

    def test_memory_keeps_the_direction_the_export_gives_it(self, gmail_agent):
        edges = {(src, dst) for src, dst, _, _ in load_n8n(gmail_agent).edges}
        assert ("Memory", "AI Agent") in edges
        assert ("AI Agent", "Memory") not in edges

    def test_acting_tool_reaches_the_delivery_boundary(self, gmail_agent):
        edges = {(src, dst) for src, dst, _, _ in load_n8n(gmail_agent).edges}
        assert ("Reply", "__end__") in edges

    def test_read_only_tool_does_not_reach_the_boundary(self, gmail_agent):
        wf = load_n8n(gmail_agent)
        assert wf.blocks["Read thread"].side_effect is False
        assert ("Read thread", "__end__") not in {(s, d) for s, d, _, _ in wf.edges}

    def test_memory_is_not_an_outward_action_despite_writing_to_postgres(self, gmail_agent):
        wf = load_n8n(gmail_agent)
        assert wf.blocks["Memory"].side_effect is False
        assert ("Memory", "__end__") not in {(s, d) for s, d, _, _ in wf.edges}

    def test_branch_node_is_the_router_and_no_second_one_is_materialised(self):
        spec = {
            "name": "branchy",
            "nodes": [
                _node("Start", "n8n-nodes-base.manualTrigger"),
                _node("If", "n8n-nodes-base.if"),
                _node("Yes", "n8n-nodes-base.set"),
                _node("No", "n8n-nodes-base.set"),
            ],
            "connections": {
                "Start": {"main": [[{"node": "If"}]]},
                "If": {"main": [[{"node": "Yes"}], [{"node": "No"}]]},
            },
        }
        wf = load_n8n(spec)
        assert wf.blocks["If"].role is Role.ROUTER
        system = wf.system()
        assert "If::router" not in system.components
        assert [c.id for c in system.by_role(Role.ROUTER)] == ["If"]
        labels = {label for _, _, label, _ in wf.edges if label}
        assert {"true", "false"} <= labels


class TestFailureLogic:
    def test_acting_agent_gets_a_commission_event(self, gmail_agent):
        wf = load_n8n(gmail_agent)
        report = wf.analyse()
        assert "BE-AI Agent-POLICY" in report.failure_model.events
        assert "BE-Reply-UNSOLICITED" in report.failure_model.events

    def test_commission_hazard_is_reworded_for_an_outward_action(self, gmail_agent):
        report = load_n8n(gmail_agent).analyse()
        hazard = next(h for h in report.hazards if h.id == "H5-Reply")
        assert "outward action" in hazard.name
        assert hazard.severity == "critical"

    def test_commission_hazard_dropped_where_there_is_no_logic_for_it(self, gmail_agent):
        wf = load_n8n(gmail_agent)
        report = wf.analyse()
        assert "H5-Memory" not in [h.id for h in report.hazards]
        assert any("No commission hazard" in note for note in wf.notes)

    def test_the_unsolicited_reply_decomposes_into_agent_and_tool(self, gmail_agent):
        report = load_n8n(gmail_agent).analyse()
        cuts = {frozenset(cs) for cs in report.cut_sets("H5-Reply")}
        assert frozenset({"BE-AI Agent-POLICY"}) in cuts
        assert frozenset({"BE-Reply-UNSOLICITED"}) in cuts

    def test_hallucination_reaches_the_boundary(self, gmail_agent):
        report = load_n8n(gmail_agent).analyse()
        events = {e for cs in report.cut_sets("H2") for e in cs}
        assert "BE-AI Agent-HALLUC" in events

    def test_model_authored_arguments_are_flagged(self, gmail_agent):
        wf = load_n8n(gmail_agent)
        assert wf.blocks["Reply"].model_authored_args is True
        assert wf.blocks["Read thread"].model_authored_args is False


class TestLedgerAndSpec:
    def test_every_node_appears_in_the_ledger_with_a_reason(self, gmail_agent):
        rows = load_n8n(gmail_agent).ledger()
        assert len(rows) == len(gmail_agent["nodes"])
        assert all(row["why"] for row in rows)

    def test_shared_credential_is_reported_as_a_common_dependency(self, gmail_agent):
        notes = " ".join(load_n8n(gmail_agent).notes)
        assert "Mail account" in notes

    def test_spec_round_trips_through_the_architecture(self, gmail_agent):
        spec = n8n_to_spec(gmail_agent)
        assert spec["nodes"]["AI Agent"]["role"] == "llm_agent"
        assert "__end__" in spec["nodes"]

    def test_json_text_and_path_are_both_accepted(self, gmail_agent, tmp_path):
        path = tmp_path / "wf.json"
        path.write_text(json.dumps(gmail_agent), encoding="utf-8")
        assert load_n8n(str(path)).name == "Mail agent"
        assert load_n8n(json.dumps(gmail_agent)).name == "Mail agent"

    def test_a_json_that_is_not_a_workflow_is_refused(self):
        with pytest.raises(ValueError, match="not an n8n workflow"):
            load_n8n({"hello": "world"})


class TestStudy:
    def test_study_is_ready_to_observe(self, gmail_agent):
        study = load_n8n(gmail_agent).study()
        assert study.report is not None
        study.observe([1, 1, 0, 1, 1, 0],
                      ["business", "business", "business", "other", "other", "other"],
                      component="AI Agent",
                      profile={"business": 0.6, "other": 0.4})
        study.run()
        assert "AI Agent" in study.evidence
        envelope = study.hazard_probability("H2")
        assert 0.0 <= envelope.lower <= envelope.upper <= 1.0
