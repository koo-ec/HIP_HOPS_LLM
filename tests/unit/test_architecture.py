"""Architecture extraction: components, ports, resources, loops."""

from __future__ import annotations


from HIP_HOPS_LLM import (
    LangGraphExtractor,
    Role,
    SystemModel,
    extract_architecture,
    find_cycles,
    is_acyclic,
    load_example,
    make_acyclic,
    parse_mermaid,
)

MERMAID = """
graph TD;
    __start__([__start__]):::first
    react_agent(react_agent)
    cot_agent(cot_agent)
    aggregator(aggregator)
    __end__([__end__]):::last
    __start__ --> react_agent;
    __start__ --> cot_agent;
    react_agent --> aggregator;
    cot_agent --> aggregator;
    aggregator --> __end__;
"""


class TestExtraction:
    def test_roles_are_classified(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        assert model.components["__start__"].role is Role.SOURCE
        assert model.components["__end__"].role is Role.SINK
        assert model.components["react_agent"].role is Role.LLM_AGENT
        assert model.components["aggregator"].role is Role.AGGREGATOR

    def test_tool_and_router_roles(self, react_spec):
        model = extract_architecture(react_spec)
        assert model.components["coder"].role is Role.TOOL
        assert model.components["generator::router"].role is Role.ROUTER

    def test_mermaid_and_spec_agree(self, parallel_spec):
        from_spec = extract_architecture(parallel_spec)
        from_mermaid = extract_architecture(MERMAID, name="m")
        assert set(from_spec.components) == set(from_mermaid.components)
        assert len(from_spec.connections) == len(from_mermaid.connections)

    def test_parse_mermaid_reads_conditional_edges(self):
        raw = parse_mermaid(
            "graph TD;\n a(a)\n b(b)\n a -. &nbsp;go&nbsp; .-> b;\n", name="t"
        )
        assert raw.edges == [("a", "b", "go", True)]

    def test_shared_snapshot_becomes_a_common_cause_group(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        groups = model.common_cause_groups()
        llm_groups = {v: m for (k, v), m in groups.items() if k == "llm"}
        assert llm_groups, "the shared model snapshot was not detected"
        members = next(iter(llm_groups.values()))
        assert {"react_agent", "cot_agent"} <= set(members)

    def test_a_system_model_passes_through_unchanged(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        assert extract_architecture(model) is model

    def test_ports_are_unique_per_incoming_connection(self, react_spec):
        model = extract_architecture(react_spec)
        end = model.components["__end__"]
        assert len(end.ports_in) == len(set(end.ports_in)), (
            "two router branches reaching END must not share an input port; a "
            "shared port collapses them into one deviation"
        )


class TestRouterMaterialisation:
    def test_router_is_materialised_from_conditional_edges(self):
        spec = {
            "name": "t",
            "nodes": {"a": {"role": "llm_agent"}, "b": {}, "__end__": {"role": "sink"}},
            "edges": [["a", "b", "go", True], ["a", "__end__", "stop", True]],
        }
        model = extract_architecture(spec)
        assert "a::router" in model.components
        assert model.components["a::router"].role is Role.ROUTER

    def test_predeclared_router_still_gets_its_feed_edge(self, react_spec):
        """Regression: a router declared as a node lost its incoming edge.

        The router component already existed, so the ``node -> node::router``
        edge was never emitted, which disconnected the graph and made the ReAct
        feedback loop vanish from the analysis without any warning.
        """
        model = extract_architecture(react_spec)
        feed = [
            c
            for c in model.connections
            if c.src == "generator" and c.dst == "generator::router"
        ]
        assert len(feed) == 1, "the generator must feed its own router exactly once"
        assert find_cycles(model), "the ReAct feedback loop must be visible"


class TestLoopElimination:
    def test_react_loop_is_found_and_cut(self, react_spec):
        model = extract_architecture(react_spec)
        assert not is_acyclic(model)
        acyclic, report = make_acyclic(model, unroll=1)
        assert is_acyclic(acyclic)
        assert report.had_cycles
        assert report.back_edges
        assert report.feedback_components

    def test_feedback_cut_component_is_added_not_deleted(self, react_spec):
        acyclic, report = make_acyclic(extract_architecture(react_spec), unroll=1)
        for cid in report.feedback_components:
            assert cid in acyclic.components
            assert acyclic.components[cid].role is Role.FEEDBACK

    def test_unrolling_deeper_replicates_the_loop_body(self, react_spec):
        model = extract_architecture(react_spec)
        one, _ = make_acyclic(model, unroll=1)
        two, _ = make_acyclic(model, unroll=2)
        assert len(two.components) > len(one.components)
        assert is_acyclic(two)

    def test_an_acyclic_graph_is_left_alone(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        acyclic, report = make_acyclic(model)
        assert not report.had_cycles
        assert set(acyclic.components) == set(model.components)


class TestExtractorClass:
    def test_extractor_applies_its_conventions(self, parallel_spec):
        extractor = LangGraphExtractor(role_overrides={"aggregator": "transform"})
        model = extractor.extract(parallel_spec)
        assert model.components["aggregator"].role is Role.TRANSFORM

    def test_with_roles_returns_a_copy(self, parallel_spec):
        base = LangGraphExtractor()
        derived = base.with_roles(aggregator="tool")
        assert base.role_overrides == {}
        assert derived.role_overrides == {"aggregator": "tool"}

    def test_with_resources_creates_a_common_cause_group(self):
        spec = {
            "name": "t",
            "nodes": {"a": {"role": "llm_agent"}, "b": {"role": "llm_agent"}},
            "edges": [["a", "b"]],
        }
        extractor = LangGraphExtractor().with_resources(
            a={"llm": "m"}, b={"llm": "m"}
        )
        groups = extractor.extract(spec).common_cause_groups()
        assert ("llm", "m") in groups

    def test_extract_acyclic_returns_a_report(self, react_spec):
        model, report = LangGraphExtractor(unroll=2).extract_acyclic(react_spec)
        assert is_acyclic(model)
        assert report.unroll == 2

    def test_call_is_extract(self, parallel_spec):
        extractor = LangGraphExtractor()
        assert set(extractor(parallel_spec).components) == set(
            extractor.extract(parallel_spec).components
        )


def test_every_bundled_example_extracts():
    from HIP_HOPS_LLM import EXAMPLES

    for key in EXAMPLES:
        model = extract_architecture(load_example(key), name=key)
        assert isinstance(model, SystemModel)
        assert model.components
        assert model.connections
