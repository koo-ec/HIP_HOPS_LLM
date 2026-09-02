"""Extraction internals: role classification, resource detection, drawables.

These are the parts that decide whether a common-cause group is found, and a
missed one turns a single point of failure into an apparently redundant
architecture. They are worth testing directly rather than only through the
bundled examples, whose answers are already known to be right.
"""

from __future__ import annotations

import functools

import pytest

from HIP_HOPS_LLM import Role, SystemModel, extract_architecture, parse_mermaid
from HIP_HOPS_LLM.architecture.model import (
    _model_id_of,
    _raw_from_drawable,
    _resolve_callable,
    classify_role,
    detect_resources,
    raw_from_spec,
    source_of_function,
)


# --------------------------------------------------------------------------- #
# Fakes that stand in for a live LangGraph drawable
# --------------------------------------------------------------------------- #
class FakeNode:
    def __init__(self, name, data=None, metadata=None):
        self.name = name
        self.data = data
        self.metadata = metadata or {}


class FakeEdge:
    def __init__(self, source, target, data=None, conditional=False):
        self.source = source
        self.target = target
        self.data = data
        self.conditional = conditional


class FakeDrawable:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


class FakeGraph:
    """Something with ``get_graph()``, like a compiled LangGraph."""

    def __init__(self, drawable):
        self._drawable = drawable

    def get_graph(self):
        return self._drawable


class TestSourceOfFunction:
    def test_it_reads_a_plain_function(self):
        def node(state):
            """a docstring"""
            return state

        assert "a docstring" in source_of_function(node)

    def test_none_gives_empty_text_not_an_error(self):
        assert source_of_function(None) == ""

    def test_a_builtin_without_source_degrades_gracefully(self):
        assert isinstance(source_of_function(len), str)

    def test_a_partial_is_unwrapped_to_the_real_function(self):
        """Regression: source_of_function did not unwrap its argument, so a
        partial yielded no source — and therefore no role hint and no detected
        resources — while looking like a node that simply had none."""
        def node(state):
            model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
            return model_id

        assert "Qwen" in source_of_function(functools.partial(node, state=None))

    def test_a_runnable_style_wrapper_is_unwrapped(self):
        def node(state):
            model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
            return model_id

        class Runnable:
            def __init__(self, f):
                self.func = f

        assert "Qwen" in source_of_function(Runnable(node))

    def test_stale_source_from_another_function_is_discarded(self):
        """A re-executed notebook cell can make linecache return someone else's
        lines; classifying a component from those is worse than from none."""

        def node(state):
            return state

        node.__name__ = "a_completely_different_name"
        assert source_of_function(node) == ""

    def test_a_bound_method_is_read(self):
        class Agent:
            def run(self, state):
                """bound method body"""
                return state

        assert "bound method body" in source_of_function(Agent().run)


class TestResolveCallable:
    def test_a_plain_callable_resolves_to_itself(self):
        def f():
            pass

        assert _resolve_callable(f) is f

    def test_a_partial_resolves_to_its_func(self):
        def f():
            pass

        assert _resolve_callable(functools.partial(f)) is f

    def test_a_non_callable_resolves_to_none(self):
        assert _resolve_callable(42) is None

    def test_recursion_is_bounded(self):
        """A self-referential wrapper must terminate rather than blow the stack."""

        class Loop:
            @property
            def func(self):
                return self

            def __call__(self):
                pass

        _resolve_callable(Loop())          # must return, not recurse forever


class TestModelIdOf:
    @pytest.mark.parametrize(
        "attr", ["name_or_path", "model_name", "model"]
    )
    def test_it_reads_the_usual_attributes(self, attr):
        obj = type("M", (), {attr: "org/model-1"})()
        assert _model_id_of(obj) == "org/model-1"

    def test_it_reads_a_nested_config(self):
        config = type("C", (), {"_name_or_path": "org/nested"})()
        obj = type("M", (), {"config": config})()
        assert _model_id_of(obj) == "org/nested"

    def test_an_object_with_nothing_useful_gives_none(self):
        assert _model_id_of(object()) is None

    def test_a_non_string_attribute_is_ignored(self):
        obj = type("M", (), {"model_name": 7})()
        assert _model_id_of(obj) is None


class TestDetectResources:
    def test_empty_source_finds_nothing(self):
        assert detect_resources("") == {}

    def test_a_literal_model_id_is_found(self):
        code = 'model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"\nout = model.generate(x)'
        assert detect_resources(code)["llm"] == "Qwen/Qwen2.5-Math-1.5B-Instruct"

    def test_two_literals_are_joined_so_the_group_is_visible(self):
        code = 'a = "org/one"\nb = "org/two"\nmodel.generate(a)'
        llm = detect_resources(code)["llm"]
        assert "org/one" in llm and "org/two" in llm

    def test_a_live_object_beats_the_literal(self):
        """globals() is authoritative: it is what makes shared-snapshot
        detection reliable when the source only names a variable."""
        code = "out = my_model.generate(prompt)"
        live = type("M", (), {"name_or_path": "org/actually-loaded"})()
        assert detect_resources(code, {"my_model": live})["llm"] == "org/actually-loaded"

    def test_an_unresolvable_variable_is_recorded_as_a_variable(self):
        resources = detect_resources("out = my_model.generate(prompt)")
        assert resources["llm"].startswith("var:")

    def test_a_tokenizer_is_recorded(self):
        code = "ids = tokenizer.apply_chat_template(messages)\nmodel.generate(ids)"
        assert "tokenizer" in detect_resources(code)

    def test_a_gpu_runtime_is_detected(self):
        assert detect_resources("x = y.to('cuda')\nmodel.generate(x)").get("runtime")

    def test_two_nodes_with_the_same_literal_share_a_resource(self):
        code = 'm = "org/same"\nmodel.generate(m)'
        assert detect_resources(code) == detect_resources(code)


class TestClassifyRole:
    def test_generate_calls_mark_an_llm_agent(self):
        role = classify_role("worker", "worker", "out = model.generate(x)", 1, False)
        assert role is Role.LLM_AGENT

    def test_eval_marks_a_tool(self):
        role = classify_role("coder", "coder", "result = eval(expr)", 1, False)
        assert role is Role.TOOL

    def test_conditional_output_marks_a_router_as_a_last_resort(self):
        """Routers are normally materialised as their own components; this rule
        only fires with materialise_routers=False or for an otherwise
        unclassifiable node."""
        role = classify_role("route", "route", "return 'a' if x else 'b'", 1, True)
        assert role is Role.ROUTER

    def test_a_stronger_signal_beats_the_conditional_hint(self):
        role = classify_role("worker", "worker", "out = model.generate(x)", 1, True)
        assert role is Role.LLM_AGENT

    def test_the_boundary_ids_are_recognised(self):
        assert classify_role("__start__", "__start__", "", 0, False) is Role.SOURCE
        assert classify_role("__end__", "__end__", "", 1, False) is Role.SINK

    def test_a_name_hint_is_used_when_there_is_no_source(self):
        assert classify_role("react_agent", "react_agent", "", 1, False) is Role.LLM_AGENT

    def test_an_unrecognisable_node_is_a_transform(self):
        assert classify_role("thing", "thing", "return state", 1, False) is Role.TRANSFORM


class TestParseMermaid:
    def test_plain_edges(self):
        raw = parse_mermaid("graph TD;\n a(a)\n b(b)\n a --> b;\n", name="t")
        assert raw.edges == [("a", "b", "", False)]

    def test_conditional_edges_carry_their_label(self):
        raw = parse_mermaid(
            "graph TD;\n a(a)\n b(b)\n a -. &nbsp;go&nbsp; .-> b;\n", name="t"
        )
        assert raw.edges == [("a", "b", "go", True)]

    def test_boundary_shapes_are_read(self):
        raw = parse_mermaid(
            "graph TD;\n __start__([__start__]):::first\n a(a)\n __start__ --> a;\n",
            name="t",
        )
        assert "__start__" in raw.nodes

    def test_nodes_only_referenced_by_an_edge_are_created(self):
        raw = parse_mermaid("graph TD;\n a --> b;\n", name="t")
        assert {"a", "b"} <= set(raw.nodes)

    def test_the_name_is_kept(self):
        assert parse_mermaid("graph TD;\n a --> b;\n", name="mine").name == "mine"


class TestRawFromDrawable:
    def test_nodes_and_edges_are_read_from_objects(self):
        drawable = FakeDrawable(
            nodes={"a": FakeNode("a"), "b": FakeNode("b")},
            edges=[FakeEdge("a", "b")],
        )
        raw = _raw_from_drawable(drawable, "t")
        assert set(raw.nodes) == {"a", "b"}
        assert raw.edges == [("a", "b", "", False)]

    def test_a_conditional_edge_is_preserved(self):
        drawable = FakeDrawable(
            nodes={"a": FakeNode("a"), "b": FakeNode("b")},
            edges=[FakeEdge("a", "b", data="go", conditional=True)],
        )
        assert _raw_from_drawable(drawable, "t").edges == [("a", "b", "go", True)]

    def test_tuple_edges_are_accepted(self):
        """Not every drawable uses objects; tuples must work too."""
        drawable = FakeDrawable(nodes=["a", "b"], edges=[("a", "b")])
        raw = _raw_from_drawable(drawable, "t")
        assert raw.edges[0][:2] == ("a", "b")

    def test_node_data_is_kept_for_source_extraction(self):
        def body(state):
            """the node body"""
            return state

        drawable = FakeDrawable(nodes={"a": FakeNode("a", data=body)}, edges=[])
        assert _raw_from_drawable(drawable, "t").node_meta["a"]["data"] is body

    def test_an_empty_drawable_is_not_an_error(self):
        raw = _raw_from_drawable(FakeDrawable(nodes={}, edges=[]), "t")
        assert raw.nodes == {} and raw.edges == []


class TestExtractionRoutes:
    def test_a_get_graph_object_is_accepted(self):
        drawable = FakeDrawable(
            nodes={
                "__start__": FakeNode("__start__"),
                "worker": FakeNode("worker"),
                "__end__": FakeNode("__end__"),
            },
            edges=[FakeEdge("__start__", "worker"), FakeEdge("worker", "__end__")],
        )
        model = extract_architecture(FakeGraph(drawable), name="live")
        assert set(model.components) == {"__start__", "worker", "__end__"}
        assert model.name == "live"

    def test_the_four_routes_agree_on_the_same_system(self, parallel_spec):
        from_spec = extract_architecture(parallel_spec, name="x")
        from_raw = SystemModel.from_spec({"name": "x", **parallel_spec})
        assert set(from_spec.components) == set(from_raw.components)
        assert extract_architecture(from_spec) is from_spec

    def test_node_functions_supply_source_the_graph_lacks(self):
        def worker(state):
            # The literal is the point: extraction reads the source, never runs it.
            return model.generate(state, "org/from-node-functions")  # noqa: F821

        spec = {
            "name": "t",
            "nodes": {"__start__": {"role": "source"}, "worker": {},
                      "__end__": {"role": "sink"}},
            "edges": [["__start__", "worker"], ["worker", "__end__"]],
        }
        model_ = extract_architecture(spec, node_functions={"worker": worker})
        assert "from-node-functions" in model_.components["worker"].source_code

    def test_globals_ns_is_consulted_for_node_source(self):
        spec = {
            "name": "t",
            "nodes": {"__start__": {"role": "source"}, "namedfn": {},
                      "__end__": {"role": "sink"}},
            "edges": [["__start__", "namedfn"], ["namedfn", "__end__"]],
        }

        def namedfn(state):
            """found via globals"""
            return state

        model = extract_architecture(spec, globals_ns={"namedfn": namedfn})
        assert "found via globals" in model.components["namedfn"].source_code

    def test_an_explicit_empty_resource_dict_is_authoritative(self):
        """A rebuild must not re-detect resources the caller deliberately cleared."""
        spec = {
            "name": "t",
            "nodes": {
                "a": {"role": "llm_agent", "resources": {},
                      "source_code": 'm = "org/x"\nmodel.generate(m)'},
                "b": {"role": "sink"},
            },
            "edges": [["a", "b"]],
        }
        assert extract_architecture(spec).components["a"].resources == {}

    def test_materialise_routers_can_be_turned_off(self):
        spec = {
            "name": "t",
            "nodes": {"a": {"role": "llm_agent"}, "b": {}, "__end__": {"role": "sink"}},
            "edges": [["a", "b", "go", True], ["a", "__end__", "stop", True]],
        }
        assert "a::router" not in extract_architecture(
            spec, materialise_routers=False
        ).components


class TestRawFromSpec:
    def test_labels_default_to_the_node_id(self):
        raw = raw_from_spec({"name": "t", "nodes": {"a": {}}, "edges": []})
        assert raw.nodes["a"] == "a"

    def test_an_explicit_label_is_kept(self):
        raw = raw_from_spec({"name": "t", "nodes": {"a": {"label": "Agent A"}}, "edges": []})
        assert raw.nodes["a"] == "Agent A"

    def test_a_two_element_edge_is_unconditional(self):
        raw = raw_from_spec({"name": "t", "nodes": {}, "edges": [["a", "b"]]})
        assert raw.edges == [("a", "b", "", False)]

    def test_a_labelled_edge_defaults_to_conditional(self):
        raw = raw_from_spec({"name": "t", "nodes": {}, "edges": [["a", "b", "go"]]})
        assert raw.edges == [("a", "b", "go", True)]

    def test_missing_sections_are_tolerated(self):
        assert raw_from_spec({"name": "t"}).nodes == {}


class TestSystemModelHelpers:
    def test_to_mermaid_renders_every_component_and_connection(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        text = model.to_mermaid()
        assert text.lstrip().startswith(("graph", "flowchart"))
        for cid in model.components:
            assert cid.split("::")[0] in text
        assert text.count("-->") + text.count(".->") >= len(model.connections)

    def test_predecessors_and_successors_are_inverse(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        for cid in model.components:
            for successor in model.successors(cid):
                assert cid in model.predecessors(successor)

    def test_sources_and_sinks_use_the_declared_roles(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        assert model.sources() == ["__start__"]
        assert model.sinks() == ["__end__"]

    def test_sources_fall_back_to_topology_when_unlabelled(self):
        spec = {"name": "t", "nodes": {"a": {}, "b": {}}, "edges": [["a", "b"]]}
        model = extract_architecture(spec)
        assert "a" in model.sources()
        assert "b" in model.sinks()

    def test_by_role_filters(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        agents = model.by_role(Role.LLM_AGENT)
        assert {c.id for c in agents} == {"react_agent", "cot_agent"}

    def test_incoming_and_outgoing_can_be_filtered_by_port(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        aggregator = model.components["aggregator"]
        for port in aggregator.ports_in:
            assert model.incoming("aggregator", port=port)

    def test_a_group_of_one_is_not_a_common_cause_group(self):
        spec = {
            "name": "t",
            "nodes": {"a": {"role": "llm_agent", "resources": {"llm": "only-me"}},
                      "b": {"role": "sink"}},
            "edges": [["a", "b"]],
        }
        assert extract_architecture(spec).common_cause_groups() == {}

    def test_architecture_table_has_a_row_per_component(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        rows = model.architecture_table()
        assert len(rows) == len(model.components)
        assert {r["component"] for r in rows} == set(model.components)

    def test_port_accessors_have_sane_defaults(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        for component in model.components.values():
            assert isinstance(component.port_in(), str)
            assert isinstance(component.port_out(), str)

    def test_connection_ids_are_unique(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        ids = [c.id for c in model.connections]
        assert len(ids) == len(set(ids))
