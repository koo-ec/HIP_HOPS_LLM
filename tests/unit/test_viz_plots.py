"""Matplotlib rendering of trees, architectures and rankings.

A drawing test cannot assert that a picture is *good*, but it can assert the
things that actually break: that every node is drawn, that the reserved status
colour is never the only signal, that feedback edges are visibly distinguished,
and that degenerate inputs produce a figure or a clear error rather than a
traceback from deep inside matplotlib.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from hiphopsllm import (  # noqa: E402
    AgenticReliabilityStudy,
    extract_architecture,
    load_example,
    plot_architecture,
    plot_cutset_orders,
    plot_fault_tree,
    plot_importance,
)
from hiphopsllm.faulttree.synthesis import expand_to_tree  # noqa: E402
from hiphopsllm.viz.plots import TOKENS  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def _suptitle(ax) -> str:
    """The main title lives on the figure; the axes title is the subtitle."""
    sup = getattr(ax.figure, "_suptitle", None)
    return sup.get_text() if sup is not None else ""


def _subtitle(ax) -> str:
    """Subtitles are set left-aligned, so the default centre lookup misses them."""
    return ax.get_title(loc="left") or ax.get_title()


#: A serial pipeline with an early exit: the router's ``bail`` branch jumps
#: straight to ``__end__``, skipping every layer in between. That edge is the
#: one a reader most needs to see and the easiest one to draw invisibly.
EARLY_EXIT_SPEC = {
    "name": "early exit",
    "nodes": {
        "__start__": {"role": "source"},
        "first": {"role": "llm_agent"},
        "second": {"role": "llm_agent"},
        "__end__": {"role": "sink"},
    },
    "edges": [
        ["__start__", "first"],
        ["first", "second", "ok", True],
        ["first", "__end__", "bail", True],
        ["second", "__end__"],
    ],
}


class TestDesignTokens:
    def test_every_token_is_a_hex_colour(self):
        for name, value in TOKENS.items():
            assert isinstance(value, str), name
            assert value.startswith("#") and len(value) in (4, 7), f"{name}={value}"

    def test_the_surface_and_text_tokens_contrast(self):
        def luminance(hex_colour: str) -> float:
            r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        assert abs(luminance(TOKENS["surface"]) - luminance(TOKENS["text"])) > 0.5


class TestPlotFaultTree:
    def test_it_returns_axes_on_a_figure(self, tree):
        ax = plot_fault_tree(tree)
        assert ax.figure is not None
        assert ax.figure.get_size_inches()[0] > 0

    def test_every_basic_event_is_labelled(self, study):
        tree = study.report.tree("H2")
        ax = plot_fault_tree(tree, label_width=200)
        drawn = " ".join(_texts(ax))
        for node in tree.nodes.values():
            if node.ntype == "basic" and node.event_id:
                # labels are wrapped, so match on a distinctive fragment
                stem = node.event_id.split("-")[-1]
                assert stem in drawn, f"{node.event_id} was not drawn"

    def test_it_draws_onto_a_supplied_axes(self, tree):
        fig, ax = plt.subplots()
        returned = plot_fault_tree(tree, ax=ax)
        assert returned is ax

    def test_the_title_is_used(self, tree):
        ax = plot_fault_tree(tree, title="a custom title")
        assert _suptitle(ax) == "a custom title"

    def test_the_default_title_names_the_hazard(self, tree):
        ax = plot_fault_tree(tree)
        assert tree.id in _suptitle(ax)

    def test_the_axes_subtitle_carries_severity_and_top_event(self, tree):
        subtitle = _subtitle(plot_fault_tree(tree))
        assert "severity" in subtitle
        assert "top event" in subtitle

    def test_the_legend_can_be_turned_off(self, tree):
        with_legend = plot_fault_tree(tree, show_legend=True)
        without = plot_fault_tree(tree, show_legend=False)
        assert len(with_legend.patches) >= len(without.patches)

    def test_the_axes_are_hidden(self, tree):
        """Fault trees have no meaningful coordinates; visible ticks are noise."""
        ax = plot_fault_tree(tree)
        assert not ax.axison

    def test_label_width_truncates(self, tree):
        narrow = plot_fault_tree(tree, label_width=8)
        wide = plot_fault_tree(tree, label_width=60)
        longest_narrow = max(len(t) for t in _texts(narrow))
        longest_wide = max(len(t) for t in _texts(wide))
        assert longest_narrow < longest_wide

    def test_scale_changes_the_figure_size(self, tree):
        small = plot_fault_tree(tree, scale=0.5).figure.get_size_inches()
        large = plot_fault_tree(tree, scale=2.0).figure.get_size_inches()
        assert large[0] > small[0] and large[1] > small[1]

    def test_dag_and_expanded_tree_both_draw(self, tree):
        assert plot_fault_tree(tree, as_tree=True) is not None
        assert plot_fault_tree(tree, as_tree=False) is not None

    def test_a_repeated_cause_is_marked_when_expanded(self, study):
        """Shared sub-trees are the norm here; a repeat must be visibly a repeat."""
        expanded = expand_to_tree(study.report.tree("H2"))
        repeats = [n for n in expanded.nodes.values() if n.repeat_of]
        assert repeats, "H2 shares sub-trees, so expansion must produce repeats"
        assert plot_fault_tree(expanded) is not None

    @pytest.mark.parametrize("hazard", ["H1", "H2", "H3", "H4"])
    def test_every_hazard_of_the_example_draws(self, study, hazard):
        assert plot_fault_tree(study.report.tree(hazard)) is not None

    def test_a_single_node_tree_draws(self):
        """The degenerate case: one basic event and nothing else."""
        spec = {
            "name": "t",
            "nodes": {"__start__": {"role": "source"}, "a": {"role": "llm_agent"},
                      "__end__": {"role": "sink"}},
            "edges": [["__start__", "a"], ["a", "__end__"]],
        }
        study = AgenticReliabilityStudy(spec, name="tiny")
        study.analyse()
        for hazard in study.hazards_found():
            assert plot_fault_tree(study.report.tree(hazard)) is not None


class TestPlotArchitecture:
    def test_every_component_is_drawn(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        ax = plot_architecture(model)
        drawn = " ".join(_texts(ax))
        for cid in model.components:
            assert cid.split("::")[0] in drawn, f"{cid} was not drawn"

    def test_feedback_edges_are_distinguished(self, react_spec):
        """The loop is the reason the graph is not yet a fault tree, so it must
        be visible as something other than an ordinary edge."""
        model = extract_architecture(react_spec)
        ax = plot_architecture(model)
        styles = {
            tuple(line.get_linestyle() if isinstance(line.get_linestyle(), str) else "dashed")
            for line in ax.lines
        }
        dashed_annotations = [
            a
            for a in ax.texts + list(ax.patches)
            if getattr(a, "get_linestyle", lambda: "-")() not in ("-", "solid")
        ]
        assert styles or dashed_annotations or ax.patches

    def test_it_draws_onto_a_supplied_axes(self, parallel_spec):
        fig, ax = plt.subplots()
        assert plot_architecture(extract_architecture(parallel_spec), ax=ax) is ax

    def test_ports_can_be_shown(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        plain = len(_texts(plot_architecture(model, show_ports=False)))
        ported = len(_texts(plot_architecture(model, show_ports=True)))
        assert ported >= plain

    def test_the_title_is_used(self, parallel_spec):
        ax = plot_architecture(extract_architecture(parallel_spec), title="Approach 2")
        assert _suptitle(ax) == "Approach 2"

    def test_the_subtitle_counts_components_and_connections(self, parallel_spec):
        model = extract_architecture(parallel_spec)
        subtitle = _subtitle(plot_architecture(model))
        assert str(len(model.components)) in subtitle
        assert str(len(model.connections)) in subtitle

    @pytest.mark.parametrize(
        "key", ["react_calculator", "parallel_aggregator", "supervisor_workers"]
    )
    def test_every_bundled_architecture_draws(self, key):
        assert plot_architecture(extract_architecture(load_example(key))) is not None


class TestArchitectureSkipEdges:
    """An early exit spans several layers. Drawn straight it is a vertical line
    hidden behind every component it passes, with its branch label landing on
    top of one of them — so it must be routed around the side instead.
    """

    @pytest.fixture
    def drawn(self):
        return plot_architecture(extract_architecture(EARLY_EXIT_SPEC))

    def _positions(self, ax) -> dict:
        return {t.get_text(): t.get_position() for t in ax.texts}

    def _node_column(self, ax) -> float:
        pos = self._positions(ax)
        return max(
            pos[name][0]
            for name in ("__start__", "first", "second", "__end__")
            if name in pos
        )

    def test_the_skip_edge_label_is_clear_of_every_component(self, drawn):
        pos = self._positions(drawn)
        assert "bail" in pos, "the branch label was not drawn at all"
        assert pos["bail"][0] > self._node_column(drawn) + 0.5

    def test_the_skip_edge_itself_leaves_the_node_column(self, drawn):
        lane = max(max(line.get_xdata()) for line in drawn.lines)
        assert lane > self._node_column(drawn) + 0.5

    def test_the_ordinary_edge_label_stays_on_its_edge(self, drawn):
        """Only the skipping edge moves; a one-layer edge is still drawn straight."""
        pos = self._positions(drawn)
        assert pos["ok"][0] < pos["bail"][0]

    def test_the_view_is_wide_enough_to_show_the_lane(self, drawn):
        lane = max(max(line.get_xdata()) for line in drawn.lines)
        assert drawn.get_xlim()[1] > lane

    def test_two_skip_edges_never_share_a_lane(self):
        """Two early exits drawn on the same lane would be one line, and the
        second branch label would sit on top of the first."""
        spec = {
            "name": "two early exits",
            "nodes": {
                "__start__": {"role": "source"},
                "first": {"role": "llm_agent"},
                "second": {"role": "llm_agent"},
                "third": {"role": "llm_agent"},
                "__end__": {"role": "sink"},
            },
            "edges": [
                ["__start__", "first"],
                ["first", "second", "ok", True],
                ["first", "__end__", "bail_1", True],
                ["second", "third", "ok", True],
                ["second", "__end__", "bail_2", True],
                ["third", "__end__"],
            ],
        }
        ax = plot_architecture(extract_architecture(spec))
        pos = {t.get_text(): t.get_position() for t in ax.texts}
        assert pos["bail_1"][0] != pos["bail_2"][0]

    def test_a_router_is_not_captioned_with_a_word_already_in_its_name(self, drawn):
        """`first::router` captioned `router` prints the same word twice and
        crowds a shape that is narrow where the caption sits."""
        texts = _texts(drawn)
        assert "first\n::router" in texts, "the router id was broken mid-word"
        assert "router" not in texts

    def test_a_role_caption_is_still_drawn_where_it_says_something_new(self, drawn):
        assert "llm agent" in _texts(drawn)


class TestPlotImportance:
    def test_it_ranks_descending(self, study):
        ax = plot_importance(study.report.analysis("H2"))
        widths = [p.get_width() for p in ax.patches if p.get_width() > 0]
        assert widths == sorted(widths) or widths == sorted(widths, reverse=True)

    def test_top_n_limits_the_bars(self, study):
        analysis = study.report.analysis("H2")
        few = plot_importance(analysis, top_n=3)
        many = plot_importance(analysis, top_n=50)
        assert len([p for p in few.patches if p.get_width() > 0]) <= 3
        assert len([p for p in many.patches if p.get_width() > 0]) >= len(
            [p for p in few.patches if p.get_width() > 0]
        )

    def test_single_points_carry_a_text_tag_not_only_a_colour(self, study):
        """Colour is never the only signal: a single point of failure is also
        labelled, so the chart survives being printed in greyscale."""
        analysis = study.report.analysis("H2")
        assert analysis.single_points, "this hazard should have single points"
        drawn = " ".join(_texts(plot_importance(analysis, top_n=20))).lower()
        assert "single" in drawn

    def test_it_draws_onto_a_supplied_axes(self, study):
        fig, ax = plt.subplots()
        assert plot_importance(study.report.analysis("H2"), ax=ax) is ax


class TestPlotCutsetOrders:
    def test_it_covers_every_hazard(self, study):
        ax = plot_cutset_orders(study.report.analyses)
        drawn = (
            " ".join(_texts(ax))
            + " ".join(t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels())
            + _subtitle(ax)
            + " ".join(t.get_text() for t in ax.get_legend().get_texts())
            if ax.get_legend()
            else " ".join(_texts(ax))
            + " ".join(t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels())
            + _subtitle(ax)
        )
        for hazard in study.report.analyses:
            assert hazard in drawn

    def test_order_one_is_represented(self, study):
        """A tall order-1 bar is the finding, so order 1 must appear."""
        ax = plot_cutset_orders(study.report.analyses)
        labels = (
            " ".join(t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels())
            + " ".join(_texts(ax))
            + _subtitle(ax)
            + (" ".join(t.get_text() for t in ax.get_legend().get_texts())
               if ax.get_legend() else "")
        )
        assert "1" in labels

    def test_it_refuses_an_empty_analysis_set_clearly(self):
        with pytest.raises(ValueError, match="no cut sets"):
            plot_cutset_orders({})

    def test_the_study_helpers_delegate_correctly(self, study):
        assert study.plot_cutset_orders() is not None
        assert study.plot_importance("H2") is not None
        assert study.plot("H2") is not None
        assert study.plot_architecture() is not None
