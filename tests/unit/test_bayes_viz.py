"""Drawing Bayesian networks, with and without Graphviz.

The claim this module makes is that ``bn.show()`` always produces a picture.
These tests hold it to that on both branches: with the Graphviz binary present
and with it forced absent, which is the case that actually bites on a stock
Windows install.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from HIP_HOPS_LLM.bayes import viz as viz_module  # noqa: E402
from HIP_HOPS_LLM.bayes.viz import (  # noqa: E402
    PALETTE,
    BayesNetView,
    _blend,
    graphviz_available,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def network(study):
    from HIP_HOPS_LLM import BayesianNetwork

    return BayesianNetwork.from_fault_tree(
        study.report.tree("H2"), study.report.failure_model, name="H2"
    )


@pytest.fixture
def no_graphviz(monkeypatch):
    """Force the matplotlib branch, whatever this machine actually has."""
    monkeypatch.setattr(viz_module, "graphviz_available", lambda: False)
    return True


class TestGraphvizDetection:
    def test_it_returns_a_bool(self):
        assert isinstance(graphviz_available(), bool)

    def test_a_missing_binary_is_false(self, monkeypatch):
        monkeypatch.setattr(viz_module.shutil, "which", lambda name: None)
        assert graphviz_available() is False

    def test_a_binary_that_will_not_run_is_false(self, monkeypatch):
        """`which` finding a file proves nothing; the binary must actually run."""
        monkeypatch.setattr(viz_module.shutil, "which", lambda name: "/usr/bin/dot")

        def explode(*args, **kwargs):
            raise OSError("not executable")

        monkeypatch.setattr(viz_module.subprocess, "run", explode)
        assert graphviz_available() is False

    def test_a_working_binary_is_true(self, monkeypatch):
        monkeypatch.setattr(viz_module.shutil, "which", lambda name: "/usr/bin/dot")
        monkeypatch.setattr(viz_module.subprocess, "run", lambda *a, **k: None)
        assert graphviz_available() is True


class TestBackendSelection:
    def test_an_unknown_backend_is_refused(self, network):
        with pytest.raises(ValueError, match="'auto', 'pyagrum' or 'matplotlib'"):
            BayesNetView(network, backend="graphviz")

    def test_auto_falls_back_when_graphviz_is_absent(self, network, no_graphviz):
        assert BayesNetView(network).resolved_backend == "matplotlib"

    def test_an_explicit_backend_is_respected(self, network, no_graphviz):
        assert BayesNetView(network, backend="matplotlib").resolved_backend == (
            "matplotlib"
        )

    def test_auto_uses_pyagrum_when_it_can(self, network, monkeypatch):
        pytest.importorskip("pyagrum")
        monkeypatch.setattr(viz_module, "graphviz_available", lambda: True)
        assert BayesNetView(network).resolved_backend == "pyagrum"

    def test_auto_falls_back_if_the_pyagrum_build_fails(self, network, monkeypatch):
        """Graphviz present but the network unbuildable must not raise from show()."""
        monkeypatch.setattr(viz_module, "graphviz_available", lambda: True)

        class Broken:
            @property
            def net(self):
                raise RuntimeError("cannot build")

            cpts = network.cpts
            name = "broken"
            top = network.top

            def _resolve_evidence(self, evidence):
                return {}

            def posterior(self, *a, **k):
                raise RuntimeError

        assert BayesNetView(Broken()).resolved_backend == "matplotlib"


class TestLayout:
    def test_roots_are_at_depth_zero_and_the_top_is_deepest(self, network):
        layers = BayesNetView(network)._layers()
        assert set(network.cpts.roots) <= set(layers[0])
        assert network.top in layers[max(layers)]

    def test_every_variable_appears_exactly_once(self, network):
        layers = BayesNetView(network)._layers()
        placed = [v for members in layers.values() for v in members]
        assert sorted(placed) == sorted(network.cpts.order)

    def test_a_node_sits_above_all_of_its_parents(self, network):
        layers = BayesNetView(network)._layers()
        depth = {v: d for d, members in layers.items() for v in members}
        for var in network.cpts.order:
            for parent in network.cpts[var].parents:
                assert depth[parent] < depth[var], f"{var} not above {parent}"

    def test_the_barycentre_pass_does_not_lose_or_duplicate_nodes(self, network):
        view = BayesNetView(network)
        few = view._layers(sweeps=0)
        many = view._layers(sweeps=8)
        assert {v for m in few.values() for v in m} == {
            v for m in many.values() for v in m
        }

    def test_more_sweeps_do_not_increase_crossings(self, network):
        """The barycentre heuristic is there to reduce edge crossings; check it
        at least never makes them worse on this graph."""
        view = BayesNetView(network)

        def crossings(layers):
            index = {v: i for members in layers.values() for i, v in enumerate(members)}
            depth = {v: d for d, members in layers.items() for v in members}
            edges = [
                (p, c)
                for c in network.cpts.order
                for p in network.cpts[c].parents
                if depth[c] - depth[p] == 1
            ]
            count = 0
            for i, (p1, c1) in enumerate(edges):
                for p2, c2 in edges[i + 1 :]:
                    if depth[p1] != depth[p2]:
                        continue
                    if (index[p1] - index[p2]) * (index[c1] - index[c2]) < 0:
                        count += 1
            return count

        assert crossings(view._layers(sweeps=6)) <= crossings(view._layers(sweeps=0))


class TestMatplotlibBackend:
    def test_show_draws_without_graphviz(self, network, no_graphviz):
        """show() displays and returns None inside a notebook; what matters is
        that it does not raise and that a figure is obtainable."""
        view = BayesNetView(network)
        view.show()
        assert view.figure() is not None

    def test_figure_returns_a_matplotlib_figure_without_displaying(
        self, network, no_graphviz
    ):
        figure = BayesNetView(network).figure()
        assert hasattr(figure, "savefig")
        assert figure.axes

    def test_every_variable_is_drawn(self, network, no_graphviz):
        ax = BayesNetView(network, max_label=200).figure().axes[0]
        drawn = " ".join(t.get_text() for t in ax.texts).replace("\n", "")
        for event_id in network.basic_events:
            assert event_id in drawn, f"{event_id} was not drawn"

    def test_long_labels_are_elided_with_a_marker(self, network, no_graphviz):
        """A truncated label must look truncated, not merely wrong."""
        ax = BayesNetView(network, max_label=12).figure().axes[0]
        drawn = " ".join(t.get_text() for t in ax.texts)
        assert "…" in drawn

    def test_a_box_is_drawn_per_variable(self, network, no_graphviz):
        figure = BayesNetView(network).figure()
        assert len(figure.axes[0].patches) == len(network.cpts.order)

    def test_probabilities_are_printed_by_default(self, network, no_graphviz):
        with_probs = BayesNetView(network).figure().axes[0]
        without = BayesNetView(network, show_probabilities=False).figure().axes[0]
        assert len(" ".join(t.get_text() for t in with_probs.texts)) > len(
            " ".join(t.get_text() for t in without.texts)
        )

    def test_evidence_nodes_are_outlined_in_the_evidence_colour(
        self, network, no_graphviz
    ):
        event = network.basic_events[0]
        figure = BayesNetView(network, evidence={event: "Fail"}).figure()
        edge_colours = {p.get_edgecolor() for p in figure.axes[0].patches}
        import matplotlib.colors as mcolors

        expected = mcolors.to_rgba(PALETTE["evidence"])
        assert any(
            all(abs(a - b) < 1e-6 for a, b in zip(c, expected)) for c in edge_colours
        ), "the observed node must be outlined in the evidence colour"

    def test_the_caption_names_the_evidence(self, network, no_graphviz):
        event = network.basic_events[0]
        ax = BayesNetView(network).figure({event: "Fail"}).axes[0]
        caption = " ".join(t.get_text() for t in ax.texts)
        assert "observed" in caption, "the caption must explain the outline"
        assert "evidence" in ax.get_title(), "the title must name the evidence"

    def test_labels_are_truncated_to_max_label(self, network, no_graphviz):
        short = BayesNetView(network, max_label=10)
        long = BayesNetView(network, max_label=60)
        longest = lambda v: max(  # noqa: E731
            len(line)
            for t in v.figure().axes[0].texts
            for line in t.get_text().split("\n")
        )
        assert longest(short) <= longest(long)

    def test_figsize_is_honoured(self, network, no_graphviz):
        figure = BayesNetView(network, figsize=(9.0, 4.0)).figure()
        assert tuple(figure.get_size_inches()) == (9.0, 4.0)

    def test_the_caption_says_when_graphviz_is_missing(self, network, no_graphviz):
        text = " ".join(
            t.get_text() for t in BayesNetView(network).figure().axes[0].texts
        )
        assert "Graphviz" in text

    def test_side_by_side_falls_back_rather_than_raising(self, network, no_graphviz):
        BayesNetView(network).side_by_side()          # must not raise


class TestFileOutputs:
    def test_png_is_written_and_non_trivial(self, network, no_graphviz, tmp_path):
        path = BayesNetView(network).to_png(str(tmp_path / "n.png"))
        assert path == str(tmp_path / "n.png")
        assert (tmp_path / "n.png").stat().st_size > 5000

    def test_dpi_changes_the_file_size(self, network, no_graphviz, tmp_path):
        BayesNetView(network).to_png(str(tmp_path / "lo.png"), dpi=60)
        BayesNetView(network).to_png(str(tmp_path / "hi.png"), dpi=200)
        assert (tmp_path / "hi.png").stat().st_size > (tmp_path / "lo.png").stat().st_size

    def test_svg_is_returned_and_optionally_written(self, network, no_graphviz, tmp_path):
        svg = BayesNetView(network).to_svg(str(tmp_path / "n.svg"))
        assert svg.lstrip().startswith(("<?xml", "<svg"))
        assert (tmp_path / "n.svg").read_text(encoding="utf-8") == svg

    def test_dot_is_valid_looking_and_needs_no_binary(self, network, no_graphviz):
        dot = BayesNetView(network).to_dot()
        assert dot.startswith("digraph") and dot.rstrip().endswith("}")
        assert dot.count("->") == sum(
            len(network.cpts[v].parents) for v in network.cpts.order
        )
        for var in network.cpts.order:
            assert f'"{var}"' in dot

    def test_dot_labels_every_node_with_its_probability(self, network, no_graphviz):
        dot = BayesNetView(network).to_dot()
        assert dot.count("label=") == len(network.cpts.order)


class TestColourBlending:
    def test_blending_at_zero_and_one_returns_the_endpoints(self):
        assert _blend("#000000", "#ffffff", 0.0) == "#000000"
        assert _blend("#000000", "#ffffff", 1.0) == "#ffffff"

    def test_the_midpoint_is_between(self):
        mid = _blend("#000000", "#ffffff", 0.5)
        assert mid == "#808080" or mid == "#7f7f7f"

    def test_a_higher_probability_shades_further(self, network, no_graphviz):
        view = BayesNetView(network)
        low, _ = view._colours("x", is_root=True, probability=0.0)
        high, _ = view._colours("x", is_root=True, probability=1.0)
        assert low != high

    def test_the_top_event_keeps_its_own_colour(self, network, no_graphviz):
        fill, line = BayesNetView(network)._colours(network.top, is_root=False)
        assert fill == PALETTE["top"] and line == PALETTE["top_line"]

    def test_roots_and_gates_are_distinguishable(self, network, no_graphviz):
        view = BayesNetView(network)
        root_fill, _ = view._colours("x", is_root=True)
        gate_fill, _ = view._colours("y", is_root=False)
        assert root_fill != gate_fill


class TestNetworkIntegration:
    def test_bn_view_and_bn_show_agree_on_backend(self, network, no_graphviz):
        assert network.view().resolved_backend == "matplotlib"
        network.show()                                # must not raise
        assert network.view().figure() is not None

    def test_view_passes_options_through(self, network, no_graphviz):
        view = network.view(max_label=12, show_probabilities=False)
        assert view.max_label == 12
        assert view.show_probabilities is False
