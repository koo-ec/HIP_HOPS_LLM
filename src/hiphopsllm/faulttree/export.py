"""Export of synthesised fault trees and reports.

``to_mermaid``
    Fault tree as a mermaid flowchart — renders in a notebook the same way the
    LangGraph diagram does.
``to_dot``
    Graphviz DOT with conventional fault-tree shapes.
``to_json``
    Machine-readable tree, for diffing across releases.
``to_openpsa_xml``
    Open-PSA MEF, so the tree can be opened in an external fault-tree tool
    (XFTA, SCRAM) rather than trusted blindly.
``markdown_report``
    The full safety-analysis document, including the loop handling and the
    common-cause notes.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence
from xml.sax.saxutils import escape as _xml_escape

from ..architecture.acyclic import CycleReport
from .analysis import TreeAnalysis, fmea_table, single_points_of_failure
from ..architecture.model import SystemModel
from .failure import FailureModel
from .synthesis import FaultTree, FTNode

__all__ = [
    "to_mermaid",
    "to_dot",
    "to_json",
    "to_openpsa_xml",
    "markdown_report",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mid(nid: str) -> str:
    return "n_" + re.sub(r"[^0-9A-Za-z]", "_", nid)


def _mlabel(text: str, width: int = 46) -> str:
    """Escape and soft-wrap a label; each source line wraps independently."""
    out: List[str] = []
    for source_line in (text or "").replace('"', "#quot;").split("\n"):
        line = ""
        for word in source_line.split(" "):
            if len(line) + len(word) + 1 > width and line:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return "<br/>".join(part for part in out if part)


_GATE_SYMBOL = {"AND": "AND &amp;", "OR": "OR ≥1"}


# --------------------------------------------------------------------------- #
# mermaid
# --------------------------------------------------------------------------- #
def to_mermaid(tree: FaultTree, direction: str = "TB", show_gate: bool = True) -> str:
    """Render the fault tree as mermaid text.

    Gates are drawn as their own nodes between an event and its causes, which is
    as close to the conventional AND/OR symbols as mermaid gets; the matplotlib
    renderer (:mod:`hiphopsllm.viz.plots`) draws the real symbols. Shared sub-trees (the
    equivalent of a transfer gate) appear once and are referenced by several
    parents, outlined so a shared cause is visible as such.
    """
    shared = set(tree.shared_nodes())
    lines = [f"graph {direction};"]
    for nid, node in tree.nodes.items():
        label = _mlabel(node.label)
        mid = _mid(nid)
        if node.ntype == "top":
            lines.append(f'    {mid}[["{label}"]]:::top')
        elif node.ntype == "basic":
            lines.append(f'    {mid}(("{label}")):::basic')
        elif node.ntype == "undeveloped":
            lines.append(f'    {mid}[/"{label}"/]:::undeveloped')
        elif node.ntype == "house":
            lines.append(f'    {mid}{{"{label}"}}:::house')
        else:
            cls = "shared" if nid in shared else "intermediate"
            lines.append(f'    {mid}["{label}"]:::{cls}')
        if show_gate and node.gate and node.children:
            lines.append(f'    {mid}_gate{{{{"{_GATE_SYMBOL.get(node.gate, node.gate)}"}}}}:::gate')
    for nid, node in tree.nodes.items():
        source = _mid(nid)
        if show_gate and node.gate and node.children:
            lines.append(f"    {source} --- {source}_gate;")
            source = f"{source}_gate"
        for child in node.children:
            lines.append(f"    {source} --> {_mid(child)};")
    lines += [
        "    classDef top fill:#8B2E2E,color:#fff,stroke:#4A1414,stroke-width:2px;",
        "    classDef intermediate fill:#254E58,color:#fff,stroke:#112D32;",
        "    classDef shared fill:#254E58,color:#fff,stroke:#E6B89C,stroke-width:3px,stroke-dasharray:4 3;",
        "    classDef basic fill:#F5E9C9,color:#112D32,stroke:#88BDBC;",
        "    classDef undeveloped fill:#E6B89C,color:#112D32,stroke:#8B5E3C,stroke-dasharray:5 3;",
        "    classDef house fill:#C5D8D1,color:#112D32,stroke:#254E58;",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# graphviz
# --------------------------------------------------------------------------- #
def to_dot(tree: FaultTree) -> str:
    """Graphviz DOT using conventional fault-tree shapes."""
    def esc(text: str) -> str:
        return (text or "").replace('"', r"\"").replace("\n", r"\n")

    lines = [
        "digraph fault_tree {",
        "  rankdir=TB;",
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [arrowhead=none];',
    ]
    for nid, node in tree.nodes.items():
        label = esc(node.label)
        if node.ntype == "top":
            shape = 'shape=box, style="bold,filled", fillcolor="#F3D2D2"'
        elif node.ntype == "basic":
            shape = 'shape=circle, style=filled, fillcolor="#F5E9C9"'
        elif node.ntype == "undeveloped":
            shape = 'shape=diamond, style=filled, fillcolor="#E6B89C"'
        elif node.ntype == "house":
            shape = 'shape=house, style=filled, fillcolor="#C5D8D1"'
        else:
            shape = 'shape=box, style=filled, fillcolor="#DCE7EA"'
        lines.append(f'  "{_mid(nid)}" [label="{label}", {shape}];')
        if node.gate and node.children:
            gshape = "invhouse" if node.gate == "OR" else "invtrapezium"
            lines.append(f'  "{_mid(nid)}_gate" [label="{node.gate}", shape={gshape}, '
                         f'style=filled, fillcolor="#ffffff", height=0.35];')
    for nid, node in tree.nodes.items():
        source = _mid(nid)
        if node.gate and node.children:
            lines.append(f'  "{source}" -> "{source}_gate";')
            source = f"{source}_gate"
        for child in node.children:
            lines.append(f'  "{source}" -> "{_mid(child)}";')
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# json
# --------------------------------------------------------------------------- #
def to_json(tree: FaultTree, analysis: Optional[TreeAnalysis] = None, indent: int = 2) -> str:
    payload: Dict[str, object] = {
        "hazard": {
            "id": tree.id,
            "name": tree.name,
            "severity": tree.hazard.severity if tree.hazard else None,
            "description": tree.hazard.description if tree.hazard else "",
            "detection": tree.hazard.detection if tree.hazard else "",
            "top_deviations": [str(d) for d in (tree.hazard.deviations if tree.hazard else [])],
        },
        "root": tree.root,
        "nodes": {
            nid: {
                "type": n.ntype,
                "gate": n.gate,
                "label": n.label,
                "children": n.children,
                "deviation": n.deviation,
                "event_id": n.event_id,
                "component": n.component,
                "detail": n.detail,
            }
            for nid, n in tree.nodes.items()
        },
        "warnings": tree.warnings,
    }
    if analysis is not None:
        payload["minimal_cut_sets"] = [sorted(cs) for cs in analysis.cuts.sets]
        payload["truncated"] = analysis.cuts.truncated
        payload["quantification"] = {
            "top_probability": analysis.quant.top_probability,
            "interval": list(analysis.quant.top_interval),
            "method": analysis.quant.method,
            "rare_event_sum": analysis.quant.rare_event_sum,
        }
        payload["importance"] = [
            {
                "event": r.event_id, "component": r.component,
                "fussell_vesely": r.fussell_vesely, "birnbaum": r.birnbaum,
                "single_point": r.single_point, "probability": r.probability,
            }
            for r in analysis.importance
        ]
    return json.dumps(payload, indent=indent)


# --------------------------------------------------------------------------- #
# Open-PSA MEF
# --------------------------------------------------------------------------- #
def to_openpsa_xml(tree: FaultTree, name: Optional[str] = None) -> str:
    """Export in Open-PSA Model Exchange Format for an external FT engine.

    Being able to re-run the cut sets in an independent tool is the cheapest
    available check on this implementation, so the export is part of the normal
    output rather than an extra.
    """
    ft_name = re.sub(r"[^0-9A-Za-z_]", "_", name or f"FT_{tree.id}")
    gates: List[str] = []
    events: Dict[str, FTNode] = {}
    houses: Dict[str, FTNode] = {}

    def ref(nid: str) -> str:
        node = tree.nodes[nid]
        if node.ntype in ("basic", "undeveloped"):
            events[nid] = node
            return f'<basic-event name="{_mid(nid)}"/>'
        if node.ntype == "house":
            houses[nid] = node
            return f'<house-event name="{_mid(nid)}"/>'
        return f'<gate name="{_mid(nid)}"/>'

    for nid, node in tree.nodes.items():
        if node.ntype in ("basic", "undeveloped", "house"):
            continue
        children = [ref(c) for c in node.children]
        if not children:
            continue
        op = (node.gate or "or").lower()
        if len(children) == 1:
            body = f"<or>{children[0]}</or>"
        else:
            body = f"<{op}>{''.join(children)}</{op}>"
        comment = _xml_escape(node.label)
        gates.append(
            f'    <define-gate name="{_mid(nid)}">\n'
            f'      <label>{comment}</label>\n'
            f'      {body}\n'
            f"    </define-gate>"
        )

    event_defs: List[str] = []
    for nid, node in events.items():
        prob = 0.05
        ev = tree.events.get(node.event_id or "")
        if ev is not None:
            prob = ev.prob
        event_defs.append(
            f'    <define-basic-event name="{_mid(nid)}">\n'
            f"      <label>{_xml_escape(node.label)}</label>\n"
            f"      <float value=\"{prob}\"/>\n"
            f"    </define-basic-event>"
        )

    if not gates:
        # a fault tree with no gate will not load; say "no modelled cause"
        gates.append(
            f'    <define-gate name="{_mid(tree.root)}">\n'
            f"      <label>no modelled cause for this hazard</label>\n"
            f'      <constant value="false"/>\n'
            f"    </define-gate>"
        )

    for nid, node in houses.items():
        # a referenced house event must also be declared, or the file will not
        # load in an external solver
        event_defs.append(
            f'    <define-house-event name="{_mid(nid)}">\n'
            f"      <label>{_xml_escape(node.label)}</label>\n"
            f'      <constant value="true"/>\n'
            f"    </define-house-event>"
        )

    return (
        '<?xml version="1.0"?>\n'
        "<opsa-mef>\n"
        f'  <define-fault-tree name="{ft_name}">\n'
        + "\n".join(gates)
        + "\n  </define-fault-tree>\n"
        "  <model-data>\n"
        + "\n".join(event_defs)
        + "\n  </model-data>\n"
        "</opsa-mef>\n"
    )


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return "_(none)_\n"
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(r.get(c, "")).replace("|", "\\|") for c in columns) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body]) + "\n"


def markdown_report(
    system: SystemModel,
    fmodel: FailureModel,
    analyses: Dict[str, TreeAnalysis],
    cycle_report: Optional[CycleReport] = None,
    title: Optional[str] = None,
    include_trees: bool = True,
) -> str:
    """The full HiP-HOPS-style analysis document."""
    out: List[str] = []
    add = out.append

    add(f"# Fault tree analysis — {title or system.name}\n")
    add("Synthesised from the LangGraph architecture with a HiP-HOPS-style "
        "compositional method: components annotated with local failure logic, trees "
        "generated by traversing the connections, cut sets and FMEA derived from the "
        "trees.\n")

    # 1. Architecture
    add("## 1. Architecture\n")
    add(_table(system.architecture_table(),
               ["component", "role", "in_ports", "out_ports", "resources"]))
    add("\n**Connections**\n")
    add(_table(
        [{"from": f"{c.src}.{c.src_port}", "to": f"{c.dst}.{c.dst_port}",
          "branch": c.label or "-",
          "kind": "conditional" if c.conditional else ("fan-in" if c.fan_in else "direct")}
         for c in system.connections],
        ["from", "to", "branch", "kind"]))

    # 2. Loop handling
    add("\n## 2. Loop handling (why this tree is static)\n")
    if cycle_report is not None:
        add("```\n" + cycle_report.summary() + "\n```\n")
    else:
        add("_No loop analysis was recorded._\n")
    add("A fault tree cannot contain a cycle. Feedback edges in the agent graph are "
        "unrolled to the stated depth and then closed by a feedback-cut component that "
        "delivers the loop's deviations to the system boundary, so the acyclic model "
        "remains conservative with respect to the cyclic system.\n")

    # 3. Common cause
    add("\n## 3. Common-cause groups\n")
    groups = system.common_cause_groups()
    if groups:
        add(_table(
            [{"resource": f"{k[0]} = {k[1]}", "components": ", ".join(v),
              "consequence": "redundancy between these components is not independent"}
             for k, v in sorted(groups.items())],
            ["resource", "components", "consequence"]))
    else:
        add("_No shared resources were detected._\n")

    # 4. Local failure logic
    add("\n## 4. Component failure annotations (IF-FMEA)\n")
    for cid in sorted(fmodel.logic):
        cfl = fmodel.logic[cid]
        if not cfl.logic:
            continue
        comp = system.components[cid]
        add(f"\n### `{cid}` — {comp.role.value}\n")
        add(_table(cfl.table(), ["output_deviation", "failure_class", "expression"]))
        for note in cfl.notes:
            add(f"\n> {note}\n")

    # 5. Hazards and trees
    add("\n## 5. Hazards, fault trees and cut sets\n")
    for hid in sorted(analyses):
        analysis = analyses[hid]
        tree, cuts, quant = analysis.tree, analysis.cuts, analysis.quant
        hazard = tree.hazard
        add(f"\n### {hid} — {tree.name}\n")
        if hazard:
            add(f"- **Severity:** {hazard.severity}\n"
                f"- **Top event:** {', '.join(str(d) for d in hazard.deviations)}\n"
                f"- **Detection:** {hazard.detection}\n\n{hazard.description}\n")
        size = tree.size()
        add(f"\nTree: {size.get('total', 0)} nodes "
            f"({size.get('basic', 0)} basic, {size.get('intermediate', 0)} intermediate, "
            f"{size.get('undeveloped', 0)} undeveloped), depth {tree.depth()}, "
            f"acyclic: {tree.verify_acyclic()}.\n")
        add(f"\n**Top-event probability (MCUB):** {quant.top_probability:.4f}"
            + (f"  interval [{quant.top_interval[0]:.4f}, {quant.top_interval[1]:.4f}]"
               if quant.imprecise else "")
            + "\n")

        by_order = cuts.by_order()
        add("\n**Minimal cut sets by order**\n")
        add(_table(
            [{"order": order, "count": len(sets),
              "cut sets": "; ".join("{" + ", ".join(sorted(cs)) + "}" for cs in sets[:6])
                          + (" ..." if len(sets) > 6 else "")}
             for order, sets in by_order.items()],
            ["order", "count", "cut sets"]))
        if cuts.order_1():
            add("\n> **Single points of failure:** "
                + ", ".join(f"`{e}`" for e in cuts.order_1()) + "\n")
        for note in cuts.notes + quant.notes:
            add(f"\n_{note}_\n")
        if tree.warnings:
            for w in tree.warnings:
                add(f"\n**Warning:** {w}\n")

        add("\n**Top contributors**\n")
        add(_table(
            [{"event": r.event_id, "component": r.component, "failure mode": r.label,
              "P": f"{r.probability:.3f}", "FV": f"{r.fussell_vesely:.3f}",
              "Birnbaum": f"{r.birnbaum:.3f}",
              "single point": "yes" if r.single_point else ""}
             for r in analysis.importance[:10]],
            ["event", "component", "failure mode", "P", "FV", "Birnbaum", "single point"]))

        if include_trees:
            add("\n<details><summary>Fault tree (mermaid)</summary>\n\n```mermaid\n")
            add(to_mermaid(tree))
            add("\n```\n\n</details>\n")

    # 6. Single points of failure
    add("\n## 6. Single points of failure across all hazards\n")
    add(_table(single_points_of_failure(analyses),
               ["hazard", "severity", "component", "event", "failure_mode", "mitigation"]))

    # 7. FMEA
    add("\n## 7. FMEA (generated from the trees)\n")
    rows = fmea_table(analyses)
    add(_table(
        [{"component": r.component, "failure mode": f"{r.event_id} — {r.failure_mode}",
          "class": r.failure_class, "P": f"{r.probability:.3f}",
          "direct effects": ", ".join(r.direct_effects) or "-",
          "further effects": ", ".join(r.further_effects) or "-",
          "max severity": r.max_severity, "mitigation": r.mitigation}
         for r in rows],
        ["component", "failure mode", "class", "P", "direct effects",
         "further effects", "max severity", "mitigation"]))

    # 8. Evidence
    add("\n## 8. Basic event register\n")
    add(_table(
        [{"event": e.id, "component": e.component, "class": e.fclass.title,
          "P": f"{e.prob:.3f}", "kind": e.kind, "evidence": e.evidence}
         for e in sorted(fmodel.events.values(), key=lambda x: x.id)],
        ["event", "component", "class", "P", "kind", "evidence"]))
    add("\n> Probabilities marked _engineering judgement_ are placeholders. They order the "
        "contributors sensibly but must be replaced by measurement before any quantitative "
        "claim is made. Events marked _measured_ are calibrated from the run's semantic "
        "uncertainty records.\n")
    for note in fmodel.notes:
        add(f"\n> {note}\n")
    return "\n".join(out)
