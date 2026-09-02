"""Optional dependencies must not be imported at a notebook cell's top level.

This has now bitten three times — IPython, langgraph and pyagrum each broke every
CI job because a cell began with a bare ``import``. A notebook is documentation
that runs in environments its author does not control, so an optional package has
to be imported inside a ``try`` or a conditional, with a stated fallback.

The check is deliberately structural rather than executing anything: it parses
each cell and looks at where the import sits in the tree.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"

#: Packages a reader's environment may legitimately lack. Everything the package
#: itself depends on (numpy, pandas, matplotlib, scipy) may be imported freely.
OPTIONAL = {
    "pyagrum",
    "langgraph",
    "langchain_openai",
    "langchain_core",
    "openai",
    "IPython",
    "google",          # google.colab
    "typing_extensions",
}

_SHELL = re.compile(r"^\s*[!%]")


def _cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(
            line for line in "".join(cell["source"]).splitlines()
            if not _SHELL.match(line)
        )
        if source.strip():
            yield index, source


def _top_level_optional_imports(source: str) -> list[str]:
    """Optional packages imported directly in the module body of a cell."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - the cells all parse
        return []
    found = []
    for node in tree.body:            # body only: nested nodes are guarded
        if isinstance(node, ast.Import):
            found += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module.split(".")[0])
    return [name for name in found if name in OPTIONAL]


@pytest.mark.parametrize(
    "path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.stem
)
def test_optional_imports_are_guarded(path):
    offenders = [
        (index, sorted(set(names)))
        for index, source in _cells(path)
        if (names := _top_level_optional_imports(source))
    ]
    assert not offenders, (
        f"{path.name} imports optional packages at a cell's top level: "
        f"{offenders}. Put them inside a try/except or an `if`, with a stated "
        "fallback — a notebook runs in environments its author does not control."
    )


@pytest.mark.parametrize(
    "path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.stem
)
def test_the_package_itself_is_imported_plainly(path):
    """The converse: `hiphopsllm` is the point, so it should not be guarded."""
    source = "\n".join(src for _, src in _cells(path))
    assert "hiphopsllm" in source, f"{path.name} never imports the package"


def test_the_optional_set_covers_what_the_extras_declare():
    """Anything in an optional extra should be on the list this test checks."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = pyproject.split("[project.optional-dependencies]", 1)[1]
    block = block.split("\n[", 1)[0]
    declared = set(re.findall(r'"([A-Za-z][A-Za-z0-9_.-]*)', block))
    interesting = {
        name.lower().replace("-", "_")
        for name in declared
        if name.lower().split("[")[0] in {"pyagrum", "langgraph", "openai"}
    }
    assert interesting <= {n.lower() for n in OPTIONAL}, (
        f"an optional dependency is not covered by this check: {interesting}"
    )
