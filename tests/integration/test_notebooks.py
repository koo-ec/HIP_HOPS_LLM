"""Execute the repository's notebooks, so a stale one fails loudly.

Notebooks that no longer run are worse than no notebooks: they are documentation
that looks maintained. These tests execute every code cell in-process, skipping
only the cells whose whole purpose is to install the package (``!git clone``,
``%pip install``), which have already happened by definition if the test itself
is running.
"""

from __future__ import annotations

import json
import pathlib
import re

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"

#: Lines that only make sense inside a hosted notebook.
_SHELL = re.compile(r"^\s*[!%]")


def _notebooks() -> list[pathlib.Path]:
    return sorted(NOTEBOOKS.glob("*.ipynb"))


def _executable_source(cell: dict) -> str:
    """The cell's code with shell/magic lines and bare-expression displays removed."""
    lines = [line for line in "".join(cell["source"]).splitlines() if not _SHELL.match(line)]
    return "\n".join(lines)


@pytest.mark.parametrize(
    "path", _notebooks(), ids=lambda p: p.stem if hasattr(p, "stem") else str(p)
)
class TestNotebookRuns:
    def test_it_is_valid_and_has_no_baked_output(self, path):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        for cell in notebook["cells"]:
            assert cell.get("id"), "every cell needs a stable id"
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [], (
                    f"{path.name} has committed output; regenerate it with "
                    "scripts/build_notebooks.py"
                )

    def test_every_code_cell_executes(self, path, tmp_path, monkeypatch):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        monkeypatch.chdir(tmp_path)
        # The Colab notebook makes real, paid API calls when a key is present.
        # Tests must never spend money or depend on a provider being up, so the
        # key is cleared and the notebook takes its documented offline path.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        namespace: dict = {
            "__name__": "__notebook__",
            # `display` is provided by IPython at run time; a print keeps the
            # cells honest here without pulling IPython into the test env.
            "display": lambda *args, **kwargs: None,
            "get_ipython": lambda: None,
        }
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = _executable_source(cell)
            if not source.strip():
                continue
            try:
                exec(compile(source, f"{path.name}:cell{index}", "exec"), namespace)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"{path.name} cell {index} raised {type(exc).__name__}: {exc}\n"
                    f"--- source ---\n{source}"
                )


def test_the_notebooks_are_up_to_date_with_their_generator(tmp_path):
    """Regenerating must reproduce what is committed, byte for byte.

    Otherwise the generator and the notebook drift, and the next person to run
    the generator produces a diff nobody intended.
    """
    import subprocess
    import sys

    committed = {p.name: p.read_text(encoding="utf-8") for p in _notebooks()}
    assert committed, "no notebooks found"

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_notebooks.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr

    stale = [
        name
        for name, text in committed.items()
        if (NOTEBOOKS / name).read_text(encoding="utf-8") != text
    ]
    assert not stale, (
        f"{stale} differ from what scripts/build_notebooks.py produces; "
        "commit the regenerated notebooks"
    )


def test_hosted_notebooks_use_an_immediately_importable_install():
    """A running kernel does not re-process the .pth file from ``pip -e``."""
    for path in _notebooks():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert '%pip install -q "./hiphopsllm-repo[bayes,graph]"' in source
        assert '%pip install -q -e "hiphopsllm-repo[bayes,graph]"' not in source
