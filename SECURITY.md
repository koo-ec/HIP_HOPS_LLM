# Security policy

## Reporting a vulnerability

Please report security issues through GitHub's private vulnerability reporting
on this repository, rather than opening a public issue.

## Scope

This is a research and analysis library. It does not run untrusted code, open
network connections, or handle credentials — with two exceptions worth knowing
about.

**Source-code inspection.** The architecture extractor reads the *source text* of
your node functions (through `inspect.getsource`) to classify roles and detect
shared model snapshots. It never executes them. If the graph you analyse contains
secrets in literals, those literals are read into the `Component.source_code`
field and will appear in a saved Markdown report. Review artefacts before
publishing them.

**Vendored HIP-LLM.** `src/HIPLLM/` and `src/hip_llm/` include provider clients
that can make paid API calls. Those paths are inert unless you enable them
explicitly and supply keys; nothing in HIP-HOPS-LLM's own code path calls them.

## Supported versions

The latest release only, while the package is pre-1.0.
