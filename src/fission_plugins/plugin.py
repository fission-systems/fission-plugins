"""Plugin interface draft (v0, unstable).

A plugin is a pure function from one decompiled function to an `EditList`.
It runs out of process: the host (`fission-postproc`, planned) reads the JSON
array from `fission_cli decomp --json`, calls the plugin per function and applies
the edits with `apply_edits`. Plugins never see the binary, only the pseudocode
and Fission's variable table.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import EditList, FunctionRecord


@runtime_checkable
class PostProcessor(Protocol):
    id: str  # unique, kebab-case, matches plugins/<id>/plugin.toml
    version: str

    def propose(self, record: FunctionRecord) -> EditList:
        """Return advisory edits. Must be deterministic for a fixed model and seed."""
        ...


class Backend(Protocol):
    """A local text-generation runtime (llama.cpp, MLX, ...)."""

    name: str

    def complete(self, prompt: str, *, max_tokens: int, grammar: str | None = None) -> str:
        """Greedy completion. `grammar` is a GBNF string where the backend supports it."""
        ...
