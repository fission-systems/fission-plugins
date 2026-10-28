"""Data contracts between Fission, a post-processing plugin and the applier.

`FunctionRecord` mirrors the subset of `fission_cli decomp --json` that plugins
may read. `EditList` is the only thing a plugin may return: a plugin never emits
replacement code, so a model cannot change what the pseudocode computes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variable:
    name: str
    kind: str  # "arg" | "reg" | "stack" as emitted by Fission
    type: str
    arg_index: int | None = None


@dataclass(frozen=True)
class FunctionRecord:
    address: str
    name: str
    code: str
    variables: tuple[Variable, ...] = ()

    @classmethod
    def from_fission_json(cls, row: dict) -> "FunctionRecord":
        """Build from one element of the `fission_cli decomp ... --json` array.

        Rows that failed to decompile carry `error` and no `code`; callers must
        skip them rather than pass them to a plugin.
        """
        if "code" not in row:
            raise ValueError(f"no code for {row.get('name')!r}: {row.get('error')}")
        variables = tuple(
            Variable(v["name"], v["kind"], v["type"], v.get("arg_index"))
            for v in row.get("variables", [])
        )
        return cls(row["address"], row["name"], row["code"], variables)


@dataclass(frozen=True)
class Comment:
    line: int  # 1-based line of `FunctionRecord.code`
    text: str


@dataclass
class EditList:
    """Advisory edits proposed by a plugin; the applier decides what is allowed."""

    function_name: str | None = None
    renames: dict[str, str] = field(default_factory=dict)  # old variable -> new
    comments: list[Comment] = field(default_factory=list)
    # Type suggestions are reported but never applied in v0.
    type_hints: dict[str, str] = field(default_factory=dict)


@dataclass
class ApplyResult:
    code: str
    applied_renames: dict[str, str]
    rejected: list[tuple[str, str]]  # (what, why)
