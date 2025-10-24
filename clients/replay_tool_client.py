"""Replay-only tool client that reuses frozen tool outputs."""
from __future__ import annotations

from typing import Any, Iterable, List

from core.schema import FrozenTool, FrozenToolRecord


class ReplayToolClient:
    """Replay tool client enforcing strict frozen-context execution."""

    def __init__(self, tool_outputs: Iterable[FrozenTool | FrozenToolRecord]):
        self._records: List[FrozenToolRecord] = []
        for entry in tool_outputs:
            if isinstance(entry, FrozenToolRecord):
                self._records.append(entry)
            else:
                self._records.append(FrozenToolRecord.from_dict(entry))
        self._cursor = 0

    def reset(self) -> None:
        """Reset the playback cursor to the first tool output."""
        self._cursor = 0

    def call(self, tool_name: str, args: Any) -> Any:
        """Return the pre-recorded output for the requested tool call."""
        if self._cursor >= len(self._records):
            raise KeyError(
                f"No recorded tool outputs remaining when calling '{tool_name}' with args={args!r}."
            )
        record = self._records[self._cursor]
        if record.tool != tool_name:
            raise KeyError(
                f"Replay mismatch: expected tool '{record.tool}' but received '{tool_name}' at index {self._cursor}."
            )
        if record.input != args:
            raise ValueError(
                "Replay mismatch: provided arguments do not match frozen input "
                f"for tool '{tool_name}' at index {self._cursor}."
            )
        self._cursor += 1
        return record.output

    def remaining(self) -> int:
        """Return the number of unused tool outputs."""
        return len(self._records) - self._cursor


__all__ = ["ReplayToolClient"]
