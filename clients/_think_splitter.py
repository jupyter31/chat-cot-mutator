"""Utilities for extracting <think> blocks from streamed responses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ThinkSplitResult:
    reasoning: str
    visible: str
    capped: bool


class ThinkSplitter:
    """Incrementally split streamed text into ``<think>`` blocks and visible output."""

    def __init__(
        self,
        start_tag: str = "<think>",
        end_tag: str = "</think>",
        cap_chars: Optional[int] = None,
    ) -> None:
        self.start_tag = start_tag
        self.end_tag = end_tag
        self._buf = ""
        self._state = "OUT"
        self._think_chunks: list[str] = []
        self._visible_chunks: list[str] = []
        self._cap = cap_chars
        self._think_len = 0
        self._end_guard = max(len(start_tag), len(end_tag)) - 1
        self._capped = False

    def feed(self, chunk: Optional[str]) -> None:
        """Consume a chunk of text produced by the model."""

        if not chunk:
            return
        s = self._buf + chunk
        i = 0
        while i < len(s):
            if self._state == "OUT":
                j = s.find(self.start_tag, i)
                if j == -1:
                    tail_from = max(len(s) - self._end_guard, i)
                    self._visible_chunks.append(s[i:tail_from])
                    self._buf = s[tail_from:]
                    return
                self._visible_chunks.append(s[i:j])
                i = j + len(self.start_tag)
                self._state = "IN"
            else:  # IN
                k = s.find(self.end_tag, i)
                if k == -1:
                    frag = s[i:]
                    if self._cap is None or self._think_len < self._cap:
                        take = frag if self._cap is None else frag[: self._cap - self._think_len]
                        if take != frag:
                            self._capped = True
                        self._think_chunks.append(take)
                        self._think_len += len(take)
                self._buf = ""
                return
            frag = s[i:k]
            if self._cap is None or self._think_len < self._cap:
                take = frag if self._cap is None else frag[: self._cap - self._think_len]
                if take != frag:
                    self._capped = True
                self._think_chunks.append(take)
                self._think_len += len(take)
                i = k + len(self.end_tag)
                self._state = "OUT"
        self._buf = s[-self._end_guard :] if len(s) >= self._end_guard else s

    def finish(self) -> ThinkSplitResult:
        """Finalize the split and return accumulated reasoning and visible text."""

        if self._state == "IN" and self._buf:
            if self._cap is None or self._think_len < self._cap:
                take = self._buf if self._cap is None else self._buf[: self._cap - self._think_len]
                if take != self._buf:
                    self._capped = True
                self._think_chunks.append(take)
                self._think_len += len(take)
        elif self._buf and self._state == "OUT":
            self._visible_chunks.append(self._buf)
        self._buf = ""

        reasoning = "".join(self._think_chunks).strip()
        visible = "".join(self._visible_chunks)
        return ThinkSplitResult(reasoning=reasoning, visible=visible, capped=self._capped)


__all__ = ["ThinkSplitter", "ThinkSplitResult"]
