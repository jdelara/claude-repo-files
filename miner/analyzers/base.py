from __future__ import annotations

from abc import ABC, abstractmethod


class Analyzer(ABC):
    """Every analyzer takes raw markdown text + light metadata and returns a
    JSON-serializable dict. Analyzers must not hit the network or depend on
    other analyzers' output -- they run independently over content already
    sitting in SQLite, so `miner analyze` never touches GitHub.
    """

    id: str = "base"

    @abstractmethod
    def analyze(self, content: str, meta: dict) -> dict:
        ...
