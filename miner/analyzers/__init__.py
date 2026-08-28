from __future__ import annotations

from typing import List

from .base import Analyzer
from .structure import StructureAnalyzer
from .notation import NotationAnalyzer


def get_all_analyzers() -> List[Analyzer]:
    """Central registry. Add new analyzers here to have them run by
    `miner analyze` automatically."""
    return [
        StructureAnalyzer(),
        NotationAnalyzer(),
    ]
