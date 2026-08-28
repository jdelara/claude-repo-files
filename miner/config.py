"""Load the YAML-based notation and diagram-language catalog."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List

import yaml

DEFAULT_NOTATIONS_PATH = Path(__file__).parent.parent / "config" / "notations.yaml"


@dataclasses.dataclass
class NotationDef:
    id: str
    name: str
    fence_languages: List[str]
    untagged_heuristics: List[str] = dataclasses.field(default_factory=list)
    boundary_markers: List[str] = dataclasses.field(default_factory=list)
    subtype_markers: dict = dataclasses.field(default_factory=dict)


def load_notations(path: Path = DEFAULT_NOTATIONS_PATH) -> List[NotationDef]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for entry in data["notations"]:
        out.append(
            NotationDef(
                id=entry["id"],
                name=entry["name"],
                fence_languages=entry.get("fence_languages", []),
                untagged_heuristics=entry.get("untagged_heuristics", []),
                boundary_markers=entry.get("boundary_markers", []),
                subtype_markers=entry.get("subtype_markers", {}),
            )
        )
    return out
