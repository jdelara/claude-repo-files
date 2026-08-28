from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from ..config import NotationDef, load_notations
from .base import Analyzer
from .notation_parsers import NOTATION_PARSERS, generic_arrow_structure

FENCE_RE = re.compile(r"^```(\S*)[ \t]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
SNIPPET_MAX_CHARS = 600


class NotationAnalyzer(Analyzer):
    """Detects diagrams-as-text / DSLs embedded in markdown: Mermaid,
    PlantUML, Graphviz/DOT, ASCII-art, BNF/EBNF, regex listings, JSON/YAML
    schemas, etc. Fully driven by config/notations.yaml -- new notations
    require no code changes; deeper structural parsing (nodes/edges/states)
    is opt-in per notation via notation_parsers.NOTATION_PARSERS.
    """

    id = "notation"

    def __init__(self, notations: Optional[List[NotationDef]] = None):
        self.notations = notations or load_notations()
        self._by_fence_lang = {}
        for n in self.notations:
            for lang in n.fence_languages:
                self._by_fence_lang[lang.lower()] = n

    def analyze(self, content: str, meta: dict) -> dict:
        detections = []

        # --- Pass 1: fenced code blocks, matched by language tag (high confidence) ---
        for lang, body in FENCE_RE.findall(content):
            lang_key = lang.strip().lower()
            notation = self._by_fence_lang.get(lang_key)
            if notation:
                detections.append(self._build_detection(notation, body, confidence="high", via="fence_language"))
                continue
            # Untagged or unrecognized-tag block: check heuristics of every notation
            # that also declares an untagged path (skip ones matched by fence already).
            for n in self.notations:
                if lang_key in [fl.lower() for fl in n.fence_languages]:
                    continue
                if self._heuristics_match(n, body):
                    detections.append(self._build_detection(n, body, confidence="medium", via="untagged_heuristic"))

        # --- Pass 2: boundary-marker blocks outside fences (e.g. @startuml/@enduml) ---
        for n in self.notations:
            if not n.boundary_markers or len(n.boundary_markers) != 2:
                continue
            start, end = n.boundary_markers
            pattern = re.compile(re.escape(start) + r"(.*?)" + re.escape(end), re.DOTALL)
            for body in pattern.findall(content):
                # avoid double-counting if this was already inside a matched fence
                if any(d["snippet"].strip().startswith(body.strip()[:40]) for d in detections if d["notation_id"] == n.id):
                    continue
                detections.append(self._build_detection(n, body, confidence="high", via="boundary_marker"))

        # --- Aggregate ---
        by_notation = {}
        for d in detections:
            by_notation.setdefault(d["notation_id"], []).append(d)

        summary = {
            nid: {
                "count": len(items),
                "confidences": sorted({i["confidence"] for i in items}),
                "subtypes": sorted({i["subtype"] for i in items if i["subtype"]}),
            }
            for nid, items in by_notation.items()
        }

        return {
            "any_notation_detected": bool(detections),
            "notation_count_total": len(detections),
            "notations_summary": summary,
            "detections": detections[:30],  # cap payload size
        }

    # -- helpers ----------------------------------------------------------

    def _heuristics_match(self, notation: NotationDef, body: str) -> bool:
        for pattern in notation.untagged_heuristics:
            try:
                if re.search(pattern, body):
                    return True
            except re.error:
                if pattern in body:
                    return True
        return False

    def _detect_subtype(self, notation: NotationDef, body: str) -> Optional[str]:
        first_lines = "\n".join(body.strip().splitlines()[:3]).lower()
        for subtype, markers in notation.subtype_markers.items():
            if any(m.lower() in first_lines for m in markers):
                return subtype
        return None

    def _build_detection(self, notation: NotationDef, body: str, confidence: str, via: str) -> dict:
        subtype = self._detect_subtype(notation, body)
        parser = NOTATION_PARSERS.get(notation.id)
        structure = parser(body, subtype) if parser else generic_arrow_structure(body)
        snippet = body.strip()
        if len(snippet) > SNIPPET_MAX_CHARS:
            snippet = snippet[:SNIPPET_MAX_CHARS] + "…"
        return {
            "notation_id": notation.id,
            "notation_name": notation.name,
            "subtype": subtype,
            "confidence": confidence,
            "detected_via": via,
            "snippet": snippet,
            "structure": structure,
        }
