"""Notation-specific structural parsers.

These are intentionally lightweight regex-based extractors, not full
grammars -- good enough to answer "how big/connected is this diagram"
without pulling in a parser dependency per DSL. Register a function here
and key it into NOTATION_PARSERS by the notation's `id` from notations.yaml
to upgrade a notation from "snippet only" to "structural parse".

Any notation id with no entry here still gets detected/counted/snippeted by
the generic engine in notation.py -- it just falls back to
`generic_arrow_structure` for the "structure" field.
"""
from __future__ import annotations

import re
from typing import Optional

_ARROW_RE = re.compile(
    r"([A-Za-z0-9_\"'.]+)\s*(?:-{1,3}>{0,1}>?|\.{2,3}>|<\|--|--\|>|<-{1,3})\s*(?:\|[^|]*\|)?\s*([A-Za-z0-9_\"'.]+)"
)
_MERMAID_SEQ_RE = re.compile(r"^\s*([\w\"'.]+)\s*-{1,2}>{1,2}\s*([\w\"'.]+)\s*:\s*(.+)$")
_BNF_RULE_RE = re.compile(r"^\s*<?([\w-]+)>?\s*::=\s*(.+)$")


def generic_arrow_structure(body: str) -> dict:
    """Fallback: count arrow-like relations and pull the distinct endpoints.
    Used for graphviz/dot, plantuml, and any unregistered notation."""
    edges = []
    nodes = set()
    for line in body.splitlines():
        m = _ARROW_RE.search(line)
        if m:
            a, b = m.group(1).strip('"\''), m.group(2).strip('"\'')
            edges.append([a, b])
            nodes.update([a, b])
    return {"node_count": len(nodes), "edge_count": len(edges), "edges": edges[:100]}


def parse_mermaid_structure(body: str, subtype: Optional[str]) -> dict:
    if subtype == "sequence":
        edges = []
        participants = set()
        for line in body.splitlines():
            m = _MERMAID_SEQ_RE.match(line)
            if m:
                a, b, msg = m.group(1), m.group(2), m.group(3).strip()
                edges.append([a, b, msg])
                participants.update([a, b])
        return {"participant_count": len(participants), "message_count": len(edges),
                "messages": edges[:100]}

    if subtype == "state":
        transitions = []
        states = set()
        for line in body.splitlines():
            m = re.match(r"^\s*([\w\[\]*]+)\s*-->\s*([\w\[\]*]+)\s*(?::\s*(.+))?$", line)
            if m:
                a, b, event = m.group(1), m.group(2), (m.group(3) or "").strip()
                transitions.append([a, b, event])
                states.update([a, b])
        return {"state_count": len(states), "transition_count": len(transitions),
                "transitions": transitions[:100]}

    if subtype == "class":
        classes = set(re.findall(r"^\s*class\s+([\w]+)", body, re.MULTILINE))
        relations = generic_arrow_structure(body)
        return {"class_count": len(classes), "classes": sorted(classes)[:50],
                "relation_count": relations["edge_count"]}

    # flowchart / er / default -- generic node/edge extraction
    return generic_arrow_structure(body)


def parse_bnf_rules(body: str, subtype: Optional[str] = None) -> dict:
    rules = []
    for line in body.splitlines():
        m = _BNF_RULE_RE.match(line)
        if m:
            rules.append(m.group(1))
    return {"rule_count": len(rules), "rules": rules[:100]}


# id (from notations.yaml) -> callable(body, subtype) -> dict
NOTATION_PARSERS = {
    "mermaid": parse_mermaid_structure,
    "plantuml": lambda body, subtype: generic_arrow_structure(body),
    "graphviz": lambda body, subtype: generic_arrow_structure(body),
    "bnf_ebnf": parse_bnf_rules,
}
