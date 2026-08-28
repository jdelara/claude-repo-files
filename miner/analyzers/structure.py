from __future__ import annotations

import re

from .base import Analyzer

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
FENCE_RE = re.compile(r"^```(\S*)\s*\n(.*?)^```", re.MULTILINE | re.DOTALL)
BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


class StructureAnalyzer(Analyzer):
    id = "structure"

    def analyze(self, content: str, meta: dict) -> dict:
        headers = [(len(h), t.strip()) for h, t in HEADER_RE.findall(content)]
        fences = FENCE_RE.findall(content)
        fence_langs = {}
        for lang, _body in fences:
            key = lang.strip().lower() or "(untagged)"
            fence_langs[key] = fence_langs.get(key, 0) + 1

        lines = content.splitlines()
        words = content.split()

        return {
            "line_count": len(lines),
            "word_count": len(words),
            "char_count": len(content),
            "header_count": len(headers),
            "max_header_depth": max((h for h, _ in headers), default=0),
            "headers": [t for _, t in headers][:50],  # cap to keep rows small
            "code_block_count": len(fences),
            "code_block_languages": fence_langs,
            "bullet_count": len(BULLET_RE.findall(content)),
            "link_count": len(LINK_RE.findall(content)),
        }
