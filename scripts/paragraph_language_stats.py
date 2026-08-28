"""Estimate the natural languages of prose in ``CLAUDE.md`` files.

The analyzer is deliberately conservative. It removes Markdown/code-like
material, ignores short paragraphs, requires fastText and Lingua to agree at
strict confidence and margin thresholds, checks non-Latin scripts, and
requires the prediction to survive a light normalization. Ambiguous cases are
reported as ``unknown`` rather than forced into a language.

Exact duplicate contents are parsed and classified once. Outputs include both
copy-weighted results and a content-deduplicated view, plus a deterministic
paragraph sample for a small manual audit.

The source database is always opened with ``mode=ro&immutable=1`` and
``PRAGMA query_only=ON``. No classifications are written to it.

Install the optional dependencies and download the official model once:

    python -m pip install -r requirements-language.txt
    python scripts/paragraph_language_stats.py --download-model \
        --db mined.db \
        --distribution-output article/natural_language_distribution.csv \
        --families-output article/document_language_families.csv \
        --audit-output article/language_manual_audit.csv \
        --summary-output article/natural_language_summary.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import html
import importlib.metadata
import json
import math
import re
import sqlite3
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


SCOPES = ("exact-claude", "markdown", "all")
FASTTEXT_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)
FASTTEXT_MODEL_BYTES = 131_266_198
FASTTEXT_MODEL_SHA256 = (
    "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
)

DEFAULT_MIN_WORDS = 20
DEFAULT_MIN_ALPHA_CHARS = 80
DEFAULT_MAX_CLASSIFIER_CHARS = 4_000
DEFAULT_MIN_CONFIDENCE = 0.90
DEFAULT_MIN_MARGIN = 0.20
DEFAULT_MIN_DOCUMENT_WORDS = 200
DEFAULT_MIN_DOCUMENT_COVERAGE = 0.60
DEFAULT_DOMINANT_LANGUAGE_SHARE = 0.85
DEFAULT_MULTILINGUAL_SECONDARY_SHARE = 0.15
DEFAULT_MULTILINGUAL_SECONDARY_WORDS = 50
DEFAULT_SENSITIVITY_THRESHOLDS = (0.80, 0.90, 0.95)

LANGUAGE_CODE_ALIASES = {
    # Lingua represents Norwegian Bokmal as ``nb``; fastText's corresponding
    # general Norwegian label is ``no``.
    "nb": "no",
}

FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
ATX_HEADING_RE = re.compile(r"^[ ]{0,3}#{1,6}(?:[ \t]+|$)(.*)$")
SETEXT_RE = re.compile(r"^[ ]{0,3}(?:=+|-+)[ \t]*$")
HORIZONTAL_RULE_RE = re.compile(
    r"^[ ]{0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$"
)
LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+(.*)$")
LINK_DEFINITION_RE = re.compile(r"^[ ]{0,3}\[[^]]+\]:[ \t]*\S+")
YAML_KEY_RE = re.compile(r"^[A-Za-z_][\w.-]*[ \t]*:")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_CODE_RE = re.compile(
    r"<(?:pre|code)\b[^>]*>.*?</(?:pre|code)>", re.IGNORECASE | re.DOTALL
)
INLINE_CODE_RE = re.compile(r"(`+)(.*?)\1", re.DOTALL)
MARKDOWN_LINK_RE = re.compile(r"!?\[([^]]*)\]\([^)]+\)")
REFERENCE_LINK_RE = re.compile(r"\[([^]]+)\]\[[^]]*\]")
AUTOLINK_RE = re.compile(r"<(?:(?:https?://|mailto:)[^>]+)>", re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
TEMPLATE_RE = re.compile(r"\{[{%].*?[}%]\}")
PATH_TOKEN_RE = re.compile(
    r"(?<!\w)(?:(?:[A-Za-z]:)?[./~][\w.@%+~:/\\-]+|"
    r"[\w.@%+~-]+(?:/|\\)[\w.@%+~:/\\-]+)(?!\w)"
)
FLAG_RE = re.compile(r"(?<!\w)--?[A-Za-z][\w-]*")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)


@dataclass(frozen=True)
class ProseParagraph:
    index: int
    text: str
    word_count: int
    alphabetic_characters: int


@dataclass(frozen=True)
class Prediction:
    language: str
    confidence: float
    margin: float


@dataclass(frozen=True)
class ParagraphDecision:
    paragraph: ProseParagraph
    detector_text: str
    fasttext: Prediction
    lingua: Prediction
    dominant_script: str
    stable_after_normalization: bool
    decision: str
    accepted_language: str | None


@dataclass(frozen=True)
class AnalysisConfig:
    min_words: int = DEFAULT_MIN_WORDS
    min_alpha_characters: int = DEFAULT_MIN_ALPHA_CHARS
    max_classifier_characters: int = DEFAULT_MAX_CLASSIFIER_CHARS
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_margin: float = DEFAULT_MIN_MARGIN
    min_document_accepted_words: int = DEFAULT_MIN_DOCUMENT_WORDS
    min_document_coverage: float = DEFAULT_MIN_DOCUMENT_COVERAGE
    dominant_language_share: float = DEFAULT_DOMINANT_LANGUAGE_SHARE
    multilingual_secondary_share: float = DEFAULT_MULTILINGUAL_SECONDARY_SHARE
    multilingual_secondary_words: int = DEFAULT_MULTILINGUAL_SECONDARY_WORDS
    sensitivity_confidence_thresholds: tuple[float, ...] = (
        DEFAULT_SENSITIVITY_THRESHOLDS
    )


@dataclass(frozen=True)
class ContentFamilyResult:
    content_hash: str
    copies: int
    representative_repo: str
    representative_path: str
    representative_url: str | None
    size_bytes: int | None
    extracted_paragraphs: int
    eligible_paragraphs: int
    eligible_words: int
    detector_agreements: int
    stable_detector_agreements: int
    accepted_paragraphs: int
    accepted_words: int
    accepted_word_coverage: float
    document_class: str
    primary_language: str | None
    dominant_language_share: float
    document_decision: str
    accepted_language_words: dict[str, int]


class LanguageDetector(Protocol):
    name: str
    version: str
    language_names: dict[str, str]

    def predict(self, text: str) -> Prediction:
        """Return the highest-ranked language, confidence, and top-two margin."""


class PredictionCache:
    """A resumable derived cache that is always separate from ``mined.db``."""

    def __init__(self, cache_path: str | Path, *, source_db_path: str | Path):
        path = Path(cache_path).resolve()
        source = Path(source_db_path).resolve()
        if path == source:
            raise ValueError("prediction cache must not be the source database")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                   detector_key TEXT NOT NULL,
                   text_sha256 TEXT NOT NULL,
                   language TEXT NOT NULL,
                   confidence REAL NOT NULL,
                   margin REAL NOT NULL,
                   PRIMARY KEY (detector_key, text_sha256)
               )"""
        )
        self._connection.commit()
        self._pending = 0

    def get(self, detector_key: str, text: str) -> Prediction | None:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self._connection.execute(
            """SELECT language, confidence, margin
               FROM predictions
               WHERE detector_key = ? AND text_sha256 = ?""",
            (detector_key, text_hash),
        ).fetchone()
        if row is None:
            return None
        return Prediction(str(row[0]), float(row[1]), float(row[2]))

    def put(self, detector_key: str, text: str, prediction: Prediction) -> None:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._connection.execute(
            """INSERT OR IGNORE INTO predictions
               (detector_key, text_sha256, language, confidence, margin)
               VALUES (?, ?, ?, ?, ?)""",
            (
                detector_key,
                text_hash,
                prediction.language,
                prediction.confidence,
                prediction.margin,
            ),
        )
        self._pending += 1
        if self._pending >= 1_000:
            self._connection.commit()
            self._pending = 0

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()


class CachedDetector:
    """Cache a detector's deterministic predictions by detector/text hash."""

    def __init__(
        self,
        detector: LanguageDetector,
        cache: PredictionCache,
        detector_key: str,
    ):
        self._detector = detector
        self._cache = cache
        self._detector_key = detector_key
        self.name = detector.name
        self.version = detector.version
        self.language_names = detector.language_names
        self.cache_hits = 0
        self.cache_misses = 0

    def predict(self, text: str) -> Prediction:
        cached = self._cache.get(self._detector_key, text)
        if cached is not None:
            self.cache_hits += 1
            return cached
        prediction = self._detector.predict(text)
        self._cache.put(self._detector_key, text, prediction)
        self.cache_misses += 1
        return prediction


def _normalize_language_code(value: str) -> str:
    code = value.strip().lower().removeprefix("__label__")
    return LANGUAGE_CODE_ALIASES.get(code, code)


def _scope_predicate(scope: str, *, alias: str = "") -> str:
    if scope not in SCOPES:
        raise ValueError(f"unsupported scope: {scope!r}")
    prefix = f"{alias}." if alias else ""
    if scope == "exact-claude":
        return (
            f"(lower({prefix}path) = 'claude.md' "
            f"OR lower({prefix}path) LIKE '%/claude.md')"
        )
    if scope == "markdown":
        return f"lower({prefix}path) LIKE '%.md'"
    return "1 = 1"


def _scope_definition(scope: str) -> str:
    return {
        "exact-claude": "case-insensitive basename equal to CLAUDE.md",
        "markdown": "case-insensitive paths ending in .md",
        "all": "all stored file records",
    }[scope]


def _readonly_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fasttext_model(model_path: str | Path) -> dict[str, Any]:
    """Verify the published fastText model by size and SHA-256."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"fastText model not found: {path}")
    size = path.stat().st_size
    digest = _sha256_file(path)
    if size != FASTTEXT_MODEL_BYTES or digest != FASTTEXT_MODEL_SHA256:
        raise ValueError(
            "unexpected fastText model: "
            f"size={size}, sha256={digest}; expected "
            f"size={FASTTEXT_MODEL_BYTES}, sha256={FASTTEXT_MODEL_SHA256}"
        )
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "download_url": FASTTEXT_MODEL_URL,
    }


def download_fasttext_model(model_path: str | Path) -> dict[str, Any]:
    """Download and verify the official model, without touching the database."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(FASTTEXT_MODEL_URL, timeout=120) as response:
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        metadata = verify_fasttext_model(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata["path"] = str(path)
    return metadata


def _fence_open(line: str) -> tuple[str, int] | None:
    match = FENCE_OPEN_RE.match(line)
    if match is None:
        return None
    marker = match.group(1)
    if marker[0] == "`" and "`" in match.group(2):
        return None
    return marker[0], len(marker)


def _fence_close(line: str, character: str, minimum_length: int) -> bool:
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return False
    match = re.fullmatch(re.escape(character) + r"+([ \t]*)", stripped)
    return match is not None and len(stripped.rstrip(" \t")) >= minimum_length


def _strip_yaml_front_matter(lines: list[str]) -> list[str]:
    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return lines
    closing: int | None = None
    for index in range(1, min(len(lines), 101)):
        if lines[index].strip() in {"---", "..."}:
            closing = index
            break
    if closing is None or not any(YAML_KEY_RE.match(line) for line in lines[1:closing]):
        return lines
    return [""] * (closing + 1) + lines[closing + 1 :]


def clean_markdown_block(value: str) -> str:
    """Remove common Markdown and code-like tokens while retaining prose."""
    text = html.unescape(value)
    text = INLINE_CODE_RE.sub(" ", text)
    text = AUTOLINK_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r" \1 ", text)
    text = REFERENCE_LINK_RE.sub(r" \1 ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = TEMPLATE_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = PATH_TOKEN_RE.sub(" ", text)
    text = FLAG_RE.sub(" ", text)
    text = re.sub(r"\[(?: |x|X)\]", " ", text)
    text = re.sub(r"^[ \t]*(?:>{1,3}[ \t]*)+", "", text)
    text = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", text)
    text = re.sub(r"[|*_~]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t:;,-")


def _word_count(value: str) -> int:
    return len(WORD_RE.findall(value))


def _alphabetic_character_count(value: str) -> int:
    return sum(character.isalpha() for character in value)


def extract_prose_paragraphs(content: str) -> list[ProseParagraph]:
    """Extract conservative prose blocks outside fenced/indented code and tables."""
    without_html_blocks = HTML_CODE_RE.sub("\n", HTML_COMMENT_RE.sub("\n", content))
    lines = _strip_yaml_front_matter(without_html_blocks.splitlines())
    paragraphs: list[ProseParagraph] = []
    current: list[str] = []
    fence_character: str | None = None
    fence_length = 0

    def flush() -> None:
        if not current:
            return
        cleaned = clean_markdown_block(" ".join(current))
        current.clear()
        if not cleaned:
            return
        paragraphs.append(
            ProseParagraph(
                index=len(paragraphs),
                text=cleaned,
                word_count=_word_count(cleaned),
                alphabetic_characters=_alphabetic_character_count(cleaned),
            )
        )

    for line in lines:
        if fence_character is not None:
            if _fence_close(line, fence_character, fence_length):
                fence_character = None
                fence_length = 0
            continue

        opening = _fence_open(line)
        if opening is not None:
            flush()
            fence_character, fence_length = opening
            continue

        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if line.startswith("\t") or line.startswith("    "):
            flush()
            continue
        if line.count("|") >= 2 or LINK_DEFINITION_RE.match(line):
            flush()
            continue
        if HORIZONTAL_RULE_RE.match(line):
            flush()
            continue
        if SETEXT_RE.match(line) and current:
            flush()
            continue

        heading = ATX_HEADING_RE.match(line)
        if heading is not None:
            flush()
            current.append(heading.group(1))
            flush()
            continue

        item = LIST_ITEM_RE.match(line)
        if item is not None:
            flush()
            current.append(item.group(1))
            continue

        current.append(stripped)

    flush()
    return paragraphs


def _truncate_detector_text(value: str, maximum_characters: int) -> str:
    if len(value) <= maximum_characters:
        return value
    truncated = value[:maximum_characters]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip()


def stability_normalization(value: str) -> str:
    """Create a letters-only, case-folded version for a stability check."""
    return " ".join(WORD_RE.findall(value.casefold()))


SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("Japanese kana", ((0x3040, 0x30FF), (0x31F0, 0x31FF))),
    ("Hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("Han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
    ("Cyrillic", ((0x0400, 0x052F),)),
    ("Arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))),
    ("Hebrew", ((0x0590, 0x05FF),)),
    ("Greek", ((0x0370, 0x03FF),)),
    ("Devanagari", ((0x0900, 0x097F),)),
    ("Bengali", ((0x0980, 0x09FF),)),
    ("Gurmukhi", ((0x0A00, 0x0A7F),)),
    ("Gujarati", ((0x0A80, 0x0AFF),)),
    ("Tamil", ((0x0B80, 0x0BFF),)),
    ("Telugu", ((0x0C00, 0x0C7F),)),
    ("Kannada", ((0x0C80, 0x0CFF),)),
    ("Malayalam", ((0x0D00, 0x0D7F),)),
    ("Thai", ((0x0E00, 0x0E7F),)),
    ("Georgian", ((0x10A0, 0x10FF), (0x2D00, 0x2D2F))),
    ("Armenian", ((0x0530, 0x058F),)),
)

SCRIPT_LANGUAGES: dict[str, set[str]] = {
    "Japanese kana": {"ja"},
    "Hangul": {"ko"},
    "Han": {"zh", "ja"},
    "Cyrillic": {"be", "bg", "kk", "ky", "mk", "mn", "ru", "sr", "tg", "uk"},
    "Arabic": {"ar", "fa", "ps", "sd", "ug", "ur"},
    "Hebrew": {"he", "yi"},
    "Greek": {"el"},
    "Devanagari": {"hi", "mr", "ne", "sa"},
    "Bengali": {"as", "bn"},
    "Gurmukhi": {"pa"},
    "Gujarati": {"gu"},
    "Tamil": {"ta"},
    "Telugu": {"te"},
    "Kannada": {"kn"},
    "Malayalam": {"ml"},
    "Thai": {"th"},
    "Georgian": {"ka"},
    "Armenian": {"hy"},
}


def dominant_script(value: str) -> str:
    alphabetic = _alphabetic_character_count(value)
    if alphabetic == 0:
        return "none"
    counts: Counter[str] = Counter()
    for character in value:
        point = ord(character)
        for name, ranges in SCRIPT_RANGES:
            if any(start <= point <= end for start, end in ranges):
                counts[name] += 1
                break
    if not counts:
        return "Latin/other"
    name, count = counts.most_common(1)[0]
    if count < 10 or count / alphabetic < 0.40:
        return "mixed"
    return name


def script_is_compatible(language: str, script: str) -> bool:
    allowed = SCRIPT_LANGUAGES.get(script)
    return allowed is None or language in allowed


class FastTextDetector:
    name = "fastText lid.176.bin"

    def __init__(self, model_path: str | Path):
        try:
            import fasttext
        except ImportError as exc:
            raise RuntimeError(
                "fasttext-predict is required; install requirements-language.txt"
            ) from exc
        self.version = importlib.metadata.version("fasttext-predict")
        self.language_names: dict[str, str] = {}
        self._model = fasttext.load_model(str(model_path))

    def predict(self, text: str) -> Prediction:
        labels, scores = self._model.predict(text.replace("\n", " "), k=2)
        if not labels:
            return Prediction("und", 0.0, 0.0)
        confidence = float(scores[0])
        runner_up = float(scores[1]) if len(scores) > 1 else 0.0
        return Prediction(
            language=_normalize_language_code(str(labels[0])),
            confidence=confidence,
            margin=max(0.0, confidence - runner_up),
        )


class LinguaDetector:
    name = "Lingua"

    def __init__(self):
        try:
            from lingua import Language, LanguageDetectorBuilder
        except ImportError as exc:
            raise RuntimeError(
                "lingua-language-detector is required; install "
                "requirements-language.txt"
            ) from exc
        self.version = importlib.metadata.version("lingua-language-detector")
        self.language_names = {
            _normalize_language_code(language.iso_code_639_1.name):
            language.name.replace("_", " ").title()
            for language in Language.all()
        }
        self._detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        )

    def predict(self, text: str) -> Prediction:
        values = self._detector.compute_language_confidence_values(text)
        if not values:
            return Prediction("und", 0.0, 0.0)
        first = values[0]
        confidence = float(first.value)
        runner_up = float(values[1].value) if len(values) > 1 else 0.0
        return Prediction(
            language=_normalize_language_code(first.language.iso_code_639_1.name),
            confidence=confidence,
            margin=max(0.0, confidence - runner_up),
        )


def paragraph_is_eligible(paragraph: ProseParagraph, config: AnalysisConfig) -> bool:
    return (
        paragraph.word_count >= config.min_words
        and paragraph.alphabetic_characters >= config.min_alpha_characters
    )


def _base_acceptance_reason(
    fasttext_prediction: Prediction,
    lingua_prediction: Prediction,
    script: str,
    stable: bool,
    config: AnalysisConfig,
) -> str:
    if fasttext_prediction.language != lingua_prediction.language:
        return "detector_disagreement"
    if fasttext_prediction.confidence < config.min_confidence:
        return "fasttext_low_confidence"
    if fasttext_prediction.margin < config.min_margin:
        return "fasttext_low_margin"
    if lingua_prediction.confidence < config.min_confidence:
        return "lingua_low_confidence"
    if lingua_prediction.margin < config.min_margin:
        return "lingua_low_margin"
    if not script_is_compatible(fasttext_prediction.language, script):
        return "script_mismatch"
    if not stable:
        return "normalization_instability"
    return "accepted"


def classify_paragraph(
    paragraph: ProseParagraph,
    fasttext_detector: LanguageDetector,
    lingua_detector: LanguageDetector,
    config: AnalysisConfig,
) -> ParagraphDecision:
    """Classify one eligible paragraph using agreement plus abstention rules."""
    detector_text = _truncate_detector_text(
        paragraph.text, config.max_classifier_characters
    )
    fasttext_prediction = fasttext_detector.predict(detector_text)
    lingua_prediction = lingua_detector.predict(detector_text)
    script = dominant_script(detector_text)

    agreed = fasttext_prediction.language == lingua_prediction.language
    compatible = agreed and script_is_compatible(
        fasttext_prediction.language, script
    )
    lowest_relevant_confidence = min(
        (config.min_confidence, *config.sensitivity_confidence_thresholds)
    )
    can_pass_a_reported_threshold = (
        fasttext_prediction.confidence >= lowest_relevant_confidence
        and lingua_prediction.confidence >= lowest_relevant_confidence
        and fasttext_prediction.margin >= config.min_margin
        and lingua_prediction.margin >= config.min_margin
    )
    stable = False
    if compatible and can_pass_a_reported_threshold:
        normalized = stability_normalization(detector_text)
        if not normalized or normalized == detector_text.casefold():
            stable = True
        else:
            normalized_fasttext = fasttext_detector.predict(normalized)
            normalized_lingua = lingua_detector.predict(normalized)
            stable = (
                normalized_fasttext.language == fasttext_prediction.language
                and normalized_lingua.language == lingua_prediction.language
            )

    decision = _base_acceptance_reason(
        fasttext_prediction,
        lingua_prediction,
        script,
        stable,
        config,
    )
    accepted_language = (
        fasttext_prediction.language if decision == "accepted" else None
    )
    return ParagraphDecision(
        paragraph=paragraph,
        detector_text=detector_text,
        fasttext=fasttext_prediction,
        lingua=lingua_prediction,
        dominant_script=script,
        stable_after_normalization=stable,
        decision=decision,
        accepted_language=accepted_language,
    )


def accepted_at_confidence(
    decision: ParagraphDecision,
    confidence_threshold: float,
    margin_threshold: float,
) -> bool:
    """Re-evaluate acceptance at one confidence threshold for sensitivity."""
    return (
        decision.fasttext.language == decision.lingua.language
        and decision.fasttext.confidence >= confidence_threshold
        and decision.lingua.confidence >= confidence_threshold
        and decision.fasttext.margin >= margin_threshold
        and decision.lingua.margin >= margin_threshold
        and script_is_compatible(
            decision.fasttext.language, decision.dominant_script
        )
        and decision.stable_after_normalization
    )


def classify_document(
    accepted_language_words: Counter[str] | dict[str, int],
    eligible_words: int,
    config: AnalysisConfig,
) -> tuple[str, str | None, float, float, str]:
    """Aggregate accepted paragraph labels into a conservative document label."""
    language_words = Counter(accepted_language_words)
    accepted_words = sum(language_words.values())
    coverage = accepted_words / eligible_words if eligible_words else 0.0
    if eligible_words == 0:
        return "unknown", None, 0.0, coverage, "no_eligible_prose"
    if accepted_words < config.min_document_accepted_words:
        return (
            "unknown",
            None,
            0.0,
            coverage,
            "insufficient_accepted_words",
        )
    if coverage < config.min_document_coverage:
        return "unknown", None, 0.0, coverage, "low_accepted_word_coverage"

    ordered = language_words.most_common()
    primary_language, primary_words = ordered[0]
    dominant_share = primary_words / accepted_words
    if dominant_share >= config.dominant_language_share:
        return (
            "primary",
            primary_language,
            dominant_share,
            coverage,
            "dominant_language",
        )

    if len(ordered) >= 2:
        _secondary_language, secondary_words = ordered[1]
        secondary_share = secondary_words / accepted_words
        if (
            secondary_share >= config.multilingual_secondary_share
            and secondary_words >= config.multilingual_secondary_words
        ):
            return (
                "multilingual",
                None,
                dominant_share,
                coverage,
                "multiple_substantial_languages",
            )

    return (
        "unknown",
        None,
        dominant_share,
        coverage,
        "no_dominant_or_substantial_secondary_language",
    )


class DeterministicSampler:
    """Keep fixed-size, SHA-ranked samples without retaining all paragraphs."""

    def __init__(self, capacity_by_stratum: dict[str, int], *, default_capacity: int = 0):
        self.capacity_by_stratum = capacity_by_stratum
        self.default_capacity = default_capacity
        self._heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
        self._sequence = 0

    def consider(
        self,
        stratum: str,
        identity: str,
        record: dict[str, Any],
    ) -> None:
        capacity = self.capacity_by_stratum.get(stratum, self.default_capacity)
        if capacity <= 0:
            return
        score = int.from_bytes(
            hashlib.sha256(f"language-audit-v1\0{identity}".encode("utf-8")).digest(),
            "big",
        )
        self._sequence += 1
        item = (-score, self._sequence, record)
        heap = self._heaps.setdefault(stratum, [])
        if len(heap) < capacity:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)

    def records(self, stratum: str) -> list[dict[str, Any]]:
        values = self._heaps.get(stratum, [])
        ordered = sorted(values, key=lambda item: -item[0])
        return [record for _negative_score, _sequence, record in ordered]


def _new_population_stats() -> dict[str, Any]:
    return {
        "files": 0,
        "extracted_paragraphs": 0,
        "eligible_paragraphs": 0,
        "eligible_words": 0,
        "detector_agreements": 0,
        "stable_agreements": 0,
        "accepted_paragraphs": 0,
        "accepted_words": 0,
        "documents_with_eligible_prose": 0,
        "paragraph_decisions": Counter(),
        "sensitivity_accepted_paragraphs": Counter(),
        "sensitivity_accepted_words": Counter(),
        "sensitivity_language_paragraphs": {},
        "sensitivity_language_words": {},
        "sensitivity_document_classes": {},
        "sensitivity_primary_document_languages": {},
        "sensitivity_document_decisions": {},
        "accepted_language_paragraphs": Counter(),
        "accepted_language_words": Counter(),
        "documents_containing_language": Counter(),
        "document_classes": Counter(),
        "primary_document_languages": Counter(),
        "document_decisions": Counter(),
    }


def _update_population_stats(
    population: dict[str, Any],
    family: ContentFamilyResult,
    paragraph_decisions: Counter[str],
    sensitivity_counts: Counter[str],
    sensitivity_language_paragraphs: dict[str, Counter[str]],
    sensitivity_language_words: dict[str, Counter[str]],
    sensitivity_documents: dict[str, tuple[str, str | None, str]],
    accepted_language_paragraphs: Counter[str],
    *,
    weight: int,
) -> None:
    population["files"] += weight
    population["extracted_paragraphs"] += family.extracted_paragraphs * weight
    population["eligible_paragraphs"] += family.eligible_paragraphs * weight
    population["eligible_words"] += family.eligible_words * weight
    population["detector_agreements"] += family.detector_agreements * weight
    population["stable_agreements"] += family.stable_detector_agreements * weight
    population["accepted_paragraphs"] += family.accepted_paragraphs * weight
    population["accepted_words"] += family.accepted_words * weight
    if family.eligible_paragraphs:
        population["documents_with_eligible_prose"] += weight
    population["document_classes"][family.document_class] += weight
    population["document_decisions"][family.document_decision] += weight
    if family.primary_language is not None:
        population["primary_document_languages"][family.primary_language] += weight

    for decision, count in paragraph_decisions.items():
        population["paragraph_decisions"][decision] += count * weight
    for threshold in sensitivity_documents:
        count = sensitivity_counts[threshold]
        population["sensitivity_accepted_paragraphs"][threshold] += count * weight
        words = sum(sensitivity_language_words[threshold].values())
        population["sensitivity_accepted_words"][threshold] += words * weight
        paragraph_counter = population["sensitivity_language_paragraphs"].setdefault(
            threshold, Counter()
        )
        word_counter = population["sensitivity_language_words"].setdefault(
            threshold, Counter()
        )
        for language, language_count in sensitivity_language_paragraphs[
            threshold
        ].items():
            paragraph_counter[language] += language_count * weight
        for language, language_words in sensitivity_language_words[threshold].items():
            word_counter[language] += language_words * weight
        document_class, primary_language, document_decision = sensitivity_documents[
            threshold
        ]
        class_counter = population["sensitivity_document_classes"].setdefault(
            threshold, Counter()
        )
        decision_counter = population[
            "sensitivity_document_decisions"
        ].setdefault(threshold, Counter())
        primary_counter = population[
            "sensitivity_primary_document_languages"
        ].setdefault(threshold, Counter())
        class_counter[document_class] += weight
        decision_counter[document_decision] += weight
        if primary_language is not None:
            primary_counter[primary_language] += weight
    for language, count in accepted_language_paragraphs.items():
        population["accepted_language_paragraphs"][language] += count * weight
    for language, words in family.accepted_language_words.items():
        population["accepted_language_words"][language] += words * weight
        population["documents_containing_language"][language] += weight


def _iter_content_families(
    connection: sqlite3.Connection,
    scope: str,
) -> Iterable[sqlite3.Row]:
    predicate = _scope_predicate(scope, alias="f")
    return connection.execute(
        f"""
        WITH scoped AS (
            SELECT
                f.id,
                CASE
                    WHEN f.content_hash IS NULL OR f.content_hash = ''
                    THEN 'file-id:' || f.id
                    ELSE f.content_hash
                END AS family_key
            FROM files AS f
            WHERE {predicate}
        ),
        families AS (
            SELECT family_key, count(*) AS copies, min(id) AS representative_id
            FROM scoped
            GROUP BY family_key
        )
        SELECT
            families.family_key,
            families.copies,
            f.repo_full_name,
            f.path,
            f.html_url,
            f.size_bytes,
            f.content
        FROM families
        INNER JOIN files AS f ON f.id = families.representative_id
        ORDER BY families.representative_id
        """
    )


def _audit_record(
    row: sqlite3.Row,
    family_key: str,
    copies: int,
    decision: ParagraphDecision,
) -> dict[str, Any]:
    return {
        "content_hash": family_key,
        "copies": copies,
        "representative_repo": row["repo_full_name"],
        "representative_path": row["path"],
        "representative_url": row["html_url"] or "",
        "paragraph_index": decision.paragraph.index,
        "word_count": decision.paragraph.word_count,
        "alphabetic_characters": decision.paragraph.alphabetic_characters,
        "dominant_script": decision.dominant_script,
        "fasttext_language": decision.fasttext.language,
        "fasttext_confidence": decision.fasttext.confidence,
        "fasttext_margin": decision.fasttext.margin,
        "lingua_language": decision.lingua.language,
        "lingua_confidence": decision.lingua.confidence,
        "lingua_margin": decision.lingua.margin,
        "stable_after_normalization": decision.stable_after_normalization,
        "automated_decision": decision.decision,
        "accepted_language": decision.accepted_language or "",
        "paragraph_text": decision.detector_text,
        "manual_language": "",
        "manual_judgment": "",
        "manual_notes": "",
    }


def _threshold_key(value: float) -> str:
    return f"{value:.2f}"


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _serialize_population(population: dict[str, Any]) -> dict[str, Any]:
    files = int(population["files"])
    extracted = int(population["extracted_paragraphs"])
    eligible = int(population["eligible_paragraphs"])
    eligible_words = int(population["eligible_words"])
    agreements = int(population["detector_agreements"])
    stable = int(population["stable_agreements"])
    accepted = int(population["accepted_paragraphs"])
    accepted_words = int(population["accepted_words"])
    sensitivity: dict[str, Any] = {}
    for threshold in sorted(
        population["sensitivity_accepted_paragraphs"], key=float
    ):
        threshold_paragraphs = int(
            population["sensitivity_accepted_paragraphs"][threshold]
        )
        threshold_words = int(population["sensitivity_accepted_words"][threshold])
        language_words = population["sensitivity_language_words"][threshold]
        language_paragraphs = population["sensitivity_language_paragraphs"][
            threshold
        ]
        total_language_words = sum(language_words.values())
        document_classes = population["sensitivity_document_classes"][threshold]
        sensitivity[threshold] = {
            "accepted_paragraphs": threshold_paragraphs,
            "accepted_paragraph_share": (
                threshold_paragraphs / eligible if eligible else 0.0
            ),
            "accepted_words": threshold_words,
            "accepted_word_coverage": (
                threshold_words / eligible_words if eligible_words else 0.0
            ),
            "language_paragraphs": _sorted_counter(language_paragraphs),
            "language_words": _sorted_counter(language_words),
            "language_word_shares": {
                language: count / total_language_words
                if total_language_words
                else 0.0
                for language, count in _sorted_counter(language_words).items()
            },
            "document_classes": _sorted_counter(document_classes),
            "document_class_shares": {
                key: value / files if files else 0.0
                for key, value in _sorted_counter(document_classes).items()
            },
            "document_decisions": _sorted_counter(
                population["sensitivity_document_decisions"][threshold]
            ),
            "primary_document_languages": _sorted_counter(
                population["sensitivity_primary_document_languages"][threshold]
            ),
        }

    return {
        "files": files,
        "documents_with_eligible_prose": int(
            population["documents_with_eligible_prose"]
        ),
        "documents_with_eligible_prose_share": (
            population["documents_with_eligible_prose"] / files if files else 0.0
        ),
        "paragraphs": {
            "extracted": extracted,
            "eligible": eligible,
            "eligible_share": eligible / extracted if extracted else 0.0,
            "eligible_words": eligible_words,
            "detector_agreements": agreements,
            "detector_agreement_share": agreements / eligible if eligible else 0.0,
            "stable_detector_agreements": stable,
            "stable_agreement_share": stable / eligible if eligible else 0.0,
            "accepted": accepted,
            "accepted_share": accepted / eligible if eligible else 0.0,
            "accepted_words": accepted_words,
            "accepted_word_coverage": (
                accepted_words / eligible_words if eligible_words else 0.0
            ),
            "decisions": _sorted_counter(population["paragraph_decisions"]),
            "sensitivity_accepted": _sorted_counter(
                population["sensitivity_accepted_paragraphs"]
            ),
            "sensitivity": sensitivity,
        },
        "documents": {
            "classes": _sorted_counter(population["document_classes"]),
            "class_shares": {
                key: value / files if files else 0.0
                for key, value in _sorted_counter(
                    population["document_classes"]
                ).items()
            },
            "decisions": _sorted_counter(population["document_decisions"]),
            "primary_languages": _sorted_counter(
                population["primary_document_languages"]
            ),
        },
        "languages": {
            "accepted_paragraphs": _sorted_counter(
                population["accepted_language_paragraphs"]
            ),
            "accepted_words": _sorted_counter(
                population["accepted_language_words"]
            ),
            "documents_containing_accepted_prose": _sorted_counter(
                population["documents_containing_language"]
            ),
        },
    }


def _language_distribution(
    full: dict[str, Any],
    unique: dict[str, Any],
    language_names: dict[str, str],
) -> list[dict[str, Any]]:
    language_codes = set(full["accepted_language_words"]) | set(
        unique["accepted_language_words"]
    )
    ordered = sorted(
        language_codes,
        key=lambda code: (
            -full["accepted_language_words"][code],
            -full["primary_document_languages"][code],
            code,
        ),
    )
    full_words = sum(full["accepted_language_words"].values())
    unique_words = sum(unique["accepted_language_words"].values())
    full_paragraphs = sum(full["accepted_language_paragraphs"].values())
    unique_paragraphs = sum(unique["accepted_language_paragraphs"].values())
    full_primary = sum(full["primary_document_languages"].values())
    unique_primary = sum(unique["primary_document_languages"].values())
    rows: list[dict[str, Any]] = []
    for rank, code in enumerate(ordered, 1):
        rows.append(
            {
                "rank": rank,
                "language_code": code,
                "language_name": language_names.get(code, code),
                "accepted_words_full": full["accepted_language_words"][code],
                "accepted_word_share_full": (
                    full["accepted_language_words"][code] / full_words
                    if full_words
                    else 0.0
                ),
                "accepted_paragraphs_full": full[
                    "accepted_language_paragraphs"
                ][code],
                "accepted_paragraph_share_full": (
                    full["accepted_language_paragraphs"][code] / full_paragraphs
                    if full_paragraphs
                    else 0.0
                ),
                "files_with_accepted_prose_full": full[
                    "documents_containing_language"
                ][code],
                "primary_documents_full": full["primary_document_languages"][code],
                "primary_document_share_full": (
                    full["primary_document_languages"][code] / full_primary
                    if full_primary
                    else 0.0
                ),
                "accepted_words_unique": unique["accepted_language_words"][code],
                "accepted_word_share_unique": (
                    unique["accepted_language_words"][code] / unique_words
                    if unique_words
                    else 0.0
                ),
                "accepted_paragraphs_unique": unique[
                    "accepted_language_paragraphs"
                ][code],
                "accepted_paragraph_share_unique": (
                    unique["accepted_language_paragraphs"][code]
                    / unique_paragraphs
                    if unique_paragraphs
                    else 0.0
                ),
                "content_families_with_accepted_prose": unique[
                    "documents_containing_language"
                ][code],
                "primary_contents_unique": unique["primary_document_languages"][code],
                "primary_content_share_unique": (
                    unique["primary_document_languages"][code] / unique_primary
                    if unique_primary
                    else 0.0
                ),
            }
        )
    return rows


def analyze_paragraph_languages(
    db_path: str | Path,
    fasttext_detector: LanguageDetector,
    lingua_detector: LanguageDetector,
    *,
    scope: str = "exact-claude",
    config: AnalysisConfig | None = None,
    audit_top_languages: int = 10,
    audit_accepted_per_language: int = 5,
    audit_disagreements: int = 20,
    audit_threshold_rejections: int = 15,
    audit_stability_or_script: int = 10,
    progress_every: int = 0,
    progress_stream: Any | None = None,
) -> tuple[
    dict[str, Any],
    list[ContentFamilyResult],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Analyze the scoped population and return summary and export rows."""
    config = config or AnalysisConfig()
    connection = _readonly_connection(db_path)
    full_population = _new_population_stats()
    unique_population = _new_population_stats()
    families: list[ContentFamilyResult] = []
    sampler = DeterministicSampler(
        {
            "detector_disagreement": audit_disagreements,
            "threshold_rejection": audit_threshold_rejections,
            "stability_or_script": audit_stability_or_script,
        },
        default_capacity=audit_accepted_per_language,
    )

    try:
        predicate = _scope_predicate(scope, alias="f")
        scoped_files = int(
            connection.execute(
                f"SELECT count(*) FROM files AS f WHERE {predicate}"
            ).fetchone()[0]
        )
        for family_number, row in enumerate(
            _iter_content_families(connection, scope), 1
        ):
            family_key = str(row["family_key"])
            copies = int(row["copies"])
            paragraphs = extract_prose_paragraphs(row["content"] or "")
            eligible_paragraphs = [
                paragraph
                for paragraph in paragraphs
                if paragraph_is_eligible(paragraph, config)
            ]
            eligible_words = sum(
                paragraph.word_count for paragraph in eligible_paragraphs
            )
            paragraph_decisions: Counter[str] = Counter()
            sensitivity_counts: Counter[str] = Counter()
            sensitivity_language_paragraphs: dict[str, Counter[str]] = {
                _threshold_key(threshold): Counter()
                for threshold in config.sensitivity_confidence_thresholds
            }
            sensitivity_language_words: dict[str, Counter[str]] = {
                _threshold_key(threshold): Counter()
                for threshold in config.sensitivity_confidence_thresholds
            }
            accepted_language_paragraphs: Counter[str] = Counter()
            accepted_language_words: Counter[str] = Counter()
            detector_agreements = 0
            stable_agreements = 0

            for paragraph in eligible_paragraphs:
                decision = classify_paragraph(
                    paragraph,
                    fasttext_detector,
                    lingua_detector,
                    config,
                )
                paragraph_decisions[decision.decision] += 1
                if decision.fasttext.language == decision.lingua.language:
                    detector_agreements += 1
                    if decision.stable_after_normalization:
                        stable_agreements += 1
                for threshold in config.sensitivity_confidence_thresholds:
                    threshold_key = _threshold_key(threshold)
                    if accepted_at_confidence(
                        decision, threshold, config.min_margin
                    ):
                        sensitivity_counts[threshold_key] += 1
                        language = decision.fasttext.language
                        sensitivity_language_paragraphs[threshold_key][language] += 1
                        sensitivity_language_words[threshold_key][
                            language
                        ] += paragraph.word_count

                if decision.accepted_language is not None:
                    language = decision.accepted_language
                    accepted_language_paragraphs[language] += 1
                    accepted_language_words[language] += paragraph.word_count
                    audit_stratum = f"accepted:{language}"
                elif decision.decision == "detector_disagreement":
                    audit_stratum = "detector_disagreement"
                elif decision.decision in {
                    "script_mismatch",
                    "normalization_instability",
                }:
                    audit_stratum = "stability_or_script"
                else:
                    audit_stratum = "threshold_rejection"

                sampler.consider(
                    audit_stratum,
                    f"{family_key}:{paragraph.index}:{audit_stratum}",
                    _audit_record(row, family_key, copies, decision),
                )

            accepted_paragraphs = sum(accepted_language_paragraphs.values())
            accepted_words = sum(accepted_language_words.values())
            (
                document_class,
                primary_language,
                dominant_share,
                coverage,
                document_decision,
            ) = classify_document(
                accepted_language_words,
                eligible_words,
                config,
            )
            sensitivity_documents: dict[
                str, tuple[str, str | None, str]
            ] = {}
            for threshold in config.sensitivity_confidence_thresholds:
                threshold_key = _threshold_key(threshold)
                (
                    threshold_document_class,
                    threshold_primary_language,
                    _threshold_dominant_share,
                    _threshold_coverage,
                    threshold_document_decision,
                ) = classify_document(
                    sensitivity_language_words[threshold_key],
                    eligible_words,
                    config,
                )
                sensitivity_documents[threshold_key] = (
                    threshold_document_class,
                    threshold_primary_language,
                    threshold_document_decision,
                )
            family = ContentFamilyResult(
                content_hash=family_key,
                copies=copies,
                representative_repo=str(row["repo_full_name"]),
                representative_path=str(row["path"]),
                representative_url=row["html_url"],
                size_bytes=(
                    int(row["size_bytes"])
                    if row["size_bytes"] is not None
                    else None
                ),
                extracted_paragraphs=len(paragraphs),
                eligible_paragraphs=len(eligible_paragraphs),
                eligible_words=eligible_words,
                detector_agreements=detector_agreements,
                stable_detector_agreements=stable_agreements,
                accepted_paragraphs=accepted_paragraphs,
                accepted_words=accepted_words,
                accepted_word_coverage=coverage,
                document_class=document_class,
                primary_language=primary_language,
                dominant_language_share=dominant_share,
                document_decision=document_decision,
                accepted_language_words=dict(
                    sorted(accepted_language_words.items())
                ),
            )
            families.append(family)
            _update_population_stats(
                full_population,
                family,
                paragraph_decisions,
                sensitivity_counts,
                sensitivity_language_paragraphs,
                sensitivity_language_words,
                sensitivity_documents,
                accepted_language_paragraphs,
                weight=copies,
            )
            _update_population_stats(
                unique_population,
                family,
                paragraph_decisions,
                sensitivity_counts,
                sensitivity_language_paragraphs,
                sensitivity_language_words,
                sensitivity_documents,
                accepted_language_paragraphs,
                weight=1,
            )

            if progress_every and family_number % progress_every == 0:
                stream = progress_stream or sys.stderr
                print(
                    f"Processed {family_number:,} content families; "
                    f"{unique_population['eligible_paragraphs']:,} eligible "
                    "unique-content paragraphs",
                    file=stream,
                    flush=True,
                )
    finally:
        connection.close()

    if full_population["files"] != scoped_files:
        raise RuntimeError(
            "copy-weighted family count does not match scoped file count: "
            f"{full_population['files']} != {scoped_files}"
        )

    language_names = dict(fasttext_detector.language_names)
    language_names.update(lingua_detector.language_names)
    distribution = _language_distribution(
        full_population, unique_population, language_names
    )
    top_codes = [
        row["language_code"] for row in distribution[:audit_top_languages]
    ]
    audit_rows: list[dict[str, Any]] = []
    for code in top_codes:
        for record in sampler.records(f"accepted:{code}"):
            audit_rows.append({"audit_stratum": f"accepted:{code}", **record})
    for stratum in (
        "detector_disagreement",
        "threshold_rejection",
        "stability_or_script",
    ):
        for record in sampler.records(stratum):
            audit_rows.append({"audit_stratum": stratum, **record})
    for index, record in enumerate(audit_rows, 1):
        record["audit_id"] = f"L{index:03d}"

    summary = {
        "scope": scope,
        "scope_definition": _scope_definition(scope),
        "scoped_files": scoped_files,
        "content_families": len(families),
        "exact_duplicate_instances": scoped_files - len(families),
        "parameters": asdict(config),
        "detectors": {
            "fasttext": {
                "name": fasttext_detector.name,
                "version": fasttext_detector.version,
            },
            "lingua": {
                "name": lingua_detector.name,
                "version": lingua_detector.version,
            },
        },
        "full_population": _serialize_population(full_population),
        "unique_content_population": _serialize_population(unique_population),
        "language_distribution": distribution,
        "manual_audit": {
            "rows": len(audit_rows),
            "top_languages_sampled": top_codes,
            "accepted_per_language_target": audit_accepted_per_language,
            "disagreement_target": audit_disagreements,
            "threshold_rejection_target": audit_threshold_rejections,
            "stability_or_script_target": audit_stability_or_script,
            "status": "awaiting_manual_review",
        },
        "database_writes": 0,
    }
    return summary, families, distribution, audit_rows


def write_distribution_csv(
    rows: Sequence[dict[str, Any]], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "rank",
        "language_code",
        "language_name",
        "accepted_words_full",
        "accepted_word_share_full",
        "accepted_paragraphs_full",
        "accepted_paragraph_share_full",
        "files_with_accepted_prose_full",
        "primary_documents_full",
        "primary_document_share_full",
        "accepted_words_unique",
        "accepted_word_share_unique",
        "accepted_paragraphs_unique",
        "accepted_paragraph_share_unique",
        "content_families_with_accepted_prose",
        "primary_contents_unique",
        "primary_content_share_unique",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_families_csv(
    families: Sequence[ContentFamilyResult], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "content_hash",
        "copies",
        "representative_repo",
        "representative_path",
        "representative_url",
        "size_bytes",
        "extracted_paragraphs",
        "eligible_paragraphs",
        "eligible_words",
        "detector_agreements",
        "stable_detector_agreements",
        "accepted_paragraphs",
        "accepted_words",
        "accepted_word_coverage",
        "document_class",
        "primary_language",
        "dominant_language_share",
        "document_decision",
        "accepted_language_words_json",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family in families:
            record = asdict(family)
            language_words = record.pop("accepted_language_words")
            record["accepted_language_words_json"] = json.dumps(
                language_words, sort_keys=True, ensure_ascii=False
            )
            writer.writerow(record)
    return output


def write_audit_csv(rows: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "audit_id",
        "audit_stratum",
        "content_hash",
        "copies",
        "representative_repo",
        "representative_path",
        "representative_url",
        "paragraph_index",
        "word_count",
        "alphabetic_characters",
        "dominant_script",
        "fasttext_language",
        "fasttext_confidence",
        "fasttext_margin",
        "lingua_language",
        "lingua_confidence",
        "lingua_margin",
        "stable_after_normalization",
        "automated_decision",
        "accepted_language",
        "paragraph_text",
        "manual_language",
        "manual_judgment",
        "manual_notes",
    )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_summary_json(summary: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def database_fingerprint(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "modified_at_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": _sha256_file(path),
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def confidence_thresholds(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated numbers between 0 and 1"
        ) from exc
    if not parsed or any(not 0.0 <= item <= 1.0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "must be comma-separated numbers between 0 and 1"
        )
    return tuple(sorted(set(parsed)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify prose paragraphs with fastText/Lingua agreement while "
            "opening mined.db read-only and leaving uncertain text unknown."
        )
    )
    parser.add_argument("--db", default="mined.db", help="SQLite database")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="exact-claude",
        help="population to analyze (default: exact-claude)",
    )
    parser.add_argument(
        "--fasttext-model",
        default="models/lid.176.bin",
        help="path to the official lid.176.bin model",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="download the official model if it is absent, then verify its hash",
    )
    parser.add_argument(
        "--prediction-cache",
        default=".analysis_cache/paragraph_language_predictions.sqlite3",
        help="separate derived prediction cache used to resume long runs",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable the separate prediction cache",
    )
    parser.add_argument(
        "--distribution-output",
        default="article/natural_language_distribution.csv",
        help="per-language CSV output",
    )
    parser.add_argument(
        "--families-output",
        default="article/document_language_families.csv",
        help="per-content-family CSV output",
    )
    parser.add_argument(
        "--audit-output",
        default="article/language_manual_audit.csv",
        help="deterministically sampled paragraph CSV for manual review",
    )
    parser.add_argument(
        "--summary-output",
        default="article/natural_language_summary.json",
        help="machine-readable JSON summary",
    )
    parser.add_argument("--min-words", type=positive_int, default=DEFAULT_MIN_WORDS)
    parser.add_argument(
        "--min-alpha-characters",
        type=positive_int,
        default=DEFAULT_MIN_ALPHA_CHARS,
    )
    parser.add_argument(
        "--max-classifier-characters",
        type=positive_int,
        default=DEFAULT_MAX_CLASSIFIER_CHARS,
    )
    parser.add_argument(
        "--min-confidence", type=probability, default=DEFAULT_MIN_CONFIDENCE
    )
    parser.add_argument("--min-margin", type=probability, default=DEFAULT_MIN_MARGIN)
    parser.add_argument(
        "--min-document-accepted-words",
        type=positive_int,
        default=DEFAULT_MIN_DOCUMENT_WORDS,
    )
    parser.add_argument(
        "--min-document-coverage",
        type=probability,
        default=DEFAULT_MIN_DOCUMENT_COVERAGE,
    )
    parser.add_argument(
        "--dominant-language-share",
        type=probability,
        default=DEFAULT_DOMINANT_LANGUAGE_SHARE,
    )
    parser.add_argument(
        "--multilingual-secondary-share",
        type=probability,
        default=DEFAULT_MULTILINGUAL_SECONDARY_SHARE,
    )
    parser.add_argument(
        "--multilingual-secondary-words",
        type=positive_int,
        default=DEFAULT_MULTILINGUAL_SECONDARY_WORDS,
    )
    parser.add_argument(
        "--sensitivity-confidence-thresholds",
        type=confidence_thresholds,
        default=DEFAULT_SENSITIVITY_THRESHOLDS,
        help="comma-separated confidence thresholds (default: 0.80,0.90,0.95)",
    )
    parser.add_argument("--audit-top-languages", type=positive_int, default=10)
    parser.add_argument(
        "--audit-accepted-per-language", type=positive_int, default=5
    )
    parser.add_argument("--audit-disagreements", type=positive_int, default=20)
    parser.add_argument(
        "--audit-threshold-rejections", type=positive_int, default=15
    )
    parser.add_argument(
        "--audit-stability-or-script", type=positive_int, default=10
    )
    parser.add_argument(
        "--progress-every",
        type=nonnegative_int,
        default=5_000,
        help="print progress after this many families; 0 disables it",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        default=15,
        help="number of languages to print (default: 15)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.fasttext_model)
    try:
        if args.download_model and not model_path.exists():
            print(f"Downloading official fastText model to {model_path} ...")
            model_metadata = download_fasttext_model(model_path)
        else:
            model_metadata = verify_fasttext_model(model_path)

        database_before = database_fingerprint(args.db)
        base_fasttext_detector = FastTextDetector(model_path)
        base_lingua_detector = LinguaDetector()
        prediction_cache: PredictionCache | None = None
        if args.no_cache:
            fasttext_detector: LanguageDetector = base_fasttext_detector
            lingua_detector: LanguageDetector = base_lingua_detector
        else:
            prediction_cache = PredictionCache(
                args.prediction_cache, source_db_path=args.db
            )
            fasttext_detector = CachedDetector(
                base_fasttext_detector,
                prediction_cache,
                (
                    f"fasttext-predict:{base_fasttext_detector.version}:"
                    f"lid.176.bin:{FASTTEXT_MODEL_SHA256}"
                ),
            )
            lingua_detector = CachedDetector(
                base_lingua_detector,
                prediction_cache,
                f"lingua:{base_lingua_detector.version}:all-languages",
            )
        config = AnalysisConfig(
            min_words=args.min_words,
            min_alpha_characters=args.min_alpha_characters,
            max_classifier_characters=args.max_classifier_characters,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            min_document_accepted_words=args.min_document_accepted_words,
            min_document_coverage=args.min_document_coverage,
            dominant_language_share=args.dominant_language_share,
            multilingual_secondary_share=args.multilingual_secondary_share,
            multilingual_secondary_words=args.multilingual_secondary_words,
            sensitivity_confidence_thresholds=(
                args.sensitivity_confidence_thresholds
            ),
        )
        try:
            summary, families, distribution, audit_rows = (
                analyze_paragraph_languages(
                    args.db,
                    fasttext_detector,
                    lingua_detector,
                    scope=args.scope,
                    config=config,
                    audit_top_languages=args.audit_top_languages,
                    audit_accepted_per_language=args.audit_accepted_per_language,
                    audit_disagreements=args.audit_disagreements,
                    audit_threshold_rejections=(
                        args.audit_threshold_rejections
                    ),
                    audit_stability_or_script=args.audit_stability_or_script,
                    progress_every=args.progress_every,
                    progress_stream=sys.stderr,
                )
            )
            cache_metadata = {
                "enabled": prediction_cache is not None,
                "path": (
                    str(prediction_cache.path)
                    if prediction_cache is not None
                    else None
                ),
                "fasttext_hits": getattr(fasttext_detector, "cache_hits", 0),
                "fasttext_misses": getattr(
                    fasttext_detector, "cache_misses", 0
                ),
                "lingua_hits": getattr(lingua_detector, "cache_hits", 0),
                "lingua_misses": getattr(lingua_detector, "cache_misses", 0),
            }
        finally:
            if prediction_cache is not None:
                prediction_cache.close()
        database_after = database_fingerprint(args.db)
        if database_before != database_after:
            raise RuntimeError("database fingerprint changed during analysis")

        distribution_path = write_distribution_csv(
            distribution, args.distribution_output
        )
        families_path = write_families_csv(families, args.families_output)
        audit_path = write_audit_csv(audit_rows, args.audit_output)
        summary.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "database": database_before,
                "fasttext_model": model_metadata,
                "prediction_cache": cache_metadata,
                "software": {
                    "python": sys.version.split()[0],
                    "script": str(Path(__file__).resolve()),
                    "script_sha256": _sha256_file(Path(__file__).resolve()),
                },
                "artifacts": {
                    "distribution_csv": str(distribution_path),
                    "families_csv": str(families_path),
                    "manual_audit_csv": str(audit_path),
                    "summary_json": str(Path(args.summary_output)),
                },
            }
        )
        summary_path = write_summary_json(summary, args.summary_output)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        sqlite3.Error,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    full = summary["full_population"]
    unique = summary["unique_content_population"]
    print(f"Scope: {summary['scope']} ({summary['scope_definition']})")
    print(
        f"Files: {summary['scoped_files']:,}; unique contents: "
        f"{summary['content_families']:,}"
    )
    print(
        "Eligible paragraphs: "
        f"{full['paragraphs']['eligible']:,} copy-weighted / "
        f"{unique['paragraphs']['eligible']:,} unique-content"
    )
    print(
        "Accepted paragraphs: "
        f"{full['paragraphs']['accepted']:,} "
        f"({full['paragraphs']['accepted_share']:.2%})"
    )
    print("Leading accepted languages by copy-weighted word count:")
    for row in distribution[: args.top]:
        print(
            f"  {row['rank']:>3}. {row['language_code']:>3} "
            f"{row['language_name']:<15} "
            f"{row['accepted_words_full']:>12,} words "
            f"({row['accepted_word_share_full']:.2%})"
        )
    print(f"Distribution CSV: {distribution_path.resolve()}")
    print(f"Content-family CSV: {families_path.resolve()}")
    print(f"Manual-audit CSV: {audit_path.resolve()} ({len(audit_rows)} rows)")
    print(f"Summary JSON: {summary_path.resolve()}")
    print(f"Database SHA-256 unchanged: {database_after['sha256']}")
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
