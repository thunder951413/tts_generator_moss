"""Text helpers shared by the generation routes and the document pipeline."""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_KANA_RE = re.compile(r"[぀-ヿ]")
_HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")

SUPPORTED_LANGUAGES = {
    "auto": "auto",
    "chinese": "Chinese",
    "english": "English",
    "japanese": "Japanese",
    "korean": "Korean",
}


def detect_language(text: str, default: str = "Chinese") -> str:
    """Pick the qwentts.cpp language tag for ``text``.

    Han characters dominate Chinese text even with embedded English words, so a
    modest Han share is enough. Kana and Hangul are unambiguous. Text that is
    mostly Latin letters is English; anything else lets the model infer it.
    """

    sample = (text or "")[:4000]
    kana = len(_KANA_RE.findall(sample))
    hangul = len(_HANGUL_RE.findall(sample))
    han = len(_CJK_RE.findall(sample))
    latin = len(_LATIN_RE.findall(sample))
    if kana >= 3 and kana * 4 >= han:
        return "Japanese"
    if hangul >= 3 and hangul >= han:
        return "Korean"
    letters = han + latin
    if letters == 0:
        return default
    if han and han * 3 >= latin:
        return "Chinese"
    if latin and latin >= han * 3:
        return "English"
    return "auto"


def normalize_language(value: str | None, text: str = "", default: str = "Chinese") -> str:
    """Map a client supplied language to a backend tag; empty/auto detects."""

    key = (value or "").strip().lower()
    if not key or key == "auto" or key == "detect":
        return detect_language(text, default=default)
    return SUPPORTED_LANGUAGES.get(key, value.strip())


# ---------------------------------------------------------------------------
# Spoken-text preparation (applied at synthesis time; stored text is untouched)
# ---------------------------------------------------------------------------

import unicodedata as _unicodedata

_FULLWIDTH_ALNUM_RE = re.compile(r"[０-９Ａ-Ｚａ-ｚ]")
_ELLIPSIS_RUN_RE = re.compile(r"(?:\.{3,}|…{2,}|(?:…\.+|\.+…)[.…]*)")
_TERMINAL_CHARS = set("。！？!?；;….，,、：:")
_CLOSING_CHARS = set("”」』）)]】》\"'’")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BARE_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]（）。，]+", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.*?)\s*#*\s*$")
_MD_LIST_RE = re.compile(r"^\s*[-*+•]\s+(?:\[[ xX]\]\s+)?")
_MD_QUOTE_RE = re.compile(r"^\s*>+\s?")
_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_MD_RULE_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_MD_EMPHASIS_RES = (
    re.compile(r"\*\*(.+?)\*\*"),
    re.compile(r"__(.+?)__"),
    re.compile(r"~~(.+?)~~"),
    re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"),
    re.compile(r"`([^`]*)`"),
)


def _strip_inline_markdown(line: str) -> str:
    line = _MD_IMAGE_RE.sub("", line)
    line = _MD_LINK_RE.sub(r"\1", line)
    line = _BARE_URL_RE.sub("", line)
    for pattern in _MD_EMPHASIS_RES:
        line = pattern.sub(r"\1", line)
    return line


def strip_markdown(text: str) -> str:
    """Remove Markdown syntax for an imported .md/.markdown document.

    Headings, emphasis, list bullets, quotes, links, images, code fences,
    horizontal rules and bare URLs are removed while the readable words stay.
    """

    lines: list[str] = []
    for raw in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _MD_FENCE_RE.match(raw) or _MD_RULE_RE.match(raw):
            continue
        line = raw
        heading = _MD_HEADING_RE.match(line)
        if heading and line.lstrip().startswith("#"):
            line = heading.group(1)
        line = _MD_QUOTE_RE.sub("", line)
        line = _MD_LIST_RE.sub("", line)
        line = _strip_inline_markdown(line)
        lines.append(line)
    return "\n".join(lines)


def _fullwidth_to_ascii(match: re.Match[str]) -> str:
    return _unicodedata.normalize("NFKC", match.group(0))


def _is_latin_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "%)]")


def prepare_tts_text(text: str) -> str:
    """Turn a stored block into the string spoken by the model.

    * paragraph breaks ("\\n") become a sentence stop ("。", or ". " after Latin
      text) when the previous line lacks terminal punctuation, otherwise a plain
      join (space between Latin words);
    * full-width digits/letters are NFKC-normalised (punctuation is untouched);
    * runs of "..." / "……" collapse to a single "…";
    * Markdown remnants (emphasis, links, images, heading marks, URLs) go away.
    """

    if not text:
        return ""
    value = _FULLWIDTH_ALNUM_RE.sub(_fullwidth_to_ascii, text)
    lines = []
    for raw in value.split("\n"):
        line = raw.strip()
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        line = _MD_LIST_RE.sub("", line)
        line = _strip_inline_markdown(line)
        line = re.sub(r"[ \t　]+", " ", line).strip()
        if line:
            lines.append(line)
    if not lines:
        return ""
    result = lines[0]
    for line in lines[1:]:
        probe = result.rstrip()
        index = len(probe) - 1
        while index >= 0 and probe[index] in _CLOSING_CHARS:
            index -= 1
        last = probe[index] if index >= 0 else ""
        latin_join = _is_latin_char(last) or (last in ".!?;:," and _is_latin_char(line[0]))
        if last in _TERMINAL_CHARS:
            joiner = " " if (last.isascii() or _is_latin_char(line[0])) and latin_join else ""
            result = probe + joiner + line
        elif _is_latin_char(last):
            result = probe + ". " + line
        else:
            result = probe + "。" + line
    result = _ELLIPSIS_RUN_RE.sub("…", result)
    return re.sub(r" {2,}", " ", result).strip()
