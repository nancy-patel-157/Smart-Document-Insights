"""Small, dependency-free helpers used by the RAG engine and the UI.

Nothing in here imports Streamlit or LangChain, so these functions can be
unit-tested without installing the heavy ML stack.
"""

import hashlib
import math
import re

# Dates such as 12/05/2025, 12-05-25, 2025-05-12, 12 May 2025, May 12, 2025, 12th March
_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"  # 12/05/2025
    r"|\d{4}[/.-]\d{1,2}[/.-]\d{1,2}"  # 2025-05-12
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?(?:\s+\d{{4}})?"  # 12 May 2025
    rf"|(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?(?:\s+\d{{4}})?"  # May 12, 2025
    r")\b",
    re.IGNORECASE,
)


def file_fingerprint(files_bytes, *settings):
    """Stable hash of uploaded file contents plus indexing settings.

    Used to skip re-indexing when the same files are processed again with the
    same chunk settings.
    """
    digest = hashlib.sha256()
    for data in files_bytes:
        digest.update(hashlib.sha256(data).digest())
    digest.update(repr(settings).encode())
    return digest.hexdigest()


def split_into_sections(text, section_chars=12000, max_sections=10):
    """Split long text into roughly equal sections for map-reduce summarising.

    Sections end on a line break where possible so sentences are not cut.
    If the text would need more than ``max_sections`` sections, the section
    size grows so the whole document is still covered.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= section_chars:
        return [text]

    size = max(section_chars, math.ceil(len(text) / max_sections))
    sections, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer to break on a newline in the last 20% of the window.
            newline = text.rfind("\n", start + int(size * 0.8), end)
            if newline != -1:
                end = newline
        piece = text[start:end].strip()
        if piece:
            sections.append(piece)
        start = end
    return sections


def sample_evenly(items, n):
    """Pick ``n`` items spread evenly across the list (keeps order)."""
    items = list(items)
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def take_until(texts, max_chars):
    """Return texts from the start until ``max_chars`` would be exceeded."""
    picked, total = [], 0
    for text in texts:
        if picked and total + len(text) > max_chars:
            break
        picked.append(text)
        total += len(text)
    return picked


def contains_date(text):
    return bool(DATE_PATTERN.search(text or ""))


def format_history(messages, max_turns=3):
    """Format the last few chat turns as plain text for a prompt."""
    recent = messages[-max_turns * 2:]
    lines = []
    for message in recent:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = (message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def source_label(metadata):
    """Human-readable label like 'report.pdf, p. 4'."""
    metadata = metadata or {}
    source = metadata.get("source", "document")
    page = metadata.get("page")
    return f"{source}, p. {page}" if page else str(source)


def friendly_error(error):
    """Turn common Gemini / network errors into a short, actionable message."""
    text = str(error)
    lowered = text.lower()
    if "api key" in lowered or "api_key_invalid" in lowered or "permission_denied" in lowered:
        return "The Gemini API key was rejected. Check the key and try again."
    if "429" in text or "quota" in lowered or "resource_exhausted" in lowered or "rate" in lowered:
        return "Gemini rate limit or quota reached. Wait a minute and try again."
    if "not found" in lowered and "model" in lowered:
        return "The configured Gemini model is not available. Set GEMINI_MODEL in your .env."
    if "timeout" in lowered or "deadline" in lowered or "connection" in lowered:
        return "Could not reach Gemini (network timeout). Check your connection and retry."
    return f"Something went wrong while calling Gemini: {text[:300]}"
