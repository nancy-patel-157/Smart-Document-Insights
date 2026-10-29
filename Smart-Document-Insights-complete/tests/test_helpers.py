"""Tests for dependency-free helpers (run without the ML stack installed)."""

import helpers
import prompts


def test_split_short_text_is_one_section():
    assert helpers.split_into_sections("hello world", section_chars=100) == ["hello world"]


def test_split_empty_text():
    assert helpers.split_into_sections("   ") == []


def test_split_covers_whole_document_and_respects_max_sections():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(2000))  # ~118k chars
    sections = helpers.split_into_sections(text, section_chars=12000, max_sections=5)
    assert len(sections) <= 6  # max_sections, plus at most one small remainder
    joined = "".join(s.replace("\n", "") for s in sections)
    assert joined == text.replace("\n", "")  # nothing lost or duplicated


def test_split_prefers_line_breaks():
    text = ("a" * 90 + "\n") * 30
    for section in helpers.split_into_sections(text, section_chars=500, max_sections=100):
        assert set(section.replace("\n", "")) == {"a"}
        assert len(section.split("\n")[-1]) == 90  # last line is complete


def test_sample_evenly_spreads_across_list():
    picked = helpers.sample_evenly(list(range(100)), 5)
    assert picked == [0, 20, 40, 60, 80]
    assert helpers.sample_evenly([1, 2], 5) == [1, 2]
    assert helpers.sample_evenly([], 3) == []
    assert helpers.sample_evenly([1, 2, 3], 0) == []


def test_take_until_limits_characters_but_keeps_first():
    assert helpers.take_until(["aaaa", "bbbb", "cccc"], 9) == ["aaaa", "bbbb"]
    assert helpers.take_until(["a" * 50], 10) == ["a" * 50]


def test_contains_date_formats():
    for text in [
        "Last date: 15/03/2025",
        "Starts on 2025-04-01",
        "Apply by 12 May 2025",
        "Results on March 5, 2026",
        "deadline 3rd June",
    ]:
        assert helpers.contains_date(text), text
    assert not helpers.contains_date("Section 4.2 covers the fee of 500 rupees")


def test_format_history_keeps_last_turns():
    messages = []
    for i in range(5):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    out = helpers.format_history(messages, max_turns=2)
    assert out == "User: q3\nAssistant: a3\nUser: q4\nAssistant: a4"


def test_source_label():
    assert helpers.source_label({"source": "a.pdf", "page": 3}) == "a.pdf, p. 3"
    assert helpers.source_label({"source": "a.pdf"}) == "a.pdf"
    assert helpers.source_label(None) == "document"


def test_fingerprint_changes_with_content_and_settings():
    base = helpers.file_fingerprint([b"abc"], 1000, 200)
    assert base == helpers.file_fingerprint([b"abc"], 1000, 200)
    assert base != helpers.file_fingerprint([b"abd"], 1000, 200)
    assert base != helpers.file_fingerprint([b"abc"], 800, 200)


def test_friendly_errors():
    assert "API key" in helpers.friendly_error(Exception("400 API_KEY_INVALID"))
    assert "rate limit" in helpers.friendly_error(Exception("429 RESOURCE_EXHAUSTED"))
    assert "Something went wrong" in helpers.friendly_error(Exception("boom"))


def test_prompts_format_without_missing_keys():
    common = dict(math_rule=prompts.MATH_RULE, language_rule=prompts.language_rule("Hindi"))
    prompts.QA_PROMPT.format(context="c {braces}", question="q", not_found="nf", **common)
    prompts.SUMMARY_PROMPT.format(text="t", **common)
    prompts.QUIZ_PROMPT.format(count=5, context="c", **common)
    prompts.ELIGIBILITY_PROMPT.format(context="c", not_found="nf", language_rule="l")
    prompts.TIMELINE_PROMPT.format(context="c", not_found="nf", language_rule="l")
    prompts.MAP_PROMPT.format(index=1, total=2, text="t")
    prompts.CONDENSE_PROMPT.format(history="h", question="q")
    assert "Hindi" in prompts.language_rule("Hindi")
    assert prompts.language_rule("Klingon") == prompts.LANGUAGES["English"]
