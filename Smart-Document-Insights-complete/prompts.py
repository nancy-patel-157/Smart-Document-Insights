"""Prompt templates. Kept in one place so wording can be tuned without touching logic."""

LANGUAGES = {
    "English": "Write the entire response in clear, simple English.",
    "Hindi": (
        "Write the entire response in fluent, natural Hindi (Devanagari script). "
        "Keep technical terms, names, numbers and dates exactly as they appear in the document."
    ),
}

MATH_RULE = (
    "If the answer contains formulas or equations, write them in LaTeX: "
    "inline math as $...$ and display math as $$...$$."
)

NOT_FOUND = {
    "English": "I could not find this in the uploaded document.",
    "Hindi": "यह जानकारी अपलोड किए गए दस्तावेज़ में नहीं मिली।",
}


def language_rule(language):
    return LANGUAGES.get(language, LANGUAGES["English"])


CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, rewrite the
follow-up as a single standalone question that can be understood without the conversation.
Keep it in the same language as the follow-up. Return only the rewritten question.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""


QA_PROMPT = """You are a careful document assistant. Answer the question using ONLY the
context below, which comes from the user's uploaded documents.

Rules:
- If the context does not contain the answer, reply exactly: "{not_found}"
- Do not use outside knowledge and do not guess.
- After each fact, cite where it came from using the label in square brackets, e.g. [report.pdf, p. 3].
- {math_rule}
- {language_rule}

Context:
{context}

Question: {question}
Answer:"""


SUMMARY_PROMPT = """Analyse the document text below and write a report in Markdown with exactly
these sections:

# Executive Summary
3-4 short paragraphs explaining what the document is about and why it matters.

# 10 Key Points
A numbered list of exactly 10 of the most important points, rules or findings.

# Action Items & Decisions
Bullet points with recommended next steps based only on the document.

Use only information from the text. {math_rule} {language_rule}

Document text:
{text}"""


MAP_PROMPT = """Summarise this part of a longer document in 8-12 dense bullet points.
Keep every important rule, number, date, eligibility condition and decision.
Write in English (it will be combined with other parts later).

Part {index} of {total}:
{text}"""


ELIGIBILITY_PROMPT = """From the document excerpts below, extract every eligibility criterion
(for example: who can apply, age limits, educational qualifications, minimum marks or
percentage, nationality, category or reservation rules, income limits, experience, documents
required).

Return a Markdown table with the columns: Criterion | Requirement | Source
- Source is the label shown before each excerpt, e.g. report.pdf, p. 3.
- One row per distinct requirement. Quote numbers and limits exactly.
- If an excerpt contradicts another, include both rows.
- If no eligibility criteria are present, reply only: "{not_found}"
- {language_rule}

Excerpts:
{context}"""


TIMELINE_PROMPT = """From the document excerpts below, extract every important date, deadline,
schedule entry and milestone.

Return a Markdown table with the columns: Date | Event | Source
- Sort rows chronologically. If a year is missing, keep the date as written.
- Only include dates that appear in the excerpts; never invent or calculate dates.
- Source is the label shown before each excerpt, e.g. report.pdf, p. 3.
- After the table, add one line highlighting the nearest or most critical deadline, if any.
- If no dates are present, reply only: "{not_found}"
- {language_rule}

Excerpts:
{context}"""


QUIZ_PROMPT = """Create {count} multiple-choice revision questions from the document excerpts
below. The excerpts are spread across the whole document, so cover different parts.

Format (Markdown):
**Q1. question text**
- A) option
- B) option
- C) option
- D) option

...then, after all questions, a section "### Answer Key" listing each answer with a one-line
explanation and its source label (e.g. report.pdf, p. 3).

Use only facts from the excerpts. {math_rule} {language_rule}

Excerpts:
{context}"""


ELIGIBILITY_QUERIES = [
    "eligibility criteria who can apply",
    "age limit minimum maximum age",
    "educational qualification minimum marks percentage",
    "nationality category reservation income criteria",
    "documents required for application",
]

TIMELINE_QUERIES = [
    "important dates and deadlines",
    "last date of registration application submission",
    "schedule timeline phases milestones",
    "result announcement date event date",
]
