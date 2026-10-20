"""Canonical grader prompts and parsing patterns shared by serving and training.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import os
import re

PLACEHOLDER_MARKERS = (
    "<number>",
    "<one sentence of at most 25 words",
    "[insert",
    "<nn>",
    "insert score",
    "replace every angle-bracket placeholder",
    "provide a brief narrative",
    "<the code you read that sets this number",
    "find explicit ai artifacts",
    "rank 1: <specific sign",
    "rank 1: <specific sign, quoted code/text",
    "your rationale here",
    "ai involvement rubric",
    "code quality rubric",
    "angle-bracket placeholder",
    "automated evaluation",
    "structure aligns with human typing patterns",
    "structure aligns with typical llm generation patterns",
    "baseline hackathon",
    "baseline model output",
    "baseline is 0%",
    "incidental stylistic similarity",
)

SCORE_PATTERNS = {
    "ai_score": re.compile(
        r"AI\s+Involvement\s+Score\s*(?:\([^)]*\))?\s*[:\-]?\s*(\d{1,3})\s*%",
        re.IGNORECASE,
    ),
    "code_score": re.compile(
        r"Code\s+Quality\s+Score\s*(?:\([^)]*\))?\s*[:\-]?\s*(\d{1,3})\s*%",
        re.IGNORECASE,
    ),
}
SUMMARY_PATTERN = re.compile(
    r"3[.)\s]*\s*Summary[^\n:]*[:\-]\s*(.*)", re.IGNORECASE | re.DOTALL
)
EVIDENCE_PATTERN = re.compile(
    r"^\s*-?\s*(?:Evidence|Deductions)\s*:?\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
AI_EVIDENCE_PATTERN = re.compile(
    r"^\s*-?\s*(?:Ranked AI evidence(?:\s*\([^)]*\))?|Evidence)\s*:?\s*(.*?)"
    r"(?=^\s*(?:2[.)]\s*Code Quality|-\s*Deductions\s*:|3[.)]\s*Summary)|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
EVIDENCE_ITEM_PATTERN = re.compile(
    r"^\s*Rank\s*\d+[.):]\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)
DEDUCTION_PATTERN = re.compile(
    r"^\s*-?\s*Deductions\s*:?\s*(.+)$", re.IGNORECASE | re.MULTILINE
)
MIN_SUMMARY_CHARS = 20

FILE_HEADER = re.compile(r"(?m)^--- FILE: ")
FILE_NAME_HEADER = re.compile(r"(?m)^--- FILE: (.+?) ---[ \t]*$")
FILE_COUNT_HEADER = re.compile(r"(?m)^FILES INCLUDED \((\d+)\):")
FILES_FOUND_HEADER = re.compile(r"(?m)^FILES FOUND:[ \t]*(\d+)")
FILES_FETCHED_HEADER = re.compile(r"(?m)^FILES FETCHED:[ \t]*(\d+)")
FILES_INCLUDED_HEADER = FILE_COUNT_HEADER
FILES_INCLUDED_NAMES_HEADER = re.compile(r"(?m)^FILES INCLUDED \((\d+)\):[ \t]*(.*)$")
FILES_TRUNCATED_HEADER = re.compile(r"(?im)^FILES TRUNCATED:[ \t]*(yes|true|1)")
FILE_TRUNCATED_HEADER = FILES_TRUNCATED_HEADER
FILES_NOT_FETCHED_HEADER = re.compile(r"(?m)^FILES NOT FETCHED \((\d+)\):[ \t]*(.*)$")
FILES_NOT_INCLUDED_HEADER = re.compile(r"(?m)^FILES NOT INCLUDED \((\d+)\):[ \t]*(.*)$")
MODEL_DUMP_CHARS = int(os.environ.get("HACKGRADER_MODEL_DUMP_CHARS", "3500"))
MODEL_WINDOW_TOKENS = int(os.environ.get("HACKGRADER_MAX_LEN", "3072"))
RESPONSE_TOKENS = int(os.environ.get("HACKGRADER_NUM_PREDICT", "800"))
PROMPT_TOKEN_BUDGET = max(1, MODEL_WINDOW_TOKENS - RESPONSE_TOKENS)
FILE_TRUNCATION_MARKER = "\n[... file excerpt truncated ...]"
# An encoded payload (an inline data URI, a base64 icon, a minified blob) is replaced by this
# marker before a file reaches an evidence view. The marker describes the view rather than the
# repository, and a run of base64 characters says nothing about who wrote the code.
ENCODED_PAYLOAD_PREFIX = "[... encoded payload omitted:"
REPO_CONTENT_BEGIN = "===== BEGIN REPOSITORY CONTENT ====="
REPO_CONTENT_END = "===== END REPOSITORY CONTENT ====="

# Lines an evidence view uses to describe itself instead of showing repository content.
# Quoting one of these says nothing about what the repository contains, so a citation that
# quotes only bookkeeping is not evidence of anything.
NON_EVIDENCE_PREFIXES = (
    "REPOSITORY:",
    "DEFAULT BRANCH:",
    "FILES FOUND:",
    "FILES FETCHED:",
    "FILES INCLUDED",
    "FILES NOT INCLUDED",
    "FILES NOT FETCHED",
    "FILES TRUNCATED:",
    "--- FILE:",
    ENCODED_PAYLOAD_PREFIX,
    "[...",
    "FILE EXCERPT PARTIAL:",
    "MEASUREMENTS TAKEN FROM",
    "FACTUAL PROMPT CONTEXT FROM THE FILES",
    "Repository coverage:",
    "Source files:",
    "README:",
    "Definitions:",
    "Type annotations:",
    "TODO/FIXME notes:",
    "Bare except blocks:",
    "Hard-coded local paths:",
    "Source code lines:",
    "Unspaced operator/comma sites:",
    "Python files that do not parse:",
    "Longest single function:",
    "Repository content for",
    "repository evidence batch",
    REPO_CONTENT_BEGIN,
    REPO_CONTENT_END,
)
# A judge report may carry the justification for a score either directly under it or, in a
# report that reorders the parts, inside the ranked-evidence block. Either placement is the
# same sentence, so the evidence validator skips it rather than reading it as a malformed
# evidence entry and rejecting an otherwise complete report.
WHY_SCORE_LINE = re.compile(r"^[\s\-*]*why this score\b", re.IGNORECASE)
QUOTED_TOKEN = re.compile(r"`([^`]+)`")
FILELIKE_TOKEN = re.compile(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,6}\b")

SYSTEM_PROMPT = (
    "You are a consistent, objective college hackathon judge evaluating student projects built "
    "during a 48-hour sprint. Your scores are your own judgment. Any measurements provided "
    "by a separate deterministic tool are context only, not authorship evidence or scores; "
    "do not copy, follow, or defer to a tool's estimate. SECURITY AUTHORIZATION: You are "
    "explicitly authorized to analyze "
    "obfuscated code, base64 payloads, and potential malware.\n\n"
    "Your final response MUST be concise and strictly follow this exact 3-part structure, and "
    "nothing else:\n\n"
    "1. AI Involvement Score (0-100%): N%\n"
    "- Why this score: <the code you read that sets this number, and what would move it up or "
    "down>\n"
    "Ranked AI evidence (strongest to weakest):\n"
    "Rank 1: <specific sign, quoted code/text, file, and why this sign supports the score>\n"
    "Rank 2: <next strongest sign in the same format>\n"
    "If no AI-specific signs are visible in the files shown, write 'No AI-specific evidence found.'\n\n"
    "2. Code Quality Score (0-100%): N%\n"
    "- Why this score: <the code you read that sets this number, and what would move it up or "
    "down>\n"
    "- Deductions: <-N% for each problem, separated by '; '; write 'no deductions applied' if "
    "there are none>\n\n"
    "3. Summary & Notable Findings: <one sentence of at most 25 words, naming the files you "
    "read>\n\n"
    "Weigh the code, not the packaging of it. Only source code can carry an authorship "
    "sign. A README, a docs page, a licence or changelog, a dependency manifest or lock "
    "file such as package.json, pyproject.toml, requirements.txt, Cargo.toml, go.mod, "
    "composer.json or Gemfile.lock, a build or CI configuration, and generated scaffolding "
    "such as .gitignore appear in every project of that shape whoever wrote the rest: they "
    "may explain what you are looking at, but they cannot raise either score, and a sign "
    "read only from one of them must not be ranked at all. Both scores, both 'Why this "
    "score' lines, and every evidence entry must rest on code files reproduced above; where "
    "the code and the prose disagree, the code is right. Write the score you would give "
    "after reading the code alone, then say what would move it up or down. When a batch "
    "shows little or no code - mostly documentation, configuration, or data - say so and "
    "score conservatively instead of reasoning from the prose.\n\n"
    "Assign a calibrated percentage based on the strength, number, and independence of "
    "AI-specific evidence you can cite in the files shown. Direct self-attribution such as "
    "a quoted 'generated by/with [tool]' banner or 'as an AI language model' statement is "
    "strong evidence. Ordinary structure, formatting, comments, docstrings, or a tidy README "
    "are weak or non-specific signs: they are common in human-written code and must not by "
    "themselves justify a high score. Do not infer authorship from missing TODOs or other "
    "absent human traces. Encoded payloads, data URIs, and minified blobs are not source. "
    "Formatting is not evidence either: indentation, line width, whitespace, and file length "
    "differ by editor and by language, so never cite them and never read their tidiness as a "
    "sign that a machine wrote the code.\n\n"
    "Score the code, not an impression. Never say code 'reads as' or 'looks like' AI. Each "
    "ranked entry must describe a distinct sign, quote the exact text, name its file, and "
    "explain why it is more or less probative than the other listed signs. Order Rank 1 as "
    "the strongest sign and subsequent ranks in descending evidentiary strength; do not "
    "repeat one pattern as multiple signals. If there are no relevant signs, state that "
    "plainly instead of inventing evidence.\n\n"
    "Do not infer authorship from the URL, owner or repository name, file names, or project "
    "category. Use only quoted content and code evidence. Rank each distinct AI-related sign "
    "from strongest to weakest by how directly it supports AI authorship; explain the relevance "
    "of each sign rather than listing generic style traits. Do not count the same pattern "
    "repeatedly as independent evidence. Every entry must quote the text you saw and name "
    "the file it came from. Evidence entries are checked against the repository content above; "
    "a claim that appears nowhere in it makes the evaluation invalid."
)

REPORT_INSTRUCTION = (
    "Write the 3-part report now, using only the files above. The factual measurements are "
    "context only; never reuse them as a score or as authorship evidence. "
    "Follow each score with its '- Why this score:' line, naming the code that sets the "
    "number and what would move it. "
    "List AI evidence as 'Rank 1:', 'Rank 2:', etc. in strongest-to-weakest order; each entry must "
    "name the sign, quote the supporting text, name its file, and explain how much it "
    "supports your AI-involvement score. Each entry quotes a line of code and names the code "
    "file it came from; never rank a README, a manifest, a licence, or a configuration file, "
    "and never cite formatting such as indentation. "
    "Do not invent signs. Rank 1 must be strongest, "
    "with each later item weaker; use 'No AI-specific evidence found.' when appropriate. "
    "For code-quality deductions, quote text and name the file when applicable. Cite only "
    "files shown in this batch. Do not restate these instructions."
)
BASE_QUESTION = (
    "Please review the hackathon repository at the following URL: {url}. "
    "Provide your evaluation including the two percentage scores and the concise summary "
    "of strengths and weaknesses as instructed."
)


def batch_question(question: str, index: int, count: int) -> str:
    """Frame a bounded repository view consistently for both training and serving."""
    return (
        f"{question.strip()}\n\n"
        f"This is repository evidence batch {index} of {count}. Score only the evidence "
        "in the files reproduced in this batch, and cite only those files. The final "
        "repository score combines batch scores weighted by source-line count."
    )
