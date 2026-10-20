"""Regex catalogues describing authorship signals and quality deductions.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import re

from .base import EXPLICIT_ARTIFACT_WEIGHT

README_BLOCK = re.compile(
    r"^--- FILE: (?:.*/)?readme[^\n]*---[ \t]*\r?\n(.*?)(?=^--- FILE: |\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
README_FIRST_PERSON = re.compile(r"\b(?:I|we|my|our|us|I've|we've|I'm)\b")
README_CASUAL = re.compile(
    r"\b(?:lol|stuff|things|a lot of stuff|kinda|gonna|wanna|btw|tbh|hack(y|ed) together)"
    r"|[!]{2,}", re.IGNORECASE,
)
README_TYPOS = re.compile(
    r"\b(?:teh|recieve|seperate|definately|occured|alot|thier|wierd|adress|sucess)\b",
    re.IGNORECASE,
)
FILE_HEADER = re.compile(r"^--- FILE: (.+?) ---$", re.MULTILINE)
FILE_BLOCK = re.compile(
    r"^--- FILE: (.+?) ---[ \t]*\r?\n(.*?)(?=^--- FILE: |\Z)",
    re.MULTILINE | re.DOTALL,
)
COMMENT_LINE = re.compile(r"^\s*(?:#|//|\*|/\*)")
SPACED_SITE = re.compile(r", | = ")
BARE_SITE = re.compile(r",(?![ \n])|(?<![=!<>+\-*/%&|^:>])=(?![=>])")
FILES_SHOWN = re.compile(r"^FILES INCLUDED \((\d+)\)", re.MULTILINE)
TRUNCATION_MARKER = "[... Remaining files omitted for context limits ...]"

AI_SIGNALS = (
    (re.compile(
        r"as an ai language model|co-authored-by:\s*(?:github\s+)?copilot"
        r"|\b100\s*%\s*(?:ai|llm)[- ]generated\b"
        r"|\b100\s*%\s+(?:of\s+)?(?:the\s+)?(?:code|project|codebase|repository)"
        r"(?:\s+is|\s+was)?\s+(?:ai|llm)[- ]generated\b"
        r"|\b(?:entirely|fully|completely)\s+(?:ai|llm)[- ]generated\b"
        r"|generated entirely (?:with|by) "
        r"|(?:written|created) (?:entirely )?(?:by|with) (?:an? )?(?:ai|llm)\b"
        r"|(?:entire|whole) (?:repository|project|codebase) was (?:created|written|generated|built) by ai\b"
        r"|(?:no|zero) human[- ]written code\b"
        r"|created through (?:conversation|collaboration) between (?:a )?human and (?:an? )?(?:ai|llm)\b"
        r"|created through collaboration between a human and ai\b"
        r"|created by ai,? with no human[- ]written code\b"
        r"|generated (?:by|with|using) "
        r"(?:(?:github|microsoft|google|openai|aws|jetbrains)\s+)?"
        r"(?:chatgpt|gpt-?[0-9]|codex|copilot|claude|cursor|gemini|an? (?:ai|llm))"
        r"|written (?:by|with) (?:(?:github|microsoft|google|openai)\s+)?"
        r"(?:gpt|chatgpt|claude|copilot|cursor|an? (?:ai|llm)|ai|llm)", re.IGNORECASE,
    ), EXPLICIT_ARTIFACT_WEIGHT, "an explicit AI-generation marker", True),
    (re.compile(r"your code here|add your (?:code|logic) here|implement this|<!-- ?(?:describe|todo: add)", re.IGNORECASE), 18, "an unfilled prompt placeholder", True),
    (re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"), 4, "emoji used inside code or comments", False),
    (re.compile(r"^\s*(?:#|//)\s*[-=*\u2500-\u257f\u2014\u2015__]{4,}", re.MULTILINE), 4, "generated-looking banner comments", False),
    (re.compile(r"except\s+Exception\s+as\s+\w+\s*:\s*\n\s*print\((?:e|exc|err)", re.MULTILINE), 10, "copy-pasted uniform exception handling", False),
)

HUMAN_SIGNALS = (
    (re.compile(r"\b(?:TODO|FIXME|HACK|XXX|NOTE TO SELF)\b"), 6, "leftover TODO/FIXME notes"),
    (re.compile(r"\b(?:print|console\.log|System\.out\.println)\s*\("), 5, "debug output left in"),
    (re.compile(r"^\s*(?:#|//)\s*(?:if|for|while|return|def|function|\})\S*\s*$", re.MULTILINE), 5, "commented-out code"),
    (re.compile(r"\b(?:teh|recieve|seperate|definately|occured|alot|thier|wierd)\b", re.IGNORECASE), 4, "typos in comments"),
    (re.compile(r"[A-Za-z]:\\Users\\|/home/[a-z0-9._-]+/"), 4, "hard-coded local paths"),
)

# A deduction for something the repository does not have has to be about the repository, not
# about the excerpt: the reader names the paths it left out, so a README listed there exists
# and only its contents are missing from the view.
MISSING_README_REASON = "no README with setup instructions"
ABSENCE_DEDUCTIONS = (
    (re.compile(r"^--- FILE: .*readme", re.MULTILINE | re.IGNORECASE), 15, MISSING_README_REASON),
)

QUALITY_DEDUCTIONS = (
    (re.compile(r"except\s*(?:Exception\s*(?:as\s+\w+)?)?\s*:|except\s*:\s*(?:pass|\.\.\.)", re.MULTILINE), 8, "bare `except:` swallowing every error"),
    (re.compile(r"(?:api[_-]?key|secret|password|passwd|token|access[_-]?key)\s*[:=]\s*['\"][A-Za-z0-9_\-/+]{8,}['\"]|sk-[A-Za-z0-9]{20,}", re.IGNORECASE), 12, "hard-coded credentials"),
    (re.compile(r"^\s*(?:#|//).*\b(?:print|console\.log|debugger)\b", re.MULTILINE), 3, "debug scaffolding left in comments"),
    (re.compile(r"localhost|127\.0\.0\.1:\d{4}"), 4, "hard-coded localhost endpoints"),
    (re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b.*(?:f\"|\+ *(?:user|name|id))"), 10, "SQL built by string interpolation"),
    (re.compile(r"\b(?:password|secret|api_key)\b\s*=\s*['\"][A-Za-z0-9!@#$%^&*]{4,}['\"]"), 10, "credentials committed in source"),
)

COUNTED_PATTERNS = {
    "todos": re.compile(r"\b(?:TODO|FIXME|HACK|XXX|NOTE TO SELF)\b"),
    "debug_prints": re.compile(r"\b(?:print|console\.log|System\.out\.println)\s*\("),
    "commented_out": re.compile(r"^\s*(?:#|//)\s*(?:if|for|while|return|def|function|\})", re.MULTILINE),
    "local_paths": re.compile(r"[A-Za-z]:\\Users\\|/home/[a-z0-9._-]+/"),
    "emoji": re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"),
    "credentials": re.compile(r"(?:api[_-]?key|secret|password|passwd|token|access[_-]?key)\s*[:=]\s*['\"][A-Za-z0-9_\-/+]{8,}['\"]|sk-[A-Za-z0-9]{20,}", re.IGNORECASE),
    "localhost": re.compile(r"localhost|127\.0\.0\.1:\d{4}"),
    "sql_interpolation": re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b.*(?:f\"|\+ *(?:user|name|id))"),
    "bare_excepts": re.compile(r"except\s*(?:Exception\s*(?:as\s+\w+)?)?\s*:|except\s*:\s*(?:pass|\.\.\.)", re.MULTILINE),
    "docs_comments": re.compile(r"^\s*(?:#|//|\*|/\*)", re.MULTILINE),
}

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "with",
    "return", "returns", "get", "gets", "set", "sets", "is", "are", "this",
    "that", "it", "its", "from", "given", "using",
}
DEF_PATTERN = re.compile(
    r"^\s*(?:def|function|func|public|private|protected|static|export\s+(?:default\s+)?"
    r"(?:function\s+)?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE,
)
DOCSTRING_PATTERN = re.compile(
    r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\([^)]*\)[^:\n]*:"
    r"[ \t]*\r?\n[ \t]*(?:'''|\"\"\")(.*?)(?:'''|\"\"\")", re.MULTILINE | re.DOTALL,
)
JSDOC_PATTERN = re.compile(r"^\s*/\*\*", re.MULTILINE)
TYPED_DEF_PATTERN = re.compile(r"def\s+\w+\s*\([^)]*:\s*\w+[^)]*\)\s*->\s*\w+")
