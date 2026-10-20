"""Shared weights, cutoffs, and source categories for detector scoring.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

BASELINE_AI_SCORE = 0
EXPLICIT_ARTIFACT_SCORE = 88
MAX_AI_SCORE = 100
MAX_QUALITY_SCORE = 100
DEDUCTION_CAP = 85
GRADED_AI_CAP = 85
AI_FLAG_THRESHOLD = 40
EXPLICIT_ARTIFACT_WEIGHT = 60
RESTATEMENT_POINTS = 18
ANNOTATION_POINTS = 6
COMMENT_POINTS = 12
README_POINTS = 14
CLEANLINESS_POINTS = 4
CORROBORATION_POINTS = 4
CORROBORATION_CAP = 12
MIN_DOCUMENTED_FUNCTIONS = 2
VOLUME_IDEAL_LINES = 60
COMMENT_DENSITY_TARGET = 0.12
TRACE_LIMIT = 4
SATURATION_FLOOR = 0.5
# Below this many non-blank source lines, comment-to-code ratios describe a handful of lines
# rather than an authorship pattern, so documentation and cleanliness do not move the score.
# It equals the point where the cleanliness volume ramp starts (VOLUME_IDEAL_LINES * SATURATION_FLOOR).
MIN_SCORED_SOURCE_LINES = 30
RESTATEMENT_FLOOR = 0.4
README_MIN_CHARS = 400
README_IDEAL_CHARS = 2500
README_STRUCTURE_IDEAL = 20

SOURCE_EXTS = (
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts", ".go",
    ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".dart", ".scala", ".sh", ".ps1", ".sql", ".vue", ".svelte", ".jai",
    ".lua", ".r", ".jl", ".gml",
)
DOCUMENTATION_EXTS = (".md", ".markdown", ".txt", ".rst", ".adoc", ".mdx")
TRACES = ("todos", "debug_prints", "commented_out", "local_paths", "emoji")
BOILERPLATE_FILES = (
    ".gitignore", ".gitattributes", ".gitkeep", ".dockerignore", ".npmignore",
    ".prettierignore", ".eslintignore", ".editorconfig",
)
