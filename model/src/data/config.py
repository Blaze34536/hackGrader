"""Configure candidate sources, split quotas, and score targets.

Target scores sit in separate human, AI, and artifact bands. A position below the midpoint
maps into the lower half of the human band, so the reachable human range runs from
``HUMAN_LABEL_BAND[0]`` to that band's midpoint: 15 to 20 at the default band, twenty
points under the ``FLAG_THRESHOLD`` boundary. That margin is the one a model has to spend
before a human-band example crosses the boundary as a false alarm, and the reliability
gate admits at most 20% false alarms on the human side, so an earlier 25-35 band left only
ten points and failed specificity once per-row error reached fifteen. Each band keeps a
margin of at least ``MIN_LABEL_BAND_MARGIN`` points from the boundary, and a labeled
example is only supervised when its own visible measurements land on its label's side of
the midpoint (``MIN_EVIDENCE_MARGIN``). Without those rules a band edge sits adjacent to
the boundary and the model is trained against its own evidence, which is what made the
validation reliability gate unreachable.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import os

DATA_DIR = "data"
DEFAULT_TRAIN = f"{DATA_DIR}/dataset.jsonl"
DEFAULT_VAL = f"{DATA_DIR}/dataset.val.jsonl"
DEFAULT_TEST = f"{DATA_DIR}/dataset.test.jsonl"
DEFAULT_CANDIDATE = f"{DATA_DIR}/dataset.candidate.jsonl"
# Human-bias benchmark repositories, held out of every partition and never harvested.
BENCHMARK_REPOS_FILE = os.environ.get(
    "HACKGRADER_BENCHMARK_REPOS", f"{DATA_DIR}/benchmark_repos.txt"
)
DEFAULT_TRAIN_CANDIDATE = f"{DATA_DIR}/dataset.train.candidate.jsonl"
DEFAULT_VAL_CANDIDATE = f"{DATA_DIR}/dataset.val.candidate.jsonl"
DEFAULT_TEST_CANDIDATE = f"{DATA_DIR}/dataset.test.candidate.jsonl"
DATA_SPLITS = ("train", "validation", "test")
VALIDATION_FRACTION = 0.12
TEST_FRACTION = 0.12
MIN_SNIPPET_CHARS = 40
AIG_CSV = (
    "hf://datasets/basakdemirok/AIGCodeSet/"
    "data/all_data_with_ada_embeddings_will_be_splitted_into_train_test_set.csv"
)
AIG_ROWS = int(os.environ.get("HACKGRADER_AIG_ROWS", "2500"))
SNIPPET_CHARS = int(os.environ.get("HACKGRADER_SNIPPET_CHARS", "1500"))
REPO_REPEAT = int(os.environ.get("HACKGRADER_REPO_REPEAT", "1"))
SYNTHETIC_REPOS = int(os.environ.get("HACKGRADER_SYNTHETIC_REPOS", "256"))
SYNTHETIC_FILES = int(os.environ.get("HACKGRADER_SYNTHETIC_FILES", "2"))
SYNTHETIC_FILE_CHARS = int(os.environ.get("HACKGRADER_SYNTHETIC_FILE_CHARS", "420"))
ARTIFACT_ROWS = int(os.environ.get("HACKGRADER_ARTIFACT_ROWS", "48"))
TARGET_CAP = int(os.environ.get("HACKGRADER_TARGET_CAP", "1000000"))
MIN_SCORE_BAND_ROWS = int(os.environ.get("HACKGRADER_MIN_SCORE_BAND_ROWS", "10"))
if MIN_SCORE_BAND_ROWS < 1:
    raise ValueError("each score band needs at least one distinct prompt")
# A snippet-only build trains the style and quality reading without the real-repository
# head. The default pipeline requires real repositories in every split, because
# repository-shaped views are what the served grader actually reads, and it now refuses to
# build them without a verified label manifest: a repository target comes from the share a
# harvest attributed, never from the detector's style reading, which on real code put most
# of both classes in the AI band. Snippet-only remains the explicit way to build without
# the repository head.
SNIPPET_ONLY = os.environ.get("HACKGRADER_SNIPPET_ONLY", "0") == "1"


def required_real_repository_floor(snippet_only: bool, explicit: str | None) -> int:
    """Return the per-split real-repository floor, refusing a contradictory setting."""
    if snippet_only:
        if explicit is not None and int(explicit):
            raise ValueError(
                "HACKGRADER_SNIPPET_ONLY and a positive "
                "HACKGRADER_MIN_REAL_REPOSITORIES_PER_SPLIT contradict each other: a "
                "snippet-only build contains no real-repository rows"
            )
        return 0
    value = 2 if explicit is None else int(explicit)
    if value < 1:
        raise ValueError("at least one independent real repository is required per split")
    return value


MIN_REAL_REPOSITORIES_PER_SPLIT = required_real_repository_floor(
    SNIPPET_ONLY, os.environ.get("HACKGRADER_MIN_REAL_REPOSITORIES_PER_SPLIT")
)
REPO_CACHE_DIR = os.environ.get("HACKGRADER_REPO_CACHE", ".cache/repos")
REPO_CACHE_HOURS = float(os.environ.get("HACKGRADER_REPO_CACHE_HOURS", "24"))
MAX_TARGET_SHARE = 0.25

# Cohort harvest settings. The harvest is what turns repository authorship from a style
# reading into a recomputable record — an agent pull request, its merge commit, and the
# commit that last touched every file examined — and its size is set by the rate limit
# rather than by ambition: a token allows 5,000 requests an hour, and the per-file rule
# spends one per examined file. Without a verified cohort the repository head has no
# supervision it can defend, so these settings decide how quickly that changes.
HARVEST_COHORT_FILE = os.environ.get(
    "HACKGRADER_HARVEST_COHORT", f"{DATA_DIR}/cohort.jsonl"
)
HARVEST_BUDGET = int(os.environ.get("HACKGRADER_HARVEST_BUDGET", "5000"))
HARVEST_MAX_FILES = int(os.environ.get("HACKGRADER_HARVEST_MAX_FILES", "40"))
HARVEST_TOUCH_WINDOW = int(os.environ.get("HACKGRADER_HARVEST_TOUCH_WINDOW", "5"))
HARVEST_HUMAN_BEFORE = os.environ.get("HACKGRADER_HARVEST_HUMAN_BEFORE", "2023-01-01")
HARVEST_PAUSE_SECONDS = float(os.environ.get("HACKGRADER_HARVEST_PAUSE", "0"))

AI_LIKE_FEATURES = {
    "readme structure": 8.0,
    "documented functions": 1.0,
    "restated docstrings": 0.5,
    "comment density": 0.12,
}
# Indentation and trailing whitespace are not human-like features. ``machine_likeness``
# inverts this map, so zero tab-indented lines read as fully machine-like and the recorded
# build log measured that inversion carrying 21% of a repository target; the served model
# answered "0 line(s) are tab indented" for ordinary projects that indent with spaces.
# Formatting is not authorship evidence, so it is not supervised here.
HUMAN_LIKE_FEATURES = {
    "readme first person": 3.0,
    "readme informal": 2.0,
    "human traces": 6.0,
}
MIN_EFFECT = 0.15

# The two maps above are saturation scales, not weights: each says the measurement at which
# a feature reads as fully machine-like or fully human-like. What a target actually rests on
# is the measured separation between the label groups, which ``feature_weights`` turns into a
# share of the target.
#
# A README is the project's own account of itself and costs nothing to write, so README shape
# must not carry a target. The recorded build log for repository rows measured the three
# README features taking 66% of the weight between them (structure 23%, first person 21%,
# informal 22%) while every code feature together took 34%, and those rows taught the grader
# to justify a score with "the README has N heading(s) and M bullet(s)". Documentation
# features therefore hold at most this share of a target; code-derived features take the
# rest, and documentation keeps its measured share only when nothing else separates the
# labels at all.
DOCUMENTATION_FEATURES = frozenset(
    {"readme structure", "readme first person", "readme informal"}
)
MAX_DOCUMENTATION_WEIGHT_SHARE = 0.10

FLAG_THRESHOLD = 40


def _band(name: str, default: str) -> tuple[int, int]:
    """Parse an inclusive score range from its environment override."""
    low, _, high = os.environ.get(name, default).partition(",")
    bounds = int(low), int(high)
    if not 0 <= bounds[0] <= bounds[1] <= 100:
        raise ValueError(f"{name} must be an ordered range inside 0 through 100")
    return bounds


AI_LABEL_BAND = _band("HACKGRADER_AI_LABEL_BAND", "45,85")
HUMAN_LABEL_BAND = _band("HACKGRADER_HUMAN_LABEL_BAND", "15,25")
ARTIFACT_BAND = _band("HACKGRADER_ARTIFACT_BAND", "88,99")
BOUNDARY_BAND = _band("HACKGRADER_BOUNDARY_BAND", "25,60")
MIN_LABEL_BAND_MARGIN = int(os.environ.get("HACKGRADER_MIN_LABEL_BAND_MARGIN", "5"))
MIN_EVIDENCE_MARGIN = float(os.environ.get("HACKGRADER_MIN_EVIDENCE_MARGIN", "0.1"))

if HUMAN_LABEL_BAND[1] >= AI_LABEL_BAND[0]:
    raise ValueError("the human and AI label bands must not overlap")
if AI_LABEL_BAND[1] >= ARTIFACT_BAND[0]:
    raise ValueError("the AI and artifact label bands must not overlap")
if not 0 <= MIN_EVIDENCE_MARGIN < 0.5:
    raise ValueError("HACKGRADER_MIN_EVIDENCE_MARGIN must be at least 0 and below 0.5")
if MIN_LABEL_BAND_MARGIN < 0:
    raise ValueError("HACKGRADER_MIN_LABEL_BAND_MARGIN cannot be negative")
if FLAG_THRESHOLD - HUMAN_LABEL_BAND[1] < MIN_LABEL_BAND_MARGIN:
    raise ValueError(
        f"HACKGRADER_HUMAN_LABEL_BAND must end at least {MIN_LABEL_BAND_MARGIN} points "
        f"below the {FLAG_THRESHOLD}% flag threshold, or ordinary score noise flips a "
        "human-labeled example into a false alarm"
    )
if AI_LABEL_BAND[0] - FLAG_THRESHOLD < MIN_LABEL_BAND_MARGIN:
    raise ValueError(
        f"HACKGRADER_AI_LABEL_BAND must start at least {MIN_LABEL_BAND_MARGIN} points "
        f"above the {FLAG_THRESHOLD}% flag threshold, or ordinary score noise flips an "
        "AI-labeled example into a missed detection"
    )
if not BOUNDARY_BAND[0] < FLAG_THRESHOLD <= BOUNDARY_BAND[1]:
    raise ValueError(
        "HACKGRADER_BOUNDARY_BAND must straddle the 40% flag threshold, or training "
        "gets no examples on one side of the decision it is being asked to make"
    )
