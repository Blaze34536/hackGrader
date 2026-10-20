"""Select useful repository paths and compact file-list metadata.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import re

from contracts.prompts import ENCODED_PAYLOAD_PREFIX
from contracts.repository_dump import (
    DOCUMENTATION_EXTENSIONS,
    OTHER_CATEGORY,
    README_CATEGORY,
    view_priority,
)

from .settings import (
    IGNORED_DIRS,
    IGNORED_EXTENSIONS,
    IGNORED_FILES,
    MAX_PATH_LIST_CHARS,
)

# A source file is printable text; a weight blob, archive, or compressed asset is mostly
# bytes that have no printable form, and its bytes are what the grader quotes back. The
# file name is not enough on its own: extensionless and mislabelled blobs are common, so
# the fetched bytes are judged too.
_BINARY_SAMPLE_CHARS = 2000
_NON_TEXT_SHARE = 0.1
_READABLE_WHITESPACE = "\n\r\t"

# An encoded payload is text, so the binary check below passes it through, and it is exactly
# what a grader quotes back as evidence: a run of base64 characters cannot be read as source
# and says nothing about who wrote the file. Replacing the run with a marker that states the
# omission keeps the view truthful about what it left out, and the marker itself is
# bookkeeping rather than repository content, so a citation of it is not evidence.
_ENCODED_PAYLOAD = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")
_ENCODED_MARKER = f"{ENCODED_PAYLOAD_PREFIX} {{count}} character(s) ...]"

# A nested README is the least valuable thing in a repository of any size: a large one holds
# a README in dozens of subprojects, and each one read is budget taken from code. Only the
# root README keeps the place its category earns.
_NESTED_README_RANK = OTHER_CATEGORY + 1


def is_candidate(path: str) -> bool:
    """Return whether a tree path is source, project metadata, or useful documentation."""
    lowered = path.lower()
    if lowered.endswith(IGNORED_EXTENSIONS) or lowered.rsplit("/", 1)[-1] in IGNORED_FILES:
        return False
    return not any(
        lowered.startswith(directory) or f"/{directory}" in lowered
        for directory in IGNORED_DIRS
    )


def looks_binary(text: str) -> bool:
    """Return whether fetched text is a binary artifact rather than readable source.

    A NUL byte anywhere is decisive: no source file carries one, while every pickled or
    compiled blob does. Otherwise the opening characters are sampled, because decoding
    drops undecodable bytes and leaves only the fragments a binary file barely resembles
    text with. The sample is bounded so a large weight file is not scanned in Python before
    it is discarded.
    """
    if not text:
        return False
    if "\x00" in text:
        return True
    sample = text[:_BINARY_SAMPLE_CHARS]
    non_text = sum(
        1 for char in sample if not char.isprintable() and char not in _READABLE_WHITESPACE
    )
    return non_text > len(sample) * _NON_TEXT_SHARE


def redact_payloads(text: str) -> str:
    """Replace long encoded payloads with a marker, so no view can quote bytes as evidence."""
    return _ENCODED_PAYLOAD.sub(
        lambda match: _ENCODED_MARKER.format(count=len(match.group(0))), text
    )


def is_documentation(path: str) -> bool:
    """Return whether a tree path is documentation rather than project source."""
    return path.lower().endswith(DOCUMENTATION_EXTENSIONS)


def selection_priority(name: str) -> tuple[int, int, str]:
    """Rank tree paths so a bounded read reaches code before any documentation.

    The read stops once the total character budget is spent, so a category's rank decides
    whether its files are read at all. Source, entry points, and tests come first, then the
    project's own metadata and configuration, then the README and every other document.
    The README is the project's description of itself: reading it ahead of the code is how
    a bounded read returns a description instead of a repository.
    """
    category, depth, lowered = view_priority(name)
    if category == README_CATEGORY and depth > 0:
        return _NESTED_README_RANK, depth, lowered
    return category, depth, lowered


def path_listing(paths: list[str]) -> str:
    """Serialize paths compactly while signaling when names exceed the header budget."""
    unique = list(dict.fromkeys(paths))
    joined = ", ".join(unique)
    if len(joined) <= MAX_PATH_LIST_CHARS:
        return joined
    return "path list exceeds header budget; names unavailable"
