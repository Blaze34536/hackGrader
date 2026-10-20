"""Build bounded model views while preserving whole-file coverage metadata.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from .prompts import FILE_NAME_HEADER, FILE_TRUNCATION_MARKER, MODEL_DUMP_CHARS
from .repository_dump import (
    UNFETCHED_CONTENT_MARKER,
    _base_header,
    _parse_metadata,
    _view_header,
    carries_authorship_evidence,
    view_priority,
)

# While other files still wait for space, a file may claim only a share of a view. A large file
# therefore cannot fill a view by itself, and several files appear in each view. Source files
# claim the larger share: three of them take five-sixths of a view, which leaves the rest for
# documentation and configuration, so packaging still appears in every view that has code in it.
# Space a file does not claim in the first pass goes to files in priority order in the second.
SOURCE_SHARE = 5 / 18
OTHER_SHARE = 1 / 6

# One file's contribution to a view: (name, file header, full body, start offset, end offset).
Slice = tuple[str, str, str, int, int]


def _make_batch(
    base: str,
    parts: list[str],
    all_fetched: list[str],
    *,
    partial_file: bool = False,
) -> str:
    """Serialize file parts with path-level and excerpt-level coverage metadata."""
    names = list(dict.fromkeys(FILE_NAME_HEADER.search(part).group(1).strip() for part in parts))
    return _view_header(base, names, all_fetched, partial_file=partial_file) + "".join(parts)


def _render(base: str, slices: list[Slice], all_fetched: list[str]) -> str:
    """Serialize each file's chosen slice, marking the view partial when any file is cut."""
    parts = []
    partial = False
    for _, file_header, body, start, end in slices:
        if start > 0 or end < len(body):
            partial = True
        excerpt = body[start:end]
        if end < len(body):
            # The marker says where the excerpt was cut. The newline after it starts the next
            # file's header on a line of its own, so the next file is still recognised.
            excerpt += FILE_TRUNCATION_MARKER + "\n"
        parts.append(file_header + excerpt)
    return _make_batch(base, parts, all_fetched, partial_file=partial)


def _grow(
    base: str,
    chosen: list[Slice],
    index: int,
    limit: int,
    all_fetched: list[str],
    budget: int,
) -> int:
    """Extend ``chosen[index]`` as far as the view allows without passing ``limit``.

    The end offset is found by bisection on the rendered view, so the result is the largest one
    that keeps the view within budget. A cut is moved back to a line break when that keeps at
    least half of the new slice, so an excerpt does not end mid-line when it need not.
    Returns the new end offset.
    """
    name, file_header, body, start, end = chosen[index]
    best = end
    low, high = end + 1, min(len(body), limit)
    while low <= high:
        middle = (low + high) // 2
        trial = [*chosen[:index], (name, file_header, body, start, middle), *chosen[index + 1 :]]
        if len(_render(base, trial, all_fetched)) <= budget:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best < len(body):
        cut = body.rfind("\n", start, best)
        if cut >= start + (best - start) // 2 and cut + 1 > end:
            best = cut + 1
    chosen[index] = (name, file_header, body, start, best)
    return best


def split_dump(dump: str, budget: int | None = None) -> list[str]:
    """Split every fetched file into bounded views that identify other omitted paths."""
    budget = MODEL_DUMP_CHARS if budget is None else max(0, budget)
    if budget <= 0 or not dump:
        return []
    meta = _parse_metadata(dump)
    files = [item for item in meta.files if UNFETCHED_CONTENT_MARKER not in item[2]]
    if not files:
        return [dump[index : index + budget] for index in range(0, len(dump), budget)]

    base = _base_header(meta)
    all_fetched = list(dict.fromkeys([name for name, _, _ in files] + meta.not_included))
    # Files are offered in priority order rather than the order the dump happened to list them.
    # The header carries the collector's fetch order, so following it let a small README travel
    # in the first view with the first code file and lead the batch the judge read first.
    queue = sorted(files, key=lambda item: view_priority(item[0]))
    offsets = {name: 0 for name, _, _ in queue}
    batches: list[str] = []

    while True:
        waiting = [item for item in queue if offsets[item[0]] < len(item[2])]
        if not waiting:
            break
        shared = len(waiting) > 1
        chosen: list[Slice] = []
        # First pass: each waiting file takes up to its share of the view, in priority order.
        for name, file_header, body in waiting:
            start = offsets[name]
            share = SOURCE_SHARE if carries_authorship_evidence(name) else OTHER_SHARE
            limit = start + int(budget * share) if shared else len(body)
            chosen.append((name, file_header, body, start, start))
            if _grow(base, chosen, len(chosen) - 1, limit, all_fetched, budget) == start:
                chosen.pop()
        # Second pass: spend the space the shares left unused, again in priority order.
        if shared:
            for index, (_, _, body, _, end) in enumerate(chosen):
                if end < len(body):
                    _grow(base, chosen, index, len(body), all_fetched, budget)
        if not chosen:
            raise ValueError(
                f"The configured {budget}-character budget cannot fit repository headers."
            )
        batch = _render(base, chosen, all_fetched)
        if len(batch) > budget:
            raise ValueError("A repository batch exceeds the configured character budget.")
        batches.append(batch)
        for name, _, _, _, end in chosen:
            offsets[name] = end
    return batches
