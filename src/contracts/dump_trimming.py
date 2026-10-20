"""Select representative file excerpts under the model context budget.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from .prompts import (
    FILE_COUNT_HEADER,
    FILE_TRUNCATION_MARKER,
    FILES_NOT_FETCHED_HEADER,
    FILES_NOT_INCLUDED_HEADER,
    MODEL_DUMP_CHARS,
)
from .repository_dump import (
    DOCUMENTATION_EXTENSIONS,
    SOURCE_EXTENSIONS,
    UNFETCHED_CONTENT_MARKER,
    _base_header,
    _file_blocks,
    _parse_metadata,
    _view_header,
    view_limits,
    view_priority,
)


def _compact_header(header: str, budget: int) -> str:
    """Shorten verbose omission paths before excerpt selection."""
    if len(header) <= budget // 3:
        return header
    header = FILE_COUNT_HEADER.sub(
        lambda match: f"FILES INCLUDED ({match.group(1)}): see reproduced file headers",
        header,
    )
    header = FILES_NOT_INCLUDED_HEADER.sub(
        lambda match: f"FILES NOT INCLUDED ({match.group(1)}): see other evidence batches",
        header,
    )
    return FILES_NOT_FETCHED_HEADER.sub(
        lambda match: f"FILES NOT FETCHED ({match.group(1)}): path list exceeds header budget",
        header,
    )


def _single_file(dump: str, budget: int, file_header: str, body: str) -> str:
    """Fit a single unlabelled file into the budget without wasting remaining space."""
    marker_size = len(FILE_TRUNCATION_MARKER)
    excerpt_size = min(len(body), max(0, budget - len(file_header) - marker_size))
    content = file_header + body[:excerpt_size]
    if excerpt_size < len(body):
        content += FILE_TRUNCATION_MARKER
    if len(content) < budget and excerpt_size < len(body):
        extra = min(budget - len(content), len(body) - excerpt_size)
        content = file_header + body[: excerpt_size + extra]
        if excerpt_size + extra < len(body):
            content += FILE_TRUNCATION_MARKER
    return content[:budget]


def trim_dump(dump: str, budget: int | None = None) -> str:
    """Select representative excerpts under a character budget while retaining identities."""
    budget = MODEL_DUMP_CHARS if budget is None else max(0, budget)
    if budget <= 0:
        return ""
    if len(dump) <= budget:
        return dump

    meta = _parse_metadata(dump)
    files = [item for item in meta.files if UNFETCHED_CONTENT_MARKER not in item[2]]
    if not files:
        return meta.header[:budget]
    if len(files) == 1 and not meta.header.strip():
        return _single_file(dump, budget, files[0][1], files[0][2])

    base = _compact_header(_base_header(meta), budget)
    all_fetched = list(dict.fromkeys([name for name, _, _ in files] + meta.not_included))
    marker_size = len(FILE_TRUNCATION_MARKER)
    selected: list[tuple[str, str, str]] = []
    selected_sizes: dict[str, int] = {}
    partial_paths: set[str] = set()

    def render() -> str:
        """Serialize selected excerpts and their matching path coverage headers."""
        parts = []
        for name, file_header, body in selected:
            size = selected_sizes[name]
            excerpt = body[:size]
            if size < len(body):
                excerpt += FILE_TRUNCATION_MARKER
            parts.append(file_header + excerpt)
        return _view_header(
            base,
            [name for name, _, _ in selected],
            all_fetched,
            partial_file=bool(partial_paths),
        ) + "".join(parts)

    for name, file_header, body in sorted(files, key=lambda item: view_priority(item[0])):
        minimum, _ = view_limits(name)
        size = min(len(body), minimum)
        selected.append((name, file_header, body))
        selected_sizes[name] = size
        if size < len(body):
            partial_paths.add(name)
        if len(render()) > budget:
            selected.pop()
            selected_sizes.pop(name)
            partial_paths.discard(name)
            available = budget - len(render()) - len(file_header) - marker_size
            if available <= 0:
                continue
            selected.append((name, file_header, body))
            selected_sizes[name] = min(len(body), available)
            if selected_sizes[name] < len(body):
                partial_paths.add(name)
            if len(render()) > budget:
                selected.pop()
                selected_sizes.pop(name)
                partial_paths.discard(name)
        if len(selected) == 2 and len(render()) > budget:
            dropped_name, _, _ = selected.pop()
            selected_sizes.pop(dropped_name)
            partial_paths.discard(dropped_name)
            if selected:
                kept_name, _, kept_body = selected[0]
                selected_sizes[kept_name] = min(
                    len(kept_body), budget - len(render()) + selected_sizes[kept_name] + marker_size
                )
                if selected_sizes[kept_name] < len(kept_body):
                    partial_paths.add(kept_name)

    def expand(name: str, limit: int) -> None:
        """Spend spare characters on the largest excerpt that still fits the budget."""
        body = next(text for path, _, text in selected if path == name)
        old_size = selected_sizes[name]
        low, high, best = old_size + 1, min(len(body), limit), old_size
        while low <= high:
            candidate = (low + high) // 2
            selected_sizes[name] = candidate
            if candidate == len(body):
                partial_paths.discard(name)
            else:
                partial_paths.add(name)
            if len(render()) <= budget:
                best = candidate
                low = candidate + 1
            else:
                high = candidate - 1
        selected_sizes[name] = best
        if best < len(body):
            partial_paths.add(name)
        else:
            partial_paths.discard(name)

    for name, _, _ in selected:
        expand(name, view_limits(name)[1])
    for name, _, _ in selected:
        expand(name, 10**12)

    if len(selected) == 1:
        name, _, body = selected[0]
        current_size = selected_sizes[name]
        next_size = min(len(body), current_size + budget - len(render()) + marker_size)
        selected_sizes[name] = next_size
        if next_size == len(body):
            partial_paths.discard(name)
        else:
            partial_paths.add(name)

    content = render()
    if len(content) < budget:
        for name, _, body in selected:
            old_size = selected_sizes[name]
            extra = min(len(body) - old_size, budget - len(content))
            if extra <= 0:
                continue
            selected_sizes[name] = old_size + extra
            if old_size + extra == len(body):
                partial_paths.discard(name)
            candidate = render()
            if len(candidate) <= budget:
                content = candidate
            else:
                selected_sizes[name] = old_size
                partial_paths.add(name)
                content = render()
    return content[:budget].rstrip()


def source_line_count(dump: str) -> int:
    """Count nonblank source lines, excluding documentation and missing-content placeholders."""
    _, files = _file_blocks(dump)
    if not files:
        return sum(bool(line.strip()) for line in dump.splitlines())
    total = 0
    for name, _, body in files:
        if UNFETCHED_CONTENT_MARKER in body or name.lower().endswith(DOCUMENTATION_EXTENSIONS):
            continue
        if name.lower().endswith(SOURCE_EXTENSIONS):
            total += sum(bool(line.strip()) for line in body.splitlines())
    return total
