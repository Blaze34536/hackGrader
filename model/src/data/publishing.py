"""Publish candidate JSONL files only after proving they fit the trainer window.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import json
import os
import tempfile
from typing import Any

from data.config import DATA_SPLITS
from data.validation import DEFAULT_MAX_LEN

from .rows import is_snippet

_TOKENIZER: Any | None = None


def candidate_tokenizer():
    """Load and configure the tokenizer used by training before building rows."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "[!] Cannot certify candidate token lengths; install model requirements first."
        ) from None
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get(
            "HACKGRADER_BASE_MODEL",
            "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
        )
    )
    from learning.inputs import configure_tokenizer

    _TOKENIZER = configure_tokenizer(tokenizer)
    return _TOKENIZER


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """Create a new JSON Lines candidate file without overwriting existing data."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)


def _write_combined_content(path: str, partitions) -> int:
    """Write tagged rows to an already-created file and remove partial output on error."""
    assigned = list(zip(DATA_SPLITS, partitions, strict=True))
    try:
        with open(path, "w", encoding="utf-8") as handle:
            for split_name, rows in assigned:
                for row in rows:
                    record = {**row, "split": split_name}
                    handle.write(json.dumps(record) + "\n")
    except BaseException:
        if os.path.exists(path):
            os.remove(path)
        raise
    return sum(len(rows) for _, rows in assigned)


def write_combined_jsonl(
    path: str,
    partitions: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    *,
    replace: bool = False,
) -> int:
    """Write three tagged partitions, atomically replacing the target only when asked."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not replace:
        with open(path, "x", encoding="utf-8"):
            pass
        try:
            return _write_combined_content(path, partitions)
        except BaseException:
            if os.path.exists(path):
                os.remove(path)
            raise

    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        total_rows = _write_combined_content(temporary_path, partitions)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return total_rows


def report_fit(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> None:
    """Reject candidate rows whose rendered chat exceeds the configured model window."""
    tokenizer = candidate_tokenizer()
    longest = 0
    over_repository = over_snippet = 0
    for row in train_rows + validation_rows:
        rendered = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        longest = max(longest, length)
        if length > DEFAULT_MAX_LEN:
            if is_snippet(row):
                over_snippet += 1
            else:
                over_repository += 1
    print(f"[i] longest rendered row {longest} token(s) against {DEFAULT_MAX_LEN}")
    if over_repository or over_snippet:
        raise SystemExit(
            f"[!] {over_repository} repository and {over_snippet} snippet row(s) exceed "
            f"{DEFAULT_MAX_LEN} tokens; refusing to publish candidates."
        )
