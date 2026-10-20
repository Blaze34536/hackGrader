"""Source-disjoint data splitting.

The public surface is deliberately small: two split functions that return training plus
one or two holdouts, and the lineage grouping primitive other modules reuse.

    from .splitting import split_by_source_with_test

Implementation modules:

``bands``       score bands and the ownership classes a target satisfies
``components``  union-find lineage grouping and per-component profiles
``quotas``      resolves per-partition floors and validates the request
``capacity``    reports every way the pool falls short, before allocating anything
``progress``    tracks one partition's coverage and its outstanding deficits
``priority``    scores how well a component closes the deficits still open
``allocator``   places whole components while keeping the remaining quotas feasible
``audit``       re-measures the finished partitions against their floors
``core``        ties the pipeline together and exposes the split functions

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from .bands import class_labels, score_band
from .components import ComponentProfile, _components
from .core import (
    SplitPlan,
    partition_rows,
    plan_partitions,
    split_by_source,
    split_by_source_with_test,
)
from .quotas import PartitionQuotas

__all__ = [
    "ComponentProfile",
    "PartitionQuotas",
    "SplitPlan",
    "_components",
    "class_labels",
    "partition_rows",
    "plan_partitions",
    "score_band",
    "split_by_source",
    "split_by_source_with_test",
]
