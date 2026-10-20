"""Group scored rows that share a source identity or prompt."""

from collections import defaultdict
from typing import Any


def lineage_clusters(predictions: list[dict[str, Any]]) -> list[str]:
    """Group examples sharing a repository URL or source-group lineage key."""
    parents = list(range(len(predictions)))
    owners: dict[str, int] = {}

    def root(index: int) -> int:
        """Find a cluster representative while compressing parent links."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(first: int, second: int) -> None:
        """Merge examples sharing one lineage identity."""
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for index, prediction in enumerate(predictions):
        identities: set[str] = set()
        source_url = prediction.get("source_url")
        if isinstance(source_url, str) and source_url.strip():
            identities.add(f"lineage:{source_url.strip().casefold().rstrip('/')}")
        groups = prediction.get("source_groups", [])
        if isinstance(groups, list):
            identities.update(
                f"lineage:{group.strip().casefold().rstrip('/')}"
                for group in groups
                if isinstance(group, str) and group.strip()
            )
        prompt_digest = prediction.get("prompt_sha256")
        if isinstance(prompt_digest, str) and prompt_digest.strip():
            identities.add(f"prompt:{prompt_digest.strip().casefold()}")
        repository_id = prediction.get("repository_id")
        if isinstance(repository_id, str) and repository_id.strip():
            normalized = repository_id.strip().casefold().rstrip("/")
            identity = (
                f"prompt:{normalized}"
                if isinstance(prompt_digest, str)
                and normalized == prompt_digest.strip().casefold()
                else f"lineage:{normalized}"
            )
            identities.add(identity)
        if not identities:
            identities.add(f"row:{index}")
        for identity in identities:
            join(index, owners.setdefault(identity, index))
    return [str(root(index)) for index in range(len(predictions))]


def cluster_values(
    values: list[float], cluster_ids: list[str] | None
) -> list[list[int]]:
    """Return row indices grouped by source identity or independent row."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(len(values)):
        identity = cluster_ids[index] if cluster_ids is not None else f"row:{index}"
        groups[identity].append(index)
    return list(groups.values())


def group_counts(
    trials: list[bool],
    successes: list[bool],
    cluster_ids: list[str] | None,
) -> dict[str, tuple[int, int]] | None:
    """Count metric trials and successes within each source cluster."""
    if len(trials) != len(successes):
        raise ValueError("metric trial and success arrays must have equal lengths")
    if cluster_ids is None:
        return None
    if len(cluster_ids) != len(trials):
        raise ValueError("cluster_ids and metric trial arrays must have equal lengths")
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for cluster, trial, success in zip(cluster_ids, trials, successes, strict=True):
        if success and not trial:
            raise ValueError("a metric success must also be counted as a trial")
        if trial:
            counts[cluster][0] += int(success)
            counts[cluster][1] += 1
    return {cluster: (values[0], values[1]) for cluster, values in counts.items()}
