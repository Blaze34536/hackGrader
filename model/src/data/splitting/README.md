# Source-disjoint splitting

The splitter assigns whole connected source components to training, validation, or independent test. Keeping a component intact prevents a repository, duplicate prompt, or derived example from appearing on both sides of an evaluation boundary.

## How rows are grouped

Rows are connected when they share a canonical source URL, rendered user prompt, or any `source_groups` identifier. Connections are transitive: if row A shares lineage with B and B shares lineage with C, all three belong to one component. The allocator can reserve or skip a component; it cannot break one apart to satisfy a row quota.

## Quotas and allocation

The public entry points are `split_by_source()` for training plus validation, and `split_by_source_with_test()` for training, validation, and independent test. Callers can set source-family, repository, row, class, boundary, and score-band minimums. The builder's normal floors come from its data and learning configuration.

The fresh-repository builder option adds test-only minimums for distinct real-repository prompts and both score classes. The `--fresh-repository-test` builder flag requests 50 distinct real test prompts and 10 prompts in each class by default. These are allocation requirements, not bypasses for any other configured gate. Grouping and allocation details are in [`components.py`](components.py), [`quotas.py`](quotas.py), [`capacity.py`](capacity.py), and [`allocator.py`](allocator.py).

The allocator fills holdouts while checking that enough unassigned source components and examples remain for every outstanding requirement. If total source-disjoint supply is inadequate, it reports the unmet category and stops. It never weakens quotas or splits a source component to force a pass. The final audit independently recounts class, score-band, lineage, repository, and target-share requirements.

## Module map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports the band, component, and public split functions. |
| `bands.py` | Defines score classes, bands, and boundary ranges. |
| `components.py` | Builds connected lineage groups and measures component supply. |
| `quotas.py` | Expands and validates per-partition minimums. |
| `capacity.py` | Reports source and score shortfalls before allocation. |
| `progress.py` | Tracks partition coverage and remaining deficits. |
| `priority.py` | Ranks components by how well they address current deficits. |
| `allocator.py` | Assigns whole components without making later quotas infeasible. |
| `audit.py` | Rechecks finished partitions against their full requirements. |
| `core.py` | Exposes split planning and public split functions. |

See the [candidate-data reference](../README.md) for the row schema and the [builder reference](../builder/README.md) for CLI options.
