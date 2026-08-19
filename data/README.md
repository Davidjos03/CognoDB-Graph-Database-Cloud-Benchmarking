# Dataset

## Source

[SNAP wiki-Vote](https://snap.stanford.edu/data/wiki-Vote.html) — the Wikipedia
who-votes-on-whose-adminship election network, downloaded from
`https://snap.stanford.edu/data/wiki-Vote.txt.gz`.

| Property | Value |
|---|---|
| Nodes | 7,115 |
| Directed relationships | 103,689 |
| Raw archive | ~1 MB gzipped |

It clears the assignment's 100,000 relationship minimum while staying small
enough to fit the smallest free tier in the comparison, so no platform is
penalised by data volume.

## Graph model (identical on every platform)

```
(:User {node_id, group_id})-[:VOTED_FOR]->(:User)
```

- `node_id` — the original SNAP node id, kept unchanged so any row can be
  traced back to the source file.
- `group_id` — `node_id % 50`, a derived low-cardinality property that gives
  the filtered lookup and the aggregation workload something indexable without
  inventing data that is not in the source.

## Files

`gdbbench download-data` produces:

| File | Committed | Contents |
|---|---|---|
| `data/raw/wiki-Vote.txt.gz` | no | cached source archive |
| `data/nodes.csv` | no | `node_id,group_id`, ascending by `node_id` |
| `data/edges.csv` | no | `source,target`, sorted |
| `data/dataset.json` | yes | provenance: source URL, archive SHA-256, counts |

The CSV files are regenerated rather than committed; `dataset.json` is
committed so a published result can be tied to the exact input that produced
it. Parsing is deterministic — comment lines are skipped, duplicate edges are
dropped and counted, and both files are written in sorted order — so the same
archive always yields byte-identical CSVs.
