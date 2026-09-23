# Trace generators

One Python script per lab. Each one runs a real reference implementation and
dumps a frame per algorithmic step — tensor shapes, values, and the formula
symbol table for that step — as JSON.

This directory is **not** staged into `public/labs/` (see
`scripts/build-labs.mjs`); the JSON is inlined into the corresponding page in
`labs/pages/` at build time instead.

## Rules (from the engine's data contract)

- Every number shown on a lab page must come from a run of one of these
  scripts. Hand-written demo data is not allowed — that is the whole point of
  the separate-player architecture.
- Each script must carry an assertion against a PyTorch reference
  implementation. Those run in CI (`.github/workflows/trace-checks.yml`,
  CPU-only torch, scoped to this directory) so a page can never drift away from
  the algorithm it claims to replay.
- The shared example model across all labs is
  `N=8, d_model=64, H=4, d_head=16, d_ff=176, vocab=32`.

`online_softmax.py` (L00) is the first one to land.
`kv_cache.py` (L05) is the second; it is also the fixture the memory-ledger
view component (`labs/assets/engine/views/ledger.js`, ticket #45) is validated
against, which is why it carries a `ledger` block on top of the usual trace.

## The `ledger` block (added by ticket #45)

A trace may declare how its memory is accounted for. Two pieces:

```jsonc
"ledger": {
  "unit": "B",
  "segments": [
    { "id": "params", "label": "参数", "color": "#5b8def", "formula": "…LaTeX…" }
  ]
},
"steps": [
  { "ledger": { "bytes": { "params": 210176, … },   // one entry per segment
                "config": { "phase": …, "n_q": …, "layers": …, "seq_len": …,
                            "kv_rows_total": … } } }
]
```

`bytes` is what the component draws. `config` is what lets a reader *recompute*
those bytes independently — without it, any cross-check would be the same
formula evaluated twice and would prove nothing. Both are required by the lint,
and the lint additionally requires `kv_rows_total == layers × seq_len`, because
the cache holds exactly one row per layer per position.

The whole block is optional: a trace without it (like L00's) lints clean, and
the rules stay inert. See `lint_ledger()` here and its JS port in
`labs/assets/engine/views/ledger.js` — **the two rule sets must be changed
together, and each needs its own sabotage case**, or the side that did not get
one is untested.

## Environment

Verified locally: `python3.12`, `numpy 1.26`, `torch 2.2.0` (CPU).
