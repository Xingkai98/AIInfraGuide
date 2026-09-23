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

## Trace sets (labs with parameter controls)

A generator may emit a *set*: one trace per slider position, plus a manifest
naming them. `online_softmax.py` does this for `(N, Bc, T)` — 18 configurations,
plus `online-softmax.manifest.json`:

```jsonc
{ "set": "online-softmax", "default": "online-softmax-N8-B4-T1",
  "params": { "N": [4, 8], "Bc": [1, 2, 4], "T": [0.5, 1, 2] },
  "traces": ["online-softmax-N4-B1-T0p5", …],
  "labels": { "online-softmax-N4-B1-T0p5": "N=4 · Bc=1 · T=0.5", … } }
```

The manifest is written by the generator, not by hand, so the set is defined in
exactly one place: the script that produces it. The page inlines the whole set
with a `<!-- traces:NAME -->` marker (see the trace-set step in
`scripts/build-labs.mjs`), which fails the build if the manifest and the trace
files have drifted apart.

The sliders therefore select among *generated* configurations — moving one
loads a different trace, it does not recompute anything in the browser. That is
what keeps "every number on the page is the output of one of these scripts"
literally true for a parameterised lab: `T` is applied as `x/T` here, not on the
page, and a configuration with no trace behind it is unreachable by construction.

## Environment

Verified locally: `python3.12`, `numpy 1.26`, `torch 2.2.0` (CPU).
