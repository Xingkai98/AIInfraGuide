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

`online_softmax.py` (L00) is the first one to land, `gemm_tiling.py` (L01) the
second.

## Two optional step fields a trace may use

`online_softmax.py` needs neither; `gemm_tiling.py` introduced both. Both live
on the step, so `resolve(trace, i)` stays a pure function of the cursor.

- **`step.spans`** — the sub-region of a resident block the step touches, one
  `[lo, hi)` range per axis (`{"A": [[0, 4], [0, 4]]}`). Lets the memory-hierarchy
  stage outline the 4×4 tile being carried rather than tinting all of A.
- **`step.flows`** — how much data crossed which layer boundary:
  `{from, to, elements, bytes?}`. Not derivable from `reads`/`writes`: an
  accumulator update and a block store both look like a write, and only one of
  them moves anything. The generator's docstring works through the two cases
  that make the derivation wrong.

## Environment

Verified locally: `python3.12`, `numpy 1.26`, `torch 2.2.0` (CPU).
