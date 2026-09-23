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

The generators that have landed so far: `online_softmax.py` (L00),
`gemm_tiling.py` (L01), `kv_cache.py` (L05), `continuous_batching.py` (L10),
`ring_allreduce.py` (L13), `flash_attention.py` (L06).

Several of them double as the fixture their view component is validated
against, which is why they carry a block on top of the usual trace: `kv_cache.py`
has `ledger` for the memory-ledger view (`labs/assets/engine/views/ledger.js`,
ticket #45), `continuous_batching.py` has `gantt` for the gantt view (ticket
#46), `ring_allreduce.py` has the ring contract (ticket #47), and
`flash_attention.py` drives the memory-hierarchy stage
(`labs/assets/engine/views/tiling-stage.js`, ticket #20). Writing that fixture
is how a view ticket breaks the circular dependency — the lab that would consume
it is blocked by the view, but trace generation does not depend on the view.

The other half of that coin is what a shared component must NOT do. Because
`tiling-stage.js` is driven by four different labs, every rule its lint carries
is gated on the trace declaring the corresponding field (`step.occ` → the
occupancy account, `step.io` → the counter, `meta.sram`/`meta.io`/
`meta.exposure` → the summaries). A rule that fired unconditionally would turn
L01's three-layer GEMM replay red for not having FlashAttention's fields. The
gates are derived from a field one level DOWN from what they gate, so the
sabotage that deletes a `meta` block can still be caught.

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

## The `occ` / `io` step fields and the `sram` / `io` / `exposure` blocks (added by ticket #20)

Three more step fields and three trace-level summaries, all optional and all
consumed by the memory-hierarchy stage and L06's panels. Like `ledger` below,
each is inert on a trace that does not declare it — which is what lets four
different labs share one view component.

- **`step.occ`** — the occupancy account: `{elements, parts: [{name, label,
  shape, elements, covers?}]}`. `parts` must name **exactly** the tensors the
  pure reconstruction says are resident in that layer at that step, so the lint
  re-derives the resident set from `state` rather than trusting the account to
  describe itself. `covers` is how a merged buffer is spelled: one allocation
  named for two tensors (L06's `S`/`P`, where `P = exp(S - m_new)` is computed in
  place) lists the tensors on it, and the rule is that those are exactly the ones
  currently resident — so an account cannot claim a shared buffer holds both
  members before the second exists, and the component needs to know nothing about
  which lab merges what.
- **`step.io`** — `{step, cum}`: HBM elements accessed by this step, and the
  running total. `cum` must equal the previous `cum` plus this `step`, so the
  per-frame numbers and the total cannot drift apart.
- **`meta.sram`** — the peak of that account, its `peak_step`, and the parts
  broken out. The lint requires `used` to equal the maximum over the steps and
  `dominant` to equal `4·B_c·d`, which is the term the tutorial's `B_c = ⌈M/4d⌉`
  bound constrains.
- **`meta.io`** — the measured totals for both implementations, the formulas that
  PREDICT them, and the curves. The lint's requirement is that measured equals
  predicted, so the page's two statements about the same quantity cannot
  disagree; the prediction is written separately from the counter precisely so
  that assertion has two sides.
- **`meta.exposure`** — the headline claim as a measurement: which tensors the
  forbidden set contains, which of them each implementation actually touched, and
  how many times. `flash_forbidden_accesses` must be 0 and
  `standard_forbidden_accesses` must NOT be — the second half is the control that
  makes the first mean anything, and the lint enforces both.

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
## The gantt contract

The gantt view (`labs/assets/engine/views/gantt.js`) is the one view whose
contract does **not** live in `trace-model.js`. Rows, tracks and time-steps are
a view's idea, not the engine's, and no other view or lab reads them — so the
rules sit beside the view, and the copy of them that the trace generator needs
sits beside the generator. `continuous_batching.py` (L10) introduces the three
fields:

- **`gantt.rows`** — `[{id, label, sub?, track?}]`. One entry per resource: a
  request in L10, a pipeline stage in L16, the compute and communication lanes
  in L17. `track` groups rows into labelled sub-lanes, which is how L17 draws
  its compute-above-communication split without a second chart.
- **`gantt.tracks`** — `[{id, label}]`, optional. Only needed when rows name a
  `track`.
- **`gantt.kinds`** (+ optional `gantt.kindsLabel`) — the activity vocabulary.
  Every bar's `kind` must be in this list or the lint rejects the trace: a kind
  with no entry is a bar with no style, which renders as a grey block that looks
  deliberate.
- **`step.bars`** — `[{row, kind, label?}]`, what each resource was doing during
  that step. Not derivable from `state`: a scheduler that computed three tokens
  and one that computed none can leave identical queue lengths behind. The
  optional `label` is what L16 (micro-batch id) and L17 (bucket number) will
  print inside the block.

Because these five rules are a second copy of the same contract, **both copies
carry a sabotage control group and the acceptance harness runs the same one
through both** (`scripts/verify-gantt.py`, the "契约 lint 的两份实现" section).
A rule that exists on one side only is a rule tested on neither.

## Environment

Verified locally: `python3.12`, `numpy 1.26`, `torch 2.2.0` (CPU).
