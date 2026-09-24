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

`gemm_tiling.py` (L01) follows the same shape for `(B_M, B_N, B_K)`: 27
configurations — the full product of {2, 4, 8} per parameter, every one of which
divides M=N=K=8 — plus `gemm-tiling.manifest.json`. The sliders are therefore
three independent controls rather than three positions of one knob, and the
filter that would skip a non-dividing tile is in the generator's `grid()`.

`kv_cache.py` (L05) follows the same shape for `(N, L, precision)` — 18
configurations, the full product of {8, 128, 256} × {2, 4} × {fp16, fp8, int8},
plus `kv-cache.manifest.json`. Two things distinguish it from the two above and
both are deliberate:

- **The default keeps the bare file name `kv-cache.json`** rather than encoding
  its parameters, because ticket #45's acceptance page and harness pin that
  filename. Nothing parses an id: the page matches a slider position to a trace
  by reading the trace's own `meta.config`, which is a stronger check than
  string parsing and needs no naming convention at all — the manifest's ids are
  for humans.
- **The three sliders move geometry, not just counts.** `N` is the sequence
  length the replay *ends* at (prompt N−4, then 4 generated tokens), so the
  replay is 20 steps in every configuration while the context the prefill
  swallows and the decode attends over spans 16× — enough for the ledger's KV
  share to visibly climb (7% → 13% → 24%). `L` and the precision move the
  ledger's byte counts in the other two directions — and fp8 and int8 price
  identically at this level (one byte per element), which the generator asserts
  as a property and prints rather than hiding behind a third position that
  looks like it should differ.
- **The context grid stops at 128, and `state` is published to 6 significant
  figures, both for page size.** L05 is the first lab whose `state` holds a
  large array of high-entropy floats: the residual stream is `[N, d_model]` and
  its entries come out of a standard-normal weight draw, so at N=256 it is
  ~330 KB of text *per prefill step* — and random mantissas do not compress, so
  it costs full price over the wire too. A 32× context span put 10.3 MB of
  traces into one inlined page, 4.4× the largest lab that had shipped and 8×
  the largest page the site had. Capping the grid at 128 and rounding `state` to
  `STATE_SIG_FIGS = 6` brings the set to 3.9 MB and the page to 2.4 MB raw —
  level with L01's 2.33 MB. (Its gzip is larger, 0.85 MB against L01's 0.20 MB:
  that is the honest cost of the subject matter, since L01's integers compress
  and these do not.) Nothing asserted downstream reads those digits — the torch
  cross-check, the ledger accounting and the attention-cost comparison all run
  on the float64 arrays before the projection. Contexts past 128 are reachable
  through the page's *predicted* panel, which is labelled as a prediction and
  drives the same closed form out to 4096.

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

The Roofline is the one view whose contract is a *trace-level block* rather than
step fields, and it follows the gantt's precedent instead: its rules live beside
the view (`views/roofline.js`) and beside the generator, both carry a sabotage
control group, and the acceptance harness runs the same table through both. It
is reached from any lab with a `meta.roofline` block; L01 is the only one so far.

L05 added two more view fixtures (`step.attn_cost` / `meta.attn_cost` for
`views/attn-shape.js`, `meta.phase_roofline` for `views/phase-roofline.js`) and
one trace-level block (`meta.kv_crossover`) — see their own sections below. The
phase Roofline is worth one note here, because it is the case where a shared
component was **not** reused as a lint target: `views/phase-roofline.js` renders
through `roofline.makeView` unchanged (the drawing is shared in full), but L05
declares a different key rather than `meta.roofline`. The reason is the rule
stated just below — `roofline.js`'s lint is **not** field-gated: it hard-requires
L01's three roles (`naive`/`tiled`/`doc`), `2MNK` FLOPs and a `meta.traffic`
block, and would report every L05 trace as broken. Rewriting another ticket's
contract file was out of scope, so the renderer is shared and the rules are
L05's own.

The other half of that coin is what a shared component must NOT do. Because
`tiling-stage.js` is driven by four different labs, every rule its lint carries
is gated on the trace declaring the corresponding field (`step.occ` → the
occupancy account, `step.io` → the counter, `meta.sram`/`meta.io`/
`meta.exposure` → the summaries). A rule that fired unconditionally would turn
L01's three-layer GEMM replay red for not having FlashAttention's fields. The
gates are derived from a field one level DOWN from what they gate, so the
sabotage that deletes a `meta` block can still be caught.

L04 (#18) is the third lab to hit the same rule from the consumer side, and it
is worth recording because it declined the shared thing twice. It renders its
parameter account through `views/ledger.js` (the bar, the byte formatting, the
table, the hover wiring — all reused, ticket #45's deliverable), but it does
**not** declare `trace.ledger`: that block's lint requires `phase` / `n_q` /
`layers` / `seq_len` / `kv_rows_total` on every step and requires
`kv_rows_total == layers × seq_len`, because for L05 that invariant *is* the KV
cache. L04 has no cache, so satisfying the rule would mean publishing a
`kv_rows_total` that names nothing — a number on a page that does not stand for
anything, which is the thing the whole contract exists to prevent. L04 declares
`step.params` / `meta.params` instead, with its own rules and its own two ports
(`views/*.js` and `decoder_block.py`), exactly as L05 did for `roofline.js`. The
ledger's `check()` still runs on L04 and its three component-level verdicts are
still reported; only the embedded lint verdict is ignored, and the page says so
in the panel rather than quietly dropping it.

**The `params …` sabotage cases are the one group in this repo that has a single
port, and that is stated rather than papered over.** The two-port rule exists
because a rule whose verdict a lab author cannot get in the browser is a rule
they will edit blind — but that argument needs *a browser-side consumer to
disagree with*, and `meta.params` has none: the component that draws it is the
ledger's generic renderer, which is `lab`-agnostic by design and cannot carry
this block's rules. So the rules live in `decoder_block.py` alone, its sabotage
cases keep their `params …` names with no JS counterpart, and
`scripts/verify-l04.py` counts the generator's own verdict for them
(`sabotages[name].python`) instead of reporting them as "caught by nobody". The
harness prints how many such cases there are, so the asymmetry is visible in the
run rather than buried in this paragraph. If a later ticket gives the parameter
account a view of its own, moving these rules beside it is the right change.

What L04 *did* add as a shared arrangement is the **one-level-down gate as a
positive rule**. `meta.decoder` exists only to be deleted: it names which steps
carry `step.ln`, `step.residual` and `step.swiglu`, and the lint — on both
sides — requires those lists to equal the steps that actually carry the fields.
That is what makes it safe for four new view components to run rules
unconditionally on `step.ln` and friends: a trace without them lints clean
because the gate says "there are none", and a trace whose gate has been edited
away is a gap rather than a silent pass. The gate is checked before the rules it
gates, and the sabotage table has three cases for it (`decoder map removed
while steps keep their fields`, `decoder map omits a step that carries step.ln`,
`decoder map claims a step that has no such field`).

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

## The `roofline` block (added by ticket #15)

L01's parameter sliders needed a picture to move, and this is it. One trace-level
block, optional, consumed by `labs/assets/engine/views/roofline.js`:

```jsonc
"meta": {
  "traffic":  { "naive_reads": 1024, "tiled_reads": 256, "tiled_ai": 1.0,
                "tiled_ai_closed": 1.0, "load_steps": 8, "smem_tile": 32, … },
  "roofline": { "peak_flops": 1.57e13, "bandwidth": 9e11, "ridge": 17.44,
                "x_label": "算术强度 (FLOP/Byte)",
                "y_label": "可达算力上界 (TFLOP/s)",
                "y_divisor": 1e12, "y_unit": "TFLOP/s",
                "points": [ { "id": "naive"|"tiled"|"doc", "ai": …,
                              "elements": …, "bytes": …, "flops": …,
                              "ceiling": …, "bound": "bandwidth"|"compute" } ] }
}
```

What the lint requires, and why each rule has two sides:

- **`ai` is `flops / bytes`, re-derived per point.** The page's parameter story
  is two statements about one quantity — "读取 1024 → 256 个元素" and "算术强度
  0.25 → 2.0" — so the point carries its own FLOP and byte counts and the rule
  recomputes the quotient from them. A page that quoted both would otherwise be
  making two independent claims about the same kernel.
- **`ceiling` is `min(peak_flops, bandwidth × ai)`, and `bound` names the roof
  that produced it.** It is a CEILING, not a measurement: nothing in this lab
  times a kernel, and the field is named and displayed so a reader is not misled
  into reading a bound as an achieved rate.
- **`naive` and `tiled` are THIS configuration's problem**, so their element
  counts must equal `meta.traffic`'s counts and their FLOPs must equal 2MNK.
  `doc` is the tutorial's 128-cubed example and is deliberately NOT tied to this
  trace's sizes — tying it would be the wrong rule, not a stricter one.
- **`y_divisor` and `y_unit` must agree with `y_label`.** A chart whose label
  reads TFLOP/s while its ticks are FLOP/s is wrong by exactly the factor
  between them, and no reader can see it.
- **The naive point's `ai` must be `2K/8K`** — a function of the problem size
  alone. This is the cross-configuration control: the tiled point is the one the
  sliders move, and its movement only means something because this one holds
  still. `gemm_tiling.py` asserts the same thing across the whole grid, at the
  set level rather than the trace level.

`B_K` is the interesting case and the block records the asymmetry honestly.
Under this traffic model each output block walks the whole K range, so B_K
cancels out of the element count — which is exactly why the tutorial's §4.3
closed form, `B_M·B_N/(2(B_M+B_N))`, has no B_K in it. So `tiled_reads` and
`tiled_ai` are invariant across B_K, while `load_steps` and `smem_tile` scale
with it; the generator asserts both directions, and the page says which is
which rather than leaving a reader to conclude the middle slider is broken.

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

## The `attn_cost` step field (added with L05's lab page, #19)

L05's central contrast is between two ways to compute the same attention:
without a cache, every step re-runs the whole prefix (`[t,d] × [t,d]` per head);
with one, only the new queries run (`[n_q,d] × [t,d]`). `views/attn-shape.js`
draws the two side by side, and this is the field it draws:

```jsonc
"steps": [
  { "attn_cost": {
      "t": 125, "n_q": 1, "H": 4, "d_head": 16, "layers": 2,
      "no_cache":   { "queries": 125, "q_shape": [125, 64], "k_shape": [125, 64],
                      "score_shape": [4, 125, 125], "score_elems": 62500,
                      "kv_read_elems": 0, "kv_write_elems": 0,
                      "attn_flops": …, "proj_flops": … },
      "with_cache": { "queries": 1,   "q_shape": [1, 64],   "k_shape": [125, 64],
                      "score_shape": [4, 1, 125], "score_elems": 500,
                      "kv_read_elems": …, "kv_write_elems": …,
                      "attn_flops": …, "proj_flops": … } } }
],
"meta": { "attn_cost": { "totals": { "no_cache": {…}, "with_cache": {…} },
                         "prefill_identical": true } }
```

**Both modes are MEASURED, not asserted.** The generator already runs an
uncached full-prefix forward at every step — that is the torch cross-check — and
`attn_cost_step()` reads each mode's score matrix and Q/K row counts off the
**real arrays of both computations**, then asserts them against the shape
formula in the same block. Publishing the contrast as a formula alone would make
it a claim about what the code does rather than a reading of what it did, and
those differ in exactly the case that matters, because `n_q` is the prompt
length at prefill and 1 at decode.

Three things the lint requires, each with a control:

- **`score_elems` is the product of its own `score_shape`**, and `attn_flops` is
  `4·H·q·t·d_h`, and `proj_flops` is `6·q·d²·L` — so "the shape got wider" and
  "the work got bigger" cannot become two independent claims on one page.
- **At prefill the two modes must be IDENTICAL** (in the computation — shapes
  and FLOPs — not in the cache traffic, which genuinely differs because one path
  writes the prompt's rows into a cache and the other has no cache at all).
  With an empty cache there is nothing to recompute, and that coincidence is the
  control the decode divergence is read against. A lab whose two columns
  differed *everywhere* would be showing a difference it never isolated.
- **Some step must diverge**, or the control above is the only thing the
  comparison ever shows.

The uncached mode charges `kv_read_elems = kv_write_elems = 0`: it has no cache,
so its cost lands in `proj_flops` instead, and saying "0" is what keeps that
visible rather than spread across both columns.

The whole field is optional, and its rules live beside the view
(`views/attn-shape.js`) — same two-port arrangement as the ledger's, with the
same sabotage table run through both by `scripts/verify-l05.py`. The generator's
sabotage names carry the owning view as a prefix (`attn …`, `roofline …`,
`crossover …`), which is how the harness attributes each case to the port that
must catch it rather than only requiring *some* port to.

## The `phase_roofline` block (added with L05's lab page, #19)

The tutorial's own claim — Prefill is compute-bound, Decode is memory-bound —
placed on a Roofline by computing it rather than quoting it
(`1.1-LLM推理基础.md` §4):

```jsonc
"meta": {
  "phase_roofline": {
    "peak_flops": 9.9e14, "bandwidth": 3.35e12, "ridge": 295.5,
    "y_divisor": 1e12, "y_unit": "TFLOP/s", "model": "NVIDIA H100 SXM…",
    "traffic_note": "按权重读取一次 + KV 全量读 + 本步新行写回来计价…",
    "points": [
      { "id": "prefill"|"decode"|"doc", "label": …, "ai": …, "flops": …,
        "bytes": …, "parts": { "proj_flops": …, "ffn_flops": …, "attn_flops": …,
                               "head_flops": …, "weights_bytes": …,
                               "kv_read_bytes": …, "kv_write_bytes": … },
        "ceiling": …, "bound": "bandwidth"|"compute",
        "config": { "n_q": …, "t": …, "prompt": …, "layers": …, "dtype": … } }
    ] } }
```

One trace-level block, like the Roofline's, and it is reached from
`meta.phase_roofline` rather than `meta.roofline` for the field-gating reason
above. Three points, and the comparison between them is the lesson:

- **`prefill`** — this configuration's prompt in one pass. Its arithmetic
  intensity is `prompt_len`-fold the decode one, because the weight bytes are
  paid ONCE for the whole prompt. This is the point the slider moves.
- **`decode`** — one token against the same weights. It sits near 1 FLOP/Byte
  and barely moves with the context: the control that makes the prefill movement
  readable as a comparison.
- **`doc`** — a 7B-class model at a 512-token prompt, i.e. the tutorial's own
  claim computed rather than quoted. It is deliberately NOT tied to this trace's
  config: it must be the same dot on every configuration, and the harness
  asserts that stillness.

The traffic model is a choice and the block publishes it in `traffic_note`
rather than leaving it implied: weights read once per pass, the KV cache read in
full, this step's new rows written back, and **the attention score matrix not
counted as HBM traffic at all** — it is produced and consumed inside the
attention kernel's SRAM, which is the assumption FlashAttention is built on
(L06) and the reason this lab can put attention on a Roofline.

What the lint requires is that the chart be internally coherent: `ai` is the
point's own FLOPs over its own bytes; `ceiling` is `min(peak, bandwidth × ai)`
and `bound` names the roof that produced it; the totals equal the sums of their
own `parts` (so one summary number cannot hide a dropped term); `y_divisor` and
`y_unit` agree with `y_label`; and the two live points describe *this*
configuration while `doc` describes the reference model. The honest answer the
block records is that in this toy model **both** live points land on the
bandwidth roof — the compute roof is reached only by the reference model — and
the page says so rather than letting a reader conclude the chart is wrong.

## The `kv_crossover` block (added with L05's lab page, #19)

```jsonc
"meta": { "kv_crossover": { "params_bytes": 210176, "per_token_bytes": 512,
                            "context": 410.5 } }
```

The context length at which the KV cache outweighs the weights, published so the
page can state it without computing it. One rule, and it is the one with teeth:
the crossover must equal `params_bytes / per_token_bytes`, **and both of those
must equal the figures the ledger block already publishes** — `params_bytes` to
the last step's `params` segment, `per_token_bytes` to `2·L·H_kv·d_h·b`. Without
the second half the page could print a crossover computed from a different model
than the bar it is drawn beside.

The block exists because at this model's size the crossover is at context ≈ 410
for 2 layers at fp16 — far beyond the 256 the sliders can replay — so the ledger
bar never crosses over on screen. A page that left the number out would leave a
reader with a picture that never shows the thing the tutorial is about. L05's
page therefore also carries a *predicted*-configuration ledger, driven by the
same closed form over a wider range; it is labelled as a prediction everywhere
it appears, and the ledger's self-check panel is what makes it legitimate (it
proves the page's cost model equals the one the trace was counted with).

The JS port of these rules lives in `views/phase-roofline.js` even though the
block is ledger subject matter: `labs/assets/engine/` is frozen for this ticket
and `views/ledger.js` is one of the files it freezes. Recorded here so whoever
unfreezes the engine can move the four `crossover …` rules next to the ledger's
own port, where the repo's convention puts them.

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
