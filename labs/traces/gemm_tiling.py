#!/usr/bin/env python3
"""Trace generator for L01 · GEMM：索引、分块与访存.

Emits one JSON file per configuration of `(B_M, B_N, B_K)`, plus a manifest
naming the set:

    labs/traces/gemm-tiling-BM<B_M>-BN<B_N>-BK<B_K>.json   one full replay each
    labs/traces/gemm-tiling.manifest.json                  the set the page inlines

M=N=K=8 throughout; only the tile shape varies. This is the lab the parameter
sliders drive, and it is the site's most direct demonstration of *why tiling
helps*: the same problem, the same arithmetic, only the blocking changes — and
the HBM traffic, the arithmetic intensity, and the point on the Roofline move
accordingly. The manifest is written by this script rather than by hand, so a
slider position can never point at a configuration that was never generated
(see the `traces:` marker in `scripts/build-labs.mjs`).

WHY THIS FILE IS NOT A CARBON COPY OF online_softmax.py
------------------------------------------------------
Two things are genuinely different, and both are about what can be *proved*.

1. The strongest correctness claim is exactness, not tolerance. Tiling is
   supposed to change when data moves and not what is computed, so this trace
   asserts that directly:

       C by the naive k-ordered loop
       C by block tiling (B_M=B_N=B_K=4)
       ->  np.array_equal(...) is True, bit for bit

   Both accumulators are plain Python floats — not numpy scalars, not a BLAS
   call — so the k-order really is 0..7 in both and IEEE-754 double arithmetic
   makes the two runs identical rather than merely close. That claim is only
   available because this is a scalar model of the kernel rather than the
   kernel, which is exactly what the replay is for.

2. The tolerance question the ticket raises ("set it from your own dtype; do
   not copy 1e-12, that is a float64 number") is answered by the dtype, and
   this trace is float64 on both sides, so it lands in the same place for the
   same reason: the residual against torch is summation order, because
   torch.matmul goes through a blocked BLAS kernel while this sums k in order.
   1e-12 relative sits three orders above the ~1e-16 that order alone can
   produce here and far below the fourth significant digit the page prints. It
   is re-derived, not inherited — and it is deliberately *not* the headline
   claim, which is the exact equality above. The torch check is kept as the
   independent cross-check: exactness against a second implementation of the
   same loop would prove nothing about the algorithm being right.

Every number quoted in a narration or a formula is computed from M/N/K/B_* here
— the HBM element traffic (2MNK naive against the tiled count), the arithmetic
intensity, every accumulator. Only the problem inputs A and B are literals,
which is the same split online_softmax.py makes for its input vector.

THE ROOFLINE BLOCK, AND WHY THE SLIDERS ARE TRACES RATHER THAN ARITHMETIC
------------------------------------------------------------------------
The ticket's fourth criterion is "改变 B_M/B_N/B_K 后，访存次数与算术强度实时重算，
Roofline 图上的点相应移动". The tempting reading is "recompute in the browser on
every slider move". That would put arithmetic on the page, and this project's
first rule is that every number shown comes from a run of one of these scripts —
so the page would be computing a formula the trace never executed, and the two
could disagree with nothing to catch it.

The reading this file implements is the one L00 established: a parameterised lab
ships a *set*, one trace per slider position, and the slider selects among
already-generated configurations. So `B_M`/`B_N`/`B_K` are applied HERE, in
`traffic()`, and each configuration publishes its own `meta.roofline` block:

    "roofline": { "peak_flops": …, "bandwidth": …, "ridge": …,
                  "points": [ {"id": "naive"|"tiled"|"doc", "ai": …, "bytes": …,
                               "ceiling": …, "bound": …} ] }

Two things about that block are load-bearing rather than decorative:

  * `ai` is not an independent number. It is `2MNK / bytes`, and the lint
    re-derives it from the byte counts rather than trusting the published
    float — so the page's two statements about the same quantity ("读取
    1024 → 256 个元素" and "算术强度 0.25 → 2.0") cannot drift apart.
  * `ceiling` is `min(peak, bandwidth × ai)`: the performance the Roofline says
    is the most this kernel could reach AT that intensity, and `bound` names
    which roof that is. It is explicitly a BOUND, not a measurement — nothing
    in this lab was timed, and the page labels the axis accordingly. The
    alternative (inventing a plausible achieved FLOP/s) would be exactly the
    hand-written demo data the contract forbids.

The grid is the product of {2, 4, 8} per parameter, filtered to combinations
that divide M/N/K. The filter is not cosmetic: a tile that does not divide the
matrix leaves a partial block at the edge, and tail handling is a different
algorithm (FlashAttention's subject, L06) — the naive-against-tiled exactness
assertion below would still hold, but the *traffic formula* would not, because
`(M/B_M)(N/B_N)(K/B_K)` counts whole blocks. So impossible tiles are skipped
rather than silently rounded.

Three replay phases, in the order the design doc states them
(`docs/plans/interactive-labs.md` §二 L01):

  1. 点积展开    C_00 = sum_k A_0k B_k0, one step per k. A and B elements go
                 HBM -> REG directly and SMEM is never touched. That absence is
                 the point of the phase, and the stage shows it as an empty
                 layer rather than as an omission.
  2. 朴素铺满    one step that prices the naive kernel: the same dot product
                 repeated for all 64 outputs, with the traffic that implies.
  3. 分块搬运    the (i,j) block loop x the k block loop, 2x2 blocks of 4x4.
                 Each k iteration is two steps: HBM -> SMEM for the A/B tiles,
                 then SMEM -> REG for the thread's fragments and the k-loop of
                 outer products that feeds the accumulator. Each block ends by
                 writing its accumulator back.

TWO FIELDS THIS COMPONENT ADDED TO THE CONTRACT
-----------------------------------------------
Both are on the step and optional, so resolve(trace, i) stays a pure function of
the cursor — there is no per-frame side table to keep in sync.

`step.spans` — the sub-region of a resident block the step touches, as one
[lo, hi) range per axis: `{"A": [[0, 4], [0, 4]]}`. The stage outlines that
4x4 corner of the 8x8 A instead of tinting the whole matrix, which is the
difference between "A is involved" and "this tile of A is being carried".
Purely presentational; absent means outline nothing.

`step.flows` — how much data crossed which layer boundary: `{from, to,
elements, bytes}`. This one was forced by the data rather than designed up
front, and the reason is worth recording, because the obvious derivation is
wrong. The tempting rule is "a move is (a read tensor's layer) -> (a written
tensor's layer), sized by the written tensors" — it works for a load step and
then breaks twice:

  * the mma step writes `a_frag`, `b_frag` AND `c_frag`; only the two fragments
    crossed the SMEM->REG boundary (8 elements). `c_frag` is an accumulator that
    never left REG, so sizing the flow by the written tensors counts 24.
  * the store of one element into C writes a 64-element tensor, so sizing by the
    written tensor counts 64 where 1 crossed.

Neither is recoverable from reads/writes alone; both are facts only the
algorithm knows. So the trace states them. (This is P01's lesson applied:
an abstract contract is not the place to be clever — let the real trace say
what the view cannot derive.) The stage derives *residency* — which blocks are
in which layer at cursor i — because that genuinely is a pure function of the
trace, and reads *flows* because that is not.

Run: python3 labs/traces/gemm_tiling.py
"""

import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# Sentinel spellings for values JSON cannot represent. These are the contract's
# strings, not a local convention -- the engine parses exactly these. No step in
# this trace needs one, but lint checks for them, so the vocabulary is here for
# the same reason it is in online_softmax.py.
NEG_INF = r"-\infty"
POS_INF = r"+\infty"
NAN = "NaN"


# ----------------------------------------------------------------- formatting
def fmt(v):
    """LaTeX form, for formula bindings. Never put this in narration."""
    f = float(v)
    if math.isnan(f):
        return r"\mathrm{NaN}"
    if math.isinf(f):
        return NEG_INF if f < 0 else POS_INF
    if f == 0:
        return "0"
    return "%.4g" % f


def disp(v):
    """Unicode form, for prose. fmt() returns LaTeX, which would render as
    literal backslashes if it leaked into a narration string (the lint
    checks for that)."""
    f = float(v)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "-∞" if f < 0 else "+∞"
    if f == 0:
        return "0"
    return "%.4g" % f


def state_val(v):
    """A numeric value in its JSON-safe form: a float, or a sentinel string."""
    f = float(v)
    if math.isnan(f):
        return NAN
    if math.isinf(f):
        return NEG_INF if f < 0 else POS_INF
    return f


def mat(values):
    """Nested-list form of a 2-D array, for `state` and `tensors[].init`."""
    return [[state_val(x) for x in row] for row in values]


def vec(values):
    return [state_val(x) for x in values]


def vec_tex(values):
    """A short 1-D vector in a `num` tier.

    Only vectors go into formulas, never a matrix: the slot's \\phantom
    reserves the width of whatever the value will be, so a 4x4 accumulator
    would reserve four times the space of the line it sits on and push the
    formula out of its panel. The accumulator is a *resident block*, so the
    stage below shows it as a grid -- which is both narrower and more to the
    point than repeating it in the formula.

    A binding value is plain LaTeX. \\slot{} and \\region{} are expanded only in
    the formula string, never inside a binding, so one must not appear here.
    """
    return r"\left[" + r",\; ".join(fmt(v) for v in values) + r"\right]"


# ------------------------------------------------------------------- the math
#
# THE INPUTS ARE LITERALS, THE RESULTS ARE NOT.
#
# Same split as online_softmax.py's `x`: A and B are fixed problem data (a lab
# needs a specific, reproducible example), and every derived number -- C, the
# accumulators, the traffic counts, the arithmetic intensity -- is computed. The
# values are two-decimal so they stay readable at four significant digits, and
# this pair was checked to give a C with no zeros and a spread that makes the
# heatmap in the tensor panel legible.
A_DATA = [
    [0.25, 0.79, 0.55, -0.55, -0.40, 0.75, -0.99, 0.64],
    [0.59, -0.06, -0.39, -0.44, -0.49, -0.11, 0.01, 0.11],
    [0.99, 0.59, 0.24, 0.98, -0.57, -0.68, 0.23, -0.91],
    [-0.93, 0.03, -0.07, 0.83, 0.26, 0.03, -0.01, -0.50],
    [-0.98, -0.62, 0.38, -0.60, -0.26, -0.99, 0.66, -0.69],
    [-0.46, 0.76, 0.02, 0.69, 0.28, 0.48, -0.82, 0.08],
    [0.02, 0.74, -0.28, 0.20, -0.88, -0.22, -0.35, -0.70],
    [0.63, -0.24, 0.96, 0.18, 0.21, 0.28, 0.35, -0.70],
]
B_DATA = [
    [-0.12, -0.52, -0.20, -0.81, 0.94, -0.57, 0.34, -0.40],
    [0.75, 0.32, -0.74, 0.69, 0.89, 0.81, 0.14, -0.71],
    [-0.62, 0.86, 0.10, -0.64, 0.77, 0.28, 0.14, -0.25],
    [-0.18, -0.52, -0.92, 0.75, -0.06, 0.10, -0.36, 0.50],
    [-0.95, -0.26, -0.94, -0.75, 0.93, 0.32, -0.14, 0.05],
    [0.75, -0.31, 0.18, 0.37, -0.29, 0.04, 0.53, 0.82],
    [-0.70, 0.87, -0.99, 0.51, 0.62, -0.73, -0.16, 0.63],
    [-0.97, 0.26, 0.59, 0.03, 0.45, -0.55, -0.60, -0.27],
]

M, N, K = 8, 8, 8

# The hardware model the Roofline is drawn against: the tutorial's own V100
# numbers (`4.1-CUDA GEMM算子性能优化.md` §2.1 — 15.7 TFLOP/s FP32, 900 GB/s HBM),
# and the same pair the summary step's `\region{RIDGE}` note already quotes. They
# live here rather than on the page because the ridge point is a number the
# trace publishes, and a trace may not publish a number it did not compute.
PEAK_FLOPS = 15.7e12
BW_BYTES = 900e9

# The tile the page opens on, and the configuration the click-through default
# lands on.
DEFAULT_TILE = (4, 4, 4)


def run_dot(A, B, i, j):
    """C_ij by the textbook k-ordered dot product, in plain Python floats.

    Plain floats, not numpy scalars, are what make the exact-equality assertion
    against the tiled version meaningful: the operation order is 0..K-1 in both
    and nothing is free to reassociate or contract an FMA.
    """
    acc = 0.0
    trail = []
    for k in range(K):
        old = acc
        av, bv = float(A[i][k]), float(B[k][j])
        acc = acc + av * bv
        trail.append(dict(k=k, a=av, b=bv, old=old, new=acc))
    return acc, trail


def run_naive(A, B):
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            C[i][j], _ = run_dot(A, B, i, j)
    return C


def run_tiled(A, B, bm, bn, bk):
    """The same C, block by block, in the same k order.

    Structure follows the block-tiling kernel in the tutorial (section 4.5),
    with the thread-level fragment from section 5.2 made explicit: per k the
    thread holds one column of A_smem and one row of B_smem in registers and
    updates the whole TM x TN tile by outer product. The inner k loop is a
    single replay step, so a frame records the last k's fragments and the
    accumulator after all B_K of them.

    `bm/bn/bk` are parameters rather than module state because the tile is what
    the sliders vary: the loop bounds below are the only place the tiling
    parameters touch the arithmetic, and every configuration runs this same
    function. The k order stays 0..K-1 in every configuration — changing the
    tile changes how the work is BLOCKED, never the order things are summed in —
    which is what lets the exact-equality assertion hold across the whole grid
    rather than only at the default.
    """
    C = [[0.0] * N for _ in range(M)]
    frames = []
    for bi in range(M // bm):
        r0 = bi * bm
        for bj in range(N // bn):
            c0 = bj * bn
            c_frag = [[0.0] * bn for _ in range(bm)]
            for bki in range(K // bk):
                k0 = bki * bk
                A_smem = [[float(A[r0 + p][k0 + q]) for q in range(bk)]
                          for p in range(bm)]
                B_smem = [[float(B[k0 + p][c0 + q]) for q in range(bn)]
                          for p in range(bk)]
                a_frag, b_frag = None, None
                for k in range(bk):
                    a_frag = [A_smem[p][k] for p in range(bm)]
                    b_frag = [B_smem[k][q] for q in range(bn)]
                    for p in range(bm):
                        for q in range(bn):
                            c_frag[p][q] = c_frag[p][q] + a_frag[p] * b_frag[q]
                frames.append(dict(
                    bi=bi, bj=bj, bk=bki, r0=r0, c0=c0, k0=k0, last=(bki == K // bk - 1),
                    A_smem=[[row[q] for q in range(bk)] for row in A_smem],
                    B_smem=[[row[q] for q in range(bn)] for row in B_smem],
                    a_frag=a_frag, b_frag=b_frag,
                    c_frag=[[c_frag[p][q] for q in range(bn)] for p in range(bm)],
                ))
            for p in range(bm):
                for q in range(bn):
                    C[r0 + p][c0 + q] = c_frag[p][q]
            # Snapshot C once per block, after that block's store, so the store
            # step carries the post-write value the contract asks for.
            frames[-1]["C_after"] = [[C[r][c] for c in range(N)] for r in range(M)]
    return C, frames


# Tolerance calibration -- see the module docstring. Both sides are float64, so
# the residual is summation order and nothing else. This is the independent
# cross-check, not the headline claim (that is the exact equality above).
_CHECK_REL_TOL = 1e-12
_CHECK_ABS_TOL = 1e-15


def torch_reference(A, B):
    """Independent reference, deliberately a different implementation: one
    library matmul over the whole matrix rather than a scalar k-loop."""
    C = torch.matmul(torch.tensor(A, dtype=torch.float64),
                     torch.tensor(B, dtype=torch.float64))
    return [[float(C[i][j]) for j in range(N)] for i in range(M)]


def traffic(M, N, K, bm, bn, bk):
    """HBM element traffic for the naive and the tiled kernel.

    Naive: every one of the MN outputs reads its own K-row of A and K-column of
    B, so 2MNK element reads. Tiled: each of the (M/BM)(N/BN)(K/BK) block
    iterations moves BM*BK + BK*BN elements, and the reuse is exactly the block
    area divided by the tile perimeter. These are the numbers the narration and
    the page quote, computed here so they cannot drift from the algorithm.

    Everything here scales with the tile, which is the point of the lab: growing
    `bm`/`bn` grows the reuse, and shrinking `bk` grows the number of block
    iterations and therefore the traffic. Both appear in
    `blocks * (bm*bk + bk*bn)`, and the page's sliders move exactly that.
    """
    blocks = (M // bm) * (N // bn) * (K // bk)
    naive_reads = 2 * M * N * K
    flops = 2 * M * N * K
    tiled_reads = tiled_read_count(M, N, K, bm, bn, bk)
    # The same quotient written the tutorial's way — tile area over tile
    # perimeter, `bm*bn / (2*(bm+bn))`. Published alongside the counted form
    # precisely so the lint can require them to agree: they are two routes to one
    # number, and a page quoting both would otherwise be making two independent
    # claims that nothing keeps in step.
    closed_ai = (bm * bn) / (2 * (bm + bn))
    counted_ai = flops / (tiled_reads * 4)
    return dict(
        blocks=blocks,
        # The number of HBM -> SMEM load OPERATIONS: one per (block, k-step)
        # iteration, each moving an A-tile and a B-tile. This is the quantity the
        # tutorial's §4.2 prices ("访问量从 O(MNK) 降到 O(MNK/BK · 常数) 量级"), and
        # unlike the element count below it DOES depend on B_K: halving B_K
        # doubles the number of loads. Published separately because the page
        # shows "访存次数" and "搬运的元素量" as two different numbers, and they
        # genuinely are two different quantities.
        load_steps=blocks,
        # The SMEM footprint of one k-step's tiles, in elements. §4.2's whole
        # reason for splitting K is that this must fit the SMEM budget, so it is
        # the number that says what a smaller B_K buys.
        smem_tile=bm * bk + bk * bn,
        naive_reads=naive_reads,
        tiled_reads=tiled_reads,
        reuse=naive_reads / tiled_reads,
        naive_flops=flops,
        naive_fma=M * N * K,
        naive_bytes=naive_reads * 4,
        tiled_bytes=tiled_reads * 4,
        naive_ai=flops / (naive_reads * 4),
        tiled_ai=counted_ai,
        tiled_ai_closed=closed_ai,
        ridge=PEAK_FLOPS / BW_BYTES,
        # The tutorial's 128x128 tile, computed rather than quoted: the same
        # counting helper on a 128-cubed problem with a 128-cubed tile, so the
        # reference marker on the Roofline is a call of this function and not a
        # literal transcribed from section 4.3.
        doc_reads=DOC_READS,
        doc_ai=(2 * DOC_N ** 3) / (DOC_READS * 4),
    )


def tiled_read_count(M, N, K, bm, bn, bk):
    """HBM element reads of the tiled kernel: whole blocks times tile perimeter.

    Factored out of `traffic()` so the tutorial's 128x128 reference is a call of
    the SAME function the live configurations go through, rather than a second
    transcription of the formula that could drift from it.
    """
    blocks = (M // bm) * (N // bn) * (K // bk)
    return blocks * (bm * bk + bk * bn)


# The tutorial's BM=BN=BK=128 example (`4.1-CUDA GEMM算子性能优化.md` §4.3), sized
# by the counting helper above. Kept as a module constant so `traffic()` can
# publish it on every configuration without recursing into itself.
DOC_N = 128
DOC_READS = tiled_read_count(DOC_N, DOC_N, DOC_N, DOC_N, DOC_N, DOC_N)


def roofline(tr):
    """The Roofline placement of this configuration, as three real points.

    The Roofline's two roofs are the hardware model: `bandwidth * ai` below the
    ridge, `peak_flops` above it. A kernel at arithmetic intensity `ai` cannot
    beat `min(peak, bandwidth * ai)` — so that minimum is what gets plotted, and
    `bound` names which roof is binding. It is a CEILING and the field is named
    for that: this lab times nothing, and a plotted "achieved FLOP/s" would be a
    number no script produced.

    Three points rather than one, because the comparison is the lesson:

      * `naive` — the untiled kernel's 0.25 FLOP/Byte. It does not move with the
        sliders, and that stillness is the control: it is what makes the tiled
        point's movement mean something.
      * `tiled` — this configuration. The only point the sliders move.
      * `doc` — the tutorial's BM=BN=128 tile, 32 FLOP/Byte. Drawn as a
        reference so a reader can see that the toy tile's whole journey stays on
        the bandwidth roof, which is the honest end to the story rather than a
        contradiction of section 4.3.
    """
    peak, bw = PEAK_FLOPS, BW_BYTES
    ridge = peak / bw

    def point(pid, label, elements, flops):
        """One Roofline dot: a byte count, the FLOPs it bought, and the roof.

        `ai` is DERIVED here from the other two rather than passed in, so the
        point cannot claim an intensity its own byte and FLOP counts contradict.
        `flops` is carried so the lint can re-derive it a third time: the chart,
        the intensity and the traffic counter are then three statements about one
        number instead of three independent ones.
        """
        bytes_ = elements * 4
        ai = float(flops) / bytes_
        ceiling = min(peak, bw * ai)
        return dict(id=pid, label=label, ai=ai, elements=elements,
                    bytes=bytes_, flops=flops, ceiling=ceiling,
                    bound="bandwidth" if ai < ridge else "compute")

    return dict(
        peak_flops=peak,
        bandwidth=bw,
        ridge=ridge,
        x_label="算术强度 (FLOP/Byte)",
        # The y axis is drawn in TFLOP/s, and the DIVISOR is published beside
        # the label rather than assumed by the view: a chart whose label reads
        # TFLOP/s while its ticks are FLOP/s is wrong in a way no reader can
        # see, and the pair is the only way to keep the two in step. See the
        # "y_divisor" rule in both lint ports.
        y_label="可达算力上界 (TFLOP/s)",
        y_divisor=1e12,
        y_unit="TFLOP/s",
        note="点画在 min(峰值算力, 带宽 × 算术强度) 上，是 Roofline 给出的上界，"
             "不是实测值 —— 本实验室没有对任何 kernel 计时。",
        points=[
            # The naive and tiled points are both this configuration's problem:
            # 2MNK FLOPs each, differing only in the bytes they moved to get
            # them. The doc point is a DIFFERENT problem (128 cubed), which is
            # why it carries its own FLOP count rather than borrowing this trace's.
            point("naive", "朴素（不分块）", tr["naive_reads"], tr["naive_flops"]),
            point("tiled", "本配置分块", tr["tiled_reads"], tr["naive_flops"]),
            point("doc", "教程 128×128", tr["doc_reads"], 2 * DOC_N ** 3),
        ],
    )


def check_against_torch(C_naive):
    ref = torch_reference(A_DATA, B_DATA)
    worst = 0.0
    for i in range(M):
        for j in range(N):
            if not math.isclose(C_naive[i][j], ref[i][j],
                                rel_tol=_CHECK_REL_TOL, abs_tol=_CHECK_ABS_TOL):
                raise AssertionError(
                    f"C[{i}][{j}]: naive={C_naive[i][j]!r} torch={ref[i][j]!r}")
            worst = max(worst, abs(C_naive[i][j] - ref[i][j]))
    return ref, worst


# ---------------------------------------------------------------- trace build
def b(slot, sym, idx, num):
    return slot, {"sym": sym, "idx": idx, "num": num}


def step(sid, title, kind, op, phase, formula, bindings, reads, writes, state,
         narration, regions=None, spans=None, flows=None):
    d = {"id": sid, "title": title, "kind": kind, "op": op, "phase": phase,
         "formula": formula, "bindings": dict(bindings),
         "reads": reads, "writes": writes, "state": state, "narration": narration}
    if regions:
        d["regions"] = regions
    if spans:
        d["spans"] = spans
    if flows:
        d["flows"] = [dict(f, bytes=f["elements"] * 4) for f in flows]
    return d


# The three layers this lab's replay moves data through, top to bottom. The ids
# are what `tensors[].at` and `step.flows[].from/to` refer to, and what the view
# component's layer config must match. L06's two-layer stage is the same
# mechanic with a shorter list -- HBM, then SRAM.
LAYERS = ("HBM", "SMEM", "REG")


def build_trace(title, source, bm, bn, bk):
    A, B = np.array(A_DATA, dtype=np.float64), np.array(B_DATA, dtype=np.float64)
    C_naive = run_naive(A, B)
    C_tiled, frames = run_tiled(A, B, bm, bn, bk)

    # The load-bearing assertion: tiling moves data differently and must not
    # change the arithmetic. See the module docstring for why this is exact
    # rather than approximate. Run for EVERY configuration, not just the
    # default: a tile that changed the answer is caught here whichever slider
    # position produced it, so the claim "the sliders move the traffic and not
    # the result" is checked 27 times rather than once.
    if not np.array_equal(np.array(C_naive), np.array(C_tiled)):
        bad = [(i, j, C_naive[i][j], C_tiled[i][j])
               for i in range(M) for j in range(N)
               if C_naive[i][j] != C_tiled[i][j]]
        raise AssertionError(f"tiled({bm},{bn},{bk}) != naive at {len(bad)} entries, "
                             f"first {bad[0]}")

    _, worst = check_against_torch(C_naive)
    tr = traffic(M, N, K, bm, bn, bk)
    roof = roofline(tr)

    steps = []

    # ------------------------------------------------------------- phase 0
    steps.append(step(
        "init", "初始化：问题规模与分块参数", "state", "init", "准备",
        {
            "sym": r"C^{(0)}_{ij} = 0,\qquad \mathrm{tile} = \slot{BM}\times\slot{BN}\times\slot{BK}",
            "idx": r"C^{(0)}_{ij} = 0,\qquad (M,N,K) = (\slot{M},\slot{N},\slot{K}),\qquad "
                   r"\mathrm{tile} = \slot{BM}\times\slot{BN}\times\slot{BK}",
            "num": r"C^{(0)}_{ij} = \slot{Z},\qquad (M,N,K) = (\slot{M},\slot{N},\slot{K}),\qquad "
                   r"\mathrm{tile} = \slot{BM}\times\slot{BN}\times\slot{BK},\qquad "
                   r"\mathrm{blocks} = \slot{NB}",
        },
        [b("Z", "0", "0", "0"),
         b("M", "M", str(M), str(M)),
         b("N", "N", str(N), str(N)),
         b("K", "K", str(K), str(K)),
         b("BM", "B_M", str(bm), str(bm)),
         b("BN", "B_N", str(bn), str(bn)),
         b("BK", "B_K", str(bk), str(bk)),
         b("NB", r"(M/B_M)(N/B_N)(K/B_K)",
           f"({M}/{bm})({N}/{bn})({K}/{bk})", str(tr["blocks"]))],
        [], ["C"], {"C": mat(np.zeros((M, N)))},
        f"A 和 B 已经在 HBM 里，C 先清零。{M}×{N}×{K} 的矩阵按 {bm}×{bn}×{bk} 分块，"
        f"C 是 {M // bm}×{N // bn} 个块，每个块沿 K 要迭代 {K // bk} 次。",
    ))

    # ------------------------------------------- phase 1: 单点积 C_00
    _, dot = run_dot(A, B, 0, 0)
    for d in dot:
        k, av, bv, old, new = d["k"], d["a"], d["b"], d["old"], d["new"]
        steps.append(step(
            f"d{k}", f"点积 C00：第 {k + 1} 个 k", "op", "fma", "点积展开",
            {
                "sym": r"\slot{NEW} \leftarrow \slot{OLD} + \region{MUL}{A_{i\slot{K}}\,B_{\slot{K}j}}",
                "idx": r"\slot{NEW} \leftarrow \slot{OLD} + \region{MUL}{A_{0\slot{K}}\,B_{\slot{K}0}}",
                "num": r"\slot{NEW} \leftarrow \slot{OLD} + \region{MUL}{\slot{AV}\cdot\slot{BV}}",
            },
            [b("K", "k", str(k), str(k)),
             b("OLD", r"C_{ij}", r"C_{00}", fmt(old)),
             b("AV", r"A_{ik}", f"A_{{0{k}}}", fmt(av)),
             b("BV", r"B_{kj}", f"B_{{{k}0}}", fmt(bv)),
             b("NEW", r"C_{ij}", r"C_{00}", fmt(new))],
            ["A", "B", "acc"], ["a_val", "b_val", "acc"],
            {"a_val": state_val(av), "b_val": state_val(bv), "acc": state_val(new)},
            f"取 A[0][{k}] = {disp(av)} 和 B[{k}][0] = {disp(bv)}，乘起来累加："
            f"{disp(old)} + {disp(av)}×{disp(bv)} = {disp(new)}。"
            f"这两个数是从 HBM 直接进寄存器的 —— 朴素实现里没有 SMEM 这一层。",
            {"MUL": f"{disp(av)} × {disp(bv)} 是要累加的那一项。两个操作数都来自 HBM，"
                    f"每算一个输出点都要重新读一遍，谁都复用不上。"},
            spans={"A": [[0, 1], [k, k + 1]], "B": [[k, k + 1], [0, 1]]},
            # HBM -> REG, straight past SMEM: 2 elements per k, and SMEM is
            # marked unused for the whole phase. That skip is the lesson.
            flows=[{"from": "HBM", "to": "REG", "elements": 2,
                    "note": f"A[0][{k}] 与 B[{k}][0] 各 1 个 float，直接进寄存器"}],
        ))

    c_after_dot = [[0.0] * N for _ in range(M)]
    c_after_dot[0][0] = dot[-1]["new"]
    steps.append(step(
        "d8", "点积结束：写回 C00", "op", "store", "点积展开",
        {
            "sym": r"\mathrm{HBM}[C][\slot{RI}][\slot{CI}] \leftarrow \slot{ACC}",
            "idx": r"\mathrm{HBM}[C][0][0] \leftarrow \slot{ACC}",
            "num": r"\mathrm{HBM}[C][0][0] \leftarrow \slot{ACC} = \slot{SUM}",
        },
        [b("RI", "i", "0", "0"), b("CI", "j", "0", "0"),
         b("ACC", r"\mathrm{acc}", r"\mathrm{acc}", fmt(dot[-1]["new"])),
         b("SUM", r"C_{ij}", r"C_{00}", fmt(C_naive[0][0]))],
        ["acc"], ["C"], {"C": mat(np.array(c_after_dot))},
        f"8 次乘加之后 C[0][0] = {disp(C_naive[0][0])}。整个矩阵有 {M * N} 个输出点，"
        f"每个都得这么来一遍。",
        spans={"C": [[0, 1], [0, 1]]},
        flows=[{"from": "REG", "to": "HBM", "elements": 1,
                "note": "一个输出点，一个 float 写回 HBM"}],
    ))

    # ------------------------------------------- phase 2: 朴素铺满
    steps.append(step(
        "naive", "朴素铺满：同样的事做 64 遍", "op", "cost", "朴素铺满",
        {
            "sym": r"\slot{NEW} = \sum_{k=0}^{K-1} A_{ik}B_{kj}\quad \text{for every }(i,j)",
            "idx": r"\text{reads} = 2MNK = \slot{RD},\qquad \text{FMA} = MNK = \slot{FL}",
            "num": r"\region{RBYTES}{\text{reads} = \slot{RD}},\qquad \text{FMA} = \slot{FL},\qquad "
                   r"\mathrm{AI} = \slot{AI}\ \mathrm{FLOP/B}",
        },
        [b("NEW", r"C_{ij}", r"C_{ij}", r"C_{ij}"),
         b("RD", r"2MNK", f"2\\times{M}\\times{N}\\times{K}", str(tr["naive_reads"])),
         b("FL", r"MNK", f"{M}\\times{N}\\times{K}", str(tr["naive_fma"])),
         b("AI", r"\frac{2K}{8K}", r"\frac{2K}{8K}", fmt(tr["naive_ai"]))],
        ["A", "B"], ["C"], {"C": mat(np.array(C_naive))},
        f"把上面的点积对全部 {M * N} 个输出点各做一次，C 就满了。代价是 "
        f"{tr['naive_reads']} 次 HBM 读取（{tr['naive_bytes']} 字节）换 "
        f"{tr['naive_fma']} 次 FMA：算术强度 {disp(tr['naive_ai'])} FLOP/Byte。"
        f"A 的同一行被不同列重复读、B 的同一列被不同行重复读，一次都没复用上。",
        {"RBYTES": f"每次乘加都要从 HBM 取 2 个 float，共 {tr['naive_reads']} 次读取，"
                   f"fp32 下就是 {tr['naive_bytes']} 字节。这是下面要消掉的量。"},
        spans={"C": [[0, M], [0, N]]},
        flows=[{"from": "HBM", "to": "HBM", "elements": tr["naive_reads"],
                "note": f"{M * N} 个输出点各读 2K 个元素，全部从 HBM 出发"}],
    ))

    # ------------------------------------------- phase 3: 分块搬运
    for f in frames:
        bi, bj, bki = f["bi"], f["bj"], f["bk"]
        r0, c0, k0 = f["r0"], f["c0"], f["k0"]
        tag = f"{bi}{bj}{bki}"
        moved = (bm * bk + bk * bn) * 4

        steps.append(step(
            f"ld{tag}", f"块({bi},{bj})·K{bki + 1}：A、B 的 tile 载入 SMEM",
            "comm", "load", "分块搬运",
            {
                "sym": r"A^{\mathrm{sm}} \leftarrow \mathrm{HBM}[A]\left[\slot{R0}:\slot{R1},\ \slot{K0}:\slot{K1}\right],"
                       r"\qquad B^{\mathrm{sm}} \leftarrow \mathrm{HBM}[B]\left[\slot{K0}:\slot{K1},\ \slot{C0}:\slot{C1}\right]",
                "idx": r"A^{\mathrm{sm}} \leftarrow \mathrm{HBM}[A]\left[\slot{R0}:\slot{R1},\ \slot{K0}:\slot{K1}\right],"
                       r"\qquad B^{\mathrm{sm}} \leftarrow \mathrm{HBM}[B]\left[\slot{K0}:\slot{K1},\ \slot{C0}:\slot{C1}\right]",
                "num": r"\region{MOVE}{A^{\mathrm{sm}},\; B^{\mathrm{sm}} \leftarrow \slot{BYTES}\ \mathrm{B}}",
            },
            [b("R0", r"(b_i)B_M", str(r0), str(r0)),
             b("R1", r"(b_i+1)B_M", str(r0 + bm), str(r0 + bm)),
             b("K0", r"(b_k)B_K", str(k0), str(k0)),
             b("K1", r"(b_k+1)B_K", str(k0 + bk), str(k0 + bk)),
             b("C0", r"(b_j)B_N", str(c0), str(c0)),
             b("C1", r"(b_j+1)B_N", str(c0 + bn), str(c0 + bn)),
             b("BYTES", r"4(B_MB_K + B_KB_N)",
               f"4\\times({bm * bk} + {bk * bn})", str(moved))],
            ["A", "B"], ["A_smem", "B_smem"],
            {"A_smem": mat(np.array(f["A_smem"])),
             "B_smem": mat(np.array(f["B_smem"]))},
            f"块({bi},{bj}) 的第 {bk + 1} 个 K 迭代：A 的 [{r0}:{r0 + bm}, {k0}:{k0 + bk}] 和 "
            f"B 的 [{k0}:{k0 + bk}, {c0}:{c0 + bn}] 从 HBM 搬进 SMEM，"
            f"共 {bm * bk + bk * bn} 个 float（{moved} 字节）。"
            f"这一对 tile 接下来会被块内 {bm * bn} 个输出点复用。",
            {"MOVE": f"这次搬运把 A、B 各一个 {bm}×{bk} 的 tile 送进 SMEM，"
                     f"合计 {bm * bk + bk * bn} 个 float，即 {moved} 字节。"
                     f"朴素实现算这 {bm * bn} 个输出点要读 {bm * bn * 2 * K} 个元素。"},
            {"A": [[r0, r0 + bm], [k0, k0 + bk]],
             "B": [[k0, k0 + bk], [c0, c0 + bn]]},
            flows=[{"from": "HBM", "to": "SMEM", "elements": bm * bk + bk * bn,
                    "note": f"A 的 {bm}×{bk} tile 与 B 的 {bk}×{bn} tile，一次搬完"}],
        ))

        steps.append(step(
            f"mma{tag}", f"块({bi},{bj})·K{bki + 1}：SMEM → REG 外积累加",
            "op", "fma", "分块搬运",
            {
                "sym": r"C^{\mathrm{frag}}_{pq} \mathrel{+}= \sum_{\slot{P}} "
                       r"A^{\mathrm{sm}}_{p\slot{P}}\,B^{\mathrm{sm}}_{\slot{P}q}",
                "idx": r"C^{\mathrm{frag}}_{pq} \mathrel{+}= \sum_{\slot{P}=\slot{P0}}^{\slot{P1}} "
                       r"A^{\mathrm{sm}}_{p\slot{P}}\,B^{\mathrm{sm}}_{\slot{P}q}",
                "num": r"\slot{AV} \leftarrow A^{\mathrm{sm}}\left[:,\slot{PL}\right],\qquad "
                       r"\slot{BV} \leftarrow B^{\mathrm{sm}}\left[\slot{PL},:\right],\qquad "
                       r"\region{OUTER}{\slot{FRAG}}",
            },
            [b("P", "p", str(bk - 1), str(bk - 1)),
             b("P0", "0", "0", "0"),
             b("P1", "B_K-1", str(bk - 1), str(bk - 1)),
             b("PL", "B_K-1", str(bk - 1), str(bk - 1)),
             b("AV", r"A^{\mathrm{sm}}[:,B_K-1]",
               f"A^{{\\mathrm{{sm}}}}[:,{bk - 1}]", vec_tex(f["a_frag"])),
             b("BV", r"B^{\mathrm{sm}}[B_K-1,:]",
               f"B^{{\\mathrm{{sm}}}}[{bk - 1},:]", vec_tex(f["b_frag"])),
             # The outer product reads the same at every tier -- what the num
             # tier substitutes is the operands, not this line. The accumulator
             # it produces is rendered by the stage as a REG-resident block.
             b("FRAG", r"C^{\mathrm{frag}} \mathrel{+}= a^{\mathrm{frag}}\otimes b^{\mathrm{frag}}",
               r"C^{\mathrm{frag}} \mathrel{+}= a^{\mathrm{frag}}\otimes b^{\mathrm{frag}}",
               r"C^{\mathrm{frag}} \mathrel{+}= a^{\mathrm{frag}}\otimes b^{\mathrm{frag}}")],
            ["A_smem", "B_smem", "c_frag"], ["a_frag", "b_frag", "c_frag"],
            {"a_frag": vec(f["a_frag"]), "b_frag": vec(f["b_frag"]),
             "c_frag": mat(np.array(f["c_frag"]))},
            f"线程把 A_smem 的第 {bk} 列 {bm} 个数、B_smem 的第 {bk} 行 {bn} 个数搬进寄存器，"
            f"然后做外积。本块的 {bk} 个 k 里，每轮读 {bm} + {bn} = {bm + bn} 个 float，"
            f"完成 {bm * bn} 次 FMA —— 这就是 SMEM 这一级买到的复用。",
            {"OUTER": f"每轮 SMEM 读 {bm + bn} 个 float 做 {bm * bn} 次 FMA，"
                      f"比例 {bm * bn} / {bm + bn} = {bm * bn // (bm + bn)} : 1，"
                      f"而朴素实现是每读 2 个 float 做 1 次 FMA。"},
            {"A_smem": [[0, bm], [bk - 1, bk]],
             "B_smem": [[bk - 1, bk], [0, bn]]},
            # Two flows, not three, and the count is the point: a_frag and
            # b_frag crossed SMEM -> REG (bm + bn elements). c_frag did not --
            # it is an accumulator that stays in REG -- which is why deriving
            # the flow from `writes` would be wrong here.
            flows=[{"from": "SMEM", "to": "REG", "elements": bm + bn,
                    "note": f"A_smem 的 {bm} 个 + B_smem 的 {bn} 个，共 {bm + bn} 个进寄存器"},
                   {"from": "REG", "to": "REG", "elements": bm * bn,
                    "note": f"寄存器内的 {bm * bn} 次 FMA，c_frag 原地累加，不跨层"}],
        ))

        # The store comes once per block, after its last k iteration.
        if f["last"]:
            steps.append(step(
                f"st{bi}{bj}", f"块({bi},{bj})：累加器写回 C",
                "comm", "store", "分块搬运",
                {
                    "sym": r"\mathrm{HBM}[C]\left[\slot{R0}:\slot{R1},\ \slot{C0}:\slot{C1}\right] "
                           r"\leftarrow C^{\mathrm{frag}}",
                    "idx": r"\mathrm{HBM}[C]\left[\slot{R0}:\slot{R1},\ \slot{C0}:\slot{C1}\right] "
                           r"\leftarrow C^{\mathrm{frag}}",
                    "num": r"\mathrm{HBM}[C]\left[\slot{R0}:\slot{R1},\ \slot{C0}:\slot{C1}\right] "
                           r"\leftarrow C^{\mathrm{frag}},\qquad C_{\slot{SI}\slot{SJ}} = \slot{SV}",
                },
                [b("R0", r"(b_i)B_M", str(r0), str(r0)),
                 b("R1", r"(b_i+1)B_M", str(r0 + bm), str(r0 + bm)),
                 b("C0", r"(b_j)B_N", str(c0), str(c0)),
                 b("C1", r"(b_j+1)B_N", str(c0 + bn), str(c0 + bn)),
                 b("SI", "i", str(r0), str(r0)),
                 b("SJ", "j", str(c0), str(c0)),
                 b("SV", r"C_{ij}", f"C_{{{r0}{c0}}}", fmt(C_tiled[r0][c0]))],
                ["c_frag"], ["C"],
                {"C": mat(np.array(f["C_after"]))},
                f"块({bi},{bj}) 的 {bm}×{bn} 个结果从寄存器写回 HBM。这 {bm * bn} 个输出点"
                f"一共从 HBM 读了 {(K // bk) * (bm * bk + bk * bn)} 个元素，"
                f"朴素实现要读 {bm * bn * 2 * K} 个。算出来的值两者逐位相同。",
                spans={"C": [[r0, r0 + bm], [c0, c0 + bn]]},
                flows=[{"from": "REG", "to": "HBM", "elements": bm * bn,
                        "note": f"{bm}×{bn} 个输出点从寄存器写回 HBM"}],
            ))

    # ------------------------------------------- 收尾：把账算清楚
    steps.append(step(
        "sum", "收尾：分块买到了什么", "op", "summary", "收尾",
        {
            "sym": r"\mathrm{reads}_{\mathrm{naive}} = 2MNK,\qquad "
                   r"\mathrm{reads}_{\mathrm{tiled}} = \frac{MN K}{B_MB_NB_K}(B_MB_K + B_KB_N)",
            "idx": r"\mathrm{reads}_{\mathrm{naive}} = \slot{RD},\qquad "
                   r"\mathrm{reads}_{\mathrm{tiled}} = \slot{TD}",
            "num": r"\slot{RDI} \rightarrow \slot{TDI}\ (\times\slot{RATIO}),\qquad "
                   r"\mathrm{AI}: \slot{AI0} \rightarrow \slot{AI1}\ \mathrm{FLOP/B},\qquad "
                   r"\region{RIDGE}{\mathrm{ridge} \approx \slot{RIDGE}}",
        },
        [b("RD", r"2MNK", f"2\\times{M}\\times{N}\\times{K}", str(tr["naive_reads"])),
         b("TD", r"\frac{MNK}{B_MB_NB_K}(B_MB_K+B_KB_N)",
           f"{tr['blocks']}\\times({bm * bk}+{bk * bn})", str(tr["tiled_reads"])),
         b("RDI", r"2MNK", str(tr["naive_reads"]), str(tr["naive_reads"])),
         b("TDI", r"\mathrm{reads}_{\mathrm{tiled}}",
           str(tr["tiled_reads"]), str(tr["tiled_reads"])),
         b("RATIO", r"\mathrm{reuse}", f"{tr['reuse']:.0f}\\times", f"{tr['reuse']:.0f}"),
         b("AI0", r"\frac{2K}{8K}", fmt(tr["naive_ai"]), fmt(tr["naive_ai"])),
         b("AI1", r"\frac{2B_MB_NB_K}{4(B_MB_K+B_KB_N)}",
           fmt(tr["tiled_ai"]), fmt(tr["tiled_ai"])),
         b("RIDGE", r"\mathrm{peak}/\mathrm{bw}", "17.4", "17.4")],
        ["A", "B"], [], {},
        f"同样一个 C，朴素实现从 HBM 读 {tr['naive_reads']} 个元素，"
        f"按 {bm}×{bn}×{bk} 分块只要 {tr['tiled_reads']} 个，少了 {tr['reuse']:.0f} 倍。"
        f"算术强度从 {disp(tr['naive_ai'])} 抬到 {disp(tr['tiled_ai'])} FLOP/Byte —— "
        f"方向和教程里说的一致，但离 V100 的平衡点 {disp(tr['ridge'])} 还差得远："
        f"这个 8×8 的例子是为了把搬运过程看清楚，真要把算力打满，tile 得大到 "
        f"128×128（算术强度 {disp(tr['doc_ai'])}）。",
        {"RIDGE": f"平衡点 = 峰值算力 / 峰值带宽。V100 上约 {disp(tr['ridge'])} FLOP/Byte，"
                  f"超过它才由算力而非带宽决定快慢。"},
        flows=[{"from": "HBM", "to": "SMEM", "elements": tr["tiled_reads"],
                "note": f"整趟回放的 HBM → SMEM 总搬运量：{tr['tiled_reads']} 个元素"},
               {"from": "SMEM", "to": "REG",
                "elements": tr["blocks"] * (bm + bn) * bk,
                "note": "整趟回放的 SMEM → REG 总搬运量"}],
    ))

    # ---------------------------------------------------------------- graph
    # Nodes carry no `step` field on purpose: ids are shared with `steps`, so
    # the engine derives the index. A redundant field could only ever drift.
    nodes = [{"id": s["id"], "title": s["title"], "kind": s["kind"],
              "op": s["op"], "phase": s["phase"]} for s in steps]

    # Data edges. `from -> to` labelled T means "the T produced by `from` is
    # consumed by `to`". The engine synthesises the sequencing edges it needs
    # for layout, so this list carries only the dependencies worth drawing. The
    # boundary between the naive pass and the tiled pass deliberately has no
    # edge: the tiled kernel re-derives C from A and B, it does not read the
    # naive result, and inventing an edge there would draw a dependency the
    # algorithm does not have.
    edges = [{"from": "init", "to": "d0", "tensor": "acc"}]
    for k in range(K - 1):
        edges.append({"from": f"d{k}", "to": f"d{k + 1}", "tensor": "acc"})
    edges.append({"from": f"d{K - 1}", "to": "d8", "tensor": "acc"})
    edges.append({"from": "init", "to": "naive", "tensor": "A"})
    for bi in range(M // bm):
        for bj in range(N // bn):
            for bki in range(K // bk):
                tag = f"{bi}{bj}{bki}"
                edges.append({"from": "init", "to": f"ld{tag}", "tensor": "A"})
                edges.append({"from": f"ld{tag}", "to": f"mma{tag}", "tensor": "A_smem"})
                if bki > 0:
                    # The K loop carries the accumulator across block iterations.
                    edges.append({"from": f"mma{bi}{bj}{bki - 1}", "to": f"mma{tag}",
                                  "tensor": "c_frag"})
            edges.append({"from": f"mma{bi}{bj}{K // bk - 1}", "to": f"st{bi}{bj}",
                          "tensor": "c_frag"})

    return {
        "meta": {
            "lab": "L01",
            "title": title,
            "source": source,
            "config": {"M": M, "N": N, "K": K, "B_M": bm, "B_N": bn, "B_K": bk},
            "reference": (
                f"C = A·B，torch float64 matmul 对拍最大绝对偏差 {worst:.3g}"
                f"（相对容差 {_CHECK_REL_TOL:g}）；朴素 k 序循环与 "
                f"B_M={bm}、B_N={bn}、B_K={bk} 分块逐位相同（np.array_equal）；"
                f"HBM 读取 {tr['naive_reads']} → {tr['tiled_reads']} 个元素，"
                f"算术强度 {tr['naive_ai']:.4g} → {tr['tiled_ai']:.4g} FLOP/B"
            ),
            "traffic": tr,
            "roofline": roof,
        },
        "tensors": {
            # A and B are inputs: no step ever writes them, so they carry `init`
            # -- the contract addition P01 found, without which render(trace, 0)
            # cannot show the input at all.
            "A": {"shape": [M, K], "dtype": "fp32", "at": "HBM", "role": "input",
                  "init": mat(A), "note": "左矩阵，全程驻留 HBM"},
            "B": {"shape": [K, N], "dtype": "fp32", "at": "HBM", "role": "input",
                  "init": mat(B), "note": "右矩阵，全程驻留 HBM"},
            "C": {"shape": [M, N], "dtype": "fp32", "at": "HBM", "role": "output",
                  "note": "输出矩阵，逐块写回"},
            # The naive phase touches HBM and registers only. That SMEM has no
            # entry here is the lesson, not an omission.
            "a_val": {"shape": [], "dtype": "fp32", "at": "REG", "role": "staging",
                      "note": "朴素实现里 A 的元素直接从 HBM 进寄存器"},
            "b_val": {"shape": [], "dtype": "fp32", "at": "REG", "role": "staging",
                      "note": "朴素实现里 B 的元素直接从 HBM 进寄存器"},
            "acc": {"shape": [], "dtype": "fp32", "at": "REG", "role": "state",
                    "init": 0.0, "note": "点积累加器，标量"},
            # The tiled phase's three hops.
            "A_smem": {"shape": [bm, bk], "dtype": "fp32", "at": "SMEM", "role": "staging",
                       "note": "A 的当前 tile，块内所有线程共享"},
            "B_smem": {"shape": [bk, bn], "dtype": "fp32", "at": "SMEM", "role": "staging",
                       "note": "B 的当前 tile，块内所有线程共享"},
            "a_frag": {"shape": [bm], "dtype": "fp32", "at": "REG", "role": "staging",
                       "note": "线程私有：A_smem 的一列"},
            "b_frag": {"shape": [bn], "dtype": "fp32", "at": "REG", "role": "staging",
                       "note": "线程私有：B_smem 的一行"},
            "c_frag": {"shape": [bm, bn], "dtype": "fp32", "at": "REG", "role": "state",
                       "note": "线程私有的累加器，跨 K 迭代保持"},
        },
        "graph": {"nodes": nodes, "edges": edges},
        "steps": steps,
    }


# -------------------------------------------------------------------- self-lint
SLOT_RE = re.compile(r"\\slot\{([A-Za-z0-9_]+)\}")
REGION_RE = re.compile(r"\\region\{([A-Za-z0-9_]+)\}")
LATEX_IN_TEXT_RE = re.compile(r"\\[a-zA-Z]+\{")
BAD_LITERAL_RE = re.compile(r"^(NaN|Infinity|-Infinity|undefined|null)$")


def layer_ids(trace):
    """The layers this trace actually uses, in the order tensors[] first names
    them. Derived rather than declared: a lab that forgets to declare a layer
    list still gets a correct stage, and a tensor stranded in a layer nothing
    else uses is visible as a one-element list rather than an error."""
    order = []
    for spec in trace["tensors"].values():
        at = spec.get("at")
        if at and at not in order:
            order.append(at)
    return order


def lint(trace):
    """Check the trace against the contract the engine consumes.

    This is the author-side half of the check; the engine carries a JS port of
    the same rules so a lab author gets the same feedback in the page. Both
    must stay in step with docs/plans/interactive-labs.md §一.
    """
    gaps, warns, infos = [], [], []
    declared = set(trace["tensors"])
    step_ids = {s["id"] for s in trace["steps"]}

    written, read = set(), set()
    for s in trace["steps"]:
        written.update(s.get("writes") or [])
        read.update(s.get("reads") or [])

    for t in sorted(read):
        if t not in declared:
            gaps.append(f'张量 "{t}" 被读（reads[]），但 tensors[] 里根本没有声明 —— '
                        f"引擎不知道它的 shape / 位置，也没有初始值")
        elif t not in written and "init" not in trace["tensors"][t]:
            gaps.append(f'张量 "{t}" 被读但从未被写，且 tensors[] 里没有 init —— '
                        f"render(trace, 0) 无法得知它的值")
    for t in sorted(written):
        if t not in declared:
            gaps.append(f'步骤写了张量 "{t}"，但 tensors[] 里没有声明')

    node_ids = {n["id"] for n in trace["graph"]["nodes"]}
    for s in trace["steps"]:
        if s["id"] not in node_ids:
            gaps.append(f'步骤 "{s["id"]}" 在 graph.nodes 里没有对应节点')
    for n in trace["graph"]["nodes"]:
        if n["id"] not in step_ids:
            warns.append(f'图节点 "{n["id"]}" 没有对应步骤，点击无处可跳')

    for s in trace["steps"]:
        sid = s["id"]
        formula = s.get("formula")
        if not formula:
            gaps.append(f'步骤 "{sid}" 没有 formula')
            continue
        for tier in ("sym", "idx", "num"):
            if not isinstance(formula.get(tier), str):
                gaps.append(f'步骤 "{sid}" 缺 formula.{tier}')
                continue
            for key in SLOT_RE.findall(formula[tier]):
                bdg = (s.get("bindings") or {}).get(key)
                if bdg is None:
                    gaps.append(f'步骤 "{sid}" 的 formula.{tier} 引用了 \\slot{{{key}}}，'
                                f"但 bindings 里没有")
                elif bdg.get(tier) is None:
                    gaps.append(f'步骤 "{sid}" 的绑定 "{key}" 缺 "{tier}" 形态')
                elif BAD_LITERAL_RE.match(str(bdg[tier])):
                    gaps.append(f'步骤 "{sid}" 绑定 "{key}".{tier} = {bdg[tier]} —— '
                                f"±∞/NaN 要用哨兵字符串")
            for key in REGION_RE.findall(formula[tier]):
                if key not in (s.get("regions") or {}):
                    gaps.append(f'步骤 "{sid}" 的 formula.{tier} 引用了 \\region{{{key}}}，'
                                f"但 step.regions 里没有")

        used = set()
        for tier in ("sym", "idx", "num"):
            used.update(SLOT_RE.findall(formula.get(tier) or ""))
        for key in (s.get("bindings") or {}):
            if key not in used:
                warns.append(f'步骤 "{sid}" 的绑定 "{key}" 没被任何档位引用')

        for key, bdg in (s.get("bindings") or {}).items():
            if set(bdg) != {"sym", "idx", "num"}:
                gaps.append(f'步骤 "{sid}" 的绑定 "{key}" 字段是 '
                            f'{{{", ".join(sorted(bdg))}}} 而不是 {{sym, idx, num}}')
            # A binding value is handed straight to KaTeX by formula.patch, so
            # it must not carry the trace's own macros: those are expanded in
            # the formula string only, and would reach KaTeX as undefined
            # control sequences. Guarded on presence -- a missing tier is
            # already a gap above, and reporting it twice would be noise.
            for tier in ("sym", "idx", "num"):
                tex = bdg.get(tier)
                if tex is not None and (SLOT_RE.search(str(tex)) or REGION_RE.search(str(tex))):
                    gaps.append(f'步骤 "{sid}" 的绑定 "{key}".{tier} 里含 \\slot / \\region —— '
                                f"这两个宏只在 formula 串里展开，绑定值会被原样交给 KaTeX")

        if LATEX_IN_TEXT_RE.search(s.get("narration") or ""):
            warns.append(f'步骤 "{sid}" 的 narration 里含 LaTeX 命令，会原样显示')

        for key, val in (s.get("state") or {}).items():
            if key not in declared:
                gaps.append(f'步骤 "{sid}" 的 state 写了未声明张量 "{key}"')
            if key not in (s.get("writes") or []):
                warns.append(f'步骤 "{sid}" 的 state 里有 "{key}"，但它不在 writes[] 里')
            # One level deep, matching trace-model.js's port exactly. The two
            # must agree, and a stricter rule here alone would be a divergence
            # nobody could see: a trace is rejected before it is written, so a
            # JS-only author would never meet the case. (That the scan does not
            # recurse into a matrix is a known limit of both ports; no trace in
            # this lab puts a sentinel inside a rank-2 value, and L06's will
            # use top-level scalars for m and l. Worth fixing in both ports
            # together, not in one.)
            for item in (val if isinstance(val, list) else [val]):
                if isinstance(item, str) and item not in (NEG_INF, POS_INF, NAN):
                    gaps.append(f'步骤 "{sid}" 的 state["{key}"] 含未知哨兵字符串 "{item}"')

        for key in (s.get("regions") or {}):
            if not any(f"\\region{{{key}}}" in (formula.get(t) or "")
                       for t in ("sym", "idx", "num")):
                warns.append(f'步骤 "{sid}" 声明了 region "{key}" 但公式里没有引用')

        # `spans` is the view component's annotation: it must name a declared
        # tensor and stay inside that tensor's declared shape, or the stage
        # would silently outline nothing.
        for key, axes in (s.get("spans") or {}).items():
            if key not in declared:
                gaps.append(f'步骤 "{sid}" 的 spans 指向未声明张量 "{key}"')
                continue
            shape = trace["tensors"][key].get("shape") or []
            if len(axes) != len(shape):
                gaps.append(f'步骤 "{sid}" 的 spans["{key}"] 给了 {len(axes)} 个轴，'
                            f"但张量是 {len(shape)} 维")
                continue
            for ax, rng in enumerate(axes):
                lo, hi = rng
                if not (0 <= lo < hi <= shape[ax]):
                    gaps.append(f'步骤 "{sid}" 的 spans["{key}"] 第 {ax} 轴 '
                                f"[{lo}, {hi}) 超出 shape {shape[ax]}")

        # `flows` names layer boundaries, so both endpoints must be layers the
        # trace actually declares tensors in. A typo would draw a rail to
        # nowhere; the stage would rather fail loudly at author time.
        for fl in (s.get("flows") or []):
            for end in ("from", "to"):
                if fl.get(end) not in layer_ids(trace):
                    gaps.append(f'步骤 "{sid}" 的 flows 有一端的层 "{fl.get(end)}" '
                                f"不在 tensors[].at 用到的层里")
            if not isinstance(fl.get("elements"), int) or fl["elements"] <= 0:
                gaps.append(f'步骤 "{sid}" 的 flows 搬运量为 {fl.get("elements")!r}，'
                            f"应为正整数")

    # ---- meta.roofline: the points the sliders move -----------------------
    #
    # Every field here is re-derived from something else in the trace rather than
    # trusted, because the page's parameter story is two statements about the
    # same quantity: "读取 1024 → 256 个元素" and "算术强度 0.25 → 2.0". If those
    # could be written independently they could disagree, and the reader would
    # have no way to tell which one the picture was drawn from.
    roof = (trace.get("meta") or {}).get("roofline")
    if not isinstance(roof, dict):
        gaps.append("meta.roofline 缺失 —— 参数滑杆的 Roofline 图没有数据")
    else:
        required = ("peak_flops", "bandwidth", "ridge", "points",
                    "x_label", "y_label", "y_divisor", "y_unit", "note")
        for key in required:
            if key not in roof:
                gaps.append(f"meta.roofline 缺 {key}")
        peak, bw = roof.get("peak_flops"), roof.get("bandwidth")
        # `have_model` gates every check below that divides by, multiplies by or
        # compares against the ridge. A trace that is MISSING the ridge has
        # already been reported once, in the required-keys loop above; walking on
        # regardless would turn that report into a KeyError from the middle of
        # the function, which reads as a broken lint rather than as a broken trace.
        have_model = (isinstance(peak, (int, float)) and isinstance(bw, (int, float))
                      and bw > 0 and isinstance(roof.get("ridge"), (int, float)))
        yd = roof.get("y_divisor")
        if not isinstance(yd, (int, float)) or not (yd > 0) or not math.isfinite(yd):
            gaps.append(f'meta.roofline.y_divisor = {yd!r} 应为正数 —— '
                        f"纵轴刻度按它换算")
        else:
            # The label and the divisor are the two halves of one statement about
            # what the y axis means; naming a unit in the label and dividing the
            # ticks by something else would make the chart unreadable by exactly
            # the factor between them.
            unit = roof.get("y_unit")
            if not isinstance(unit, str) or not unit:
                gaps.append(f'meta.roofline.y_unit = {unit!r} 应为非空字符串')
            elif unit not in (roof.get("y_label") or ""):
                gaps.append(f'meta.roofline.y_label 里没有写明单位 "{unit}" —— '
                            f"刻度按 {yd:g} 换算，读者无从知道单位是什么")
        if have_model:
            if not math.isclose(roof["ridge"], peak / bw, rel_tol=1e-12):
                gaps.append(f'meta.roofline.ridge = {roof.get("ridge")!r} 应为 '
                            f"峰值算力 / 带宽 = {peak / bw!r}")
            # The hardware model must be the one `traffic()` priced the kernel
            # against, or the chart and the summary step would draw two ridges.
            if not (math.isclose(peak, PEAK_FLOPS, rel_tol=1e-12)
                    and math.isclose(bw, BW_BYTES, rel_tol=1e-12)):
                gaps.append(f"meta.roofline 的硬件模型 ({peak!r}, {bw!r}) 与生成器里的 "
                            f"V100 常数 ({PEAK_FLOPS!r}, {BW_BYTES!r}) 不一致")
        elif "ridge" in roof:
            gaps.append("meta.roofline 有 ridge 却没有像样的 peak_flops / bandwidth —— "
                        "平衡点无从核对")
        pts = roof.get("points")
        ids = [p.get("id") for p in pts] if isinstance(pts, list) else []
        # Everything below re-derives a point's numbers from the point's own
        # counts, so it needs the list to be the right three things first; a
        # missing key is reported once above rather than crashing the walk.
        if ids != ["naive", "tiled", "doc"]:
            gaps.append(f"meta.roofline.points 的 id 是 {ids!r}，应该是 "
                        f"['naive', 'tiled', 'doc']（顺序也参与渲染）")
        elif not have_model:
            pass
        else:
            cfg_ = trace["meta"]["config"]
            flops = 2 * cfg_["M"] * cfg_["N"] * cfg_["K"]
            by_id = {p["id"]: p for p in pts}
            for pid, p in by_id.items():
                bs, ai = p.get("bytes"), p.get("ai")
                pf = p.get("flops")
                if not isinstance(bs, int) or bs <= 0:
                    gaps.append(f'meta.roofline.points["{pid}"].bytes = {bs!r} 应为正整数')
                    continue
                if p.get("elements", 0) * 4 != bs:
                    gaps.append(f'meta.roofline.points["{pid}"] 的 elements 与 bytes 不符')
                # The point's intensity must be its OWN flops over its OWN bytes.
                # This is the rule that keeps "读取 1024 → 256 个元素" and "算术强度
                # 0.25 → 2.0" from becoming two unrelated claims.
                if not isinstance(pf, (int, float)):
                    gaps.append(f'meta.roofline.points["{pid}"].flops = {pf!r} 不是数')
                    continue
                if not math.isclose(ai, pf / bs, rel_tol=1e-12):
                    gaps.append(f'meta.roofline.points["{pid}"].ai = {ai!r} 与它自己的 '
                                f"flops / bytes = {pf} / {bs} = {pf / bs!r} 不一致 —— "
                                f"图上的点和它标称的访存量是两套说法")
                    continue
                want = "bandwidth" if ai < roof["ridge"] else "compute"
                if p.get("bound") != want:
                    gaps.append(f'meta.roofline.points["{pid}"].bound = '
                                f'{p.get("bound")!r}，而在 AI = {ai!r} 上起作用的是 '
                                f'"{want}" 的屋顶')
                want_ceiling = min(peak, bw * ai)
                if not math.isclose(p.get("ceiling", -1), want_ceiling, rel_tol=1e-12):
                    gaps.append(f'meta.roofline.points["{pid}"].ceiling = '
                                f'{p.get("ceiling")!r} 应为 min(峰值, 带宽×AI) = '
                                f"{want_ceiling!r}")
            # The naive and tiled points are THIS configuration's problem, so
            # their element counts are the traffic counts and their FLOPs are
            # 2MNK. The doc point is the tutorial's 128-cubed example and is
            # deliberately not tied to this trace's sizes.
            tr_ = trace["meta"]["traffic"]
            for pid, key in (("tiled", "tiled_reads"), ("naive", "naive_reads")):
                if by_id[pid].get("elements") != tr_[key]:
                    gaps.append(f'meta.roofline 的 {pid} 点用了 '
                                f'{by_id[pid].get("elements")!r} 个元素，'
                                f'而 meta.traffic.{key} = {tr_[key]!r}')
                if by_id[pid].get("flops") != flops:
                    gaps.append(f'meta.roofline 的 {pid} 点用了 '
                                f'{by_id[pid].get("flops")!r} FLOPs，而本配置是 '
                                f"2MNK = {flops}")
            if not math.isclose(tr_["tiled_ai"], tr_["tiled_ai_closed"], rel_tol=1e-12):
                gaps.append(f'meta.traffic 的两种算法不一致：计数式 {tr_["tiled_ai"]!r} vs '
                            f'闭式 B_MB_N/(2(B_M+B_N)) = {tr_["tiled_ai_closed"]!r}')
            if not math.isclose(by_id["tiled"]["ai"], tr_["tiled_ai"], rel_tol=1e-12):
                gaps.append(f'meta.roofline 的 tiled 点 AI = {by_id["tiled"]["ai"]!r} 与 '
                            f'meta.traffic.tiled_ai = {tr_["tiled_ai"]!r} 不一致')
            # Where the reading starts: the naive kernel has no tile at all, so
            # its point must be the same one on every configuration. That
            # stillness is the control the tiled point's movement is read
            # against, and it is checkable here because the naive intensity falls
            # out of the problem size alone: 2MNK FLOPs over 2MNK fp32 reads.
            naive_ai = flops / (2 * cfg_["M"] * cfg_["N"] * cfg_["K"] * 4)
            if not math.isclose(by_id["naive"]["ai"], naive_ai, rel_tol=1e-12):
                gaps.append(f'meta.roofline 的 naive 点 AI = {by_id["naive"]["ai"]!r} '
                            f"应恒为 2MNK/8MNK = {naive_ai!r}，与分块参数无关")

    infos.append(f'共 {len(trace["steps"])} 步 / {len(trace["graph"]["nodes"])} 节点 / '
                 f'{len(trace["graph"]["edges"])} 数据边')
    # The info line is built from whatever survived; a trace broken by a sabotage
    # must produce GAPs, not an exception from the reporting that follows them.
    if isinstance(roof, dict) and isinstance(roof.get("points"), list) \
            and isinstance(roof.get("ridge"), (int, float)):
        tiled_pt = next((p for p in roof["points"] if p.get("id") == "tiled"), None)
        if isinstance(tiled_pt, dict) and isinstance(tiled_pt.get("ceiling"), (int, float)):
            infos.append(f'Roofline：AI {tiled_pt["ai"]:.4g} FLOP/B（ridge '
                         f'{roof["ridge"]:.4g}）· 上界 '
                         f'{tiled_pt["ceiling"] / 1e12:.4g} TFLOP/s · '
                         f'{tiled_pt["bound"]}-bound')
    infos.append(f'state 写过的张量：{", ".join(sorted(written))}')
    return gaps, warns, infos


# Each entry breaks one lint rule on the real trace. Kept at module scope so the
# report can name how many were exercised.
SABOTAGE_CASES = {
    "read-but-never-written tensor undeclared": lambda t: t["tensors"].pop("A"),
    "read-but-never-written tensor loses its init": lambda t: t["tensors"]["B"].pop("init"),
    "state entry for an undeclared tensor": lambda t: t["steps"][2]["state"].update({"nope": 1.0}),
    "write of an undeclared tensor": lambda t: t["steps"][2]["writes"].append("ghost"),
    "binding missing a tier": lambda t: t["steps"][2]["bindings"].update({"OLD": {"sym": "a"}}),
    "binding collapsed to a 2-tuple": lambda t: t["steps"][2]["bindings"].update(
        {"NEW": {"num": "0.83"}}),
    "binding carrying a \\slot macro": lambda t: t["steps"][2]["bindings"].update(
        {"AV": {"sym": r"\slot{X}", "idx": "a", "num": "1"}}),
    "\\slot with no binding": lambda t: t["steps"][2]["formula"].update(
        {"num": t["steps"][2]["formula"]["num"] + r"\slot{GHOST}"}),
    "\\region with no description": lambda t: t["steps"][3]["formula"].update(
        {"sym": t["steps"][3]["formula"]["sym"] + r"\region{GHOST}{x}"}),
    "step with no graph node": lambda t: t["graph"]["nodes"].pop(0),
    "formula tier missing": lambda t: t["steps"][0]["formula"].pop("idx"),
    # d1 (index 2) carries scalar state, which is where the rule applies.
    "unknown sentinel string": lambda t: t["steps"][2]["state"].update({"acc": "-inf"}),
    "span names an undeclared tensor": lambda t: t["steps"][2].update({"spans": {"GHOST": [[0, 1]]}}),
    "span out of the tensor's shape": lambda t: t["steps"][2].update({"spans": {"A": [[0, 1], [0, 99]]}}),
    "span with the wrong rank": lambda t: t["steps"][2].update({"spans": {"A": [[0, 1]]}}),
    "flow naming an unknown layer": lambda t: t["steps"][2].update(
        {"flows": [{"from": "HBM", "to": "L2CACHE", "elements": 4}]}),
    "flow with a non-positive element count": lambda t: t["steps"][2].update(
        {"flows": [{"from": "HBM", "to": "REG", "elements": 0}]}),
    # --- meta.roofline: the parameter sliders' chart ------------------------
    # Tagged with a "roofline " prefix so a parity harness can tell which cases
    # belong to the new contract without hard-coding an index.
    "roofline block missing": lambda t: t["meta"].pop("roofline"),
    "roofline missing the ridge point": lambda t: t["meta"]["roofline"].pop("ridge"),
    "roofline ridge not peak/bandwidth": lambda t: t["meta"]["roofline"].update({"ridge": 5.0}),
    "roofline points in the wrong order": lambda t: t["meta"]["roofline"].update(
        {"points": list(reversed(t["meta"]["roofline"]["points"]))}),
    "roofline point with no counterpart in traffic": lambda t: _roof_point(
        t, "tiled").update({"elements": 7}),
    "roofline ai disagreeing with its own bytes": lambda t: _roof_point(
        t, "tiled").update({"ai": 99.0}),
    "roofline ceiling not min(peak, bandwidth*ai)": lambda t: _roof_point(
        t, "naive").update({"ceiling": 1.0}),
    "roofline naming the wrong roof": lambda t: _roof_point(
        t, "tiled").update({"bound": "compute"}),
    "roofline hardware model diverging from the generator's": lambda t: t["meta"][
        "roofline"].update({"peak_flops": 1.0}),
    "roofline y-axis divisor missing": lambda t: t["meta"]["roofline"].pop("y_divisor"),
    "roofline y-axis unit not named in the label": lambda t: t["meta"]["roofline"].update(
        {"y_label": "可达算力上界"}),
}


def _roof_point(trace, pid):
    """The one point in `meta.roofline.points` with this id.

    Addressed by id rather than by index so a case keeps pointing at the point it
    means if the list order ever changes -- and so the "points in the wrong
    order" case above cannot silently retarget the others.
    """
    return next(p for p in trace["meta"]["roofline"]["points"] if p["id"] == pid)


def sabotage_checks(trace):
    """Prove the lint is not a function that always returns zero.

    A clean report only means something if the same lint fails on a trace that
    is actually broken, so every rule is exercised against a deliberately
    broken copy of the real trace. Returns a list of failures (empty == live).
    """
    import copy

    failures = []

    def broken(mutate):
        t = copy.deepcopy(trace)
        mutate(t)
        return lint(t)[0]

    for name, mutate in SABOTAGE_CASES.items():
        try:
            gaps = broken(mutate)
        except Exception as exc:  # a mutation that does not even lint is a failure
            failures.append(f"{name}: lint raised {exc!r}")
            continue
        if not gaps:
            failures.append(f"{name}: broke the trace but lint reported 0 gaps")
    return failures


# The non-finite tagging `--js-lint` needs. Nothing in an L01 trace is
# non-finite -- no sentinels are used at all -- but two sabotages put a bare
# `Infinity` in, and `json.dumps` writes that as the token `Infinity`, which is
# legal JavaScript and ILLEGAL JSON: a payload carrying one could not be parsed
# by the probe at all, and the harness would fail before either lint ran. The
# tags keep the payload valid while delivering the same value; the probe decodes
# `{"__nonfinite__": "inf"}` back into `Infinity`. Same device, and the same
# reason, as flash_attention.py's -- that is where the case needing it appeared.
_NONFINITE_TAGS = {float("nan"): "nan", float("inf"): "inf", float("-inf"): "-inf"}


def _tag_nonfinite(value):
    if isinstance(value, float):
        for probe, tag in _NONFINITE_TAGS.items():
            # `probe == value` is False for every NaN, so identity is the only
            # test that works for the NaN case.
            if probe != probe:
                if value != value:
                    return {"__nonfinite__": tag}
            elif value == probe and math.copysign(1, value) == math.copysign(1, probe):
                return {"__nonfinite__": tag}
        return value
    if isinstance(value, dict):
        return {k: _tag_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tag_nonfinite(v) for v in value]
    return value


def _payload_json(payload):
    return json.dumps(_tag_nonfinite(payload), ensure_ascii=False)


# ------------------------------------------------------------------------ main
SOURCE = ("docs/guides/模块二-CUDA编程与算子优化/4.1-CUDA GEMM算子性能优化.md"
          " + 模块一-前置知识/第2章-数学基础.md §3")
SET = "gemm-tiling"

# The parameter grid the page's sliders walk. A full product of {2, 4, 8} per
# parameter, filtered to combinations that divide M=N=K=8 -- all 27 of them,
# which is why 8 is the problem size the lab picked. A tile that did not divide
# would leave a partial block, and tail handling is L06's subject, not L01's
# (see the module docstring); the filter is here so the grid stays correct if
# the problem size ever moves.
#
# A full product rather than a hand-picked list, because that is what makes the
# sliders three independent controls instead of three positions of one knob: a
# reader can hold B_K fixed and watch B_M move, which is exactly the comparison
# the tutorial's §4.3 closed form is about.
GRID_VALUES = (2, 4, 8)


def grid():
    out = []
    for bm in GRID_VALUES:
        for bn in GRID_VALUES:
            for bk in GRID_VALUES:
                if M % bm or N % bn or K % bk:
                    continue
                out.append((bm, bn, bk))
    return out


def cfg_id(bm, bn, bk):
    return f"{SET}-BM{bm}-BN{bn}-BK{bk}"


def label(bm, bn, bk):
    return f"B_M={bm} · B_N={bn} · B_K={bk}"


def main():
    only_json = "--js-lint" in sys.argv
    real_stdout = sys.stdout
    if only_json:
        # Every human-readable line goes to stderr so stdout carries the JSON
        # payload and nothing else -- the parity harness pipes it straight into
        # `node`. A stray progress line would make the payload unparseable, which
        # reads as a harness bug rather than as this mode's own contract.
        sys.stdout = sys.stderr

    fails, written, built = [], [], {}
    for (bm, bn, bk) in grid():
        name = cfg_id(bm, bn, bk)
        trace = build_trace(f"GEMM 分块与数据搬运（M=N=K=8，{label(bm, bn, bk)}）",
                            SOURCE, bm, bn, bk)
        gaps, warns, infos = lint(trace)
        # Every configuration's control group runs, not just the default's: the
        # grid is 27 traces, and a rule that only bit at B_M=4 would otherwise be
        # reported live on the strength of a trace it never saw.
        sabotages = sabotage_checks(trace)

        tr = trace["meta"]["traffic"]
        roof = trace["meta"]["roofline"]
        tiled_pt = next(p for p in roof["points"] if p["id"] == "tiled")
        print(f"\n{name}: {len(trace['steps'])} 步 / {len(trace['graph']['nodes'])} 节点, "
              f"{len(json.dumps(trace, ensure_ascii=False))} bytes")
        print(f"  分块 {bm}×{bn}×{bk} · {tr['blocks']} 个块 · HBM 读取 "
              f"{tr['naive_reads']} → {tr['tiled_reads']} 个元素"
              f"（复用 {tr['reuse']:.4g}×）")
        print(f"  AI {tr['naive_ai']:.4g} → {tr['tiled_ai']:.4g} FLOP/B"
              f"（闭式 {tr['tiled_ai_closed']:.4g}）· ridge {roof['ridge']:.4g} · "
              f"{tiled_pt['bound']}-bound，上界 {tiled_pt['ceiling'] / 1e12:.4g} TFLOP/s")
        print(f"  {trace['meta']['reference']}")
        for line in infos:
            print(f"  info  {line}")
        for line in warns:
            print(f"  warn  {line}")
        for line in gaps:
            print(f"  GAP   {line}")
            fails.append(f"{name}: {line}")
        if sabotages:
            fails.append(f"{name}: lint 对照组失效")
            for line in sabotages:
                print(f"  LINT-DEAD  {line}")
        else:
            print(f"  lint: {len(gaps)} gap / {len(warns)} warn；"
                  f"对照 {len(SABOTAGE_CASES)} 种破坏全部被抓到")
        built[name] = trace

    # The cross-configuration claim a single-trace lint cannot make: the sliders
    # must actually MOVE something. A grid in which every tile produced the same
    # traffic would lint perfectly and teach nothing, so it is asserted here --
    # and the naive point's stillness is asserted alongside it, because that is
    # what makes the tiled point's movement readable as a comparison rather than
    # as a chart that redraws itself arbitrarily.
    if not only_json and not fails:
        tiled_ais = {n: next(p for p in t["meta"]["roofline"]["points"]
                             if p["id"] == "tiled")["ai"] for n, t in built.items()}
        naive_ais = {next(p for p in t["meta"]["roofline"]["points"]
                          if p["id"] == "naive")["ai"] for t in built.values()}
        if len(set(tiled_ais.values())) < 2:
            fails.append("整张网格的 tiled 点 AI 完全相同 —— 滑杆什么也移动不了")
            print("  GAP   滑杆移动不了 Roofline 上的点：整张网格 AI 相同")
        if len(naive_ais) != 1:
            fails.append(f"朴素点的 AI 随分块参数变了：{sorted(naive_ais)}")
            print(f"  GAP   朴素点本该与分块参数无关，却有 {len(naive_ais)} 个值")

        # ---- what B_K does and does not move, asserted in both directions ----
        #
        # This is the one place the lab has to be careful, because the honest
        # answer is asymmetric. Under this traffic model each output block walks
        # the whole K range, so chunking K changes only how the loads are
        # GROUPED: total element traffic is `MNK(1/B_M + 1/B_N)` and `B_K`
        # cancels — which is exactly why the tutorial's §4.3 closed form,
        # `B_M·B_N/(2(B_M+B_N))`, has no B_K in it at all.
        #
        # What B_K does move is the number of load operations (halving B_K
        # doubles them) and the SMEM tile `B_M·B_K + B_K·B_N` — the quantity
        # §4.2 says is the reason to split K in the first place. Both are
        # published, both are shown, and both are asserted here.
        #
        # Asserting the INVARIANCE as well as the sensitivity is the point: a
        # future edit that quietly made B_K change the traffic would break the
        # closed form the page prints, and one that made it change nothing at all
        # would leave a dead slider. Neither can pass this.
        by_params = {}
        for name, t in built.items():
            c = t["meta"]["config"]
            key = (c["B_M"], c["B_N"])
            by_params.setdefault(key, []).append(
                (c["B_K"], t["meta"]["traffic"]["tiled_reads"],
                 t["meta"]["traffic"]["tiled_ai"],
                 t["meta"]["traffic"]["load_steps"],
                 t["meta"]["traffic"]["smem_tile"]))
        bad_inv, bad_sens = [], []
        for key, rows in sorted(by_params.items()):
            rows.sort()
            if len({r[1] for r in rows}) != 1 or len({r[2] for r in rows}) != 1:
                bad_inv.append((key, [(r[0], r[1], r[2]) for r in rows]))
            # Across the B_K values, both the load count and the SMEM tile must
            # actually differ -- otherwise the B_K slider would be inert.
            if len(rows) > 1 and (len({r[3] for r in rows}) < 2 or len({r[4] for r in rows}) < 2):
                bad_sens.append((key, [(r[0], r[3], r[4]) for r in rows]))
        if bad_inv:
            fails.append(f"B_K 改变了元素搬运量 —— 与 §4.3 闭式矛盾：{bad_inv[:1]}")
            print(f"  GAP   B_K 只该改变分组，不该改变元素量，却有 {len(bad_inv)} 组在变")
        if bad_sens:
            fails.append(f"B_K 什么也没改变（滑杆是死的）：{bad_sens[:1]}")
            print(f"  GAP   B_K 滑杆没有可移动的量：{len(bad_sens)} 组")
        if not fails:
            bk_rows = sorted({(c["B_K"], t["meta"]["traffic"]["load_steps"],
                               t["meta"]["traffic"]["smem_tile"])
                              for t in built.values()
                              for c in [t["meta"]["config"]]})
            print(f"\n滑杆确实有东西可移动：B_M / B_N 把 tiled 点 AI 从 "
                  f"{min(tiled_ais.values()):.4g} 拉到 {max(tiled_ais.values()):.4g} "
                  f"FLOP/B（{len(set(tiled_ais.values()))} 个不同值）；对照的朴素点恒为 "
                  f"{naive_ais.pop():.4g}")
            print("  B_K 的作用相反且同样已验证：元素搬运量与 AI 对它完全不变"
                  "（§4.3 的闭式里没有 B_K），但它把载入次数与 SMEM tile 的大小"
                  f"按 1/B_K 缩放 —— {len({r[0] for r in bk_rows})} 个 B_K 值给出 "
                  f"{sorted({r[1] for r in bk_rows})} 种载入次数")

    if only_json:
        # `base` names which trace the sabotages were cut from, so the JS side can
        # tell "the lint missed this" from "the mutation changed nothing".
        base_name = cfg_id(*DEFAULT_TILE)
        base = built[base_name]
        payload = {"traces": built, "base": base_name, "sabotages": {}}
        for name, mutate in SABOTAGE_CASES.items():
            broken = copy.deepcopy(base)
            try:
                mutate(broken)
            except Exception as exc:
                payload["sabotages"][name] = {"error": repr(exc)}
                continue
            payload["sabotages"][name] = {
                "trace": broken,
                "changed": json.dumps(broken, sort_keys=True)
                           != json.dumps(base, sort_keys=True),
            }
        real_stdout.write(_payload_json(payload))
        return 0

    if fails:
        print("\nlint / 断言未通过，拒绝写文件：", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        return 1

    for name, trace in built.items():
        out = HERE / f"{name}.json"
        out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n")
        written.append(name)

    manifest = {
        "set": SET,
        "lab": "L01",
        "default": cfg_id(*DEFAULT_TILE),
        "params": {
            "B_M": sorted({bm for (bm, _, _) in grid()}),
            "B_N": sorted({bn for (_, bn, _) in grid()}),
            "B_K": sorted({bk for (_, _, bk) in grid()}),
        },
        "traces": written,
        "labels": {cfg_id(bm, bn, bk): label(bm, bn, bk) for (bm, bn, bk) in grid()},
    }
    (HERE / f"{SET}.manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"{SET}.manifest.json: {len(written)} 个配置，默认 {manifest['default']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
