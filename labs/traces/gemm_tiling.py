#!/usr/bin/env python3
"""Trace generator for L01 · GEMM：索引、分块与访存.

Emits one JSON file:

    labs/traces/gemm-tiling.json   M=N=K=8, B_M=B_N=B_K=4 -> 32 steps

The lab this belongs to (L01) needs a trace, and a trace does not need a lab —
it is data, the lab is rendering. So the generator lives here, next to the L00
one, and the L01 page consumes the JSON.

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

Every number quoted in a narration or a formula is computed from M/N/K here —
the HBM element traffic (2MNK naive against the tiled count), the arithmetic
intensity, every accumulator. Only the problem inputs A and B are literals,
which is the same split online_softmax.py makes for its input vector.

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
BM, BN, BK = 4, 4, 4


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


def run_tiled(A, B):
    """The same C, block by block, in the same k order.

    Structure follows the block-tiling kernel in the tutorial (section 4.5),
    with the thread-level fragment from section 5.2 made explicit: per k the
    thread holds one column of A_smem and one row of B_smem in registers and
    updates the whole TM x TN tile by outer product. The inner k loop is a
    single replay step, so a frame records the last k's fragments and the
    accumulator after all B_K of them.
    """
    C = [[0.0] * N for _ in range(M)]
    frames = []
    for bi in range(M // BM):
        r0 = bi * BM
        for bj in range(N // BN):
            c0 = bj * BN
            c_frag = [[0.0] * BN for _ in range(BM)]
            for bk in range(K // BK):
                k0 = bk * BK
                A_smem = [[float(A[r0 + p][k0 + q]) for q in range(BK)]
                          for p in range(BM)]
                B_smem = [[float(B[k0 + p][c0 + q]) for q in range(BN)]
                          for p in range(BK)]
                a_frag, b_frag = None, None
                for k in range(BK):
                    a_frag = [A_smem[p][k] for p in range(BM)]
                    b_frag = [B_smem[k][q] for q in range(BN)]
                    for p in range(BM):
                        for q in range(BN):
                            c_frag[p][q] = c_frag[p][q] + a_frag[p] * b_frag[q]
                frames.append(dict(
                    bi=bi, bj=bj, bk=bk, r0=r0, c0=c0, k0=k0, last=(bk == K // BK - 1),
                    A_smem=[[row[q] for q in range(BK)] for row in A_smem],
                    B_smem=[[row[q] for q in range(BN)] for row in B_smem],
                    a_frag=a_frag, b_frag=b_frag,
                    c_frag=[[c_frag[p][q] for q in range(BN)] for p in range(BM)],
                ))
            for p in range(BM):
                for q in range(BN):
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


def traffic(M, N, K, BM, BN, BK):
    """HBM element traffic for the naive and the tiled kernel.

    Naive: every one of the MN outputs reads its own K-row of A and K-column of
    B, so 2MNK element reads. Tiled: each of the (M/BM)(N/BN)(K/BK) block
    iterations moves BM*BK + BK*BN elements, and the reuse is exactly the block
    area divided by the tile perimeter. These are the numbers the narration and
    the page quote, computed here so they cannot drift from the algorithm.
    """
    blocks = (M // BM) * (N // BN) * (K // BK)
    naive_reads = 2 * M * N * K
    tiled_reads = blocks * (BM * BK + BK * BN)
    return dict(
        blocks=blocks,
        naive_reads=naive_reads,
        tiled_reads=tiled_reads,
        reuse=naive_reads / tiled_reads,
        naive_flops=2 * M * N * K,
        naive_fma=M * N * K,
        naive_bytes=naive_reads * 4,
        tiled_bytes=tiled_reads * 4,
        naive_ai=(2 * M * N * K) / (naive_reads * 4),
        tiled_ai=(2 * M * N * K) / (tiled_reads * 4),
        # V100's ridge point: 15.7 TFLOP/s / 900 GB/s. The toy tile stays far
        # below it, which is the honest end to the story rather than a
        # contradiction of the tutorial's 128x128 example.
        ridge=15.7e12 / 900e9,
        doc_ai=(128 * 128) / (2 * (128 + 128)),
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


def build_trace(title, source):
    A, B = np.array(A_DATA, dtype=np.float64), np.array(B_DATA, dtype=np.float64)
    C_naive = run_naive(A, B)
    C_tiled, frames = run_tiled(A, B)

    # The load-bearing assertion: tiling moves data differently and must not
    # change the arithmetic. See the module docstring for why this is exact
    # rather than approximate.
    if not np.array_equal(np.array(C_naive), np.array(C_tiled)):
        bad = [(i, j, C_naive[i][j], C_tiled[i][j])
               for i in range(M) for j in range(N)
               if C_naive[i][j] != C_tiled[i][j]]
        raise AssertionError(f"tiled != naive at {len(bad)} entries, first {bad[0]}")

    _, worst = check_against_torch(C_naive)
    tr = traffic(M, N, K, BM, BN, BK)

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
         b("BM", "B_M", str(BM), str(BM)),
         b("BN", "B_N", str(BN), str(BN)),
         b("BK", "B_K", str(BK), str(BK)),
         b("NB", r"(M/B_M)(N/B_N)(K/B_K)",
           f"({M}/{BM})({N}/{BN})({K}/{BK})", str(tr["blocks"]))],
        [], ["C"], {"C": mat(np.zeros((M, N)))},
        f"A 和 B 已经在 HBM 里，C 先清零。{M}×{N}×{K} 的矩阵按 {BM}×{BN}×{BK} 分块，"
        f"C 是 {M // BM}×{N // BN} 个块，每个块沿 K 要迭代 {K // BK} 次。",
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
        bi, bj, bk = f["bi"], f["bj"], f["bk"]
        r0, c0, k0 = f["r0"], f["c0"], f["k0"]
        tag = f"{bi}{bj}{bk}"
        moved = (BM * BK + BK * BN) * 4

        steps.append(step(
            f"ld{tag}", f"块({bi},{bj})·K{bk + 1}：A、B 的 tile 载入 SMEM",
            "comm", "load", "分块搬运",
            {
                "sym": r"A^{\mathrm{sm}} \leftarrow \mathrm{HBM}[A]\left[\slot{R0}:\slot{R1},\ \slot{K0}:\slot{K1}\right],"
                       r"\qquad B^{\mathrm{sm}} \leftarrow \mathrm{HBM}[B]\left[\slot{K0}:\slot{K1},\ \slot{C0}:\slot{C1}\right]",
                "idx": r"A^{\mathrm{sm}} \leftarrow \mathrm{HBM}[A]\left[\slot{R0}:\slot{R1},\ \slot{K0}:\slot{K1}\right],"
                       r"\qquad B^{\mathrm{sm}} \leftarrow \mathrm{HBM}[B]\left[\slot{K0}:\slot{K1},\ \slot{C0}:\slot{C1}\right]",
                "num": r"\region{MOVE}{A^{\mathrm{sm}},\; B^{\mathrm{sm}} \leftarrow \slot{BYTES}\ \mathrm{B}}",
            },
            [b("R0", r"(b_i)B_M", str(r0), str(r0)),
             b("R1", r"(b_i+1)B_M", str(r0 + BM), str(r0 + BM)),
             b("K0", r"(b_k)B_K", str(k0), str(k0)),
             b("K1", r"(b_k+1)B_K", str(k0 + BK), str(k0 + BK)),
             b("C0", r"(b_j)B_N", str(c0), str(c0)),
             b("C1", r"(b_j+1)B_N", str(c0 + BN), str(c0 + BN)),
             b("BYTES", r"4(B_MB_K + B_KB_N)",
               f"4\\times({BM * BK} + {BK * BN})", str(moved))],
            ["A", "B"], ["A_smem", "B_smem"],
            {"A_smem": mat(np.array(f["A_smem"])),
             "B_smem": mat(np.array(f["B_smem"]))},
            f"块({bi},{bj}) 的第 {bk + 1} 个 K 迭代：A 的 [{r0}:{r0 + BM}, {k0}:{k0 + BK}] 和 "
            f"B 的 [{k0}:{k0 + BK}, {c0}:{c0 + BN}] 从 HBM 搬进 SMEM，"
            f"共 {BM * BK + BK * BN} 个 float（{moved} 字节）。"
            f"这一对 tile 接下来会被块内 {BM * BN} 个输出点复用。",
            {"MOVE": f"这次搬运把 A、B 各一个 {BM}×{BK} 的 tile 送进 SMEM，"
                     f"合计 {BM * BK + BK * BN} 个 float，即 {moved} 字节。"
                     f"朴素实现算这 {BM * BN} 个输出点要读 {BM * BN * 2 * K} 个元素。"},
            {"A": [[r0, r0 + BM], [k0, k0 + BK]],
             "B": [[k0, k0 + BK], [c0, c0 + BN]]},
            flows=[{"from": "HBM", "to": "SMEM", "elements": BM * BK + BK * BN,
                    "note": f"A 的 {BM}×{BK} tile 与 B 的 {BK}×{BN} tile，一次搬完"}],
        ))

        steps.append(step(
            f"mma{tag}", f"块({bi},{bj})·K{bk + 1}：SMEM → REG 外积累加",
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
            [b("P", "p", str(BK - 1), str(BK - 1)),
             b("P0", "0", "0", "0"),
             b("P1", "B_K-1", str(BK - 1), str(BK - 1)),
             b("PL", "B_K-1", str(BK - 1), str(BK - 1)),
             b("AV", r"A^{\mathrm{sm}}[:,B_K-1]",
               f"A^{{\\mathrm{{sm}}}}[:,{BK - 1}]", vec_tex(f["a_frag"])),
             b("BV", r"B^{\mathrm{sm}}[B_K-1,:]",
               f"B^{{\\mathrm{{sm}}}}[{BK - 1},:]", vec_tex(f["b_frag"])),
             # The outer product reads the same at every tier -- what the num
             # tier substitutes is the operands, not this line. The accumulator
             # it produces is rendered by the stage as a REG-resident block.
             b("FRAG", r"C^{\mathrm{frag}} \mathrel{+}= a^{\mathrm{frag}}\otimes b^{\mathrm{frag}}",
               r"C^{\mathrm{frag}} \mathrel{+}= a^{\mathrm{frag}}\otimes b^{\mathrm{frag}}",
               r"C^{\mathrm{frag}} \mathrel{+}= a^{\mathrm{frag}}\otimes b^{\mathrm{frag}}")],
            ["A_smem", "B_smem", "c_frag"], ["a_frag", "b_frag", "c_frag"],
            {"a_frag": vec(f["a_frag"]), "b_frag": vec(f["b_frag"]),
             "c_frag": mat(np.array(f["c_frag"]))},
            f"线程把 A_smem 的第 {BK} 列 {BM} 个数、B_smem 的第 {BK} 行 {BN} 个数搬进寄存器，"
            f"然后做外积。本块的 {BK} 个 k 里，每轮读 {BM} + {BN} = {BM + BN} 个 float，"
            f"完成 {BM * BN} 次 FMA —— 这就是 SMEM 这一级买到的复用。",
            {"OUTER": f"每轮 SMEM 读 {BM + BN} 个 float 做 {BM * BN} 次 FMA，"
                      f"比例 {BM * BN} / {BM + BN} = {BM * BN // (BM + BN)} : 1，"
                      f"而朴素实现是每读 2 个 float 做 1 次 FMA。"},
            {"A_smem": [[0, BM], [BK - 1, BK]],
             "B_smem": [[BK - 1, BK], [0, BN]]},
            # Two flows, not three, and the count is the point: a_frag and
            # b_frag crossed SMEM -> REG (BM + BN elements). c_frag did not --
            # it is an accumulator that stays in REG -- which is why deriving
            # the flow from `writes` would be wrong here.
            flows=[{"from": "SMEM", "to": "REG", "elements": BM + BN,
                    "note": f"A_smem 的 {BM} 个 + B_smem 的 {BN} 个，共 {BM + BN} 个进寄存器"},
                   {"from": "REG", "to": "REG", "elements": BM * BN,
                    "note": f"寄存器内的 {BM * BN} 次 FMA，c_frag 原地累加，不跨层"}],
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
                 b("R1", r"(b_i+1)B_M", str(r0 + BM), str(r0 + BM)),
                 b("C0", r"(b_j)B_N", str(c0), str(c0)),
                 b("C1", r"(b_j+1)B_N", str(c0 + BN), str(c0 + BN)),
                 b("SI", "i", str(r0), str(r0)),
                 b("SJ", "j", str(c0), str(c0)),
                 b("SV", r"C_{ij}", f"C_{{{r0}{c0}}}", fmt(C_tiled[r0][c0]))],
                ["c_frag"], ["C"],
                {"C": mat(np.array(f["C_after"]))},
                f"块({bi},{bj}) 的 {BM}×{BN} 个结果从寄存器写回 HBM。这 {BM * BN} 个输出点"
                f"一共从 HBM 读了 {(K // BK) * (BM * BK + BK * BN)} 个元素，"
                f"朴素实现要读 {BM * BN * 2 * K} 个。算出来的值两者逐位相同。",
                spans={"C": [[r0, r0 + BM], [c0, c0 + BN]]},
                flows=[{"from": "REG", "to": "HBM", "elements": BM * BN,
                        "note": f"{BM}×{BN} 个输出点从寄存器写回 HBM"}],
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
           f"{tr['blocks']}\\times({BM * BK}+{BK * BN})", str(tr["tiled_reads"])),
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
        f"按 {BM}×{BN}×{BK} 分块只要 {tr['tiled_reads']} 个，少了 {tr['reuse']:.0f} 倍。"
        f"算术强度从 {disp(tr['naive_ai'])} 抬到 {disp(tr['tiled_ai'])} FLOP/Byte —— "
        f"方向和教程里说的一致，但离 V100 的平衡点 {disp(tr['ridge'])} 还差得远："
        f"这个 8×8 的例子是为了把搬运过程看清楚，真要把算力打满，tile 得大到 "
        f"128×128（算术强度 {disp(tr['doc_ai'])}）。",
        {"RIDGE": f"平衡点 = 峰值算力 / 峰值带宽。V100 上约 {disp(tr['ridge'])} FLOP/Byte，"
                  f"超过它才由算力而非带宽决定快慢。"},
        flows=[{"from": "HBM", "to": "SMEM", "elements": tr["tiled_reads"],
                "note": f"整趟回放的 HBM → SMEM 总搬运量：{tr['tiled_reads']} 个元素"},
               {"from": "SMEM", "to": "REG",
                "elements": tr["blocks"] * (BM + BN) * BK,
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
    for bi in range(M // BM):
        for bj in range(N // BN):
            for bk in range(K // BK):
                tag = f"{bi}{bj}{bk}"
                edges.append({"from": "init", "to": f"ld{tag}", "tensor": "A"})
                edges.append({"from": f"ld{tag}", "to": f"mma{tag}", "tensor": "A_smem"})
                if bk > 0:
                    # The K loop carries the accumulator across block iterations.
                    edges.append({"from": f"mma{bi}{bj}{bk - 1}", "to": f"mma{tag}",
                                  "tensor": "c_frag"})
            edges.append({"from": f"mma{bi}{bj}{K // BK - 1}", "to": f"st{bi}{bj}",
                          "tensor": "c_frag"})

    return {
        "meta": {
            "lab": "L01",
            "title": title,
            "source": source,
            "config": {"M": M, "N": N, "K": K, "B_M": BM, "B_N": BN, "B_K": BK},
            "reference": (
                f"C = A·B，torch float64 matmul 对拍最大绝对偏差 {worst:.3g}"
                f"（相对容差 {_CHECK_REL_TOL:g}）；朴素 k 序循环与 "
                f"B_M=B_N=B_K={BM} 分块逐位相同（np.array_equal）；"
                f"HBM 读取 {tr['naive_reads']} → {tr['tiled_reads']} 个元素"
            ),
            "traffic": tr,
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
            "A_smem": {"shape": [BM, BK], "dtype": "fp32", "at": "SMEM", "role": "staging",
                       "note": "A 的当前 tile，块内所有线程共享"},
            "B_smem": {"shape": [BK, BN], "dtype": "fp32", "at": "SMEM", "role": "staging",
                       "note": "B 的当前 tile，块内所有线程共享"},
            "a_frag": {"shape": [BM], "dtype": "fp32", "at": "REG", "role": "staging",
                       "note": "线程私有：A_smem 的一列"},
            "b_frag": {"shape": [BN], "dtype": "fp32", "at": "REG", "role": "staging",
                       "note": "线程私有：B_smem 的一行"},
            "c_frag": {"shape": [BM, BN], "dtype": "fp32", "at": "REG", "role": "state",
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

    infos.append(f'共 {len(trace["steps"])} 步 / {len(trace["graph"]["nodes"])} 节点 / '
                 f'{len(trace["graph"]["edges"])} 数据边')
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
}


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


# ------------------------------------------------------------------------ main
SOURCE = ("docs/guides/模块二-CUDA编程与算子优化/4.1-CUDA GEMM算子性能优化.md"
          " + 模块一-前置知识/第2章-数学基础.md §3")
FILENAME = "gemm-tiling.json"


def main():
    trace = build_trace("GEMM 分块与数据搬运（M=N=K=8，B_M=B_N=B_K=4）", SOURCE)
    gaps, warns, infos = lint(trace)

    # Lint and the sabotage control group run BEFORE the file is touched: a
    # trace that does not pass must not be able to reach labs/.
    sabotages = sabotage_checks(trace)
    for line in gaps:
        print(f"  GAP   {line}")
    if gaps or sabotages:
        for line in sabotages:
            print(f"  LINT-DEAD  {line}")
        print("lint or its control group failed -- nothing written", file=sys.stderr)
        return 1

    out = HERE / FILENAME
    out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n")

    print(f"{out.name}: {len(trace['steps'])} 步 / {len(trace['graph']['nodes'])} 节点, "
          f"{out.stat().st_size} bytes")
    print(f"  {trace['meta']['reference']}")
    for line in infos:
        print(f"  info  {line}")
    for line in warns:
        print(f"  warn  {line}")
    print(f"  lint: {len(gaps)} gap / {len(warns)} warn")
    print(f"  lint 对照组：{len(SABOTAGE_CASES)} 种破坏全部被抓到"
          f"（证明上面的 0 不是 lint 永远报 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
