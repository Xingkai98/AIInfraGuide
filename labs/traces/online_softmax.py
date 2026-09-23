#!/usr/bin/env python3
"""Trace generator for L00 · Online Softmax.

Emits one JSON file per configuration of `(N, Bc, T)`, plus a manifest naming
the set:

    labs/traces/online-softmax-N<N>-B<Bc>-T<T>.json   one full replay each
    labs/traces/online-softmax.manifest.json          the set the page inlines

The page's parameter sliders select among these; every configuration the
sliders can reach has a trace here, and the manifest is how the page knows that
(see the `traces:` marker in `scripts/build-labs.mjs`). The generator writes the
manifest itself so a slider position can never point at a trace that was never
generated.

Three things this script produces beyond the raw replay, all of them real
arithmetic rather than narration:

  * **The correction factor, amplified.** Every block that raises `m` rescales
    the running `(l, acc)` by `corr = exp(m_old - m_new)`. Alongside the correct
    recursion this script runs a *shadow* recursion that omits the rescale — the
    classic bug — and records, per step, what each variant currently reports as
    the answer plus how far apart they are. `meta.correction` carries the final
    deviation if you never rescale at all.

  * **A three-way comparison.** Naive softmax (no max subtraction, fed logits
    shifted by +1000 — a shift softmax is invariant to, so the answer must not
    change and yet the naive implementation overflows), the three-pass safe
    version, and online. Each is run for real on the same input and indexed
    against a shared axis: elements read from HBM. The naive run is *asserted*
    to produce a non-finite result — if it ever stopped overflowing, the
    comparison would lose its point and this script would refuse to write.

  * **Progress.** Every step records `k`, how many elements of `x` have been
    read from HBM by the end of that step. That is the axis the comparison
    panel aligns the three methods on.

Every number here is computed, not typed in: the numpy recursion is asserted
step by step against an independent torch implementation (which derives each
statistic from the *prefix* of the input rather than from the previous block, so
a mistake in the recursion cannot cancel out), and the final output is asserted
against `torch.softmax`. The script refuses to write a file if those assertions,
the self-lint, or the lint's control group fail.

Contract notes (see `docs/plans/interactive-labs.md` §一 "轨迹数据模型"):

  * `tensors[].init` is mandatory for anything read but never written. `x` is
    read by the first block load and written by no step; without `init`,
    render(trace, 0) cannot show the input at all.
  * +/-inf and NaN are emitted as the strings "-\\infty" / "+\\infty" / "NaN".
    Bare `-Infinity` is not legal JSON. The sentinels appear in `state`,
    `corr`, and `compare` values; formula bindings are LaTeX fragments and
    spell `-\\infty` themselves.
  * `step.state` is "every tensor value that changed during this step, in full"
    -- never a delta. That is what makes arbitrary cursor jumps a pure function
    of (trace, cursor) with no undo log.

The script is deterministic (fixed inputs, no sampling), which is what lets CI
re-run it and require the committed JSON to be reproduced byte for byte — see
.github/workflows/trace-checks.yml.

Run: python3 labs/traces/online_softmax.py
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
# strings, not a local convention -- the engine parses exactly these.
NEG_INF = r"-\infty"
POS_INF = r"+\infty"
NAN = "NaN"

# The design doc's shared example model: every lab replays the same vector, so a
# reader's sense of scale accumulates across the course. N=4 slices the front of
# the same vector rather than picking new numbers.
SHARED_X = [0.83, -0.20, 1.40, 0.35, 0.12, -1.05, 2.10, 0.44]

# The naive path's demonstration. Softmax is invariant to adding a constant to
# every logit, so a +1000 shift must not change the answer -- but an
# implementation that exponentiates first blows past float64's range (exp(709)
# is already the ceiling) and returns inf, then inf/inf = NaN.
#
# This is not an artificial pathology. The shift stands in for "the logits are
# large", which is the normal state of affairs in a real attention head: with
# the shared 8-element vector the naive path happens to work, and the shift is
# what moves it into the regime a real one operates in. The trace publishes
# BOTH runs so the claim "same code, only the logit magnitude differs" is a pair
# of measured numbers rather than an assertion.
NAIVE_SHIFT = 1000.0


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
    literal backslashes if it leaked into a narration string (the lint checks
    for that)."""
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
    if v is None:
        return None
    f = float(v)
    if math.isnan(f):
        return NAN
    if math.isinf(f):
        return NEG_INF if f < 0 else POS_INF
    return f


def state_list(values):
    return [state_val(v) for v in values]


def mx(values):
    return r"\left[" + ", ".join(fmt(v) for v in values) + r"\right]"


def paren(v):
    """Wrap negatives: `0.357\\cdot -0.2` reads as a typo, `0.357\\cdot(-0.2)` does not."""
    return rf"\left({fmt(v)}\right)" if v < 0 else fmt(v)


def plus_terms(pairs, sep=" + "):
    """LaTeX sum of p*v products."""
    return sep.join(f"{fmt(p)}\\cdot {paren(v)}" for p, v in pairs)


def plus_terms_disp(pairs, sep=" + "):
    return sep.join(
        f"{disp(p)}·{disp(v)}" if v >= 0 else f"{disp(p)}·({disp(v)})" for p, v in pairs
    )


def pct(x):
    """A relative deviation as prose. Four significant digits at most, because
    that is the precision every other number in the trace is shown at."""
    return "%.4g%%" % (100.0 * x)


# ------------------------------------------------------------------- the math
def run_online_softmax(z, bc):
    """The algorithm under replay: blocked online softmax, one pass.

    Returns the per-block trajectory, so every step of the trace is a real
    intermediate of this loop rather than a reconstruction.

    `l_bad` / `acc_bad` are the shadow recursion: the same exponentials, added
    into a history that is never rescaled. They are what "如果不修正" costs,
    measured rather than asserted.
    """
    nb = len(z) // bc
    m, l, acc = float("-inf"), 0.0, 0.0
    l_bad, acc_bad = 0.0, 0.0
    run = []
    for j in range(nb):
        blk = z[j * bc:(j + 1) * bc]
        m_old, l_old, acc_old = m, l, acc
        l_bad_old, acc_bad_old = l_bad, acc_bad
        blkmax = float(blk.max())
        m = max(m_old, blkmax)
        # First block has no history: l_old is 0 and the correction factor is
        # defined to be 0 so the update reduces to the plain sum.
        corr = math.exp(m_old - m) if m_old != float("-inf") else 0.0
        p = np.exp(blk - m)
        l = l_old * corr + float(p.sum())
        acc = acc_old * corr + float((p * blk).sum())
        # The shadow run adds the same two quantities and rescales nothing.
        l_bad = l_bad_old + float(p.sum())
        acc_bad = acc_bad_old + float((p * blk).sum())
        run.append(
            dict(
                j=j, n=j + 1, blk=blk, m_old=m_old, blkmax=blkmax, m_new=m, corr=corr,
                l_old=l_old, l_new=l, acc_old=acc_old, acc_new=acc,
                l_bad=l_bad, acc_bad=acc_bad,
                p=p, psum=float(p.sum()), wsum=float((p * blk).sum()),
            )
        )
    return run, acc / l, m, l, acc


def torch_reference(z, bc):
    """Independent per-block statistics and final output, in torch.

    Deliberately a different implementation from the loop above: the reference
    derives each statistic from the *prefix* of z rather than from the previous
    block's state, so a mistake in the recursion cannot cancel out.
    """
    xt = torch.tensor([float(v) for v in z], dtype=torch.float64)
    blocks = []
    for j in range(len(z) // bc):
        seen = xt[:(j + 1) * bc]
        blk = xt[j * bc:(j + 1) * bc]
        m = seen.max()
        e = torch.exp(seen - m)
        blocks.append(
            dict(m=float(m), l=float(e.sum()), acc=float((e * seen).sum()),
                 blkmax=float(blk.max()))
        )
    p = torch.softmax(xt, dim=0)
    return blocks, float((p * xt).sum())


# Tolerance calibration (the open question left by issue #37).
#
# #37 suggested "roughly 1e-5 relative for fp32, 1e-2 for fp16". Those are
# targets for a lab whose *arithmetic* runs in that precision. This trace is
# computed in float64 throughout — both the recursion and the torch reference —
# so the only error between them is summation order, and the tolerance can be
# far tighter than the suggestion without being flaky.
#
# It is tight deliberately: at 4 significant digits of display precision, a
# 1e-5 tolerance would hide a wrong formula that merely rounds to the same
# number on screen. 1e-12 relative catches a genuinely different computation
# while still leaving summation-order noise three orders of magnitude of room.
_CHECK_REL_TOL = 1e-12
_CHECK_ABS_TOL = 1e-15


def close(a, b):
    return math.isclose(a, b, rel_tol=_CHECK_REL_TOL, abs_tol=_CHECK_ABS_TOL)


def check_against_torch(z, bc, run, out):
    """Assert the recursion matches the torch reference, per block and at the end.

    Per-block assertions matter: the final output alone would still match if two
    opposite errors cancelled, and the trace publishes every intermediate.
    """
    ref_blocks, ref_out = torch_reference(z, bc)
    fields = {"m_new": "m", "l_new": "l", "acc_new": "acc", "blkmax": "blkmax"}
    for r, ref in zip(run, ref_blocks):
        for got_key, want_key in fields.items():
            if not close(r[got_key], ref[want_key]):
                raise AssertionError(
                    f"block {r['n']} {got_key}: recursion={r[got_key]!r} "
                    f"torch={ref[want_key]!r}"
                )
    if not close(out, ref_out):
        raise AssertionError(f"output: recursion={out!r} torch softmax={ref_out!r}")
    return ref_out


# ------------------------------------------------------------- the comparison
def run_naive(z, shift=NAIVE_SHIFT):
    """Naive softmax: exponentiate the raw logits, then normalise.

    One pass, N reads, no max subtraction — the implementation the whole lab
    exists to argue against. numpy would warn on the overflow; that warning is
    the result we are here to publish, so it is suppressed rather than avoided.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.exp(z + shift)
        s = float(np.sum(p))
        out = float(np.sum(p * z) / s)
    return s, out


def run_safe_three_pass(z):
    """The version a careful engineer writes first: three separate scans.

    Pass 1 finds the maximum, pass 2 sums the exponentials at that maximum,
    pass 3 accumulates the weighted sum. Correct, and it reads x three times —
    which is the cost the online recursion exists to remove.
    """
    m = float(np.max(z))
    e = np.exp(z - m)
    l = float(np.sum(e))
    acc = float(np.sum(e * z))
    return m, l, acc, acc / l


def naive_frames(z, shift=NAIVE_SHIFT):
    """The naive method's state after each element of its single pass.

    `out` is the running answer (running sums divided, not a partial sum) so it
    is the same kind of quantity the other two methods report at the same k.
    """
    frames = []
    for k in range(len(z) + 1):
        if k == 0:
            frames.append({"k": 0, "sum_exp": 0.0, "out": None})
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.exp(z[:k] + shift)
            s = float(np.sum(p))
            out = float(np.sum(p * z[:k]) / s)
        frames.append({"k": k, "sum_exp": state_val(s), "out": state_val(out)})
    return frames


def safe_frames(z):
    """The three-pass method's state at each k, across its three scans.

    k counts elements read from HBM, so the scans are laid end to end on one
    axis: 0..N is pass 1, N..2N is pass 2, 2N..3N is pass 3. That is what makes
    "three passes costs three times the reads" a thing the panel can show
    instead of say.
    """
    n = len(z)
    m = float(np.max(z))
    e = np.exp(z - m)
    l_all = float(np.sum(e))
    frames = [{"k": 0, "pass": 1, "m": NEG_INF, "l": None, "out": None}]
    for k in range(1, n + 1):
        frames.append({"k": k, "pass": 1, "m": float(np.max(z[:k])),
                       "l": None, "out": None})
    for k in range(1, n + 1):
        frames.append({"k": n + k, "pass": 2, "m": m,
                       "l": float(np.sum(e[:k])), "out": None})
    for k in range(1, n + 1):
        acc_k = float(np.sum(e[:k] * z[:k]))
        frames.append({"k": 2 * n + k, "pass": 3, "m": m, "l": l_all,
                       "out": acc_k / l_all})
    return frames


def online_frames(run, bc):
    """The online recursion's state after each block, on the same k axis."""
    frames = [{"k": 0, "m": NEG_INF, "l": 0.0, "acc": 0.0, "out": None}]
    for r in run:
        frames.append({"k": r["n"] * bc, "m": state_val(r["m_new"]),
                       "l": state_val(r["l_new"]), "acc": state_val(r["acc_new"]),
                       "out": state_val(r["acc_new"] / r["l_new"])})
    return frames


def build_compare(z, bc, run, ref_out, out, l_bad, acc_bad):
    """Run all three methods for real and package their trajectories.

    The naive result is asserted non-finite rather than merely reported: the
    comparison's whole teaching claim is that the un-shifted implementation
    overflows, and a script that published a finite number there while the page
    still called it "溢出" would be exactly the hand-written demo data this
    project forbids.
    """
    n = len(z)
    naive_sum, naive_out = run_naive(z)
    safe_m, safe_l, safe_acc, safe_out = run_safe_three_pass(z)

    # The control for the naive row: the SAME function, on the same logits,
    # without the offset. If this also failed, the naive path would just be
    # broken code and the overflow would teach nothing about logit magnitude.
    unshifted_sum, unshifted_out = run_naive(z, shift=0.0)
    shifted_sum, shifted_out = run_naive(z, shift=NAIVE_SHIFT)

    # "Overflowed" means non-finite — NaN or ±inf, a NaN arising from inf/inf.
    # Testing `isfinite` would conflate that with a finite wrong answer, which is
    # a different (and much less interesting) failure.
    if math.isfinite(naive_out):
        raise AssertionError(
            f"朴素路径没有溢出（得到 {naive_out!r}）—— 对照模式的教学前提不成立，"
            f"检查 NAIVE_SHIFT 或输入量级"
        )
    if math.isfinite(shifted_out):
        raise AssertionError(
            f"带偏移的朴素路径没有溢出（得到 {shifted_out!r}）——"
            f"它本该与不带偏移的那份分道扬镳"
        )
    if not close(unshifted_out, ref_out):
        raise AssertionError(
            f"不带偏移的朴素路径 {unshifted_out!r} != torch {ref_out!r} —— "
            f"对照组的说明不成立：本该只是量级问题，不是代码错误"
        )
    if not close(safe_out, ref_out):
        raise AssertionError(f"三遍安全版 {safe_out!r} != torch {ref_out!r}")
    if not close(out, ref_out):
        raise AssertionError(f"online 版 {out!r} != torch {ref_out!r}")

    uncorrected_out = acc_bad / l_bad

    return {
        "axis": "已从 HBM 读取的元素数 k",
        "axis_max": 3 * n,
        "reference": "torch.softmax(x/T)·(x/T)，float64",
        "methods": [
            {
                "id": "naive",
                "label": "朴素 softmax（不减最大值）",
                "passes": 1,
                "reads_factor": 1,
                "reads": n,
                "status": "overflow",
                "note": f"先 exp 再归一，不减最大值。同样的代码在原始 logits 上算得对，"
                        f"一旦 logits 整体变大（这里加 {NAIVE_SHIFT:g} 模拟）就先溢出 —— "
                        f"而 softmax 对整体平移是不变的。",
                "final": state_val(naive_out),
                "matches_reference": False,
                "frames": naive_frames(z),
            },
            {
                "id": "safe",
                "label": "三遍安全 softmax",
                "passes": 3,
                "reads_factor": 3,
                "reads": 3 * n,
                "status": "ok",
                "note": "先扫一遍求最大值，再扫一遍求和，最后扫一遍归一化。正确，"
                        "但要把 x 读三遍 —— 这正是 online 递推要消掉的代价。",
                "final": state_val(safe_out),
                "matches_reference": True,
                "frames": safe_frames(z),
            },
            {
                "id": "online",
                "label": "Online softmax（本页回放）",
                "passes": 1,
                "reads_factor": 1,
                "reads": n,
                "status": "ok",
                "note": "一遍扫描，每块用新的最大值修正历史统计量。读数与朴素版相同，"
                        "结果与三遍版相同。",
                "final": state_val(out),
                "matches_reference": True,
                "frames": online_frames(run, bc),
            },
        ],
        # The naive row's control. `softmax(x + c) == softmax(x)` for any
        # constant c, so the two numbers below are the same function on inputs
        # that differ only by a constant — and one of them is NaN. That is the
        # whole argument against subtracting nothing, as two measured values.
        "shift_invariance": {
            "offset": NAIVE_SHIFT,
            "note": f"同一份朴素实现，只把 logits 整体加 {NAIVE_SHIFT:g}。"
                    f"数学上 softmax(x + c) = softmax(x)，两个结果应该一模一样。",
            "unshifted_final": state_val(unshifted_out),
            "unshifted_sum_exp": state_val(unshifted_sum),
            "shifted_final": state_val(shifted_out),
            "shifted_sum_exp": state_val(shifted_sum),
            "reference": state_val(out),
        },
        "uncorrected": {
            "label": "Online（漏掉修正因子）",
            "note": "同一遍扫描，但 ℓ 与 acc 的历史永不缩放 —— 即忘记乘 "
                    "e^(m_old − m_new)。与 online 版读取代价完全相同。",
            "final": state_val(uncorrected_out),
            "bias_abs": abs(uncorrected_out - out),
            "bias_rel": abs(uncorrected_out - out) / abs(out),
        },
    }


# ---------------------------------------------------------------- trace build
def b(slot, sym, idx, num):
    return slot, {"sym": sym, "idx": idx, "num": num}


def step(sid, title, kind, op, phase, formula, bindings, reads, writes, state,
         narration, regions=None, corr=None, k=0):
    d = {"id": sid, "title": title, "kind": kind, "op": op, "phase": phase,
         "formula": formula, "bindings": dict(bindings),
         "reads": reads, "writes": writes, "state": state, "narration": narration,
         "k": k}
    if regions:
        d["regions"] = regions
    if corr:
        d["corr"] = corr
    return d


def corr_kind(j, blkmax, m_old):
    """Which of the three roles the correction factor plays on this block.

    Naming them is the point: a reader who only ever sees "e^(m_old - m_new)"
    highlighted cannot tell the block where it matters from the block where it is
    1 by construction.
    """
    if j == 0:
        return "init"
    return "rescale" if blkmax > m_old else "identity"


def corr_block(run_i, j, l_new, acc_new):
    """The correction factor and what omitting it costs, at this step.

    `o_correct` and `o_uncorrected` are both running answers (the running sums
    divided) rather than partial sums, so they are the same kind of quantity and
    subtracting them is meaningful from the first block on. On block 1 they are
    identical, which is itself the lesson: the rescale only starts to matter
    once m has moved.
    """
    kind = corr_kind(j, run_i["blkmax"], run_i["m_old"])
    if kind == "identity":
        kind = "identity"
    o_correct = acc_new / l_new
    o_bad = run_i["acc_bad"] / run_i["l_bad"]
    bias = abs(o_bad - o_correct)
    return {
        "kind": kind,
        "factor": state_val(run_i["corr"]),
        "m_old": state_val(run_i["m_old"]),
        "m_new": state_val(run_i["m_new"]),
        "o_correct": o_correct,
        "o_uncorrected": o_bad,
        "bias_abs": bias,
        "bias_rel": bias / abs(o_correct) if o_correct else 0.0,
    }


CORR_NOTE = {
    "init": "第一块没有历史：ℓ 旧值是 0，因子按定义为 0，更新化简成直接求和。",
    "identity": "本步 m 没有变，因子恰好是 1 —— 乘以 1 是恒等操作，不修正也没有偏差。",
    "rescale": "本步 m 被抬高，历史 ℓ 与 acc 是按旧的 m 累加的，必须乘这个因子拉回来。",
}


def build_trace(raw, n, bc, temp, title, source):
    x_raw = np.asarray(raw[:n], dtype=np.float64)
    # Temperature enters where it mathematically belongs: softmax(x/T). Doing it
    # here rather than in the page keeps "every number is the output of this
    # script" literally true — the page never divides anything.
    z = x_raw / temp
    if len(z) % bc:
        raise ValueError(f"N={len(z)} is not a multiple of Bc={bc}")

    nb = len(z) // bc
    run, out, m_end, l_end, acc_end = run_online_softmax(z, bc)
    ref_out = check_against_torch(z, bc, run, out)
    compare = build_compare(z, bc, run, ref_out, out,
                            run[-1]["l_bad"], run[-1]["acc_bad"])

    steps = []

    # --- kind: "state" -- the initialiser. Included because m starts at -inf
    # rather than 0, which is the one fact a reader has to know before the first
    # block makes sense.
    steps.append(step(
        "init", "初始化", "state", "init", "准备",
        {t: r"m^{(0)} = \slot{m0},\quad \ell^{(0)} = \slot{l0},\quad "
            r"\mathrm{acc}^{(0)} = \slot{a0}" for t in ("sym", "idx", "num")},
        [b("m0", NEG_INF, NEG_INF, NEG_INF),
         b("l0", "0", "0", "0"),
         b("a0", "0", "0", "0")],
        [], ["m", "l", "acc"],
        {"m": state_val(float("-inf")), "l": 0.0, "acc": 0.0},
        "开始时 m 是 -∞ 而不是 0。如果这里写 0，遇到一整块全是负数的输入，"
        "最大值就会被 0 顶掉，后面 exp 全溢出。",
        k=0,
    ))

    for r in run:
        j, nb_i, blk = r["j"], r["n"], r["blk"]

        # --- kind: "comm" -- reads x, writes only the staging buffer. This is
        # the step that has no data-dependency predecessor and therefore forces
        # the engine to synthesise sequence edges for layout. See the layout
        # regression sample.
        steps.append(step(
            f"ld{nb_i}", f"第 {nb_i} 块载入 SRAM", "comm", "load", "取数",
            {
                "sym": r"x^{(j)} \leftarrow \mathrm{HBM}[\,x\,]\left[\,(j-1)B_c \,:\, jB_c\,\right]",
                "idx": r"x^{(\slot{J})} \leftarrow \mathrm{HBM}[\,x\,]\left[\slot{LO} : \slot{HI}\right]",
                "num": r"x^{(\slot{J})} \leftarrow \mathrm{HBM}[\,x\,]\left[\slot{LO} : \slot{HI}\right] "
                       r"= \slot{VALS}",
            },
            [b("J", "j", str(nb_i), str(nb_i)),
             b("LO", r"(j-1)B_c", str(j * bc), str(j * bc)),
             b("HI", r"jB_c", str((j + 1) * bc), str((j + 1) * bc)),
             b("VALS", r"[\cdot]", r"[\cdot]", mx(blk))],
            ["x"], ["x_blk"],
            {"x_blk": state_list(blk)},
            f"把 x 的第 {nb_i} 块（下标 {j * bc}..{(j + 1) * bc - 1}）从 HBM 搬进 SRAM。"
            f"这一步不改任何标量状态，只是把数据挪个位置。"
            + (f"注意 x 已经是 x_raw / T（T = {temp:g}）之后的值，"
               f"温度只在这里体现为数值本身。" if temp != 1 else ""),
            k=nb_i * bc,
        ))

        steps.append(step(
            f"m{nb_i}", f"第 {nb_i} 块：更新运行最大值 m", "op", "max", "统计量更新",
            {
                "sym": r"m^{(j)} = \max\left(\slot{MOLD},\; \slot{BLKMAX}\right)",
                "idx": r"m^{(\slot{J})} = \max\left(\slot{MOLD},\; \slot{BLKMAX}\right)",
                "num": r"m^{(\slot{J})} = \max\left(\slot{MOLD},\; \slot{BLKMAX}\right) = \slot{MNEW}",
            },
            [b("J", "j", str(nb_i), str(nb_i)),
             b("MOLD", r"m^{(j-1)}", rf"m^{{{j}}}", fmt(r["m_old"])),
             b("BLKMAX", r"\max_i x^{(j)}_i", rf"\max_i x^{{{nb_i}}}_i", fmt(r["blkmax"])),
             b("MNEW", r"m^{(j)}", rf"m^{{{nb_i}}}", fmt(r["m_new"]))],
            ["x_blk", "m"], ["m"],
            {"m": state_val(r["m_new"])},
            f"本块最大值 {disp(r['blkmax'])}，"
            + ("比当前的 m 大，m 要抬上去。" if r["blkmax"] > r["m_old"]
               else "没超过当前的 m，m 不变。")
            + ("这是第一块，m 还是 -∞，所以直接取本块最大值。" if r["m_old"] == float("-inf") else "")
            + (f" m 一动，下一步的 ℓ 与 acc 就必须乘修正因子 —— 这是本步真正的后果。"
               if corr_kind(j, r["blkmax"], r["m_old"]) == "rescale" else ""),
            k=nb_i * bc,
        ))

        # --- the correction factor, boxed by \region{CORR}
        cb = corr_block(r, j, r["l_new"], r["acc_new"])
        corr_note = (
            f"{cb['kind'].upper()} · e^(m_old − m_new) = e^({disp(r['m_old'])} − "
            f"{disp(r['m_new'])}) = {disp(r['corr'])}。{CORR_NOTE[cb['kind']]} "
            f"本步为止：修正后 O = {disp(cb['o_correct'])}，"
            f"不修正 O = {disp(cb['o_uncorrected'])}，偏差 {disp(cb['bias_abs'])}"
            f"（相对 {pct(cb['bias_rel'])}）。"
        )
        steps.append(step(
            f"l{nb_i}", f"第 {nb_i} 块：更新指数和 ℓ", "op", "sum-exp", "统计量更新",
            {
                "sym": r"\ell^{(j)} = \slot{LOLD}\cdot "
                       r"\region{CORR}{\exp\left(m^{(j-1)} - m^{(j)}\right)} "
                       r"+ \sum_i \exp\left(x^{(j)}_i - m^{(j)}\right)",
                "idx": r"\ell^{(\slot{J})} = \slot{LOLD}\cdot "
                       r"\region{CORR}{\exp\left(\slot{MOLD} - \slot{MNEW}\right)} + \slot{SUM}",
                "num": r"\ell^{(\slot{J})} = \slot{LOLD}\cdot "
                       r"\region{CORR}{\exp\left(\slot{MOLD} - \slot{MNEW}\right)} "
                       r"+ \slot{SUM} = \slot{LNEW}",
            },
            [b("J", "j", str(nb_i), str(nb_i)),
             b("LOLD", r"\ell^{(j-1)}", rf"\ell^{{{j}}}", fmt(r["l_old"])),
             b("MOLD", r"m^{(j-1)}", rf"m^{{{j}}}", fmt(r["m_old"])),
             b("MNEW", r"m^{(j)}", rf"m^{{{nb_i}}}", fmt(r["m_new"])),
             # The num form is the un-evaluated substitution, not the result --
             # the formula already ends with "= \slot{LNEW}". A binding that
             # repeats its own result renders as "= 1.357 = 1.357".
             b("SUM", r"\sum_i \exp(x^{(j)}_i - m^{(j)})",
               rf"\sum_i \exp(x^{{{nb_i}}}_i - m^{{{nb_i}}})",
               " + ".join(fmt(v) for v in r["p"])),
             b("LNEW", r"\ell^{(j)}", rf"\ell^{{{nb_i}}}", fmt(r["l_new"]))],
            ["x_blk", "m", "l"], ["l"],
            {"l": state_val(r["l_new"])},
            (f"m 变了，之前累加的 ℓ 是按旧的 m 算的，所以要乘修正因子 {disp(r['corr'])} 把它拉回来；"
             f"本块的 " + " + ".join(fmt(v) for v in r["p"]) + f" = {disp(r['psum'])} 再加进去。"
             if j > 0 and cb["kind"] == "rescale" else
             (f"m 没变，修正因子是 {disp(r['corr'])}，ℓ 旧值原样保留，只加本块的 "
              + " + ".join(fmt(v) for v in r["p"]) + f" = {disp(r['psum'])}。"
              if j > 0 else
              f"第一块没有历史，修正因子是 0（ℓ 旧值是 0），只加本块的 "
              + " + ".join(fmt(v) for v in r["p"]) + f" = {disp(r['psum'])}。")),
            regions={"CORR": corr_note},
            corr=cb,
            k=nb_i * bc,
        ))

        steps.append(step(
            f"a{nb_i}", f"第 {nb_i} 块：更新累加器 acc", "op", "fma", "输出更新",
            {
                "sym": r"\mathrm{acc}^{(j)} = \slot{AOLD}\cdot "
                       r"\region{CORR}{\exp\left(m^{(j-1)} - m^{(j)}\right)} "
                       r"+ \sum_i \exp\left(x^{(j)}_i - m^{(j)}\right)\cdot x^{(j)}_i",
                "idx": r"\mathrm{acc}^{(\slot{J})} = \slot{AOLD}\cdot "
                       r"\region{CORR}{\exp\left(\slot{MOLD} - \slot{MNEW}\right)} + \slot{WSUM}",
                "num": r"\mathrm{acc}^{(\slot{J})} = \slot{AOLD}\cdot "
                       r"\region{CORR}{\exp\left(\slot{MOLD} - \slot{MNEW}\right)} "
                       r"+ \slot{WSUM} = \slot{ANEW}",
            },
            [b("J", "j", str(nb_i), str(nb_i)),
             b("AOLD", r"\mathrm{acc}^{(j-1)}", rf"\mathrm{{acc}}^{{{j}}}", fmt(r["acc_old"])),
             b("MOLD", r"m^{(j-1)}", rf"m^{{{j}}}", fmt(r["m_old"])),
             b("MNEW", r"m^{(j)}", rf"m^{{{nb_i}}}", fmt(r["m_new"])),
             b("WSUM", r"\sum_i \exp(x^{(j)}_i - m^{(j)})\cdot x^{(j)}_i",
               rf"\sum_i \exp(x^{{{nb_i}}}_i - m^{{{nb_i}}})\cdot x^{{{nb_i}}}_i",
               plus_terms(list(zip(r["p"], blk)))),
             b("ANEW", r"\mathrm{acc}^{(j)}", rf"\mathrm{{acc}}^{{{nb_i}}}", fmt(r["acc_new"]))],
            ["x_blk", "m", "acc"], ["acc"],
            {"acc": state_val(r["acc_new"])},
            (f"和 ℓ 用同一个修正因子 {disp(r['corr'])}，把旧的 acc 缩回来，再加上本块的加权和 "
             + plus_terms_disp(list(zip(r["p"], blk))) + f" = {disp(r['wsum'])}。"
             + "这就是 FlashAttention 里对 O 做 rescale 的那一步。"
             if j > 0 and cb["kind"] == "rescale" else
             (f"修正因子是 {disp(r['corr'])}（恒等），旧的 acc 原样保留，加上本块的加权和 "
              + plus_terms_disp(list(zip(r["p"], blk))) + f" = {disp(r['wsum'])}。"
              if j > 0 else
              f"第一块，旧的 acc 是 0，直接取本块加权和 "
              + plus_terms_disp(list(zip(r["p"], blk))) + f" = {disp(r['wsum'])}。")),
            regions={"CORR": corr_note},
            corr=cb,
            k=nb_i * bc,
        ))

    steps.append(step(
        "fin", "收尾：归一化", "op", "div", "收尾",
        {t: (r"O = \slot{ACC} \,/\, \slot{L}" + (r" = \slot{ONUM}" if t == "num" else ""))
         for t in ("sym", "idx", "num")},
        [b("ACC", r"\mathrm{acc}^{(J)}", rf"\mathrm{{acc}}^{{{nb}}}", fmt(acc_end)),
         b("L", r"\ell^{(J)}", rf"\ell^{{{nb}}}", fmt(l_end)),
         b("ONUM", "O", "O", fmt(out))],
        ["acc", "l"], ["O"],
        {"O": state_val(out)},
        f"整条 trace 只在这里除一次。O = {disp(acc_end)} / {disp(l_end)} = {disp(out)}，"
        f"就是 softmax 加权平均，与 torch 的参考实现逐位一致。"
        f"三遍安全版得到同一个数，代价是 3 倍的读数；朴素版读数相同但结果是 NaN。",
        k=len(z),
    ))

    # ---------------------------------------------------------------- graph
    # Nodes carry no `step` field on purpose: ids are shared with `steps`, so the
    # engine derives the index. A redundant field could only ever drift.
    nodes = [{"id": s["id"], "title": s["title"], "kind": s["kind"],
              "op": s["op"], "phase": s["phase"]} for s in steps]

    edges = [{"from": "init", "to": "m1", "tensor": "m"},
             {"from": "init", "to": "l1", "tensor": "l"},
             {"from": "init", "to": "a1", "tensor": "acc"}]
    for i in range(1, nb + 1):
        edges += [{"from": f"ld{i}", "to": f"m{i}", "tensor": "x_blk"},
                  {"from": f"m{i}", "to": f"l{i}", "tensor": "m"},
                  {"from": f"m{i}", "to": f"a{i}", "tensor": "m"},
                  {"from": f"l{i}", "to": f"a{i}", "tensor": "l"}]
        if i < nb:
            edges += [{"from": f"m{i}", "to": f"m{i + 1}", "tensor": "m"},
                      {"from": f"l{i}", "to": f"l{i + 1}", "tensor": "l"},
                      {"from": f"a{i}", "to": f"a{i + 1}", "tensor": "acc"}]
        else:
            edges += [{"from": f"a{i}", "to": "fin", "tensor": "acc"},
                      {"from": f"l{i}", "to": "fin", "tensor": "l"}]

    # Counted per BLOCK, not per step: both the l step and the a step carry the
    # same factor, so counting corr blocks would double every rescale.
    per_block_corr = [
        {"n": i + 1,
         "kind": corr_kind(r["j"], r["blkmax"], r["m_old"]),
         "factor": state_val(r["corr"])}
        for i, r in enumerate(run)
    ]
    rescales = sum(1 for e in per_block_corr if e["kind"] == "rescale")
    uncorrected_out = run[-1]["acc_bad"] / run[-1]["l_bad"]

    return {
        "meta": {
            "lab": "L00",
            "title": title,
            "source": source,
            "config": {"N": len(z), "Bc": bc, "T": temp, "n_blocks": nb},
            "input": {
                "raw": [float(v) for v in x_raw],
                "scaled": [float(v) for v in z],
                "note": f"送入 softmax 的实际输入是 x/T（T = {temp:g}），"
                        f"即 tensors.x；x_raw 是未经温度缩放的原始 logits。",
            },
            "reference": f"O = softmax(x/T)·(x/T) = {ref_out!r} (torch float64, "
                         f"per-block statistics also cross-checked)",
            "correction": {
                "rescales": rescales,
                # What the "forget the rescale" bug costs at the end, computed by
                # running the buggy recursion to completion rather than by
                # estimating it from the largest factor.
                "final_o_correct": out,
                "final_o_uncorrected": uncorrected_out,
                "final_bias_abs": abs(uncorrected_out - out),
                "final_bias_rel": abs(uncorrected_out - out) / abs(out),
                "note": "「不修正」= 同一条递推里 ℓ 与 acc 的历史永不缩放。",
                "per_block": per_block_corr,
            },
        },
        "tensors": {
            # `init` is the contract addition: no step writes x, but the first
            # block load reads it, so without this render(trace, 0) has nothing
            # to show for the input.
            "x": {"shape": [len(z)], "dtype": "fp32", "at": "HBM", "role": "input",
                  "init": [float(v) for v in z],
                  "note": f"整条序列，全程驻留 HBM（已按 x/T 缩放，T = {temp:g}）"},
            "x_blk": {"shape": [bc], "dtype": "fp32", "at": "SRAM", "role": "staging",
                      "note": "当前块的临时副本，每块覆盖一次"},
            "m": {"shape": [], "dtype": "fp32", "at": "register", "role": "state",
                  "note": "运行最大值，标量"},
            "l": {"shape": [], "dtype": "fp32", "at": "register", "role": "state",
                  "note": "指数和，标量"},
            "acc": {"shape": [], "dtype": "fp32", "at": "register", "role": "state",
                    "note": "加权累加器，标量"},
            "O": {"shape": [], "dtype": "fp32", "at": "register", "role": "output",
                  "note": "归一化后的输出，标量"},
        },
        "graph": {"nodes": nodes, "edges": edges},
        "compare": compare,
        "steps": steps,
    }


# -------------------------------------------------------------------- self-lint
SLOT_RE = re.compile(r"\\slot\{([A-Za-z0-9_]+)\}")
REGION_RE = re.compile(r"\\region\{([A-Za-z0-9_]+)\}")
LATEX_IN_TEXT_RE = re.compile(r"\\[a-zA-Z]+\{")
BAD_LITERAL_RE = re.compile(r"^(NaN|Infinity|-Infinity|undefined|null)$")

CORR_KEYS = {"kind", "factor", "m_old", "m_new", "o_correct", "o_uncorrected",
             "bias_abs", "bias_rel"}
CORR_KINDS = ("init", "identity", "rescale")
METHOD_IDS = ("naive", "safe", "online")


def value_ok(v):
    """A trace value is either a finite number, null (not produced yet), or one
    of the three sentinels.

    A bare non-finite float is NOT acceptable even though `json.dumps` will
    happily write `NaN` / `Infinity` — those are legal JavaScript but illegal
    JSON, and they are exactly what the sentinel vocabulary exists to replace.
    """
    if v is None:
        return True
    if isinstance(v, bool):
        return False
    if isinstance(v, str):
        return v in (NEG_INF, POS_INF, NAN)
    if isinstance(v, int):
        return True
    return isinstance(v, float) and math.isfinite(v)


def lint(trace):
    """Check the trace against the contract the engine and the page consume.

    This is the author-side half of the check; the engine carries a JS port of
    the same rules so a lab author gets the same feedback in the page. Both must
    stay in step with docs/plans/interactive-labs.md §一.
    """
    gaps, warns, infos = [], [], []
    declared = set(trace["tensors"])
    step_ids = {s["id"] for s in trace["steps"]}
    n = trace["meta"]["config"]["N"]

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

    node_ids = {n_["id"] for n_ in trace["graph"]["nodes"]}
    for s in trace["steps"]:
        if s["id"] not in node_ids:
            gaps.append(f'步骤 "{s["id"]}" 在 graph.nodes 里没有对应节点')
    for n_ in trace["graph"]["nodes"]:
        if n_["id"] not in step_ids:
            warns.append(f'图节点 "{n_["id"]}" 没有对应步骤，点击无处可跳')

    # ---- progress axis: k must be a real HBM read count, never decreasing.
    # A step reported as having read less than the step before it would put the
    # comparison panel's cursor in two places at once.
    last_k = 0
    for s in trace["steps"]:
        if "k" not in s:
            gaps.append(f'步骤 "{s["id"]}" 没有 k（截至本步已从 HBM 读取的元素数）—— '
                        f"对照面板无法把它对齐到共同的时间轴")
            continue
        if not isinstance(s["k"], int) or not 0 <= s["k"] <= n:
            gaps.append(f'步骤 "{s["id"]}" 的 k = {s["k"]!r} 不是 0..{n} 的整数')
        elif s["k"] < last_k:
            gaps.append(f'步骤 "{s["id"]}" 的 k = {s["k"]} 比上一步的 {last_k} 还小 —— '
                        f"k 是已读元素数，只能单调不减")
        last_k = max(last_k, s["k"] if isinstance(s["k"], int) else last_k)
    if trace["steps"] and trace["steps"][-1].get("k") != n:
        gaps.append(f'最后一步的 k = {trace["steps"][-1].get("k")!r} 而不是 N = {n} —— '
                    f"整条输入必须都被读过")

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

        if LATEX_IN_TEXT_RE.search(s.get("narration") or ""):
            warns.append(f'步骤 "{sid}" 的 narration 里含 LaTeX 命令，会原样显示')

        for key, val in (s.get("state") or {}).items():
            if key not in declared:
                gaps.append(f'步骤 "{sid}" 的 state 写了未声明张量 "{key}"')
            if key not in (s.get("writes") or []):
                warns.append(f'步骤 "{sid}" 的 state 里有 "{key}"，但它不在 writes[] 里')
            for item in (val if isinstance(val, list) else [val]):
                if isinstance(item, str) and item not in (NEG_INF, POS_INF, NAN):
                    gaps.append(f'步骤 "{sid}" 的 state["{key}"] 含未知哨兵字符串 "{item}"')

        for key in (s.get("regions") or {}):
            if not any(f"\\region{{{key}}}" in (formula.get(t) or "")
                       for t in ("sym", "idx", "num")):
                warns.append(f'步骤 "{sid}" 声明了 region "{key}" 但公式里没有引用')

        # ---- the correction factor. It may only be attached to a step whose
        # formula actually shows it, and its numbers must be real: the whole
        # point of the amplifier panel is that the deviation is measured.
        corr = s.get("corr")
        if corr is not None:
            if set(corr) != CORR_KEYS:
                gaps.append(f'步骤 "{sid}" 的 corr 字段是 '
                            f'{{{", ".join(sorted(corr))}}} 而不是 '
                            f'{{{", ".join(sorted(CORR_KEYS))}}}')
            if corr.get("kind") not in CORR_KINDS:
                gaps.append(f'步骤 "{sid}" 的 corr.kind = {corr.get("kind")!r} 不属于 '
                            f'{" / ".join(CORR_KINDS)}')
            if "CORR" not in (s.get("regions") or {}):
                gaps.append(f'步骤 "{sid}" 带了 corr，却没有 \\region{{CORR}} 的说明 —— '
                            f"被框住的那段没有文字解释")
            for key in ("factor", "o_correct", "o_uncorrected", "bias_abs", "bias_rel"):
                if not value_ok(corr.get(key)):
                    gaps.append(f'步骤 "{sid}" 的 corr.{key} = {corr.get(key)!r} '
                                f"不是数、null 或哨兵字符串")
            if corr.get("kind") == "identity" and corr.get("factor") != 1.0:
                gaps.append(f'步骤 "{sid}" 的 corr.kind 是 identity，但因子是 '
                            f'{corr.get("factor")!r} 而不是 1')
            if corr.get("kind") == "rescale" and not (
                    isinstance(corr.get("factor"), float) and corr["factor"] < 1.0):
                gaps.append(f'步骤 "{sid}" 的 corr.kind 是 rescale，因子 {corr.get("factor")!r} '
                            f"却不小于 1 —— m 抬高时 e^(m_old − m_new) 必然小于 1")

    # ---- trace-level correction summary
    meta_corr = (trace.get("meta") or {}).get("correction")
    if not isinstance(meta_corr, dict):
        gaps.append("meta.correction 缺失 —— 放大器面板报不出整条 trace 的最终偏差")
    else:
        for key in ("final_o_correct", "final_o_uncorrected",
                    "final_bias_abs", "final_bias_rel", "per_block", "rescales"):
            if key not in meta_corr:
                gaps.append(f"meta.correction 缺 {key}")
        per_block = meta_corr.get("per_block")
        if not isinstance(per_block, list) or len(per_block) != \
                trace["meta"]["config"]["n_blocks"]:
            gaps.append(f'meta.correction.per_block 有 '
                        f'{len(per_block) if isinstance(per_block, list) else "?"} 项，'
                        f'应该是 n_blocks = {trace["meta"]["config"]["n_blocks"]} 项')
        else:
            for entry in per_block:
                if entry.get("kind") not in CORR_KINDS:
                    gaps.append(f'meta.correction.per_block 里 kind = {entry.get("kind")!r} 非法')
            stepped = [e for e in per_block if e.get("kind") == "rescale"]
            if meta_corr.get("rescales") != len(stepped):
                gaps.append(f'meta.correction.rescales = {meta_corr.get("rescales")!r} '
                            f'与 per_block 里 rescale 的项数 {len(stepped)} 不一致')

    # ---- the comparison block: three real runs on one shared axis
    cmp_ = trace.get("compare")
    if not isinstance(cmp_, dict):
        gaps.append("compare 缺失 —— 对照模式没有数据")
    else:
        methods = cmp_.get("methods") or []
        ids = [m.get("id") for m in methods]
        if ids != list(METHOD_IDS):
            gaps.append(f"compare.methods 的 id 是 {ids!r}，"
                        f"应该是 {list(METHOD_IDS)!r}（顺序也参与渲染）")
        axis_max = cmp_.get("axis_max")
        # Each method's frames run from k=0 to its OWN read count, not to the
        # shared axis_max: a method that has already read everything it needs
        # has no further states, and the panel holds its last one. The axis is
        # what the three are compared ON, not a length each must fill.
        for m in methods:
            mid = m.get("id")
            frames = m.get("frames") or []
            if not frames:
                gaps.append(f'compare.methods["{mid}"] 没有 frames')
                continue
            if frames[0].get("k") != 0 or frames[-1].get("k") != m.get("reads"):
                gaps.append(f'compare.methods["{mid}"] 的 frames 覆盖 k = '
                            f'{frames[0].get("k")}..{frames[-1].get("k")}，'
                            f'应该从 0 到它自己的 reads = {m.get("reads")!r}')
            if m.get("reads") != m.get("passes", 0) * n:
                gaps.append(f'compare.methods["{mid}"] 的 reads = {m.get("reads")!r} '
                            f'与 passes × N = {m.get("passes")!r} × {n} 不一致')
            ks = [f.get("k") for f in frames]
            if ks != sorted(ks) or len(set(ks)) != len(ks):
                gaps.append(f'compare.methods["{mid}"] 的 frames 不是按 k 严格递增')
            if not value_ok(m.get("final")):
                gaps.append(f'compare.methods["{mid}"].final = {m.get("final")!r} 非法')
            for f in frames:
                for key, v in f.items():
                    if key == "k":
                        continue
                    if not value_ok(v):
                        gaps.append(f'compare.methods["{mid}"] 的 frame k={f.get("k")} '
                                    f"字段 {key} = {v!r} 非法")
        if axis_max != max((m.get("reads") or 0) for m in methods):
            gaps.append(f"compare.axis_max = {axis_max!r}，应该是三条轨迹里最大的 reads")

        by_id = {m.get("id"): m for m in methods}
        # What the control is checked against. `online`'s final was asserted
        # against torch during generation, so it is the trace's own best
        # available statement of the right answer.
        online_final = (by_id.get("online") or {}).get("final")

        # The shift-invariance control. Not optional: it is what turns "the
        # naive version overflows" from an assertion into a demonstration, and
        # the page prints it unconditionally.
        si = cmp_.get("shift_invariance")
        if not isinstance(si, dict):
            gaps.append("compare.shift_invariance 缺失 —— 对照面板无法说明朴素版"
                        "是真的量级问题还是代码本来就写错了")
        else:
            for key in ("offset", "unshifted_final", "unshifted_sum_exp", "shifted_final",
                        "shifted_sum_exp", "reference", "note"):
                if key not in si:
                    gaps.append(f"compare.shift_invariance 缺 {key}")
            for key in ("unshifted_final", "shifted_final", "reference", "unshifted_sum_exp",
                        "shifted_sum_exp"):
                if key in si and not value_ok(si[key]):
                    gaps.append(f"compare.shift_invariance.{key} = {si[key]!r} 非法")
            shifted = si.get("shifted_final")
            if isinstance(shifted, (int, float)) and math.isfinite(shifted):
                gaps.append(f"compare.shift_invariance.shifted_final = {shifted!r} 是有限值 —— "
                            f"带上偏移的朴素路径本该溢出，否则这个对照没有说服力")
            # The two "this one is right" numbers the panel prints side by side
            # must actually agree, and agree with the run the generator
            # asserted against torch. Otherwise the page would be presenting a
            # control that contradicts the result it is meant to support.
            for key in ("unshifted_final", "reference"):
                v = si.get(key)
                if isinstance(v, (int, float)) and isinstance(online_final, (int, float)) \
                        and not close(v, online_final):
                    gaps.append(f"compare.shift_invariance.{key} = {v!r} 与 online 版的终值 "
                                f"{online_final!r} 不一致 —— 对照组的「本该算对」不成立")

        if "naive" in by_id and isinstance(by_id["naive"].get("final"), str) \
                and by_id["naive"]["final"] == POS_INF:
            warns.append("compare 的朴素版最终值是 +∞ 而不是 NaN —— 检查是否真的复现了溢出")

    infos.append(f'共 {len(trace["steps"])} 步 / {len(trace["graph"]["nodes"])} 节点 / '
                 f'{len(trace["graph"]["edges"])} 数据边 / '
                 f'{meta_corr.get("rescales") if isinstance(meta_corr, dict) else "?"} 次修正')
    infos.append(f'state 写过的张量：{", ".join(sorted(written))}')
    return gaps, warns, infos


def _method(trace, mid):
    """Look a comparison method up by id. Used by the sabotage cases so they
    keep pointing at the method they mean if the order ever changes."""
    return next(m for m in trace["compare"]["methods"] if m["id"] == mid)


def _corr_step(trace):
    """Index of any step carrying a `corr` block.

    Deliberately not "a step of kind X": which kinds occur is a property of the
    input and the block size — Bc=4 over eight elements has no block where m
    holds, so no `identity` step exists to address. The sabotage cases therefore
    set the fields they need explicitly instead of borrowing them.
    """
    return next(i for i, s in enumerate(trace["steps"]) if s.get("corr"))


# Each entry breaks one lint rule on the real trace. Kept at module scope so the
# report can name how many were exercised.
SABOTAGE_CASES = {
    "read-but-never-written tensor with no init": lambda t: t["tensors"].pop("x"),
    "state entry for an undeclared tensor": lambda t: t["steps"][2]["state"].update({"nope": 1.0}),
    "binding missing a tier": lambda t: t["steps"][2]["bindings"].update({"MOLD": {"sym": "a"}}),
    "binding collapsed to a 2-tuple": lambda t: t["steps"][2]["bindings"].update(
        {"MNEW": {"num": "0.83"}}),
    "\\slot with no binding": lambda t: t["steps"][2]["formula"].update(
        {"num": t["steps"][2]["formula"]["num"] + r"\slot{GHOST}"}),
    "\\region with no description": lambda t: t["steps"][3]["formula"].update(
        {"sym": t["steps"][3]["formula"]["sym"] + r"\region{GHOST}{x}"}),
    "step with no graph node": lambda t: t["graph"]["nodes"].pop(0),
    "formula tier missing": lambda t: t["steps"][0]["formula"].pop("idx"),
    "raw -Infinity instead of the sentinel": lambda t: t["steps"][0]["state"].update({"m": "-Infinity"}),
    "unknown sentinel string": lambda t: t["steps"][0]["state"].update({"m": "-inf"}),
    # --- the fields the two new views read
    "corr attached to a step with no \\region{CORR}": lambda t: t["steps"][2].update(
        {"corr": t["steps"][_corr_step(t)]["corr"]}),
    "corr.kind outside the vocabulary": lambda t: t["steps"][
        _corr_step(t)]["corr"].update({"kind": "maybe"}),
    "corr claims identity with a factor that is not 1": lambda t: t["steps"][
        _corr_step(t)]["corr"].update({"kind": "identity", "factor": 0.5}),
    "corr claims rescale with a factor above 1": lambda t: t["steps"][
        _corr_step(t)]["corr"].update({"kind": "rescale", "factor": 1.4}),
    "corr.bias_abs is a bare Infinity": lambda t: t["steps"][
        _corr_step(t)]["corr"].update({"bias_abs": float("inf")}),
    "corr.o_uncorrected is a bare NaN": lambda t: t["steps"][
        _corr_step(t)]["corr"].update({"o_uncorrected": float("nan")}),
    "corr missing a field": lambda t: t["steps"][
        _corr_step(t)]["corr"].pop("o_uncorrected"),
    "k goes backwards": lambda t: t["steps"][4].update({"k": 0}),
    "k beyond N": lambda t: t["steps"][4].update({"k": 999}),
    "last step has not read all of x": lambda t: t["steps"][-1].update({"k": 0}),
    "step without k": lambda t: t["steps"][2].pop("k"),
    "meta.correction missing": lambda t: t["meta"].pop("correction"),
    "meta.correction.per_block out of step with blocks": lambda t: t["meta"]["correction"].update(
        {"per_block": t["meta"]["correction"]["per_block"][:-1]}),
    "meta.correction.rescales miscounted": lambda t: t["meta"]["correction"].update({"rescales": 99}),
    "compare missing": lambda t: t.pop("compare"),
    "compare with only two methods": lambda t: t["compare"].update(
        {"methods": t["compare"]["methods"][:2]}),
    "compare methods out of order": lambda t: t["compare"].update(
        {"methods": list(reversed(t["compare"]["methods"]))}),
    "compare frames stopping short of the method's own reads": lambda t: _method(
        t, "safe")["frames"].pop(),
    "compare frame with a bare NaN float": lambda t: _method(
        t, "naive")["frames"][1].update({"out": float("nan")}),
    "compare reads inconsistent with passes": lambda t: _method(
        t, "safe").update({"reads": 2 * t["meta"]["config"]["N"]}),
    "compare axis_max not the widest method": lambda t: t["compare"].update({"axis_max": 1}),
    "shift-invariance control reporting a finite shifted result": lambda t: t["compare"][
        "shift_invariance"].update({"shifted_final": 1.0}),
    "shift-invariance control with a bare NaN": lambda t: t["compare"][
        "shift_invariance"].update({"shifted_final": float("nan")}),
    "shift-invariance control missing a field": lambda t: t["compare"][
        "shift_invariance"].pop("unshifted_final"),
    "shift-invariance control absent": lambda t: t["compare"].pop("shift_invariance"),
    "shift-invariance control unshifted result wrong": lambda t: t["compare"][
        "shift_invariance"].update({"unshifted_final": 99.0}),
}


def sabotage_checks(trace):
    """Prove the lint is not a function that always returns zero.

    A clean report only means something if the same lint fails on a trace that
    is actually broken, so each rule is exercised against a deliberately broken
    copy of the real trace. Returns a list of failures (empty == lint is live).
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
        except Exception as exc:  # a mutation that does not even lint is a failure too
            failures.append(f"{name}: lint raised {exc!r}")
            continue
        if not gaps:
            failures.append(f"{name}: broke the trace but lint reported 0 gaps")
    return failures


# ------------------------------------------------------------------------ main
#
# The parameter grid the page's sliders walk. Every combination here is
# generated and inlined, so a slider position always has a real trace behind it.
# Bc must divide N — a partial final block would change the algorithm (that is
# FlashAttention's tail handling, not L00's subject), so impossible pairs are
# skipped rather than silently rounded.
#
# T is applied as x/T before anything else, which is what temperature means for
# a softmax.
def configs():
    out = []
    for n in (4, 8):
        for bc in (1, 2, 4):
            if n % bc:
                continue
            for temp in (0.5, 1.0, 2.0):
                out.append((n, bc, temp))
    return out


def temp_tag(temp):
    return ("%g" % temp).replace(".", "p")


def cfg_id(n, bc, temp):
    return f"online-softmax-N{n}-B{bc}-T{temp_tag(temp)}"


SOURCE = "docs/guides/模块二-CUDA编程与算子优化/5.2-CUDA Online Softmax实现.md"
DEFAULT_CFG = (8, 4, 1.0)


def main():
    failed = False
    written = []
    for (n, bc, temp) in configs():
        name = cfg_id(n, bc, temp)
        title = f"Online Softmax（1-D，N={n}，Bc={bc}，T={temp:g}）"
        trace = build_trace(SHARED_X, n, bc, temp, title, SOURCE)
        gaps, warns, infos = lint(trace)

        out = HERE / f"{name}.json"
        out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n")
        written.append(name)

        corr = trace["meta"]["correction"]
        print(f"{out.name}: {len(trace['steps'])} 步, {out.stat().st_size} bytes")
        print(f"  {trace['meta']['reference']}")
        print(f"  修正因子生效 {corr['rescales']} 次（共 {trace['meta']['config']['n_blocks']} 块）；"
              f"不修正的最终偏差 {corr['final_bias_abs']:.4g}"
              f"（相对 {100 * corr['final_bias_rel']:.4g}%）")
        cmp_ = trace["compare"]
        print("  对照：" + " · ".join(
            f"{m['label'].split('（')[0]}={m['final']!r}({m['status']}, {m['passes']} 遍)"
            for m in cmp_["methods"]))
        for line in infos:
            print(f"  info  {line}")
        for line in warns:
            print(f"  warn  {line}")
        for line in gaps:
            print(f"  GAP   {line}")
        if gaps:
            failed = True
        print(f"  lint: {len(gaps)} gap / {len(warns)} warn")

        sabotages = sabotage_checks(trace)
        if sabotages:
            failed = True
            for line in sabotages:
                print(f"  LINT-DEAD  {line}")
        else:
            print(f"  lint 对照组：{len(SABOTAGE_CASES)} 种破坏全部被抓到"
                  f"（证明上面的 0 不是 lint 永远报 0）")
        print()

    manifest = {
        "set": "online-softmax",
        "lab": "L00",
        "default": cfg_id(*DEFAULT_CFG),
        "params": {
            "N": sorted({n for (n, _, _) in configs()}),
            "Bc": sorted({bc for (_, bc, _) in configs()}),
            "T": sorted({t for (_, _, t) in configs()}),
        },
        "traces": written,
        "labels": {cfg_id(n, bc, t): f"N={n} · Bc={bc} · T={t:g}" for (n, bc, t) in configs()},
    }
    manifest_path = HERE / "online-softmax.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"online-softmax.manifest.json: {len(written)} 个配置, 默认 {manifest['default']}")

    if failed:
        print("lint reported gaps -- fix the trace before it can be published",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
