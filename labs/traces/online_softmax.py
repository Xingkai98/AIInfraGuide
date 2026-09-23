#!/usr/bin/env python3
"""Trace generator for L00 · Online Softmax.

Emits one JSON file per configuration:

    labs/traces/online-softmax.json       N=4, Bc=2 -> 10 steps
    labs/traces/online-softmax-long.json  N=8, Bc=1 -> 34 steps

The first is the regression sample that ships in
`labs/pages/00-online-softmax.html`. The second is the same algorithm on the
design doc's shared example vector, long enough (>10 layers) that the DAG has to
wrap -- it is the layout regression sample, and it is real arithmetic, not a
padded copy of the short one.

Every number here is computed, not typed in: the numpy recursion below is
asserted step by step against an independent torch implementation, and the final
output is asserted against `torch.softmax`. The script refuses to write a file
if those assertions or the self-lint fail.

Contract notes that this script is the first to exercise (see
`docs/plans/interactive-labs.md` §一 "轨迹数据模型"):

  * `tensors[].init` is mandatory for anything read but never written. `x` is
    read by the first block load and written by no step; without `init`,
    render(trace, 0) cannot show the input at all.
  * +/-inf and NaN are emitted as the strings "-\\infty" / "+\\infty" / "NaN".
    Bare `-Infinity` is not legal JSON. The sentinels appear only in `state`
    values -- formula bindings are LaTeX fragments and spell `-\\infty`
    themselves.
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


# ------------------------------------------------------------------- the math
def run_online_softmax(x, bc):
    """The algorithm under replay: blocked online softmax, one pass.

    Returns the per-block trajectory, so every step of the trace is a real
    intermediate of this loop rather than a reconstruction.
    """
    nb = len(x) // bc
    m, l, acc = float("-inf"), 0.0, 0.0
    run = []
    for j in range(nb):
        blk = x[j * bc:(j + 1) * bc]
        m_old, l_old, acc_old = m, l, acc
        blkmax = float(blk.max())
        m = max(m_old, blkmax)
        # First block has no history: l_old is 0 and the correction factor is
        # defined to be 0 so the update reduces to the plain sum.
        corr = math.exp(m_old - m) if m_old != float("-inf") else 0.0
        p = np.exp(blk - m)
        l = l_old * corr + float(p.sum())
        acc = acc_old * corr + float((p * blk).sum())
        run.append(
            dict(
                j=j, n=j + 1, blk=blk, m_old=m_old, blkmax=blkmax, m_new=m, corr=corr,
                l_old=l_old, l_new=l, acc_old=acc_old, acc_new=acc,
                p=p, psum=float(p.sum()), wsum=float((p * blk).sum()),
            )
        )
    return run, acc / l, m, l, acc


def torch_reference(x, bc):
    """Independent per-block statistics and final output, in torch.

    Deliberately a different implementation from the loop above: the reference
    derives each statistic from the *prefix* of x rather than from the previous
    block's state, so a mistake in the recursion cannot cancel out.
    """
    xt = torch.tensor([float(v) for v in x], dtype=torch.float64)
    blocks = []
    for j in range(len(x) // bc):
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


def check_against_torch(x, bc, run, out):
    """Assert the recursion matches the torch reference, per block and at the end.

    Per-block assertions matter: the final output alone would still match if two
    opposite errors cancelled, and the trace publishes every intermediate.
    """
    ref_blocks, ref_out = torch_reference(x, bc)

    def close(a, b):
        return math.isclose(a, b, rel_tol=_CHECK_REL_TOL, abs_tol=_CHECK_ABS_TOL)
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


# ---------------------------------------------------------------- trace build
def b(slot, sym, idx, num):
    return slot, {"sym": sym, "idx": idx, "num": num}


def step(sid, title, kind, op, phase, formula, bindings, reads, writes, state,
         narration, regions=None):
    d = {"id": sid, "title": title, "kind": kind, "op": op, "phase": phase,
         "formula": formula, "bindings": dict(bindings),
         "reads": reads, "writes": writes, "state": state, "narration": narration}
    if regions:
        d["regions"] = regions
    return d


def build_trace(x, bc, title, source):
    x = np.asarray(x, dtype=np.float64)
    if len(x) % bc:
        raise ValueError(f"N={len(x)} is not a multiple of Bc={bc}")
    nb = len(x) // bc
    run, out, m_end, l_end, acc_end = run_online_softmax(x, bc)
    ref_out = check_against_torch(x, bc, run, out)

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
    ))

    for r in run:
        j, n, blk = r["j"], r["n"], r["blk"]

        # --- kind: "comm" -- reads x, writes only the staging buffer. This is
        # the step that has no data-dependency predecessor and therefore forces
        # the engine to synthesise sequence edges for layout. See the layout
        # regression sample.
        steps.append(step(
            f"ld{n}", f"第 {n} 块载入 SRAM", "comm", "load", "取数",
            {
                "sym": r"x^{(j)} \leftarrow \mathrm{HBM}[\,x\,]\left[\,(j-1)B_c \,:\, jB_c\,\right]",
                "idx": r"x^{(\slot{J})} \leftarrow \mathrm{HBM}[\,x\,]\left[\slot{LO} : \slot{HI}\right]",
                "num": r"x^{(\slot{J})} \leftarrow \mathrm{HBM}[\,x\,]\left[\slot{LO} : \slot{HI}\right] "
                       r"= \slot{VALS}",
            },
            [b("J", "j", str(n), str(n)),
             b("LO", r"(j-1)B_c", str(j * bc), str(j * bc)),
             b("HI", r"jB_c", str((j + 1) * bc), str((j + 1) * bc)),
             b("VALS", r"[\cdot]", r"[\cdot]", mx(blk))],
            ["x"], ["x_blk"],
            {"x_blk": state_list(blk)},
            f"把 x 的第 {n} 块（下标 {j * bc}..{(j + 1) * bc - 1}）从 HBM 搬进 SRAM。"
            f"这一步不改任何标量状态，只是把数据挪个位置。",
        ))

        steps.append(step(
            f"m{n}", f"第 {n} 块：更新运行最大值 m", "op", "max", "统计量更新",
            {
                "sym": r"m^{(j)} = \max\left(\slot{MOLD},\; \slot{BLKMAX}\right)",
                "idx": r"m^{(\slot{J})} = \max\left(\slot{MOLD},\; \slot{BLKMAX}\right)",
                "num": r"m^{(\slot{J})} = \max\left(\slot{MOLD},\; \slot{BLKMAX}\right) = \slot{MNEW}",
            },
            [b("J", "j", str(n), str(n)),
             b("MOLD", r"m^{(j-1)}", rf"m^{{{j}}}", fmt(r["m_old"])),
             b("BLKMAX", r"\max_i x^{(j)}_i", rf"\max_i x^{{{n}}}_i", fmt(r["blkmax"])),
             b("MNEW", r"m^{(j)}", rf"m^{{{n}}}", fmt(r["m_new"]))],
            ["x_blk", "m"], ["m"],
            {"m": state_val(r["m_new"])},
            f"本块最大值 {disp(r['blkmax'])}，"
            + ("比当前的 m 大，m 要抬上去。" if r["blkmax"] > r["m_old"]
               else "没超过当前的 m，m 不变。")
            + ("这是第一块，m 还是 -∞，所以直接取本块最大值。" if r["m_old"] == float("-inf") else ""),
        ))

        # --- the correction factor, boxed by \region{CORR}
        corr_note = (f"修正因子 e^(m_old - m_new) = {disp(r['corr'])}"
                     + ("，第一块时恒为 0" if j == 0 else ""))
        steps.append(step(
            f"l{n}", f"第 {n} 块：更新指数和 ℓ", "op", "sum-exp", "统计量更新",
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
            [b("J", "j", str(n), str(n)),
             b("LOLD", r"\ell^{(j-1)}", rf"\ell^{{{j}}}", fmt(r["l_old"])),
             b("MOLD", r"m^{(j-1)}", rf"m^{{{j}}}", fmt(r["m_old"])),
             b("MNEW", r"m^{(j)}", rf"m^{{{n}}}", fmt(r["m_new"])),
             # The num form is the un-evaluated substitution, not the result --
             # the formula already ends with "= \slot{LNEW}". A binding that
             # repeats its own result renders as "= 1.357 = 1.357".
             b("SUM", r"\sum_i \exp(x^{(j)}_i - m^{(j)})",
               rf"\sum_i \exp(x^{{{n}}}_i - m^{{{n}}})",
               " + ".join(fmt(v) for v in r["p"])),
             b("LNEW", r"\ell^{(j)}", rf"\ell^{{{n}}}", fmt(r["l_new"]))],
            ["x_blk", "m", "l"], ["l"],
            {"l": state_val(r["l_new"])},
            (f"m 变了，之前累加的 ℓ 是按旧的 m 算的，所以要乘修正因子 {disp(r['corr'])} 把它拉回来；"
             f"本块的 " + " + ".join(fmt(v) for v in r["p"]) + f" = {disp(r['psum'])} 再加进去。"
             if j > 0 else
             f"第一块没有历史，修正因子是 0（ℓ 旧值是 0），只加本块的 "
             + " + ".join(fmt(v) for v in r["p"]) + f" = {disp(r['psum'])}。"),
            {"CORR": corr_note},
        ))

        steps.append(step(
            f"a{n}", f"第 {n} 块：更新累加器 acc", "op", "fma", "输出更新",
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
            [b("J", "j", str(n), str(n)),
             b("AOLD", r"\mathrm{acc}^{(j-1)}", rf"\mathrm{{acc}}^{{{j}}}", fmt(r["acc_old"])),
             b("MOLD", r"m^{(j-1)}", rf"m^{{{j}}}", fmt(r["m_old"])),
             b("MNEW", r"m^{(j)}", rf"m^{{{n}}}", fmt(r["m_new"])),
             b("WSUM", r"\sum_i \exp(x^{(j)}_i - m^{(j)})\cdot x^{(j)}_i",
               rf"\sum_i \exp(x^{{{n}}}_i - m^{{{n}}})\cdot x^{{{n}}}_i",
               plus_terms(list(zip(r["p"], blk)))),
             b("ANEW", r"\mathrm{acc}^{(j)}", rf"\mathrm{{acc}}^{{{n}}}", fmt(r["acc_new"]))],
            ["x_blk", "m", "acc"], ["acc"],
            {"acc": state_val(r["acc_new"])},
            (f"和 ℓ 用同一个修正因子 {disp(r['corr'])}，把旧的 acc 缩回来，再加上本块的加权和 "
             + plus_terms_disp(list(zip(r["p"], blk))) + f" = {disp(r['wsum'])}。"
             + "这就是 FlashAttention 里对 O 做 rescale 的那一步。"),
            {"CORR": corr_note},
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
        f"就是 softmax 加权平均，与 torch 的参考实现逐位一致。",
    ))

    # ---------------------------------------------------------------- graph
    # Nodes carry no `step` field on purpose: ids are shared with `steps`, so the
    # engine derives the index. A redundant field could only ever drift.
    nodes = [{"id": s["id"], "title": s["title"], "kind": s["kind"],
              "op": s["op"], "phase": s["phase"]} for s in steps]

    edges = [{"from": "init", "to": "m1", "tensor": "m"},
             {"from": "init", "to": "l1", "tensor": "l"},
             {"from": "init", "to": "a1", "tensor": "acc"}]
    for n in range(1, nb + 1):
        edges += [{"from": f"ld{n}", "to": f"m{n}", "tensor": "x_blk"},
                  {"from": f"m{n}", "to": f"l{n}", "tensor": "m"},
                  {"from": f"m{n}", "to": f"a{n}", "tensor": "m"},
                  {"from": f"l{n}", "to": f"a{n}", "tensor": "l"}]
        if n < nb:
            edges += [{"from": f"m{n}", "to": f"m{n + 1}", "tensor": "m"},
                      {"from": f"l{n}", "to": f"l{n + 1}", "tensor": "l"},
                      {"from": f"a{n}", "to": f"a{n + 1}", "tensor": "acc"}]
        else:
            edges += [{"from": f"a{n}", "to": "fin", "tensor": "acc"},
                      {"from": f"l{n}", "to": "fin", "tensor": "l"}]

    return {
        "meta": {
            "lab": "L00",
            "title": title,
            "source": source,
            "config": {"N": len(x), "Bc": bc, "n_blocks": nb},
            "reference": f"O = softmax(x)·x = {ref_out!r} (torch float64, "
                         f"per-block statistics also cross-checked)",
        },
        "tensors": {
            # `init` is the contract addition: no step writes x, but the first
            # block load reads it, so without this render(trace, 0) has nothing
            # to show for the input.
            "x": {"shape": [len(x)], "dtype": "fp32", "at": "HBM", "role": "input",
                  "init": [float(v) for v in x], "note": "整条序列，全程驻留 HBM"},
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
        "steps": steps,
    }


# -------------------------------------------------------------------- self-lint
SLOT_RE = re.compile(r"\\slot\{([A-Za-z0-9_]+)\}")
REGION_RE = re.compile(r"\\region\{([A-Za-z0-9_]+)\}")
LATEX_IN_TEXT_RE = re.compile(r"\\[a-zA-Z]+\{")
BAD_LITERAL_RE = re.compile(r"^(NaN|Infinity|-Infinity|undefined|null)$")


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

    infos.append(f'共 {len(trace["steps"])} 步 / {len(trace["graph"]["nodes"])} 节点 / '
                 f'{len(trace["graph"]["edges"])} 数据边')
    infos.append(f'state 写过的张量：{", ".join(sorted(written))}')
    return gaps, warns, infos


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
CONFIGS = [
    dict(
        filename="online-softmax.json",
        x=[0.83, -0.20, 1.40, 0.35],
        bc=2,
        title="Online Softmax（1-D，N=4，Bc=2）",
    ),
    dict(
        # The design doc's shared example vector, at Bc=2: 4 blocks, 18 steps.
        # Same algorithm on real arithmetic -- its job is to be long enough
        # (18 layers) that the DAG must wrap, instead of degenerating into the
        # 8:1 strip a single-row layout produces past ~10 layers.
        filename="online-softmax-long.json",
        x=[0.83, -0.20, 1.40, 0.35, 0.12, -1.05, 2.10, 0.44],
        bc=2,
        title="Online Softmax（1-D，N=8，Bc=2 · 布局回归样例）",
    ),
]

SOURCE = "docs/guides/模块二-CUDA编程与算子优化/5.2-CUDA Online Softmax实现.md"


def main():
    failed = False
    for cfg in CONFIGS:
        trace = build_trace(cfg["x"], cfg["bc"], cfg["title"], SOURCE)
        gaps, warns, infos = lint(trace)

        out = HERE / cfg["filename"]
        out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n")

        steps = len(trace["steps"])
        print(f"{out.name}: {steps} 步 / {len(trace['graph']['nodes'])} 节点, "
              f"{out.stat().st_size} bytes")
        print(f"  {trace['meta']['reference']}")
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

    if failed:
        print("lint reported gaps -- fix the trace before it can be published",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
