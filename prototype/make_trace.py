#!/usr/bin/env python3
"""PROTOTYPE trace generator for issue #12.

Emits prototype/trace.js (window.PROTO_TRACE = {...}).

Not a real labs/traces/*.py script: no torch, no CI assertions. But the numbers
ARE computed here with numpy, not typed in by hand -- the online-softmax
recursion is run for real and checked against a reference softmax at the bottom,
so the page shows real arithmetic even though the trace is throwaway.

Chosen algorithm: 1-D online softmax, x in R^4, block size Bc = 2 (2 blocks).
10 graph nodes (2 structural + 4 per block), which is the "<= 10 nodes" budget
the ticket asks about.

Emitted as a .js file rather than .json on purpose: JSON cannot represent -inf
and m starts at -inf. Python's json.dumps emits the bare token `-Infinity`,
which is valid JS, so the escape hatch is free.
"""

import json
import math
from pathlib import Path

import numpy as np

X = np.array([0.83, -0.20, 1.40, 0.35], dtype=np.float64)
BC = 2
NB = len(X) // BC


def fmt(v):
    """LaTeX form. Never put this in narration -- use disp()."""
    if v == float("-inf"):
        return r"-\infty"
    if v == float("inf"):
        return r"+\infty"
    if v == 0:
        return "0"
    return "%.4g" % v


def disp(v):
    """Unicode form for prose. fmt() returns LaTeX, which would render as
    literal backslashes if it leaked into a narration string."""
    if v == float("-inf"):
        return "-∞"
    if v == float("inf"):
        return "+∞"
    if v == 0:
        return "0"
    return "%.4g" % v


def mx(values):
    return r"\left[" + ", ".join(fmt(v) for v in values) + r"\right]"


def paren(v):
    """Wrap negatives: `0.357\\cdot -0.2` reads as a typo, `0.357\\cdot(-0.2)` does not."""
    return rf"\left({fmt(v)}\right)" if v < 0 else fmt(v)


def plus_terms(pairs, sep=" + "):
    """LaTeX sum of p*v products."""
    return sep.join(f"{fmt(p)}\\cdot {paren(v)}" for p, v in pairs)


def plus_terms_disp(pairs, sep=" + "):
    return sep.join(f"{disp(p)}·{disp(v)}" if v >= 0 else f"{disp(p)}·({disp(v)})" for p, v in pairs)


# ---------------------------------------------------------------- real run
m = float("-inf")
l = 0.0
acc = 0.0
run = []

for j in range(NB):
    blk = X[j * BC:(j + 1) * BC]
    m_old, l_old, acc_old = m, l, acc
    blkmax = float(blk.max())
    m = max(m_old, blkmax)
    corr = math.exp(m_old - m) if m_old != float("-inf") else 0.0
    p = np.exp(blk - m)
    l = l_old * corr + float(p.sum())
    acc = acc_old * corr + float((p * blk).sum())
    run.append(dict(j=j, n=j + 1, blk=blk, m_old=m_old, blkmax=blkmax, m_new=m, corr=corr,
                    l_old=l_old, l_new=l, acc_old=acc_old, acc_new=acc, p=p,
                    psum=float(p.sum()), wsum=float((p * blk).sum())))

O = acc / l  # softmax-weighted mean of x: what FA computes for a single query with V = x

# reference check -- proof for the report, not a CI assertion
_p = np.exp(X - X.max())
ref = float((_p * X).sum() / _p.sum())
assert abs(ref - O) < 1e-12, (ref, O)


# ---------------------------------------------------------------- builders
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


steps = []

# --- the state node (kind = "state"), deliberately included
steps.append(step(
    "init", "初始化", "state", "init", "准备",
    {t: r"m^{(0)} = \slot{m0},\quad \ell^{(0)} = \slot{l0},\quad "
        r"\mathrm{acc}^{(0)} = \slot{a0}" for t in ("sym", "idx", "num")},
    [b("m0", r"-\infty", r"-\infty", r"-\infty"),
     b("l0", "0", "0", "0"),
     b("a0", "0", "0", "0")],
    [], ["m", "l", "acc"],
    {"m": float("-inf"), "l": 0.0, "acc": 0.0},
    "开始时 m 是 -∞ 而不是 0。如果这里写 0，遇到一整块全是负数的输入，"
    "最大值就会被 0 顶掉，后面 exp 全溢出。",
))

for r in run:
    j, n = r["j"], r["n"]
    blk = r["blk"]

    # --- the "comm" node: does kind=comm earn its keep, or is it awkward?
    steps.append(step(
        f"ld{n}", f"第 {n} 块载入 SRAM", "comm", "load", "取数",
        {
            "sym": r"x^{(j)} \leftarrow \mathrm{HBM}[\,x\,]\left[\,(j-1)B_c \,:\, jB_c\,\right]",
            "idx": r"x^{(\slot{J})} \leftarrow \mathrm{HBM}[\,x\,]\left[\slot{LO} : \slot{HI}\right]",
            "num": r"x^{(\slot{J})} \leftarrow \mathrm{HBM}[\,x\,]\left[\slot{LO} : \slot{HI}\right] "
                   r"= \slot{VALS}",
        },
        [b("J", "j", str(n), str(n)),
         b("LO", r"(j-1)B_c", str(j * BC), str(j * BC)),
         b("HI", r"jB_c", str((j + 1) * BC), str((j + 1) * BC)),
         b("VALS", r"[\cdot]", r"[\cdot]", mx(blk))],
        ["x"], ["x_blk"],
        {"x_blk": [float(v) for v in blk]},
        f"把 x 的第 {n} 块（下标 {j*BC}..{(j+1)*BC-1}）从 HBM 搬进 SRAM。"
        f"这一步不改任何标量状态，只是把数据挪个位置。",
    ))

    # --- m
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
        {"m": r["m_new"]},
        f"本块最大值 {disp(r['blkmax'])}，"
        + ("比当前的 m 大，m 要抬上去。" if r["blkmax"] > r["m_old"]
           else "没超过当前的 m，m 不变。")
        + ("这是第一块，m 还是 -∞，所以直接取本块最大值。" if r["m_old"] == float("-inf") else ""),
    ))

    # --- l  (region CORR = the correction factor, the thing L00 wants to amplify)
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
         # num form is the un-evaluated substitution, NOT the final value --
         # the enclosing formula already supplies the trailing "= \slot{LNEW}".
         # (Prototype finding: a binding whose num form repeats its own result
         # makes the rendered formula read "= 1.357 = 1.357".)
         b("SUM", r"\sum_i \exp(x^{(j)}_i - m^{(j)})",
           rf"\sum_i \exp(x^{{{n}}}_i - m^{{{n}}})",
           " + ".join(fmt(v) for v in r["p"])),
         b("LNEW", r"\ell^{(j)}", rf"\ell^{{{n}}}", fmt(r["l_new"]))],
        ["x_blk", "m", "l"], ["l"],
        {"l": r["l_new"]},
        (f"m 变了，之前累加的 ℓ 是按旧的 m 算的，所以要乘修正因子 {disp(r['corr'])} 把它拉回来；"
         f"本块的 " + " + ".join(fmt(v) for v in r["p"]) + f" = {disp(r['psum'])} 再加进去。"
         if j > 0 else
         f"第一块没有历史，修正因子是 0（ℓ 旧值是 0），只加本块的 "
         + " + ".join(fmt(v) for v in r["p"]) + f" = {disp(r['psum'])}。"),
        {"CORR": f"修正因子 e^(m_old - m_new) = {disp(r['corr'])}"
                 + ("，第一块时恒为 0" if j == 0 else "")},
    ))

    # --- acc
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
        {"acc": r["acc_new"]},
        (f"和 ℓ 用同一个修正因子 {disp(r['corr'])}，把旧的 acc 缩回来，再加上本块的加权和 "
         + plus_terms_disp(list(zip(r["p"], blk))) + f" = {disp(r['wsum'])}。"
         + "这就是 FlashAttention 里对 O 做 rescale 的那一步。"),
        {"CORR": f"修正因子 e^(m_old - m_new) = {disp(r['corr'])}"
                 + ("，第一块时恒为 0" if j == 0 else "")},
    ))

steps.append(step(
    "fin", "收尾：归一化", "op", "div", "收尾",
    {t: (r"O = \slot{ACC} \,/\, \slot{L}"
         + (r" = \slot{ONUM}" if t == "num" else "")) for t in ("sym", "idx", "num")},
    [b("ACC", r"\mathrm{acc}^{(J)}", r"\mathrm{acc}^{(2)}", fmt(acc)),
     b("L", r"\ell^{(J)}", r"\ell^{(2)}", fmt(l)),
     b("ONUM", "O", "O", fmt(O))],
    ["acc", "l"], ["O"],
    {"O": O},
    f"整条 trace 只在这里除一次。O = {disp(acc)} / {disp(l)} = {disp(O)}，"
    f"就是 softmax 加权平均，和 numpy 的三遍参考实现逐位一致。",
))

# ---------------------------------------------------------------- graph
nodes = [{"id": s["id"], "step": i, "title": s["title"], "kind": s["kind"],
          "op": s["op"], "phase": s["phase"]} for i, s in enumerate(steps)]

edges = [{"from": "init", "to": "m1", "tensor": "m"},
         {"from": "init", "to": "l1", "tensor": "l"},
         {"from": "init", "to": "a1", "tensor": "acc"}]
for n in range(1, NB + 1):
    edges += [{"from": f"ld{n}", "to": f"m{n}", "tensor": "x_blk"},
              {"from": f"m{n}", "to": f"l{n}", "tensor": "m"},
              {"from": f"m{n}", "to": f"a{n}", "tensor": "m"},
              {"from": f"l{n}", "to": f"a{n}", "tensor": "l"}]
    if n < NB:
        edges += [{"from": f"m{n}", "to": f"m{n+1}", "tensor": "m"},
                  {"from": f"l{n}", "to": f"l{n+1}", "tensor": "l"},
                  {"from": f"a{n}", "to": f"a{n+1}", "tensor": "acc"}]
    else:
        edges += [{"from": f"a{n}", "to": "fin", "tensor": "acc"},
                  {"from": f"l{n}", "to": "fin", "tensor": "l"}]

trace = {
    "meta": {
        "title": "Online Softmax（1-D，Bc=2）",
        "source": "prototype for issue #12 — NOT a labs/traces script",
        "config": {"N": len(X), "Bc": BC, "n_blocks": NB},
        "reference": f"softmax-weighted mean of x = {O!r} (numpy, 3-pass)",
    },
    "tensors": {
        # `init` is an addition to the draft contract -- see the report. Without
        # it a pure render(trace, i) cannot know x's value at all: no step writes x.
        "x": {"shape": [len(X)], "dtype": "fp32", "at": "HBM", "role": "input",
              "init": [float(v) for v in X], "note": "整条序列，全程驻留 HBM"},
        "x_blk": {"shape": [BC], "dtype": "fp32", "at": "SRAM", "role": "staging",
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

# ---------------------------------------------------------------- emit
out = Path(__file__).with_name("trace.js")
out.write_text(
    "// GENERATED by make_trace.py -- do not edit by hand.\n"
    "// Regenerate: python3 prototype/make_trace.py\n"
    "window.PROTO_TRACE = "
    + json.dumps(trace, indent=1, ensure_ascii=False) + ";\n"
)
print(f"wrote {out} ({out.stat().st_size} bytes)")
print(f"steps={len(steps)} nodes={len(nodes)} edges={len(edges)}")
print(f"x={list(X)}  m={m} l={l} acc={acc} O={O}")
print(f"numpy reference O = {ref}  match={abs(ref-O) < 1e-12}")
for r in run:
    print(f"  block{r['n']}: m_old={r['m_old']} blkmax={r['blkmax']} m_new={r['m_new']} "
          f"corr={r['corr']:.6f} l:{r['l_old']:.6f}->{r['l_new']:.6f} "
          f"acc:{r['acc_old']:.6f}->{r['acc_new']:.6f}")
