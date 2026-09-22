#!/usr/bin/env python3
"""R02 — Generate simulated lab traces and measure their serialized size.

The traces are *shape-faithful simulations*, not real PyTorch runs: the math is
genuine (online softmax / FlashAttention recurrences, executed step by step) but
the inputs come from a seeded RNG instead of a model. That is enough to pin down
the wire size, which is what R02 is about.

Two labs are generated, matching the design doc's parameter table:

  L06 FlashAttention   N=8, d=16, B_r=4, B_c=4   -> outer 2 x inner 2 = 4 steps
  L00 Online Softmax   N=8, B_c=4                -> 2 blocks + normalise = 3 steps

Float precision is the dominant term in trace size, so each trace is emitted at
four precisions and measured. Only `sig6` is a real candidate for shipping;
the others bracket the decision.

Usage:
    python3 docs/research/scripts/gen_trace.py [--outdir DIR] [--write]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import struct
import sys

import numpy as np

SEED = 20260922


# --------------------------------------------------------------------------
# float formatting
# --------------------------------------------------------------------------

def round_sig(x: float, sig: int) -> float:
    """Round to `sig` significant digits."""
    if x == 0.0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def round_f32(x: float) -> float:
    """Round-trip through float32 — what a GPU kernel would actually hold."""
    return struct.unpack("f", struct.pack("f", x))[0]


def clean(x):
    """Recursively replace non-finite floats with None so the JSON stays valid."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


class Quant:
    """Applies one precision policy to every float in a trace."""

    def __init__(self, name: str, fn):
        self.name = name
        self.fn = fn

    def __call__(self, x):
        if isinstance(x, bool):
            return x
        if isinstance(x, float):
            return self.fn(x)
        if isinstance(x, np.floating):
            return self.fn(float(x))
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.ndarray):
            return self(x.tolist())
        if isinstance(x, dict):
            return {k: self(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self(v) for v in x]
        return x


PRECISIONS = [
    Quant("full", lambda v: v),                        # Python repr, ~17 sig digits
    Quant("f32", round_f32),                           # float32 round-trip
    Quant("sig6", lambda v: round_sig(v, 6)),
    Quant("dec4", lambda v: round(v, 4)),
]


def fmt_vec(a: np.ndarray) -> dict:
    return {"shape": list(a.shape), "values": a}


# --------------------------------------------------------------------------
# L00 — Online Softmax
# --------------------------------------------------------------------------

def build_online_softmax(N: int = 8, Bc: int = 4) -> dict:
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.0, 2.0, size=N).astype(np.float64)

    steps = []
    m = -math.inf
    l = 0.0
    O = np.zeros(N, dtype=np.float64)

    n_blocks = math.ceil(N / Bc)
    for j in range(n_blocks):
        lo, hi = j * Bc, min((j + 1) * Bc, N)
        x_blk = x[lo:hi]
        m_prev, l_prev = m, l
        m_new = max(m, float(x_blk.max()))
        corr = 0.0 if not math.isfinite(m_prev) else math.exp(m_prev - m_new)

        # rescale everything written so far, then write this block's exponentials
        O = O * corr
        O[lo:hi] = np.exp(x_blk - m_new)
        l = l * corr + float(np.exp(x_blk - m_new).sum())
        m = m_new

        steps.append({
            "id": f"step{j + 1}.block",
            "title": f"第 {j + 1} 块：更新运行最大值 m 与归一化常数 ℓ",
            "formula": r"m^{(j)} = \max\left(m^{(j-1)},\; \max_i x^{(j)}_i\right)",
            "bindings": {
                "j": j + 1,
                r"m^{(j-1)}": m_prev,
                r"\max_i x^{(j)}_i": float(x_blk.max()),
                r"m^{(j)}": m_new,
                r"e^{m^{(j-1)} - m^{(j)}}": corr,
                r"\ell^{(j)}": l,
            },
            "reads": ["x", "x_block"],
            "writes": ["m", "l", "O"],
            "state": {"m": m, "l": l},
            "narration": (
                "第一次进来时 m 是 -∞，修正因子为 0，所以历史累加量整体作废。"
                if j == 0 else
                f"m 从 {m_prev:.4f} 抬到 {m_new:.4f}，历史累加量整体乘上修正因子 {corr:.6f}。"
            ),
            "frame": {
                "x": fmt_vec(x),
                "x_block": fmt_vec(x_blk),
                "O": fmt_vec(O.copy()),
                "scalars": {"m_prev": m_prev, "m_new": m_new, "l": l, "corr": corr},
            },
        })

    O_final = O / l
    steps.append({
        "id": "step3.normalize",
        "title": "收尾：用 ℓ 归一化",
        "formula": r"O = \frac{O}{\ell}",
        "bindings": {r"\ell": l},
        "reads": ["O", "l"], "writes": ["O"],
        "state": {"m": m, "l": l},
        "narration": "全部块扫完之后才做唯一一次除法，这就是 online softmax 的全部代价。",
        "frame": {"O": fmt_vec(O_final), "scalars": {"l": l}},
    })

    return {
        "meta": {
            "title": "Online Softmax",
            "source": "docs/guides/第2章-数学基础/5.3",
            "config": {"N": N, "Bc": Bc},
        },
        "tensors": {
            "x": {"shape": [N], "dtype": "fp32", "label": "HBM"},
            "x_block": {"shape": [Bc], "dtype": "fp32", "label": "SRAM"},
            "O": {"shape": [N], "dtype": "fp32", "label": "SRAM"},
            "m": {"shape": [], "dtype": "fp32", "label": "SRAM"},
            "l": {"shape": [], "dtype": "fp32", "label": "SRAM"},
        },
        "graph": {
            "nodes": [
                {"id": "x", "op": "input", "phase": "输入"},
                *[{"id": f"step{j + 1}.block", "op": "block-max-exp", "phase": "分块递推"}
                  for j in range(n_blocks)],
                {"id": "step3.normalize", "op": "div", "phase": "归一化"},
            ],
            "edges": [
                *[{"from": "x", "to": f"step{j + 1}.block", "tensor": "x_block"}
                  for j in range(n_blocks)],
                *[{"from": f"step{j + 1}.block", "to": "step3.normalize", "tensor": "O"}
                  for j in range(n_blocks)],
            ],
        },
        "steps": steps,
    }


# --------------------------------------------------------------------------
# L06 — FlashAttention V1
# --------------------------------------------------------------------------

def build_flash_attention(N: int = 8, d: int = 16, Br: int = 4, Bc: int = 4) -> dict:
    rng = np.random.default_rng(SEED + 1)
    Q = rng.normal(0.0, 0.5, size=(N, d))
    K = rng.normal(0.0, 0.5, size=(N, d))
    V = rng.normal(0.0, 0.5, size=(N, d))

    n_i, n_j = math.ceil(N / Br), math.ceil(N / Bc)
    steps = []

    for i in range(n_i):
        qlo, qhi = i * Br, min((i + 1) * Br, N)
        Q_i = Q[qlo:qhi]
        m_i = np.full(Q_i.shape[0], -math.inf)
        l_i = np.zeros(Q_i.shape[0])
        O_i = np.zeros_like(Q_i)

        for j in range(n_j):
            klo, khi = j * Bc, min((j + 1) * Bc, N)
            K_j, V_j = K[klo:khi], V[klo:khi]
            m_prev, l_prev = m_i.copy(), l_i.copy()

            S_ij = Q_i @ K_j.T / math.sqrt(d)
            m_new = np.maximum(m_i, S_ij.max(axis=1))
            corr = np.exp(m_prev - m_new)          # -inf -> 0, the first-block case
            P_ij = np.exp(S_ij - m_new[:, None])
            l_i = l_prev * corr + P_ij.sum(axis=1)
            O_i = O_i * corr[:, None] + P_ij @ V_j
            m_i = m_new

            steps.append({
                "id": f"step{i * n_j + j + 1}.s{i + 1}{j + 1}",
                "title": f"外循环 i={i + 1} / 内循环 j={j + 1}：S 与 P 只在 SRAM 里出现",
                "formula": (
                    r"S_{ij} = Q_i K_j^\top / \sqrt{d},\quad "
                    r"P_{ij} = \exp(S_{ij} - m^{(new)}),\quad "
                    r"O_i \leftarrow O_i e^{m^{(old)} - m^{(new)}} + P_{ij} V_j"
                ),
                "bindings": {
                    "i": i + 1, "j": j + 1, "d": d,
                    r"\sqrt{d}": math.sqrt(d),
                    r"m^{(old)}": list(m_prev),
                    r"m^{(new)}": list(m_new),
                    r"e^{m^{(old)}-m^{(new)}}": list(corr),
                },
                "reads": ["Q_i", "K_j", "V_j", "O_i", "m_i", "l_i"],
                "writes": ["O_i", "m_i", "l_i"],
                # S and P are deliberately NOT in `state`: they never leave SRAM.
                "state": {
                    "m_i": list(m_i), "l_i": list(l_i),
                    "sram_bytes": (Q_i.nbytes + K_j.nbytes + V_j.nbytes
                                   + O_i.nbytes + S_ij.nbytes),
                },
                "narration": (
                    f"载入 K_{j + 1}, V_{j + 1}（各 {Bc}×{d}），算出的 S/P 是 "
                    f"{S_ij.shape[0]}×{S_ij.shape[1]}，用完即弃，从不落回 HBM。"
                ),
                "frame": {
                    "Q_i": fmt_vec(Q_i), "K_j": fmt_vec(K_j), "V_j": fmt_vec(V_j),
                    "S_ij": fmt_vec(S_ij), "P_ij": fmt_vec(P_ij),
                    "O_i": fmt_vec(O_i.copy()),
                    "scalars": {
                        "m_prev": list(m_prev), "m_new": list(m_new),
                        "l_prev": list(l_prev), "l": list(l_i), "corr": list(corr),
                    },
                },
            })

    return {
        "meta": {
            "title": "FlashAttention V1",
            "source": "docs/guides/6.1-FlashAttention V1详解",
            "config": {"N": N, "d": d, "Br": Br, "Bc": Bc},
        },
        "tensors": {
            "Q_i": {"shape": [Br, d], "dtype": "fp32", "label": "SRAM"},
            "K_j": {"shape": [Bc, d], "dtype": "fp32", "label": "SRAM"},
            "V_j": {"shape": [Bc, d], "dtype": "fp32", "label": "SRAM"},
            "S_ij": {"shape": [Br, Bc], "dtype": "fp32", "label": "SRAM"},
            "P_ij": {"shape": [Br, Bc], "dtype": "fp32", "label": "SRAM"},
            "O_i": {"shape": [Br, d], "dtype": "fp32", "label": "SRAM"},
            "m_i": {"shape": [Br], "dtype": "fp32", "label": "SRAM"},
            "l_i": {"shape": [Br], "dtype": "fp32", "label": "SRAM"},
        },
        "graph": {
            "nodes": [
                {"id": "Q", "op": "input", "phase": "输入"},
                *[{"id": f"step{i * n_j + j + 1}.s{i + 1}{j + 1}",
                   "op": "fa-block", "phase": f"Q 块 {i + 1} × K/V 块 {j + 1}"}
                  for i in range(n_i) for j in range(n_j)],
            ],
            "edges": [
                {"from": "Q", "to": "step1.s11", "tensor": "Q_i"},
                *[{"from": f"step{k}.s{k // n_j + 1}{k % n_j + 1}",
                   "to": f"step{k + 1}.s{(k + 1) // n_j + 1}{(k + 1) % n_j + 1}",
                   "tensor": "O_i"}
                  for k in range(1, n_i * n_j)],
            ],
        },
        "steps": steps,
    }


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure(obj) -> dict:
    """Serialized size of one quantised trace, at several JSON layouts."""
    out = {}
    for layout, kwargs in (
        ("compact", {"separators": (",", ":"), "ensure_ascii": False}),
        ("pretty", {"indent": 2, "ensure_ascii": False}),
    ):
        blob = json.dumps(clean(obj), **kwargs).encode("utf-8")
        out[layout] = {
            "raw": len(blob),
            "gzip": len(gzip.compress(blob, 9)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="write trace JSONs here (one file per lab, sig6 precision)")
    ap.add_argument("--write", action="store_true", help="actually write the files")
    args = ap.parse_args()

    labs = {
        "L06-flash-attention": build_flash_attention(),
        "L00-online-softmax": build_online_softmax(),
    }

    report = {}
    print(f"{'lab':<22} {'prec':<6} {'layout':<8} {'raw':>9} {'gzip':>9}")
    print("-" * 60)
    for name, trace in labs.items():
        report[name] = {}
        for q in PRECISIONS:
            m = measure(q(trace))
            report[name][q.name] = m
            for layout in ("compact", "pretty"):
                print(f"{name:<22} {q.name:<6} {layout:<8} "
                      f"{m[layout]['raw']:>9} {m[layout]['gzip']:>9}")

    # Sanity: the trace must still be valid JSON after quantisation, and the
    # final L00 output must be a real softmax (i.e. it sums to ~1).
    x = np.array(build_online_softmax()["steps"][0]["frame"]["x"]["values"])
    got = np.array(build_online_softmax()["steps"][2]["frame"]["O"]["values"])
    want = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    assert np.allclose(got, want, atol=1e-9), "online softmax trace is wrong"
    print("\n[check] L00 trace output matches numpy softmax to 1e-9")

    if args.write and args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        for name, trace in labs.items():
            q = next(p for p in PRECISIONS if p.name == "sig6")
            blob = json.dumps(clean(q(trace)), separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8")
            path = os.path.join(args.outdir, f"{name}.trace.json")
            with open(path, "wb") as fh:
                fh.write(blob)
            print(f"[write] {path}  {len(blob)} bytes")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "size-report.json"), "w") as fh:
            json.dump(report, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
