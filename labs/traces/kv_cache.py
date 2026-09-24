#!/usr/bin/env python3
"""Trace generator for L05 · KV Cache and autoregressive decoding.

Emits a trace SET — one trace per slider position, plus the manifest that names
them (see "Trace sets" in `labs/traces/README.md`):

    labs/traces/kv-cache.json                 N=8,  L=2, fp16  (the default)
    labs/traces/kv-cache-N128-L2-fp16.json    …
    labs/traces/kv-cache.manifest.json        written by this script

over `(N, L, precision)` = {8, 128, 256} × {2, 4} × {fp16, fp8, int8}. The
context slider walks N, which is the sequence length at the end of the replay
(prompt N-4, then 4 generated tokens), so the *shape* of the replay is the same
in every configuration — 5 phases × 4 sub-steps = 20 steps — while the context
the prefill swallows and the decode attends over grows by 32×.

**Why the default keeps the bare name `kv-cache`.** Ticket #45's acceptance page
(`labs/pages/view-ledger.html`) and its harness (`scripts/verify-ledger.py`) pin
that filename and inline it by name. Naming the default configuration's file
after its parameters would have renamed an artifact another ticket shipped, for
no gain: the page never parses an id. It matches a slider position to a trace by
reading the trace's own `meta.config`, which is a stronger check than string
parsing and needs no naming convention at all.

This is also the fixture the **memory-ledger view component** (`labs/assets/
engine/views/ledger.js`, shipped by ticket #45) is validated against. The view
is a renderer: it consumes `trace.ledger` (the segment declarations) and
`step.ledger` (the byte counts), and it carries a pure re-implementation of the
accounting. The two must agree exactly, which is only a meaningful check because
this script computes its side from real numpy arrays rather than from the same
formula re-typed. Two more of this lab's views carry their contracts the same
way: `views/attn-shape.js` (the with/without-cache attention comparison, from
`step.attn_cost` / `meta.attn_cost`) and `views/phase-roofline.js` (the
Prefill/Decode Roofline, from `meta.phase_roofline`).

Four independent checks run before anything is written:

  1. **torch, no cache.** The cached path here is asserted, token by token,
     against a torch implementation that recomputes the whole prefix from
     scratch with a full causal mask. That is the lab's central claim made
     numeric: with a cache and without one, the last position's logits are the
     same number. A bug in the cache bookkeeping cannot cancel out, because the
     reference never touches a cache.
  2. **numpy, both paths.** The cached path is also compared against this
     script's own no-cache full-prefix forward, so a mistake that lives in the
     numpy code alone is caught before torch is even consulted.
  3. **The ledger's byte accounting**, asserted twice — once against the real
     arrays (`arr.size`), once against the closed-form formulas the two guides
     give (params = Σ numel(Wᵢ)·b, KV = 2·L·H_kv·d_h·N·B·b). The two are written
     so that a mismatch in either direction fails.
  4. **The contract lint**, plus a control group of deliberately broken copies
     that the lint must catch. Without the control group a clean report would
     only prove the lint always returns zero.

The script refuses to write a file if any of the four fails.

## Two deliberate contract decisions this script has to make

**The growing KV cache is not snapshotted into `step.state`.** The data model
requires `state` to hold full values, never deltas, so a faithful replay would
re-snapshot the whole cache every step — O(N²) volume that dwarfs the rest of
the trace (≈10⁴ floats here, and it grows with the square of the context). The
design doc leaves this call to L05:

    "KV cache 逐 token 追加若不想记全量快照（O(N²) 体积），需要 delta + 反向操作。
     → L05/L09 定。契约预留可选的 snapshot / inverse 字段，但现在不实现。"

Since the component this trace exists to validate needs the cache's *size*, not
its contents, the cache's byte accounting is recorded per step (`step.ledger`)
and `step.state` carries only the rows appended by that step (`k_new` / `v_new`)
— genuinely new tensors, not deltas of an existing one. The full-snapshot path
stays unimplemented, exactly as the design doc says.

**Variable-length tensors declare their terminal shape.** `tokens`, `h` and
`attn` all grow during the replay, and the contract has one static `shape` per
tensor. They therefore declare the shape they end at; the tensor grid lays out
the cells that actually exist, so only the *label* reads as capacity. This is a
real gap in the data model (a per-step shape override, or `shape: [null, …]`,
would close it) and is called out here rather than papered over — see the note
in `labs/README.md`.

## The with/without-cache comparison, and why it is MEASURED

The lab's central contrast is between two ways to compute the same attention:
without a cache, every step re-runs the whole prefix (`[t,d] × [t,d]` per head);
with one, only the new queries run (`[n_q,d] × [t,d]`). Publishing that contrast
as a shape formula would make it an assertion about what the code does rather
than a reading of what it did — and the two differ in exactly the case that
matters, because `n_q` is `prompt_len` at prefill and 1 at decode.

So `step.attn_cost` is read off the **real arrays of both paths**: the cached
path's score matrix is `live["attn"].arrays["scores"]`, and the no-cache path's
is the score matrix of the full-prefix forward this script already runs for its
torch cross-check. Both are asserted against the shape formula in the same
block, so a mismatch in either direction fails the build. The comparison is
therefore a measurement of two computations that both happened.

At prefill the two modes are IDENTICAL — with an empty cache there is no history
to recompute — and that coincidence is asserted too. It is the control that
makes the decode divergence mean something: a lab whose two modes differed
everywhere would be showing a difference it never isolated.

## The phase Roofline, and why it is not `meta.roofline`

`meta.phase_roofline` places Prefill and Decode on the tutorial's Roofline
(`1.1-LLM推理基础.md` §4): prefill's arithmetic intensity rises with the prompt
(weights are read once for every token in the prompt), while decode sits at
≈1 FLOP/Byte no matter how long the context gets — the two statements the
tutorial makes, computed here instead of quoted.

It is a *separate key* rather than `meta.roofline` on purpose. The shared
Roofline lint in `labs/assets/engine/views/roofline.js` is bound to L01's GEMM
roles (`naive`/`tiled`/`doc`, `2MNK` FLOPs, `meta.traffic` counters) and would
report every L05 trace as broken. The repo's own rule for a shared component is
that its rules are *gated on the field the trace declares* — that is how
`tiling-stage.js` serves four labs without turning three of them red — and
roofline.js's lint is not gated. Rather than rewrite another ticket's contract
file, L05 declares a different key, and `views/phase-roofline.js` (which renders
through `roofline.js`'s `makeView`, unchanged) carries L05's rules.

## Precision and tolerance

The reference arithmetic runs in **float64**, so the only error between the
numpy path and the torch path is summation order and library internals; the
tolerance can be far tighter than any tolerance quoted for fp16/fp8 inference.
It is tight deliberately: at the 4 significant digits the lab displays, a loose
tolerance would hide a wrong formula that happens to round to the same number
on screen. `_REL_TOL` below is calibrated against an actual measured maximum
rather than guessed — see the constant.

The **byte accounting is a separate matter**: it uses the *deployment* dtype
(fp16 = 2 bytes per element), not the dtype the reference computes in. Storage
cost is what the ledger is about, and a checkpoint is served in fp16 regardless
of the precision a reference implementation uses to check it. The multiplier is
declared once, in `meta.config`, and every byte count in the trace is
`element_count × that_multiplier`.

Run: python3 labs/traces/kv_cache.py
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# Sentinel spellings for values JSON cannot represent. Same vocabulary as L00.
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
    """Unicode form, for prose."""
    f = float(v)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "-∞" if f < 0 else "+∞"
    if f == 0:
        return "0"
    return "%.4g" % f


# The significant figures every value in `state` is published to.
#
# WHY THIS EXISTS, and it is a size decision with an honesty cost that this
# comment is here to pay. `state` holds full values (never deltas — see the
# module docstring), and L05's `h` is a [N, d_model] residual stream whose
# entries come out of a standard-normal weight draw at full float64 precision.
# At N=128 that is ~150 KB of ~17-significant-figure text PER STEP, which both
# dwarfs every other lab's frames and — because random mantissas do not
# compress — costs 197 KB gzipped per trace. Inlined as a set of 18 it put
# 10.3 MB into one page, 4.4x the largest lab that had shipped.
#
# Six significant figures is two orders of margin above the four the page
# displays and far below the values' own noise floor, and it is the same
# decision L06's generator documents ("rounded to two decimals for the same two
# reasons L01's are — they stay readable at the four significant digits the page
# prints, and they keep the file small"). Nothing downstream is asserted against
# these digits: the torch cross-check, the ledger's byte accounting and the
# attention-cost comparison all run on the float64 arrays BEFORE this projection,
# so what rounds here is only what a reader can see. A trace whose published
# values had to be exact would need this number raised, and the size budget
# re-measured — not a silent edit to 17.
STATE_SIG_FIGS = 6


def state_val(v):
    """A numeric value in its JSON-safe form: a float, or a sentinel string.

    Rounded to `STATE_SIG_FIGS` — see that constant for why, and for what it
    costs. `-0.0` is normalised to `0.0`: the tensor inspector tints negatives
    and positives differently, and a signed zero that is negative only in its
    sign bit would tint a cell red for a value that is zero.
    """
    f = float(v)
    if math.isnan(f):
        return NAN
    if math.isinf(f):
        return NEG_INF if f < 0 else POS_INF
    if f == 0:
        return 0.0
    return float("%.*g" % (STATE_SIG_FIGS, f))


def state_list(values):
    return [state_val(v) for v in np.asarray(values).ravel()]


def state_mat(a):
    """A 2-D array as a list of rows -- the tensor inspector renders rank-2
    values by rows, and nesting it here keeps the JSON readable."""
    return [state_list(row) for row in np.asarray(a)]


def mx(values):
    return r"\left[" + ", ".join(fmt(v) for v in values) + r"\right]"


# ------------------------------------------------------------------- the model
#
# The shared example model (docs/plans/interactive-labs.md §二):
#   N=8, d_model=64, H=4, d_head=16, d_ff=176, vocab=32, B=1
#
# The replay adds a layer count of its own. Two layers, because one layer makes
# "the cache grows per layer" invisible, and the ledger is the thing this lab
# has to show. The layer *slider* on the page spans far past this -- the trace
# fixes what the replay runs, not what the component can compute.
CONFIG = {
    "N": 8,            # sequence length at the end of the replay
    "prompt_len": 4,   # tokens the prompt is worth
    "gen_len": 4,      # tokens generated, so prompt_len + gen_len == N
    "L": 2,
    "d_model": 64,
    "H": 4,
    "d_head": 16,
    "d_ff": 176,
    "vocab": 32,
    "B": 1,
    "eps": 1e-5,
}

# The model's geometry and the replay's SHAPE are fixed across the set — only
# the three slider parameters move. `gen_len` in particular is constant, so
# every configuration replays 5 phases × 4 sub-steps = 20 steps and the context
# slider changes how much context the prefill swallows, not how many steps the
# reader has to click through.
GEOMETRY = {k: CONFIG[k] for k in ("d_model", "H", "d_head", "d_ff", "vocab", "B", "eps")}

# Deployment dtypes: what the ledger prices. See the module docstring for why
# these are not the dtype the reference computes in.
#
# fp8 and int8 price IDENTICALLY at this level — one byte per element — and the
# set publishes that as a property rather than hiding it (see the cross-grid
# assertions in `main`). They are both in the slider because the tutorial names
# both, and the page says which of their difference this ledger can and cannot
# see: the byte width is the same; what differs is the scale / zero-point
# bookkeeping, which is a precision-format question rather than a memory-size
# one. A slider that silently dropped the third position would have been the
# dishonest option.
PRECISIONS = {
    "fp16": {"name": "fp16", "weight_bytes": 2, "kv_bytes": 2, "act_bytes": 2},
    "fp8": {"name": "fp8", "weight_bytes": 1, "kv_bytes": 1, "act_bytes": 1},
    "int8": {"name": "int8", "weight_bytes": 1, "kv_bytes": 1, "act_bytes": 1},
}

# The grid the page's sliders walk: the full product, so the three sliders are
# three independent controls rather than three positions of one knob.
#
# THE CONTEXT GRID IS CAPPED AT 128 BY PAGE SIZE, and that is a real constraint
# rather than a preference. A trace's volume is dominated by the residual-stream
# snapshot each prefill sub-step publishes (`h`, [prompt_len, d_model], at full
# float64 precision) — ~330 KB per step at N=256, against L01's whole 8x8 GEMM
# frame. A 32x span (8, 128, 256) put 10.3 MB of traces into one inlined page,
# 4.4x the largest lab that shipped before it and 8x the biggest page the site
# had (docs/research/size-budget.md §4: "一个 lab 页面在这个分布里不构成任何
# 异常" — at that size this one was the anomaly). 16x keeps a three-position
# slider and a visible ledger story (KV share 7% -> 13% -> 24%) at 4.0 MB.
# Contexts past 128 are reachable through the page's *predicted* panel, which is
# labelled as a prediction and drives the same closed form out to 4096.
GRID_N = (8, 64, 128)
GRID_L = (2, 4)
GRID_PREC = ("fp16", "fp8", "int8")
DEFAULT_TRIPLE = (8, 2, "fp16")

SET = "kv-cache"


def config_for(N, L, prec):
    """The config one grid point runs. Geometry is fixed; only the knobs move."""
    cfg = dict(GEOMETRY)
    cfg.update({"N": N, "L": L, "prompt_len": N - CONFIG["gen_len"],
                "gen_len": CONFIG["gen_len"]})
    assert cfg["prompt_len"] >= 1, (N, L, prec)
    assert cfg["d_head"] * cfg["H"] == cfg["d_model"], cfg
    return cfg


def cfg_id(N, L, prec):
    """The trace's name. The default keeps the bare set name — see the module
    docstring: ticket #45 pins `kv-cache.json`, and the page matches on
    `meta.config` rather than on this string anyway."""
    if (N, L, prec) == DEFAULT_TRIPLE:
        return SET
    return f"{SET}-N{N}-L{L}-{prec}"


def label(N, L, prec):
    return f"上下文 {N} · {L} 层 · {prec}"

PROMPT_IDS = [3, 7, 1, 12]
WEIGHT_SEED = 20240521


def init_weights(cfg):
    """Deterministic weights. Inputs to the computation, like L00's x vector --
    nothing derived is typed in."""
    rng = np.random.default_rng(WEIGHT_SEED)
    d, H, dh, dff, V = (cfg["d_model"], cfg["H"], cfg["d_head"], cfg["d_ff"], cfg["vocab"])
    scale = 1.0 / math.sqrt(d)

    def w(*shape):
        return (rng.standard_normal(shape) * scale).astype(np.float64)

    p = {"embed": w(V, d), "lm_head": w(V, d), "lnf_g": np.ones(d), "lnf_b": np.zeros(d)}
    for l in range(cfg["L"]):
        p[f"Wq{l}"] = w(d, d)
        p[f"Wk{l}"] = w(d, d)
        p[f"Wv{l}"] = w(d, d)
        p[f"Wo{l}"] = w(d, d)
        p[f"Wg{l}"] = w(dff, d)
        p[f"Wu{l}"] = w(dff, d)
        p[f"Wd{l}"] = w(d, dff)
        p[f"ln1g{l}"] = np.ones(d)
        p[f"ln1b{l}"] = np.zeros(d)
        p[f"ln2g{l}"] = np.ones(d)
        p[f"ln2b{l}"] = np.zeros(d)
    assert set(p) and H * dh == d
    return p


def analytic_param_count(cfg):
    """The closed-form parameter count, written independently of how the model
    is built. `init_weights` is asserted against this, so "the ledger's params
    figure equals the parameter count" is a real statement rather than a
    tautology."""
    d, H, dh, dff, V, L = (cfg["d_model"], cfg["H"], cfg["d_head"], cfg["d_ff"],
                           cfg["vocab"], cfg["L"])
    per_layer = 4 * d * d + 3 * d * dff + 4 * d   # Wq,Wk,Wv,Wo + Wg,Wu,Wd + ln1/ln2 (γ,β)
    return V * d + L * per_layer + 2 * d + V * d  # embed + layers + ln_f + lm_head


# ------------------------------------------------------------------- the math
def layernorm(x, g, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def softmax_rows(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


class Live:
    """The arrays a step has allocated and not yet freed.

    The byte accounting is taken from this, not from a shape formula, so
    "activations cost X" is a statement about arrays that exist. The formula is
    then a *second* opinion about the same number (`live_specs`), and the two
    are asserted equal -- either could be wrong alone.
    """

    def __init__(self):
        self.arrays = {}

    def keep(self, name, arr):
        self.arrays[name] = np.asarray(arr)
        return self.arrays[name]

    @property
    def elements(self):
        return sum(int(a.size) for a in self.arrays.values())


def live_specs(phase, cfg, n, t):
    """The shape-level view of the same working set, for the second opinion.

    `n` queries against `t` cached keys. `H*d_head == d_model`, so the per-token
    cost collapses to a clean multiple of d_model.
    """
    d, dff, H, V = cfg["d_model"], cfg["d_ff"], cfg["H"], cfg["vocab"]
    return {
        # QKV projections: the residual stream in, LN out, the new K and V.
        "kv": {"h_in": (n, d), "ln1": (n, d), "k": (n, H, d // H), "v": (n, H, d // H)},
        # attention: the score matrix and the softmax weights are both [H, n, t]
        "attn": {"h_in": (n, d), "q": (n, d), "scores": (H, n, t), "w": (H, n, t),
                 "out": (n, d), "proj": (n, d), "h1": (n, d)},
        # FFN: gate and up are [n, d_ff]; the SwiGLU product is one more [n, d_ff]
        "ffn": {"h1": (n, d), "ln2": (n, d), "gate": (n, dff), "up": (n, dff),
                "swiglu": (n, dff), "down": (n, d), "h2": (n, d)},
        # the head runs on the last position only
        "head": {"h_final": (n, d), "lnf": (n, d), "logits": (V,)},
    }[phase]


def cached_forward(prefix_ids, cfg, p, cache_k, cache_v, phase_name):
    """One forward pass over `prefix_ids`, reusing and extending the cache.

    Returns (logits, attn_row, new_cache_k, new_cache_v, live_by_phase), where
    `live_by_phase` maps "kv"/"attn"/"ffn"/"head" to the Live recorder for that
    sub-step. The sub-step boundaries are the replay's step boundaries; the
    accounting follows them rather than re-deriving shapes after the fact.
    """
    d, H, dh, dff, eps = cfg["d_model"], cfg["H"], cfg["d_head"], cfg["d_ff"], cfg["eps"]
    n = len(prefix_ids)
    t_prev = 0 if cache_k is None else cache_k[0].shape[0]
    t_new = t_prev + n
    x = p["embed"][np.asarray(prefix_ids)]           # [n, d]

    live = {k: Live() for k in ("kv", "attn", "ffn", "head")}
    live["kv"].keep("h_in", x)

    K_layers, V_layers = [], []
    for l in range(cfg["L"]):
        # ---- kv sub-step: project K and V, append to this layer's cache
        kv = Live()
        kv.keep("h_in", x)
        ln1 = kv.keep("ln1", layernorm(x, p[f"ln1g{l}"], p[f"ln1b{l}"], eps))
        k_l = kv.keep("k", ln1 @ p[f"Wk{l}"].T)      # [n, d]
        v_l = kv.keep("v", ln1 @ p[f"Wv{l}"].T)
        k_l = k_l.reshape(n, H, dh)
        v_l = v_l.reshape(n, H, dh)
        live["kv"] = kv

        prev_k = np.zeros((0, H, dh)) if cache_k is None else cache_k[l]
        prev_v = np.zeros((0, H, dh)) if cache_v is None else cache_v[l]
        K = np.concatenate([prev_k, k_l], axis=0)    # [t_new, H, dh]
        V = np.concatenate([prev_v, v_l], axis=0)
        K_layers.append(K)
        V_layers.append(V)

        # ---- attn sub-step: every query against the whole cache, causal
        at = Live()
        at.keep("h_in", x)
        q = at.keep("q", ln1 @ p[f"Wq{l}"].T).reshape(n, H, dh)
        scores = at.keep("scores", np.einsum("qhd,khd->hqk", q, K) / math.sqrt(dh))
        causal = np.arange(t_new)[None, :] > (t_prev + np.arange(n))[:, None]
        scores = np.where(causal[None, :, :], -np.inf, scores)
        w = at.keep("w", softmax_rows(scores))
        out = at.keep("out", np.einsum("hqk,khd->qhd", w, V).reshape(n, d))
        proj = at.keep("proj", out @ p[f"Wo{l}"].T)
        h1 = at.keep("h1", x + proj)                 # residual
        live["attn"] = at

        # ---- ffn sub-step: pre-norm SwiGLU, second residual
        ff = Live()
        ff.keep("h1", h1)
        ln2 = ff.keep("ln2", layernorm(h1, p[f"ln2g{l}"], p[f"ln2b{l}"], eps))
        gate = ff.keep("gate", ln2 @ p[f"Wg{l}"].T)
        up = ff.keep("up", ln2 @ p[f"Wu{l}"].T)
        swiglu = ff.keep("swiglu", silu(gate) * up)
        down = ff.keep("down", swiglu @ p[f"Wd{l}"].T)
        x = ff.keep("h2", h1 + down)
        live["ffn"] = ff

    # ---- head sub-step: final norm, lm_head, next-token logits
    hd = Live()
    hd.keep("h_final", x)
    lnf = hd.keep("lnf", layernorm(x, p["lnf_g"], p["lnf_b"], eps))
    logits_all = lnf @ p["lm_head"].T                # [n, V]
    logits = hd.keep("logits", logits_all[-1])       # only the last position matters
    live["head"] = hd

    # attention row of the final layer's last query, over the whole context
    attn_row = w[:, -1, :].reshape(H, t_new)         # [H, t_new]
    return logits, attn_row, K_layers, V_layers, live, t_new


# ------------------------------------------------------- torch reference (no cache)
def torch_forward_uncached(prompt_ids, cfg, p):
    """Recompute the whole prefix from scratch, in torch, with no cache at all.

    Deliberately structured unlike the numpy path: one [t, t] score matrix with
    an explicit -inf causal mask, `F.softmax`, and every position's logits
    produced at once. The numpy path keeps a cache and computes one query row;
    this one has never seen a cache. If the two agree at the last position, the
    cache is not changing the answer.
    """
    d, H, dh, dff, eps = cfg["d_model"], cfg["H"], cfg["d_head"], cfg["d_ff"], cfg["eps"]
    ids = torch.tensor(list(prompt_ids), dtype=torch.int64)
    tw = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in p.items()}

    t = len(prompt_ids)
    x = tw["embed"][ids]

    def ln(a, g, b):
        mu = a.mean(dim=-1, keepdim=True)
        var = a.var(dim=-1, unbiased=False, keepdim=True)
        return (a - mu) / torch.sqrt(var + eps) * g + b

    for l in range(cfg["L"]):
        ln1 = ln(x, tw[f"ln1g{l}"], tw[f"ln1b{l}"])
        q = (ln1 @ tw[f"Wq{l}"].T).reshape(t, H, dh)
        k = (ln1 @ tw[f"Wk{l}"].T).reshape(t, H, dh)
        v = (ln1 @ tw[f"Wv{l}"].T).reshape(t, H, dh)
        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(dh)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", w, v).reshape(t, d)
        h1 = x + out @ tw[f"Wo{l}"].T
        ln2 = ln(h1, tw[f"ln2g{l}"], tw[f"ln2b{l}"])
        gate = ln2 @ tw[f"Wg{l}"].T
        up = ln2 @ tw[f"Wu{l}"].T
        x = h1 + (torch.nn.functional.silu(gate) * up) @ tw[f"Wd{l}"].T

    lnf = ln(x, tw["lnf_g"], tw["lnf_b"])
    return (lnf @ tw["lm_head"].T)[-1]


# Measured maximum deviation between the numpy cache path and the torch no-cache
# path, over every published step: **2.3e-15** absolute, against a logit vector
# whose largest entry is ≈2.4 — i.e. about ten float64 ULPs, from summation
# order alone.
#
# The tolerance is 1e-12, roughly 400× the measured noise and, deliberately,
# three to four orders of magnitude *below* the four significant digits the lab
# displays. The second half of that matters more than the first: at display
# precision a loose tolerance would let a wrong formula through whenever it
# happened to round to the same number on screen, which is exactly the failure
# this check exists to catch. Every wrong-formula mistake worth worrying about
# here — a dropped 1/√d_h, a missing causal mask, an off-by-one in the cache
# append — moves the logits by O(1), so nothing is lost by being this tight.
_REL_TOL = 1e-12


def max_rel_dev(got, want):
    """How far apart two logit vectors are, relative to the vector's own scale.

    Two choices here, both deliberate:

    * **The whole vector is compared, not its argmax.** Two computations that
      pick the same next token while disagreeing about the distribution would
      pass a token-only check — and it is the distribution, not the argmax,
      that the following step consumes.
    * **The scale is the vector's largest magnitude, not each element's own.**
      A per-element relative error divides by numbers that are legitimately
      near zero (a logit sitting at 3e-16 against a 2.4 peak), so it reports
      meaningless ratios in the hundreds of percent. Normalising by ‖logits‖∞
      measures the thing that matters: how much of the distribution moved.
    """
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    if got.shape != want.shape:
        raise AssertionError(f"shape mismatch: numpy {got.shape} vs torch {want.shape}")
    return float(np.max(np.abs(got - want)) / max(float(np.max(np.abs(want))), 1e-300))


# ------------------------------------------------------------------- the ledger
#
# The byte accounting. Two implementations of every number, asserted equal:
#
#   real   -- `arr.size` over the arrays that actually exist in the forward pass
#   closed -- the formulas the two guides state (1.1 §2.3, 3.8 §6.4)
#
# Storage is priced at the deployment dtype (fp16, 2 bytes), not at the float64
# the reference computes in. See the module docstring.
def ledger_counts(p, live, caches_k, caches_v):
    """Per-segment element counts for one step, read off the real arrays.

    `p` is every weight tensor in the model, `live` the arrays this sub-step has
    allocated, `caches_*` the per-layer K/V caches as they stand at the end of
    the step. Nothing here is derived from a shape formula — that is the whole
    point, since the shape formula is the thing being checked.
    """
    return {
        "params": sum(int(w.size) for w in p.values()),
        "kv_cache": sum(int(c.size) for c in list(caches_k) + list(caches_v)),
        "activations": live.elements,
    }


def closed_form_counts(cfg, phase, n, t, kv_rows_total):
    """The same three numbers from the closed forms, independently.

    `params` is the parameter-count formula; `kv_cache` is the guide's
    `2 · L · H_kv · d_h · N · B`; `activations` is the enumerated working set of
    one sub-step, which is the only honest way to price activations -- there is
    no textbook closed form for them, so the enumeration *is* the definition and
    the real arrays are what keep it honest.
    """
    d, dff, H, V, B, dh, L = (cfg["d_model"], cfg["d_ff"], cfg["H"], cfg["vocab"],
                              cfg["B"], cfg["d_head"], cfg["L"])
    act = sum(int(np.prod(shape)) for shape in live_specs(phase, cfg, n, t).values())
    return {
        "params": analytic_param_count(cfg),
        "kv_cache": 2 * H * dh * kv_rows_total * B,      # 2 = K and V
        "activations": act,
    }


def to_bytes(counts, deploy):
    """Element counts -> bytes at the deployment dtype."""
    return {
        "params": counts["params"] * deploy["weight_bytes"],
        "kv_cache": counts["kv_cache"] * deploy["kv_bytes"],
        "activations": counts["activations"] * deploy["act_bytes"],
    }


# --------------------------------------- the two-mode attention cost (shapes)
def attn_cost(cfg, n_q, t):
    """Attention's shapes and cost in both modes, as a formula.

    `n_q` queries against `t` keys. Without a cache the queries are every
    position that exists (`t` of them) and the keys are those same `t`; with one
    they are only this step's (`n_q` — `prompt_len` at prefill, 1 at decode).
    The per-head inner product is the same operation either way, which is what
    makes the two counts comparable at all.

    This is the SHAPE side. `step.attn_cost` publishes these fields beside
    counts read off the arrays both paths really allocated, and asserts the two
    agree — see `attn_cost_step`.
    """
    d, H, dh, L = cfg["d_model"], cfg["H"], cfg["d_head"], cfg["L"]

    def mode(q, cached):
        return {
            "queries": q,
            "q_shape": [q, d],
            "k_shape": [t, d],
            # One row per query, one column per key, one plane per head: this is
            # the matrix the tutorial draws as `[t,d] × [t,d]` vs `[1,d] × [t,d]`.
            "score_shape": [H, q, t],
            "score_elems": H * q * t,
            # The uncached mode has no cache, so it has no cache traffic — its
            # cost lands in `proj_flops` instead, and saying "0" here is what
            # keeps that visible rather than spread across both columns.
            "kv_read_elems": 2 * L * H * dh * t if cached else 0,
            "kv_write_elems": 2 * L * H * dh * q if cached else 0,
            # QK^T and W·V, two matmuls of 2·H·q·t·d_h FLOPs each.
            "attn_flops": 4 * H * q * t * dh,
            # The projections this mode must redo: Q, K and V for all q queries
            # in every layer, i.e. 3·2·q·d²·L. It is THIS term — not the
            # attention itself — that makes the uncached path quadratic in the
            # context, and the page says so rather than letting the score matrix
            # take all the blame.
            "proj_flops": 6 * q * d * d * L,
        }

    return {"t": t, "n_q": n_q, "H": H, "d_head": dh, "layers": L,
            "no_cache": mode(t, False), "with_cache": mode(n_q, True)}


def attn_cost_step(cfg, n_q, t, cache_live, nocache_live):
    """`attn_cost`, with its formula cross-checked against both paths' arrays.

    `cache_live` / `nocache_live` are the `Live` recorders from the cached
    forward and from the full-prefix uncached forward — two computations that
    really ran, in the same step. A mismatch in either direction raises, so the
    picture on the page cannot describe a computation that did not happen.
    """
    cost = attn_cost(cfg, n_q, t)
    for name, live, want_q, want_k in (
            ("no_cache", nocache_live, t, t),
            ("with_cache", cache_live, n_q, n_q)):
        got = int(np.asarray(live["attn"].arrays["scores"]).size)
        if got != cost[name]["score_elems"]:
            raise AssertionError(
                f"attn_cost[{name}]: 实算分数矩阵 {got} 个元素，"
                f"公式 {cost[name]['score_elems']}（{cost[name]['score_shape']}）")
        q_rows = int(np.asarray(live["attn"].arrays["q"]).shape[0])
        k_rows = int(np.asarray(live["kv"].arrays["k"]).shape[0])
        if (q_rows, k_rows) != (want_q, want_k):
            raise AssertionError(
                f"attn_cost[{name}]: 实算 Q/K 行数 {q_rows}/{k_rows}，"
                f"公式 {want_q}/{want_k}")
    return cost


# ------------------------------------------------------- the phase Roofline
#
# The hardware model, quoted from the tutorial's own worked example
# (`1.1-LLM推理基础.md` §4.2: H100 SXM, BF16 dense 990 TFLOP/s, HBM3 3.35 TB/s
# → ridge ≈ 295 FLOP/Byte). Constants beside the generator, like L01's V100
# pair, so the ridge is a number this script computes rather than transcribes.
PEAK_FLOPS = 990e12
BW_BYTES = 3.35e12

# The reference point the sliders must NOT move: a 7B-class model (Llama-7B's
# shape) reading a 512-token prompt — the tutorial's "成百上千 Token" prefill.
# It answers the question the toy model cannot ("does anything reach the
# compute roof?") without pretending the toy is that model.
DOC_MODEL = {"L": 32, "d_model": 4096, "H": 32, "d_head": 128,
             "d_ff": 11008, "vocab": 32000, "B": 1}
DOC_PROMPT = 512


def phase_cost(cfg, deploy, n_q, t):
    """FLOPs and HBM bytes for one forward pass: `n_q` queries over `t` keys.

    THE TRAFFIC MODEL, stated because it is a choice and the page repeats it:
    the weights are read once per pass, the KV cache is read in full, and this
    step's new rows are written back. The attention score matrix is NOT counted
    as HBM traffic — it is produced and consumed inside the attention kernel's
    SRAM, which is the assumption FlashAttention is built on (L06) and the
    reason this lab can put attention on a Roofline at all. Counting it would
    price a tensor no implementation here materialises in HBM.

    The `parts` come back beside the totals so the lint can re-add them: a total
    that disagrees with its own sum is exactly what one summary number hides.
    """
    d, H, dh, dff, V, L = (cfg["d_model"], cfg["H"], cfg["d_head"], cfg["d_ff"],
                           cfg["vocab"], cfg["L"])

    proj_flops = L * 8 * n_q * d * d              # Wq,Wk,Wv,Wo: d×d each
    ffn_flops = L * 6 * n_q * d * dff             # Wg,Wu: d×d_ff; Wd: d_ff×d
    attn_flops = L * 4 * H * n_q * t * dh         # QK^T and W·V
    head_flops = 2 * n_q * d * V                  # lm_head
    weights_bytes = analytic_param_count(cfg) * deploy["weight_bytes"]
    kv_read_bytes = 2 * L * H * dh * t * deploy["kv_bytes"]
    kv_write_bytes = 2 * L * H * dh * n_q * deploy["kv_bytes"]

    parts = {"proj_flops": proj_flops, "ffn_flops": ffn_flops,
             "attn_flops": attn_flops, "head_flops": head_flops,
             "weights_bytes": weights_bytes, "kv_read_bytes": kv_read_bytes,
             "kv_write_bytes": kv_write_bytes}
    flops = proj_flops + ffn_flops + attn_flops + head_flops
    n_bytes = weights_bytes + kv_read_bytes + kv_write_bytes
    return {"flops": flops, "bytes": n_bytes, "ai": flops / n_bytes, "parts": parts}


def roofline_phases(cfg, deploy):
    """The Roofline placement of this configuration's two phases.

    Three points, and the comparison between them is the lesson:

      * `prefill` — this configuration's prompt, processed in one pass. Its
        arithmetic intensity is `n_q` times the decode one, because the weight
        bytes are paid ONCE for the whole prompt: that is what the prompt buys.
        It is the point the slider moves.
      * `decode` — one token against the same weights. Its intensity sits at
        ≈1 FLOP/Byte and barely moves with the context, which is the control
        that makes the prefill movement readable as a comparison.
      * `doc` — a 7B-class model at a 512-token prompt, i.e. the tutorial's own
        claim computed rather than quoted. It is deliberately NOT tied to this
        trace's config: it must be the same dot on every configuration, and the
        harness asserts that.

    Note what the honest answer is here, because the page has to say it: in this
    toy model BOTH phase points land on the bandwidth roof — the compute roof is
    reached only by the reference model. The difference the lab can show is the
    one that is real at every scale: prefill's intensity is `prompt_len`-fold
    decode's, and it grows with the prompt while decode's does not.
    """
    peak, bw = PEAK_FLOPS, BW_BYTES
    ridge = peak / bw
    phase_costs = {
        "prefill": (cfg, deploy, cfg["prompt_len"], cfg["prompt_len"], cfg["prompt_len"]),
        "decode": (cfg, deploy, 1, cfg["N"], cfg["N"]),
        # fp16 regardless of this configuration's slider position: the reference
        # describes the reference model, and letting it follow the precision
        # slider would make the control move.
        "doc": (DOC_MODEL, PRECISIONS["fp16"], DOC_PROMPT, DOC_PROMPT, DOC_PROMPT),
    }
    # Short labels, on purpose: the chart draws them beside a dot and the axis
    # domain is derived from the points, so a long one runs off the frame at the
    # configurations where the live points sit far right. What each point IS
    # belongs in the breakdown table below the chart, which has room for it.
    labels = {"prefill": "Prefill", "decode": "Decode", "doc": "7B 参考"}

    points = []
    for pid, (pcfg, pdep, n_q, t, prompt) in phase_costs.items():
        cost = phase_cost(pcfg, pdep, n_q, t)
        ai = cost["ai"]
        points.append({
            "id": pid, "label": labels[pid], "ai": ai,
            "flops": cost["flops"], "bytes": cost["bytes"],
            "parts": cost["parts"],
            "ceiling": min(peak, bw * ai),
            "bound": "bandwidth" if ai < ridge else "compute",
            "config": {"n_q": n_q, "t": t, "prompt": prompt, "layers": pcfg["L"],
                       "dtype": pdep["name"]},
        })

    return {
        "peak_flops": peak, "bandwidth": bw, "ridge": ridge,
        "x_label": "算术强度 (FLOP/Byte)",
        "y_label": "可达算力上界 (TFLOP/s)",
        "y_divisor": 1e12, "y_unit": "TFLOP/s",
        "model": "NVIDIA H100 SXM：BF16 稠密 990 TFLOP/s · HBM3 3.35 TB/s "
                 "（教程 1.1 §4.2 的算例）",
        "traffic_note": "按权重读取一次 + KV 全量读 + 本步新行写回来计价；"
                        "注意力分数矩阵不进出 HBM（它在 kernel 的 SRAM 里产生并消费，"
                        "这正是 FlashAttention 的前提），所以不计入访存。",
        "note": "点画在 min(峰值算力, 带宽 × 算术强度) 上，是 Roofline 给出的上界，"
                "不是实测值 —— 本实验室没有对任何 kernel 计时。",
        "points": points,
    }


def kv_crossover(cfg, deploy):
    """The context length at which the cache outweighs the weights.

    Published so the page can state it without computing it. At this model's
    size the crossover is far beyond the longest context the sliders offer
    (≈410 for 2 layers at fp16): the ledger bar never crosses over on screen,
    and a page that left the number out would leave the reader with a picture
    that never shows the thing the tutorial is about.
    """
    params_bytes = analytic_param_count(cfg) * deploy["weight_bytes"]
    per_token = 2 * cfg["L"] * cfg["H"] * cfg["d_head"] * deploy["kv_bytes"]
    return {"params_bytes": params_bytes, "per_token_bytes": per_token,
            "context": params_bytes / per_token}


# The segment declarations the component consumes. `formula` is the provenance
# the acceptance criteria ask for: every figure on screen traces to one of these.
LEDGER_SEGMENTS = [
    {
        "id": "params",
        "label": "参数",
        "color": "#5b8def",
        "formula": r"\sum_i \mathrm{numel}(W_i)\times b_w",
    },
    {
        "id": "kv_cache",
        "label": "KV Cache",
        "color": "#d6408c",
        "formula": r"2\times L\times H_{kv}\times d_h\times N\times B\times b_{kv}",
    },
    {
        "id": "activations",
        "label": "激活",
        "color": "#e08b2c",
        "formula": r"\sum_j \mathrm{numel}(a_j)\times b_a\ \ (\text{本步工作集})",
    },
]


# ---------------------------------------------------------------- trace build
def b(slot, sym, idx, num):
    return slot, {"sym": sym, "idx": idx, "num": num}


def step(sid, title, kind, op, phase, formula, bindings, reads, writes, state,
         narration, regions=None, ledger=None, attn_cost=None):
    d = {"id": sid, "title": title, "kind": kind, "op": op, "phase": phase,
         "formula": formula, "bindings": dict(bindings),
         "reads": reads, "writes": writes, "state": state, "narration": narration}
    if regions:
        d["regions"] = regions
    if ledger is not None:
        d["ledger"] = ledger
    if attn_cost is not None:
        d["attn_cost"] = attn_cost
    return d


def shape_str(*dims):
    return "[" + ",".join(str(x) for x in dims) + "]"


def build_trace(cfg, p, deploy, prompt_ids):
    d, H, dh, dff, V, L = (cfg["d_model"], cfg["H"], cfg["d_head"], cfg["d_ff"],
                           cfg["vocab"], cfg["L"])
    P = cfg["prompt_len"]
    G = cfg["gen_len"]

    tokens = list(prompt_ids)
    assert len(tokens) == P

    cache_k, cache_v = None, None
    steps = []
    checks = []
    no_cache_phases = {}

    phases = [("prefill", "Prefill", P, P)]
    for g in range(1, G + 1):
        phases.append((f"d{g}", f"Decode {g}", 1, P + g))
    # phase tuple: (id, label, queries-this-step, context length after this step)

    for pid, plabel, n_q, t_after in phases:
        ids_this = tokens[len(tokens) - n_q:]
        before = tokens[:len(tokens) - n_q]

        logits, attn_row, cache_k, cache_v, live, t_new = cached_forward(
            ids_this, cfg, p, cache_k, cache_v, pid)
        assert t_new == t_after, (t_new, t_after)

        # --- check 1 & 2: the cache path agrees with both uncached references.
        #
        # The no-cache forward's `live` recorders are kept, not discarded: at
        # this phase it recomputed the whole prefix, and its attention working
        # set is the OTHER COLUMN of the comparison this lab is about. Reading
        # the shapes off that run is what makes the contrast a measurement of
        # two computations rather than a claim about one.
        ctx = before + ids_this
        torch_out = torch_forward_uncached(ctx, cfg, p)
        checks.append((f"{pid}.vs-torch", logits, np.asarray(torch_out.numpy())))
        no_cache_logits, _, _, _, no_cache_live, _ = cached_forward(ctx, cfg, p, None, None, pid)
        checks.append((f"{pid}.vs-own-no-cache", logits, no_cache_logits))
        no_cache_phases[pid] = {"live": no_cache_live, "n_q": len(ctx), "t": len(ctx)}

        sampled = int(np.argmax(logits))
        tokens = ctx + [sampled]
        kv_rows_total = sum(k.shape[0] for k in cache_k)

        # --- the ledger, for each of the four sub-steps of this phase
        def make_ledger(sub):
            counts_real = ledger_counts(p, live[sub], cache_k, cache_v)
            counts_closed = closed_form_counts(cfg, sub, n_q, t_new, kv_rows_total)
            for k in counts_real:
                if counts_real[k] != counts_closed[k]:
                    raise AssertionError(
                        f"{pid}.{sub}: 账本分项 {k} 实算={counts_real[k]} "
                        f"闭式={counts_closed[k]}")
            return {"bytes": to_bytes(counts_real, deploy),
                    "config": {"phase": sub, "n_q": n_q, "layers": cfg["L"],
                               "seq_len": t_new, "kv_rows_total": kv_rows_total}}

        led_kv = make_ledger("kv")
        led_attn = make_ledger("attn")
        led_ffn = make_ledger("ffn")
        led_head = make_ledger("head")

        # --- the with/without-cache shapes, both read off real arrays
        cost = attn_cost_step(cfg, n_q, t_new, live, no_cache_phases[pid]["live"])

        # ------------------------------------------------------------- kv
        new_k_row = cache_k[-1][-1]
        new_v_row = cache_v[-1][-1]
        steps.append(step(
            f"{pid}.kv", f"{plabel}：本步 K/V 追加进缓存", "comm", "append", plabel,
            {
                "sym": r"K^{(l)} \leftarrow \left[\,\slot{KOLD}\,;\,"
                       r"\region{APPEND}{\mathrm{LN}(h)\,W_k^{(l)}}\,\right],\qquad "
                       r"V^{(l)} \leftarrow \left[\,\slot{VOLD}\,;\,\mathrm{LN}(h)\,W_v^{(l)}\,\right]",
                "idx": r"K^{(\slot{L})}_{[\slot{TOLD},H,d_h]} \leftarrow "
                       r"\left[\,\slot{KOLD}\,;\,\slot{KNEW}_{[\slot{NQ},H,d_h]}\,\right]"
                       r"\ \to\ K^{(\slot{L})}_{[\slot{TNEW},H,d_h]}",
                "num": r"\slot{KNEW} = \mathrm{shape}\ \slot{KSHAPE}"
                       r",\quad \slot{VNEW} = \mathrm{shape}\ \slot{VSHAPE}"
                       r"\quad(\slot{NQ}\ \text{个新 token}, 每层 \slot{TOLD} \to \slot{TNEW})",
            },
            [b("L", "l", "0/1", "0/1"),
             b("TOLD", r"t_{\mathrm{prev}}", str(t_after - n_q), str(t_after - n_q)),
             b("TNEW", r"t", str(t_after), str(t_after)),
             b("NQ", "n_q", str(n_q), str(n_q)),
             b("KOLD", r"K^{(l)}_{[t_{\mathrm{prev}},H,d_h]}",
               f"K^{{(l)}}_{{[{t_after - n_q},H,d_h]}}",
               f"{t_after - n_q}\\times{H}\\times{dh}"),
             b("VOLD", r"V^{(l)}_{[t_{\mathrm{prev}},H,d_h]}",
               f"V^{{(l)}}_{{[{t_after - n_q},H,d_h]}}",
               f"{t_after - n_q}\\times{H}\\times{dh}"),
             b("KNEW", r"\mathrm{LN}(h)W_k^{(l)}",
               r"\mathrm{LN}(h^{(n_q)})W_k^{(l)}", shape_str(n_q, H, dh)),
             b("VNEW", r"\mathrm{LN}(h)W_v^{(l)}",
               r"\mathrm{LN}(h^{(n_q)})W_v^{(l)}", shape_str(n_q, H, dh)),
             b("KSHAPE", r"\cdot", r"\cdot", shape_str(n_q, H, dh)),
             b("VSHAPE", r"\cdot", r"\cdot", shape_str(n_q, H, dh))],
            ["tokens"], ["k_new", "v_new"],
            {"k_new": state_mat(new_k_row), "v_new": state_mat(new_v_row)},
            f"这一步只做一件事：把本步 {n_q} 个 token 的 K、V 算出来，追加到 {L} 层各自的缓存上"
            f"（每层从 {t_after - n_q} 行长到 {t_after} 行）。历史部分一个字节都不动 —— "
            f"这就是 KV Cache 的全部内容，也是账本里 KV Cache 分项唯一的增长来源。",
            {"APPEND": f"每层新增 {n_q} 行 × H={H} 头 × d_h={dh}"},
            led_kv,
        ))

        # ----------------------------------------------------------- attn
        steps.append(step(
            f"{pid}.attn", f"{plabel}：查询与全量 K/V 做注意力", "op", "attention", plabel,
            {
                "sym": r"S = \frac{Q K^{\top}}{\sqrt{d_h}},\qquad "
                       r"W = \mathrm{softmax}\left(\region{MASK}{S + M}\right),\qquad "
                       r"O = W V W_o",
                "idx": r"Q_{[\slot{NQ},H,d_h]} \times K^{\top}_{[\slot{T},H,d_h]} "
                       r"\to S + M_{[\slot{H},\slot{NQ},\slot{T}]} "
                       r"\to W_{[\slot{H},\slot{NQ},\slot{T}]}",
                "num": r"S_{\max} = \slot{SMAX} \to "
                       r"W_{[\slot{H},\slot{NQ},\slot{T}]} \to O_{[\slot{NQ},\slot{d}]}",
            },
            [b("NQ", "n_q", str(n_q), str(n_q)),
             b("T", "t", str(t_new), str(t_new)),
             b("H", "H", str(H), str(H)),
             b("d", "d_{model}", str(d), str(d)),
             b("SMAX", r"\max S", r"\max S", fmt(float(np.max(live["attn"].arrays["scores"]))))],
            ["tokens", "k_new"], ["h", "attn"],
            {"h": state_mat(live["attn"].arrays["h1"]),
             "attn": state_mat(attn_row)},
            (f"有 cache 之后，本步只有 {n_q} 个 Query，却要对 {t_new} 个 Key 做内积，"
             f"所以分数矩阵是 [{H}, {n_q}, {t_new}] —— 行是查询、列是上下文。"
             + (f"Prefill 时 {n_q} 个查询一次算完，矩阵最宽也最贵。"
                if n_q > 1 else
                f"Decode 时只有 1 个查询，矩阵退化成一行；上下文越长这一行越长，"
                f"这正是 decode 每步都要读全量 KV 的原因。")
             + (f" 同一步不做缓存的话要算 {H}×{t_new}×{t_new} 的分数矩阵 —— "
                f"本步是 Prefill，缓存是空的，两条路重合，"
                f"所以这一刻还看不出 1/{(n_q if n_q > 1 else 1)} 的差别。"
                if n_q == cost["no_cache"]["queries"] else
                f" 同一步不做缓存的话，要把 {t_new} 个 token 重新投影一遍、"
                f"算 {H}×{t_new}×{t_new} 的分数矩阵 —— 那就是 {n_q} 个查询变 "
                f"{t_new} 个查询，本页的 shape 对比图把这两栏并排画出来。")),
            {"MASK": f"因果掩码 M：被遮住的位置 -∞，softmax 后权重为 0。"
                     f"本步 {n_q} 个查询，每个只看得到自己及之前的 {t_new} 个位置"},
            led_attn,
            cost,
        ))

        # ------------------------------------------------------------ ffn
        new_h2 = live["ffn"].arrays["h2"]
        steps.append(step(
            f"{pid}.ffn", f"{plabel}：前馈网络 + 第二个残差", "op", "ffn", plabel,
            {
                "sym": r"h_1 = h + O,\qquad h_2 = h_1 + \mathrm{SwiGLU}(\mathrm{LN}(h_1))",
                "idx": r"\mathrm{gate}_{[\slot{NQ},\slot{DFF}]},\ "
                       r"\mathrm{up}_{[\slot{NQ},\slot{DFF}]} \to "
                       r"\mathrm{down}_{[\slot{NQ},\slot{d}]}",
                "num": r"h_2\ \text{shape}\ \slot{SHAPE}",
            },
            [b("NQ", "n_q", str(n_q), str(n_q)),
             b("DFF", "d_{ff}", str(dff), str(dff)),
             b("d", "d_{model}", str(d), str(d)),
             b("SHAPE", r"\cdot", r"\cdot", shape_str(n_q, d))],
            ["h"], ["h"],
            {"h": state_mat(new_h2)},
            f"SwiGLU 的中间宽度是 d_ff={dff}，比 d_model={d} 宽 —— "
            f"激活分项里 FFN 那几块（gate、up、SwiGLU 乘积）就是这里来的。"
            f"它们和 K/V 缓存不同，用完即弃，所以只进『本步工作集』，不进『常驻』。",
            None,
            led_ffn,
        ))

        # ----------------------------------------------------------- head
        steps.append(step(
            f"{pid}.head", f"{plabel}：取最后一个位置，采样下一个 token",
            "op", "head", plabel,
            {
                "sym": r"\mathrm{logits} = \mathrm{LN}_f(h)\,W_{lm}^{\top},\qquad "
                       r"x_{t+1} = \arg\max \mathrm{logits}",
                "idx": r"\mathrm{logits}_{[\slot{V}]} = "
                       r"h_{[1,\slot{d}]}\,W_{lm}^{\top},\quad x_{t+1} = \slot{TOK}",
                "num": r"\arg\max \mathrm{logits} = \slot{TOK}\ "
                       r"(\text{共 } \slot{T} \text{ 个 token})",
            },
            [b("V", "vocab", str(V), str(V)),
             b("d", "d_{model}", str(d), str(d)),
             b("TOK", r"x_{t+1}", "x_{t+1}", str(sampled)),
             b("T", "t", str(len(tokens)), str(len(tokens)))],
            ["h"], ["logits", "token_new", "tokens"],
            {"logits": state_list(logits), "token_new": sampled,
             "tokens": list(tokens)},
            f"只取最后一个位置的 hidden state 过 LM Head，得到 {V} 维 logits，"
            f"取最大得到 token {sampled}。序列现在是 {len(tokens)} 个 token，"
            f"而每层的缓存只有 {t_new} 行（{L} 层合计 {kv_rows_total} 行）—— "
            f"缓存比序列少一行，因为刚采样出来的这个 token 还没被喂回模型，"
            f"它的 K/V 要到下一步才算。"
            + (" 达到 max_new_tokens，生成结束。" if len(tokens) == cfg["N"] else ""),
            None,
            led_head,
        ))

    # --------------------------------------------------------------- graph
    nodes = [{"id": s["id"], "title": s["title"], "kind": s["kind"],
              "op": s["op"], "phase": s["phase"]} for s in steps]

    edges = []
    for pid, _, _, _ in phases:
        edges += [{"from": f"{pid}.kv", "to": f"{pid}.attn", "tensor": "k_new"},
                  {"from": f"{pid}.attn", "to": f"{pid}.ffn", "tensor": "h"},
                  {"from": f"{pid}.ffn", "to": f"{pid}.head", "tensor": "h"}]
    for i in range(len(phases) - 1):
        a, bnext = phases[i][0], phases[i + 1][0]
        edges += [{"from": f"{a}.head", "to": f"{bnext}.kv", "tensor": "tokens"}]
    # The cache is the only other thing carried across phases; drawn separately
    # from `tokens` because it is the whole subject of the lab.
    for i in range(len(phases) - 1):
        a, bnext = phases[i][0], phases[i + 1][0]
        edges.append({"from": f"{a}.kv", "to": f"{bnext}.kv", "tensor": "kv_cache"})

    max_kv_rows_total = max(s["ledger"]["config"]["kv_rows_total"] for s in steps)
    prefill_cost, decode_cost = steps[1]["attn_cost"], steps[5]["attn_cost"]
    # The comparison's totals: the two modes' score-matrix elements summed over
    # the whole replay. Both a per-step measurement and its sum are published so
    # the page can print either without adding a column up itself — and so the
    # lint can require the sums to equal the steps they came from.
    # Only the attention sub-step carries the comparison — it is the only step
    # where the two modes differ — so the totals are over those steps.
    cost_steps = [s for s in steps if "attn_cost" in s]
    attn_totals = {}
    for mode in ("no_cache", "with_cache"):
        attn_totals[mode] = {
            key: sum(s["attn_cost"][mode][key] for s in cost_steps)
            for key in ("score_elems", "attn_flops", "proj_flops",
                        "kv_read_elems", "kv_write_elems")
        }

    phase_roof = roofline_phases(cfg, deploy)
    cross = kv_crossover(cfg, deploy)
    cfg_out = dict(cfg)
    cfg_out.update({
        "dtype": deploy["name"],
        "weight_bytes": deploy["weight_bytes"],
        "kv_bytes": deploy["kv_bytes"],
        "act_bytes": deploy["act_bytes"],
    })

    return {
        "meta": {
            "lab": "L05",
            "title": f"KV Cache 与自回归生成（Prompt {cfg['prompt_len']} → 生成 "
                     f"{cfg['gen_len']}，{cfg['L']} 层 {deploy['name']}，{len(steps)} 步）",
            "source": "docs/guides/模块一-前置知识/transformer/3.8 从Transformer到LLM自回归生成深入理解.md"
                      " · docs/guides/模块四-推理优化/第1章-LLM推理基础/1.1-LLM推理基础.md",
            "config": cfg_out,
            "reference": "逐步对拍 torch 无缓存全前缀重算（float64）",
            "notes": [
                f"参考实现在 float64 下跑（对拍用），账本按部署精度 {deploy['name']} 计价"
                f"（{deploy['kv_bytes']} 字节/元素）。",
                f"KV Cache 末态每层 {cfg['N']} 行、{cfg['L']} 层合计 {max_kv_rows_total} 行 —— "
                f"序列有 {cfg['N']} 个 token 时缓存只有 {cfg['N']} 行/层，"
                f"因为被采样的最后一个 token 没有再被喂回去。",
                "变长张量（tokens / h / attn）按终态声明 shape，"
                "网格里铺开的是当前真实存在的单元格，只有标签是容量。",
            ],
            "attn_cost": {
                "totals": attn_totals,
                "note": "无缓存一栏是逐步重算整段前缀那个参照实现的真实工作集"
                        "（它就是上面被对拍的那个实现）；有缓存一栏是本回放自己的。"
                        "两栏都在同一步里真的算过。",
                "prefill_identical": (
                    prefill_cost["no_cache"]["score_elems"] ==
                    prefill_cost["with_cache"]["score_elems"]),
            },
            "phase_roofline": phase_roof,
            "kv_crossover": cross,
        },
        "ledger": {
            "unit": "B",
            "segments": [dict(s) for s in LEDGER_SEGMENTS],
        },
        "tensors": {
            "tokens": {"shape": [cfg["N"]], "dtype": "int32", "at": "HBM",
                       "role": "input", "init": list(prompt_ids),
                       "note": f"已处理的 token 序列，终态 {cfg['N']} 个；回放中逐步增长"},
            "h": {"shape": [cfg["N"], cfg["d_model"]], "dtype": "fp32", "at": "SRAM",
                  "role": "state",
                  "note": "残差流。回放中只有当前这一步的 token 在其中，标签是终态容量"},
            "k_new": {"shape": [cfg["H"], cfg["d_head"]], "dtype": "fp32", "at": "SRAM",
                      "role": "staging", "note": "本步算出的 K（最后一层的最后一行）"},
            "v_new": {"shape": [cfg["H"], cfg["d_head"]], "dtype": "fp32", "at": "SRAM",
                      "role": "staging", "note": "本步算出的 V"},
            "attn": {"shape": [cfg["H"], cfg["N"]], "dtype": "fp32", "at": "SRAM",
                     "role": "staging",
                     "note": "最后一层最后一个查询对全部上下文的注意力权重；标签是终态容量"},
            "logits": {"shape": [cfg["vocab"]], "dtype": "fp32", "at": "SRAM",
                       "role": "output", "note": "最后一个位置的词表分布"},
            "token_new": {"shape": [], "dtype": "int32", "at": "register",
                          "role": "output", "note": "本步采样出的 token id"},
        },
        "graph": {"nodes": nodes, "edges": edges},
        "steps": steps,
        "_checks": checks,   # popped by main(); not part of the contract
    }


# -------------------------------------------------------------------- self-lint
SLOT_RE = re.compile(r"\\slot\{([A-Za-z0-9_]+)\}")
REGION_RE = re.compile(r"\\region\{([A-Za-z0-9_]+)\}")
LATEX_IN_TEXT_RE = re.compile(r"\\[a-zA-Z]+\{")
BAD_LITERAL_RE = re.compile(r"^(NaN|Infinity|-Infinity|undefined|null)$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$")


def lint(trace):
    """The author-side half of the contract check.

    Same rules as `labs/assets/engine/trace-model.js`, which carries the JS port
    so a lab author editing a trace in the browser gets the same verdict. The
    ledger rules added by ticket #45 live in *both* places too -- the JS half in
    `labs/assets/engine/views/ledger.js`, because the ledger is that file's
    subject and `trace-model.js` was frozen once the engine landed.
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

    gaps += lint_ledger(trace)
    gaps += lint_attn_cost(trace)
    gaps += lint_phase_roofline(trace)
    gaps += lint_crossover(trace)

    infos.append(f'共 {len(trace["steps"])} 步 / {len(trace["graph"]["nodes"])} 节点 / '
                 f'{len(trace["graph"]["edges"])} 数据边')
    infos.append(f'state 写过的张量：{", ".join(sorted(written))}')
    # Informational only, so it must survive a trace whose ledger block is
    # malformed -- a lint that crashes on the very input it is meant to report
    # on is worse than one that misses the rule.
    block = trace.get("ledger") or {}
    if isinstance(block.get("segments"), list) and steps_with_ledger(trace):
        seg = [str(s.get("id")) for s in block["segments"]]
        last = trace["steps"][-1].get("ledger") or {}
        total = sum(last.get("bytes", {}).values())
        infos.append(f'账本分项 {" / ".join(seg)}；末步合计 '
                     f'{total:,} {block.get("unit", "?")}')
    return gaps, warns, infos


def steps_with_ledger(trace):
    return [s for s in trace["steps"] if isinstance(s, dict) and "ledger" in s]


def lint_ledger(trace):
    """The ledger's own contract, added by ticket #45.

    The block is optional: a lab with no memory accounting simply omits it, and
    every rule below stays inert. That is what keeps L00 (which predates the
    field) passing the same lint.
    """
    gaps = []
    block = trace.get("ledger")
    steps_with = steps_with_ledger(trace)

    if block is None:
        if steps_with:
            gaps.append(f'trace 没有 ledger 块，但 {len(steps_with)} 个步骤带了 step.ledger —— '
                        f"分项没有声明，组件只能显示一堆没有名字的数字")
        return gaps

    unit = block.get("unit")
    if not isinstance(unit, str) or not unit:
        gaps.append('ledger.unit 缺失：分项数值的单位没有声明，组件无法标注')
    segments = block.get("segments")
    if not isinstance(segments, list) or not segments:
        gaps.append('ledger.segments 缺失或为空：至少要有一个分项')
        return gaps

    ids = []
    for seg in segments:
        sid = seg.get("id")
        if not isinstance(sid, str) or not sid:
            gaps.append(f"ledger.segments 里有分项缺少 id：{seg!r}")
            continue
        ids.append(sid)
        if not isinstance(seg.get("label"), str) or not seg["label"]:
            gaps.append(f'ledger 分项 "{sid}" 缺 label —— 图例上会是一片空白')
        if not COLOR_RE.match(str(seg.get("color", ""))):
            gaps.append(f'ledger 分项 "{sid}" 的 color={seg.get("color")!r} 不是 '
                        f'#rgb / #rrggbb —— CSS 会静默丢弃它，色块变成透明')
        if not isinstance(seg.get("formula"), str) or not seg["formula"]:
            gaps.append(f'ledger 分项 "{sid}" 缺 formula —— '
                        f"数值就无法追溯到显存公式，这正是本组件存在的理由")
    if len(set(ids)) != len(ids):
        dups = sorted({i for i in ids if ids.count(i) > 1})
        gaps.append(f"ledger.segments 有重复 id：{', '.join(dups)}")

    declared = set(ids)
    for s in trace["steps"]:
        led = s.get("ledger")
        if led is None:
            gaps.append(f'步骤 "{s["id"]}" 没有 ledger —— '
                        f"「每个分项都要有值」是账本能画成堆叠条的前提")
            continue
        if not isinstance(led, dict):
            gaps.append(f'步骤 "{s["id"]}" 的 ledger 不是对象（{type(led).__name__}）')
            continue
        bytes_map = led.get("bytes")
        if not isinstance(bytes_map, dict):
            gaps.append(f'步骤 "{s["id"]}" 的 ledger 缺 bytes（或不是对象）')
            continue
        got = set(bytes_map)
        for missing in sorted(declared - got):
            gaps.append(f'步骤 "{s["id"]}" 的 ledger 少了分项 "{missing}"')
        for extra in sorted(got - declared):
            gaps.append(f'步骤 "{s["id"]}" 的 ledger 有未声明的分项 "{extra}"')
        for key, val in bytes_map.items():
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                gaps.append(f'步骤 "{s["id"]}" 的 ledger.bytes["{key}"]={val!r} —— '
                            f"字节数必须是非负整数")
        cfg = led.get("config")
        if not isinstance(cfg, dict):
            gaps.append(f'步骤 "{s["id"]}" 的 ledger 缺 config —— '
                        f"没有它就无法独立重算这一步的字节数，对拍会退化成自证")
            continue
        for key in ("phase", "n_q", "seq_len", "kv_rows_total"):
            if key not in cfg:
                gaps.append(f'步骤 "{s["id"]}" 的 ledger.config 缺 "{key}"')
        for key in ("n_q", "seq_len", "kv_rows_total"):
            v = cfg.get(key)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                gaps.append(f'步骤 "{s["id"]}" 的 ledger.config["{key}"]={v!r} —— '
                            f"必须是非负整数")

        # An internal-consistency rule that costs nothing to state and catches a
        # whole class of bookkeeping slip: the cache holds one row per layer per
        # position, so the two counts in `config` cannot disagree. Without this,
        # `kv_rows_total` could be half what it should be and every downstream
        # check would still pass -- they all take it as given.
        layers = cfg.get("layers")
        if not isinstance(layers, int) or isinstance(layers, bool) or layers < 1:
            gaps.append(f'步骤 "{s["id"]}" 的 ledger.config 缺 "layers"（正整数）—— '
                        f"没有它就无法校验 kv_rows_total")
        elif isinstance(cfg.get("seq_len"), int) and isinstance(cfg.get("kv_rows_total"), int):
            if cfg["kv_rows_total"] != layers * cfg["seq_len"]:
                gaps.append(
                    f'步骤 "{s["id"]}" 的 ledger.config 自相矛盾：'
                    f'kv_rows_total={cfg["kv_rows_total"]} 但 layers×seq_len='
                    f'{layers}×{cfg["seq_len"]}={layers * cfg["seq_len"]} —— '
                    f"每层每个位置恰好一行，这两个数必须相等")
    return gaps


_COST_SHAPE_KEYS = ("q_shape", "k_shape", "score_shape")
_COST_INT_KEYS = ("queries", "score_elems", "kv_read_elems", "kv_write_elems",
                  "attn_flops", "proj_flops")

# The fields that describe the COMPUTATION rather than the cache traffic. The
# prefill control compares exactly these, and not the whole mode object: at
# prefill the two modes genuinely disagree about the cache — the cached path
# writes the prompt's rows into it, the uncached path has no cache at all — so
# comparing the traffic would make the control fail for a reason that is not a
# difference in the work done. What prefill must not differ in is the shapes
# and the FLOPs, and that is what is asserted.
_WORK_KEYS = ("queries", "q_shape", "k_shape", "score_shape", "score_elems",
              "attn_flops", "proj_flops")


def _work_of(mode):
    return {k: mode[k] for k in _WORK_KEYS}


def _cost_shape_elems(shape):
    n = 1
    for x in shape:
        n *= x
    return n


def lint_attn_cost(trace):
    """The with/without-cache contract, added with L05's lab page.

    Optional, like the ledger block: a trace that does not declare `attn_cost`
    on its attention step lints clean and every rule here stays inert. That is
    what lets this lab's other views share lint entry points with traces that
    never heard of the comparison.

    Two rules carry the weight, and they point in opposite directions:

      * the two modes' costs must be re-derivable from the shapes they publish,
        so the picture on the page cannot show a matrix its own shape fields
        contradict;
      * at prefill the two modes must be EQUAL — with an empty cache there is
        nothing to recompute — which is the control the decode divergence is
        read against. A lab whose two columns differed in every step would be
        showing a difference it never isolated.
    """
    gaps = []
    steps = [s for s in trace.get("steps", []) if isinstance(s, dict)]
    with_cost = [s for s in steps if "attn_cost" in s]
    if not with_cost:
        if (trace.get("meta") or {}).get("attn_cost"):
            gaps.append("trace 没有步骤带 attn_cost，但 meta.attn_cost 存在 —— "
                        "汇总没有逐帧依据")
        return gaps

    for s in with_cost:
        sid, c = s["id"], s["attn_cost"]
        # Per-step, so a step whose fields are the wrong TYPE is reported once
        # and skipped, rather than either crashing the derived checks or
        # suppressing them for every step that follows.
        local = []
        for key in ("t", "n_q", "H", "d_head", "layers", "no_cache", "with_cache"):
            if key not in c:
                local.append(f'步骤 "{sid}" 的 attn_cost 缺 "{key}"')
        for mode in ("no_cache", "with_cache"):
            m = c.get(mode)
            if not isinstance(m, dict):
                local.append(f'步骤 "{sid}" 的 attn_cost["{mode}"] 不是对象')
                continue
            for key in _COST_SHAPE_KEYS:
                sh = m.get(key)
                if not (isinstance(sh, list) and sh and
                        all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                            for x in sh)):
                    local.append(f'步骤 "{sid}" 的 attn_cost["{mode}"].{key}={sh!r} '
                                 f"不是正整数的 shape")
            for key in _COST_INT_KEYS:
                v = m.get(key)
                if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
                    local.append(f'步骤 "{sid}" 的 attn_cost["{mode}"].{key}={v!r} '
                                 f"必须是非负整数")
        if local:
            gaps += local
            continue

        nc, wc = c["no_cache"], c["with_cache"]
        # The score matrix's element count IS its shape's product. This is the
        # rule that stops "the shape got wider" and "the work got bigger" from
        # becoming two independent claims on the same page.
        for name, m in (("no_cache", nc), ("with_cache", wc)):
            if m["score_shape"] != [c["H"], m["queries"], c["t"]]:
                gaps.append(f'步骤 "{sid}" 的 {name}.score_shape={m["score_shape"]} '
                            f'应为 [H,queries,t]=[{c["H"]},{m["queries"]},{c["t"]}]')
            elif m["score_elems"] != _cost_shape_elems(m["score_shape"]):
                gaps.append(f'步骤 "{sid}" 的 {name}.score_elems={m["score_elems"]} '
                            f'与 shape {m["score_shape"]} 的乘积不符')
            # The score matrix's element count IS its shape's product. This is
            # the rule that stops "the shape got wider" and "the work got
            # bigger" from becoming two independent claims on the same page.
            if m["queries"] != m["score_shape"][1] or m["q_shape"][0] != m["queries"]:
                gaps.append(f'步骤 "{sid}" 的 {name}：q_shape={m["q_shape"]} 与 '
                            f'queries={m["queries"]} 不一致')
            if m["k_shape"][0] != c["t"]:
                gaps.append(f'步骤 "{sid}" 的 {name}.k_shape={m["k_shape"]} —— '
                            f"K 的行数应为上下文 t={c['t']}")
        # The uncached path's query count IS the context: it recomputes every
        # position, which is the whole reason its score matrix is square.
        if nc["queries"] != c["t"]:
            gaps.append(f'步骤 "{sid}"：无缓存时 queries={nc["queries"]} 应等于 '
                        f"上下文 t={c['t']}（它重算整段前缀）")
        if wc["queries"] != c["n_q"]:
            gaps.append(f'步骤 "{sid}"：有缓存时 queries={wc["queries"]} '
                        f"应等于本步新查询数 n_q={c['n_q']}")
        # The cache traffic, re-derived from the ledger's own KV formula. It is
        # the term the uncached mode does not pay — it re-projects K and V
        # instead — so a page that charged it to both modes, or to neither,
        # would be telling a different story about why the cache costs memory.
        kv_read, kv_write = 2 * c["layers"] * c["H"] * c["d_head"] * c["t"], \
                            2 * c["layers"] * c["H"] * c["d_head"] * c["n_q"]
        if (wc["kv_read_elems"], wc["kv_write_elems"]) != (kv_read, kv_write):
            gaps.append(f'步骤 "{sid}" 的 with_cache KV 访存 '
                        f'({wc["kv_read_elems"]}, {wc["kv_write_elems"]}) 应为 '
                        f"({kv_read}, {kv_write}) = 2·L·H·d_h·(t, n_q)")
        if (nc["kv_read_elems"], nc["kv_write_elems"]) != (0, 0):
            gaps.append(f'步骤 "{sid}" 的 no_cache KV 访存应为 0 —— '
                        f"不缓存就没有缓存可读可写，它的代价落在投影 FLOPs 上")
        # The projections are the term that makes the uncached path quadratic,
        # so they are re-derived from t rather than trusted. `q_shape` is [q, d],
        # and each of Q/K/V is a d×d projection, so the count is 6·q·d²·L.
        for name, m in (("no_cache", nc), ("with_cache", wc)):
            d = m["q_shape"][1]
            want_proj = 6 * m["queries"] * d * d * c["layers"]
            if m["proj_flops"] != want_proj:
                gaps.append(f'步骤 "{sid}" 的 {name}.proj_flops={m["proj_flops"]} '
                            f"应为 6·q·d²·L={want_proj}")

    # The prefill control. Named by the step's own id prefix rather than by an
    # index: an index would silently point at a different step the day the
    # replay's shape changes.
    prefills = [s for s in with_cost if s["id"].startswith("prefill")]
    for s in prefills:
        nc, wc = s["attn_cost"]["no_cache"], s["attn_cost"]["with_cache"]
        if _work_of(nc) != _work_of(wc):
            gaps.append(f'步骤 "{s["id"]}"：Prefill 时缓存是空的，两种模式必须做'
                        f"同样的工作，但 no_cache={nc['score_elems']} "
                        f"with_cache={wc['score_elems']} 个分数矩阵元素")
    if not prefills:
        gaps.append("没有任何 prefill 步骤带 attn_cost —— 两种模式的重合点无法校验")
    else:
        # ...and the divergence must actually happen somewhere, or the control
        # above is the only thing the comparison ever shows.
        diverged = [s for s in with_cost
                    if _work_of(s["attn_cost"]["no_cache"]) !=
                    _work_of(s["attn_cost"]["with_cache"])]
        if not diverged:
            gaps.append("所有步骤两种模式都相同 —— decode 阶段没有出现分歧，"
                        "这份 trace 演示不了这个 lab 的对比")

    meta = (trace.get("meta") or {}).get("attn_cost")
    if isinstance(meta, dict) and isinstance(meta.get("totals"), dict):
        for mode in ("no_cache", "with_cache"):
            tot = meta["totals"].get(mode)
            if not isinstance(tot, dict):
                gaps.append(f'meta.attn_cost.totals 缺 "{mode}"')
                continue
            for key in ("score_elems", "attn_flops", "proj_flops",
                        "kv_read_elems", "kv_write_elems"):
                want = sum(s["attn_cost"][mode][key] for s in with_cost)
                if tot.get(key) != want:
                    gaps.append(f'meta.attn_cost.totals["{mode}"].{key}={tot.get(key)} '
                                f"与逐步之和 {want} 不符")
    elif with_cost:
        gaps.append("步骤带了 attn_cost，但 meta.attn_cost.totals 缺失 —— "
                    "页面上的合计会无处可查")
    return gaps


_PHASE_IDS = ("prefill", "decode", "doc")


def lint_phase_roofline(trace):
    """L05's own Roofline contract, added with the lab page.

    It is a *separate* key from `meta.roofline` on purpose — see the module
    docstring — so this rule set is the one that governs the chart L05 draws,
    and `views/phase-roofline.js` carries its port.

    The rules are the ones that make the chart mean what the page says it
    means: the ceiling is `min(peak, bandwidth·ai)` and `bound` names the roof
    that produced it; `ai` is the point's own FLOPs over its own bytes; the
    totals equal their own parts; and the reference point is the SAME dot on
    every configuration, because that stillness is what the two live points
    move against.
    """
    gaps = []
    roof = (trace.get("meta") or {}).get("phase_roofline")
    if not isinstance(roof, dict):
        gaps.append("meta.phase_roofline 缺失 —— Prefill/Decode 的 Roofline 没有数据")
        return gaps

    required = ("peak_flops", "bandwidth", "ridge", "points", "x_label", "y_label",
                "y_divisor", "y_unit", "note", "traffic_note", "model")
    for k in required:
        if k not in roof:
            gaps.append(f"meta.phase_roofline 缺 {k} —— 轴或说明会显示 undefined")

    peak, bw = roof.get("peak_flops"), roof.get("bandwidth")
    if not (isinstance(peak, (int, float)) and isinstance(bw, (int, float)) and bw > 0):
        gaps.append("meta.phase_roofline 的 peak_flops / bandwidth 不是正数")
        return gaps
    if not math.isclose(peak, PEAK_FLOPS, rel_tol=1e-12) or \
       not math.isclose(bw, BW_BYTES, rel_tol=1e-12):
        gaps.append(f"meta.phase_roofline 的硬件模型 ({peak!r}, {bw!r}) 与生成器里的 "
                    f"H100 常数 ({PEAK_FLOPS!r}, {BW_BYTES!r}) 不一致")
    ridge = peak / bw
    if not math.isclose(roof.get("ridge", 0), ridge, rel_tol=1e-9):
        gaps.append(f"meta.phase_roofline.ridge={roof.get('ridge')!r} 应为 "
                    f"峰值算力 / 带宽 = {ridge}")

    yd = roof.get("y_divisor")
    if not (isinstance(yd, (int, float)) and yd > 0):
        gaps.append(f"meta.phase_roofline.y_divisor={yd!r} 应为正数")
    else:
        unit = roof.get("y_unit")
        if not isinstance(unit, str) or not unit:
            gaps.append(f"meta.phase_roofline.y_unit={unit!r} 应为非空字符串")
        elif unit not in str(roof.get("y_label", "")):
            gaps.append(f'meta.phase_roofline.y_label 里没有写明单位 "{unit}"')

    pts = roof.get("points")
    ids = [p.get("id") for p in pts] if isinstance(pts, list) else []
    if ids != list(_PHASE_IDS):
        gaps.append(f"meta.phase_roofline.points 的 id 是 {ids!r}，"
                    f"应该是 {list(_PHASE_IDS)!r} —— 顺序也参与渲染")
        return gaps

    for p in pts:
        pid = p["id"]
        for k in ("label", "ai", "flops", "bytes", "parts", "ceiling", "bound", "config"):
            if k not in p:
                gaps.append(f'meta.phase_roofline.points["{pid}"] 缺 {k}')
        if gaps:
            continue
        flops, nbytes, parts = p["flops"], p["bytes"], p["parts"]
        if not (isinstance(nbytes, int) and nbytes > 0):
            gaps.append(f'meta.phase_roofline.points["{pid}"].bytes={nbytes!r} 应为正整数')
            continue
        want_parts = ("proj_flops", "ffn_flops", "attn_flops", "head_flops",
                      "weights_bytes", "kv_read_bytes", "kv_write_bytes")
        if set(parts) != set(want_parts):
            gaps.append(f'meta.phase_roofline.points["{pid}"].parts 的键是 '
                        f'{sorted(parts)}，应为 {sorted(want_parts)}')
            continue
        # The total is its own parts, added up: one summary number cannot hide a
        # dropped term if the terms are published beside it and re-added here.
        flops_from_parts = sum(parts[k] for k in
                               ("proj_flops", "ffn_flops", "attn_flops", "head_flops"))
        bytes_from_parts = sum(parts[k] for k in
                               ("weights_bytes", "kv_read_bytes", "kv_write_bytes"))
        if flops != flops_from_parts:
            gaps.append(f'meta.phase_roofline.points["{pid}"].flops={flops} 与各分项之和 '
                        f"{flops_from_parts} 不符")
        if nbytes != bytes_from_parts:
            gaps.append(f'meta.phase_roofline.points["{pid}"].bytes={nbytes} 与各分项之和 '
                        f"{bytes_from_parts} 不符")
        if not math.isclose(p["ai"], flops / nbytes, rel_tol=1e-12):
            gaps.append(f'meta.phase_roofline.points["{pid}"].ai={p["ai"]!r} 与它自己的 '
                        f"flops / bytes = {flops / nbytes} 不一致")
        want_bound = "bandwidth" if p["ai"] < ridge else "compute"
        if p["bound"] != want_bound:
            gaps.append(f'meta.phase_roofline.points["{pid}"].bound={p["bound"]!r}，'
                        f'而在 AI={p["ai"]:.4g} 上起作用的是 "{want_bound}" 屋顶')
        want_ceiling = min(peak, bw * p["ai"])
        if not math.isclose(p["ceiling"], want_ceiling, rel_tol=1e-9):
            gaps.append(f'meta.phase_roofline.points["{pid}"].ceiling={p["ceiling"]!r} '
                        f"应为 min(峰值, 带宽×AI)={want_ceiling}")
        # Weights are read once per pass, so a point's weight traffic is its
        # parameters at its own dtype — the term that makes prefill's intensity
        # scale with the prompt and decode's not.
        cfgp = p.get("config") or {}
        if not isinstance(cfgp.get("dtype"), str):
            gaps.append(f'meta.phase_roofline.points["{pid}"].config.dtype 缺失')

    # The live points describe THIS configuration: their query count and context
    # are the replay's own. The reference point is a different model and is
    # deliberately not tied to this trace's sizes — tying it would be the wrong
    # rule, not a stricter one.
    by_id = {p["id"]: p for p in pts}
    cfg = trace.get("meta", {}).get("config") or {}
    pre_cfg = by_id["prefill"].get("config") or {}
    for key in ("n_q", "t", "prompt"):
        if pre_cfg.get(key) != cfg.get("prompt_len"):
            gaps.append(f'meta.phase_roofline 的 prefill 点用了 {key}={pre_cfg.get(key)}，'
                        f'而本配置的 prompt 是 {cfg.get("prompt_len")}')
    if pre_cfg.get("layers") != cfg.get("L") or pre_cfg.get("dtype") != cfg.get("dtype"):
        gaps.append(f"meta.phase_roofline 的 prefill 点与本配置的层数/精度不一致："
                    f'{pre_cfg} vs L={cfg.get("L")}, dtype={cfg.get("dtype")}')
    dec_cfg = by_id["decode"].get("config") or {}
    if dec_cfg.get("layers") != cfg.get("L") or dec_cfg.get("dtype") != cfg.get("dtype"):
        gaps.append(f"meta.phase_roofline 的 decode 点与本配置的层数/精度不一致："
                    f'{dec_cfg} vs L={cfg.get("L")}, dtype={cfg.get("dtype")}')
    dec_cfg = by_id["decode"].get("config") or {}
    if dec_cfg.get("n_q") != 1 or dec_cfg.get("t") != cfg.get("N"):
        gaps.append(f'meta.phase_roofline 的 decode 点用了 n_q={dec_cfg.get("n_q")}, '
                    f't={dec_cfg.get("t")}，而本配置的 decode 是 1 个 token 对 '
                    f'{cfg.get("N")} 个位置')
    # The reference point must be the same model on every configuration.
    doc_cfg = by_id["doc"].get("config") or {}
    if (doc_cfg.get("layers"), doc_cfg.get("t"), doc_cfg.get("prompt"),
            doc_cfg.get("dtype")) != (DOC_MODEL["L"], DOC_PROMPT, DOC_PROMPT, "fp16"):
        gaps.append(f"meta.phase_roofline 的 doc 参考点随配置变了：{doc_cfg} != "
                    f'{{"layers": {DOC_MODEL["L"]}, "t": {DOC_PROMPT}, '
                    f'"prompt": {DOC_PROMPT}, "dtype": "fp16"}}')

    # The lab's headline claim, as a relation between the two live points:
    # prefill does `prompt_len` times decode's work per byte, because the weight
    # bytes are paid once for the whole prompt.
    dec = by_id["decode"]
    if dec["ai"] > by_id["prefill"]["ai"] + 1e-12:
        gaps.append(f'meta.phase_roofline：decode 的 AI={dec["ai"]:.4g} '
                    f'不该高于 prefill 的 {by_id["prefill"]["ai"]:.4g}')

    if not isinstance(roof.get("traffic_note"), str) or not roof["traffic_note"]:
        gaps.append("meta.phase_roofline.traffic_note 缺失 —— "
                    "访存口径没有声明，图上的数字无从解释")
    return gaps


def lint_crossover(trace):
    """The published crossover point, re-derived from the ledger's own numbers.

    Two rules, and the second is the one with teeth: the crossover must equal
    `params_bytes / per_token_bytes`, AND both of those must equal the figures
    the ledger block already publishes at the last step. Without the second,
    the page could print a crossover computed from a different model than the
    one the bar above it is drawn from.
    """
    gaps = []
    cross = (trace.get("meta") or {}).get("kv_crossover")
    if not isinstance(cross, dict):
        gaps.append("meta.kv_crossover 缺失 —— 页面上「KV 何时反超参数」无处可查")
        return gaps
    for k in ("params_bytes", "per_token_bytes", "context"):
        if not isinstance(cross.get(k), (int, float)) or cross[k] <= 0:
            gaps.append(f"meta.kv_crossover.{k}={cross.get(k)!r} 应为正数")
    if gaps:
        return gaps
    if not math.isclose(cross["context"], cross["params_bytes"] / cross["per_token_bytes"],
                        rel_tol=1e-12):
        gaps.append(f'meta.kv_crossover.context={cross["context"]!r} 应为 '
                    f'params_bytes / per_token_bytes = '
                    f'{cross["params_bytes"] / cross["per_token_bytes"]}')

    # Tie it to the ledger: params is the last step's `params` segment (it does
    # not move with the context), and the per-token cost is the KV cache of a
    # one-token step at this configuration's layer count and dtype.
    steps = [s for s in trace.get("steps", []) if isinstance(s, dict) and s.get("ledger")]
    if not steps:
        return gaps
    led = steps[-1]["ledger"]
    last_params = led["bytes"].get("params")
    if last_params != cross["params_bytes"]:
        gaps.append(f'meta.kv_crossover.params_bytes={cross["params_bytes"]} 与末步账本的 '
                    f"参数分项 {last_params} 不符 —— 两处说的是同一个模型的参数")
    cfg = trace.get("meta", {}).get("config") or {}
    want_per_token = 2 * cfg.get("L", 0) * cfg.get("H", 0) * cfg.get("d_head", 0) * \
        cfg.get("kv_bytes", 0)
    if cross["per_token_bytes"] != want_per_token:
        gaps.append(f'meta.kv_crossover.per_token_bytes={cross["per_token_bytes"]} 应为 '
                    f"2·L·H_kv·d_h·b = {want_per_token}")
    return gaps


# Each entry breaks one rule on the real trace. Kept at module scope so the
# report can name how many were exercised.
SABOTAGE_CASES = {
    # -- the rules that predate this ticket (shared with L00)
    "read-but-never-written tensor with no init": lambda t: t["tensors"].pop("tokens"),
    "state entry for an undeclared tensor": lambda t: t["steps"][1]["state"].update({"nope": 1.0}),
    "binding missing a tier": lambda t: t["steps"][1]["bindings"].update({"TOLD": {"sym": "a"}}),
    "binding collapsed to a 2-tuple": lambda t: t["steps"][1]["bindings"].update(
        {"TNEW": {"num": "5"}}),
    "\\slot with no binding": lambda t: t["steps"][1]["formula"].update(
        {"num": t["steps"][1]["formula"]["num"] + r"\slot{GHOST}"}),
    "\\region with no description": lambda t: t["steps"][1]["formula"].update(
        {"sym": t["steps"][1]["formula"]["sym"] + r"\region{GHOST}{x}"}),
    "step with no graph node": lambda t: t["graph"]["nodes"].pop(0),
    "formula tier missing": lambda t: t["steps"][1]["formula"].pop("idx"),
    "raw -Infinity instead of the sentinel": lambda t: t["steps"][1]["state"].update({"h": "-Infinity"}),
    "unknown sentinel string": lambda t: t["steps"][1]["state"].update({"h": "-inf"}),
    # -- the ledger rules this ticket adds
    "ledger block removed but steps keep theirs": lambda t: t.pop("ledger"),
    "ledger.unit missing": lambda t: t["ledger"].pop("unit"),
    "segment id duplicated": lambda t: t["ledger"]["segments"].append(
        dict(t["ledger"]["segments"][0])),
    "segment color not a CSS colour": lambda t: t["ledger"]["segments"][0].update(
        {"color": "cornflowerblue"}),
    "segment formula missing": lambda t: t["ledger"]["segments"][0].pop("formula"),
    "segment label missing": lambda t: t["ledger"]["segments"][0].pop("label"),
    "a step is missing its ledger": lambda t: t["steps"][2].pop("ledger"),
    "a step's ledger drops a segment": lambda t: t["steps"][2]["ledger"]["bytes"].pop("kv_cache"),
    "a step's ledger invents a segment": lambda t: t["steps"][2]["ledger"]["bytes"].update(
        {"ghost": 1}),
    "negative byte count": lambda t: t["steps"][2]["ledger"]["bytes"].update({"params": -1}),
    "float byte count": lambda t: t["steps"][2]["ledger"]["bytes"].update({"params": 1.5}),
    "boolean byte count": lambda t: t["steps"][2]["ledger"]["bytes"].update({"params": True}),
    "ledger.config missing so bytes cannot be recomputed": lambda t: t["steps"][2]["ledger"].pop(
        "config"),
    "ledger.config missing a key": lambda t: t["steps"][2]["ledger"]["config"].pop("kv_rows_total"),
    "ledger.config missing the layer count": lambda t: t["steps"][2]["ledger"]["config"].pop("layers"),
    "kv_rows_total disagrees with layers x seq_len": lambda t: t["steps"][2]["ledger"]["config"].update(
        {"kv_rows_total": t["steps"][2]["ledger"]["config"]["kv_rows_total"] // 2}),
    "ledger.config with a non-integer": lambda t: t["steps"][2]["ledger"]["config"].update(
        {"seq_len": "5"}),
    # -- the with/without-cache comparison (this lab page's own rules)
    #
    # Every mutation names a concrete step rather than "the attention step", so
    # that a change to the replay's shape is a loud failure here rather than a
    # silent no-op that makes the lint look alive when it never ran.
    "attn attn_cost totals removed while steps keep theirs": lambda t: t["meta"].pop("attn_cost"),
    "attn attn_cost removed from a decode step": lambda t: t["steps"][5].pop("attn_cost"),
    "attn score shape contradicting its own element count": lambda t: t["steps"][1][
        "attn_cost"]["with_cache"].update({"score_elems":
                                           t["steps"][1]["attn_cost"]["with_cache"]["score_elems"] + 1}),
    "attn cached mode claiming the uncached query count": lambda t: t["steps"][5][
        "attn_cost"]["with_cache"].update({"queries": t["steps"][5]["attn_cost"]["t"]}),
    "attn uncached mode not recomputing the whole prefix": lambda t: t["steps"][5][
        "attn_cost"]["no_cache"].update({"queries": 1}),
    "attn prefill's two modes diverging": lambda t: t["steps"][1][
        "attn_cost"]["with_cache"].update({"score_elems": 7}),
    "attn no step diverging at all": lambda t: [
        t["steps"][i]["attn_cost"].update({"with_cache": dict(
            t["steps"][i]["attn_cost"]["no_cache"])}) for i in (5, 9, 13, 17)],
    "attn score shape not [H, queries, t]": lambda t: t["steps"][1][
        "attn_cost"]["with_cache"].update({"score_shape": [4, 1, 4]}),
    "attn FLOPs not 4Hqtd_h": lambda t: t["steps"][5][
        "attn_cost"]["with_cache"].update({"attn_flops": 1}),
    "attn K rows disagreeing with the context": lambda t: t["steps"][9][
        "attn_cost"]["with_cache"].update({"k_shape": [3, 64]}),
    "attn cached mode charging a KV read it does not do":
        lambda t: t["steps"][5]["attn_cost"]["with_cache"].update({"kv_read_elems": 0}),
    "attn uncached projection FLOPs not scaling with t": lambda t: t["steps"][5][
        "attn_cost"]["no_cache"].update({"proj_flops": 6 * t["steps"][5]["attn_cost"]["t"]}),
    "attn a step's cost fields collapsed to a string": lambda t: t["steps"][5][
        "attn_cost"]["with_cache"].update({"queries": "1"}),
    "attn totals disagreeing with the steps they summarise": lambda t: t["meta"][
        "attn_cost"]["totals"]["with_cache"].update({"score_elems": 1}),
    "attn totals missing a mode": lambda t: t["meta"]["attn_cost"]["totals"].pop("no_cache"),
    # -- the phase Roofline (this lab page's own rules)
    "roofline block missing": lambda t: t["meta"].pop("phase_roofline"),
    "roofline ridge not peak/bandwidth": lambda t: t["meta"][
        "phase_roofline"].update({"ridge": 5.0}),
    "roofline hardware diverging from the generator's": lambda t: t["meta"][
        "phase_roofline"].update({"peak_flops": 1.0}),
    "roofline points in the wrong order": lambda t: t["meta"][
        "phase_roofline"].update({"points": list(reversed(t["meta"]["phase_roofline"]["points"]))}),
    "roofline point ai disagreeing with its own flops/bytes": lambda t: t["meta"][
        "phase_roofline"]["points"][0].update({"ai": 1.0}),
    "roofline point ceiling not min(peak, bandwidth*ai)": lambda t: t["meta"][
        "phase_roofline"]["points"][1].update({"ceiling": 1.0}),
    "roofline point naming the wrong roof": lambda t: t["meta"][
        "phase_roofline"]["points"][1].update({"bound": "compute"}),
    "roofline point total disagreeing with its parts": lambda t: t["meta"][
        "phase_roofline"]["points"][0]["parts"].update({"ffn_flops": 1}),
    "roofline point missing a traffic term": lambda t: t["meta"][
        "phase_roofline"]["points"][0]["parts"].pop("kv_read_bytes"),
    "roofline point y-axis divisor missing": lambda t: t["meta"][
        "phase_roofline"].pop("y_divisor"),
    "roofline point y-axis unit not named in the label": lambda t: t["meta"][
        "phase_roofline"].update({"y_label": "可达算力上界"}),
    "roofline traffic note missing": lambda t: t["meta"]["phase_roofline"].pop("traffic_note"),
    "decode point tied to the reference model": lambda t: t["meta"][
        "phase_roofline"]["points"][1]["config"].update({"t": 512}),
    "prefill point not using this config's prompt": lambda t: t["meta"][
        "phase_roofline"]["points"][0]["config"].update({"n_q": 1}),
    "prefill point not using this config's layers": lambda t: t["meta"][
        "phase_roofline"]["points"][0]["config"].update({"layers": 64}),
    "reference point moving with the configuration": lambda t: t["meta"][
        "phase_roofline"]["points"][2]["config"].update(
        {"layers": t["meta"]["config"]["L"], "t": t["meta"]["config"]["N"]}),
    "reference point following the precision slider": lambda t: t["meta"][
        "phase_roofline"]["points"][2]["config"].update({"dtype": "fp8"}),
    "decode intensity exceeding prefill's": lambda t: t["meta"][
        "phase_roofline"]["points"][1].update(
        {"ai": t["meta"]["phase_roofline"]["points"][0]["ai"] * 2}),
    "roofline point config missing its prompt length": lambda t: t["meta"][
        "phase_roofline"]["points"][0]["config"].pop("prompt"),
    # -- the crossover point
    "crossover block missing": lambda t: t["meta"].pop("kv_crossover"),
    "crossover not equal to params / per_token": lambda t: t["meta"][
        "kv_crossover"].update({"context": 1.0}),
    "crossover computed from a different parameter count": lambda t: t["meta"][
        "kv_crossover"].update({"params_bytes": t["meta"]["kv_crossover"]["params_bytes"] + 2}),
    "crossover per-token cost not 2LHd_hb": lambda t: t["meta"][
        "kv_crossover"].update({"per_token_bytes": 8}),
}


def sabotage_checks(trace):
    """Prove the lint is not a function that always returns zero."""
    import copy

    failures = []

    def broken(mutate):
        t = copy.deepcopy(trace)
        mutate(t)
        return lint(t)[0]

    for name, mutate in SABOTAGE_CASES.items():
        try:
            gaps = broken(mutate)
        except Exception as exc:
            failures.append(f"{name}: lint raised {exc!r}")
            continue
        if not gaps:
            failures.append(f"{name}: broke the trace but lint reported 0 gaps")
    return failures


# ------------------------------------------------------------------- ledger check
def ledger_checks(trace, deploy):
    """The independent checks the acceptance criteria ask for.

    Two of them, and they are different in kind:

    * **sum == predicted total.** `stack()` in the component sums whatever
      segments it is handed; `model()` computes the total directly. Feeding the
      component's own prediction through its own summation and comparing against
      the direct total catches a dropped or double-counted segment in the
      renderer, which is the failure mode a stacked bar actually has.
    * **scaling laws.** Doubling the context must double KV Cache exactly and
      leave params untouched; doubling the layer count must double KV Cache
      exactly and change params by a known amount. These are properties of the
      formulas, not of one evaluation, so they catch a formula that is merely
      *self-consistent* while being wrong.

    Both are re-run in JavaScript, in `labs/assets/engine/views/ledger.js`, and
    the page's self-check panel shows the verdict. Running them here too means a
    broken accounting fails the generator rather than only the browser.
    """
    results = []
    meta_cfg = trace.get("meta", {}).get("config") or {}
    base = {"L": meta_cfg.get("L"), "H": GEOMETRY["H"], "d_head": GEOMETRY["d_head"],
            "d_model": GEOMETRY["d_model"], "d_ff": GEOMETRY["d_ff"],
            "vocab": GEOMETRY["vocab"], "B": GEOMETRY["B"],
            "kv_bytes": deploy["kv_bytes"], "act_bytes": deploy["act_bytes"]}

    def predict(t, phase="head", n_q=1, **over):
        """The shape-formula prediction for one configuration.

        `over` is merged into the base *before* the count, so an override
        genuinely replaces the base value. (Merging it afterwards is how the
        first version of this check managed to compare fp16 against fp16 and
        report "halving changes nothing" as a passing row.)
        """
        c = dict(base)
        c.update(over)
        L, H, dh, d = c["L"], c["H"], c["d_head"], c["d_model"]
        if "kv_rows_total" not in c:
            c["kv_rows_total"] = L * t
        counts = closed_form_counts(
            {"L": L, "H": H, "d_head": dh, "d_model": d,
             "d_ff": c["d_ff"], "vocab": c["vocab"], "B": c["B"]},
            phase, n_q, t, c["kv_rows_total"])
        # Bytes at *this* configuration's dtype -- what the other half of the
        # check needs to actually differ from the base.
        return {"params": counts["params"] * deploy["weight_bytes"],
                "kv_cache": counts["kv_cache"] * c["kv_bytes"],
                "activations": counts["activations"] * c["act_bytes"]}

    # 1. every published step: the formula prediction must equal the bytes this
    #    script counted from the real arrays.
    for s in trace["steps"]:
        led = s["ledger"]
        got = predict(led["config"]["seq_len"], phase=led["config"]["phase"],
                      n_q=led["config"]["n_q"], kv_rows_total=led["config"]["kv_rows_total"])
        if got != led["bytes"]:
            raise AssertionError(f'{s["id"]}: 公式预测 {got} != trace 实算 {led["bytes"]}')

    # 2. sum of segments == the directly computed total, at the last step.
    last_cfg = trace["steps"][-1]["ledger"]["config"]
    last = trace["steps"][-1]["ledger"]["bytes"]
    direct = predict(last_cfg["seq_len"], phase=last_cfg["phase"],
                     n_q=last_cfg["n_q"], kv_rows_total=last_cfg["kv_rows_total"])
    results.append(("各分项之和 == 该配置下的总显存预测值",
                    sum(last.values()) == sum(direct.values()),
                    f'{sum(last.values()):,} B'))

    # 3. scaling laws. Each isolates one factor, so a formula that is merely
    #    self-consistent while being wrong still has to survive all four.
    t0 = 128
    a = predict(t0)
    b2 = predict(2 * t0)
    results.append(("上下文长度翻倍 → KV Cache 精确翻倍，参数不动",
                    b2["kv_cache"] == 2 * a["kv_cache"] and b2["params"] == a["params"],
                    f'{a["kv_cache"]:,} → {b2["kv_cache"]:,} B'))

    for base_L in (2, 4):
        two_layer = predict(7, L=base_L)
        four_layer = predict(7, L=2 * base_L)
        results.append((f"层数翻倍（{base_L} → {2 * base_L}）→ KV Cache 精确翻倍",
                        four_layer["kv_cache"] == 2 * two_layer["kv_cache"],
                        f'{two_layer["kv_cache"]:,} → {four_layer["kv_cache"]:,} B'))

    fp16 = predict(7, kv_bytes=2)
    fp8 = predict(7, kv_bytes=1)
    results.append(("KV Cache 精度减半（fp16 → fp8）→ 字节数精确减半",
                    fp16["kv_cache"] == 2 * fp8["kv_cache"] and fp8["kv_cache"] > 0,
                    f'{fp8["kv_cache"]:,} → {fp16["kv_cache"]:,} B'))

    # This is the GQA/MQA claim the tutorial makes ("H_kv 在公式里是线性因子，所以
    # 32 头共享 8 组 KV 就把缓存砍到 1/4"), expressed as a property: doubling
    # H_kv at fixed d_head doubles the cache, and *only* the cache.
    mha = predict(7, H=GEOMETRY["H"])
    gqa_half = predict(7, H=GEOMETRY["H"] // 2)
    results.append(("KV 头减半（MHA → GQA）→ KV Cache 精确减半，参数不动",
                    mha["kv_cache"] == 2 * gqa_half["kv_cache"] and
                    mha["params"] == gqa_half["params"] and gqa_half["kv_cache"] > 0,
                    f'{gqa_half["kv_cache"]:,} → {mha["kv_cache"]:,} B'))

    return results


# --------------------------------------------------- the `--js-lint` payload
#
# The non-finite tagging `--js-lint` needs. L05 traces do carry sentinels in
# `state` (-∞ in the causal mask's neighbourhood), and `json.dumps` writes a
# bare `Infinity` for a float infinity — legal JavaScript, illegal JSON — so a
# payload carrying one could not be parsed at all. Tagged on the way out, and
# reversed by the harness before the trace reaches the lint.
def _tag_nonfinite(value):
    if isinstance(value, float):
        if math.isnan(value):
            return {"__nonfinite__": "nan"}
        if math.isinf(value):
            return {"__nonfinite__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, dict):
        return {k: _tag_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tag_nonfinite(v) for v in value]
    return value


def _payload_json(payload):
    return json.dumps(_tag_nonfinite(payload), ensure_ascii=False)


# ------------------------------------------------------------------------ set
PROMPT_SEED = 20240522


def grid():
    """Every configuration the page's sliders can reach, in grid order.

    A full product, like L01's: three independent controls rather than three
    positions of one knob. The order is the nesting order of the loops
    (context → layers → precision), which is also the order the manifest lists
    them and the order the trace files are written in.
    """
    return [(N, L, prec) for N in GRID_N for L in GRID_L for prec in GRID_PREC]


def prompt_for(cfg, prec):
    """A deterministic prompt of the right length.

    Derived from a fixed seed and the configuration, so two runs of this script
    produce byte-identical traces — which is what `check-traces-reproducible.py`
    verifies. The precision is folded in by its INDEX rather than by hashing the
    string: `hash()` is salted per process, and a trace set that changed between
    runs would fail the reproducibility check in a way that looks like a code
    bug.
    """
    rng = np.random.default_rng(PROMPT_SEED + cfg["N"] * 101 + cfg["L"] * 17
                                + GRID_PREC.index(prec))
    return [int(x) for x in rng.integers(0, cfg["vocab"], size=cfg["prompt_len"])]


def build_one(N, L, prec):
    """One configuration: its trace, plus the cross-check pairs it produced."""
    cfg = config_for(N, L, prec)
    deploy = PRECISIONS[prec]
    p = init_weights(cfg)
    real_params = sum(int(w.size) for w in p.values())
    want_params = analytic_param_count(cfg)
    if real_params != want_params:
        raise AssertionError(f"参数个数：模型实际 {real_params} / 闭式 {want_params}")
    trace = build_trace(cfg, p, deploy, prompt_for(cfg, prec))
    return trace, trace.pop("_checks")


def main():
    only_json = "--js-lint" in sys.argv
    real_stdout = sys.stdout
    if only_json:
        # Every human-readable line goes to stderr so stdout carries the JSON
        # payload and nothing else -- the parity harness pipes it straight into
        # `node`. Same contract as L01's generator.
        sys.stdout = sys.stderr

    fails = []
    built = {}
    for (N, L, prec) in grid():
        name = cfg_id(N, L, prec)
        trace, checks = build_one(N, L, prec)
        cfg = config_for(N, L, prec)
        deploy = PRECISIONS[prec]
        print(f"\n{name}: 上下文 {N} · {L} 层 · {prec} · {len(trace['steps'])} 步 · "
              f"{len(json.dumps(trace, ensure_ascii=False)):,} bytes")

        # --- the published precision, asserted against what the page shows ---
        #
        # The page renders four significant figures; `STATE_SIG_FIGS` must stay
        # strictly above that or a reader would be looking at values the trace
        # rounded past what is drawn. This is the guard on the size cut, and it
        # is cheap: a projection that quietly dropped below display precision
        # would otherwise show up only as numbers that look slightly off.
        if STATE_SIG_FIGS <= 4:
            fails.append(f"{name}: STATE_SIG_FIGS={STATE_SIG_FIGS} 不高于页面显示的 4 位有效数字")

        # --- 1 & 2: numpy cache path vs torch no-cache and vs numpy no-cache
        worst, worst_at = 0.0, ""
        for cname, got, want in checks:
            dev = max_rel_dev(got, want)
            if dev > worst:
                worst, worst_at = dev, cname
            if dev > _REL_TOL:
                fails.append(f"{name}: {cname} 最大相对偏差 {dev:.3e} > {_REL_TOL:.0e}")
        print(f"  对拍：{len(checks)} 项全部一致（每一步 {trace['meta']['config']['vocab']} "
              f"维 logits 逐元素对 torch 无缓存全前缀重算），实测最大相对偏差 "
              f"{worst:.3e}（{worst_at}）")

        # --- ledger cross-check: real arrays vs closed form, per step (asserted
        # in build_trace), plus the scaling laws.
        # `what`, not `label`: the module-level `label()` builds a
        # configuration's human-readable name, and shadowing it here is how the
        # first version of this loop managed to crash while writing a manifest
        # it had already computed.
        for what, ok, detail in ledger_checks(trace, deploy):
            if not ok:
                fails.append(f"{name}: {what}")
            print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  — {detail}" if detail else ""))

        last = trace["steps"][-1]["ledger"]["bytes"]
        print(f"  末步账本： " + " / ".join(f"{k}={v:,}" for k, v in last.items()) +
              f"  合计 {sum(last.values()):,} B")
        c = trace["meta"]["kv_crossover"]
        print(f"  交叉点：参数 {c['params_bytes']:,} B / 每 token KV {c['per_token_bytes']} B "
              f"→ 上下文 {c['context']:,.0f} 之后 KV 反超参数")

        # --- 3: lint + control group
        gaps, warns, infos = lint(trace)
        for line in infos:
            print(f"  info  {line}")
        for line in warns:
            print(f"  warn  {line}")
        for line in gaps:
            print(f"  GAP   {line}")
            fails.append(f"{name}: {line}")
        sabotages = sabotage_checks(trace)
        if sabotages:
            fails.append(f"{name}: lint 对照组失效")
            for line in sabotages:
                print(f"  LINT-DEAD  {line}")
        else:
            print(f"  lint: {len(gaps)} gap / {len(warns)} warn；"
                  f"对照 {len(SABOTAGE_CASES)} 种破坏全部被抓到")
        built[name] = trace

    # ------------------------------------------------------ cross-configuration
    #
    # What one trace cannot check about itself: that the sliders move something,
    # and that what they move is the thing they are about. A set in which every
    # context gave the same KV cache would lint perfectly and teach nothing.
    if not fails:
        def last_bytes(t, seg):
            return t["steps"][-1]["ledger"]["bytes"][seg]

        kv_by_n = {}
        for g in grid():
            N, L, prec = g
            kv_by_n.setdefault((L, prec), {})[N] = last_bytes(built[cfg_id(*g)], "kv_cache")
        for (L, prec), series in sorted(kv_by_n.items(), key=lambda x: str(x[0])):
            ns = sorted(series)
            # KV Cache is `2·L·H_kv·d_h·N·b`: linear in the context through the
            # origin. The slope between the two extreme points is exact, and the
            # zero intercept is checked by comparing the smallest N's value
            # against the formula directly — a `+ c` term would survive the
            # slope and die here.
            want = 2 * L * GEOMETRY["H"] * GEOMETRY["d_head"] * PRECISIONS[prec]["kv_bytes"]
            slope = (series[ns[-1]] - series[ns[0]]) / (ns[-1] - ns[0])
            if slope != want:
                fails.append(f"L={L} {prec}: KV Cache 对上下文长度的斜率 {slope} "
                             f"应为 2·L·H·d_h·b = {want}")
            if series[ns[0]] != want * ns[0]:
                fails.append(f"L={L} {prec}: N={ns[0]} 的 KV Cache {series[ns[0]]} "
                             f"应为 {want}×{ns[0]}")
        print(f"\n跨配置：KV Cache 对上下文长度严格线性（斜率 2·L·H_kv·d_h·b），"
              f"{(len(GRID_L) * len(GRID_PREC))} 组 (L, 精度) 各验一次")

        # Layer count doubles the cache and only the cache.
        for N in GRID_N:
            for prec in GRID_PREC:
                two = last_bytes(built[cfg_id(N, 2, prec)], "kv_cache")
                four = last_bytes(built[cfg_id(N, 4, prec)], "kv_cache")
                if four != 2 * two:
                    fails.append(f"N={N} {prec}: 层数翻倍后 KV Cache {two} → {four} 不是精确翻倍")
        # Precision: fp8 and int8 are the same width, and the set says so.
        for N in GRID_N:
            for L in GRID_L:
                fp16 = last_bytes(built[cfg_id(N, L, "fp16")], "kv_cache")
                for p8 in ("fp8", "int8"):
                    if last_bytes(built[cfg_id(N, L, p8)], "kv_cache") * 2 != fp16:
                        fails.append(f"N={N} L={L}: {p8} 的 KV Cache 不是 fp16 的一半")
                if (last_bytes(built[cfg_id(N, L, "fp8")], "kv_cache") !=
                        last_bytes(built[cfg_id(N, L, "int8")], "kv_cache")):
                    fails.append(f"N={N} L={L}: fp8 与 int8 的 KV Cache 应等宽")
        print("跨配置：层数翻倍 → KV Cache 精确翻倍；fp16 → fp8/int8 精确减半"
              "（fp8 与 int8 等宽，这是精度格式的差别，不是字节数的差别）")

        # The parameters do not move with the context — the control that makes
        # the cache's growth readable as growth of THAT segment. Asserted per
        # (L, precision) group rather than by counting distinct values: fp8 and
        # int8 share a byte width, so the set of distinct parameter totals is
        # smaller than the number of groups, and a count would confuse the two.
        for L in GRID_L:
            for prec in GRID_PREC:
                vals = {last_bytes(built[cfg_id(N, L, prec)], "params") for N in GRID_N}
                if len(vals) != 1:
                    fails.append(f"L={L} {prec}: 参数分项随上下文长度变了：{sorted(vals)}")
        print(f"跨配置：参数分项在每个 (层数, 精度) 组内都不随上下文变化 —— "
              f"对照组（KV 那条在动，这条不动）")

        # The lab's central contrast, at the set level: the uncached path's work
        # grows with the square of the context while the cached path's does not.
        ratios = {}
        for N in GRID_N:
            t = built[cfg_id(N, 2, "fp16")]
            tot = t["meta"]["attn_cost"]["totals"]
            ratios[N] = tot["no_cache"]["score_elems"] / tot["with_cache"]["score_elems"]
            print(f"  上下文 {N}: 分数矩阵元素 无缓存 {tot['no_cache']['score_elems']:,} vs "
                  f"有缓存 {tot['with_cache']['score_elems']:,} → {ratios[N]:.2f}×")
        ns = sorted(ratios)
        if not all(ratios[a] < ratios[b] for a, b in zip(ns, ns[1:])):
            fails.append(f"上下文变长时「无缓存/有缓存」的工作量倍数应单调增大，实得 {ratios}")
        print(f"跨配置：上下文越长，不做缓存的代价越大（{ratios[ns[0]]:.1f}× → "
              f"{ratios[ns[-1]]:.1f}×）")

        # The reference point on the phase Roofline is the same dot everywhere.
        doc_pts = {(p["flops"], p["bytes"]) for t in built.values()
                   for p in t["meta"]["phase_roofline"]["points"] if p["id"] == "doc"}
        if len(doc_pts) != 1:
            fails.append(f"Roofline 的参考点随配置变了：{sorted(doc_pts)}")
        print("跨配置：Roofline 的参考点（7B / 512 prompt）在所有配置上都是同一个点 —— "
              "两个活点移动的对照")

    if only_json:
        base_name = cfg_id(*DEFAULT_TRIPLE)
        base = built[base_name]
        payload = {"traces": built, "base": base_name, "grid": [list(g) for g in grid()],
                   "sabotages": {}}
        import copy as _copy
        for name, mutate in SABOTAGE_CASES.items():
            broken = _copy.deepcopy(base)
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

    written = []
    for (N, L, prec) in grid():
        name = cfg_id(N, L, prec)
        trace = built[name]
        out = HERE / f"{name}.json"
        out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n")
        written.append(name)
        print(f"  {out.name}: {out.stat().st_size:,} bytes")

    manifest = {
        "set": SET,
        "lab": "L05",
        "default": cfg_id(*DEFAULT_TRIPLE),
        "params": {"N": sorted(GRID_N), "L": sorted(GRID_L), "prec": list(GRID_PREC)},
        "traces": written,
        "labels": {cfg_id(*g): label(*g) for g in grid()},
        # The page inlines the whole set and matches a slider position to a
        # trace by reading the trace's own `meta.config` — the ids are for
        # humans and for the manifest's internal consistency, never parsed.
        "note": "配置 id 只给人看；页面按 meta.config 匹配滑杆位置，不解析 id。",
    }
    (HERE / f"{SET}.manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"\n{SET}.manifest.json: {len(written)} 个配置，默认 {manifest['default']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
