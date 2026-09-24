#!/usr/bin/env python3
"""Trace generator for L04 · Decoder Block: residual / LayerNorm / SwiGLU.

Emits a trace SET — one trace per slider position, plus the manifest that names
them (see "Trace sets" in `labs/traces/README.md`):

    labs/traces/decoder-block-d64-dff176.json   the default (the shared example model)
    labs/traces/decoder-block-d32-dff88.json    …
    labs/traces/decoder-block.manifest.json     written by this script

over `(d_model, d_ff)` = {32, 64, 128} × {88, 176, 352}. The d_ff values are the
tutorial's own `(8/3)·d_model` rounded up to a multiple of 8 — 32→85.33→88,
64→170.67→176 (the shared example model's own number), 128→341.33→352 — so the
ratio the tutorial argues about is the ratio the sliders actually sweep, and the
middle of the grid is the configuration every other lab replays.

## The replay

One Pre-Norm block, `x → LN → MHA → +x → LN → SwiGLU → +x`, in 14 steps. The
MHA is carried in four of them (projections, scores+mask, softmax→PV→W_O)
because the block has to be complete for the two residuals to mean anything, but
attention is L03's subject and this lab does not linger on it. The other ten
steps are the ticket's: two LayerNorms, two residual adds, and five SwiGLU steps
that put the gate, up and down paths on screen one at a time.

## What is measured, and what the control for it is

Four blocks, each published per step, each with a rule that has two sides:

**`step.residual`** — the residual add. The ticket's first criterion is
"两侧 shape 相同时在图上被确认，**并演示若 shape 不匹配会怎样**", and the
second half is the one with content: it is not enough to say the two sides are
equal. So the step publishes the *actual outcome* of adding each of eight
candidate right-hand shapes to this step's real left-hand value, executed here:

    identical      x + F(x)          -> the residual add
    broadcast      x + [d]           -> numpy AND torch SILENTLY broadcast
    broadcast      x + [1, d]        -> …
    broadcast      x + [N, 1]        -> …
    error          x + [N, d/2]      -> ValueError, verbatim
    error          x + [2N, d]       -> ValueError, verbatim
    error          x + [N, d, 1]     -> ValueError, verbatim

The broadcast rows are the point. `[d]` is a per-feature bias and adding it is
legitimate arithmetic — but it is NOT a residual connection, the result has the
right shape, and nothing raises. A page that only showed "the shapes are equal,
so it works" would leave a reader believing the shape check is a formality. The
error rows carry numpy's own exception text, so the failure is quoted rather
than described. The identical row is cross-checked against an explicit
double-loop sum, so "numpy adds elementwise" is measured rather than assumed.

**`step.ln`** — LayerNorm. The ticket's second criterion is the one it calls
最容易被讲糊: which axis the mean and variance are taken over. The step
publishes, for the real array: the per-token μ and σ (N of each, all different),
and then the statistics of the normalized result **broken out by both axes** —
`(x−μ)/√(σ²+ε)` over `d_model` leaves every ROW at mean 0 / std 1 while the
COLUMNS are not, and that asymmetry is what makes the axis legible rather than
asserted. The control is the counterfactual, actually computed: normalizing over
the *token* axis instead flips the two, and the rule requires the flip to be
present. Without it, "the rows came out at 0/1" is equally consistent with a
picture that normalized nothing at all.

**`step.swiglu`** — the FFN. `SwiGLU(x) = W_down(SiLU(W_gate x) ⊙ W_up x)`.
Each of the five FFN steps publishes all three weight shapes, all three
intermediate shapes, and which of the three paths it is on, so the picture can
label the dimensions at every hop. The elementwise multiply's own step carries
the element it multiplies (row, column, both operands, the product), read off
the real arrays — the `⊙` is not a claim about what happened, it is one cell of
it shown.

**`meta.norm_contrast`** — Pre-Norm vs Post-Norm. This is the ticket's `#40`
material: `3.6-LayerNorm与残差连接深入理解.md` has two mermaid graphs with
identical node names whose only difference is where the LayerNorm sits. They are
hand-converted here rather than parsed, and the reason is recorded in
`labs/traces/README.md` — but the conversion is not a redrawing. Two things are
computed on the real thing:

  * **the residual-stream scale by depth.** The same block stack is run under
    both placements for depths 1…32 and the RMS of the residual stream is
    recorded after every block. Post-Norm holds at 1.0000 by construction —
    that is what a LayerNorm output IS, and it is the control that makes the
    other curve readable. Pre-Norm's grows (≈1.5 → ≈8 in this model), which is
    the "各层方差随深度增长" the tutorial's §4.3 cites from Xiong et al.
  * **the gradient scale by depth.** `L = ⟨y_L, v⟩` for a fixed random `v`, so
    the gradient arriving back at each block boundary is a measurement rather
    than an illustration. Under Pre-Norm it comes back amplified ≈20×; under
    Post-Norm ≈100–700×. The probe is ONE direction, not an expectation — so
    the generator re-runs the whole sweep on two further probe seeds and
    requires the ordering to hold, which is how "the difference is a property of
    the architecture, not of this draw" is checked rather than assumed.

The flow lists are published as data (`["x","LN","SubLayer","Add"]` vs
`["x","SubLayer","Add","LN"]`) and the lint requires the two to be permutations
of one another differing at exactly one index, whose element is the norm — so
"only the norm position differs" is enforced, not written in prose above a
picture that might have drifted.

**`meta.params` / `step.params`** — the parameter count. L04 renders this
through the **existing** memory-ledger component (`labs/assets/engine/views/
ledger.js`, ticket #45) rather than growing a second stacked bar, and the
accounting is counted twice as that component requires: once from the real
weight arrays' `arr.size`, once from the tutorial's closed form
`4d² + 3·d·d_ff + 2d` (`3.7 §6.1`). Both are published, per segment, so the
page's slider recomputation has something to be checked against.

## Why `meta.params` and not `trace.ledger`

`views/ledger.js` is frozen by this ticket, and its **lint** is bound to L05's
configuration keys — it requires `phase` / `n_q` / `layers` / `seq_len` /
`kv_rows_total` on every step and requires `kv_rows_total == layers × seq_len`,
because for L05 that invariant is the KV cache's whole content. L04 has no KV
cache. Satisfying that rule would mean publishing a `kv_rows_total` that names
nothing, which is exactly the kind of number this project's contract exists to
keep off a page. So L04 renders through the component (the bar, `stack()`, its
byte formatting and its table) and declares its own block, with its own rules
and its own two ports — the arrangement L05 itself used for `roofline.js` when
that view's lint turned out to be bound to L01's roles. The component's
`check()` still runs and its three verdicts are still reported; only its
embedded `lint` verdict is ignored here, and the page says so.

## Checks that run before anything is written

  1. **torch, whole block.** The numpy forward is asserted against a torch
     implementation of the same block built independently (its own parameters,
     its own LayerNorm, its own causal mask). A bug in one cannot cancel.
  2. **numpy, the residual add.** The `identical` row is re-derived with an
     explicit double loop, so the elementwise claim is measured.
  3. **The parameter count, twice** — real arrays vs the closed form.
  4. **The contract lint**, plus a control group of deliberately broken copies
     the lint must catch, and every one of those copies must actually differ
     from the trace it was made from.
  5. **The cross-configuration claims** — that d_ff moves only the FFN segment,
     that d_model moves the attention segment quadratically, and that the
     Pre/Post ordering holds on probe seeds the published trace does not use.

The script refuses to write a file if any of the five fails.

## Size

The residual stream is `[N, d_model]` and the FFN activations are `[N, d_ff]`,
so at the top of the grid a trace carries ~15k floats of high-entropy state.
`STATE_SIG_FIGS = 6` (L05's cut, for L05's reason: random mantissas do not
compress) is applied to `state` only — every assertion below runs on the
float64 arrays before the projection.

Run: python3 labs/traces/decoder_block.py
"""

import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

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
    """Unicode form, for prose. `fmt()` returns LaTeX, which would render as
    literal backslashes if it leaked into a narration string (lint checks)."""
    f = float(v)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "-∞" if f < 0 else "+∞"
    if f == 0:
        return "0"
    return "%.4g" % f


STATE_SIG_FIGS = 6


def state_val(v):
    """A number in its JSON-safe form: a float, or a sentinel string."""
    f = float(v)
    if math.isnan(f):
        return NAN
    if math.isinf(f):
        return NEG_INF if f < 0 else POS_INF
    return float("%.*g" % (STATE_SIG_FIGS, f))


def state_arr(a):
    """Nested-list form of an array of any rank, for `state` / `init`.

    The attention score matrices are `[H, N, N]` — one matrix per head — and a
    rank-2-only helper is how this file's first version managed to raise
    "only length-1 arrays can be converted to Python scalars" four frames away
    from the tensor declaration that had the wrong rank on it.
    """
    arr = np.asarray(a)
    if arr.ndim == 0:
        return state_val(arr)
    return [state_arr(sub) for sub in arr]


def state_mat(a):
    """Rank-2 alias, kept because a trace reader should be able to see at the
    call site which arrays are matrices and which are rank-3."""
    return state_arr(a)


def vec_tex(values, limit=10):
    """A short 1-D vector in a `num` tier.

    Only vectors go into formulas, never a matrix: a slot's \\phantom reserves
    the width of the value that will land in it, so an `[8, d_model]` matrix
    would reserve far more room than the line it sits on. The arrays are shown
    as grids in the view panels, which is both narrower and more to the point.
    """
    vs = list(values)
    if len(vs) > limit:
        return (r"\left[" + r",\; ".join(fmt(v) for v in vs[:limit]) +
                r",\; \dots\right]")
    return r"\left[" + r",\; ".join(fmt(v) for v in vs) + r"\right]"


def shape_str(*dims):
    return r"\times ".join(str(int(x)) for x in dims)


# ------------------------------------------------------------------- the model
#
# The shared example model, plus the two axes the L04 sliders sweep. `d_head`
# follows `d_model / H` rather than being fixed, so the head count is the thing
# held constant and the head width is the thing that moves with the model —
# which is what a reader changing `d_model` on a real model would be doing.
BASE = {
    "N": 8,
    "H": 4,
    "vocab": 32,
    "eps": 1e-5,
    "depth": 1,
}

GRID_D = (32, 64, 128)
GRID_DFF = (88, 176, 352)

DEFAULT_PAIR = (64, 176)
SET = "decoder-block"

WEIGHT_SEED = 20240613


def config_for(d_model, d_ff):
    cfg = dict(BASE)
    cfg["d_model"] = d_model
    cfg["d_head"] = d_model // cfg["H"]
    cfg["d_ff"] = d_ff
    return cfg


def cfg_id(d_model, d_ff):
    return f"{SET}-d{d_model}-dff{d_ff}"


def label(d_model, d_ff):
    return f"d_model={d_model} · d_ff={d_ff}"


def init_weights(cfg):
    """The block's parameters, at the tutorial's own initialization.

    `W_*` are drawn at `1/√d_model` (PyTorch's default for `nn.Linear`), which
    is what makes the residual-stream-growth measurement below meaningful rather
    than contrived: the growth is what that initialization produces, not what a
    large draw was chosen to produce. LayerNorm's γ/β are drawn non-trivially on
    purpose — the axis claim is stated about `(x−μ)/√(σ²+ε)` BEFORE this affine
    step (`step.ln` publishes both), so a learned γ/β costs the picture nothing
    and keeps the formula panel from being a formula with nothing in it.
    """
    d, dff = cfg["d_model"], cfg["d_ff"]
    rng = np.random.default_rng(WEIGHT_SEED + d * 131 + dff)
    s = 1.0 / math.sqrt(d)
    p = {
        "W_q": rng.standard_normal((d, d)) * s,
        "W_k": rng.standard_normal((d, d)) * s,
        "W_v": rng.standard_normal((d, d)) * s,
        "W_o": rng.standard_normal((d, d)) * s,
        "W_gate": rng.standard_normal((d, dff)) * s,
        "W_up": rng.standard_normal((d, dff)) * s,
        "W_down": rng.standard_normal((dff, d)) * s,
        "ln1_g": 1.0 + rng.standard_normal(d) * 0.1,
        "ln1_b": rng.standard_normal(d) * 0.05,
        "ln2_g": 1.0 + rng.standard_normal(d) * 0.1,
        "ln2_b": rng.standard_normal(d) * 0.05,
    }
    return p


def input_ids(cfg):
    """A deterministic input, fixed per configuration.

    Two decimal-free random draws from a fixed seed, so two runs of this script
    produce the same trace. The scale is 1.0 (not the 0.1 a "nice" demo would
    use) because a LayerNorm whose input already has unit variance is a
    LayerNorm that is hard to tell apart from the identity.
    """
    rng = np.random.default_rng(WEIGHT_SEED + cfg["d_model"] * 7 + cfg["N"])
    return rng.standard_normal((cfg["N"], cfg["d_model"]))


def analytic_param_count(cfg):
    """`3.7 §6.1`'s closed form for one block, split into the three segments.

        P_attn = 4·d²      (W_Q, W_K, W_V, W_O)
        P_ffn  = 3·d·d_ff  (W_gate, W_up, W_down)
        P_norm = 2·d       (γ and β, twice)
    """
    d, dff = cfg["d_model"], cfg["d_ff"]
    return {"attn": 4 * d * d, "ffn": 3 * d * dff, "norm": 2 * d * 2}


SEGMENTS = [
    {"id": "attn", "label": "注意力（4 个 d×d 投影）", "color": "#5b8def",
     "formula": r"4\,d^2"},
    {"id": "ffn", "label": "SwiGLU FFN（3 个 d×d_{ff} 矩阵）", "color": "#b45309",
     "formula": r"3\,d\,d_{ff}"},
    {"id": "norm", "label": "LayerNorm（两组 γ、β）", "color": "#0f7b52",
     "formula": r"2\times 2\,d"},
]


# ------------------------------------------------------------------- the math
def layernorm(x, g, b, eps):
    """LayerNorm, with the reduction axis written out rather than left to a
    library default: `axis=-1` over the last dimension, i.e. over `d_model`,
    one mean and one variance PER TOKEN. `step.ln` publishes the same two
    numbers computed this way, and the counterfactual of computing them over
    axis 0 instead."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normalized = (x - mu) / np.sqrt(var + eps)
    return g * normalized + b, normalized, mu, var


def layernorm_over_tokens(x, g, b, eps):
    """The counterfactual `step.ln` publishes: the same formula with the
    reduction one axis over. Used for nothing but the control — if this one
    also came out with rows at mean 0 / std 1, the primary statistic would not
    be evidence about which axis is normalized."""
    mu = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b


def softmax_rows(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def causal_mask(n):
    """`+0` on and below the diagonal, `-∞` above it. The sentinel vocabulary in
    the trace contract exists for exactly this array."""
    return np.where(np.tril(np.ones((n, n))) > 0, 0.0, -np.inf)


def attention_forward(x, p, cfg, traced=None):
    """Masked multi-head self-attention over the whole block's input.

    `traced`, when given, is a dict this fills with the intermediates so the
    trace can publish what the real arrays held rather than a re-derivation.
    """
    n, d, H = cfg["N"], cfg["d_model"], cfg["H"]
    dh = cfg["d_head"]
    q = x @ p["W_q"]
    k = x @ p["W_k"]
    v = x @ p["W_v"]
    qh = q.reshape(n, H, dh).transpose(1, 0, 2)
    kh = k.reshape(n, H, dh).transpose(1, 0, 2)
    vh = v.reshape(n, H, dh).transpose(1, 0, 2)
    mask = causal_mask(n)
    scores = np.einsum("hid,hjd->hij", qh, kh) / math.sqrt(dh) + mask
    probs = softmax_rows(scores)
    ctx = np.einsum("hij,hjd->hid", probs, vh).transpose(1, 0, 2).reshape(n, d)
    out = ctx @ p["W_o"]
    if traced is not None:
        traced.update({"q": q, "k": k, "v": v, "mask": mask, "scores": scores,
                       "probs": probs, "ctx": ctx, "out": out})
    return out


def ffn_forward(x, p, cfg, traced=None):
    """`SwiGLU(x) = W_down(SiLU(W_gate x) ⊙ W_up x)` — `3.7 §4`, transcribed.

    The three paths are kept as separate arrays all the way through, because
    the picture this lab draws is the three of them and a fused `F.silu` call
    would leave nothing to draw.
    """
    gate = x @ p["W_gate"]
    up = x @ p["W_up"]
    activated = silu(gate)
    gated = activated * up
    out = gated @ p["W_down"]
    if traced is not None:
        traced.update({"gate": gate, "up": up, "silu": activated,
                       "gated": gated, "out": out})
    return out


def block_forward(x, p, cfg, traced=None):
    """One Pre-Norm block, `y = x + F(LN(x))` twice.

    The two placements the lab contrasts are the two ways of writing this
    function; see `norm_contrast()` for the other one.

    The recorder is tested with `is not None`, not with truthiness: the caller
    hands in an EMPTY dict, an empty dict is falsy, and the truthiness version
    therefore recorded nothing at all while looking correct. (Its symptom was a
    `KeyError: 'mha'` several frames away, in the trace builder.)
    """
    rec = {} if traced is not None else None
    h1, norm1, mu1, var1 = layernorm(x, p["ln1_g"], p["ln1_b"], cfg["eps"])
    a = attention_forward(h1, p, cfg, None if rec is None else rec.setdefault("mha", {}))
    y1 = x + a
    h2, norm2, mu2, var2 = layernorm(y1, p["ln2_g"], p["ln2_b"], cfg["eps"])
    f = ffn_forward(h2, p, cfg, None if rec is None else rec.setdefault("ffn", {}))
    y2 = y1 + f
    if rec is not None:
        rec.update({"norm1_out": norm1, "mu1": mu1, "var1": var1,
                    "norm2_out": norm2, "mu2": mu2, "var2": var2,
                    "attn_out": a, "y1": y1, "ffn_out": f, "y2": y2})
        traced.update(rec)
    return y2, traced


# ------------------------------------------------------- torch reference block
_REL_TOL = 1e-11


def torch_block_forward(x, p, cfg):
    """The same block, in torch, written from the tutorial's own code shape.

    `3.7 §4` prints the block as `h = x + attn(norm1(x))` /
    `out = h + ffn(norm2(h))` with `F.silu` and `F.softmax`; this is that, with
    the parameters and the input handed over as tensors. It exists to be an
    independent second implementation, so it does NOT reuse `layernorm()`,
    `silu()` or `softmax_rows()` above.
    """
    import torch
    import torch.nn.functional as F

    n, d, H, dh = cfg["N"], cfg["d_model"], cfg["H"], cfg["d_head"]
    T = {k: torch.tensor(np.asarray(v), dtype=torch.float64) for k, v in p.items()}
    xt = torch.tensor(np.asarray(x), dtype=torch.float64)

    def ln(v, g, b):
        mu = v.mean(dim=-1, keepdim=True)
        var = v.var(dim=-1, unbiased=False, keepdim=True)
        return g * (v - mu) / torch.sqrt(var + cfg["eps"]) + b

    def attn(v):
        q = v @ T["W_q"]
        k = v @ T["W_k"]
        w = v @ T["W_v"]
        qh = q.reshape(n, H, dh).transpose(0, 1)
        kh = k.reshape(n, H, dh).transpose(0, 1)
        wh = w.reshape(n, H, dh).transpose(0, 1)
        s = (qh @ kh.transpose(-1, -2)) / math.sqrt(dh)
        s = s + torch.triu(torch.full((n, n), float("-inf"), dtype=torch.float64), 1)
        pr = F.softmax(s, dim=-1)
        c = (pr @ wh).transpose(0, 1).reshape(n, d)
        return c @ T["W_o"]

    def ffn(v):
        return (F.silu(v @ T["W_gate"]) * (v @ T["W_up"])) @ T["W_down"]

    y1 = xt + attn(ln(xt, T["ln1_g"], T["ln1_b"]))
    y2 = y1 + ffn(ln(y1, T["ln2_g"], T["ln2_b"]))
    return y2.numpy()


def max_rel_dev(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    scale = np.maximum(1.0, np.maximum(np.abs(got), np.abs(want)))
    return float(np.max(np.abs(got - want) / scale)) if got.size else 0.0


# =================================================== step.residual: the shapes
#
# The candidate right-hand sides, and why each one is here. The first is the
# step's own operand; the next three are the ones that SILENTLY change the
# operation; the last three are the ones that raise. Every outcome is recorded
# by running it.
RESIDUAL_CANDIDATES = [
    ("same", "相同 shape（本步真正的残差）", "both", lambda cfg: (cfg["N"], cfg["d_model"])),
    ("bcast-feature", "每个特征一个偏置 [d_model]", "broadcast", lambda cfg: (cfg["d_model"],)),
    ("bcast-row", "整行一个标量 [1, d_model]", "broadcast", lambda cfg: (1, cfg["d_model"])),
    ("bcast-col", "每个 token 一个标量 [N, 1]", "broadcast", lambda cfg: (cfg["N"], 1)),
    ("dims-half", "特征维减半 [N, d_model/2]", "error",
     lambda cfg: (cfg["N"], cfg["d_model"] // 2)),
    ("rows-twice", "token 数翻倍 [2N, d_model]", "error",
     lambda cfg: (2 * cfg["N"], cfg["d_model"])),
    ("rank-3", "多一个尾维 [N, d_model, 1]", "error",
     lambda cfg: (cfg["N"], cfg["d_model"], 1)),
]


def residual_outcome(lhs, rhs_shape, rng):
    """Run `lhs + rhs` for real and report what happened.

    Not a shape-compatibility table transcribed from the broadcasting rules:
    numpy is asked, and the exception's own text is what gets published. The
    torch column exists because "numpy broadcasts" is a claim about one library
    until the other one is consulted — and this lab's whole point is that the
    silent case is silent in both.

    The outcome is classified by the OPERAND shapes, not by the result shape.
    That distinction is the whole content of this block: `[d]` added to `[N,d]`
    produces `[N,d]`, so "the result changed shape" would call a broadcast a
    plain add — which is precisely the silent case the page exists to show.
    (That bug was in this function's first version: all three broadcast cases
    were reported as `same`, and the lint — which requires at least one of each
    outcome — refused to write the trace.)
    """
    rhs_shape = tuple(int(s) for s in rhs_shape)
    rhs = rng.standard_normal(rhs_shape)
    identical = rhs_shape == tuple(lhs.shape)
    out = {"rhs_shape": list(rhs_shape)}
    try:
        res = lhs + rhs
        out["outcome"] = "same" if identical else "broadcast"
        out["result_shape"] = [int(s) for s in res.shape]
        out["sample"] = state_mat(res[: min(2, res.shape[0])])
    except ValueError as exc:
        out["outcome"] = "error"
        out["message"] = str(exc)
        out["result_shape"] = None
    try:
        import torch
        tl = torch.tensor(np.asarray(lhs), dtype=torch.float64)
        tr = torch.tensor(np.asarray(rhs), dtype=torch.float64)
        try:
            torch_shape = tuple((tl + tr).shape)
        except RuntimeError:
            out["torch"] = "error"
        else:
            out["torch"] = "same" if torch_shape == tuple(lhs.shape) else "broadcast"
    except Exception:                                       # pragma: no cover
        out["torch"] = "unavailable"
    return out


def residual_block(cfg, lhs, rhs):
    """`step.residual`: the confirmation for the real operands, and the eight
    measured counterexamples."""
    rng = np.random.default_rng(WEIGHT_SEED + 907 + cfg["d_model"])
    # Independent re-derivation of the real add: an explicit double loop, so
    # "numpy adds elementwise" is a measurement and not the expression that is
    # being displayed.
    manual = [[lhs[i][j] + rhs[i][j] for j in range(lhs.shape[1])]
              for i in range(lhs.shape[0])]
    loop_dev = max_rel_dev(np.asarray(manual), lhs + rhs)

    cases = []
    for name, desc, expect, shapefn in RESIDUAL_CANDIDATES:
        o = dict(residual_outcome(lhs, shapefn(cfg), rng), id=name, label=desc,
                 expect=expect)
        cases.append(o)
    identical = [c for c in cases if c["outcome"] == "same"]
    return {
        "lhs_shape": [int(s) for s in lhs.shape],
        "rhs_shape": [int(s) for s in rhs.shape],
        "result_shape": [int(s) for s in (lhs + rhs).shape],
        "equal": list(lhs.shape) == list(rhs.shape),
        "element_loop_max_rel_dev": loop_dev,
        "cases": cases,
        "silent": [c["id"] for c in cases if c["outcome"] == "broadcast"],
        "raised": [c["id"] for c in cases if c["outcome"] == "error"],
        "identical_count": len(identical),
    }


# ======================================================== step.ln: the axis
def ln_axis_block(cfg, x, normalized, mu, var):
    """`step.ln`: which axis the statistics are taken over, made measurable.

    Three statistic pairs, all computed on real arrays:

      `row`           the normalized result's per-token statistics. This is
                      what LayerNorm over `d_model` produces.
      `column`        the same array's per-feature statistics. Under the real
                      normalization these are NOT 0/1, and that asymmetry is
                      the evidence about the axis.
      `counterfactual`  the same formula with the reduction one axis over
                      (`layernorm_over_tokens`). Its column statistics come out
                      at 0/1 and its rows do not — the flip. Without this, a
                      picture whose rows read 0/1 is equally consistent with one
                      that normalized nothing.
    """
    def stats(a, axis):
        m = a.mean(axis=axis)
        s = a.std(axis=axis)
        return {"mean_max_abs": float(np.max(np.abs(m))),
                "std_min": float(np.min(s)), "std_max": float(np.max(s))}

    over_tokens = layernorm_over_tokens(x, np.ones(cfg["d_model"]),
                                        np.zeros(cfg["d_model"]), cfg["eps"])
    return {
        "axis": "d_model",
        "axis_index": -1,
        "tokens": int(cfg["N"]),
        "d_model": int(cfg["d_model"]),
        "eps": cfg["eps"],
        "mu": [state_val(v) for v in mu.ravel()],
        "sigma": [state_val(v) for v in np.sqrt(var).ravel()],
        "mu_spread": float(np.max(mu) - np.min(mu)),
        "sigma_spread": float(np.max(np.sqrt(var)) - np.min(np.sqrt(var))),
        "row": stats(normalized, 1),
        "column": stats(normalized, 0),
        "counterfactual_row": stats(over_tokens, 1),
        "counterfactual_column": stats(over_tokens, 0),
    }


# ==================================================== step.swiglu: three paths
SWIGLU_PATHS = [
    {"id": "gate", "label": "gate 路径", "expr": r"\mathrm{SiLU}(xW_{gate})",
     "color": "#5b8def"},
    {"id": "up", "label": "up 路径", "expr": r"xW_{up}", "color": "#b45309"},
    {"id": "down", "label": "down 路径", "expr": r"W_{down}\,(\,\cdot\,)",
     "color": "#0f7b52"},
]


def swiglu_path_block(cfg, p, which, traced):
    """`step.swiglu`: all three weight shapes, all three intermediate shapes,
    and which path this step is on — so the picture can label every hop."""
    d, dff = cfg["d_model"], cfg["d_ff"]
    gate, up, act, gated = (traced["gate"], traced["up"], traced["silu"],
                            traced["gated"])
    gi = (int(round(0.55 * (cfg["N"] - 1))), int(round(0.61 * (dff - 1))))
    uj = (int(round(0.30 * (cfg["N"] - 1))), int(round(0.17 * (dff - 1))))
    return {
        "path": which,
        "paths": SWIGLU_PATHS,
        "d_model": int(d),
        "d_ff": int(dff),
        "shapes": {"x": [int(cfg["N"]), int(d)],
                   "W_gate": [int(dff), int(d)], "W_up": [int(dff), int(d)],
                   "W_down": [int(d), int(dff)],
                   "gate": [int(cfg["N"]), int(dff)],
                   "up": [int(cfg["N"]), int(dff)],
                   "silu": [int(cfg["N"]), int(dff)],
                   "gated": [int(cfg["N"]), int(dff)],
                   "out": [int(cfg["N"]), int(d)]},
        "elements": {"gate": int(gate.size), "up": int(up.size),
                     "silu": int(act.size), "gated": int(gated.size)},
        # Full float64 precision, NOT `state_val`'s six significant figures.
        # These three numbers are the evidence for the ⊙ — the rule re-derives
        # `product == a × b` from them — so rounding them would make the check
        # fire on its own display precision. (Six figures is enough to read and
        # not enough to multiply.)
        "multiply": {
            "row": gi[0], "col": gi[1],
            "a": float(act[gi]), "b": float(up[gi]),
            "product": float(gated[gi]),
            "product_matches": bool(float(gated[gi]) == float(act[gi]) * float(up[gi])),
            "rows_checked": int(gated.size),
        },
        "sample": {"up_col": uj[1], "row": uj[0],
                   "gate": state_val(act[uj]), "up": state_val(up[uj]),
                   "gated": state_val(gated[uj])},
    }


# ================================================= meta.norm_contrast: Pre/Post
DEPTHS = (1, 2, 4, 8, 16, 32)
PROBE_SEEDS = (20240614, 20240615, 20240616)
FLOW_PRE = ["x", "LN", "SubLayer", "Add"]
FLOW_POST = ["x", "SubLayer", "Add", "LN"]


def _init_stack(d, dff, depth, seed):
    """A stack of `depth` blocks, at the tutorial's `nn.Linear` init, with
    LayerNorm at γ=1 / β=0 — the state the tutorial's own `RMSNorm` module
    starts in, and the state the theory in `3.6 §4.3` is stated about
    ("各层输出的方差会随深度线性增长" is an initialization-time claim).

    Returned as `{block_index: {name: array}}` rather than a flat name-keyed
    dict, because a flat dict is how this function's first version handed a
    depth-8 stack the weights of block 17: `"d17".endswith("7")` is true, and a
    string suffix is not an index.

    Shapes follow `nn.Linear`'s convention — `(out_features, in_features)` — so
    every hop below is a `t @ W.T`, which is what the tutorial's own module
    does. Getting two of them the other way round is how the first version of
    this function produced `(8x32) and (88x32) cannot be multiplied`.
    """
    rng = np.random.default_rng(seed)
    s = 1.0 / math.sqrt(d)
    return {b: {
        "q": rng.standard_normal((d, d)) * s,
        "k": rng.standard_normal((d, d)) * s,
        "v": rng.standard_normal((d, d)) * s,
        "o": rng.standard_normal((d, d)) * s,
        "g": rng.standard_normal((dff, d)) * s,
        "u": rng.standard_normal((dff, d)) * s,
        "d": rng.standard_normal((d, dff)) * s,
    } for b in range(depth)}


def norm_contrast(cfg, probe_seed=PROBE_SEEDS[0], stack=None):
    """The Pre-Norm / Post-Norm contrast, measured by torch autograd.

    Two series per placement, over depths 1…32:

      `residual_rms[k]`   the RMS of the residual stream after block k. This is
                          a forward quantity, and Post-Norm holding at 1.0000 is
                          not a coincidence to be admired — it is what a
                          LayerNorm output IS, which is exactly why it is the
                          control that makes the other curve readable.
      `grad_norm[k]`      ‖∂L/∂(residual entering block k)‖ for `L = ⟨y_D, v⟩`
                          with `v` a fixed random direction. A single direction
                          is a sample, not the expected norm; the caller re-runs
                          this on other seeds and requires the ordering to
                          survive, which is what keeps the headline from being
                          an artefact of one draw.

    The stack of weights is drawn ONCE per probe seed and shared by every depth,
    so depth 8's first eight blocks are the same blocks depth 32's first eight
    are. Drawing them per depth would make the two curves differ partly because
    they were looking at different models.
    """
    import torch

    n, d, H, dff = cfg["N"], cfg["d_model"], cfg["H"], cfg["d_ff"]
    dh = d // H
    dt = torch.float64
    out = {}
    if stack is None:
        stack = _init_stack(d, dff, max(DEPTHS), WEIGHT_SEED + probe_seed)
    probe_rng = np.random.default_rng(probe_seed)
    v = torch.tensor(probe_rng.standard_normal((n, d)), dtype=dt)
    # The INPUT is the first residual-stream boundary, so it has to be a leaf
    # that wants a gradient: `retain_grad()` on a non-requiring tensor raises,
    # and the boundary at block 0 is the one the whole comparison is anchored
    # on (it is the "gradient reaching the earliest layer" the tutorial's §4.2
    # formulas are about).
    # that wants a gradient: `retain_grad()` on a non-requiring tensor raises,
    # and the boundary at block 0 is the one the whole comparison is anchored
    # on (it is the "gradient reaching the earliest layer" the tutorial's §4.2
    # formulas are about).
    x0 = torch.tensor(input_ids(cfg), dtype=dt).requires_grad_(True)

    for mode in ("pre", "post"):
        rms_series, grad_series, display = [], [], []
        for depth in DEPTHS:
            W = {f"{name}{b}": torch.tensor(stack[b][name], dtype=dt).requires_grad_(True)
                 for b in range(depth) for name in stack[b]}

            def ln(t, eps=cfg["eps"]):
                mu = t.mean(dim=-1, keepdim=True)
                var = t.var(dim=-1, unbiased=False, keepdim=True)
                return (t - mu) / torch.sqrt(var + eps)

            def sub(t, b):
                q = t @ W[f"q{b}"].T
                k = t @ W[f"k{b}"].T
                w = t @ W[f"v{b}"].T
                qh = q.reshape(n, H, dh).transpose(0, 1)
                kh = k.reshape(n, H, dh).transpose(0, 1)
                wh = w.reshape(n, H, dh).transpose(0, 1)
                s = qh @ kh.transpose(-1, -2) / math.sqrt(dh)
                s = s + torch.triu(torch.full((n, n), float("-inf"), dtype=dt), 1)
                c = (torch.softmax(s, -1) @ wh).transpose(0, 1).reshape(n, d)
                return c @ W[f"o{b}"].T

            def ffn(t, b):
                return (torch.nn.functional.silu(t @ W[f"g{b}"].T) *
                        (t @ W[f"u{b}"].T)) @ W[f"d{b}"].T

            cur = x0
            bounds = []
            for b in range(depth):
                cur.retain_grad()
                bounds.append(cur)
                if mode == "pre":
                    y1 = cur + sub(ln(cur), b)
                    y2 = y1 + ffn(ln(y1), b)
                else:
                    y1 = ln(cur + sub(cur, b))
                    y2 = ln(y1 + ffn(y1, b))
                cur = y2
            rms_series.append(float(cur.pow(2).mean().sqrt()))
            (cur * v).sum().backward()
            g = [float(bd.grad.norm()) for bd in bounds]
            grad_series.append(g)
            # The chart's y series: how much the gradient is amplified between
            # the outermost boundary and the innermost one, for a stack this
            # deep, in log10. Published rather than computed in the browser —
            # same reason the whole trace architecture exists — and logarithmic
            # because by depth 32 the two placements sit three orders of
            # magnitude apart, which a linear axis would draw as one flat line
            # lying on the x axis with the other invisible underneath it.
            display.append(math.log10(max(g[0], 1e-300) / max(g[-1], 1e-300)))

        out[mode] = {
            "id": mode,
            "label": "Pre-Norm" if mode == "pre" else "Post-Norm",
            "flow": list(FLOW_PRE if mode == "pre" else FLOW_POST),
            "formula": (r"y = x + F(\mathrm{LN}(x))" if mode == "pre"
                        else r"y = \mathrm{LN}(x + F(x))"),
            "residual_rms": rms_series,
            "grad_norm": grad_series,
            "grad_amp_log": display,
            # The DEEPEST stack, not the first one. `grad_series[0]` is the
            # depth-1 run, which has exactly one boundary, so its first and
            # last entry are the same number and the ratio is 1 by construction
            # — which is what the first version of this line reported for both
            # placements, making the one number the whole contrast rests on
            # carry no information.
            "grad_in_over_out": grad_series[-1][0] / grad_series[-1][-1],
            "rms_growth": rms_series[0] / rms_series[-1],
        }

    # Where the norm sits in the Post-Norm flow — 3 for the four-node graph in
    # `3.6 §4.1`. NOT the first index at which the two lists differ: moving one
    # element to the end shifts the two after it, so the first positional
    # difference is index 1 even though the norm itself lands at index 3.
    return {
        "depths": list(DEPTHS),
        "modes": [out["pre"], out["post"]],
        "difference": {"only": "LN", "at_index": FLOW_POST.index("LN"),
                       "note": "两条数据流的节点名完全相同，只有 LayerNorm 的位置不同 —— "
                               "这段对照是教程 3.6 §4.1 两段 mermaid 的差异本身。"},
        "probe": {
            "loss": r"L = \langle y_D,\, v \rangle",
            "direction": "固定随机方向 v（单方向，是采样不是期望）",
            "seed": probe_seed,
            "note": "梯度是回传到每个 block 边界时的范数，沿深度逐个记录。",
        },
        "order": {"grad": "post>pre"},
    }


def contrast_ordering_holds(cfg, seeds=PROBE_SEEDS):
    """Re-run the sweep on other probe seeds and require the ordering the page
    states. One direction is a sample; the claim is about the architecture, so
    it has to survive draws the published trace does not use."""
    for seed in seeds:
        c = norm_contrast(cfg, probe_seed=seed)
        pre, post = c["modes"][0], c["modes"][1]
        if not post["grad_in_over_out"] > pre["grad_in_over_out"]:
            return False, seed
    return True, None


# ------------------------------------------------------- the parameter account
DEPLOY_BYTES = 2          # fp16 — the tutorial's `3.7 §8` working dtype


def param_block(cfg, p):
    """`step.params` — the account the ledger component draws.

    Three quantities per segment, and the distinction between them is the whole
    point of this block:

      `bytes`        the PARAMETER COUNT counted from the real weight arrays
                     (`arr.size`). This is what the ticket's fifth criterion is
                     about and what the readout prints.
      `closed_form`  the same count from the tutorial's `3.7 §6.1` expression
                     `4d² + 3·d·d_ff + 4d`. Published beside `bytes` so the page
                     can show that the two agree rather than assert it.
      `priced`       `bytes × DEPLOY_BYTES` — what those weights COST, which is
                     the tutorial's `3.7 §8` formula (`显存 = 参数量 × 每参数字节数`)
                     and what the ledger component draws.

    Why the ledger gets `priced` and not `bytes`: `views/ledger.js` is a MEMORY
    ledger — its formatter is named `formatBytes`, it appends a `KiB`/`MiB`
    suffix above 1024, and every consumer prices something in bytes. Handing it
    a parameter count with a unit of "个参数" produces `50,432 个参数 · 49.3 KiB`,
    which is a real number wearing a meaningless unit — and the page would be
    printing it. Pricing the same account in bytes is both what the component is
    for and an extra teaching point (`3.7 §8` is about exactly this), so the two
    numbers are published separately and neither is dressed up as the other.
    """
    real = {}
    for key, seg in (("W_q", "attn"), ("W_k", "attn"), ("W_v", "attn"),
                     ("W_o", "attn")):
        real[seg] = real.get(seg, 0) + int(np.asarray(p[key]).size)
    for key in ("W_gate", "W_up", "W_down"):
        real["ffn"] = real.get("ffn", 0) + int(np.asarray(p[key]).size)
    for key in ("ln1_g", "ln1_b", "ln2_g", "ln2_b"):
        real["norm"] = real.get("norm", 0) + int(np.asarray(p[key]).size)
    closed = analytic_param_count(cfg)
    if real != closed:
        raise AssertionError(f"参数量：实算 {real} != 闭式 {closed}")
    order = ("attn", "ffn", "norm")
    return {
        "bytes": {k: real[k] for k in order},
        "closed_form": {k: closed[k] for k in order},
        "elements": sum(real.values()),
        "priced": {k: real[k] * DEPLOY_BYTES for k in order},
        "priced_total": sum(real.values()) * DEPLOY_BYTES,
        "deploy_bytes": DEPLOY_BYTES,
        "config": {"d_model": cfg["d_model"], "d_ff": cfg["d_ff"],
                   "H": cfg["H"], "d_head": cfg["d_head"]},
    }


# ---------------------------------------------------------------- trace build
def b(slot, sym, idx, num):
    return slot, {"sym": sym, "idx": idx, "num": num}


def step(sid, title, kind, op, phase, formula, bindings, reads, writes, state,
         narration, regions=None, extra=None):
    d = {"id": sid, "title": title, "kind": kind, "op": op, "phase": phase,
         "formula": formula, "bindings": dict(bindings),
         "reads": reads, "writes": writes, "state": state,
         "narration": narration}
    if regions:
        d["regions"] = regions
    if extra:
        d.update(extra)
    return d


def build_trace(cfg, p, x):
    d, H, dh, dff = cfg["d_model"], cfg["H"], cfg["d_head"], cfg["d_ff"]
    n = cfg["N"]
    traced = {}
    y2, traced = block_forward(x, p, cfg, traced)
    ref = torch_block_forward(x, p, cfg)
    dev = max_rel_dev(y2, ref)

    mha, ffn = traced["mha"], traced["ffn"]
    steps = []

    def add(*a, **kw):
        steps.append(step(*a, **kw))

    # ---------------------------------------------------------------- 0 · 输入
    add("x.in", "输入：残差流的入口", "data", "input", "block",
        {"sym": r"\region{RESIDUAL}{X} \in \mathbb{R}^{\slot{NT}\times \slot{DM}}",
         "idx": r"X_{[\slot{NT},\slot{DM}]}",
         "num": r"X_{[\slot{NT},\slot{DM}]}\ \text{—— 本步只声明输入，后面每一步都在它上面加}"},
        [b("NT", "N", str(n), str(n)),
         b("DM", r"d_{\mathrm{model}}", str(d), str(d))],
        [], ["x"],
        {"x": state_mat(x)},
        f"整个 block 的输入是一条 [N={n}, d_model={d}] 的残差流。后面每一步都在它上面"
        f"就地加东西，形状从头到尾不变 —— 这正是残差连接要求两侧 shape 完全相同的原因，"
        f"也是 d_model 一路不变的原因。",
        {"RESIDUAL": "残差流：形状不变的加法通道"})

    # ------------------------------------------------------------- 1 · LN 1
    add("x.ln1", "第一处 LayerNorm：在 d_model 维上、逐 token 归一化", "op", "layernorm",
        "ln1",
        {"sym": r"\region{AXIS}{\hat{X}} = \frac{X - \mu}{\sqrt{\sigma^2 + \varepsilon}},"
                r"\qquad \mathrm{LN}(X) = \gamma \odot \hat{X} + \beta",
         "idx": r"\mu^{(i)} = \frac{1}{\slot{DM}}\sum_{j=1}^{\slot{DM}} X_{i,j},"
                r"\qquad \sigma^{(i)2} = \frac{1}{\slot{DM}}\sum_{j=1}^{\slot{DM}}"
                r"\left(X_{i,j} - \mu^{(i)}\right)^2",
         "num": r"\mu = \slot{MU},\qquad \sigma = \slot{SIGMA}"},
        [b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("MU", r"\mu^{(1..N)}", r"\mu^{(1..N)}",
           vec_tex(traced["mu1"].ravel(), 4)),
         b("SIGMA", r"\sigma^{(1..N)}", r"\sigma^{(1..N)}",
           vec_tex(np.sqrt(traced["var1"]).ravel(), 4))],
        ["x"], ["h1"], {"h1": state_mat(traced["norm1_out"])},
        f"μ 和 σ 各算出 {n} 个 —— 每个 token 一对，在这个 token 自己的 {d} 个特征上求。"
        f"8 个 μ 互不相同（{disp(np.min(traced['mu1']))}…{disp(np.max(traced['mu1']))}），"
        f"σ 也是：把整个 [N, d_model] 求一个全局均值是错的，跨 token 求（轴选错）更错。"
        f"归一化之后每一行的均值≈0、标准差≈1，而每一列不是 —— 这个不对称就是"
        f"「沿哪个轴」的可判据，右侧面板把两个轴都量出来了。",
        {"AXIS": f"归约轴 = 最后一维（{d} 个特征），逐 token 独立；μ、σ 各有 {n} 个"},
        extra={"ln": ln_axis_block(cfg, x, traced["norm1_out"], traced["mu1"],
                                   traced["var1"])})

    # ---------------------------------------------------------- 2 · Q K V
    add("x.qkv", "Q / K / V 三个投影", "op", "project", "mha",
        {"sym": r"\region{HEADS}{Q = \mathrm{LN}(X)W_q},\quad "
                r"K = \mathrm{LN}(X)W_k,\quad V = \mathrm{LN}(X)W_v",
         "idx": r"W_q, W_k, W_v \in \mathbb{R}^{\slot{DM}\times \slot{DM}}",
         "num": r"Q,K,V \in \mathbb{R}^{\slot{NT}\times \slot{DM}},"
                r"\quad \text{每头 } d_h = \slot{DH}"},
        [b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("NT", "N", str(n), str(n)),
         b("DH", r"d_h", str(dh), str(dh))],
        ["h1"], ["q", "k", "v"],
        {"q": state_mat(mha["q"]), "k": state_mat(mha["k"]), "v": state_mat(mha["v"])},
        f"归一化后的 h1 分别乘三个 [{d},{d}] 的矩阵，得到三个 [{n},{d}] 的 Q、K、V。"
        f"三个输出形状完全相同 —— 它们只是角色不同，后面的 RoPE 只作用在 Q、K 上。",
        {"HEADS": f"H={H} 个头，每头 d_h={dh}；投影后再拆头"})

    # --------------------------------------------------- 3 · scores + mask
    add("x.scores", "S = QKᵀ/√d_h，加因果掩码", "op", "attention", "mha",
        {"sym": r"S = \frac{QK^{\top}}{\sqrt{d_h}} + \region{MASK}{M},\qquad "
                r"M_{ij} = 0\ (j \le i),\ -\infty\ (j > i)",
         "idx": r"S_{[H, \slot{NT}, \slot{NT}]} = "
                r"\frac{Q_{[H,N,d_h]}K^{\top}_{[H,d_h,N]}}{\sqrt{\slot{DH}}} + M",
         "num": r"S \in \mathbb{R}^{[\slot{H},\slot{NT},\slot{NT}]},\qquad "
                r"M_{1,:} = \slot{MROW}"},
        [b("NT", "N", str(n), str(n)),
         b("H", "H", str(H), str(H)),
         b("DH", r"d_h", str(dh), str(dh)),
         b("MROW", r"M_{1,:}", r"M_{1,:}",
           r"\left[0,\; -\infty,\; -\infty,\; \dots\right]")],
        ["q", "k"], ["scores", "mask"],
        {"scores": state_arr(mha["scores"]), "mask": state_mat(mha["mask"])},
        f"每个头各算一张 [{n},{n}] 的分数矩阵，共 {H} 张。掩码把上三角置为 −∞："
        f"softmax 之后 e^(−∞)=0，于是第 i 个 token 只能看到 0..i。"
        f"−∞ 在 trace 里用哨兵字符串表示，因为裸 Infinity 是非法 JSON。",
        {"MASK": "上三角 −∞，保证因果性"})

    # --------------------------------------------------- 4 · softmax → PV
    add("x.attn", "softmax → P·V → 合并头 → W_O", "op", "attention", "mha",
        {"sym": r"\region{MERGE}{\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_h}} "
                r"+ M\right)V}W_o",
         "idx": r"O_{[\slot{NT},\slot{DM}]} = \left(P_{[H,N,N]}V_{[H,N,d_h]}\right)"
                r"_{\text{合并头}} W_o",
         "num": r"\text{每行权重和} = 1,\qquad O \in \mathbb{R}^{[\slot{NT},\slot{DM}]}"},
        [b("NT", "N", str(n), str(n)),
         b("DM", r"d_{\mathrm{model}}", str(d), str(d))],
        ["scores", "v"], ["probs", "ctx", "attn_out"],
        {"probs": state_arr(mha["probs"]), "ctx": state_mat(mha["ctx"]),
         "attn_out": state_mat(mha["out"])},
        f"每行 softmax 之后是 {n} 个非负的权重，和为 1（被掩掉的位置权重是 0）。"
        f"加权求和后把 {H} 个头拼回 [{n},{d}]，再过一个 W_O。"
        f"注意输出又回到了 [{n},{d}] —— 只有这样它才能和残差流相加。",
        {"MERGE": f"H 个头 × d_h={dh} 拼回 d_model={d}"})

    # ------------------------------------------------------- 5 · 残差 1
    res1 = residual_block(cfg, x, mha["out"])
    add("x.res1", "残差相加：x + MHA(LN(x))", "op", "residual", "ln1",
        {"sym": r"Y_1 = \region{SAME}{\slot{LHS} + \slot{RHS}}",
         "idx": r"Y_1^{[\slot{NT},\slot{DM}]} = \region{SAME}{"
                r"X^{[\slot{NT},\slot{DM}]} + \mathrm{Attn}^{[\slot{NT},\slot{DM}]}}",
         "num": r"[\slot{NT},\slot{DM}] + [\slot{NT},\slot{DM}] = [\slot{NT},\slot{DM}]"
                r"\quad\text{—— 两侧逐元素相加}"},
        [b("NT", "N", str(n), str(n)),
         b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("LHS", r"X_{[N,d]}", r"X_{[%d,%d]}" % (n, d), shape_str(n, d)),
         b("RHS", r"\mathrm{Attn}_{[N,d]}", r"\mathrm{Attn}_{[%d,%d]}" % (n, d),
           shape_str(n, d))],
        ["x", "attn_out"], ["y1"], {"y1": state_mat(traced["y1"])},
        f"两侧形状完全相同（都是 [{n},{d}]），逐元素相加得到同样形状的 y1 —— "
        f"这就是 d_model 一路不变的原因。右侧面板把「不相同会怎样」也真跑了一遍："
        f"[{d}] 会<b>静默广播</b>成一次逐特征加法（结果形状正确、不报错，但它不是残差连接），"
        f"而 [{n},{d//2}] 会直接抛 ValueError。",
        {"SAME": f"两侧都必须是 [{n},{d}]：形状是相等判断，不是广播"},
        extra={"residual": res1})

    # ------------------------------------------------------------- 6 · LN 2
    add("x.ln2", "第二处 LayerNorm：同样在 d_model 维上逐 token 做", "op", "layernorm",
        "ln2",
        {"sym": r"\region{PREENORM}{\hat{Y}_1} = "
                r"\frac{Y_1 - \mu}{\sqrt{\sigma^2 + \varepsilon}},\qquad "
                r"\mathrm{LN}(Y_1) = \gamma_2 \odot \hat{Y}_1 + \beta_2",
         "idx": r"\mu^{(i)} = \frac{1}{\slot{DM}}\sum_{j=1}^{\slot{DM}} (Y_1)_{i,j},"
                r"\qquad \sigma^{(i)2} = \frac{1}{\slot{DM}}\sum_{j=1}^{\slot{DM}}"
                r"\left((Y_1)_{i,j} - \mu^{(i)}\right)^2",
         "num": r"\mu = \slot{MU},\qquad \sigma = \slot{SIGMA}"},
        [b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("MU", r"\mu^{(1..N)}", r"\mu^{(1..N)}",
           vec_tex(traced["mu2"].ravel(), 4)),
         b("SIGMA", r"\sigma^{(1..N)}", r"\sigma^{(1..N)}",
           vec_tex(np.sqrt(traced["var2"]).ravel(), 4))],
        ["y1"], ["h2"], {"h2": state_mat(traced["norm2_out"])},
        f"第二个子层前再归一化一次，轴和第一处完全一样。注意这里归一化的对象是"
        f"<b>已经加上残差之后</b>的 y1 —— Pre-Norm 的残差流本身是干净的，"
        f"LN 挂在子层的入口而不是残差路径上。这个顺序就是 L04 对照实验的全部内容。",
        {"PREENORM": "Pre-Norm：LN 在子层入口，残差路径上没有归一化"},
        extra={"ln": ln_axis_block(cfg, traced["y1"], traced["norm2_out"],
                                   traced["mu2"], traced["var2"])})

    # ------------------------------------------------------- 7 · gate 路径
    sw = swiglu_path_block(cfg, p, "gate", ffn)
    add("x.gate", "SwiGLU · gate 路径：W_gate 投影", "op", "linear", "ffn",
        {"sym": r"\region{GATE}{G} = \mathrm{LN}(Y_1)\,W_{gate}",
         "idx": r"G_{[\slot{NT},\slot{DFF}]} = \mathrm{LN}(Y_1)_{[\slot{NT},\slot{DM}]}"
                r"W_{gate}^{[\slot{DM},\slot{DFF}]}",
         "num": r"[\slot{NT},\slot{DM}] \times [\slot{DM},\slot{DFF}] = "
                r"[\slot{NT},\slot{DFF}]"},
        [b("NT", "N", str(n), str(n)),
         b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("DFF", r"d_{ff}", str(dff), str(dff))],
        ["h2"], ["gate"], {"gate": state_mat(ffn["gate"])},
        f"SwiGLU 的第一条路径：先升维到 [{n},{dff}]。三条路径的分工在这里开始 —— "
        f"gate 这条会过一个 SiLU，up 那条不会，最后两者逐元素相乘。",
        {"GATE": f"W_gate 的形状是 [{dff},{d}]，升维到 d_ff={dff}"},
        extra={"swiglu": sw})

    # --------------------------------------------------------- 8 · up 路径
    sw = swiglu_path_block(cfg, p, "up", ffn)
    add("x.up", "SwiGLU · up 路径：W_up 投影（不过激活）", "op", "linear", "ffn",
        {"sym": r"\region{UP}{U} = \mathrm{LN}(Y_1)\,W_{up}",
         "idx": r"U_{[\slot{NT},\slot{DFF}]} = \mathrm{LN}(Y_1)_{[\slot{NT},\slot{DM}]}"
                r"W_{up}^{[\slot{DM},\slot{DFF}]}",
         "num": r"[\slot{NT},\slot{DM}] \times [\slot{DM},\slot{DFF}] = "
                r"[\slot{NT},\slot{DFF}]"},
        [b("NT", "N", str(n), str(n)),
         b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("DFF", r"d_{ff}", str(dff), str(dff))],
        ["h2"], ["up"], {"up": state_mat(ffn["up"])},
        f"第二条路径，形状和 gate 一模一样（都是 [{n},{dff}]），但没有激活函数 —— "
        f"它是被门控的那一路。两条路径形状相同不是巧合：它们接下来要逐元素相乘。",
        {"UP": f"与 gate 同形 [{dff}]，是门控的对象"},
        extra={"swiglu": sw})

    # ------------------------------------------------------- 9 · SiLU
    sw = swiglu_path_block(cfg, p, "gate", ffn)
    add("x.silu", "gate 路径过 SiLU（Swish）", "op", "activation", "ffn",
        {"sym": r"\region{SILU}{\mathrm{SiLU}(g)} = g \odot \sigma(g) = "
                r"\frac{g}{1 + e^{-g}}",
         "idx": r"\mathrm{SiLU}(G)_{[\slot{NT},\slot{DFF}]},\qquad "
                r"G \in \mathbb{R}^{[\slot{NT},\slot{DFF}]}",
         "num": r"\mathrm{SiLU}(-1) = -0.2689,\quad \mathrm{SiLU}(0) = 0,\quad "
                r"\mathrm{SiLU}(1) = 0.7311\ \text{（形状不变）}"},
        [b("NT", "N", str(n), str(n)),
         b("DFF", r"d_{ff}", str(dff), str(dff))],
        ["gate"], ["silu"], {"silu": state_mat(ffn["silu"])},
        f"SiLU 是逐元素的，形状不变。它给 gate 这条路径加上了非线性 —— "
        f"没有它，两条线性路径逐元素相乘仍然只是一个双线性形式，"
        f"整个 FFN 会退化。",
        {"SILU": "逐元素，形状不变；这是门控里的非线性来源"},
        extra={"swiglu": sw})

    # ------------------------------------------------- 10 · 逐元素乘
    sw = swiglu_path_block(cfg, p, "gate", ffn)
    mul = sw["multiply"]
    add("x.gated", "逐元素相乘 gate ⊙ up（门控）", "op", "mul", "ffn",
        {"sym": r"S = \region{ELEMENTWISE}{\mathrm{SiLU}(G) \odot U}",
         "idx": r"S^{[\slot{NT},\slot{DFF}]}_{i,j} = "
                r"\mathrm{SiLU}(G)^{[\slot{NT},\slot{DFF}]}_{i,j} \cdot "
                r"U^{[\slot{NT},\slot{DFF}]}_{i,j}",
         "num": r"S_{\slot{RI},\slot{CJ}} = \slot{A} \times \slot{BB} = \slot{P}"
                r"\quad\text{（共 } \slot{NELEM} \text{ 个元素，逐位置相乘）}"},
        [b("NT", "N", str(n), str(n)),
         b("DFF", r"d_{ff}", str(dff), str(dff)),
         b("RI", "i", str(mul["row"]), str(mul["row"])),
         b("CJ", "j", str(mul["col"]), str(mul["col"])),
         b("A", r"\mathrm{SiLU}(G)", r"\mathrm{SiLU}(G)_{%d,%d}" % (mul["row"], mul["col"]),
           fmt(mul["a"])),
         b("BB", "U", r"U_{%d,%d}" % (mul["row"], mul["col"]), fmt(mul["b"])),
         b("P", "S", r"S_{%d,%d}" % (mul["row"], mul["col"]), fmt(mul["product"])),
         b("NELEM", r"N d_{ff}", f"{n}\\times{dff}", str(n * dff))],
        ["silu", "up"], ["gated"], {"gated": state_mat(ffn["gated"])},
        f"门控就是这一步：两条 [{n},{dff}] 的路径按位置相乘，形状不变。"
        f"公式面板里代的是矩阵里真实的一个位置 —— "
        f"第 ({mul['row']},{mul['col']}) 个元素：{disp(mul['a'])} × {disp(mul['b'])} = "
        f"{disp(mul['product'])}，整个矩阵的 {n*dff} 个位置都是这么来的。",
        {"ELEMENTWISE": f"⊙ 是逐元素乘，不是矩阵乘：两侧 shape 必须相同 [{n},{dff}]"},
        extra={"swiglu": sw})

    # ------------------------------------------------------- 11 · down 路径
    sw = swiglu_path_block(cfg, p, "down", ffn)
    add("x.down", "SwiGLU · down 路径：W_down 降维回 d_model", "op", "linear", "ffn",
        {"sym": r"\region{DOWN}{F} = S\,W_{down}",
         "idx": r"F_{[\slot{NT},\slot{DM}]} = S_{[\slot{NT},\slot{DFF}]}"
                r"W_{down}^{[\slot{DFF},\slot{DM}]}",
         "num": r"[\slot{NT},\slot{DFF}] \times [\slot{DFF},\slot{DM}] = "
                r"[\slot{NT},\slot{DM}]"},
        [b("NT", "N", str(n), str(n)),
         b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("DFF", r"d_{ff}", str(dff), str(dff))],
        ["gated"], ["ffn_out"], {"ffn_out": state_mat(ffn["out"])},
        f"第三条路径把 [{n},{dff}] 降回 [{n},{d}] —— 必须降回 d_model，"
        f"否则下一步的残差加法两侧形状不同，整个 block 就不成立了。"
        f"SwiGLU 用三个矩阵而标准 FFN 用两个，所以 d_ff 取 (8/3)·d_model 才能让参数量持平。",
        {"DOWN": f"W_down 的形状是 [{d},{dff}]，把 d_ff 降回 d_model"},
        extra={"swiglu": sw})

    # ------------------------------------------------------- 12 · 残差 2
    res2 = residual_block(cfg, traced["y1"], ffn["out"])
    add("x.res2", "残差相加：y1 + SwiGLU(LN(y1))", "op", "residual", "ln2",
        {"sym": r"Y_2 = \region{SAME}{\slot{LHS} + \slot{RHS}}",
         "idx": r"Y_2^{[\slot{NT},\slot{DM}]} = \region{SAME}{"
                r"Y_1^{[\slot{NT},\slot{DM}]} + \mathrm{FFN}^{[\slot{NT},\slot{DM}]}}",
         "num": r"[\slot{NT},\slot{DM}] + [\slot{NT},\slot{DM}] = [\slot{NT},\slot{DM}]"
                r"\quad\text{—— 同样要求两侧完全相同}"},
        [b("NT", "N", str(n), str(n)),
         b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("LHS", r"Y_1{}_{[N,d]}", r"Y_1{}_{[%d,%d]}" % (n, d), shape_str(n, d)),
         b("RHS", r"\mathrm{FFN}_{[N,d]}", r"\mathrm{FFN}_{[%d,%d]}" % (n, d),
           shape_str(n, d))],
        ["y1", "ffn_out"], ["y2"], {"y2": state_mat(traced["y2"])},
        f"第二个残差加，约束和第一个一样：两侧必须都是 [{n},{d}]。"
        f"整个 block 到此结束，输入和输出形状完全相同 —— 所以才能堆叠任意多层。",
        {"SAME": "形状相同才能堆叠；这是「d_model 一路不变」的第二处确认"},
        extra={"residual": res2})

    # ------------------------------------------------------- 13 · 汇总
    par = param_block(cfg, p)
    add("x.out", "本 block 的参数量：按 d_model / d_ff 重算", "data", "summary",
        "block",
        {"sym": r"\region{PARAMS}{P_{\mathrm{block}}} = "
                r"\underbrace{4d^2}_{\text{Attention}} + "
                r"\underbrace{3\,d\,d_{ff}}_{\text{SwiGLU}} + "
                r"\underbrace{2\times 2d}_{\text{LayerNorm}}",
         "idx": r"4\times\slot{DM}^2 + 3\times\slot{DM}\times\slot{DFF} + "
                r"2\times 2\times\slot{DM}",
         "num": r"\slot{PA} + \slot{PF} + \slot{PN} = \slot{PT}\ \text{个参数}"},
        [b("DM", r"d_{\mathrm{model}}", str(d), str(d)),
         b("DFF", r"d_{ff}", str(dff), str(dff)),
         b("PA", r"4d^2", f"4\\times{d}^2", f"{par['closed_form']['attn']:,}".replace(",", r"\,")),
         b("PF", r"3dd_{ff}", f"3\\times{d}\\times{dff}", f"{par['closed_form']['ffn']:,}".replace(",", r"\,")),
         b("PN", r"4d", f"2\\times 2\\times{d}", f"{par['closed_form']['norm']:,}".replace(",", r"\,")),
         b("PT", r"P_{\mathrm{block}}", r"P_{\mathrm{block}}",
           f"{par['elements']:,}".replace(",", r"\,"))],
        [], [], {},
        f"整个 block 的参数量就是这三个分项之和。滑动 d_model 会让注意力分项按平方走、"
        f"LayerNorm 分项按线性走；滑动 d_ff 只动 SwiGLU 那一项。"
        f"下面的账本组件把三项堆在一起，滑杆一动它就地重算 —— "
        f"而且每个数字都在生成期被数过两遍：一遍从真实权重的 arr.size，一遍用这条闭式。",
        {"PARAMS": "4d² + 3·d·d_ff + 4d（MHA + SwiGLU + 两组 γ、β）"})

    # ---- the account rides on every step, so the ledger bar has bytes to draw
    for s in steps:
        s["params"] = {"bytes": par["bytes"], "closed_form": par["closed_form"],
                       "elements": par["elements"], "priced": par["priced"],
                       "priced_total": par["priced_total"],
                       "deploy_bytes": par["deploy_bytes"], "config": par["config"]}

    graph = build_graph(steps)
    meta = {
        "lab": "L04",
        "title": "Decoder Block：残差 / LayerNorm / SwiGLU",
        "config": {"d_model": d, "d_ff": dff, "H": H, "d_head": dh,
                   "N": n, "vocab": cfg["vocab"], "eps": cfg["eps"],
                   "depth": cfg["depth"]},
        "residual_candidates": [c[0] for c in RESIDUAL_CANDIDATES],
        "norm_contrast": norm_contrast(cfg),
        "params": {
            **par,
            "segments": SEGMENTS,
            "unit": "个参数",
            "grid": {"d_model": list(GRID_D), "d_ff": list(GRID_DFF)},
            "formula": r"P_{\mathrm{block}} = 4d^2 + 3\,d\,d_{ff} + 4d",
        },
        "decoder": {
            "flow_pre": list(FLOW_PRE),
            "flow_post": list(FLOW_POST),
            "steps_with_ln": [s["id"] for s in steps if "ln" in s],
            "steps_with_residual": [s["id"] for s in steps if "residual" in s],
            "steps_with_swiglu": [s["id"] for s in steps if "swiglu" in s],
        },
        "reference": {
            "torch": "整块前向与 torch 独立实现逐元素对拍",
            "max_rel_dev": dev,
        },
    }
    return {"meta": meta, "tensors": tensors_for(cfg), "graph": graph,
            "steps": steps}, dev


def tensors_for(cfg):
    d, dff, n, H = cfg["d_model"], cfg["d_ff"], cfg["N"], cfg["H"]
    def T(shape, at, init=None):
        t = {"shape": list(shape), "at": at}
        if init is not None:
            t["init"] = init
        return t
    return {
        "x": T([n, d], "残差流（输入）"),
        "h1": T([n, d], "第一个子层的输入"),
        "q": T([n, d], "多头注意力的 Q"),
        "k": T([n, d], "多头注意力的 K"),
        "v": T([n, d], "多头注意力的 V"),
        "mask": T([n, n], "因果掩码（上三角 −∞）"),
        "scores": T([H, n, n], "注意力分数，每个头一张（已加掩码）"),
        "probs": T([H, n, n], "softmax 后的注意力权重，每个头一张"),
        "ctx": T([n, d], "加权求和并合并头"),
        "attn_out": T([n, d], "注意力子层输出"),
        "y1": T([n, d], "第一个残差加的结果"),
        "h2": T([n, d], "第二个子层的输入"),
        "gate": T([n, dff], "SwiGLU gate 路径"),
        "up": T([n, dff], "SwiGLU up 路径"),
        "silu": T([n, dff], "gate 过 SiLU"),
        "gated": T([n, dff], "逐元素相乘的结果"),
        "ffn_out": T([n, d], "SwiGLU 降维后"),
        "y2": T([n, d], "block 输出（也是下一个 block 的输入）"),
    }


def build_graph(steps):
    """Data-dependency edges between the steps, labelled with the tensor that
    crosses. The engine adds the synthetic sequence edges itself for layering;
    these are the real ones and they are what the DAG draws."""
    order = [s["id"] for s in steps]
    edges = [
        ("x.in", "x.ln1", "x"),
        ("x.ln1", "x.qkv", "h1"),
        ("x.qkv", "x.scores", "q"),
        ("x.scores", "x.attn", "scores"),
        ("x.attn", "x.res1", "attn_out"),
        ("x.in", "x.res1", "x"),
        ("x.res1", "x.ln2", "y1"),
        ("x.ln2", "x.gate", "h2"),
        ("x.ln2", "x.up", "h2"),
        ("x.gate", "x.silu", "gate"),
        ("x.silu", "x.gated", "silu"),
        ("x.up", "x.gated", "up"),
        ("x.gated", "x.down", "gated"),
        ("x.res1", "x.res2", "y1"),
        ("x.down", "x.res2", "ffn_out"),
        ("x.res2", "x.out", "y2"),
    ]
    nodes = []
    for s in steps:
        nodes.append({"id": s["id"], "kind": s["kind"], "op": s["op"],
                      "title": s["title"], "phase": s["phase"],
                      "label": s["id"]})
    placed = {n["id"] for n in nodes}
    edges = [{"from": a, "to": c, "tensor": t} for a, c, t in edges
             if a in placed and c in placed]
    return {"nodes": nodes, "edges": edges}


# -------------------------------------------------------------------- self-lint
SLOT_RE = re.compile(r"\\slot\{([A-Za-z0-9_]+)\}")
REGION_RE = re.compile(r"\\region\{([A-Za-z0-9_]+)\}")
LATEX_IN_TEXT_RE = re.compile(r"\\[a-zA-Z]+\{")
BAD_LITERAL_RE = re.compile(r"^(NaN|Infinity|-Infinity|undefined|null)$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$")


def lint(trace):
    """The author-side half of the contract check.

    The base rules are the same as `labs/assets/engine/trace-model.js`, which
    carries the JS port. The four blocks this lab adds are linted here AND in
    the view that renders each one (`labs/assets/engine/views/shape-guard.js`,
    `norm-axis.js`, `swiglu-paths.js`, `norm-contrast.js`) — the two ports must
    agree, and each needs its own sabotage case or the side without one is
    untested.
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
            gaps.append(f'张量 "{t}" 被读（reads[]），但 tensors[] 里根本没有声明')
        elif t not in written and "init" not in trace["tensors"][t]:
            gaps.append(f'张量 "{t}" 被读但从未被写，且 tensors[] 里没有 init')
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
                    gaps.append(f'步骤 "{sid}" 的 formula.{tier} 引用了 '
                                f"\\slot{{{key}}}，但 bindings 里没有")
                elif bdg.get(tier) is None:
                    gaps.append(f'步骤 "{sid}" 的绑定 "{key}" 缺 "{tier}" 形态')
                elif BAD_LITERAL_RE.match(str(bdg[tier])):
                    gaps.append(f'步骤 "{sid}" 绑定 "{key}".{tier} = {bdg[tier]} —— '
                                f"±∞/NaN 要用哨兵字符串")
            for key in REGION_RE.findall(formula[tier]):
                if key not in (s.get("regions") or {}):
                    gaps.append(f'步骤 "{sid}" 的 formula.{tier} 引用了 '
                                f"\\region{{{key}}}，但 step.regions 里没有")

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
                if isinstance(item, list):
                    for x in item:
                        if isinstance(x, str) and x not in (NEG_INF, POS_INF, NAN):
                            gaps.append(f'步骤 "{sid}" 的 state["{key}"] 含未知哨兵 '
                                        f'字符串 "{x}"')
                elif isinstance(item, str) and item not in (NEG_INF, POS_INF, NAN):
                    gaps.append(f'步骤 "{sid}" 的 state["{key}"] 含未知哨兵字符串 '
                                f'"{item}"')

        for key in (s.get("regions") or {}):
            if not any(f"\\region{{{key}}}" in (formula.get(t) or "")
                       for t in ("sym", "idx", "num")):
                warns.append(f'步骤 "{sid}" 声明了 region "{key}" 但公式里没有引用')

    gaps += lint_exposure(trace)
    gaps += lint_residual(trace)
    gaps += lint_ln_axis(trace)
    gaps += lint_swiglu(trace)
    gaps += lint_norm_contrast(trace)
    gaps += lint_params(trace)

    infos.append(f'共 {len(trace["steps"])} 步 / {len(trace["graph"]["nodes"])} 节点 / '
                 f'{len(trace["graph"]["edges"])} 数据边')
    infos.append(f'state 写过的张量：{", ".join(sorted(written))}')
    return gaps, warns, infos


def steps_with(trace, field):
    return [s for s in trace["steps"] if field in s]


def lint_exposure(trace):
    """`meta.decoder` is the one-level-down gate.

    The per-step rules below fire on `step.ln` / `step.residual` /
    `step.swiglu`. Gating them on the field itself would make the sabotage that
    deletes `meta.decoder` invisible, so the gate lives one level down: if the
    trace carries any of those fields, `meta.decoder` must exist and must list
    exactly the steps that carry them. (Same arrangement `tiling-stage.js` uses
    to serve four labs — see `labs/traces/README.md`.)
    """
    gaps = []
    d = (trace.get("meta") or {}).get("decoder")
    roles = {"ln": "steps_with_ln", "residual": "steps_with_residual",
             "swiglu": "steps_with_swiglu"}
    present = {f: [s["id"] for s in trace["steps"] if f in s] for f in roles}
    if not any(present.values()):
        if d is not None:
            gaps.append("meta.decoder 存在，但没有任何步骤带 step.ln / step.residual / "
                        "step.swiglu —— 声明与数据不符")
        return gaps
    if not isinstance(d, dict):
        gaps.append("trace 有步骤带本票的步骤字段，但 meta.decoder 缺失 —— "
                    "「哪些步是什么角色」无处可查，逐字段的规则就无法门控")
        return gaps
    for field, key in roles.items():
        listed = d.get(key)
        if not isinstance(listed, list):
            gaps.append(f'meta.decoder.{key} 缺失或不是数组（步骤里确有 {field} 字段）')
            continue
        if sorted(listed) != sorted(present[field]):
            gaps.append(f'meta.decoder.{key} = {listed} 与真正带该字段的步骤 '
                        f'{present[field]} 不一致 —— 声明与数据各说各话')
    return gaps


# ------------------------------------------------------- step.residual rules
def lint_residual(trace):
    """What `views/shape-guard.js` renders, and the only rules it may carry.

    The important one is not "the two shapes are equal" — that is the
    confirmation, and a rule that only checked it would pass on a page where
    the counterexamples had quietly disappeared. The rules are:

      * the two operand shapes are equal, and the result's shape is that shape;
      * the elementwise claim is measured (`element_loop_max_rel_dev` ~ 0), not
        assumed;
      * **at least one candidate broadcasts silently and at least one raises**,
        each with a real message — the second half of the ticket's first
        criterion, enforced rather than described;
      * a broadcast candidate's result shape must NOT equal the operand shape on
        the axis it broadcast (otherwise it was not a broadcast), and an error
        candidate must carry a non-empty message;
      * the `same` candidate is unique: exactly one candidate is the residual
        add itself, so a page cannot present two rows as "the residual".
    """
    gaps = []
    for s in steps_with(trace, "residual"):
        sid, r = s["id"], s["residual"]
        if r.get("lhs_shape") != r.get("rhs_shape"):
            gaps.append(f'步骤 "{sid}" 的残差两侧 shape 不同：'
                        f'{r.get("lhs_shape")} vs {r.get("rhs_shape")}')
        if r.get("result_shape") != r.get("lhs_shape"):
            gaps.append(f'步骤 "{sid}" 的残差结果 shape {r.get("result_shape")} '
                        f'与操作数 {r.get("lhs_shape")} 不同')
        if not r.get("equal"):
            gaps.append(f'步骤 "{sid}" 的 residual.equal 为假')
        if not isinstance(r.get("element_loop_max_rel_dev"), (int, float)) or \
                r["element_loop_max_rel_dev"] > 1e-12:
            gaps.append(f'步骤 "{sid}" 的逐元素相加与双循环重算不一致'
                        f'（最大相对偏差 {r.get("element_loop_max_rel_dev")}）')
        cases = r.get("cases")
        if not isinstance(cases, list) or len(cases) < 4:
            gaps.append(f'步骤 "{sid}" 的 residual.cases 缺失或太短 —— '
                        f'没有对照组，「形状相同」这句话就没有对照')
            continue
        ids = [c.get("id") for c in cases]
        if len(set(ids)) != len(ids):
            gaps.append(f'步骤 "{sid}" 的 residual.cases 有重复 id：{ids}')
        outcomes = [c.get("outcome") for c in cases]
        for want in ("same", "broadcast", "error"):
            if want not in outcomes:
                gaps.append(f'步骤 "{sid}" 的 residual.cases 里没有 "{want}" 的结果 —— '
                            f'题目要求「演示 shape 不匹配会怎样」，'
                            f'缺了它这一步就只剩下「它们相等」')
        if outcomes.count("same") != 1:
            gaps.append(f'步骤 "{sid}" 有 {outcomes.count("same")} 个候选被标为 same，'
                        f'真正是残差加的只能有一个')
        for c in cases:
            if c.get("outcome") == "same":
                if c.get("result_shape") != r.get("lhs_shape"):
                    gaps.append(f'步骤 "{sid}" 候选 "{c.get("id")}" 自称 same，'
                                f'结果 shape 却是 {c.get("result_shape")}')
            elif c.get("outcome") == "broadcast":
                if c.get("result_shape") != r.get("lhs_shape"):
                    gaps.append(f'步骤 "{sid}" 候选 "{c.get("id")}" 自称广播，'
                                f'但结果 shape {c.get("result_shape")} 与操作数相同 —— '
                                f'那它并没有走广播路径')
                if list(c.get("rhs_shape") or []) == list(r.get("lhs_shape") or []):
                    gaps.append(f'步骤 "{sid}" 候选 "{c.get("id")}" 自称广播，'
                                f'但两侧 shape 相同')
            elif c.get("outcome") == "error":
                if not (c.get("message") or "").strip():
                    gaps.append(f'步骤 "{sid}" 候选 "{c.get("id")}" 报错却没有留下'
                                f'异常文本 —— 那就不是跑出来的')
                if c.get("result_shape") is not None:
                    gaps.append(f'步骤 "{sid}" 候选 "{c.get("id")}" 报错却有结果 shape')
            else:
                gaps.append(f'步骤 "{sid}" 候选 "{c.get("id")}" 的 outcome '
                            f'{c.get("outcome")!r} 不是 same / broadcast / error')
            if c.get("torch") not in ("same", "broadcast", "error"):
                gaps.append(f'步骤 "{sid}" 候选 "{c.get("id")}" 没有 torch 侧的结果 —— '
                            f'「numpy 会广播」是一个库的结论，除非另一个库也问了')
        silent = r.get("silent")
        raised = r.get("raised")
        if not isinstance(silent, list) or not silent:
            gaps.append(f'步骤 "{sid}" 没有列出静默广播的候选')
        if not isinstance(raised, list) or not raised:
            gaps.append(f'步骤 "{sid}" 没有列出报错的候选')
        if isinstance(silent, list) and isinstance(raised, list) and \
                set(silent) & set(raised):
            gaps.append(f'步骤 "{sid}" 某个候选同时被列为广播和报错')
    return gaps


# ---------------------------------------------------------- step.ln rules
_AXIS_TOL = 1e-6


def lint_ln_axis(trace):
    """What `views/norm-axis.js` renders.

    The rule with teeth is the asymmetry: the normalized array's ROW statistics
    must be 0/1 and its COLUMN statistics must not be, and the counterfactual
    (the same formula reducing over the other axis) must show the flip. A trace
    whose rows read 0/1 while the counterfactual also read 0/1 would be
    consistent with having normalized nothing at all, and this is what refuses
    it.
    """
    gaps = []
    for s in steps_with(trace, "ln"):
        sid, ln = s["id"], s["ln"]
        if ln.get("axis") != "d_model":
            gaps.append(f'步骤 "{sid}" 的 ln.axis = {ln.get("axis")!r}，'
                        f'本 lab 的归一化轴是 d_model')
        n, d = ln.get("tokens"), ln.get("d_model")
        for key, want in (("mu", n), ("sigma", n)):
            v = ln.get(key)
            if not isinstance(v, list) or len(v) != want:
                gaps.append(f'步骤 "{sid}" 的 ln.{key} 有 '
                            f'{len(v) if isinstance(v, list) else "?"} 个值，'
                            f'应该是每个 token 一个（{want} 个）')
        if not (isinstance(n, int) and n > 1):
            gaps.append(f'步骤 "{sid}" 的 ln.tokens = {n!r}，至少要有 2 个 token —— '
                        f'只有一个 token 时「逐 token」和「全局」无法区分')
        for key in ("mu_spread", "sigma_spread"):
            v = ln.get(key)
            if not isinstance(v, (int, float)) or v <= 0:
                gaps.append(f'步骤 "{sid}" 的 ln.{key} = {v!r} —— 所有 token 的统计量'
                            f'完全相同的话，逐 token 这件事就看不出来')
        row, col = ln.get("row") or {}, ln.get("column") or {}
        crow, ccol = ln.get("counterfactual_row") or {}, ln.get("counterfactual_column") or {}
        if not row or not col or not crow or not ccol:
            gaps.append(f'步骤 "{sid}" 的 ln 缺 row / column / counterfactual 统计 —— '
                        f'没有它们，「沿哪个轴」就没有可判据')
            continue
        if abs(row.get("mean_max_abs", 1)) > _AXIS_TOL:
            gaps.append(f'步骤 "{sid}"：归一化后每一行的均值应为 0，'
                        f'实测最大 |行均值| = {row.get("mean_max_abs")}')
        for k in ("std_min", "std_max"):
            if abs(row.get(k, 0) - 1.0) > 1e-3:
                gaps.append(f'步骤 "{sid}"：归一化后每一行的标准差应为 1，'
                            f'实测 {k} = {row.get(k)}')
        # The columns must NOT also read 0/1 -- this is the discriminating half.
        col_centred = abs(col.get("mean_max_abs", 0)) <= _AXIS_TOL
        col_unit = abs(col.get("std_min", 0) - 1.0) <= 1e-3 and \
            abs(col.get("std_max", 0) - 1.0) <= 1e-3
        if col_centred and col_unit:
            gaps.append(f'步骤 "{sid}"：归一化后每一列也是 0/1 —— '
                        f'那说明这条 trace 无法区分「沿 d_model」和「沿 token」')
        # ...and the counterfactual must show the flip, or the asymmetry above
        # is not evidence about the axis.
        if not (abs(crow.get("mean_max_abs", 0)) > _AXIS_TOL or
                abs(crow.get("std_min", 0) - 1.0) > 1e-3 or
                abs(crow.get("std_max", 0) - 1.0) > 1e-3):
            gaps.append(f'步骤 "{sid}"：沿 token 轴归一化的对照组，行统计也落在 0/1 —— '
                        f'这个对照组没有翻转，上面那个不对称就不构成证据')
        if not (abs(ccol.get("mean_max_abs", 1)) <= _AXIS_TOL and
                abs(ccol.get("std_min", 0) - 1.0) <= 1e-3 and
                abs(ccol.get("std_max", 0) - 1.0) <= 1e-3):
            gaps.append(f'步骤 "{sid}"：沿 token 轴归一化的对照组，列统计没有落在 0/1 —— '
                        f'对照implementation 本身写错了')
    return gaps


# ------------------------------------------------------ step.swiglu rules
def lint_swiglu(trace):
    """What `views/swiglu-paths.js` renders.

    Three weight shapes, three intermediate shapes, all named — and the
    elementwise claim measured. Two rules have teeth:

      * **the three paths' shapes compose**: `x[d] × W_gate[dff,d] → gate[dff]`,
        `gate ⊙ up` needs `gate`'s shape to equal `up`'s, and `W_down[d,dff]`
        takes `gated[dff]` back to `[d]`. A trace where `up` came out `[N, d]`
        would draw a `⊙` between two different-shaped matrices.
      * **the elementwise sample is real**: `product == a × b` at the published
        cell, and that cell is inside the matrix.
    """
    gaps = []
    steps = steps_with(trace, "swiglu")
    if not steps:
        return gaps
    for s in steps:
        sid, sw = s["id"], s["swiglu"]
        shape = sw.get("shapes") or {}
        want = {"W_gate": ["d_ff", "d_model"], "W_up": ["d_ff", "d_model"],
                "W_down": ["d_model", "d_ff"]}
        n, d, dff = (shape.get("x") or [None, None])[0], sw.get("d_model"), sw.get("d_ff")
        if not (isinstance(n, int) and isinstance(d, int) and isinstance(dff, int)):
            gaps.append(f'步骤 "{sid}" 的 swiglu 缺 d_model / d_ff / x.shape')
            continue
        for k, (a, b) in want.items():
            got = shape.get(k)
            exp = [dff if a == "d_ff" else d, dff if b == "d_ff" else d]
            if got != exp:
                gaps.append(f'步骤 "{sid}" 的 swiglu.shapes["{k}"] = {got}，'
                            f'应为 {exp}')
        for k in ("gate", "up", "silu", "gated"):
            if shape.get(k) != [n, dff]:
                gaps.append(f'步骤 "{sid}" 的 swiglu.shapes["{k}"] = {shape.get(k)}，'
                            f'应为 [{n}, {dff}] —— 这一路是升维后的中间量')
        if shape.get("out") != [n, d]:
            gaps.append(f'步骤 "{sid}" 的 swiglu.shapes["out"] = {shape.get("out")}，'
                        f'应为 [{n}, {d}] —— down 路径必须降回 d_model')
        # ⊙ needs the two operands to be the same shape.
        if shape.get("silu") != shape.get("up"):
            gaps.append(f'步骤 "{sid}"：逐元素相乘的两侧 shape 不同 '
                        f'({shape.get("silu")} vs {shape.get("up")})，'
                        f'⊙ 要求它们完全相同')
        nelem = (sw.get("elements") or {}).get("gated")
        if nelem != n * dff:
            gaps.append(f'步骤 "{sid}" 的 swiglu.elements["gated"] = {nelem}，'
                        f'应为 N×d_ff = {n * dff}')
        if sw.get("path") not in [p["id"] for p in SWIGLU_PATHS]:
            gaps.append(f'步骤 "{sid}" 的 swiglu.path = {sw.get("path")!r} '
                        f'不属于 {[p["id"] for p in SWIGLU_PATHS]}')
        m = sw.get("multiply") or {}
        if not isinstance(m.get("row"), int) or not isinstance(m.get("col"), int):
            gaps.append(f'步骤 "{sid}" 的 swiglu.multiply 缺行列下标')
            continue
        if not (0 <= m["row"] < n and 0 <= m["col"] < dff):
            gaps.append(f'步骤 "{sid}" 的 swiglu.multiply 位置 ({m["row"]},{m["col"]}) '
                        f'越出 [{n},{dff}]')
        for k in ("a", "b", "product"):
            if not isinstance(m.get(k), (int, float)):
                gaps.append(f'步骤 "{sid}" 的 swiglu.multiply.{k} 不是数')
        if not m.get("product_matches"):
            gaps.append(f'步骤 "{sid}" 的 swiglu.multiply 声称 S = a×b，实测不成立 —— '
                        f'那这个逐元素乘就不是跑出来的')
        got = m.get("a", 0) * m.get("b", 0)
        if isinstance(m.get("product"), (int, float)) and \
                abs(got - m["product"]) > 1e-9 * max(1.0, abs(got)):
            gaps.append(f'步骤 "{sid}" 的 swiglu.multiply：a×b = {got} '
                        f'但 product = {m.get("product")} —— 两个字段互相矛盾')
        if m.get("rows_checked") != n * dff:
            gaps.append(f'步骤 "{sid}" 的 swiglu.multiply.rows_checked = '
                        f'{m.get("rows_checked")}，应为 {n * dff}')
    paths = {s["swiglu"].get("path") for s in steps}
    for want in ("gate", "up", "down"):
        if want not in paths:
            gaps.append(f'回放里没有任何一步走 {want} 路径 —— 三条路径没有走全')
    return gaps


# ------------------------------------------------- meta.norm_contrast rules
def lint_norm_contrast(trace):
    """What `views/norm-contrast.js` renders.

    The two rules that matter:

      * **the two flows differ at exactly one index and that element is the
        norm.** This is the `#40` mermaid pair's whole content, published as
        data: identical node names, one difference. Enforced, so the picture
        cannot drift from the claim written above it.
      * **the ordering is the lesson**: Post-Norm's input/output gradient ratio
        must exceed Pre-Norm's, and Post-Norm's residual RMS must hold flat
        while Pre-Norm's grows. A trace that published two identical series
        would render a chart showing no difference, which is the one thing this
        block exists to show.
    """
    gaps = []
    nc = (trace.get("meta") or {}).get("norm_contrast")
    if nc is None:
        if steps_with(trace, "ln"):
            gaps.append("trace 有 LayerNorm 步骤，但 meta.norm_contrast 缺失 —— "
                        "Pre/Post 对照无处可查")
        return gaps
    modes = nc.get("modes")
    depths = nc.get("depths")
    if not isinstance(depths, list) or len(depths) < 2:
        gaps.append("meta.norm_contrast.depths 缺失或不足两个点，画不出趋势")
        return gaps
    if not isinstance(modes, list) or len(modes) != 2:
        gaps.append("meta.norm_contrast.modes 不是两条")
        return gaps
    ids = [m.get("id") for m in modes]
    if ids != ["pre", "post"]:
        gaps.append(f"meta.norm_contrast.modes 的 id 是 {ids}，应为 ['pre', 'post']")
    flows = []
    for m in modes:
        flow = m.get("flow")
        if not isinstance(flow, list) or not flow:
            gaps.append(f'meta.norm_contrast["{m.get("id")}"] 缺 flow')
            continue
        flows.append(flow)
        if "LN" not in flow:
            gaps.append(f'meta.norm_contrast["{m.get("id")}"].flow 里没有 LN')
        src = nc.get("depths")
        for key in ("residual_rms", "grad_norm"):
            series = m.get(key)
            if not isinstance(series, list) or len(series) != len(src):
                gaps.append(f'meta.norm_contrast["{m.get("id")}"].{key} 有 '
                            f'{len(series) if isinstance(series, list) else "?"} 项，'
                            f'应为 {len(src)}（每个深度一项）')
        gn = m.get("grad_norm")
        if isinstance(gn, list):
            for i, g in enumerate(gn):
                if not isinstance(g, list) or len(g) != src[i]:
                    gaps.append(f'meta.norm_contrast["{m.get("id")}"].grad_norm[{i}] '
                                f'（深度 {src[i]}）有 '
                                f'{len(g) if isinstance(g, list) else "?"} 个边界值，'
                                f'应该每个 block 边界一个')
                    break
        # The charted curve and the headline ratio are two publications of one
        # quantity: `grad_amp_log[-1]` is log10(grad_in_over_out). A page that
        # printed one and drew the other would be making two claims.
        amp = m.get("grad_amp_log")
        ratio = m.get("grad_in_over_out")
        if not isinstance(amp, list) or len(amp) != len(src):
            gaps.append(f'meta.norm_contrast["{m.get("id")}"].grad_amp_log 有 '
                        f'{len(amp) if isinstance(amp, list) else "?"} 项，'
                        f'应为 {len(src)}')
        else:
            want = math.log10(ratio) if isinstance(ratio, (int, float)) and ratio > 0 \
                else None
            if want is not None and abs(amp[-1] - want) > 1e-6 * max(1.0, abs(want)):
                gaps.append(f'meta.norm_contrast["{m.get("id")}"].grad_amp_log '
                            f'最深处 {amp[-1]}，而 grad_in_over_out = {ratio} '
                            f'对应 {want:.6g} —— 画的曲线和结论行的数字脱节')
            if amp[0] < -1e-6:
                gaps.append(f'meta.norm_contrast["{m.get("id")}"].grad_amp_log[0] = '
                            f'{amp[0]} 是负的 —— 最外层边界相对自己必然是 1（log 为 0）')
        ratio = m.get("grad_in_over_out")
        if not isinstance(ratio, (int, float)) or not ratio > 0:
            gaps.append(f'meta.norm_contrast["{m.get("id")}"].grad_in_over_out '
                        f'= {ratio!r} 不是正数')
    if len(flows) == 2:
        # "Only the norm's position differs" is a statement about a LIST of
        # node names, where moving one element to another index shifts every
        # element between them. Comparing the two lists position by position
        # therefore reports three differences for the four-node graphs — which
        # is what this rule did in its first version, and it was rejecting a
        # correct trace. The right reading is the one `#40` states: the two
        # graphs have the SAME node set, and the norm sits at a different
        # index. So: strip the norm from both and the remainders must be
        # identical, and the norm's index must differ.
        a, b = flows
        if sorted(a) != sorted(b):
            gaps.append(f"两条数据流的节点集合不同：{a} vs {b} —— "
                        f"「只有 LN 位置不同」描述的是同一个 block")
        elif a.index("LN") == b.index("LN"):
            gaps.append(f"两条数据流里 LN 都在第 {a.index('LN')} 位：{a} vs {b} —— "
                        f"Pre/Post-Norm 的差别就是它的位置")
        else:
            rest_a = [x for x in a if x != "LN"]
            rest_b = [x for x in b if x != "LN"]
            if rest_a != rest_b:
                gaps.append(f"除 LN 之外两条数据流也不一样：{rest_a} vs {rest_b} —— "
                            f"结点名应该完全相同，只有 norm 位置不同")
            at = (nc.get("difference") or {}).get("at_index")
            if at != b.index("LN"):
                gaps.append(f"meta.norm_contrast.difference.at_index = {at}，"
                            f"Post-Norm 的 LN 实际在第 {b.index('LN')} 位")
            if (nc.get("difference") or {}).get("only") != "LN":
                gaps.append("meta.norm_contrast.difference.only 不是 LN")
    if len(modes) == 2:
        pre, post = modes[0], modes[1]
        gp, gq = pre.get("grad_in_over_out"), post.get("grad_in_over_out")
        if isinstance(gp, (int, float)) and isinstance(gq, (int, float)) and not gq > gp:
            gaps.append(f"Post-Norm 的梯度放大 {gq} 没有大于 Pre-Norm 的 {gp} —— "
                        f"这正是这张图要展示的差别，反过来就不成立了")
        rp, rq = pre.get("residual_rms"), post.get("residual_rms")
        if isinstance(rp, list) and isinstance(rq, list) and len(rp) > 1:
            for i in range(1, len(rp)):
                if not rp[i] > rp[i - 1]:
                    gaps.append(f"Pre-Norm 的残差流 RMS 不是随深度单调上升："
                                f"{rp[i - 1]} → {rp[i]}")
                    break
            if not all(abs(v - rq[0]) < 1e-3 for v in rq):
                gaps.append(f"Post-Norm 的残差流 RMS 没有保持恒定：{rq} —— "
                            f"LN 的输出本来就是单位方差，这是那张图的对照组")
        pr = nc.get("probe") or {}
        if not pr.get("loss") or not pr.get("direction"):
            gaps.append("meta.norm_contrast.probe 缺 loss / direction —— "
                        "梯度这个数是量出来的，得说清量的是什么")
    if (nc.get("order") or {}).get("grad") != "post>pre":
        gaps.append("meta.norm_contrast.order.grad 不是 'post>pre'")
    return gaps


# ------------------------------------------------------------ meta.params rules
def lint_params(trace):
    """What the page renders through the **existing** ledger component.

    L04 declares its own key rather than `trace.ledger` because that block's
    lint is bound to L05's KV configuration keys; the reasoning is in this
    file's docstring and in `labs/traces/README.md`. The rules here are the ones
    the parameter account actually needs:

      * every step carries the account (the component is redrawn per step);
      * the segments are declared once, with ids/colours/formulas;
      * `bytes` and `closed_form` agree — the count from the real arrays and the
        count from `3.7 §6.1`'s expression are two statements about one number;
      * `elements` is the sum of the parts, and the parts are the closed form
        evaluated at this trace's own `config` — so a page that slid `d_model`
        without moving the numbers is caught.
    """
    gaps = []
    block = (trace.get("meta") or {}).get("params")
    with_step = steps_with(trace, "params")
    if block is None:
        if with_step:
            gaps.append("步骤带了 params，但 meta.params 缺失 —— 分项没有声明")
        return gaps
    segs = block.get("segments")
    if not isinstance(segs, list) or not segs:
        gaps.append("meta.params.segments 缺失或为空")
        return gaps
    ids = []
    for seg in segs:
        sid = seg.get("id")
        if not isinstance(sid, str) or not sid:
            gaps.append(f"meta.params.segments 里有分项缺 id：{seg}")
            continue
        ids.append(sid)
        if not seg.get("label"):
            gaps.append(f'meta.params 分项 "{sid}" 缺 label')
        if not COLOR_RE.match(str(seg.get("color") or "")):
            gaps.append(f'meta.params 分项 "{sid}" 的 color={seg.get("color")!r} '
                        f"不是 #rgb / #rrggbb")
        if not seg.get("formula"):
            gaps.append(f'meta.params 分项 "{sid}" 缺 formula')
    if len(set(ids)) != len(ids):
        gaps.append(f"meta.params.segments 有重复 id：{ids}")
    if not block.get("unit"):
        gaps.append("meta.params.unit 缺失")
    cfg = block.get("config") or {}
    d, dff = cfg.get("d_model"), cfg.get("d_ff")
    if not (isinstance(d, int) and isinstance(dff, int)):
        gaps.append("meta.params.config 缺 d_model / d_ff —— 没有它就无法独立重算")
        return gaps
    want = analytic_param_count({"d_model": d, "d_ff": dff})
    for k in ids:
        if k not in want:
            gaps.append(f'meta.params 有分项 "{k}"，但闭式里没有这一项')
    for k, v in want.items():
        if k in ids and (block.get("closed_form") or {}).get(k) != v:
            gaps.append(f'meta.params.closed_form["{k}"] = '
                        f'{(block.get("closed_form") or {}).get(k)}，'
                        f'用本配置的闭式应为 {v}')
        if k in ids and (block.get("bytes") or {}).get(k) != v:
            gaps.append(f'meta.params.bytes["{k}"] = {(block.get("bytes") or {}).get(k)}，'
                        f'与实算的 {v} 不符')
    total = sum(want[k] for k in ids if k in want)
    if block.get("elements") != total:
        gaps.append(f'meta.params.elements = {block.get("elements")}，'
                    f'各分项之和应为 {total}')
    # The byte pricing, which is what the ledger component draws. Two rules:
    # `priced` is `bytes × deploy_bytes`, and `deploy_bytes` is the multiplier
    # the trace declares — so the count and the cost cannot drift into being two
    # independent numbers on one page.
    mul = block.get("deploy_bytes")
    if not isinstance(mul, int) or mul <= 0:
        gaps.append(f'meta.params.deploy_bytes = {mul!r} 不是正整数 —— '
                    f'「每个参数几个字节」是 3.7 §8 那条公式的乘数')
        return gaps
    priced = block.get("priced")
    if not isinstance(priced, dict):
        gaps.append("meta.params.priced 缺失 —— 账本组件画的是字节数，不是参数个数")
    else:
        for k in ids:
            if k in want and priced.get(k) != want[k] * mul:
                gaps.append(f'meta.params.priced["{k}"] = {priced.get(k)}，'
                            f'应为 bytes × {mul} = {want[k] * mul}')
        if block.get("priced_total") != sum(priced.get(k, 0) for k in ids if k in want):
            gaps.append(f'meta.params.priced_total = {block.get("priced_total")}，'
                        f'各分项之和应为 '
                        f'{sum(priced.get(k, 0) for k in ids if k in want)}')
    for s in with_step:
        p = s["params"]
        if p.get("bytes") != block.get("bytes"):
            gaps.append(f'步骤 "{s["id"]}" 的 params.bytes 与 meta.params 不一致 —— '
                        f'整个 block 的参数量不随步骤变化，两份说法必须相同')
        if p.get("closed_form") != block.get("closed_form"):
            gaps.append(f'步骤 "{s["id"]}" 的 params.closed_form 与 meta.params 不一致')
        if p.get("config") != block.get("config"):
            gaps.append(f'步骤 "{s["id"]}" 的 params.config 与 meta.params 不一致')
        if p.get("priced") != block.get("priced"):
            gaps.append(f'步骤 "{s["id"]}" 的 params.priced 与 meta.params 不一致')
    if len(with_step) != len(trace["steps"]):
        gaps.append(f"只有 {len(with_step)}/{len(trace['steps'])} 个步骤带 params —— "
                    f"账本组件每帧都要重画，每一步都需要它")
    return gaps


# ------------------------------------------------------------- sabotage table
#
# Every entry breaks one rule above, on the real trace, and is named after the
# view that owns the rule so the acceptance harness can attribute it:
#
#   `shapeguard …`  -> views/shape-guard.js
#   `normaxis …`    -> views/norm-axis.js
#   `swiglu …`      -> views/swiglu-paths.js
#   `contrast …`    -> views/norm-contrast.js
#   `params …`      -> meta.params (the page's own block; no view owns it)
#   unprefixed      -> the base contract (trace-model.js)
SABOTAGE_CASES = {
    # -- base contract
    "read-but-never-written tensor with no init": lambda t: t["tensors"].pop("h1"),
    "state entry for an undeclared tensor": lambda t: t["steps"][1]["state"].update({"nope": 1.0}),
    "binding missing a tier": lambda t: t["steps"][1]["bindings"].update({"MU": {"sym": "a"}}),
    "binding collapsed to a 2-tuple": lambda t: t["steps"][1]["bindings"].update({"SIGMA": {"num": "1"}}),
    "\\slot with no binding": lambda t: t["steps"][1]["formula"].update(
        {"num": t["steps"][1]["formula"]["num"] + r"\slot{GHOST}"}),
    "\\region with no description": lambda t: t["steps"][1]["formula"].update(
        {"sym": t["steps"][1]["formula"]["sym"] + r"\region{GHOST}{x}"}),
    "step with no graph node": lambda t: t["graph"]["nodes"].pop(0),
    "formula tier missing": lambda t: t["steps"][1]["formula"].pop("idx"),
    "raw -Infinity instead of the sentinel": lambda t: t["steps"][3]["state"].update(
        {"mask": "-Infinity"}),
    "unknown sentinel string": lambda t: t["steps"][3]["state"].update({"mask": "-inf"}),
    # -- meta.decoder: the one-level-down gate
    "decoder map removed while steps keep their fields": lambda t: t["meta"].pop("decoder"),
    "decoder map omits a step that carries step.ln": lambda t: t["meta"]["decoder"][
        "steps_with_ln"].pop(1),
    "decoder map claims a step that has no such field": lambda t: t["meta"]["decoder"][
        "steps_with_ln"].append(t["steps"][2]["id"]),
    # -- step.residual (views/shape-guard.js)
    #
    # Every shape these reach for is DERIVED from the trace under test, never
    # written as a literal. The first version used the default configuration's
    # `[8, 32]`, which at `d_model=32` IS the operand shape — so the case
    # silently became a no-op and the generator reported "lint 对照组失效" for
    # exactly the three smallest configurations. That is the same failure
    # `labs/README.md` records for L00's hard-coded sabotage table.
    "shapeguard residual block removed from a step": lambda t: t["steps"][5].pop("residual"),
    "shapeguard the two sides given different shapes": lambda t: t["steps"][5][
        "residual"].update({"_gt": 0, "rhs_shape": _wrong_shape(t, 5)}),
    "shapeguard result shape not the operand shape": lambda t: t["steps"][5][
        "residual"].update({"result_shape": _wrong_shape(t, 5)}),
    "shapeguard equal flag cleared while the shapes still match": lambda t: t["steps"][5][
        "residual"].update({"equal": False}),
    "shapeguard elementwise cross-check broken": lambda t: t["steps"][5][
        "residual"].update({"element_loop_max_rel_dev": 0.5}),
    "shapeguard every candidate declared same": lambda t: [c.update({"outcome": "same"})
        for c in t["steps"][5]["residual"]["cases"]],
    "shapeguard the broadcast counterexample removed": lambda t: [
        c.update({"outcome": "same", "result_shape": t["steps"][5]["residual"]["lhs_shape"]})
        for c in t["steps"][5]["residual"]["cases"] if c["outcome"] == "broadcast"],
    "shapeguard the error counterexample removed": lambda t: [
        c.update({"outcome": "same", "result_shape": t["steps"][5]["residual"]["lhs_shape"]})
        for c in t["steps"][5]["residual"]["cases"] if c["outcome"] == "error"],
    "shapeguard an error case loses its message": lambda t: [
        c.pop("message") for c in t["steps"][5]["residual"]["cases"]
        if c["outcome"] == "error"],
    # A broadcast is "the operand shapes differ but the result took the left
    # one's shape". Making the right-hand shape EQUAL to the left one therefore
    # makes it not a broadcast any more. (The case this replaced set the
    # RESULT to the operand shape, which is what every broadcast already has —
    # a no-op that the harness's "did this sabotage change the trace" check
    # caught.)
    "shapeguard a broadcast case given the operand shape": lambda t: [
        c.update({"rhs_shape": t["steps"][5]["residual"]["lhs_shape"]})
        for c in t["steps"][5]["residual"]["cases"] if c["outcome"] == "broadcast"],
    "shapeguard silent list emptied": lambda t: t["steps"][5]["residual"].update({"silent": []}),
    "shapeguard raised list emptied": lambda t: t["steps"][5]["residual"].update({"raised": []}),
    "shapeguard a case loses its torch result": lambda t: t["steps"][5]["residual"][
        "cases"][0].pop("torch"),
    "shapeguard a case gets an unknown outcome": lambda t: t["steps"][5]["residual"][
        "cases"][0].update({"outcome": "maybe"}),
    "shapeguard duplicate candidate ids": lambda t: t["steps"][5]["residual"]["cases"][1].update(
        {"id": t["steps"][5]["residual"]["cases"][0]["id"]}),
    "shapeguard second residual add left unchecked": lambda t: t["steps"][12][
        "residual"].update({"result_shape": _wrong_shape(t, 12)}),
    # -- step.ln (views/norm-axis.js)
    "normaxis ln block removed from a step": lambda t: t["steps"][1].pop("ln"),
    "normaxis axis named as the token axis": lambda t: t["steps"][1]["ln"].update(
        {"axis": "tokens"}),
    "normaxis mu has one value instead of one per token": lambda t: t["steps"][1]["ln"].update(
        {"mu": t["steps"][1]["ln"]["mu"][:1]}),
    "normaxis per-token statistics all identical": lambda t: t["steps"][1]["ln"].update(
        {"mu_spread": 0.0}),
    "normaxis row statistics not centred": lambda t: t["steps"][1]["ln"]["row"].update(
        {"mean_max_abs": 0.4}),
    "normaxis row standard deviation not 1": lambda t: t["steps"][1]["ln"]["row"].update(
        {"std_max": 3.0}),
    "normaxis columns also read 0/1 so the axis is undecidable": lambda t: t["steps"][1][
        "ln"]["column"].update({"mean_max_abs": 0.0, "std_min": 1.0, "std_max": 1.0}),
    "normaxis counterfactual stops flipping": lambda t: t["steps"][1]["ln"][
        "counterfactual_column"].update({"std_min": 4.0, "std_max": 5.0}),
    "normaxis second LayerNorm left unchecked": lambda t: t["steps"][6]["ln"]["row"].update(
        {"std_min": 5.0}),
    # -- step.swiglu (views/swiglu-paths.js)
    "swiglu block removed from a step": lambda t: t["steps"][7].pop("swiglu"),
    "swiglu up path given the wrong shape": lambda t: t["steps"][8]["swiglu"]["shapes"].update(
        {"up": [8, 64]}),
    "swiglu W_down shape transposed": lambda t: t["steps"][11]["swiglu"]["shapes"].update(
        {"W_down": [352, 64]}),
    "swiglu down path does not return to d_model": lambda t: t["steps"][11]["swiglu"][
        "shapes"].update({"out": [8, 352]}),
    "swiglu elementwise operands given different shapes": lambda t: t["steps"][10]["swiglu"][
        "shapes"].update({"up": [8, 64]}),
    "swiglu element count wrong": lambda t: t["steps"][10]["swiglu"]["elements"].update(
        {"gated": 8}),
    "swiglu multiply cell outside the matrix": lambda t: t["steps"][10]["swiglu"][
        "multiply"].update({"row": 99}),
    "swiglu multiply claims a product it did not compute": lambda t: t["steps"][10]["swiglu"][
        "multiply"].update({"product": 12345.0}),
    "swiglu multiply flag cleared": lambda t: t["steps"][10]["swiglu"]["multiply"].update(
        {"product_matches": False}),
    "swiglu element count does not cover the matrix": lambda t: t["steps"][10]["swiglu"][
        "multiply"].update({"rows_checked": 1}),
    "swiglu unknown path id": lambda t: t["steps"][7]["swiglu"].update({"path": "sideways"}),
    "swiglu a path is never taken": lambda t: [s["swiglu"].update({"path": "gate"})
        for s in t["steps"] if "swiglu" in s],
    # -- meta.norm_contrast (views/norm-contrast.js)
    "contrast block removed": lambda t: t["meta"].pop("norm_contrast"),
    "contrast only one mode published": lambda t: t["meta"]["norm_contrast"]["modes"].pop(),
    "contrast modes in the wrong order": lambda t: t["meta"]["norm_contrast"]["modes"].reverse(),
    "contrast the two flows are identical": lambda t: t["meta"]["norm_contrast"]["modes"][1].update(
        {"flow": list(t["meta"]["norm_contrast"]["modes"][0]["flow"])}),
    "contrast the flows differ in more than the norm": lambda t: t["meta"]["norm_contrast"][
        "modes"][1].update({"flow": ["x", "Add", "SubLayer", "LN"]}),
    "contrast the one difference is not the norm": lambda t: t["meta"]["norm_contrast"][
        "modes"][1].update({"flow": ["x", "LN", "Add", "SubLayer"]}),
    "contrast difference.at_index does not match the data": lambda t: t["meta"][
        "norm_contrast"]["difference"].update({"at_index": 0}),
    "contrast difference.only is not the norm": lambda t: t["meta"]["norm_contrast"][
        "difference"].update({"only": "Add"}),
    "contrast one flow loses its norm node": lambda t: t["meta"]["norm_contrast"][
        "modes"][0]["flow"].remove("LN"),
    "contrast a series is short one depth": lambda t: t["meta"]["norm_contrast"][
        "modes"][0]["residual_rms"].pop(),
    "contrast a gradient row is short one depth": lambda t: t["meta"]["norm_contrast"][
        "modes"][0]["grad_norm"][0].pop(),
    "contrast the charted curve is short one depth": lambda t: t["meta"]["norm_contrast"][
        "modes"][0]["grad_amp_log"].pop(),
    "contrast the charted curve disagrees with the headline ratio": lambda t: t["meta"][
        "norm_contrast"]["modes"][0]["grad_amp_log"].__setitem__(
        len(t["meta"]["norm_contrast"]["depths"]) - 1, 0.0),
    "contrast Post-Norm's gradient amplification is not larger": lambda t: t["meta"][
        "norm_contrast"]["modes"][1].update({"grad_in_over_out": 1.0}),
    "contrast Pre-Norm's residual RMS stops growing": lambda t: t["meta"]["norm_contrast"][
        "modes"][0].update({"residual_rms": [1.0] * len(t["meta"]["norm_contrast"]["depths"])}),
    "contrast Post-Norm's residual RMS drifts": lambda t: _set_series(
        t["meta"]["norm_contrast"]["modes"][1], "residual_rms", 0, 9.0),
    "contrast the charted curve goes negative at the outermost boundary": lambda t: t[
        "meta"]["norm_contrast"]["modes"][0]["grad_amp_log"].__setitem__(0, -0.5),
    "contrast probe description removed": lambda t: t["meta"]["norm_contrast"]["probe"].pop("loss"),
    "contrast the stated ordering is reversed": lambda t: t["meta"]["norm_contrast"].update(
        {"order": {"grad": "pre>post"}}),
    "contrast depths list emptied": lambda t: t["meta"]["norm_contrast"].update({"depths": [1]}),
    # -- meta.params (the page's own account, drawn by views/ledger.js)
    "params block removed while steps keep theirs": lambda t: t["meta"].pop("params"),
    "params.unit missing": lambda t: t["meta"]["params"].pop("unit"),
    "params segment id duplicated": lambda t: t["meta"]["params"]["segments"].append(
        dict(t["meta"]["params"]["segments"][0])),
    "params segment colour is not a CSS colour": lambda t: t["meta"]["params"][
        "segments"][0].update({"color": "cornflowerblue"}),
    "params segment formula missing": lambda t: t["meta"]["params"]["segments"][0].pop("formula"),
    "params segment label missing": lambda t: t["meta"]["params"]["segments"][0].pop("label"),
    "params closed form disagrees with this config": lambda t: t["meta"]["params"][
        "closed_form"].update({"ffn": 1}),
    "params bytes disagree with the real arrays": lambda t: t["meta"]["params"]["bytes"].update(
        {"attn": 1}),
    "params total is not the sum of its parts": lambda t: t["meta"]["params"].update(
        {"elements": 1}),
    "params config loses d_model": lambda t: t["meta"]["params"]["config"].pop("d_model"),
    "params a step disagrees with the block total": lambda t: t["steps"][4]["params"][
        "bytes"].update({"ffn": 1}),
    "params a step loses its account": lambda t: t["steps"][4].pop("params"),
}


def _wrong_shape(trace, step_index):
    """A shape that is definitely NOT the residual step's operand shape.

    The natural wrong answer is "halve the last dimension", and it is wrong in
    one configuration: at `d_model=2` halving gives 1, which still differs, so
    it is fine — but a version of this helper that returned a literal `[N, d]`
    computed from the DEFAULT config is what made three sabotage cases stop
    biting. Deriving it from the trace under test is the fix.
    """
    shape = list(trace["steps"][step_index]["residual"]["lhs_shape"])
    shape[-1] = shape[-1] // 2 if shape[-1] > 1 else shape[-1] + 1
    return shape


def _set_series(mode, key, index, value):
    """In-place replacement of one entry in a published series.

    A direct `[...][0] = 9.0` is not available inside a lambda, and the two
    obvious alternatives are both wrong here: `.insert(0, 9.0)` makes the series
    one entry LONGER, which the lint would reject for the wrong reason (length
    against `depths`, not the drift this case is about), and a `pop`/`insert`
    pair written as an `or` chain returns after the `pop` because a popped float
    is truthy. Both were tried and both reported "caught" for a rule that was
    not the one under test.
    """
    mode[key][index] = value


def sabotage_checks(trace):
    """Prove the lint is not a function that always returns zero.

    Also reports the mutations that turned out to be NO-OPS. A sabotage written
    against a literal that stopped biting proves nothing when the lint reports
    zero — it is a different bug from "the lint missed it", and reporting them
    together would hide the second behind the first.
    """
    failures = []
    before = json.dumps(trace, sort_keys=True)
    for name, mutate in SABOTAGE_CASES.items():
        t = copy.deepcopy(trace)
        try:
            mutate(t)
        except Exception as exc:
            failures.append(f"{name}: 破坏本身失败 {exc!r}")
            continue
        if json.dumps(t, sort_keys=True) == before:
            failures.append(f"{name}: 破坏是空操作（trace 没变）")
            continue
        try:
            gaps, _, _ = lint(t)
        except Exception as exc:
            failures.append(f"{name}: lint 抛异常 {exc!r}")
            continue
        if not gaps:
            failures.append(f"{name}: 破坏了 trace 但 lint 报 0 gap")
    return failures


# ------------------------------------------------------------------------ set
def grid():
    """Every configuration the sliders can reach, in grid order.

    A full product, like L01's and L05's: two independent controls rather than
    two positions of one knob.
    """
    return [(d, dff) for d in GRID_D for dff in GRID_DFF]


# --------------------------------------------------- the `--js-lint` payload
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


def build_one(d_model, d_ff):
    cfg = config_for(d_model, d_ff)
    p = init_weights(cfg)
    real = sum(int(np.asarray(w).size) for w in p.values())
    want = sum(analytic_param_count(cfg).values())
    if real != want:
        raise AssertionError(f"参数个数：模型实际 {real} / 闭式 {want}")
    x = input_ids(cfg)
    trace, dev = build_trace(cfg, p, x)
    return trace, dev


def main():
    only_json = "--js-lint" in sys.argv
    real_stdout = sys.stdout
    if only_json:
        # Every human-readable line goes to stderr so stdout carries the JSON
        # payload and nothing else. Same contract as L01's and L05's generators.
        sys.stdout = sys.stderr

    fails = []
    built = {}
    devs = {}
    for (d_model, d_ff) in grid():
        name = cfg_id(d_model, d_ff)
        trace, dev = build_one(d_model, d_ff)
        cfg = config_for(d_model, d_ff)
        built[name] = trace
        devs[name] = dev
        par = trace["meta"]["params"]
        print(f"\n{name}: d_model={d_model} · d_ff={d_ff} · "
              f"{len(trace['steps'])} 步 · "
              f"{len(json.dumps(trace, ensure_ascii=False)):,} bytes")

        # --- 1: the whole block against torch
        if dev > _REL_TOL:
            fails.append(f"{name}: 整块前向与 torch 的最大相对偏差 {dev:.3e} > "
                         f"{_REL_TOL:.0e}")
        print(f"  对拍：整块 {cfg['N']}×{d_model} 前向与 torch 独立实现逐元素一致，"
              f"最大相对偏差 {dev:.3e}")

        # --- 2: the parameter count, both ways
        print(f"  参数量：attn={par['bytes']['attn']:,} + ffn={par['bytes']['ffn']:,} + "
              f"norm={par['bytes']['norm']:,} = {par['elements']:,}"
              f"（闭式 4d²+3dd_ff+4d 同值）")

        # --- the residual counterexamples, reported by outcome
        res = [s for s in trace["steps"] if "residual" in s]
        kinds = {}
        for s in res:
            for c in s["residual"]["cases"]:
                kinds[c["outcome"]] = kinds.get(c["outcome"], 0) + 1
        print(f"  残差 shape 对照：{len(res)} 处残差加，各试了 "
              f"{len(res[0]['residual']['cases'])} 个右操作数 —— "
              f"same {kinds.get('same', 0)} / broadcast {kinds.get('broadcast', 0)} / "
              f"error {kinds.get('error', 0)}")

        # --- the LN axis asymmetry
        lnsteps = [s for s in trace["steps"] if "ln" in s]
        for s in lnsteps:
            ln = s["ln"]
            print(f"  {s['id']}: 逐 token μ 极差 {ln['mu_spread']:.4g}，"
                  f"行统计 |mean|≤{ln['row']['mean_max_abs']:.2e} "
                  f"std∈[{ln['row']['std_min']:.4f},{ln['row']['std_max']:.4f}]；"
                  f"列统计 |mean|≤{ln['column']['mean_max_abs']:.4f} "
                  f"std∈[{ln['column']['std_min']:.4f},{ln['column']['std_max']:.4f}]")

        # --- the Pre/Post ordering, on probe seeds this trace does not publish
        ok, bad_seed = contrast_ordering_holds(cfg)
        if not ok:
            fails.append(f"{name}: probe seed {bad_seed} 上 Post-Norm 的梯度放大不再"
                         f"大于 Pre-Norm —— 这个结论就只是某次抽样的产物")
        nc = trace["meta"]["norm_contrast"]
        pre, post = nc["modes"]
        print(f"  Pre/Post：Pre 残差流 RMS {pre['residual_rms'][0]:.3f}→"
              f"{pre['residual_rms'][-1]:.3f}（×{pre['rms_growth']:.2f}），"
              f"Post 恒为 {post['residual_rms'][0]:.4f}；"
              f"梯度回传放大 Pre {pre['grad_in_over_out']:.1f}× vs "
              f"Post {post['grad_in_over_out']:.0f}×")

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

    # ------------------------------------------------------ cross-configuration
    #
    # What one trace cannot check about itself: that the sliders move the thing
    # they claim to move. A set where every d_model gave the same FFN segment
    # would lint perfectly and teach nothing.
    if not fails:
        def seg(t, k):
            return t["meta"]["params"]["bytes"][k]

        # d_ff moves the FFN segment linearly and leaves the other two fixed.
        for d_model in GRID_D:
            for k in ("attn", "norm"):
                vals = {seg(built[cfg_id(d_model, dff)], k) for dff in GRID_DFF}
                if len(vals) != 1:
                    fails.append(f"d_model={d_model}: {k} 分项随 d_ff 变了：{sorted(vals)}")
            series = {dff: seg(built[cfg_id(d_model, dff)], "ffn") for dff in GRID_DFF}
            grid_dff = sorted(series)
            lo, hi = grid_dff[0], grid_dff[-1]
            slope = (series[hi] - series[lo]) / (hi - lo)
            if slope != 3 * d_model:
                fails.append(f"d_model={d_model}: FFN 分项对 d_ff 的斜率 {slope} "
                             f"应为 3d = {3 * d_model}")
        print("\n跨配置：d_ff 只动 SwiGLU 分项（斜率 3d），注意力与 LayerNorm 不动 —— "
              f"对照组")

        # d_model moves attention quadratically, FFN linearly, norm linearly.
        for dff in GRID_DFF:
            a = seg(built[cfg_id(GRID_D[0], dff)], "attn")
            b = seg(built[cfg_id(GRID_D[-1], dff)], "attn")
            ratio = GRID_D[-1] / GRID_D[0]
            if b != a * ratio * ratio:
                fails.append(f"d_ff={dff}: 注意力参数量 {a}→{b} 不是按 d² 走的"
                             f"（d 变了 {ratio}×，应变成 {ratio * ratio}×）")
        print("跨配置：d_model 翻倍 → 注意力分项 ×4（4d²），SwiGLU 与 LayerNorm 按线性")

        # The parameter totals differ across the grid -- a slider that produced
        # the same block every time would pass every rule above.
        totals = {t["meta"]["params"]["elements"] for t in built.values()}
        if len(totals) != len(list(grid())):
            fails.append(f"网格里有配置的参数量完全相同：{len(totals)} 个不同值 / "
                         f"{len(list(grid()))} 个配置")

        # The Pre/Post contrast is a property of the architecture, so its
        # ordering must hold in EVERY generated configuration, not just the one
        # it was tuned on.
        for name, t in built.items():
            modes = t["meta"]["norm_contrast"]["modes"]
            if not modes[1]["grad_in_over_out"] > modes[0]["grad_in_over_out"]:
                fails.append(f"{name}: Post-Norm 的梯度放大不再大于 Pre-Norm")

        # The residual counterexamples must be the same story at every size:
        # the same three outcomes, from the same set of candidate shapes.
        shapes = {tuple(sorted((c["outcome"], tuple(c["rhs_shape"]))
                               for c in s["residual"]["cases"]))
                  for t in built.values() for s in t["steps"] if "residual" in s}
        bycfg = {}
        for name, t in built.items():
            r = next(s for s in t["steps"] if "residual" in s)["residual"]
            bycfg[name] = [(c["outcome"], c["id"]) for c in r["cases"]]
        if len({tuple(v) for v in bycfg.values()}) != 1:
            fails.append(f"不同配置下残差的候选结果集合不同：{bycfg}")
        else:
            print(f"跨配置：残差的 {len(next(iter(bycfg.values())))} 个候选在 "
                  f"{len(bycfg)} 个配置下 outcome 完全一致 "
                  f"（相同的广播/报错名单）")

    if only_json:
        base_name = cfg_id(*DEFAULT_PAIR)
        base = built[base_name]
        payload = {"traces": built, "base": base_name,
                   "grid": [list(g) for g in grid()], "sabotages": {}}
        for name, mutate in SABOTAGE_CASES.items():
            broken = copy.deepcopy(base)
            try:
                mutate(broken)
            except Exception as exc:
                payload["sabotages"][name] = {"error": repr(exc)}
                continue
            # Whether the PYTHON port catches it, recorded here rather than left
            # to the caller. Most of these rules have a JS port beside the view
            # that renders them, but the `params …` cases have none — no shared
            # component owns that block's rules (see the docstring on why L04
            # does not reuse `trace.ledger`), so its only port is this file.
            # Without this field the harness would report them as "caught by
            # nobody" while the authority actually rejects every one.
            payload["sabotages"][name] = {
                "trace": broken,
                "changed": json.dumps(broken, sort_keys=True)
                           != json.dumps(base, sort_keys=True),
                "python": len(lint(broken)[0]) > 0,
            }
        real_stdout.write(_payload_json(payload))
        return 0

    if fails:
        print("\nlint / 断言未通过，拒绝写文件：", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        return 1

    written = []
    for (d_model, d_ff) in grid():
        name = cfg_id(d_model, d_ff)
        out = HERE / f"{name}.json"
        out.write_text(json.dumps(built[name], indent=1, ensure_ascii=False) + "\n")
        written.append(name)
        print(f"  {out.name}: {out.stat().st_size:,} bytes")

    manifest = {
        "set": SET,
        "lab": "L04",
        "default": cfg_id(*DEFAULT_PAIR),
        "params": {"d_model": sorted(GRID_D), "d_ff": sorted(GRID_DFF)},
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
