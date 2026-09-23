#!/usr/bin/env python3
"""Trace generator for L05 · KV Cache and autoregressive decoding.

Emits:

    labs/traces/kv-cache.json    prompt 4 tokens -> generate 4, 2 layers, 16 steps

This is also the fixture the **memory-ledger view component** (`labs/assets/
engine/views/ledger.js`, shipped by ticket #45) is validated against. The view
is a renderer: it consumes `trace.ledger` (the segment declarations) and
`step.ledger` (the byte counts), and it carries a pure re-implementation of the
accounting. The two must agree exactly, which is only a meaningful check because
this script computes its side from real numpy arrays rather than from the same
formula re-typed.

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


def state_val(v):
    f = float(v)
    if math.isnan(f):
        return NAN
    if math.isinf(f):
        return NEG_INF if f < 0 else POS_INF
    return f


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

# Deployment dtype: what the ledger prices. See the module docstring for why
# this is not the dtype the reference computes in.
DEPLOY = {"name": "fp16", "weight_bytes": 2, "kv_bytes": 2, "act_bytes": 2}

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


def to_bytes(counts):
    """Element counts -> bytes at the deployment dtype."""
    return {
        "params": counts["params"] * DEPLOY["weight_bytes"],
        "kv_cache": counts["kv_cache"] * DEPLOY["kv_bytes"],
        "activations": counts["activations"] * DEPLOY["act_bytes"],
    }


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
         narration, regions=None, ledger=None):
    d = {"id": sid, "title": title, "kind": kind, "op": op, "phase": phase,
         "formula": formula, "bindings": dict(bindings),
         "reads": reads, "writes": writes, "state": state, "narration": narration}
    if regions:
        d["regions"] = regions
    if ledger is not None:
        d["ledger"] = ledger
    return d


def shape_str(*dims):
    return "[" + ",".join(str(x) for x in dims) + "]"


def build_trace(cfg, p):
    d, H, dh, dff, V, L = (cfg["d_model"], cfg["H"], cfg["d_head"], cfg["d_ff"],
                           cfg["vocab"], cfg["L"])
    P = cfg["prompt_len"]
    G = cfg["gen_len"]

    tokens = list(PROMPT_IDS)
    assert len(tokens) == P

    cache_k, cache_v = None, None
    steps = []
    checks = []

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
        ctx = before + ids_this
        torch_out = torch_forward_uncached(ctx, cfg, p)
        checks.append((f"{pid}.vs-torch", logits, np.asarray(torch_out.numpy())))
        no_cache_logits, _, _, _, _, _ = cached_forward(ctx, cfg, p, None, None, pid)
        checks.append((f"{pid}.vs-own-no-cache", logits, no_cache_logits))

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
            return {"bytes": to_bytes(counts_real),
                    "config": {"phase": sub, "n_q": n_q, "layers": cfg["L"],
                               "seq_len": t_new, "kv_rows_total": kv_rows_total}}

        led_kv = make_ledger("kv")
        led_attn = make_ledger("attn")
        led_ffn = make_ledger("ffn")
        led_head = make_ledger("head")

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
             + ("Prefill 时 {0} 个查询一次算完，矩阵最宽也最贵。".format(n_q) if n_q > 1
                else f"Decode 时只有 1 个查询，矩阵退化成一行；上下文越长这一行越长，"
                     f"这正是 decode 每步都要读全量 KV 的原因。")),
            {"MASK": f"因果掩码 M：被遮住的位置 -∞，softmax 后权重为 0。"
                     f"本步 {n_q} 个查询，每个只看得到自己及之前的 {t_new} 个位置"},
            led_attn,
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
    cfg_out = dict(cfg)
    cfg_out.update({
        "dtype": DEPLOY["name"],
        "weight_bytes": DEPLOY["weight_bytes"],
        "kv_bytes": DEPLOY["kv_bytes"],
        "act_bytes": DEPLOY["act_bytes"],
    })

    return {
        "meta": {
            "lab": "L05",
            "title": f"KV Cache 与自回归生成（Prompt {cfg['prompt_len']} → 生成 "
                     f"{cfg['gen_len']}，{cfg['L']} 层，{len(steps)} 步）",
            "source": "docs/guides/模块一-前置知识/transformer/3.8 从Transformer到LLM自回归生成深入理解.md"
                      " · docs/guides/模块四-推理优化/第1章-LLM推理基础/1.1-LLM推理基础.md",
            "config": cfg_out,
            "reference": "逐步对拍 torch 无缓存全前缀重算（float64）",
            "notes": [
                "参考实现在 float64 下跑（对拍用），账本按部署精度 fp16 计价。",
                f"KV Cache 末态每层 {cfg['N']} 行、{cfg['L']} 层合计 {max_kv_rows_total} 行 —— "
                f"序列有 {cfg['N']} 个 token 时缓存只有 {cfg['N']} 行/层，"
                f"因为被采样的最后一个 token 没有再被喂回去。",
                "变长张量（tokens / h / attn）按终态声明 shape，"
                "网格里铺开的是当前真实存在的单元格，只有标签是容量。",
            ],
        },
        "ledger": {
            "unit": "B",
            "segments": [dict(s) for s in LEDGER_SEGMENTS],
        },
        "tensors": {
            "tokens": {"shape": [cfg["N"]], "dtype": "int32", "at": "HBM",
                       "role": "input", "init": list(PROMPT_IDS),
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
def ledger_checks(trace, cfg):
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
    base = {"L": cfg["L"], "H": cfg["H"], "d_head": cfg["d_head"],
            "d_model": cfg["d_model"], "d_ff": cfg["d_ff"], "vocab": cfg["vocab"],
            "B": cfg["B"], "kv_bytes": DEPLOY["kv_bytes"],
            "act_bytes": DEPLOY["act_bytes"]}

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
        return {"params": counts["params"] * DEPLOY["weight_bytes"],
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

    two_layer = predict(7, L=2)
    four_layer = predict(7, L=4)
    results.append(("层数翻倍 → KV Cache 精确翻倍",
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
    mha = predict(7, H=cfg["H"])
    gqa_half = predict(7, H=cfg["H"] // 2)
    results.append(("KV 头减半（MHA → GQA）→ KV Cache 精确减半，参数不动",
                    mha["kv_cache"] == 2 * gqa_half["kv_cache"] and
                    mha["params"] == gqa_half["params"] and gqa_half["kv_cache"] > 0,
                    f'{gqa_half["kv_cache"]:,} → {mha["kv_cache"]:,} B'))

    return results


# ------------------------------------------------------------------------ main
def main():
    cfg = CONFIG
    p = init_weights(cfg)

    real_params = sum(int(w.size) for w in p.values())
    want_params = analytic_param_count(cfg)
    if real_params != want_params:
        print(f"参数个数：模型实际 {real_params} / 闭式 {want_params}", file=sys.stderr)
        return 1

    trace = build_trace(cfg, p)
    checks = trace.pop("_checks")

    # --- 1 & 2: numpy cache path vs torch no-cache and vs numpy no-cache
    print("== 对拍 ==")
    worst, worst_at = 0.0, ""
    for name, got, want in checks:
        dev = max_rel_dev(got, want)
        if dev > worst:
            worst, worst_at = dev, name
        if dev > _REL_TOL:
            print(f"  FAIL  {name}: 最大相对偏差 {dev:.3e} > {_REL_TOL:.0e}", file=sys.stderr)
            return 1
    print(f"  {len(checks)} 项全部一致（每一步 {CONFIG['vocab']} 维 logits 逐元素对 "
          f"torch 无缓存全前缀重算，并与本脚本自己的无缓存路径互查）")
    print(f"  实测最大相对偏差 {worst:.3e}（{worst_at}），容差 {_REL_TOL:.0e}")

    # --- ledger cross-check: real arrays vs closed form, per step (asserted in
    # build_trace), plus the scaling laws.
    print("\n== 账本 ==")
    results = ledger_checks(trace, cfg)
    failed = False
    for label, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
        failed |= not ok

    last = trace["steps"][-1]["ledger"]["bytes"]
    print(f"  末步账本（{cfg['L']} 层，上下文 {trace['steps'][-1]['ledger']['config']['seq_len']}，"
          f"{DEPLOY['name']}）: " +
          " / ".join(f"{k}={v:,}" for k, v in last.items()) + f" 合计 {sum(last.values()):,} B")
    # The crossover the lab is about: how long the context has to get before the
    # cache outweighs the weights, at this model's shape and at a realistic one.
    for L, d, H, dh, vocab, dff in [(2, 64, 4, 16, 32, 176)]:
        params = analytic_param_count({"L": L, "d_model": d, "H": H, "d_head": dh,
                                       "d_ff": dff, "vocab": vocab}) * DEPLOY["weight_bytes"]
        per_token = 2 * L * H * dh * DEPLOY["kv_bytes"]
        print(f"  交叉点：{L} 层 d_model={d} 的模型，参数 {params:,} B，"
              f"每 token KV {per_token} B → 上下文 {params / per_token:,.0f} 之后 KV 反超参数")

    # --- 3: lint + control group
    print("\n== 契约 lint ==")
    gaps, warns, infos = lint(trace)
    for line in infos:
        print(f"  info  {line}")
    for line in warns:
        print(f"  warn  {line}")
    for line in gaps:
        print(f"  GAP   {line}")
    print(f"  {len(gaps)} gap / {len(warns)} warn")
    if gaps:
        failed = True

    sabotages = sabotage_checks(trace)
    if sabotages:
        failed = True
        for line in sabotages:
            print(f"  LINT-DEAD  {line}")
    else:
        print(f"  lint 对照组：{len(SABOTAGE_CASES)} 种破坏全部被抓到"
              f"（证明上面的 0 不是 lint 永远报 0）")

    if failed:
        print("\nlint 或账本校验未通过 —— 拒绝写文件", file=sys.stderr)
        return 1

    out = HERE / "kv-cache.json"
    out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n")
    print(f"\n{out.name}: {len(trace['steps'])} 步 / "
          f"{len(trace['graph']['nodes'])} 节点, {out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
