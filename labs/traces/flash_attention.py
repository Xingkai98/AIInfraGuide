#!/usr/bin/env python3
"""Trace generator for L06 · FlashAttention V1 (the flagship lab).

Emits one JSON file per configuration, plus a manifest naming the set:

    labs/traces/flash-attention-N<N>-d<D>-M<M>.json   one full replay each
    labs/traces/flash-attention.manifest.json         the set the page inlines

The page's sliders (N, d, SRAM 容量 M) select among these; every configuration
the sliders can reach has a trace here, and the manifest is how the page knows
that. The generator writes the manifest itself, so a slider position can never
point at a trace that was never generated.

WHAT THIS SCRIPT EXISTS TO PROVE
--------------------------------
The lab's central claim is *`S` and `P` never fall back to HBM*. A claim like
that is worth exactly what its verification is worth, so it is not asserted in
prose anywhere -- it is measured, and the measurement carries a control group:

1. **Every HBM access goes through a counted object.** `Mem` below is a dict of
   numpy arrays that records one `(op, tensor, elements)` entry per access. The
   two implementations under comparison -- standard attention and FlashAttention
   -- are *real* numpy code that can only reach memory through it. So the set of
   tensor names in an access log is the set of tensors that were in HBM during
   that run, by construction rather than by declaration. The generator asserts
   that FlashAttention's log names no `S` and no `P`, and that *the same log
   object* on the standard implementation does -- the second half is the control
   that makes the first half mean something (a query that can never return a hit
   proves nothing).

2. **The same claim is written into the trace as residency** (`tensors[].at`),
   so the stage derives which blocks are in which layer rather than being told.
   The acceptance harness then walks every frame of the rendered replay and reads
   the HBM row out of the DOM, with a deliberately broken trace as the control.

3. **The IO counter is the same log, summed.** The design doc asks for `O(N^2)`
   against `O(N^2 d^2 / M)`; those two expressions are *predictions* (written out
   below as `predict_*`) and the trace publishes the *measured* counts next to
   them. The generator asserts the two agree for every configuration in the grid
   -- a formula that stopped describing the algorithm would break that assertion
   rather than quietly relabelling the measurement.

LOOP ORDER
----------
This replays V1's, from the tutorial (§4.3, §5.2): **outer over K/V blocks,
inner over Q blocks**, which is why `O_i`, `m_i`, `l_i` are re-read and re-written
every inner iteration. That is V1's cost and V2's whole diff.

(The ticket for this lab states the order the other way round -- Q outer, K/V
inner -- which is V2's. The tutorial's pseudocode, the design doc's own L07 entry
["循环顺序从 K 外 Q 内改成 Q 外 K 内"], and the fact that L07 would otherwise have
nothing to diff all agree that V1 is K/V-outer, so that is what is replayed. The
discrepancy is called out in `LOOP_ORDER_NOTE` and in the page's lead.)

WHY THE BLOCK-SIZE FORMULA IS NOT THE WHOLE STORY
-------------------------------------------------
The tutorial derives `B_c = ceil(M / 4d)` from "SRAM must hold `Q_i, K_j, V_j,
O_i`, about `4 B d` elements". That is a bound on the DOMINANT term, and the
tutorial says so ("这里忽略了相对较小的 S_ij 和 2B_r"). This script computes the
exact account as well -- `(2B_r + 2B_c) d + B_r B_c + 2B_r`, every term measured
off the actual block shapes -- and publishes both, because the gap between them
is why a real kernel leaves slack. The stage's budget bar renders it.

CORRECTNESS
-----------
FlashAttention is an exact algorithm, not an approximation. The generator
asserts the replayed result against two independent implementations:

  * a straightforward numpy attention (same library, different algorithm: one
    softmax over the whole row instead of a blocked recurrence);
  * `torch.softmax` + `torch.matmul` (different library again).

Both are float64, so the only error between them and the recurrence is the order
in which the row is summed; the tolerance is derived from that (see
`_CHECK_REL_TOL`) and the residual is asserted to sit under a bound derived from
the term count, the way `ring_allreduce.py` does it. And the classic bug --
dropping the correction factor `e^(m_old - m_new)` -- is run as a *shadow
recursion* alongside: its deviation is measured per step and at the end, so
"online softmax needs the rescale" is a pair of numbers rather than a sentence.

Every number in a narration or a formula is computed here. The only literals are
the seed that produces Q, K and V.

Run: python3 labs/traces/flash_attention.py
     python3 labs/traces/flash_attention.py --js-lint   # parity payload
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
# strings, not a local convention -- the engine parses exactly these.
NEG_INF = r"-\infty"
POS_INF = r"+\infty"
NAN = "NaN"

# The design doc's shared example model (docs/plans/interactive-labs.md §二 L06):
# N=8, d_model=64, H=4, d_head=16. L00 replays the same shape of data and L01 the
# same scale, so a reader's sense of size carries over. This lab replays ONE head
# (head 0); `d_head` is the `d` of every shape below.
MODEL = {"d_model": 64, "H": 4}

# The only literals that reach the arithmetic: a scale and a seed. Values are
# rounded to two decimals for the same two reasons L01's are -- they stay
# readable at the four significant digits the page prints, and they keep the
# inlined JSON small (a 2-decimal float serialises to six characters instead of
# eighteen).
_INIT_SCALE = 0.7

# The seed is *chosen*, not typed in, and the choice is a requirement of the lab
# rather than a matter of taste: the correction factor only does anything on a
# block whose row maximum rises, and with small row-blocks a random input can
# leave every configuration in the grid with no such block at all. Then the
# shadow recursion would agree with the correct one exactly, the "forgetting the
# rescale costs this much" claim would measure zero, and L00's mechanism would be
# invisible in its own flagship lab. `_pick_seed` searches for an input where
# every configuration the page offers has at least one rescale and at least one
# block where m holds (so both roles are on screen); `build_trace` asserts it.
_SEED_CANDIDATES = 2000
_SEED = None


_CANDIDATE = None


def _kinds_of(n, d, m):
    plan = block_plan(n, d, m)
    if plan is None:
        return set()
    Q, K, V = make_qkv(n, d, _CANDIDATE)
    mem = Mem(blank_store(n, d, Q, K, V))
    _, frames = flash_attention(Q, K, V, plan["Br"], plan["Bc"], mem, record=True)
    return {correction_kind(f) for f in frames}


def _pick_seed():
    """The first seed that makes the correction factor visible where it matters.

    The requirement is precise, and both halves are needed:

      * **every** configuration in the grid must contain a `rescale` step. This is
        the lab's link back to L00, and the criterion the ticket states outright
        ("修正因子在 m 变化的步上被正确触发"). A cell without one would render a
        replay in which the mechanism the lab exists to reuse never fires.
      * the **default** configuration must also contain an `identity` step. On a
        block where no row's maximum rose, alpha is 1 and forgetting it costs
        nothing -- which is exactly why the bug survives casual testing, and the
        only way that claim gets a number is if such a block is on the default
        timeline.

    `identity` is deliberately not demanded of every cell. It requires ALL B_r
    rows to hold at once, so it gets rare as B_r grows; forcing it everywhere
    would be a requirement on the input rather than on the algorithm.
    """
    global _CANDIDATE, _SEED
    if _SEED is not None:
        return _SEED
    need = ("rescale",)
    for candidate in range(20260600, 20260600 + _SEED_CANDIDATES):
        _CANDIDATE = candidate
        if not all(need[0] in _kinds_of(*cell) for cell in grid()):
            continue
        if "identity" not in _kinds_of(8, 16, 256):
            continue
        _SEED = candidate
        return _SEED
    raise AssertionError(
        f"{_SEED_CANDIDATES} 个种子没能在每个配置里都触发修正因子 —— "
        f"检查输入量级或网格")


def make_qkv(n, d, seed=None):
    """Q, K, V for one head, as fixed problem data.

    A seeded `Generator` rather than a hand-written table: with N up to 16 and d
    up to 16 that would be 768 literals, and the one thing the reader is *not*
    meant to be looking at is the input values. PCG64 is specified and stable, so
    the committed JSON regenerates byte for byte -- which is what
    `.github/workflows/trace-checks.yml` requires. The scale keeps
    `S = QK^T/sqrt(d)` in a range where `exp` neither underflows to zero nor
    saturates, so the heat map in the tensor inspector has something to show.
    """
    if seed is None:
        seed = _pick_seed()
    rng = np.random.default_rng(seed)
    return tuple(
        np.round(rng.uniform(-_INIT_SCALE, _INIT_SCALE, size=(n, d)), 2)
        for _ in range(3)
    )


# ----------------------------------------------------------------- formatting
def fmt(v):
    """LaTeX form, for formula bindings. Never put this in narration."""
    if v is None:
        return r"\mathrm{NaN}"
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
    if v is None:
        return "NaN"
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


def mat(values):
    return [[state_val(x) for x in row] for row in values]


def vec(values):
    return [state_val(x) for x in values]


def vec_tex(values, limit=4):
    """A short vector in a `num` tier.

    Only vectors go into formulas, never a matrix: the slot's \\phantom reserves
    the width of whatever value will land there, so a `B_r x d` output block
    would reserve a multiple of the line it sits on and push the formula out of
    its panel. Tall vectors are elided in the middle rather than truncated --
    `[a, b, ..., z]` still reports the shape and both ends, where a bare first
    three would quietly misrepresent the row.
    """
    parts = [fmt(v) for v in values]
    if len(parts) > limit:
        parts = parts[:2] + [r"\ldots"] + parts[-1:]
    return r"\left[" + r",\; ".join(parts) + r"\right]"


def pct(x):
    return "%.4g%%" % (100.0 * x)


def plus_terms_disp(terms, sep=" + "):
    return sep.join(
        f"{disp(p)}·" + (f"({disp(v)})" if v < 0 else f"{disp(v)}") for p, v in terms
    )


# ============================================================== the HBM model
class Mem:
    """Global memory, as an object the algorithm cannot reach around.

    One entry per ACCESS with its element count -- not one per element, which
    would be millions of records for the standard implementation at N=128 and
    would drown the very thing the log is for. An access is a block transfer, so
    a block transfer is what is recorded; that is also the granularity the IO
    argument is made at.

    `store` is the backing dict, so a read returns real data and a write really
    changes what a later read sees: the two implementations below are the actual
    algorithms, not access patterns shaped like them.
    """

    def __init__(self, store):
        self.store = store
        self.log = []

    def _count(self, name, sl):
        return int(np.asarray(self.store[name][sl]).size)

    def read(self, name, sl):
        # A COPY, not a view, and that is the model rather than a precaution:
        # HBM and SRAM are different memories, so a load genuinely copies and the
        # buffer a later step overwrites is not the same object the earlier step
        # read. Returning a view would make every recorded frame alias live
        # storage -- `O_i`, `m_i`, `l_i` captured at step k would silently show
        # whatever the last step wrote, which looks right in the final result and
        # is wrong in every intermediate the replay exists to show.
        self.log.append(("read", name, self._count(name, sl)))
        return np.array(self.store[name][sl])

    def write(self, name, sl, value):
        self.log.append(("write", name, self._count(name, sl)))
        self.store[name][sl] = value

    def mark(self):
        """A cursor into the log, for attributing accesses to replay steps."""
        return len(self.log)

    def since(self, mark):
        return sum(elements for _, _, elements in self.log[mark:])

    def totals(self):
        reads = sum(e for op, _, e in self.log if op == "read")
        writes = sum(e for op, _, e in self.log if op == "write")
        by_tensor = {}
        for op, name, elements in self.log:
            slot = by_tensor.setdefault(name, {"read": 0, "write": 0})
            slot[op] += elements
        return {
            "reads": reads,
            "writes": writes,
            "elements": reads + writes,
            "accesses": len(self.log),
            "tensors": sorted(by_tensor),
            "by_tensor": by_tensor,
        }


def blank_store(n, d, Q, K, V, with_sp=True):
    store = {"Q": Q.copy(), "K": K.copy(), "V": V.copy(),
             "O": np.zeros((n, d)), "m": np.zeros(n), "l": np.zeros(n)}
    if with_sp:
        store["S"] = np.zeros((n, n))
        store["P"] = np.zeros((n, n))
    return store


def standard_attention(Q, K, V, mem):
    """The baseline FlashAttention replaces: three kernels, S and P materialised.

    Deliberately the un-fused version, because that is what the IO argument is
    about -- a fused baseline *is* FlashAttention. Each `mem` call is a kernel's
    traffic: QK^T reads Q and K and writes the N x N scores; the softmax reads
    them and writes the N x N probabilities; PV reads those and V and writes O.

    The softmax is the safe (max-subtracted) one: this function is also the
    numerical reference the recurrence is checked against, and a reference that
    overflows would check nothing.
    """
    scale = 1.0 / math.sqrt(Q.shape[1])
    Qm = mem.read("Q", (slice(None), slice(None)))
    Km = mem.read("K", (slice(None), slice(None)))
    S = (Qm @ Km.T) * scale
    mem.write("S", (slice(None), slice(None)), S)
    Sm = mem.read("S", (slice(None), slice(None)))
    P = np.exp(Sm - Sm.max(axis=1, keepdims=True))
    P = P / P.sum(axis=1, keepdims=True)
    mem.write("P", (slice(None), slice(None)), P)
    Pm = mem.read("P", (slice(None), slice(None)))
    Vm = mem.read("V", (slice(None), slice(None)))
    O = Pm @ Vm
    mem.write("O", (slice(None), slice(None)), O)
    return np.asarray(O)


def predict_standard(n, d):
    """The IO the standard implementation must produce, from its access shape.

    Written out separately from the implementation on purpose: this is the
    prediction and `Mem` is the measurement, so the assertion that they agree is
    a check on both. (An implementation whose traffic stopped matching its own
    description is exactly the drift the assertion exists to catch.)
    """
    return 2 * n * d + n * n + n * n + n * n + n * n + n * d + n * d


def predict_flash(n, d, br, bc):
    """Same, for the FlashAttention recurrence.

    `Nd + 2N` to lay down the zeroed O / m / l; `2 B_c d` per outer iteration;
    and `3 B_r d + 4 B_r` per inner one -- Q_i, O_i, m_i, l_i in, O_i, m_i, l_i
    out. The inner term is the one that scales as `N^2 d^2 / M`: it is paid
    `T_r T_c` times and `T_c` grows as `1/B_c` as `M` shrinks.
    """
    tr, tc = n // br, n // bc
    return (n * d + 2 * n
            + tc * 2 * bc * d
            + tr * tc * (3 * br * d + 4 * br))


def flash_attention(Q, K, V, br, bc, mem, correct=True, record=False):
    """V1's forward pass, one replay frame per (i, j) pair and per sub-step.

    The loop order is the tutorial's (§4.3): **outer over K/V blocks, inner over
    Q blocks**. `O_i`, `m_i`, `l_i` are therefore read at the top of every inner
    iteration and written back at the bottom of it -- V1's redundant traffic, and
    the thing V2 removes.

    `correct=False` runs the shadow recursion: the identical code with
    `alpha = 1` everywhere, i.e. the classic "forgot to rescale the running
    statistics after m moved" bug. Everything else -- including the access log --
    is the same, which is what makes the deviation it produces a statement about
    the correction factor and not about a different program.
    """
    n, d = Q.shape[0], Q.shape[1]
    scale = 1.0 / math.sqrt(d)
    tr, tc = n // br, n // bc

    frames = []
    # Laying down O, m, l costs the same traffic in both variants.
    mem.write("O", (slice(None), slice(None)), np.zeros((n, d)))
    mem.write("m", (slice(None),), np.full(n, -np.inf))
    mem.write("l", (slice(None),), np.zeros(n))

    for j in range(tc):
        r0, r1 = j * bc, (j + 1) * bc
        mark = mem.mark()
        Kj = mem.read("K", (slice(r0, r1), slice(None)))
        Vj = mem.read("V", (slice(r0, r1), slice(None)))
        kv_cost = mem.since(mark)

        for i in range(tr):
            c0, c1 = i * br, (i + 1) * br
            mark = mem.mark()
            Qi = mem.read("Q", (slice(c0, c1), slice(None)))
            Oi = mem.read("O", (slice(c0, c1), slice(None)))
            mi = mem.read("m", (slice(c0, c1),))
            li = mem.read("l", (slice(c0, c1),))
            ld_cost = mem.since(mark)

            # The three compute blocks touch no memory at all -- which is the
            # point of the lab, so their cost is measured rather than assumed to
            # be zero.
            mark = mem.mark()
            S = (Qi @ Kj.T) * scale
            s_cost = mem.since(mark)

            mark = mem.mark()
            m_tilde = S.max(axis=1)
            m_new = np.maximum(mi, m_tilde)
            mx_cost = mem.since(mark)

            # The correction factor. m_old is -inf on a row's first block, and
            # exp(-inf) is exactly 0, which is why the initialisation is -inf
            # rather than 0: 0 would be a legal maximum and would shift every
            # exponential.
            alpha = np.exp(mi - m_new) if correct else np.ones_like(mi)

            mark = mem.mark()
            P = np.exp(S - m_new[:, None])
            p_cost = mem.since(mark)

            mark = mem.mark()
            l_new = li * alpha + P.sum(axis=1)
            l_cost = mem.since(mark)

            mark = mem.mark()
            O_new = (alpha * li / l_new)[:, None] * Oi + (P / l_new[:, None]) @ Vj
            o_cost = mem.since(mark)

            mark = mem.mark()
            mem.write("O", (slice(c0, c1), slice(None)), O_new)
            mem.write("m", (slice(c0, c1),), m_new)
            mem.write("l", (slice(c0, c1),), l_new)
            wb_cost = mem.since(mark)

            if record:
                frames.append({
                    "i": i, "j": j, "c0": c0, "c1": c1, "r0": r0, "r1": r1,
                    "Qi": Qi, "Kj": Kj, "Vj": Vj, "Oi": Oi, "mi": mi, "li": li,
                    "S": S, "m_tilde": m_tilde, "m_new": m_new,
                    "P": P, "l_new": l_new, "O_new": O_new,
                    "alpha": np.exp(mi - m_new),
                    "cost": {"kv": kv_cost, "ld": ld_cost, "s": s_cost,
                             "mx": mx_cost, "p": p_cost, "l": l_cost,
                             "o": o_cost, "wb": wb_cost},
                })

    return np.asarray(mem.store["O"]), frames


# Tolerance calibration. Both sides are float64 and the two implementations sum
# a row in different orders, so the residual is summation order and nothing else:
# ~1e-16 per term times the number of terms. 1e-12 relative sits three orders
# above that and far below the fourth significant digit the page prints -- tight
# enough that a wrong formula rounding to the same displayed number is still
# caught (the argument from issue #37, re-derived for this dtype rather than
# copied).
_CHECK_REL_TOL = 1e-12
_CHECK_ABS_TOL = 1e-15


def close(a, b):
    return math.isclose(a, b, rel_tol=_CHECK_REL_TOL, abs_tol=_CHECK_ABS_TOL)


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def row_sum_bound(n):
    """A derived ceiling on the relative error of one output element.

    Each `O[p][q]` is a weighted sum of `N` terms; the recurrence adds them one
    block at a time with a rescale in between, the standard version in one pass.
    Two different orders of at most `2N` roundings on quantities of magnitude
    ~1 differ by at most about `4 (2N) eps`. This is a *bound*, not a fitted
    tolerance: the assertion is that the measured residual sits under it, so a
    real algorithmic error (orders of magnitude larger) cannot hide beneath it.
    """
    return 4.0 * (2 * n) * float(np.finfo(np.float64).eps)


def torch_reference(Q, K, V):
    """Independent reference, deliberately a different library and route.

    The result crosses back through `.tolist()` rather than `.numpy()`, and that
    is deliberate rather than stylistic. `.numpy()` goes through torch's numpy
    BRIDGE, which is compiled against the numpy ABI the wheel was built for, so
    the call breaks with "Numpy is not available" whenever the installed numpy
    and the torch wheel disagree on major version -- after torch has imported
    fine and every pure-torch operation here has already run, which makes it read
    as a broken script instead of a version skew. `.tolist()` is a plain
    tensor-to-Python conversion with no numpy involved, so this reference works
    under either numpy. (The workflow pins `numpy<2` against `torch==2.2.0` for
    the same reason; this is the half that does not depend on the pin holding.)
    """
    Qt = torch.tensor(Q, dtype=torch.float64)
    Kt = torch.tensor(K, dtype=torch.float64)
    Vt = torch.tensor(V, dtype=torch.float64)
    S = Qt @ Kt.T / math.sqrt(Qt.shape[1])
    P = torch.softmax(S, dim=1)
    return np.array((P @ Vt).tolist(), dtype=np.float64)


# =============================================================== trace build
def b(slot, sym, idx, num):
    return slot, {"sym": sym, "idx": idx, "num": num}


def step(sid, title, kind, op, phase, formula, bindings, reads, writes, state,
         narration, regions=None, spans=None, flows=None, occ=None, io=None,
         corr=None):
    d = {"id": sid, "title": title, "kind": kind, "op": op, "phase": phase,
         "formula": formula, "bindings": dict(bindings),
         "reads": reads, "writes": writes, "state": state, "narration": narration}
    if regions:
        d["regions"] = regions
    if spans:
        d["spans"] = spans
    if flows:
        d["flows"] = [dict(f, bytes=f["elements"] * 4) for f in flows]
    if occ:
        d["occ"] = occ
    if io is not None:
        d["io"] = io
    if corr is not None:
        d["corr"] = corr
    return d


# The two layers this lab's replay moves data through. The ids are what
# `tensors[].at` and `step.flows[].from/to` refer to, and what the page's stage
# config must match. Two, not L01's three: there is no register level here -- the
# whole argument is about the boundary between HBM and SRAM.
LAYERS = ("HBM", "SRAM")

# The one buffer S and P share. Named once so the occupancy account, the lint and
# the page all spell it the same way.
SP_BUFFER = "S/P 共用缓冲"

LOOP_ORDER_NOTE = (
    "循环顺序是 V1 的：外循环遍历 K/V 块，内循环遍历 Q 块（教程 4.3 / 5.2）。"
    "代价是 O_i、m_i、ℓ_i 每进一个内层迭代都要读写一次 HBM —— 这正是 V2 要消掉的。"
)

# name, human label, shape key. `SP` is the merged S/P buffer: two names, one
# B_r x B_c allocation, because P = exp(S - m_new) is computed in place. That is
# why the tutorial's SRAM table lists only S_ij for this term.
SRAM_PARTS = [
    ("Q_i", "当前 Q 块", "Q"),
    ("K_j", "当前 K 块", "K"),
    ("V_j", "当前 V 块", "V"),
    ("O_i", "累积输出块", "O"),
    (SP_BUFFER, "S_ij / P_ij 共用一块缓冲", "SP"),
    ("m_i", "行最大值", "m"),
    ("l_i", "行指数和", "l"),
]


def occ_parts(live, br, bc, d):
    """The SRAM account at one step: what is resident, and how many elements.

    Counted from the block shapes the algorithm actually holds, not from the
    formula. `live` is the set of SRAM tensor names with a value at this step.

    A part carries `covers` when it accounts for tensors other than its own name
    -- here the merged S/P buffer. Declaring the covered set rather than letting
    the view infer it from the name is what keeps the merge convention out of the
    view component (which must not know that this lab merges S and P) AND out of
    a naming heuristic (which would break the first time a lab named something
    else). The view reads a list; the trace says which tensors are on that
    allocation.
    """
    by_shape = {"Q": (br, d), "K": (bc, d), "V": (bc, d), "O": (br, d),
                "SP": (br, bc), "m": (br,), "l": (br,)}
    parts = []
    for name, label, shape_key in SRAM_PARTS:
        if shape_key == "SP":
            covers = sorted({"S", "P"} & live)
            if not covers:
                continue
        else:
            if name not in live:
                continue
            covers = None
        shape = by_shape[shape_key]
        elements = 1
        for s in shape:
            elements *= s
        part = {"name": name, "label": label, "shape": list(shape),
                "elements": elements}
        if covers:
            part["covers"] = covers
        parts.append(part)
    return {"elements": sum(p["elements"] for p in parts), "parts": parts}


def occ_peak(steps):
    """The most SRAM any single step needs, and which step that is."""
    best = None
    for s in steps:
        occ = s.get("occ")
        if not occ:
            continue
        if best is None or occ["elements"] > best["elements"]:
            best = {"elements": occ["elements"], "parts": occ["parts"], "id": s["id"]}
    return best or {"elements": 0, "parts": [], "id": None}


def block_plan(n, d, m):
    """`B_c`, `B_r`, `T_r`, `T_c` for an SRAM of `M` elements -- or None.

    The tutorial's derivation, verbatim: `B_c = ceil(M / 4d)`, `B_r = min(B_c, d)`.
    Two configurations are then rejected rather than silently rounded, and the
    page reports both to the reader:

      * `B_c > N` -- a block larger than the sequence is not a smaller replay, it
        is a different algorithm (everything fits at once, so there is no
        blocking left to watch);
      * `B_c` (or `B_r`) not dividing `N` -- real kernels mask the tail, which is
        a second mechanism with its own story. Rounding here would quietly change
        what the trace shows, so those cells stay empty.
    """
    bc = math.ceil(m / (4 * d))
    br = min(bc, d)
    if bc < 1 or bc > n or br < 1:
        return None
    if n % bc or n % br:
        return None
    return {"Bc": bc, "Br": br, "Tr": n // br, "Tc": n // bc}


def correction_kind(fr):
    """Which of the three roles the correction factor plays for this (i, j).

    THE FACTOR IS PER ROW, THE STEP IS NOT. A `B_r x B_c` block carries one alpha
    per row -- each row has its own running maximum -- so a block routinely has
    some rows whose m rose and some whose did not. The engine's contract wants one
    `kind` per step, so the honest reduction is:

        `rescale`  the moment ANY row's maximum rose (the step did rescaling work)
        `identity` only when EVERY row's alpha is exactly 1 (the step's rescale
                   really was a no-op for the whole block)

    Collapsing to "some row moved" would call a block identity whose third row was
    rescaled by 0.3, which is the kind of summary that reads well and is false.
    `step_corr` reports the numbers for one real row; `rows_scaled` reports how
    many rows the reduction is abstracting over.
    """
    if fr["j"] == 0:
        return "init"
    return "rescale" if bool(np.any(fr["alpha"] < 1.0)) else "identity"


def rows_scaled(fr):
    """How many of this block's `B_r` rows actually got rescaled."""
    return int(np.count_nonzero(fr["alpha"] < 1.0))


def corr_note(kind, alpha, scaled, br):
    rows = f"{scaled}/{br} 行的 α < 1" if kind != "init" else f"{br} 行共用定义值 0"
    return {
        "init": f"INIT · 第一块没有历史，α 按定义为 0（e^(−∞ − m) = 0），"
                f"更新化简成直接求和。",
        "identity": f"IDENTITY · {rows} —— 本块每一行的 m 都没动，α = 1，"
                    f"乘以 1 是恒等操作，漏掉它本步也不会出错。",
        "rescale": f"RESCALE · {rows}，最小 {disp(alpha)}。抬高了 m 的那几行，"
                   f"历史 ℓ_i 与 O_i 必须乘上 α 才等价于「一开始就用新 m 归一」。",
    }[kind]


def step_corr(fr, fr_bad):
    """The `corr` block for one (i, j) -- L00's amplifier, on the 2-D recurrence.

    `o_correct` and `o_uncorrected` are the Frobenius norms of the running O_i in
    the correct and the shadow recursion. A norm rather than one element because
    O_i is a matrix and any single entry would be an arbitrary sample of it; the
    two variants are the SAME quantity computed two ways, so subtracting them is
    meaningful from the first block on (where they are equal, which is itself the
    lesson).

    factor / m_old / m_new describe ONE row -- the one that had to be scaled
    hardest (`argmin alpha`). That keeps the three numbers consistent with each
    other: they are the same row's factor and the maxima it was computed from,
    rather than a factor from one row and a maximum from another. On an identity
    block every alpha is 1 and every row is the same story, so the choice does not
    matter; on a rescale block it names the row whose numbers a reader should
    reason about. `meta.correction.per_block` carries how many rows the step's
    single `kind` is abstracting over.
    """
    kind = correction_kind(fr)
    r = int(np.argmin(fr["alpha"]))
    o_ok = float(np.linalg.norm(fr["O_new"]))
    o_bad = float(np.linalg.norm(fr_bad["O_new"]))
    bias = abs(o_bad - o_ok)
    return {
        "kind": kind,
        "factor": state_val(float(fr["alpha"][r])),
        "m_old": state_val(float(fr["mi"][r])),
        "m_new": state_val(float(fr["m_new"][r])),
        "o_correct": o_ok,
        "o_uncorrected": o_bad,
        "bias_abs": bias,
        "bias_rel": bias / o_ok if o_ok else 0.0,
    }


def build_correction(frames, frames_bad, O_flash, O_bad):
    """The trace-level correction summary: per block, and at the end.

    Per (i, j) pair, in replay order, because that is the unit the rescale happens
    in -- alpha is applied to a whole row-block between one K/V block and the
    next. The end-state deviation is reported elementwise as well, because that
    is the concrete form of the claim: `max|O_correct - O_uncorrected|`.
    """
    bad_by = {(f["i"], f["j"]): f for f in frames_bad}
    per_block = []
    for f in sorted(frames, key=lambda f: (f["j"], f["i"])):
        kind = correction_kind(f)
        o_ok = float(np.linalg.norm(f["O_new"]))
        o_bad = float(np.linalg.norm(bad_by[(f["i"], f["j"])]["O_new"]))
        per_block.append({
            "n": len(per_block) + 1,
            "kind": kind,
            "factor": state_val(float(np.min(f["alpha"]))),
            "rows_scaled": rows_scaled(f),
            "o_correct": o_ok,
            "o_uncorrected": o_bad,
            "bias_abs": abs(o_bad - o_ok),
            "bias_rel": abs(o_bad - o_ok) / o_ok if o_ok else 0.0,
        })
    final_abs = max_abs_diff(O_bad, O_flash)
    return {
        "rescales": sum(1 for e in per_block if e["kind"] == "rescale"),
        "rows": sum(e["rows_scaled"] for e in per_block),
        "rows_total": sum(len(f["alpha"]) for f in frames),
        "final_o_correct": float(np.linalg.norm(O_flash)),
        "final_o_uncorrected": float(np.linalg.norm(O_bad)),
        "final_bias_abs": final_abs,
        "final_bias_rel": final_abs / float(np.max(np.abs(O_flash))),
        "note": "「不修正」= 同一条递推里 α 恒为 1，其余一字不改。factor 取本块 min α、"
                "rows_scaled 数出本块有几行的 α 严格小于 1（α 是逐行的，kind 是整块的）；"
                "整条 trace 的偏差按输出矩阵的逐元素最大绝对差计；逐块的 o_correct / "
                "o_uncorrected 是各自 O_i 的 Frobenius 范数。",
        "per_block": per_block,
    }


# The IO curves depend only on `d` (and the Bc values offered for it), not on the
# configuration being built, so they are computed once per `d` and reused. The
# curves are a real run at every point -- see `curve_counts`.
_CURVE_CACHE = {}


def curve_counts(d, bc_values, ns):
    key = (d, tuple(bc_values), tuple(ns))
    if key in _CURVE_CACHE:
        return _CURVE_CACHE[key]
    # A different data set from the replayed configurations', on purpose: the
    # curve is about IO as a function of N, and drawing it from the same 16x16
    # input the replay uses would suggest the two are the same measurement.
    rng = np.random.default_rng(_pick_seed() + 1)
    cache = {}

    def data(n):
        if n not in cache:
            cache[n] = tuple(
                np.round(rng.uniform(-_INIT_SCALE, _INIT_SCALE, size=(n, d)), 2)
                for _ in range(3)
            )
        return cache[n]

    flash = []
    law_rows = []
    for bc in bc_values:
        br = min(bc, d)
        points = []
        for n in ns:
            if n < bc or n % bc or n % br:
                continue
            Q, K, V = data(n)
            store = blank_store(n, d, Q, K, V)
            mem = Mem(store)
            flash_attention(Q, K, V, br, bc, mem, correct=True, record=False)
            points.append({"N": n, "elements": mem.totals()["elements"]})
        flash.append({"Bc": bc, "Br": br, "points": points})
        at_last = [p for p in points if p["N"] == ns[-1]]
        if at_last:
            law_rows.append({"Bc": bc, "Br": br, "N": ns[-1],
                             "elements": at_last[0]["elements"],
                             "times_bc": at_last[0]["elements"] * bc})

    standard = []
    for n in ns:
        Q, K, V = data(n)
        store = blank_store(n, d, Q, K, V)
        mem = Mem(store)
        standard_attention(Q, K, V, mem)
        standard.append({"N": n, "elements": mem.totals()["elements"]})

    out = {"ns": list(ns), "standard": standard, "flash": flash, "law_rows": law_rows}
    _CURVE_CACHE[key] = out
    return out


def build_trace(n, d, m, title, source):
    plan = block_plan(n, d, m)
    if plan is None:
        raise ValueError(f"no plan for N={n} d={d} M={m}")
    bc, br = plan["Bc"], plan["Br"]
    tr, tc = plan["Tr"], plan["Tc"]
    n_pairs = tr * tc

    Qa, Ka, Va = make_qkv(n, d)

    # ------------------------------------------------------------- the runs
    mem = Mem(blank_store(n, d, Qa, Ka, Va))
    O_flash, frames = flash_attention(Qa, Ka, Va, br, bc, mem, correct=True, record=True)
    measured_flash = mem.totals()

    # The shadow recursion: same code, `alpha` forced to 1, its own memory so its
    # (wrong) writes propagate exactly as the bug would.
    mem_bad = Mem(blank_store(n, d, Qa, Ka, Va))
    O_bad, frames_bad = flash_attention(Qa, Ka, Va, br, bc, mem_bad,
                                        correct=False, record=True)

    # The baseline, through the same counted object.
    mem_std = Mem(blank_store(n, d, Qa, Ka, Va))
    O_std = standard_attention(Qa, Ka, Va, mem_std)
    measured_std = mem_std.totals()
    O_torch = torch_reference(Qa, Ka, Va)

    # ------------------------------------------------------------ assertions
    # (1) The headline: FA's HBM log names no S and no P; the SAME log object on
    # the standard implementation does. The second clause is the control -- a
    # check that can never fire proves nothing about what it is checking.
    forbidden = {"S", "P"}
    fa_hits = [t for t in measured_flash["tensors"] if t in forbidden]
    std_hits = [t for t in measured_std["tensors"] if t in forbidden]
    if fa_hits:
        raise AssertionError(
            f"FlashAttention 的 HBM 访问日志里出现了 {fa_hits} —— "
            f"「S、P 不落回 HBM」的主张不成立")
    if std_hits != ["P", "S"]:
        raise AssertionError(
            f"标准实现的 HBM 日志里 S / P 是 {std_hits} —— 对照组失效，"
            f"上面那条「没有出现」就没有区分力")
    fa_sp_accesses = sum(1 for _, name, _ in mem.log if name in forbidden)
    std_sp_accesses = sum(1 for _, name, _ in mem_std.log if name in forbidden)
    if fa_sp_accesses != 0 or std_sp_accesses == 0:
        raise AssertionError(
            f"S/P 访问次数：FA {fa_sp_accesses}、标准实现 {std_sp_accesses}")

    # (2) The IO counter is measured, and the prediction describes the
    # measurement. A formula that drifted from the algorithm fails here.
    if measured_flash["elements"] != predict_flash(n, d, br, bc):
        raise AssertionError(
            f"FA 实测 IO {measured_flash['elements']} != 预测 "
            f"{predict_flash(n, d, br, bc)}")
    if measured_std["elements"] != predict_standard(n, d):
        raise AssertionError(
            f"标准实测 IO {measured_std['elements']} != 预测 {predict_standard(n, d)}")

    # (3a) FlashAttention is exact: it must land on the numpy baseline to
    # summation-order precision, and both must land on torch.
    bound = row_sum_bound(n) * float(np.max(np.abs(O_torch)))
    d_std = max_abs_diff(O_flash, O_std)
    d_torch = max_abs_diff(O_flash, O_torch)
    d_xcheck = max_abs_diff(O_std, O_torch)
    for label, delta in (("numpy 标准实现", d_std), ("torch 参考实现", d_torch)):
        if delta > bound:
            raise AssertionError(
                f"FA 与 {label} 的最大偏差 {delta:.3e} 超过推导上界 {bound:.3e} —— "
                f"分块递推本该只是换个求和顺序")

    # (3b) The control for (3a): the shadow recursion -- same code, alpha left at
    # 1 -- must be caught. If it were not, the equality above would only be
    # evidence that the check is blind.
    #
    # This is also where the selected seed earns its keep. With `T_c == 1` the
    # row maximum is never raised after the first block, alpha is 1 everywhere,
    # and the shadow recursion is bit-for-bit the correct one -- so the deviation
    # would be zero and this assertion would fail. `replayable_plan` refuses that
    # cell and `_pick_seed` guarantees a raise happens, so the assertion below is
    # a statement about the algorithm rather than about the input happening to be
    # uninteresting.
    kinds = {correction_kind(f) for f in frames}
    if "rescale" not in kinds:
        raise AssertionError(
            "整条回放没有任何一块抬高行最大值 —— 修正因子不会触发，"
            "L00 的机制在这个 lab 里看不见。检查输入或分块参数")
    d_shadow = max_abs_diff(O_bad, O_torch)
    if d_shadow <= bound:
        raise AssertionError(
            f"漏掉修正因子的影子递推偏差 {d_shadow:.3e} 没有超过上界 {bound:.3e} —— "
            f"对拍没有区分力")
    if mem_bad.totals()["elements"] != measured_flash["elements"]:
        raise AssertionError("影子递推的读取代价与正确版不同 —— 它本该只差一个乘法")

    # (4) The asymptotics, measured rather than quoted: at fixed d the IO of the
    # whole trace set scales as 1/B_c. This is O(N^2 d^2 / M) with a concrete
    # constant, and the constant is what the page prints.
    law_ns = [8, 16, 32, 64, 128]
    bc_values = sorted({block_plan(nn, d, mm)["Bc"]
                        for (nn, dd, mm) in grid() if dd == d}
                       | {block_plan(nn, d, mm)["Bc"]
                          for (nn, dd, mm) in all_cells() if dd == d
                          if block_plan(nn, d, mm)})
    curves = curve_counts(d, bc_values, law_ns)
    law_rows = curves["law_rows"]
    if len(law_rows) < 2:
        raise AssertionError("曲线上的 B_c 取值不足两个，1/B_c 的对比无从谈起")
    spread = (max(r["times_bc"] for r in law_rows)
              / min(r["times_bc"] for r in law_rows) - 1.0)
    if spread > 0.35:
        raise AssertionError(
            f"IO x B_c 在 N={law_ns[-1]} 上的相对跨度 {spread:.3f} 过大 —— "
            f"O(N^2 d^2 / M) 的常数说法不成立：{law_rows}")
    # The control for the law: the standard implementation's IO must NOT scale
    # with B_c, or "x B_c is constant" would be a property of the axis rather
    # than of FlashAttention.
    if len({curves["standard"][-1]["elements"] for _ in law_rows}) != 1:
        raise AssertionError("对照失效：标准实现的 IO 竟然随 B_c 变化")

    correction = build_correction(frames, frames_bad, O_flash, O_bad)

    # ----------------------------------------------------------------- steps
    steps = []
    A = math.sqrt(d)
    live = set()               # SRAM tensors with a value, at the cursor
    running = 0

    def io_of(cost):
        return {"step": cost, "cum": running}

    def push(s):
        steps.append(s)

    init_cost = n * d + 2 * n
    running = init_cost
    push(step(
        "init", "初始化：分块参数与 SRAM 预算", "state", "init", "准备",
        {
            "sym": r"B_c = \left\lceil \frac{M}{4d} \right\rceil,\qquad "
                   r"B_r = \min(B_c,\, d),\qquad T_r = \frac{N}{B_r},\qquad "
                   r"T_c = \frac{N}{B_c}",
            "idx": r"N = \slot{N},\qquad B_c = \left\lceil \frac{\slot{M}}"
                   r"{4 \times \slot{D}} \right\rceil = \slot{BC},\qquad "
                   r"B_r = \min(\slot{BC}, \slot{D}) = \slot{BR},\qquad "
                   r"T_r = \slot{TR},\qquad T_c = \slot{TC}",
            "num": r"N = \slot{N},\qquad B_c = \left\lceil "
                   r"\frac{\region{BC}{\slot{M}}}{\region{BC}{4 \times \slot{D}}} \right\rceil "
                   r"= \slot{BC},\qquad T_r \times T_c = \slot{NP} \text{ 个 tile 对},\qquad "
                   r"\mathrm{IO}_{\mathrm{std}} = \slot{IOSTD}",
        },
        [b("N", "N", str(n), str(n)),
         b("D", "d", str(d), str(d)),
         b("M", "M", str(m), str(m)),
         # The num tier makes the division explicit (M over 4d) because that is
         # where the tutorial's 4 comes from; the sym tier keeps it symbolic.
         b("BC", r"\lceil M/4d \rceil", f"\\lceil {m}/\\left(4 \\times {d}\\right) \\rceil",
           str(bc)),
         b("BR", r"\min(B_c,d)", f"\\min({bc},{d})", str(br)),
         b("TR", "T_r", str(tr), str(tr)),
         b("TC", "T_c", str(tc), str(tc)),
         b("NP", "T_r T_c", f"{tr} \\times {tc}", str(n_pairs)),
         b("IOSTD", r"4Nd + 4N^2", f"4\\cdot{n}\\cdot{d} + 4\\cdot{n}^2",
           str(predict_standard(n, d)))],
        ["Q", "K", "V"], ["O", "m", "l"],
        {"O": mat(np.zeros((n, d))), "m": vec(np.full(n, -np.inf)),
         "l": vec(np.zeros(n))},
        f"Q、K、V 已经躺在 HBM 里（形状 [{n}, {d}] —— 共享模型 N={n}、d_model="
        f"{MODEL['d_model']}、H={MODEL['H']}、d_head={d}，本 lab 只回放第 0 个头）。"
        f"SRAM 容量 M = {m} 个元素，按教程的公式 B_c = ⌈M/4d⌉ = ⌈{m}/{4 * d}⌉ = {bc}，"
        f"B_r = min(B_c, d) = {br}。于是 Q 有 {tr} 个块、K/V 有 {tc} 个块，"
        f"内层外层一共 {n_pairs} 个 tile 对。O、m、ℓ 先在 HBM 里清成 0 / −∞ / 0，"
        f"这一步花掉 {init_cost} 个元素的 HBM 写入。",
        {"BC": f"分母的 4 来自「SRAM 要同时放 Q_i、K_j、V_j、O_i 四块，每块 B×d」——"
               f"教程 4.1 节的推导。它只约束这一项，S 与 m、ℓ 是它省略的零头；"
               f"旁边的占用账会把零头也摆出来。"},
        occ=occ_parts(live, br, bc, d),
        io={"step": init_cost, "cum": running},
    ))

    for j in range(tc):
        r0, r1 = j * bc, (j + 1) * bc
        fr_j = _frame(frames, 0, j)
        cost = fr_j["cost"]["kv"]
        running += cost
        live |= {"K_j", "V_j"}
        push(step(
            f"kv{j}", f"外循环 j={j}：载入 K_{j}、V_{j} 到 SRAM", "comm", "load", "外循环",
            {
                "sym": r"K_j \leftarrow \mathrm{HBM}[K]\left[\slot{R0}:\slot{R1},\ :\right],"
                       r"\qquad V_j \leftarrow \mathrm{HBM}[V]\left[\slot{R0}:\slot{R1},\ :\right]",
                "idx": r"K_{\slot{J}} \leftarrow \mathrm{HBM}[K]\left[\slot{R0}:\slot{R1},\ :\right],"
                       r"\qquad V_{\slot{J}} \leftarrow \mathrm{HBM}[V]\left[\slot{R0}:\slot{R1},\ :\right]",
                "num": r"K_{\slot{J}},\ V_{\slot{J}} \leftarrow \slot{ELEMS} \text{ 个元素}"
                       r"\quad (\mathrm{IO} + \slot{COST})",
            },
            [b("J", "j", str(j), str(j)),
             b("R0", r"j B_c", str(r0), str(r0)),
             b("R1", r"(j+1) B_c", str(r1), str(r1)),
             b("ELEMS", r"2 B_c d", f"2\\cdot{bc}\\cdot{d}", str(cost)),
             b("COST", r"\mathrm{IO}", str(cost), str(cost))],
            ["K", "V"], ["K_j", "V_j"],
            {"K_j": mat(fr_j["Kj"]), "V_j": mat(fr_j["Vj"])},
            f"外层的第 {j + 1} 个块：K 的第 {r0}..{r1 - 1} 行和 V 的同一段从 HBM 搬进 SRAM，"
            f"各 {bc}×{d}，合计 {cost} 个元素。这一对 K/V 块会被内层的全部 {tr} 个 Q 块复用，"
            f"所以整趟回放 K、V 各自只从 HBM 读一次 —— 代价全在 Q 和 O 那边。",
            spans={"K": [[r0, r1], [0, d]], "V": [[r0, r1], [0, d]]},
            flows=[{"from": "HBM", "to": "SRAM", "elements": cost,
                    "note": f"K、V 各一个 {bc}×{d} 的块，{cost} 个 float；整个外循环只付一次"}],
            occ=occ_parts(live, br, bc, d),
            io=io_of(cost),
        ))

        for i in range(tr):
            c0, c1 = i * br, (i + 1) * br
            fr = _frame(frames, i, j)
            fbad = _frame(frames_bad, i, j)
            tag = f"{i}{j}"
            kind = correction_kind(fr)
            alpha_min = float(np.min(fr["alpha"]))
            scaled_rows = rows_scaled(fr)
            cr = step_corr(fr, fbad)
            corr_regions = {"CORR": corr_note(kind, alpha_min, scaled_rows, br)}

            # ---- load Q_i, O_i, m_i, l_i ----------------------------------
            cost = fr["cost"]["ld"]
            running += cost
            live |= {"Q_i", "O_i", "m_i", "l_i"}
            push(step(
                f"ld{tag}", f"(i={i}, j={j})：载入 Q_{i}、O_{i}、m_{i}、ℓ_{i}",
                "comm", "load", "内循环",
                {
                    "sym": r"Q_i, O_i \leftarrow \mathrm{HBM}\left[\slot{R0}:\slot{R1},\ :\right],"
                           r"\qquad m_i, \ell_i \leftarrow \mathrm{HBM}\left[\slot{R0}:\slot{R1}\right]",
                    "idx": r"Q_{\slot{I}}, O_{\slot{I}} \leftarrow \mathrm{HBM}\left[\slot{R0}:\slot{R1},\ :\right],"
                           r"\qquad m_{\slot{I}}, \ell_{\slot{I}} \leftarrow \mathrm{HBM}\left[\slot{R0}:\slot{R1}\right]",
                    "num": r"\mathrm{IO} + \slot{COST},\qquad "
                           r"2B_rd + 2B_r = 2\cdot\slot{BR}\cdot\slot{D} + 2\cdot\slot{BR} "
                           r"= \slot{COST}",
                },
                [b("I", "i", str(i), str(i)),
                 b("R0", r"i B_r", str(c0), str(c0)),
                 b("R1", r"(i+1) B_r", str(c1), str(c1)),
                 b("BR", "B_r", str(br), str(br)),
                 b("D", "d", str(d), str(d)),
                 b("COST", r"2B_rd + 2B_r", f"2\\cdot{br}\\cdot{d} + 2\\cdot{br}",
                   str(cost))],
                ["Q", "O", "m", "l"], ["Q_i", "O_i", "m_i", "l_i"],
                {"Q_i": mat(fr["Qi"]), "O_i": mat(fr["Oi"]),
                 "m_i": vec(fr["mi"]), "l_i": vec(fr["li"])},
                (f"内层第 {i} 个 Q 块。这里搬的不只是 Q_i：V1 的循环顺序是 K/V 在外、Q 在内，"
                 f"所以每个内层迭代开始时，上一轮外循环留下的 O_i、m_i、ℓ_i 都要重新从 HBM "
                 f"读回来。这 {cost} 个元素是 V1 为「内层循环」付的过路费 —— V2 把顺序调过来，"
                 f"就是为了省掉它。"
                 if j > 0 else
                 f"内层第 {i} 个 Q 块：Q 的第 {c0}..{c1 - 1} 行读进 SRAM，O_i、m_i、ℓ_i 此刻"
                 f"还是 init 时的初值（0 / −∞ / 0）。这 {cost} 个元素里只有 Q_i 是真需要读的。"),
                spans={"Q": [[c0, c1], [0, d]], "O": [[c0, c1], [0, d]],
                       "m": [[c0, c1]], "l": [[c0, c1]]},
                flows=[{"from": "HBM", "to": "SRAM", "elements": cost,
                        "note": f"Q_i 与 O_i 各 {br}×{d}，m_i、ℓ_i 各 {br} 个标量"}],
                occ=occ_parts(live, br, bc, d),
                io=io_of(cost),
            ))

            # ---- S_ij = Q_i K_j^T / sqrt(d) -------------------------------
            cost = fr["cost"]["s"]
            running += cost
            live.add("S")
            push(step(
                f"s{tag}", f"(i={i}, j={j})：S_ij = Q_i K_jᵀ / √d —— 只落在 SRAM",
                "op", "matmul", "内循环",
                {
                    "sym": r"S_{ij} = \frac{\region{SP}{\slot{QL} \; K_j^\top}}{\sqrt{d}}"
                           r"\ \in \mathbb{R}^{B_r \times B_c}",
                    "idx": r"S_{\slot{II}\slot{JJ}} = \frac{Q_{\slot{II}} K_{\slot{JJ}}^\top}"
                           r"}{\sqrt{\slot{D}}} \in \mathbb{R}^{\slot{BR} \times \slot{BC}}",
                    "num": r"S_{\slot{II}\slot{JJ}}[0,0] = "
                           r"\frac{\slot{S00}}{\sqrt{\slot{D}}} = \slot{SNUM}",
                },
                [b("II", "i", str(i), str(i)),
                 b("JJ", "j", str(j), str(j)),
                 b("D", "d", str(d), str(d)),
                 b("BR", "B_r", str(br), str(br)),
                 b("BC", "B_c", str(bc), str(bc)),
                 b("QL", r"Q_i[0]", r"Q_{i}[0,:]", vec_tex(list(fr["Qi"][0]))),
                 b("S00", r"(Q_i K_j^\top)[0,0]",
                   r"(Q_{i} K_{j}^\top)[0,0]",
                   fmt(float(fr["Qi"][0] @ fr["Kj"][0]))),
                 b("SNUM", r"S_{ij}[0,0]", r"S_{ij}[0,0]", fmt(float(fr["S"][0][0])))],
                ["Q_i", "K_j"], ["S"],
                {"S": mat(fr["S"])},
                f"内积：Q_i（{br}×{d}）乘 K_jᵀ（{d}×{bc}）得到 {br}×{bc} 的分数块 S_ij，"
                f"再除以 √d = {disp(A)} 缩放。这一步的读全部来自 SRAM —— Q_i 和 K_j 都已经"
                f"在里面了。产生的 S_ij 也<b>只写在 SRAM</b>：{br}×{bc} = {br * bc} 个元素，"
                f"一个都没碰 HBM（访问日志里的增量是 {cost}）。"
                f"标准实现在这一步会把 {n}×{n} = {n * n} 个元素写回 HBM。",
                {"SP": "S_ij 是 FlashAttention 藏起来的第一个中间矩阵。整趟回放结束，"
                       "HBM 的访问日志里都不会出现它的名字 —— 这不是「我们没画」，"
                       "是访问日志里真的没有。"},
                flows=[{"from": "SRAM", "to": "SRAM", "elements": br * bc,
                        "note": f"S_ij 的 {br}×{bc} 个元素在 SRAM 内产生，不跨层"}],
                occ=occ_parts(live, br, bc, d),
                io=io_of(cost),
            ))

            # ---- rowmax + running max --------------------------------------
            cost = fr["cost"]["mx"]
            running += cost
            push(step(
                f"mx{tag}", f"(i={i}, j={j})：行最大值 m_i 抬到 m_new", "op", "max",
                "内循环",
                {
                    "sym": r"\tilde{m}_{ij} = \mathrm{rowmax}(S_{ij}),\qquad "
                           r"m_i^{\mathrm{new}} = \max\left(\slot{MOLD},\ \tilde{m}_{ij}\right)",
                    "idx": r"\tilde{m}_{\slot{II}\slot{JJ}} = \mathrm{rowmax}(S_{\slot{II}\slot{JJ}}),"
                           r"\qquad m_{\slot{II}}^{\mathrm{new}} = "
                           r"\max\left(\slot{MOLD},\ \tilde{m}_{\slot{II}\slot{JJ}}\right)",
                    "num": r"m_{\slot{II}}^{\mathrm{new}} = \max\left(\slot{MOLD},\ \slot{MT}\right)"
                           r" = \slot{MNEW}",
                },
                [b("II", "i", str(i), str(i)),
                 b("JJ", "j", str(j), str(j)),
                 b("MOLD", r"m_i", r"m_{i}", vec_tex(fr["mi"])),
                 b("MT", r"\tilde{m}_{ij}", r"\tilde{m}_{ij}", vec_tex(fr["m_tilde"])),
                 b("MNEW", r"m_i^{\mathrm{new}}", r"m_{i}^{\mathrm{new}}",
                   vec_tex(fr["m_new"]))],
                ["S", "m_i"], ["m_i"],
                {"m_i": vec(fr["m_new"])},
                (f"每行取最大值 {disp(float(np.max(fr['m_tilde'])))}，与当前 m_i 比较后抬到 "
                 f"{disp(float(np.max(fr['m_new'])))}。m 一动，前面按旧 m 累出来的 ℓ_i 和 O_i "
                 f"就都失去参照了 —— 下一步的修正因子就是为这件事准备的。"
                 if kind == "rescale" else
                 (f"第一块：m_i 还是 −∞，所以直接取本块的行最大值 "
                  f"{disp(float(np.max(fr['m_new'])))}。这正是 init 用 −∞ 而不是 0 的原因 ——"
                  f"0 是个合法的最大值，会把所有指数整体压偏。"
                  if kind == "init" else
                  f"本块行最大值 {disp(float(np.max(fr['m_tilde'])))} 没有超过当前 m_i，"
                  f"m 保持不变 —— 下一步的修正因子会恰好是 1。")),
                flows=[{"from": "SRAM", "to": "SRAM", "elements": br,
                        "note": f"{br} 行各一个最大值，全程在 SRAM"}],
                occ=occ_parts(live, br, bc, d),
                io=io_of(cost),
            ))

            # ---- P_ij = exp(S - m_new) ------------------------------------
            cost = fr["cost"]["p"]
            running += cost
            live.add("P")
            push(step(
                f"p{tag}", f"(i={i}, j={j})：P_ij = exp(S_ij − m_new) —— 就地覆盖 S",
                "op", "exp", "内循环",
                {
                    "sym": r"P_{ij} = \region{SP}{\exp\left(S_{ij} - m_i^{\mathrm{new}}\right)}"
                           r"\ \in \mathbb{R}^{B_r \times B_c}",
                    "idx": r"P_{\slot{II}\slot{JJ}} = \exp\left(S_{\slot{II}\slot{JJ}} - "
                           r"m_{\slot{II}}^{\mathrm{new}}\right)",
                    "num": r"P_{\slot{II}\slot{JJ}}[0,0] = \exp\left(\slot{S00} - \slot{M0}\right)"
                           r" = \slot{P00}",
                },
                [b("II", "i", str(i), str(i)),
                 b("JJ", "j", str(j), str(j)),
                 b("S00", r"S_{ij}[0,0]", r"S_{ij}[0,0]", fmt(float(fr["S"][0][0]))),
                 b("M0", r"m_i^{\mathrm{new}}[0]", r"m_{i}^{\mathrm{new}}[0]",
                   fmt(float(fr["m_new"][0]))),
                 b("P00", r"P_{ij}[0,0]", r"P_{ij}[0,0]", fmt(float(fr["P"][0][0])))],
                ["S", "m_i"], ["P"],
                {"P": mat(fr["P"])},
                f"逐元素减掉本行的新最大值再取指数 —— 因为减过最大值，所有指数都落在 (0, 1]，"
                f"不会溢出。真实 kernel 让 P 就地覆盖 S 的缓冲：两个名字，一块 {br}×{bc} 的 "
                f"SRAM，所以下面的占用账里它们算一份。HBM 访问量依然是 {cost}。",
                {"SP": "P_ij 是 FlashAttention 藏起来的第二个中间矩阵。它和 S_ij 一样，"
                       "整趟回放不会在 HBM 的访问日志里出现。"},
                flows=[{"from": "SRAM", "to": "SRAM", "elements": br * bc,
                        "note": f"P_ij 就地覆盖 S_ij 的 {br}×{bc} 缓冲"}],
                occ=occ_parts(live, br, bc, d),
                io=io_of(cost),
            ))

            # ---- ell <- alpha * ell + rowsum(P) ---------------------------
            cost = fr["cost"]["l"]
            running += cost
            push(step(
                f"l{tag}", f"(i={i}, j={j})：ℓ_i 用修正因子重缩放后并入本块", "op", "sum-exp",
                "内循环",
                {
                    "sym": r"\ell_i^{\mathrm{new}} = \slot{LOLD}\cdot "
                           r"\region{CORR}{\exp\left(m_i - m_i^{\mathrm{new}}\right)} "
                           r"+ \mathrm{rowsum}\left(P_{ij}\right)",
                    "idx": r"\ell_{\slot{II}}^{\mathrm{new}} = \slot{LOLD}\cdot "
                           r"\region{CORR}{\exp\left(\slot{MOLD} - \slot{MNEW}\right)} "
                           r"+ \slot{SUM}",
                    "num": r"\ell_{\slot{II}}^{\mathrm{new}} = \slot{LOLD}\cdot "
                           r"\region{CORR}{\exp\left(\slot{MOLD} - \slot{MNEW}\right)} "
                           r"+ \slot{SUM} = \slot{LNEW}",
                },
                [b("II", "i", str(i), str(i)),
                 b("LOLD", r"\ell_i", r"\ell_{i}", vec_tex(fr["li"])),
                 b("MOLD", r"m_i", r"m_{i}", vec_tex(fr["mi"])),
                 b("MNEW", r"m_i^{\mathrm{new}}", r"m_{i}^{\mathrm{new}}",
                   vec_tex(fr["m_new"])),
                 b("SUM", r"\mathrm{rowsum}(P_{ij})", r"\mathrm{rowsum}(P_{ij})",
                   vec_tex(fr["P"].sum(axis=1))),
                 b("LNEW", r"\ell_i^{\mathrm{new}}", r"\ell_{i}^{\mathrm{new}}",
                   vec_tex(fr["l_new"]))],
                ["P", "m_i", "l_i"], ["l_i"],
                {"l_i": vec(fr["l_new"])},
                (f"本块有 {scaled_rows}/{br} 行的 m 被抬高，这些行之前累加的 ℓ_i 是按旧 m "
                 f"算的，所以先乘修正因子 α = e^(m_old − m_new)（本块最小 {disp(alpha_min)}）"
                 f"把它们拉回来，再加上本块的指数和 "
                 f"{disp(float(np.max(fr['P'].sum(axis=1))))}。"
                 if kind == "rescale" else
                 (f"第一块：ℓ_i 旧值是 0，α 按定义为 0（e^(−∞ − m) = 0），更新化简成直接取"
                  f"本块的指数和 {disp(float(np.max(fr['P'].sum(axis=1))))}。"
                  if kind == "init" else
                  f"m 没动，α = e^0 = 1 —— 旧的 ℓ_i 原样保留，只加上本块的指数和 "
                  f"{disp(float(np.max(fr['P'].sum(axis=1))))}。这一步漏掉 α 不会出错，"
                  f"这也正是「忘了乘」特别难被发现的原因。")),
                regions=corr_regions,
                flows=[{"from": "SRAM", "to": "SRAM", "elements": br,
                        "note": f"{br} 行的指数和在 SRAM 内累加，不跨层"}],
                occ=occ_parts(live, br, bc, d),
                io=io_of(cost),
                corr=cr,
            ))

            # ---- O_i <- (alpha*l_i/l_new) O_i + (P/l_new) V_j -------------
            cost = fr["cost"]["o"]
            running += cost
            push(step(
                f"o{tag}", f"(i={i}, j={j})：O_i 用同一个 α 重缩放后累加 P_ij V_j",
                "op", "matmul", "内循环",
                {
                    "sym": r"O_i^{\mathrm{new}} = \frac{\region{CORR}{\alpha}\,\ell_i}"
                           r"{\ell_i^{\mathrm{new}}}\, O_i + \frac{P_{ij} V_j}"
                           r"{\ell_i^{\mathrm{new}}}",
                    "idx": r"O_{\slot{II}}^{\mathrm{new}} = "
                           r"\frac{\region{CORR}{\alpha}\,\slot{LOLD}}{\slot{LNEW}}\, \slot{OOLD} "
                           r"+ \frac{P_{\slot{II}\slot{JJ}} V_{\slot{JJ}}}{\slot{LNEW}}",
                    "num": r"O_{\slot{II}}^{\mathrm{new}}[0,0] = \slot{O00}",
                },
                [b("II", "i", str(i), str(i)),
                 b("JJ", "j", str(j), str(j)),
                 b("LOLD", r"\ell_i", r"\ell_{i}", vec_tex(fr["li"])),
                 b("LNEW", r"\ell_i^{\mathrm{new}}", r"\ell_{i}^{\mathrm{new}}",
                   vec_tex(fr["l_new"])),
                 b("OOLD", r"O_i[0]", r"O_{i}[0,:]", vec_tex(list(fr["Oi"][0]))),
                 b("O00", r"O_i^{\mathrm{new}}[0,0]", r"O_{i}^{\mathrm{new}}[0,0]",
                   fmt(float(fr["O_new"][0][0])))],
                ["P", "V_j", "O_i", "l_i"], ["O_i"],
                {"O_i": mat(fr["O_new"])},
                (f"输出块用同一组逐行的 α 重缩放：第 0 行历史那部分的系数是 "
                 f"α·ℓ_i/ℓ_i^new = "
                 f"{disp(float(fr['alpha'][0] * fr['li'][0] / fr['l_new'][0]))}，本块的贡献是 "
                 f"P_ij V_j / ℓ_i^new。整趟回放里只有这一步和上一步用到 α —— 它和 L00 的"
                 f"修正因子放大器是同一个量。"
                 if kind == "rescale" else
                 (f"第一块：O_i 从 0 出发，直接取 P_ij V_j / ℓ_i^new。"
                  if kind == "init" else
                  f"α = 1，历史部分原样保留，只叠加本块的 P_ij V_j / ℓ_i^new。")),
                regions=corr_regions,
                flows=[{"from": "SRAM", "to": "SRAM", "elements": br * bc + br * d,
                        "note": f"P_ij V_j 是 SRAM 内的 {br}×{d} 次乘加，不跨层"}],
                occ=occ_parts(live, br, bc, d),
                io=io_of(cost),
                corr=cr,
            ))

            # ---- write O_i, m_i, l_i back ---------------------------------
            cost = fr["cost"]["wb"]
            running += cost
            push(step(
                f"wb{tag}", f"(i={i}, j={j})：把 O_i、m_i、ℓ_i 写回 HBM", "comm", "store",
                "内循环",
                {
                    "sym": r"\mathrm{HBM}[O]\left[\slot{R0}:\slot{R1}\right] \leftarrow O_i,"
                           r"\qquad \mathrm{HBM}[m],\,\mathrm{HBM}[\ell] \leftarrow m_i,\, \ell_i",
                    "idx": r"\mathrm{HBM}[O]\left[\slot{R0}:\slot{R1}\right] \leftarrow "
                           r"O_{\slot{I}}^{\mathrm{new}},\qquad \mathrm{HBM}[m], "
                           r"\mathrm{HBM}[\ell] \leftarrow m_{\slot{I}}^{\mathrm{new}}, "
                           r"\ell_{\slot{I}}^{\mathrm{new}}",
                    "num": r"\mathrm{IO} + \slot{COST},\qquad "
                           r"B_rd + 2B_r = \slot{BR}\cdot\slot{D} + 2\cdot\slot{BR} = \slot{COST}",
                },
                [b("I", "i", str(i), str(i)),
                 b("R0", r"i B_r", str(c0), str(c0)),
                 b("R1", r"(i+1) B_r", str(c1), str(c1)),
                 b("BR", "B_r", str(br), str(br)),
                 b("D", "d", str(d), str(d)),
                 b("COST", r"B_rd + 2B_r", f"{br}\\cdot{d} + 2\\cdot{br}", str(cost))],
                ["O_i", "m_i", "l_i"], ["O", "m", "l"],
                {"O": mat(_O_after(i, j, frames, n)),
                 "m": vec(_row_after(i, j, frames, n, "m_new")),
                 "l": vec(_row_after(i, j, frames, n, "l_new"))},
                f"这一轮算完的东西立刻写回 HBM：O 的第 {c0}..{c1 - 1} 行（{br}×{d}）、m 和 ℓ "
                f"的同一段（各 {br} 个标量），共 {cost} 个元素。"
                + (f"下一个外循环（j={j + 1}）一开始，它们又会被原样读回来 —— 这就是 V1 低效"
                   f"的地方，也是 V2 把循环顺序调过来之后消失的那部分流量。"
                   if j + 1 < tc else
                   "这是最后一个外循环，O 就留在 HBM 里等着收尾。"),
                spans={"O": [[c0, c1], [0, d]], "m": [[c0, c1]], "l": [[c0, c1]]},
                flows=[{"from": "SRAM", "to": "HBM", "elements": cost,
                        "note": f"O_i 的 {br}×{d} 加上 m、ℓ 各 {br} 个，全部写回 HBM"}],
                occ=occ_parts(live, br, bc, d),
                io=io_of(cost),
            ))

    if running != measured_flash["elements"]:
        raise AssertionError(
            f"逐步累加的 IO {running} != 计数器的总数 {measured_flash['elements']} —— "
            f"逐帧数字与总量对不上")

    peak = occ_peak(steps)
    ratio = measured_std["elements"] / measured_flash["elements"]
    slower_hint = (
        f"注意：在这个尺寸上 FA 的 IO 反而更多 —— d={d}、B_c={bc} 时 O(N²d²/M) 的常数还没有"
        f"压过 O(N²)，要 M > 4d² = {4 * d * d} 才翻过来（旁边那条实测曲线把它画出来了）。"
        if measured_flash["elements"] > measured_std["elements"] else
        f"这个配置下 FA 的 IO 已经比标准实现少 —— 分块把 {n}×{n} 的 S、P 从 HBM 里彻底拿掉了。")

    push(step(
        "sum", "收尾：S、P 没去过 HBM，IO 差在这里", "op", "summary", "收尾",
        {
            "sym": r"\mathrm{IO}_{\mathrm{std}} = 4Nd + 4N^2 = \slot{IOSTD},\qquad "
                   r"\mathrm{IO}_{\mathrm{FA}} = Nd + 2N + 2B_cdT_c + "
                   r"(3B_rd + 4B_r)T_rT_c = \slot{IOFA}",
            "idx": r"\mathrm{IO}_{\mathrm{std}} = \slot{IOSTD},\qquad "
                   r"\mathrm{IO}_{\mathrm{FA}} = \slot{IOFA},\qquad "
                   r"\region{SPC}{\mathrm{IO}_{\mathrm{FA}} \times B_c \approx \slot{LAW}}",
            "num": r"\mathrm{IO}_{\mathrm{std}} = \slot{IOSTD},\qquad "
                   r"\mathrm{IO}_{\mathrm{FA}} = \slot{IOFA},\qquad "
                   r"\region{SPC}{S, P \text{ 在 HBM 里出现的次数} = \slot{SPC}}",
        },
        [b("IOSTD", r"4Nd + 4N^2", f"4\\cdot{n}\\cdot{d} + 4\\cdot{n}^2",
           str(measured_std["elements"])),
         b("IOFA", r"Nd + 2N + 2B_cdT_c + (3B_rd+4B_r)T_rT_c",
           f"{predict_flash(n, d, br, bc)}", str(measured_flash["elements"])),
         b("LAW", r"\text{实测常数}", str(law_rows[-1]["times_bc"]),
           str(law_rows[-1]["times_bc"])),
         b("SPC", r"\text{次}", "0", "0")],
        ["Q", "K", "V"], [],
        {},
        f"回放结束。两个实现的输出一致 —— 与 torch 参考实现的最大偏差是 {disp(d_torch)}，"
        f"在设计推算的求和顺序上界 {disp(bound)} 之内（numpy 标准实现与 torch 之间是 "
        f"{disp(d_xcheck)}）。但 HBM 访问量差了 {disp(ratio)} 倍：标准实现 "
        f"{measured_std['elements']} 个元素，FlashAttention {measured_flash['elements']} 个。"
        f"差别全在 S、P 上：标准实现把它们各写一遍、各读一遍，FlashAttention 的访问日志里"
        f"这两个名字共出现 <b>0</b> 次。{slower_hint}",
        {"SPC": "S 和 P 在 FA 的 HBM 访问日志里各出现 0 次。同一个记录对象在标准实现上"
                "记录了它们 —— 对照组见 meta.exposure。"},
        flows=[{"from": "HBM", "to": "SRAM", "elements": measured_flash["elements"],
                "note": f"整趟回放的 HBM 访问总量：{measured_flash['elements']} 个元素"}],
        occ=occ_parts(live, br, bc, d),
        io={"step": 0, "cum": running},
    ))

    nodes = [{"id": s["id"], "title": s["title"], "kind": s["kind"],
              "op": s["op"], "phase": s["phase"]} for s in steps]

    # Data edges only. The K/V load feeds every Q block that consumes it; the
    # write-back feeds the next outer iteration's read of the same rows -- which
    # is V1's defect made into an edge, so a reader can see the cycle.
    edges = [{"from": "init", "to": "kv0", "tensor": "K"}]
    for j in range(tc):
        if j:
            edges.append({"from": f"kv{j - 1}", "to": f"kv{j}", "tensor": "K"})
        for i in range(tr):
            tag = f"{i}{j}"
            edges.append({"from": f"kv{j}", "to": f"s{tag}", "tensor": "K_j"})
            edges.append({"from": f"ld{tag}", "to": f"s{tag}", "tensor": "Q_i"})
            edges.append({"from": f"s{tag}", "to": f"mx{tag}", "tensor": "S"})
            edges.append({"from": f"mx{tag}", "to": f"p{tag}", "tensor": "m_i"})
            edges.append({"from": f"p{tag}", "to": f"l{tag}", "tensor": "P"})
            edges.append({"from": f"l{tag}", "to": f"o{tag}", "tensor": "l_i"})
            edges.append({"from": f"o{tag}", "to": f"wb{tag}", "tensor": "O_i"})
            if i + 1 < tr:
                edges.append({"from": f"wb{tag}", "to": f"ld{i + 1}{j}", "tensor": "O"})
            elif j + 1 < tc:
                edges.append({"from": f"wb{tag}", "to": f"ld0{j + 1}", "tensor": "O"})
            else:
                edges.append({"from": f"wb{tag}", "to": "sum", "tensor": "O"})

    return {
        "meta": {
            "lab": "L06",
            "title": title,
            "source": source,
            "model": {
                "N": n, "d_model": MODEL["d_model"], "H": MODEL["H"], "d_head": d,
                "note": f"贯穿示例模型 N={n}、d_model={MODEL['d_model']}、H={MODEL['H']}、"
                        f"d_head={d}；本 lab 回放第 0 个头，所有形状里的 d 都是 d_head。",
            },
            "config": {"N": n, "d": d, "M": m, "Bc": bc, "Br": br, "Tr": tr, "Tc": tc,
                       "n_blocks": n_pairs, "n_pairs": n_pairs, "n_steps": len(steps)},
            "loop_order": LOOP_ORDER_NOTE,
            "reference": (
                f"FA 输出与 numpy 标准实现最大偏差 {d_std:.3g}、与 torch.softmax 参考 "
                f"{d_torch:.3g}，均在设计推算的求和顺序上界 {bound:.3g} 之内"
                f"（相对容差 {_CHECK_REL_TOL:g}，float64）；漏掉修正因子的影子递推偏差 "
                f"{d_shadow:.3g}，被同一判据抓到；HBM 访问 S/P：FA 0 次、标准实现 "
                f"{std_sp_accesses} 次；IO 标准 {measured_std['elements']} → FA "
                f"{measured_flash['elements']} 个元素"
            ),
            "exposure": {
                "claim": "S 和 P 在整趟回放里从不落到 HBM",
                "forbidden": sorted(forbidden),
                "flash_hbm_tensors": measured_flash["tensors"],
                "standard_hbm_tensors": measured_std["tensors"],
                "flash_forbidden_accesses": fa_sp_accesses,
                "standard_forbidden_accesses": std_sp_accesses,
                "note": "两份数字出自同一个 Mem 记录对象：FlashAttention 与标准 Attention 是"
                        "两段真实的 numpy 代码，只能通过它访问内存。标准实现那份是对照组 ——"
                        "它必须出现 S、P，否则 FA 那边的 0 就没有区分力。",
            },
            "io": {
                "model": "把 HBM 建模成一个只能按块读写的对象，每次读写记一条 (op, 张量, 元素数)。"
                         "标准实现与 FlashAttention 都真的跑在上面，计数是数出来的，不是填公式。",
                "standard": measured_std,
                "flash": measured_flash,
                "predict": {
                    "standard": predict_standard(n, d),
                    "flash": predict_flash(n, d, br, bc),
                    "standard_expr": "4Nd + 4N^2",
                    "flash_expr": "Nd + 2N + 2 B_c d T_c + (3 B_r d + 4 B_r) T_r T_c",
                },
                "asymptotics": {"standard": "O(N^2)", "flash": "O(N^2 d^2 / M)"},
                "ratio": ratio,
                "curves": {
                    "ns": curves["ns"], "standard": curves["standard"],
                    "flash": curves["flash"],
                    "note": "同一份计数模型在 N = "
                            + "、".join(str(x) for x in curves["ns"])
                            + " 上逐点跑出来的曲线：标准实现的读数与 B_c 无关，FA 的读数随 "
                              "B_c 下降 —— 这就是 O(N²d²/M) 里那个 M 的来历。",
                },
                "law": {
                    "statement": f"固定 d={d} 与 N={law_ns[-1]}，实测 IO × B_c 近似恒定",
                    "rows": law_rows,
                    "spread": spread,
                    "note": "把 O(N²d²/M) 落成一句可检验的话：M = 4 B_c d，所以 IO × B_c 应当"
                            "与 B_c 无关。上表是数出来的，误差只在 ⌈·⌉ 的取整上。",
                },
            },
            "sram": {
                "M": m,
                "dominant": 4 * bc * d,
                "used": peak["elements"],
                "peak_step": peak["id"],
                "parts": peak["parts"],
                "formula": "B_c = ceil(M / 4d)",
                "exact": "(2B_r + 2B_c) d + B_r B_c + 2B_r",
                "note": f"教程的公式只约束主导项 4B_c·d = {4 * bc * d}；完整的占用账还要算上"
                        f"S/P 共用的 {br}×{bc} 缓冲与 m、ℓ，峰值 {peak['elements']} 个元素"
                        f"（在 {peak['id']} 步）。",
            },
            "correction": correction,
        },
        "tensors": {
            # HBM. Q, K, V are inputs (no step writes them, so they carry `init`);
            # O, m, l are written by the init step and then updated every inner
            # iteration. That O/m/l live in HBM at all is V1's defect -- V2's diff
            # is exactly that this traffic disappears.
            "Q": {"shape": [n, d], "dtype": "fp32", "at": "HBM", "role": "input",
                  "init": mat(Qa), "note": "Query，本 lab 取第 0 个头，整趟驻留 HBM"},
            "K": {"shape": [n, d], "dtype": "fp32", "at": "HBM", "role": "input",
                  "init": mat(Ka), "note": "Key，外层每个块只读一次"},
            "V": {"shape": [n, d], "dtype": "fp32", "at": "HBM", "role": "input",
                  "init": mat(Va), "note": "Value，与 K 同步载入"},
            "O": {"shape": [n, d], "dtype": "fp32", "at": "HBM", "role": "output",
                  "note": "输出。V1 每轮内层迭代都要写回、下一个外循环再读回来"},
            "m": {"shape": [n], "dtype": "fp32", "at": "HBM", "role": "state",
                  "note": "行最大值，逐行维护，随 O 一起往返 HBM"},
            "l": {"shape": [n], "dtype": "fp32", "at": "HBM", "role": "state",
                  "note": "行指数和，随 O 一起往返 HBM"},
            # SRAM.
            "Q_i": {"shape": [br, d], "dtype": "fp32", "at": "SRAM", "role": "staging",
                    "note": "当前 Q 块，每个外层迭代重新读一次"},
            "K_j": {"shape": [bc, d], "dtype": "fp32", "at": "SRAM", "role": "staging",
                    "note": "当前 K 块"},
            "V_j": {"shape": [bc, d], "dtype": "fp32", "at": "SRAM", "role": "staging",
                    "note": "当前 V 块"},
            "O_i": {"shape": [br, d], "dtype": "fp32", "at": "SRAM", "role": "staging",
                    "note": "累积输出块，算完立刻写回 HBM"},
            # S and P are two names for one buffer: P = exp(S - m_new) is computed
            # in place, which is why the tutorial's SRAM table lists only S_ij.
            # Both are declared, both live in SRAM, and the occupancy account
            # counts them once.
            "S": {"shape": [br, bc], "dtype": "fp32", "at": "SRAM", "role": "staging",
                  "note": "分数块 S_ij = Q_i K_jᵀ/√d；与 P 共用缓冲，全程不落 HBM"},
            "P": {"shape": [br, bc], "dtype": "fp32", "at": "SRAM", "role": "staging",
                  "note": "概率块 P_ij = exp(S_ij − m_new)，就地覆盖 S；全程不落 HBM"},
            "m_i": {"shape": [br], "dtype": "fp32", "at": "SRAM", "role": "state",
                    "note": "当前块的行最大值"},
            "l_i": {"shape": [br], "dtype": "fp32", "at": "SRAM", "role": "state",
                    "note": "当前块的行指数和"},
        },
        "graph": {"nodes": nodes, "edges": edges},
        "steps": steps,
    }


# ---------------------------------------------------------- frame/step helpers
def _frame(frames, i, j):
    return next(f for f in frames if f["i"] == i and f["j"] == j)


def _O_after(i, j, frames, n):
    """The full HBM `O` after this (i, j) pair's write-back.

    The write only touches rows [i*Br, (i+1)*Br), but `state` carries the whole
    tensor -- full values, never deltas, which is what makes the reconstruction a
    pure function of the cursor.
    """
    O = np.zeros((n, frames[0]["O_new"].shape[1]))
    for f in sorted(frames, key=lambda f: (f["j"], f["i"])):
        if (f["j"], f["i"]) <= (j, i):
            O[f["c0"]:f["c1"], :] = f["O_new"]
    return O


def _row_after(i, j, frames, n, key):
    out = np.zeros(n)
    for f in sorted(frames, key=lambda f: (f["j"], f["i"])):
        if (f["j"], f["i"]) <= (j, i):
            out[f["c0"]:f["c1"]] = f[key]
    return out


# -------------------------------------------------------------------- self-lint
SLOT_RE = re.compile(r"\\slot\{([A-Za-z0-9_]+)\}")
REGION_RE = re.compile(r"\\region\{([A-Za-z0-9_]+)\}")
LATEX_IN_TEXT_RE = re.compile(r"\\[a-zA-Z]+\{")
BAD_LITERAL_RE = re.compile(r"^(NaN|Infinity|-Infinity|undefined|null)$")

# The engine's step-level correction contract: exactly these eight keys. See the
# same list in labs/assets/engine/trace-model.js.
CORR_KEYS = {"kind", "factor", "m_old", "m_new", "o_correct", "o_uncorrected",
             "bias_abs", "bias_rel"}
CORR_KINDS = ("init", "identity", "rescale")

# The trace-level `per_block` entries carry a subset: they are a per-block
# summary, not a per-step state, so they have no m_old / m_new to report.
PER_BLOCK_KEYS = {"n", "kind", "factor", "rows_scaled", "o_correct",
                  "o_uncorrected", "bias_abs", "bias_rel"}

SP_ALIAS = SP_BUFFER


def value_ok(v):
    if v is None:
        return True
    if isinstance(v, bool):
        return False
    if isinstance(v, str):
        return v in (NEG_INF, POS_INF, NAN)
    if isinstance(v, int):
        return True
    return isinstance(v, float) and math.isfinite(v)


def resolve_names(trace, upto):
    """Which tensors hold a value after step `upto` -- the same rule the engine's
    `resolve()` uses, re-implemented here so the occupancy account can be checked
    against it rather than against itself."""
    live = set()
    for name, spec in trace["tensors"].items():
        if "init" in spec:
            live.add(name)
    for i in range(upto + 1):
        for key in (trace["steps"][i].get("state") or {}):
            live.add(key)
    return live


def lint(trace):
    """Check the trace against the contract the engine and the page consume.

    The author-side half; `labs/assets/engine/trace-model.js` carries a JS port of
    the ENGINE's rules and `labs/assets/engine/views/tiling-stage.js` a port of
    the STAGE's (`spans` / `flows` / `occ`, the three fields this lab's view
    reads). All must stay in step; the acceptance harness runs one sabotage table
    through all of them.
    """
    gaps, warns, infos = [], [], []
    declared = set(trace["tensors"])
    step_ids = {s["id"] for s in trace["steps"]}
    cfg = trace["meta"]["config"]
    n = cfg["N"]

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

    node_ids = {nd["id"] for nd in trace["graph"]["nodes"]}
    for s in trace["steps"]:
        if s["id"] not in node_ids:
            gaps.append(f'步骤 "{s["id"]}" 在 graph.nodes 里没有对应节点')
    for nd in trace["graph"]["nodes"]:
        if nd["id"] not in step_ids:
            warns.append(f'图节点 "{nd["id"]}" 没有对应步骤，点击无处可跳')

    # The layer vocabulary. Every `at` and every flow endpoint must be one of the
    # two the page's stage is configured with, or a rail is drawn to nowhere.
    layers = []
    for spec in trace["tensors"].values():
        at = spec.get("at")
        if at and at not in layers:
            layers.append(at)
    if sorted(layers) != ["HBM", "SRAM"]:
        gaps.append(f"张量声明用到的层是 {layers} —— 这份 trace 只该用 HBM 与 SRAM 两层")

    # The headline claim, as a lint rule: S and P must be declared, and declared
    # in SRAM. A trace that moved either to HBM would still render, and would
    # still be wrong.
    for name in ("S", "P"):
        spec = trace["tensors"].get(name)
        if spec is None:
            gaps.append(f'张量 "{name}" 没有声明 —— 「S、P 不落回 HBM」这条主张'
                        f"需要它在 trace 里可见（且 at=SRAM）")
        elif spec.get("at") != "SRAM":
            gaps.append(f'张量 "{name}" 的 at = {spec.get("at")!r} 而不是 "SRAM" —— '
                        f"这是整个 lab 的核心主张，不能靠别处的文字补偿")

    sram_names = {t for t, sp in trace["tensors"].items() if sp.get("at") == "SRAM"}
    shape_by = {
        "Q_i": (cfg["Br"], cfg["d"]), "K_j": (cfg["Bc"], cfg["d"]),
        "V_j": (cfg["Bc"], cfg["d"]), "O_i": (cfg["Br"], cfg["d"]),
        "m_i": (cfg["Br"],), "l_i": (cfg["Br"],),
        SP_ALIAS: (cfg["Br"], cfg["Bc"]),
    }

    last_io = 0
    for idx, s in enumerate(trace["steps"]):
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
            for tier in ("sym", "idx", "num"):
                tex = bdg.get(tier)
                if tex is not None and (SLOT_RE.search(str(tex)) or REGION_RE.search(str(tex))):
                    gaps.append(f'步骤 "{sid}" 的绑定 "{key}".{tier} 里含 \\slot / \\region —— '
                                f"这两个宏只在 formula 串里展开")

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

        # ---- the correction factor: exactly the engine's eight keys, and the
        # kind must agree with the factor it reports.
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
                gaps.append(f'步骤 "{sid}" 带了 corr，却没有 \\region{{CORR}} 的说明')
            for key in ("factor", "m_old", "m_new", "o_correct", "o_uncorrected",
                        "bias_abs", "bias_rel"):
                if not value_ok(corr.get(key)):
                    gaps.append(f'步骤 "{sid}" 的 corr.{key} = {corr.get(key)!r} '
                                f"不是数、null 或哨兵字符串")
            if corr.get("kind") == "identity" and corr.get("factor") != 1.0:
                gaps.append(f'步骤 "{sid}" 的 corr.kind 是 identity，但因子是 '
                            f'{corr.get("factor")!r} 而不是 1')
            if corr.get("kind") == "rescale" and not (
                    isinstance(corr.get("factor"), float) and corr["factor"] < 1.0):
                gaps.append(f'步骤 "{sid}" 的 corr.kind 是 rescale，因子 '
                            f'{corr.get("factor")!r} 却不小于 1')

        # ---- spans: the sub-region of an HBM tensor this step moves ---------
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

        # ---- flows: layer boundaries, so both ends must be real layers ------
        for fl in (s.get("flows") or []):
            for end in ("from", "to"):
                if fl.get(end) not in layers:
                    gaps.append(f'步骤 "{sid}" 的 flows 有一端的层 "{fl.get(end)}" '
                                f"不在 tensors[].at 用到的层里")
            if not isinstance(fl.get("elements"), int) or fl["elements"] <= 0:
                gaps.append(f'步骤 "{sid}" 的 flows 搬运量为 {fl.get("elements")!r}，'
                            f"应为正整数")

        # ---- io: the running counter is the sum of the per-step deltas ------
        io = s.get("io")
        if not isinstance(io, dict) or set(io) != {"step", "cum"}:
            gaps.append(f'步骤 "{sid}" 的 io 字段是 '
                        f'{sorted(io) if isinstance(io, dict) else io!r}，'
                        f"而不是 {{step, cum}}")
        else:
            if not isinstance(io["step"], int) or io["step"] < 0:
                gaps.append(f'步骤 "{sid}" 的 io.step = {io["step"]!r} 应为非负整数')
            if io["cum"] < last_io:
                gaps.append(f'步骤 "{sid}" 的 io.cum = {io["cum"]} 比上一步的 {last_io} 小 —— '
                            f"累计访问量只能单调不减")
            elif io["cum"] != last_io + io["step"]:
                gaps.append(f'步骤 "{sid}" 的 io.cum = {io["cum"]} 与上一步的 {last_io} '
                            f"加 io.step {io['step']} 对不上")
            last_io = io["cum"]

        # ---- occ: the SRAM account must name exactly what is resident -------
        occ = s.get("occ")
        if occ is None:
            gaps.append(f'步骤 "{sid}" 没有 occ —— SRAM 占用账缺这一步')
            continue
        if set(occ) != {"elements", "parts"}:
            gaps.append(f'步骤 "{sid}" 的 occ 字段是 {sorted(occ)}，而不是 {{elements, parts}}')
            continue
        live_sram = resolve_names(trace, idx) & sram_names
        covered, total, sp_hit = set(), 0, False
        for part in occ.get("parts") or []:
            name = part.get("name")
            if name in shape_by and name != SP_ALIAS:
                covered.add(name)
                if part.get("covers"):
                    gaps.append(f'步骤 "{sid}" 的 occ["{name}"] 带 covers —— '
                                f"一个张量自己就是一个分配，没有可合并的对象")
            elif name not in shape_by and not part.get("covers"):
                gaps.append(f'步骤 "{sid}" 的 occ 里有一项 "{name}"，既不是 SRAM 张量，'
                            f'也没有声明它 covers 了哪些张量')
                continue
            else:
                # The merged buffer: covers says which tensors are on it, and the
                # rule is that those are exactly the ones currently resident --
                # not both regardless. P is allocated (in place, over S) one step
                # after S is, and the account has to be able to show that.
                sp_hit = name in shape_by
                covers = set(part["covers"])
                if not covers <= sram_names:
                    gaps.append(f'步骤 "{sid}" 的 occ["{name}"].covers = '
                                f'{sorted(covers)} 里有非 SRAM 张量')
                covered |= covers
                expect_covers = sorted(live_sram & covers)
                if sorted(covers) != expect_covers:
                    gaps.append(f'步骤 "{sid}" 的 occ["{name}"].covers = '
                                f'{sorted(covers)}，而此刻驻留 SRAM 的是 {expect_covers}')
            expect = list(shape_by[name])
            if list(part.get("shape") or []) != expect:
                gaps.append(f'步骤 "{sid}" 的 occ["{name}"].shape = '
                            f'{part.get("shape")!r} 与配置推出的 {expect} 不一致')
            want = 1
            for s_ in expect:
                want *= s_
            if part.get("elements") != want:
                gaps.append(f'步骤 "{sid}" 的 occ["{name}"].elements = '
                            f'{part.get("elements")!r} 应为 {want}')
            total += part.get("elements") or 0
        if sp_hit and not (live_sram & {"S", "P"}):
            gaps.append(f'步骤 "{sid}" 的 occ 算了共享缓冲，但 S、P 此刻都不在 SRAM 里')
        if covered != live_sram:
            gaps.append(f'步骤 "{sid}" 的 occ 覆盖了 {sorted(covered)}，'
                        f"而此刻驻留 SRAM 的是 {sorted(live_sram)}")
        if total != occ.get("elements"):
            gaps.append(f'步骤 "{sid}" 的 occ.elements = {occ.get("elements")!r} '
                        f"与各项之和不符（{total}）")

    # ---- meta.exposure: the claim and its control must both be published ----
    ex = trace["meta"].get("exposure")
    if not isinstance(ex, dict):
        gaps.append("meta.exposure 缺失 —— 「S、P 不落回 HBM」的实测证据没有随 trace 发布")
    else:
        for key in ("forbidden", "flash_hbm_tensors", "standard_hbm_tensors",
                    "flash_forbidden_accesses", "standard_forbidden_accesses"):
            if key not in ex:
                gaps.append(f"meta.exposure 缺 {key}")
        if ex.get("flash_forbidden_accesses") != 0:
            gaps.append(f"meta.exposure.flash_forbidden_accesses = "
                        f"{ex.get('flash_forbidden_accesses')!r}，应为 0")
        if not ex.get("standard_forbidden_accesses"):
            gaps.append("meta.exposure.standard_forbidden_accesses 为 0 —— "
                        "对照组失效，FA 那边的 0 就没有区分力")
        for name in ex.get("forbidden") or []:
            if name not in (ex.get("standard_hbm_tensors") or []):
                gaps.append(f"meta.exposure.standard_hbm_tensors 里没有 {name} —— "
                            f"对照组本该记录它")

    # ---- meta.io: measured and predicted must agree, and the law must hold ---
    io = trace["meta"].get("io")
    if not isinstance(io, dict):
        gaps.append("meta.io 缺失 —— IO 计数器没有数据")
    else:
        std = io.get("standard") or {}
        fla = io.get("flash") or {}
        pred = io.get("predict") or {}
        if std.get("elements") != pred.get("standard"):
            gaps.append(f"meta.io.standard.elements = {std.get('elements')!r} "
                        f"与预测 {pred.get('standard')!r} 不一致 —— 实测与公式脱节")
        if fla.get("elements") != pred.get("flash"):
            gaps.append(f"meta.io.flash.elements = {fla.get('elements')!r} "
                        f"与预测 {pred.get('flash')!r} 不一致 —— 实测与公式脱节")
        if not std.get("elements") or not fla.get("elements"):
            gaps.append("meta.io 的 standard / flash 元素数为 0")
        curves = io.get("curves") or {}
        for key in ("ns", "standard", "flash"):
            if key not in curves:
                gaps.append(f"meta.io.curves 缺 {key}")
        if not (curves.get("standard") and curves.get("flash")):
            gaps.append("meta.io.curves 的曲线为空 —— 对比图画不出来")
        law = io.get("law") or {}
        rows = law.get("rows") or []
        if len(rows) < 2:
            gaps.append(f"meta.io.law.rows 只有 {len(rows)} 行 —— "
                        f"1/B_c 的说法至少要两个不同块大小")
        else:
            times = [r.get("times_bc") for r in rows]
            if any((not isinstance(t, (int, float)) or t <= 0) for t in times):
                gaps.append(f"meta.io.law.rows 的 times_bc 非正：{times}")
            elif max(times) / min(times) - 1 > 0.35:
                gaps.append(f"meta.io.law.rows 的 IO×B_c 跨度 "
                            f"{max(times) / min(times) - 1:.3f} 过大 —— "
                            f"O(N^2 d^2 / M) 的常数说法不成立")

    # ---- meta.sram: the account's peak must be one of the per-step accounts --
    sram = trace["meta"].get("sram")
    if not isinstance(sram, dict):
        gaps.append("meta.sram 缺失 —— 占用账的峰值没有随 trace 发布")
    else:
        for key in ("M", "dominant", "used", "peak_step", "parts"):
            if key not in sram:
                gaps.append(f"meta.sram 缺 {key}")
        peak = occ_peak(trace["steps"])
        if sram.get("used") != peak["elements"]:
            gaps.append(f"meta.sram.used = {sram.get('used')!r} 与逐帧峰值 "
                        f"{peak['elements']} 不一致")
        if sram.get("peak_step") != peak["id"]:
            gaps.append(f"meta.sram.peak_step = {sram.get('peak_step')!r}，"
                        f"而最占 SRAM 的一步是 {peak['id']!r}")
        if sram.get("dominant") != 4 * cfg["Bc"] * cfg["d"]:
            gaps.append(f"meta.sram.dominant = {sram.get('dominant')!r} 应为 "
                        f"4·B_c·d = {4 * cfg['Bc'] * cfg['d']}")
        if sram.get("M") != trace["meta"]["config"]["M"]:
            gaps.append(f"meta.sram.M = {sram.get('M')!r} 与 config.M = "
                        f"{trace['meta']['config']['M']!r} 不一致")
        if not sram.get("parts"):
            gaps.append("meta.sram.parts 为空 —— 占用账画不出分解")

    # ---- meta.correction: L00's amplifier summary, on this recurrence -------
    mc = trace["meta"].get("correction")
    if not isinstance(mc, dict):
        gaps.append("meta.correction 缺失 —— 修正因子放大器报不出结论")
    else:
        for key in ("final_o_correct", "final_o_uncorrected", "final_bias_abs",
                    "final_bias_rel", "per_block", "rescales"):
            if key not in mc:
                gaps.append(f"meta.correction 缺 {key}")
        per_block = mc.get("per_block")
        if not isinstance(per_block, list) or len(per_block) != cfg["n_blocks"]:
            gaps.append(f"meta.correction.per_block 有 "
                        f"{len(per_block) if isinstance(per_block, list) else '?'} 项，"
                        f"应该是 n_blocks = {cfg['n_blocks']} 项")
        else:
            for entry in per_block:
                if set(entry) != PER_BLOCK_KEYS:
                    gaps.append(f"meta.correction.per_block 里一项的字段是 "
                                f"{sorted(entry)}，而不是 {sorted(PER_BLOCK_KEYS)}")
                    break
                if entry["kind"] not in CORR_KINDS:
                    gaps.append(f"meta.correction.per_block 里 kind = {entry['kind']!r} 非法")
                    break
            stepped = sum(1 for e in per_block if e.get("kind") == "rescale")
            if mc.get("rescales") != stepped:
                gaps.append(f"meta.correction.rescales = {mc.get('rescales')!r} "
                            f"与 per_block 里 rescale 的项数 {stepped} 不一致")
        if not mc.get("final_bias_abs") or not mc.get("final_bias_rel"):
            gaps.append("meta.correction 的最终偏差为 0 —— 「漏掉修正因子」本该有可测的代价，"
                        "否则 L00 的放大器无从谈起")

    # ---- config arithmetic: the grid and the plan must agree ----------------
    if cfg["Br"] != min(cfg["Bc"], cfg["d"]):
        gaps.append(f"config.Br = {cfg['Br']} 应为 min(B_c, d) = {min(cfg['Bc'], cfg['d'])}")
    if cfg["Tr"] * cfg["Br"] != n or cfg["Tc"] * cfg["Bc"] != n:
        gaps.append(f"T_r B_r = {cfg['Tr'] * cfg['Br']}、T_c B_c = {cfg['Tc'] * cfg['Bc']}，"
                    f"应都等于 N = {n}")
    if cfg["n_blocks"] != cfg["Tr"] * cfg["Tc"]:
        gaps.append(f"config.n_blocks = {cfg['n_blocks']} 应为 T_r × T_c = "
                    f"{cfg['Tr'] * cfg['Tc']}")
    if len(trace["steps"]) != cfg["n_steps"]:
        gaps.append(f"config.n_steps = {cfg['n_steps']} 与实际步数 {len(trace['steps'])} 不符")

    infos.append(f'共 {len(trace["steps"])} 步 / {len(trace["graph"]["nodes"])} 节点 / '
                 f'{len(trace["graph"]["edges"])} 数据边 / '
                 f'{mc.get("rescales") if isinstance(mc, dict) else "?"} 次修正')
    infos.append(f'SRAM 峰值 {sram.get("used") if isinstance(sram, dict) else "?"} 个元素'
                 f'（公式主导项 {4 * cfg["Bc"] * cfg["d"]}）· '
                 f'IO 标准 {io.get("standard", {}).get("elements") if isinstance(io, dict) else "?"}'
                 f' → FA {io.get("flash", {}).get("elements") if isinstance(io, dict) else "?"}')
    infos.append(f'state 写过的张量：{", ".join(sorted(written))}')
    return gaps, warns, infos


# Each entry breaks one lint rule on a real trace. Kept at module scope so the
# report can name how many were exercised, and so the parity harness can hand the
# same table to the JS ports.
#
# Every mutation must genuinely change the trace -- a no-op proves nothing on
# either side, and `sabotage_meta_checks` is what rules that out.
SABOTAGE_CASES = {
    # --- the engine's contract (trace-model.js ports all of these) -----------
    "read-but-never-written tensor with no init": lambda t: t["tensors"].pop("K"),
    "state entry for an undeclared tensor": lambda t: t["steps"][2]["state"].update(
        {"nope": 1.0}),
    "write of an undeclared tensor": lambda t: t["steps"][2]["writes"].append("ghost"),
    "binding missing a tier": lambda t: t["steps"][2]["bindings"].update({"MOLD": {"sym": "a"}}),
    "binding collapsed to a 2-tuple": lambda t: t["steps"][2]["bindings"].update(
        {"MNEW": {"num": "0.83"}}),
    "\\slot with no binding": lambda t: t["steps"][2]["formula"].update(
        {"num": t["steps"][2]["formula"]["num"] + r"\slot{GHOST}"}),
    "\\region with no description": lambda t: t["steps"][3]["formula"].update(
        {"sym": t["steps"][3]["formula"]["sym"] + r"\region{GHOST}{x}"}),
    "step with no graph node": lambda t: t["graph"]["nodes"].pop(0),
    "formula tier missing": lambda t: t["steps"][0]["formula"].pop("idx"),
    "unknown sentinel string": lambda t: t["steps"][0]["state"].update({"m": "-inf"}),
    "raw -Infinity instead of the sentinel": lambda t: t["steps"][0]["state"].update(
        {"m": "-Infinity"}),
    # --- the correction summary (trace-model.js ports these too) -------------
    "corr attached to a step with no \\region{CORR}": lambda t: t["steps"][2].update(
        {"corr": _any_corr_step(t)["corr"]}),
    "corr.kind outside the vocabulary": lambda t: _any_corr_step(t)["corr"].update(
        {"kind": "maybe"}),
    "corr claims rescale with a factor above 1": lambda t: _corr_of_kind(
        t, "rescale")["corr"].update({"factor": 1.4}),
    "corr missing a field": lambda t: _any_corr_step(t)["corr"].pop("o_uncorrected"),
    "corr.o_uncorrected is a bare NaN": lambda t: _any_corr_step(t)["corr"].update(
        {"o_uncorrected": float("nan")}),
    "meta.correction missing": lambda t: t["meta"].pop("correction"),
    "meta.correction.per_block out of step with blocks": lambda t: t["meta"][
        "correction"].update({"per_block": t["meta"]["correction"]["per_block"][:-1]}),
    "meta.correction.rescales miscounted": lambda t: t["meta"]["correction"].update(
        {"rescales": 99}),
    "meta.correction with a zero final bias": lambda t: t["meta"]["correction"].update(
        {"final_bias_abs": 0.0}),
    # --- the stage's contract (tiling-stage.js ports spans / flows / occ) ----
    "span names an undeclared tensor": lambda t: _kv_step(t).update(
        {"spans": {"GHOST": [[0, 1], [0, 1]]}}),
    "span out of the tensor's shape": lambda t: _kv_step(t).update(
        {"spans": {"K": [[0, 1], [0, 999]]}}),
    "span with the wrong rank": lambda t: _kv_step(t).update({"spans": {"K": [[0, 1]]}}),
    "flow naming an unknown layer": lambda t: _kv_step(t).update(
        {"flows": [{"from": "HBM", "to": "L2CACHE", "elements": 4}]}),
    "flow with a non-positive element count": lambda t: _kv_step(t).update(
        {"flows": [{"from": "HBM", "to": "SRAM", "elements": 0}]}),
    "occ naming an undeclared tensor": lambda t: _peak_step(t)["occ"]["parts"].append(
        {"name": "GHOST", "label": "x", "shape": [1], "elements": 1}),
    "occ missing a resident block": lambda t: _peak_step(t)["occ"]["parts"].pop(),
    "occ total not the sum of its parts": lambda t: _peak_step(t)["occ"].update(
        {"elements": _peak_step(t)["occ"]["elements"] + 1}),
    "occ part shape not matching the config": lambda t: _peak_step(t)[
        "occ"]["parts"][1].update({"shape": [99, 99]}),
    "occ counting the shared buffer twice": lambda t: _peak_step(t)["occ"]["parts"].append(
        dict(_sp_part(t), label="dup")),
    "occ sharing a buffer that is not resident yet": lambda t: _step_by_id(
        t, "s00")["occ"]["parts"].append(dict(_sp_part(t), label="premature")),
    "occ claiming a plain tensor shares a buffer": lambda t: _peak_step(t)["occ"][
        "parts"][0].update({"covers": ["S", "P"]}),
    # --- this lab's own fields ----------------------------------------------
    "S moved out of SRAM": lambda t: t["tensors"]["S"].update({"at": "HBM"}),
    "P removed from the trace": lambda t: t["tensors"].pop("P"),
    "a third layer in use": lambda t: t["tensors"]["S"].update({"at": "SMEM"}),
    "meta.exposure missing": lambda t: t["meta"].pop("exposure"),
    "meta.exposure claims FA touched S": lambda t: t["meta"]["exposure"].update(
        {"flash_forbidden_accesses": 1}),
    "meta.exposure without a control group": lambda t: t["meta"]["exposure"].update(
        {"standard_forbidden_accesses": 0}),
    "meta.exposure's control never saw S": lambda t: t["meta"]["exposure"].update(
        {"standard_hbm_tensors": [x for x in t["meta"]["exposure"]["standard_hbm_tensors"]
                                  if x != "S"]}),
    "meta.io missing": lambda t: t["meta"].pop("io"),
    "meta.io measured disagreeing with the prediction": lambda t: t["meta"]["io"][
        "flash"].update({"elements": t["meta"]["io"]["flash"]["elements"] + 8}),
    "meta.io.law collapsed to one block size": lambda t: t["meta"]["io"]["law"].update(
        {"rows": t["meta"]["io"]["law"]["rows"][:1]}),
    "meta.io.law with a non-constant product": lambda t: t["meta"]["io"]["law"][
        "rows"][-1].update(
        {"times_bc": t["meta"]["io"]["law"]["rows"][-1]["times_bc"] * 4}),
    "meta.io.curves emptied": lambda t: t["meta"]["io"]["curves"].update({"flash": []}),
    "meta.sram missing": lambda t: t["meta"].pop("sram"),
    "meta.sram peak out of step with the steps": lambda t: t["meta"]["sram"].update(
        {"used": t["meta"]["sram"]["used"] + 1}),
    "meta.sram peak_step pointing at the wrong step": lambda t: t["meta"]["sram"].update(
        {"peak_step": "init"}),
    "meta.sram dominant term wrong": lambda t: t["meta"]["sram"].update({"dominant": 1}),
    "step without an io counter": lambda t: t["steps"][2].pop("io"),
    "io.cum not the running sum": lambda t: t["steps"][4]["io"].update(
        {"cum": t["steps"][4]["io"]["cum"] + 5}),
    "io.cum going backwards": lambda t: t["steps"][4]["io"].update({"cum": -1}),
    "step without an occ account": lambda t: t["steps"][2].pop("occ"),
    "config.Br not min(B_c, d)": lambda t: t["meta"]["config"].update({"Br": 99}),
    "config.n_blocks not T_r x T_c": lambda t: t["meta"]["config"].update({"n_blocks": 3}),
    "config.n_steps out of step with the steps": lambda t: t["meta"]["config"].update(
        {"n_steps": 7}),
}


def _kv_step(t):
    return next(s for s in t["steps"] if s["id"].startswith("kv"))


def _step_by_id(t, sid):
    return next(s for s in t["steps"] if s["id"] == sid)


def _sp_part(t):
    """The merged S/P part, as the trace itself states it -- copied rather than
    reconstructed, so a sabotage case cannot silently stop matching the trace."""
    cfg = t["meta"]["config"]
    return next(p for p in _peak_step(t)["occ"]["parts"] if p.get("covers"))


def _peak_step(t):
    """The step with the widest SRAM account, so popping or adding a part is
    always a visible change."""
    return next(s for s in t["steps"] if s["id"] == t["meta"]["sram"]["peak_step"])


def _any_corr_step(t):
    return next(s for s in t["steps"] if s.get("corr"))


def _corr_of_kind(t, kind):
    found = [s for s in t["steps"] if s.get("corr") and s["corr"]["kind"] == kind]
    return found[0] if found else _any_corr_step(t)


def sabotage_meta_checks(trace):
    """Confirm every mutation actually changes the trace.

    A mutation that lands on a no-op -- because the case was written against a
    literal this configuration happens to share -- proves nothing when the lint
    reports 0 gaps on both sides. Reported separately from "the lint missed it",
    because they are different bugs with different fixes.
    """
    base = json.dumps(trace, sort_keys=True)
    failures = []
    for name, mutate in SABOTAGE_CASES.items():
        broken = copy.deepcopy(trace)
        try:
            mutate(broken)
        except Exception as exc:
            failures.append(f"{name}: 破坏无法施加（{exc!r}）")
            continue
        if json.dumps(broken, sort_keys=True) == base:
            failures.append(f"{name}: 破坏后 trace 没有变化 —— 空操作")
    return failures


def sabotage_checks(trace):
    """Prove the lint is not a function that always returns zero."""
    failures = []
    for name, mutate in SABOTAGE_CASES.items():
        broken = copy.deepcopy(trace)
        try:
            mutate(broken)
        except Exception as exc:
            failures.append(f"{name}: lint raised {exc!r}")
            continue
        if not lint(broken)[0]:
            failures.append(f"{name}: broke the trace but lint reported 0 gaps")
    return failures


# ------------------------------------------------------------------------ main
# The parameter grid the page's sliders walk. M is the primary control -- it is
# what the tutorial's formula takes -- and B_c, B_r, T_r, T_c all follow from it.
# A combination is generated only when the derived plan is one the replay can
# actually show; the page reports the empty cells, with the reason, rather than
# pretending.
NS = [8, 16]
DS = [8, 16]
MS = [128, 256, 512, 1024]

# How many (i, j) tile pairs a replay may have. Seven steps per pair plus the
# outer loads, so 16 pairs is a ~119-step timeline -- long, but it is the only
# way the M slider gets two working positions per (N, d): `B_c = N` is a single
# block (no blocking left to watch) and `B_c = N/4` is 16 pairs, so a coarser cap
# would leave every (N, d) with exactly one usable M and nothing to slide.
#
# The configurations beyond the cap are exactly the "M so small the blocks are
# tiny" ones. Their story -- IO climbs as B_c shrinks -- is told by the measured
# curve, which runs at several N without being replayed as a timeline.
MAX_PAIRS = 16

# The curve's N axis. The crossover between the two IO curves happens around
# N ~ 4d^2/B_c * ... for the larger B_c, and several of these points sit below it
# on purpose: a curve that only ever showed FlashAttention winning would be a
# worse teaching artifact than one that shows where the win starts.
CURVE_NS = [8, 16, 32, 64, 128]

SOURCE = "docs/guides/模块二-cuda编程与算子优化/6.1-FlashAttention V1详解.md"


def all_cells():
    return [(n, d, m) for n in NS for d in DS for m in MS]


def replayable_plan(n, d, m):
    """The plan, or None when this cell is not one the page replays.

    Three ways a cell drops out, all reported to the reader by `skip_reason`
    rather than silently rounded into something else:

      * `T_c == 1` -- the whole sequence fits in one K/V block, so there is no
        outer loop and (more to the point) the row maximum never rises after the
        first block: the correction factor, which is this lab's link back to L00,
        would never fire. A single block is not a smaller replay of the algorithm,
        it is a different one.
      * more than `MAX_PAIRS` tile pairs -- a timeline nobody would scrub through.
      * `block_plan` refusing it at all (see there).
    """
    plan = block_plan(n, d, m)
    if plan is None or plan["Tc"] < 2:
        return None
    if plan["Tr"] * plan["Tc"] > MAX_PAIRS:
        return None
    return plan


def grid():
    """Every (N, d, M) the page offers, minus the cells the replay cannot show."""
    return [cell for cell in all_cells() if replayable_plan(*cell)]


def skip_reason(n, d, m):
    plan = block_plan(n, d, m)
    if plan is None:
        bc = math.ceil(m / (4 * d))
        if bc > n:
            return (f"B_c = ⌈{m}/(4×{d})⌉ = {bc} 比 N = {n} 还大 —— 一块就装下全部，"
                    f"回放里没有分块过程可看")
        return (f"B_c = ⌈{m}/(4×{d})⌉ = {bc} 不能整除 N = {n} —— 分块会留下不满的尾巴，"
                f"那是另一套 mask 机制，本 lab 不做近似")
    if plan["Tc"] < 2:
        return (f"B_c = {plan['Bc']} 时整个序列只够 {plan['Tc']} 个 K/V 块 —— "
                f"没有外循环，行最大值也不会再被抬高，修正因子永远不会触发")
    pairs = plan["Tr"] * plan["Tc"]
    return (f"B_c = {plan['Bc']} 太小，要 {pairs} 个 tile 对"
            f"（约 {pairs * 7 + plan['Tc'] + 2} 步）—— 本页只回放 ≤ {MAX_PAIRS} 对的组合；"
            f"它的 IO 由旁边的实测曲线覆盖")


def skipped():
    return {cfg_id(n, d, m): skip_reason(n, d, m)
            for (n, d, m) in all_cells() if replayable_plan(n, d, m) is None}


def cfg_id(n, d, m):
    return f"flash-attention-N{n}-d{d}-M{m}"


# Non-finite floats in the parity payload, as strings the JS probe turns back
# into the real thing.
#
# The table below deliberately includes sabotages that put a bare `NaN` or
# `Infinity` into the trace -- "the lint rejects it" is a rule, and a rule with no
# case against it is a rule tested on neither side. But `json.dumps` writes those
# as the bare tokens `NaN` / `Infinity`, which are legal JavaScript and ILLEGAL
# JSON, so a payload containing one cannot be parsed by the probe at all: the
# harness would fail before either lint ran, and would read as a harness bug
# rather than as the very thing the case is about. Tagging them keeps the payload
# valid while delivering the same value; `--js-lint`'s consumer decodes
# `{"__nonfinite__": "nan"}` back into `NaN`.
_NONFINITE_TAGS = {
    float("nan"): "nan", float("inf"): "inf", float("-inf"): "-inf",
}


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
    for (n, d, m) in grid():
        name = cfg_id(n, d, m)
        title = f"FlashAttention V1（N={n}，d_head={d}，SRAM M={m} 个元素）"
        trace = build_trace(n, d, m, title, SOURCE)
        gaps, warns, infos = lint(trace)

        # Order matters: a dead mutation also reports "0 gaps" below, and the two
        # are different bugs with different fixes.
        dead = sabotage_meta_checks(trace)
        sabotages = [] if dead else sabotage_checks(trace)

        cfg = trace["meta"]["config"]
        io = trace["meta"]["io"]
        sram = trace["meta"]["sram"]
        mc = trace["meta"]["correction"]
        exposure = trace["meta"]["exposure"]
        print(f"\n{name}: {len(trace['steps'])} 步 · "
              f"{len(json.dumps(trace, ensure_ascii=False))} bytes")
        print(f"  B_c={cfg['Bc']} B_r={cfg['Br']} T_r={cfg['Tr']} T_c={cfg['Tc']} · "
              f"SRAM 峰值 {sram['used']} 个元素（公式主导项 {sram['dominant']}）")
        print(f"  IO 标准 {io['standard']['elements']} → FA {io['flash']['elements']}"
              f"（{io['ratio']:.3f}×）· 实测与预测一致")
        print(f"  S/P 的 HBM 访问：FA {exposure['flash_forbidden_accesses']} 次 · "
              f"标准实现 {exposure['standard_forbidden_accesses']} 次（对照组）")
        print(f"  修正因子生效 {mc['rescales']}/{cfg['n_blocks']} 对 · "
              f"漏掉它的最终偏差 {mc['final_bias_abs']:.4g}"
              f"（相对 {pct(mc['final_bias_rel'])}）")
        print("  IO×B_c @N={}: ".format(io["law"]["rows"][-1]["N"])
              + " · ".join(f"B_c={r['Bc']}→{r['times_bc']:.4g}" for r in io["law"]["rows"])
              + f"（跨度 {io['law']['spread']:.3f}）")
        print(f"  {trace['meta']['reference']}")
        for line in infos:
            print(f"  info  {line}")
        for line in warns:
            print(f"  warn  {line}")
        for line in gaps:
            print(f"  GAP   {line}")
            fails.append(f"{name}: {line}")
        if dead:
            fails.append(f"{name}: 有破坏用例是空操作")
            for line in dead:
                print(f"  SABOTAGE-DEAD  {line}")
        elif sabotages:
            fails.append(f"{name}: lint 对照组失效")
            for line in sabotages:
                print(f"  LINT-DEAD  {line}")
        else:
            print(f"  lint: {len(gaps)} gap / {len(warns)} warn；"
                  f"对照 {len(SABOTAGE_CASES)} 种破坏：逐个确认改变了 trace，"
                  "且全部被抓到（Python 侧）")
        built[name] = trace

    if only_json:
        # `base` names which trace the sabotages were cut from, so the JS side can
        # tell "the lint missed this" from "the mutation changed nothing".
        base_name = list(built)[0]
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

    skip = skipped()
    manifest = {
        "set": "flash-attention",
        "lab": "L06",
        "default": cfg_id(8, 16, 256),
        "params": {"N": NS, "d": DS, "M": MS},
        "traces": written,
        "labels": {cfg_id(n, d, m): f"N={n} · d={d} · M={m}" for (n, d, m) in grid()},
        "skipped": skip,
        "notes": (
            "M 是 SRAM 容量（元素个数），B_c / B_r / T_r / T_c 全由它按教程的公式推出："
            "B_c = ⌈M/4d⌉、B_r = min(B_c, d)。滑杆上的每一格合法组合都有一份 trace；"
            "skipped 里列的是不做回放的格子与原因（块比序列还大、块不能整除 N，"
            "或块太小以致回放步数过多）。"
        ),
        # The curve set depends only on `d`, but is carried in every trace's meta
        # (it is what the page renders next to the trace it is showing); the
        # manifest repeats it once so a page can draw the comparison before any
        # single trace is picked.
        "curves": built[written[0]]["meta"]["io"]["curves"],
    }
    (HERE / "flash-attention.manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")

    total = sum((HERE / f"{n}.json").stat().st_size for n in written)
    print(f"\nflash-attention.manifest.json: {len(written)} 个配置, "
          f"默认 {manifest['default']}, {len(skip)} 格跳过")
    print(f"  共 {total / 1024:.0f} KiB JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
