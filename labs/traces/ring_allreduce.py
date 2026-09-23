#!/usr/bin/env python3
"""Trace generator for L13 · Ring AllReduce 逐帧回放.

Emits one JSON file per rank count, plus a manifest naming the set:

    labs/traces/ring-allreduce-N<2|4|8>.json
    labs/traces/ring-allreduce.manifest.json

Each file is a complete frame-by-frame replay of a real AllReduce over `N`
ranks: ReduceScatter's `N-1` steps, then AllGather's `N-1` steps, one step per
frame, every buffer's contents printed in full.

WHAT IS COMPUTED HERE

The ring schedule is executed on real `torch` tensors. At each step every rank
sends one chunk downstream and receives one chunk from upstream, and the
receiving buffer is updated in place (`+=` during ReduceScatter, `=` during
AllGather). The numbers under each rank's buffer in the trace ARE those
tensors, not a reconstruction of them.

A step is SIMULTANEOUS, and that matters. During AllGather, rank `r-1` receives
a chunk in the same step in which rank `r` reads that very slot from it — so a
naive in-place loop that processes the sends in rank order copies a value that
has already been overwritten once. The implementation therefore snapshots every
source value before applying any write, which is what "one step = one
simultaneous exchange" means. See `run_schedule`.

WHAT THE NUMBERS ARE CHECKED AGAINST

The design doc asks for a PyTorch cross-check; for a collective the useful
version of that is stronger than a tolerance, so there are three layers:

  * **The same-order reference is exact.** `reference_ring(x, n)`
    independently computes, for each chunk, the sum in the order the ring
    actually accumulates it, and the schedule's output must equal it with
    `torch.equal` — zero tolerance. This is the assertion that the SCHEDULE is
    right: a wrong chunk index, a missing step, or a `copy` where an `add`
    belongs all change the order and are caught here, bit for bit.

  * **The order-free reference is a tolerance, and the tolerance is derived,
    not copied.** `x.sum(dim=0)` sums in torch's own order, which is a
    different permutation of the same N addends. The standard bound on
    reordering `k` fp32 addends is `(k-1) * eps * SUM|x_i|` with
    `eps = 2^-24 ~ 5.96e-8`, so for `N = 8` here it is ~2.5e-6; the tolerance
    is `1e-5`, a little under 4× that. The generator prints the bound and the
    measured residual together so the margin is visible rather than asserted.
    (L00's `1e-12` is a float64 number and would be meaningless here — fp32
    resolves nothing near it.)
    The residual is then *shown* to be pure summation order by re-running on
    **integer-valued** fp32 input, where every partial sum is exactly
    representable and the two paths must agree with `torch.equal`. Summing
    reals in two orders and summing integers in two orders differ in exactly
    one respect — rounding — so the second run isolates it.

  * **The control topologies must agree too.** The same tensor reduced by a
    centralized hub (`naive_allreduce`) and by a binary tree
    (`tree_allreduce`) must reach the same values as the ring, to the same
    tolerance. Ring AllReduce is a *schedule* over one reduction; if the three
    schedules disagreed, the lab's premise would be false.

WHERE THE ALGORITHM ITSELF IS ASSERTED

Beyond the arithmetic, the propositions that would make the lab vacuous if
false are checked on the actual run, and the script refuses to write a file
when one fails:

  * every rank ends holding **all N chunks, each complete** (count == N);
  * at the end of ReduceScatter every rank holds **exactly one** complete
    chunk, and it is chunk `(r+1) mod N` — the "different ranks own different
    chunks" fact the second phase depends on;
  * every rank sends exactly one chunk per step, `to == (r+1) mod N`, and the
    chunk size is `S/N` — so each rank's total traffic is exactly
    `2(N-1)*S/N` elements, which is the bandwidth-optimality claim itself;
  * the naive hub's busiest link carries `2(N-1)*S`, i.e. **N times** the
    ring's, which is the whole quantitative comparison the lab exists to show;
  * the chunk-completion counts used by the view are recomputed from an
    explicit per-chunk **set of contributing ranks** (a bitmask), not from the
    counter the view folds — so "count == 4" and "ranks {0,1,2,3} are in it"
    are two independent derivations that must agree.

The traffic and timing models are stated as models. `ALPHA_US` (per-step issue
latency) and `BETA_GBS` (per-link one-way bandwidth) are order-of-magnitude
figures for NVLink with NCCL, not measurements; they exist so the Ring-vs-Tree
crossover — the reason NCCL ships both — is a computed number rather than a
sentence. The generator prints the crossover so the assumption is visible.

Contract notes (see docs/plans/interactive-labs.md §一 "轨迹数据模型"):

  * `tensors[].init` is mandatory for anything read but never written. `grad`
    carries the input; the per-rank buffers `buf_<r>` are written from step 0.
  * `step.state` is "every tensor value that changed during this step, in
    full" — never a delta. That is what makes arbitrary cursor jumps a pure
    function of (trace, cursor) with no undo log.
  * Two fields are this view's own contract, linted on **both** sides — the
    Python rules below and the JS port in `labs/assets/engine/views/ring.js`.
    (A view's contract belongs beside the view; `trace-model.js` is the
    engine's own lint and belongs to no single lab.) Both sides carry a
    sabotage control group, because a rule tested on one side only is a rule
    tested on neither.
      - `trace.ring`     — the rank list, the chunk size, the reduction being
        performed (`op`: which tensor, how many elements), and the two control
        schedules (`compare.naive`, `compare.tree`).
      - `step.transfers` — what crossed which link during this step:
        `{from, to, chunk, elements, mode?}`. `mode` is `add` (ReduceScatter:
        the received chunk is accumulated) or `copy` (AllGather and the naive
        broadcast: the received chunk REPLACES the local one), and it is not
        derivable — the same shape of transfer means different arithmetic in
        the two phases, which is exactly why it is declared.

The script is deterministic (fixed arithmetic, no sampling), which is what lets
CI re-run it and require the committed JSON to be reproduced byte for byte —
see .github/workflows/trace-checks.yml.

Run: python3 labs/traces/ring_allreduce.py
"""

import json
import math
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ the model

# Rank counts the lab offers. The ticket's 2/4/8.
N_VALUES = [2, 4, 8]
# Elements in each rank's gradient tensor. Small on purpose: every tutorial
# says "cut S into N chunks", and the only way for a reader to check that claim
# against the picture is for a chunk to be small enough to read. S must be
# divisible by every N above, so a chunk is a whole number of elements at each
# setting rather than a padded one.
S = 8
DTYPE = torch.float32
ITEM_BYTES = 4                      # fp32
EPS_FP32 = 2 ** -24                 # unit roundoff: 2^-p with p = 24 significand bits
# The tolerance the order-free reference is compared at. Derived, not copied.
# Reordering `k` fp32 addends cannot move the result by more than
# `(k-1)·eps·Σ|x_i|`; for k = N = 8 here that bound is ~2.5e-6 (the generator
# prints it alongside the measured residual). 1e-5 is a little under 4× that,
# which leaves room for the bound being a worst case while still being six
# orders of magnitude below the O(1) move a real arithmetic mistake produces —
# the gap between "summation order" and "wrong" is what the tolerance has to sit
# inside, and its exact value does not matter. (L00's 1e-12 is a float64 number
# and would be meaningless here.)
TOL = 1e-5
# The size the per-rank gradient "would really be" in a training run. Used only
# for the timing/busbw readout, where S = 8 elements would make every topology
# finish in nanoseconds and hide the crossover entirely. Every number derived
# from it is labelled as a scaled projection on the page.
NOMINAL_BYTES = 100 * 1024 * 1024
# Per-step issue latency (one link-level transfer's fixed cost) and one-way
# per-link bandwidth. Order-of-magnitude NVLink + NCCL figures, not measured.
ALPHA_US = 5.0
BETA_GBS = 300.0

# Sentinel spellings for values JSON cannot represent. These are the contract's
# strings, not a local convention — the engine parses exactly these.
NEG_INF = r"-\infty"
POS_INF = r"+\infty"
NAN = "NaN"

MODE_ADD = "add"
MODE_COPY = "copy"
MODES = [MODE_ADD, MODE_COPY]

PHASE_INIT = "准备"
PHASE_REDUCE = "ReduceScatter"
PHASE_GATHER = "AllGather"
# The schedule's own phase ids, shared by every topology in the trace.
STEP_REDUCE = "reduce"
STEP_GATHER = "gather"

# The layouts the engine draws. A `comm` step is one that moves data across a
# link; the init step is `state` because it only rearranges what is already
# there (the same vocabulary L01 and L10 use).
KIND_COMM = "comm"
KIND_STATE = "state"


# ------------------------------------------------------------- the schedules
#
# A schedule is a list of steps; a step is a list of sends. This is the ONLY
# description of a collective in this file — the simulation, the buffer
# bookkeeping, the traffic accounting and the view's per-step contract all read
# this same structure, so a schedule cannot be right for one of them and wrong
# for another.


def ring_schedule(n):
    """Ring AllReduce: N-1 ReduceScatter steps, then N-1 AllGather steps.

    At ReduceScatter step `k`, rank `r` sends chunk `(r-k) mod N` to rank
    `r+1` and receives chunk `(r-k-1) mod N` from rank `r-1`, accumulating it.
    The chunk that has visited every rank by the end is `(r+1) mod N`, which is
    why rank `r` is the one left holding chunk `(r+1) mod N`.

    AllGather repeats the rotation one place over: at step `k`, rank `r` sends
    chunk `(r-k+1) mod N` — the one it holds complete — downstream, and copies
    chunk `(r-k) mod N` from upstream.
    """
    steps = []
    for k in range(n - 1):
        steps.append({
            "phase": STEP_REDUCE,
            "k": k,
            "transfers": [{"from": r, "to": (r + 1) % n, "chunk": (r - k) % n,
                       "mode": MODE_ADD} for r in range(n)],
        })
    for k in range(n - 1):
        steps.append({
            "phase": STEP_GATHER,
            "k": k,
            "transfers": [{"from": r, "to": (r + 1) % n, "chunk": (r - k + 1) % n,
                       "mode": MODE_COPY} for r in range(n)],
        })
    return steps


def naive_schedule(n):
    """Centralized AllReduce: everyone to rank 0, rank 0 back to everyone.

    This is the tutorial's §4.1 strawman, and it is modelled as `2(N-1)`
    transfers rather than two broadcasts on purpose. Rank 0's link is the
    bottleneck: it receives `N-1` tensors and then sends `N-1` tensors, and
    those transmissions are serialized on that one link whether or not the NIC
    opens several connections. Two transfers per peer is therefore the honest
    count of what the hub has to push, and it also gives this schedule the same
    number of steps as the ring's — so the two cumulative curves can be read at
    the same tick instead of being an apples-to-oranges overlay.

    One chunk, because a hub has no reason to cut the tensor up: it moves the
    whole thing.
    """
    steps = []
    for i in range(1, n):
        steps.append({"phase": STEP_REDUCE, "k": i - 1,
                      "transfers": [{"from": i, "to": 0, "chunk": 0,
                                 "mode": MODE_ADD}]})
    for i in range(1, n):
        steps.append({"phase": STEP_GATHER, "k": i - 1,
                      "transfers": [{"from": 0, "to": i, "chunk": 0,
                                 "mode": MODE_COPY}]})
    return steps


def tree_parent(r):
    """Heap layout: rank 0 is the root, rank r's parent is (r-1)//2."""
    return (r - 1) // 2


def tree_depth(r):
    return (r + 1).bit_length() - 1


def tree_schedule(n):
    """Binary-tree AllReduce: reduce leaf -> root, then broadcast root -> leaf.

    Each step is one whole level of the tree moving at once, which is what
    makes the latency `O(log N)` instead of `O(N)`. One chunk, and it is the
    WHOLE tensor: a tree reduces by handing up a complete partial sum, not by
    rotating slices, so there is nothing to cut up. That is the structural
    difference between the two topologies, and the buffer table makes it
    visible — N cells for the ring, one for the tree.
    """
    depths = [tree_depth(r) for r in range(n)]
    max_depth = max(depths)
    steps = []
    for d in range(max_depth, 0, -1):
        sends = [{"from": r, "to": tree_parent(r), "chunk": 0,
                  "mode": MODE_ADD} for r in range(n) if depths[r] == d]
        steps.append({"phase": STEP_REDUCE, "k": max_depth - d, "transfers": sends})
    for d in range(0, max_depth):
        sends = []
        for r in range(n):
            if depths[r] != d:
                continue
            for child in (2 * r + 1, 2 * r + 2):
                if child < n:
                    sends.append({"from": r, "to": child, "chunk": 0,
                                  "mode": MODE_COPY})
        steps.append({"phase": STEP_GATHER, "k": d, "transfers": sends})
    return steps


def tree_edges(n):
    """Parent-pointing edges, i.e. the reduce tree the primer draws.

    Emitted with rank IDS, not integer positions: the view draws what the trace
    declares, and a trace whose edge list spoke a different vocabulary from its
    rank list would be a silent mismatch the lint exists to catch.
    """
    return [{"from": f"r{r}", "to": f"r{tree_parent(r)}"} for r in range(1, n)]


SCHEDULES = {"ring": ring_schedule, "naive": naive_schedule, "tree": tree_schedule}


# ------------------------------------------------------------- the execution


def run_schedule(x, n, chunk_elements, schedule):
    """Execute a schedule on real tensors. Returns (buffers, frames).

    `x` is the (N, S) gradient tensor, one row per rank. Buffers are laid out
    (N, chunks, chunk_elements) — rank-major, so `buf[r][c]` is the chunk `c`
    that rank `r` currently holds, which is exactly what the table draws.

    The snapshot before each step is what makes a step one SIMULTANEOUS
    exchange rather than a loop over ranks, and it is written that way on
    purpose: the correct in-place implementation is correct only because of a
    property of the ring's rotation (see `read_write_disjoint`), and an
    implementation that silently depends on the schedule never changing shape
    is one refactor away from being wrong. The snapshot does not depend on it.

    Whether the two agree is not asserted from intuition. `read_write_disjoint`
    checks the property directly, and `hazard_demo` shows what happens without
    it — so the equivalence is a checked claim with a control group rather than
    a comment.
    """
    if S % n:
        raise ValueError(f"S={S} 不能被 N={n} 整除")
    chunks = S // chunk_elements
    if chunks * chunk_elements != S:
        raise ValueError(f"chunk_elements={chunk_elements} 除不尽 S={S}")
    buf = x.reshape(n, chunks, chunk_elements).clone()
    frames = [buf.clone()]
    for st in schedule:
        src = [(s["from"], s["chunk"], buf[s["from"], s["chunk"]].clone())
               for s in st["transfers"]]
        for s, (_, _, val) in zip(st["transfers"], src):
            if s["mode"] == MODE_ADD:
                buf[s["to"], s["chunk"]] = buf[s["to"], s["chunk"]] + val
            else:
                buf[s["to"], s["chunk"]] = val
        frames.append(buf.clone())
    return buf, frames


def run_schedule_sequential(x, n, chunk_elements, schedule):
    """The same run, applying each transfer as it is reached instead of from a
    snapshot.

    Kept because the difference between it and `run_schedule` is a claim worth
    checking rather than assuming: they agree here, and `read_write_disjoint`
    says exactly why. An in-place loop is only safe when no slot a step WRITES
    is also a slot the same step READS from that rank.
    """
    chunks = S // chunk_elements
    buf = x.reshape(n, chunks, chunk_elements).clone()
    for st in schedule:
        for s in st["transfers"]:
            if s["mode"] == MODE_ADD:
                buf[s["to"], s["chunk"]] = (buf[s["to"], s["chunk"]] +
                                            buf[s["from"], s["chunk"]])
            else:
                buf[s["to"], s["chunk"]] = buf[s["from"], s["chunk"]].clone()
    return buf


def read_write_clobbers(schedule):
    """Where a transfer would read a slot an EARLIER transfer in the same step
    has already written.

    This is the property that decides whether an in-place loop and a
    snapshot-and-apply loop agree, and getting its shape right matters: the
    thing that clobbers is not "a rank reading the slot it writes" (a transfer
    reads the SOURCE rank's slot and writes the DESTINATION's, so those are
    never the same slot). It is one transfer's read landing on the slot a
    previous transfer in the same step overwrote — i.e. a pair with

        t_prev.to == t.from  and  t_prev.chunk == t.chunk

    with `t_prev` earlier in the step's order. Returns one entry per clobber.

    For the ring it is empty by construction. AllGather step `k` transfers,
    in rank order, are `t_r: read buf[r][(r-k+1)%n] -> write buf[r+1][(r-k+1)%n]`,
    so the previous transfer wrote `buf[r][(r-k)%n]` — adjacent chunk, never the
    one `t_r` reads. (That is the same fact as the rotation being by one.)
    """
    violations = []
    for si, st in enumerate(schedule):
        written = set()
        for t in st["transfers"]:
            slot = (t.get("from"), t.get("chunk"))
            if slot in written:
                violations.append({"step": si, "rank": t.get("from"),
                                   "chunk": t.get("chunk")})
            written.add((t.get("to"), t.get("chunk")))
    return violations


# Kept under the name the docstrings above use.
def read_write_disjoint(n, schedule):
    return read_write_clobbers(schedule)


def hazard_demo(x, n, chunk_elements):
    """A schedule that DOES step on itself, and the two runs that disagree.

    The ring avoids the clobber by construction, so the control group for "the
    snapshot is what makes a step simultaneous" has to be constructed rather
    than found. This does the smallest thing that creates one: in the first
    AllGather step, every rank sends the same chunk. Now transfer 0 writes
    `buf[1][0]`, and transfer 1 reads `buf[1][0]` — a read of a slot this same
    step already overwrote. The two implementations must then disagree; if
    they agree, the snapshot is doing nothing and the claim it supports is
    empty.
    """
    sched = [{"phase": s["phase"], "k": s["k"],
              "transfers": [dict(t) for t in s["transfers"]]} for s in
             ring_schedule(n)]
    gather = [s for s in sched if s["phase"] == STEP_GATHER]
    if not gather:
        return None
    for t in gather[0]["transfers"]:
        t["chunk"] = 0
    clash = read_write_clobbers(sched)
    if not clash:
        return None                     # the mutation did not create a clash
    good, _ = run_schedule(x, n, chunk_elements, sched)
    seq = run_schedule_sequential(x, n, chunk_elements, sched)
    return {"clobbers": clash, "disagree": not torch.equal(good, seq)}


def counts_from_sends(n, chunks, schedule):
    """Chunk completion counts, folded from the transfer list alone.

    Seeds every chunk of every rank at 1 — a rank holds its own contribution to
    all N chunks from the start — and then applies the same two rules the view
    folds: `add` sums the source's count in, `copy` replaces it. This is the
    counter the VIEW computes; `masks_from_sends` below derives the same thing
    a structurally different way, and the two must agree.
    """
    counts = [[1] * chunks for _ in range(n)]
    for st in schedule:
        for s in st["transfers"]:
            if s["mode"] == MODE_ADD:
                counts[s["to"]][s["chunk"]] += counts[s["from"]][s["chunk"]]
            else:
                counts[s["to"]][s["chunk"]] = counts[s["from"]][s["chunk"]]
    return counts


def masks_from_sends(n, chunks, schedule):
    """Which ranks' data is in each buffer, as an explicit set. The independent
    derivation: `popcount(mask) == count` is a real cross-check, because a
    counter can drift and a set cannot."""
    masks = [[{r} for _ in range(chunks)] for r in range(n)]
    for st in schedule:
        for s in st["transfers"]:
            if s["mode"] == MODE_ADD:
                masks[s["to"]][s["chunk"]] = (masks[s["to"]][s["chunk"]] |
                                              masks[s["from"]][s["chunk"]])
            else:
                masks[s["to"]][s["chunk"]] = set(masks[s["from"]][s["chunk"]])
    return masks


def per_step_bytes(n, chunk_elements, schedule):
    """Bytes each rank SENDS per step, and the busiest link's per-step load.

    The busiest link is what decides when the collective finishes under a
    bandwidth bound, so it — not the mean — is the quantity the accumulator
    plots. For the ring every rank is equally loaded; for the hub and the tree
    one rank carries the rest.
    """
    per_rank = []
    for st in schedule:
        sent = [0] * n
        for s in st["transfers"]:
            sent[s["from"]] += chunk_elements * ITEM_BYTES
        per_rank.append(sent)
    return per_rank


def cumulative_curves(n, chunk_elements, schedule, links):
    """The traffic account for one schedule, folded from its transfer list.

    Three quantities, kept separate because conflating them is exactly the
    mistake this comparison invites:

      * `per_rank_cumulative` / `per_rank_total` — how many bytes each rank has
        SENT by each step. This is what the accumulator draws, step by step.

      * `busiest_cumulative` / `busiest_total` — the same curve for the rank
        that sends the most. For the ring every rank is that rank; for the hub
        it is rank 0 and the other ranks are flat.

      * `link_load_total` — the AGGREGATE bytes moved, divided by how many
        links carried them (`links`). This is the number the tutorial's
        "朴素 2(N-1)S vs Ring 2(N-1)S/N" is about, and it is the right way to
        state it: all three schedules move the SAME total (asserted below), and
        what differs is how many links it is spread across. The hub pours the
        whole thing down one link; the ring spreads it over N. That is where
        the factor of N comes from, and stating it as "aggregate ÷ links" is
        what keeps the factor honest — comparing the hub's two-directional link
        load against a ring rank's one-directional send would also yield N, but
        only by measuring two different things.

    All of it is folded from the transfer list, and the view folds the same
    list the same way — so this is the value the acceptance harness compares the
    browser's fold against, rather than a second opinion.
    """
    per_step = per_step_bytes(n, chunk_elements, schedule)
    running = [0] * n
    series = []
    for sent in per_step:
        running = [a + b for a, b in zip(running, sent)]
        series.append(list(running))
    total = sum(running)
    if links < 1:
        raise ValueError("cumulative_curves: links must be at least 1")
    return {
        "per_step": per_step,
        "per_rank_cumulative": series,
        "busiest_cumulative": [max(step) for step in series],
        "per_rank_total": list(running),
        "busiest_total": max(running) if running else 0,
        "total": total,
        "links": links,
        "link_load_total": total // links,
        "steps": len(per_step),
    }


# ---------------------------------------------------------------- references


def reference_ring(x, n):
    """The same sum, computed in the order the ring accumulates it.

    Chunk `j` starts at rank `j` and is accumulated at ranks `j+1`, `j+2`, ...
    `j+N-1` (mod N), so the partial sums are `x[j] + x[j+1] + ...`. Reproducing
    that order is what makes the comparison EXACT rather than approximate: this
    asserts the SCHEDULE, and the tolerance check further down asserts the
    arithmetic.

    Every rank ends holding every chunk, and chunk `j` has the same value
    everywhere, so the reduced chunk is broadcast across ranks to form the
    expected final buffer state.
    """
    width = S // n
    vals = []
    for j in range(n):
        chunk_of = lambda r: x[r].reshape(n, width)[j]      # noqa: E731
        # The additions happen one at a time, in the ring's visiting order.
        # `torch.stack(...).sum()` would look tidier and would silently use
        # torch's own reduction order, which is the one thing this function
        # exists not to do.
        acc = chunk_of(j).clone()
        for d in range(1, n):
            acc = acc + chunk_of((j + d) % n)
        vals.append(acc)
    final = torch.empty(n, n, width, dtype=x.dtype)
    for j in range(n):
        final[:, j] = vals[j]
    return final


def reference_allreduce(x):
    """The order-free truth: the elementwise sum over ranks."""
    return x.sum(dim=0)


def naive_allreduce(x, n):
    """Centralized AllReduce executed on real tensors (hub = rank 0)."""
    chunks = 1
    buf = x.reshape(n, chunks, S).clone()
    for i in range(1, n):
        buf[0, 0] = buf[0, 0] + buf[i, 0]
    for i in range(1, n):
        buf[i, 0] = buf[0, 0].clone()
    return buf


def tree_allreduce(x, n):
    """Binary-tree AllReduce executed on real tensors."""
    buf = x.reshape(n, 1, S).clone()
    depths = [tree_depth(r) for r in range(n)]
    for d in range(max(depths), 0, -1):
        src = [(r, buf[r, 0].clone()) for r in range(n) if depths[r] == d]
        for r, val in src:
            p = tree_parent(r)
            buf[p, 0] = buf[p, 0] + val
    for d in range(0, max(depths)):
        src = [(r, buf[r, 0].clone()) for r in range(n) if depths[r] == d]
        for r, val in src:
            for child in (2 * r + 1, 2 * r + 2):
                if child < n:
                    buf[child, 0] = val.clone()
    return buf


def make_input(n, integral=False):
    """The per-rank gradient rows. Deterministic — no sampling anywhere.

    Two flavours. `integral` produces small integers as fp32; every partial sum
    of such values is exactly representable, so two different summation orders
    MUST agree bit for bit. That is what turns "the residual is summation
    order" from an assertion into a demonstration, since the same comparison on
    fractional values legitimately differs in the last bits.
    """
    rows = []
    for r in range(n):
        if integral:
            rows.append([float(((r * 7 + i * 3) % 11) - 5) for i in range(S)])
        else:
            rows.append([0.5 * math.sin(1.7 * r + 2.3 * i + 0.4)
                         - 0.25 * math.cos(0.9 * r * i + 1.1)
                         for i in range(S)])
    return torch.tensor(rows, dtype=DTYPE)


# ------------------------------------------------------------------ the lint
#
# The ring view's slice of the contract — the author-side half. The other half
# is the JS port in `labs/assets/engine/views/ring.js` (`LabEngine.ring.lint`),
# and it holds the same rules. A rule that exists on one side only is a rule
# tested on neither, which is the failure mode this pair exists to prevent;
# when a rule changes it changes in both, and both sabotage tables grow a case
# for it.
#
# Only THIS view's rules are here (rings, ranks, transfers, control schedules).
# The general trace contract — declared tensors, graph/step agreement,
# three-tier bindings — is the engine's own lint and every generator's job
# alike, and duplicating it would just be a second copy to keep in sync.

RANK_RE = re.compile(r"^[A-Za-z0-9_-]+$")
XFER_KEYS = {"from", "to", "chunk", "elements"}
XFER_OPTIONAL = {"mode"}


def _is_pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _is_non_neg_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _lint_transfers(label, steps, rank_ids, chunks, gaps):
    """The rules that must hold of ANY schedule carried in this trace.

    Shared by the ring's own steps and by the two control schedules, because
    all three are the same kind of object — a list of steps, each a list of
    transfers — and a rule that held for one but not the others would be a rule
    about nothing. `steps` is a list of `{phase, k, transfers}`.
    """
    if not isinstance(steps, list) or not steps:
        gaps.append(f"{label} 不是非空数组 —— 这条调度没有任何一步可放")
        return
    for si, st in enumerate(steps):
        sid = f"{label}[{si}]"
        if not isinstance(st, dict):
            gaps.append(f"{sid} 不是对象")
            continue
        if st.get("phase") not in (STEP_REDUCE, STEP_GATHER):
            gaps.append(f'{sid}.phase = {st.get("phase")!r} 不是 '
                        f'{STEP_REDUCE!r} / {STEP_GATHER!r} —— 视图按它给时间轴上色')
        if not _is_non_neg_int(st.get("k")):
            gaps.append(f'{sid}.k = {st.get("k")!r} 不是非负整数')
        xfers = st.get("transfers")
        if not isinstance(xfers, list):
            gaps.append(f"{sid}.transfers 不是数组")
            continue
        seen = set()
        for xi, x in enumerate(xfers):
            xid = f"{sid}.transfers[{xi}]"
            if not isinstance(x, dict):
                gaps.append(f"{xid} 不是对象")
                continue
            extra = set(x) - XFER_KEYS - XFER_OPTIONAL
            if extra:
                gaps.append(f"{xid} 多了字段 {{{', '.join(sorted(extra))}}} —— "
                            "契约只认 {from, to, chunk, elements, mode?}")
            missing = XFER_KEYS - set(x)
            if missing:
                gaps.append(f"{xid} 缺 {{{', '.join(sorted(missing))}}}")
                continue
            if x["from"] not in rank_ids:
                gaps.append(f'{xid}.from = {x["from"]!r} 不是本 trace 声明的 rank')
            if x["to"] not in rank_ids:
                gaps.append(f'{xid}.to = {x["to"]!r} 不是本 trace 声明的 rank')
            if x["from"] == x["to"]:
                gaps.append(f'{xid} 的 from 与 to 都是 {x["from"]!r} —— '
                            "自己发给自己不是一次跨链路传输")
            if not _is_non_neg_int(x["chunk"]) or x["chunk"] >= chunks:
                gaps.append(f'{xid}.chunk = {x["chunk"]!r} 不在 0..{chunks - 1} 里')
            if not _is_pos_int(x["elements"]):
                gaps.append(f'{xid}.elements = {x["elements"]!r} 不是正整数')
            mode = x.get("mode", MODE_ADD)
            if mode not in MODES:
                gaps.append(f'{xid}.mode = {mode!r} 不是 {MODES} 之一 —— '
                            "add（累加）与 copy（覆盖）是两种不同的算术")
            key = (x.get("from"), x.get("to"), x.get("chunk"))
            if key in seen:
                gaps.append(f"{xid} 与同一 tick 的另一条传输重复（{key}）—— "
                            "同一 tick 同一条链路上一个 chunk 只传一次")
            seen.add(key)
    # The end state every AllReduce must reach: every rank holds every chunk,
    # fully reduced. Folded from the transfers with the same two rules the view
    # uses, and seeded at 1 because a rank starts holding its own contribution.
    # A gap, not a warning: a schedule that stops short of the complete
    # reduction is not a partially-correct collective, it is a different
    # algorithm wearing the same name.
    #
    # The fold is skipped when a transfer is malformed, because it would either
    # raise or fold nonsense — and the malformation is already reported above,
    # so the verdict is unchanged either way. Indexing the ranks by their
    # declared position (rather than by name) is also what the view does.
    index_of = {rid: i for i, rid in enumerate(rank_ids)}
    ok = all(
        x.get("from") in index_of and x.get("to") in index_of
        and _is_non_neg_int(x.get("chunk")) and x["chunk"] < chunks
        and x.get("mode", MODE_ADD) in MODES
        for st in steps if isinstance(st, dict)
        for x in (st.get("transfers") or []) if isinstance(x, dict)
    ) and all(isinstance(st, dict) and isinstance(st.get("transfers"), list)
             for st in steps)
    if not ok:
        return
    indexed = [{"transfers": [
        {"from": index_of[x["from"]], "to": index_of[x["to"]],
         "chunk": x["chunk"], "mode": x.get("mode", MODE_ADD)}
        for x in st["transfers"]]} for st in steps]
    counts = counts_from_sends(len(rank_ids), chunks, indexed)
    for r in range(len(rank_ids)):
        for cidx in range(chunks):
            if counts[r][cidx] != len(rank_ids):
                gaps.append(f'{label} 结束时 rank "{rank_ids[r]}" 的 chunk {cidx} '
                            f"只含 {counts[r][cidx]} 份数据"
                            f"（应为 {len(rank_ids)}）—— "
                            "这条调度没有完成一次完整的 AllReduce")
                return


def _lint_ring_rotation(label, steps, rank_ids, chunk_elements, gaps):
    """The rules that make a schedule a RING, and not just a valid collective.

    Gated on `algorithm == 'ring'` because they are false of the tree by
    construction (a tree's leaves send and its interior receives; that is the
    point of it). Kept separate from `_lint_transfers` so "valid schedule" and
    "is a ring" cannot be confused for one rule — the first bug a reader would
    hit is a schedule that is a valid collective of the wrong shape.

    The downstream of rank `i` is taken from the DECLARED rank order
    (`rank_ids[i] -> rank_ids[i+1]`), not from parsing the rank's name: the
    ring lives in the `ranks` list, and a trace that named its ranks "gpu0.."
    would otherwise silently escape every check here.
    """
    n = len(rank_ids)
    successor = {rank_ids[i]: rank_ids[(i + 1) % n] for i in range(n)}
    for si, st in enumerate(steps):
        xfers = st.get("transfers") or []
        if not xfers:
            continue
        if len(xfers) != n:
            gaps.append(f"{label}[{si}] 有 {len(xfers)} 条传输，"
                        f"环形是 {n} 张卡一步各发一条")
        senders = [x.get("from") for x in xfers]
        if len(set(senders)) != len(senders):
            gaps.append(f"{label}[{si}] 里有 rank 一步发了不止一块 —— "
                        "环形每步每卡只发一块（这也是「每卡每步 S/N」的前提）")
        for x in xfers:
            if x.get("from") in successor and x.get("to") != successor[x["from"]]:
                gaps.append(f'{label}[{si}]：rank {x["from"]} 发给了 {x["to"]}，'
                            f'环上的下游应当是 {successor[x["from"]]}')
            if x.get("elements") != chunk_elements:
                gaps.append(f'{label}[{si}]：一条传输搬了 {x.get("elements")} 个元素，'
                            f"但环形每步搬一个 chunk = {chunk_elements} 个 —— "
                            "«每步传 S/N» 这个数字就是带宽最优性的全部依据")


def lint(trace):
    gaps, warns, infos = [], [], []
    declared = set(trace["tensors"])
    ring = trace.get("ring")

    if not isinstance(ring, dict):
        gaps.append("ring 缺失 —— 环形视图没有卡数、chunk 大小与归约对象的定义，"
                    "一帧都画不出来")
        return gaps, warns, infos

    n = ring.get("n")
    if not _is_pos_int(n) or n < 2:
        gaps.append(f"ring.n = {n!r} 不是 ≥ 2 的整数")
        return gaps, warns, infos

    ranks = ring.get("ranks")
    rank_ids = []
    if not isinstance(ranks, list):
        gaps.append("ring.ranks 不是数组")
    else:
        for i, rk in enumerate(ranks):
            rid = rk.get("id") if isinstance(rk, dict) else None
            if not isinstance(rid, str) or not rid:
                gaps.append(f"ring.ranks[{i}] 没有 id")
                continue
            if not RANK_RE.match(rid):
                gaps.append(f"ring.ranks[{i}].id = {rid!r} 含非法字符"
                            "（只允许字母数字下划线连字符）")
            if rid in rank_ids:
                gaps.append(f'ring.ranks 里 id "{rid}" 重复 —— '
                            "两条传输会指向同一张卡")
            rank_ids.append(rid)
            if not (rk.get("label") if isinstance(rk, dict) else None):
                warns.append(f'ring.ranks[{i}] ("{rid}") 没有 label，只能显示 id')
        if len(ranks) != n:
            gaps.append(f"ring.ranks 有 {len(ranks)} 项，但 ring.n = {n} —— "
                        "拓扑上画的节点数与调度里的卡数不是一回事")

    chunk_elements = ring.get("chunk_elements")
    if not _is_pos_int(chunk_elements):
        gaps.append(f"ring.chunk_elements = {chunk_elements!r} 不是正整数")
        return gaps, warns, infos
    if S % chunk_elements:
        gaps.append(f"ring.chunk_elements = {chunk_elements} 除不尽 S = {S}")

    op = ring.get("op")
    if not isinstance(op, dict):
        gaps.append("ring.op 缺失 —— 视图说不出这一次归约作用在哪个张量上")
    else:
        for key in ("id", "label", "tensor", "elements", "bytes"):
            if key not in op:
                gaps.append(f'ring.op 缺 "{key}"')
        if op.get("tensor") not in declared:
            gaps.append(f'ring.op.tensor = {op.get("tensor")!r} 不在 tensors 里 —— '
                        "「一次归约作用于一个梯度张量」这句话就落不了地")
        if not _is_pos_int(op.get("elements")):
            gaps.append(f'ring.op.elements = {op.get("elements")!r} 不是正整数')
        if not _is_pos_int(op.get("bytes")):
            gaps.append(f'ring.op.bytes = {op.get("bytes")!r} 不是正整数')
        elif _is_pos_int(op.get("elements")) and \
                op["bytes"] != op["elements"] * ITEM_BYTES:
            gaps.append(f'ring.op.bytes = {op["bytes"]} 与 elements × {ITEM_BYTES} = '
                        f'{op["elements"] * ITEM_BYTES} 不一致')

    algorithm = ring.get("algorithm")
    if algorithm not in ("ring", "tree"):
        gaps.append(f"ring.algorithm = {algorithm!r} 不是 'ring' / 'tree'")

    chunks = S // chunk_elements if _is_pos_int(chunk_elements) else 1
    if algorithm == "ring" and _is_pos_int(op.get("elements") if isinstance(op, dict)
                                          else None) and _is_pos_int(chunk_elements):
        if chunks != n:
            gaps.append(f'ring.algorithm 是 ring，但 chunk 数 {chunks} 与卡数 {n} 不等 —— '
                        "环形要求 S 正好切成 N 块")

    # ---- the trace's own steps, flattened into the schedule vocabulary
    own = []
    for s in trace["steps"]:
        sid = s["id"]
        xfers = s.get("transfers")
        if xfers is None:
            gaps.append(f'步骤 "{sid}" 没有 transfers —— '
                        "环形视图在这一帧不知道有没有数据过链")
            continue
        if not isinstance(xfers, list):
            gaps.append(f'步骤 "{sid}" 的 transfers 不是数组')
            continue
        for w in (s.get("writes") or []):
            if w not in declared:
                gaps.append(f'步骤 "{sid}" 写了未声明的张量 "{w}"')
        own.append({
            "phase": s.get("phase_id", STEP_REDUCE),
            "k": s.get("k", 0) if _is_non_neg_int(s.get("k")) else -1,
            "transfers": xfers,
        })
    if own:
        _lint_transfers("trace.steps", own, rank_ids, chunks, gaps)
        if algorithm == "ring" and rank_ids:
            _lint_ring_rotation("trace.steps", own, rank_ids, chunk_elements, gaps)

    # ---- the control schedules
    compare = ring.get("compare")
    if compare is not None:
        if not isinstance(compare, dict):
            gaps.append("ring.compare 不是对象")
        else:
            for key in ("naive", "tree"):
                if key not in compare:
                    continue
                entry = compare[key]
                label = f"ring.compare.{key}"
                if not isinstance(entry, dict):
                    gaps.append(f"{label} 不是对象")
                    continue
                if not entry.get("label"):
                    gaps.append(f"{label} 缺 label —— 对照曲线的图例会是一片空白")
                if key == "tree":
                    edges = entry.get("edges")
                    if not isinstance(edges, list) or not edges:
                        gaps.append(f"{label}.edges 不是非空数组 —— 树拓扑画不出来")
                    else:
                        listed = {e.get("from") for e in edges if isinstance(e, dict)}
                        for e in edges:
                            if not isinstance(e, dict):
                                gaps.append(f"{label}.edges 里有非对象项")
                                continue
                            if e.get("from") not in rank_ids:
                                gaps.append(f'{label}.edges 的 from = {e.get("from")!r} '
                                            "不是声明的 rank")
                            if e.get("to") not in rank_ids:
                                gaps.append(f'{label}.edges 的 to = {e.get("to")!r} '
                                            "不是声明的 rank")
                        missing = [r for r in rank_ids[1:] if r not in listed]
                        if missing:
                            gaps.append(f"{label}.edges 里 {missing} 没有父节点 —— "
                                        "这棵树接不到根上")
                # A control schedule may carry its own chunk size (the tree and
                # the hub both move the whole tensor in one piece), so the chunk
                # count is derived from what the entry declares rather than
                # assumed to be the ring's.
                entry_ce = entry.get("chunk_elements")
                entry_chunks = (S // entry_ce
                                if _is_pos_int(entry_ce) and S % entry_ce == 0
                                else chunks)
                _lint_transfers(f"{label}.steps", entry.get("steps"),
                                rank_ids, entry_chunks, gaps)

    infos.append(f"N={n} · S={S} 元素 · chunk={chunk_elements} 元素 · "
                 f"{len(trace['steps'])} 步")
    for key in ("naive", "tree"):
        entry = (compare or {}).get(key)
        if isinstance(entry, dict):
            infos.append(f'对照 {key}：{len(entry.get("steps") or [])} 步')
    return gaps, warns, infos


def _first_xfer_step(trace):
    for i, s in enumerate(trace["steps"]):
        if s.get("transfers"):
            return i
    return 0


SABOTAGE_CASES = {
    "ring 整块缺失": lambda t: t.pop("ring"),
    "ring.n 不是整数": lambda t: t["ring"].update({"n": "4"}),
    "ring.n 小于 2": lambda t: t["ring"].update({"n": 1}),
    "ring.ranks 与 n 不符": lambda t: t["ring"].update({"ranks": t["ring"]["ranks"][:-1]}),
    "rank id 重复": lambda t: t["ring"]["ranks"].append(
        dict(t["ring"]["ranks"][0])),
    "rank id 含非法字符": lambda t: t["ring"]["ranks"][0].update({"id": "rank 0"}),
    "rank 没有 id": lambda t: t["ring"]["ranks"][0].pop("id"),
    "chunk_elements 为 0": lambda t: t["ring"].update({"chunk_elements": 0}),
    "chunk 数不等于卡数": lambda t: t["ring"].update({"chunk_elements": 8}),
    "ring.op 缺失": lambda t: t["ring"].pop("op"),
    "ring.op.tensor 未声明": lambda t: t["ring"]["op"].update({"tensor": "grad2"}),
    "ring.op.bytes 与 elements 不符": lambda t: t["ring"]["op"].update({"bytes": 1}),
    "ring.algorithm 非法": lambda t: t["ring"].update({"algorithm": "star"}),
    "步骤没有 transfers": lambda t: t["steps"][_first_xfer_step(t)].pop("transfers"),
    "一条传输发给不存在的 rank": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"][0].update({"to": "r99"}),
    "一条传输发给自己": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"][0].update({"to": t["steps"][_first_xfer_step(t)]["transfers"][0]["from"]}),
    "chunk 下标越界": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"][0].update({"chunk": 99}),
    "elements 是 0": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"][0].update({"elements": 0}),
    "mode 不在词表里": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"][0].update({"mode": "merge"}),
    "传输多了个字段": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"][0].update({"color": "red"}),
    "同一 tick 重复传同一个 chunk": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"].append(dict(t["steps"][_first_xfer_step(t)]["transfers"][0])),
    "发错了下游（不是环上的下一个）": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"][0].update({"to": t["ring"]["ranks"][0]["id"]}),
    # Relative to the trace's OWN chunk size, never a literal. A literal 1
    # here is a no-op at N = S (where a chunk already is one element), and a
    # no-op mutation is a sabotage case that tests nothing — see
    # `sabotage_meta_checks`, which exists because this one was written as a
    # literal first and silently stopped biting at N = 8.
    "每步搬的不是一个 chunk": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"][0].update(
            {"elements": t["ring"]["chunk_elements"] + 1}),
    "某一步少了一条传输": lambda t: t["steps"][_first_xfer_step(t)][
        "transfers"].pop(),
    "对照调度缺 label": lambda t: t["ring"]["compare"]["naive"].pop("label"),
    "树的边指向不存在的 rank": lambda t: t["ring"]["compare"]["tree"]["edges"][
        0].update({"to": "r99"}),
    "树有一个节点接不到根上": lambda t: t["ring"]["compare"]["tree"].update(
        {"edges": t["ring"]["compare"]["tree"]["edges"][1:]}),
    "标签不是非负整数": lambda t: t["ring"]["compare"]["naive"]["steps"][0].update({"k": -1}),
}


def _step_ref(r_i):
    return f"r{r_i}"


def sabotage_meta_checks(trace):
    """Does every mutation actually CHANGE the trace?

    A sabotage case is only evidence if the thing it sabotages is there to
    begin with. A case written against a literal — `elements = 1`, say — is a
    real break of a trace whose chunk size is 4 and a silent no-op on one whose
    chunk size is 1, and the difference between those two situations is
    invisible from `lint`'s verdict alone: both report "0 gaps", and the
    failure reads as though the LINT had stopped working when in fact the
    MUTATION stopped doing anything.

    So the two are checked separately. This one compares the mutated trace
    against the original and reports the mutation as dead if nothing moved;
    `sabotage_checks` below then asks whether the lint objected. Keeping them
    apart is what makes the failure message say which of the two broke.

    Compares serialized form rather than object identity, so a mutation that
    writes the same value back is caught too — that is the same no-op wearing a
    different hat.
    """
    import copy

    dead = []
    before = json.dumps(trace, ensure_ascii=False, sort_keys=True)
    for name, mutate in SABOTAGE_CASES.items():
        t = copy.deepcopy(trace)
        try:
            mutate(t)
        except Exception as exc:
            dead.append(f"{name}: 破坏本身失败 {exc!r}")
            continue
        if json.dumps(t, ensure_ascii=False, sort_keys=True) == before:
            dead.append(f"{name}: 破坏之后 trace 逐字节没变 —— "
                        "这个用例在本配置下是空操作，等于没测")
    return dead


def sabotage_checks(trace):
    """Prove the lint is not a function that always returns zero.

    Every mutation breaks exactly one rule on a copy of the real trace. A rule
    no breakage can trip is a rule that is not being tested, so every one must
    be caught or the script refuses to write.

    Callers run `sabotage_meta_checks` first: a mutation that changed nothing
    reports "0 gaps" here too, and would be misreported as the lint failing.
    """
    import copy

    failures = []
    for name, mutate in SABOTAGE_CASES.items():
        t = copy.deepcopy(trace)
        try:
            mutate(t)
        except Exception as exc:
            failures.append(f"{name}: 破坏本身失败 {exc!r}")
            continue
        try:
            gaps = lint(t)[0]
        except Exception as exc:
            failures.append(f"{name}: lint 抛出 {exc!r}")
            continue
        if not gaps:
            failures.append(f"{name}: 破坏了 trace 但 lint 报 0 gap")
    return failures


def lint_sabotage_only(trace):
    """Inline check: the sabotage table is applied to the real trace before any
    file is written, so a rule that stopped firing stops the write."""
    return sabotage_checks(trace)


# ----------------------------------------------------------------- formulas
#
# One formula per step kind, with the step's own index substituted. The
# rank-relative chunk index stays symbolic on purpose: it is DIFFERENT for
# every rank (that is the whole algorithm), so a single number would be a lie.
# The scalars that really are uniform across ranks — the chunk size, the bytes
# per rank per step, how many chunks are complete — are the slots, and they are
# what the reader is meant to carry away.

def _f(x):
    """A float32 under the JSON contract: a finite number. Commas stripped, so
    a value can be dropped straight into a `num` binding."""
    return repr(round(float(x), 6)).replace(",", "")


def rs_formula(n, chunk_elements, k):
    left = n - 1 - k
    return {
        "sym": ("\\underbrace{\\text{buf}_r\\!\\left[(r-k-1)\\bmod N\\right] "
                "\\mathrel{+}= \\text{buf}_{r-1}\\!\\left[(r-k-1)\\bmod N\\right]}"
                "_{\\text{收：上游那一块，就地累加}}"
                "\\qquad"
                "\\underbrace{\\text{buf}_r\\!\\left[(r-k)\\bmod N\\right] \\to r+1}"
                "_{\\text{发：自己那一块，传给下游}}"
                "\\qquad |\\text{chunk}| = \\frac{\\slot{S}}{\\slot{N}} = \\slot{CH}"),
        "idx": ("\\underbrace{\\text{buf}_r\\!\\left[(r-\\slot{K}-1)\\bmod \\slot{N}\\right] "
                "\\mathrel{+}= \\text{buf}_{r-1}\\!\\left[(r-\\slot{K}-1)\\bmod \\slot{N}\\right]}"
                "_{\\text{收：上游那一块，就地累加}}"
                "\\qquad"
                "\\underbrace{\\text{buf}_r\\!\\left[(r-\\slot{K})\\bmod \\slot{N}\\right] \\to r+1}"
                "_{\\text{发：自己那一块，传给下游}}"
                "\\qquad |\\text{chunk}| = \\frac{\\slot{S}}{\\slot{N}} = \\slot{CH}"),
        "num": ("\\underbrace{\\text{buf}_r\\!\\left[(r-\\slot{K}-1)\\bmod \\slot{N}\\right] "
                "\\mathrel{+}= \\text{buf}_{r-1}\\!\\left[(r-\\slot{K}-1)\\bmod \\slot{N}\\right]}"
                "_{\\text{每卡每步收发各 }\\slot{CH}\\text{ 个元素}}"
                "\\qquad"
                "\\text{本步每卡收发 } \\slot{BYTES}\\ \\mathrm{B}"
                "\\qquad \\text{已归约 } \\slot{RED} / \\slot{N} \\text{ 块}"),
        "bindings": {
            "S": {"sym": "S", "idx": "S", "num": str(S)},
            "N": {"sym": "N", "idx": "N", "num": str(n)},
            "K": {"sym": "k", "idx": "k", "num": str(k)},
            "CH": {"sym": "S/N", "idx": f"{S}/{n}", "num": str(chunk_elements)},
            "BYTES": {"sym": "(S/N)\\cdot b", "idx": f"({S}/{n})\\times {ITEM_BYTES}",
                      "num": str(chunk_elements * ITEM_BYTES)},
            "RED": {"sym": "k+1", "idx": f"{k}+1", "num": str(k + 1)},
        },
        "regions": {
            # At k = 0 there is no previous step to point back at: the chunk
            # being received still holds only the sender's own contribution,
            # and this receive is the first accumulation into it.
            "RECV": (f"第 {k} 步收的那一块，此刻还只含上游卡自己那一份数据；"
                     f"这一步把它累加进来，这一块的计数从 1 变成 2。"
                     if k == 0 else
                     f"第 {k} 步收的那一块，是上游卡在第 {k - 1} 步攒出来的；"
                     f"每收一次，这一块里含的数据份数就 +1。"),
            "REMAIN": f"ReduceScatter 还剩 {left} 步。全部 {n - 1} 步走完之后，"
                      f"每张卡手里恰好有 1 块是「全局求和完成」的 —— "
                      f"而且是各不相同的那一块。",
        },
    }


def ag_formula(n, chunk_elements, k):
    left = n - 1 - k
    return {
        "sym": ("\\underbrace{\\text{buf}_r\\!\\left[(r-k)\\bmod N\\right] "
                "\\leftarrow \\text{buf}_{r-1}\\!\\left[(r-k)\\bmod N\\right]}"
                "_{\\text{收：上游已归约好的那一块（覆盖，不是累加）}}"
                "\\qquad"
                "\\underbrace{\\text{buf}_r\\!\\left[(r-k+1)\\bmod N\\right] \\to r+1}"
                "_{\\text{发：自己手里已归约好的那一块}}"
                "\\qquad |\\text{chunk}| = \\slot{CH}"),
        "idx": ("\\underbrace{\\text{buf}_r\\!\\left[(r-\\slot{K})\\bmod \\slot{N}\\right] "
                "\\leftarrow \\text{buf}_{r-1}\\!\\left[(r-\\slot{K})\\bmod \\slot{N}\\right]}"
                "_{\\text{收：上游已归约好的那一块（覆盖，不是累加）}}"
                "\\qquad"
                "\\underbrace{\\text{buf}_r\\!\\left[(r-\\slot{K}+1)\\bmod \\slot{N}\\right] \\to r+1}"
                "_{\\text{发：自己手里已归约好的那一块}}"
                "\\qquad |\\text{chunk}| = \\slot{CH}"),
        "num": ("\\text{本步每卡收发 } \\slot{BYTES}\\ \\mathrm{B}"
                "\\qquad \\text{已有 } \\slot{HAVE} / \\slot{N} \\text{ 块归约结果}"
                "\\qquad \\text{本步搬的是纯拷贝，不做任何加法}"),
        "bindings": {
            "N": {"sym": "N", "idx": "N", "num": str(n)},
            "K": {"sym": "k", "idx": "k", "num": str(k)},
            "CH": {"sym": "S/N", "idx": f"{S}/{n}", "num": str(chunk_elements)},
            "BYTES": {"sym": "(S/N)\\cdot b", "idx": f"({S}/{n})\\times {ITEM_BYTES}",
                      "num": str(chunk_elements * ITEM_BYTES)},
            "HAVE": {"sym": "k+1", "idx": f"{k}+1", "num": str(k + 1)},
        },
        "regions": {
            "COPY": "这一阶段是「覆盖」而不是「累加」：收到的块已经是全局求和结果，"
                    "再加一次就重复计了。传输契约里的 mode 字段记的就是这件事。",
            "REMAIN": f"AllGather 还剩 {left} 步。走完之后每张卡的 {n} 块全部齐全。",
        },
    }


def init_formula(n, chunk_elements):
    return {
        "sym": ("\\text{buf}_r \\leftarrow \\mathrm{chunk}\\left(\\text{grad}_r, N\\right)"
                "\\qquad \\text{grad} \\in \\mathbb{R}^{N \\times \\slot{S}}"
                "\\quad\\longrightarrow\\quad \\text{buf} \\in \\mathbb{R}^{N \\times N \\times \\slot{CH}}"),
        "idx": ("\\text{buf}_r \\leftarrow \\mathrm{chunk}\\left(\\text{grad}_r, \\slot{N}\\right)"
                "\\qquad \\text{grad} \\in \\mathbb{R}^{\\slot{N} \\times \\slot{S}}"
                "\\quad\\longrightarrow\\quad \\text{buf} \\in \\mathbb{R}^{\\slot{N} \\times \\slot{N} \\times \\slot{CH}}"),
        "num": ("\\underbrace{\\text{每卡 } \\slot{S} \\text{ 个元素}}_{\\text{自己那份梯度}}"
                "\\quad\\xrightarrow{\\;\\text{切成 } \\slot{N} \\text{ 块}\\;}\\quad"
                "\\underbrace{\\slot{N} \\text{ 块} \\times \\slot{CH} \\text{ 元素}}"
                "_{\\text{每块大小 } S/N}"),
        "bindings": {
            "S": {"sym": "S", "idx": "S", "num": str(S)},
            "N": {"sym": "N", "idx": "N", "num": str(n)},
            "CH": {"sym": "S/N", "idx": f"{S}/{n}", "num": str(chunk_elements)},
        },
        "regions": {
            "SEED": "每一块里现在只有本卡自己那一份数据（计数 1）。整个算法要做的，"
                    "就是让每块里的份数从 1 涨到 N。",
            "WHY": f"为什么要切 {n} 块：每步只搬 1/{n} 的数据，链路才不会像"
                   "中心化方案那样被一张卡的带宽卡死。",
        },
    }


# -------------------------------------------------------------- trace build


def rank_specs(n):
    return [{"id": f"r{i}", "label": f"rank {i}"} for i in range(n)]


def tensor_specs(n, chunks, chunk_elements):
    """`grad` plus one buffer tensor per rank.

    The buffers are real tensors in the trace, not a private side-channel,
    which buys two things: the engine's tensor inspector can show them, and the
    ring view reads their values out of `resolve(trace, cursor)` like every
    other view — so the numbers under the topology come from the same pure
    reconstruction as the rest of the page.
    """
    spec = {
        "grad": {
            "shape": [n, S], "dtype": "fp32", "at": "HBM", "role": "input",
            "note": f"{n} 张卡各自的梯度，每卡 {S} 个元素",
        },
    }
    for r in range(n):
        spec[f"buf_{r}"] = {
            "shape": [chunks, chunk_elements], "dtype": "fp32", "at": "GPU",
            "role": "buffer",
            "note": f"rank {r} 的缓冲区：{chunks} 块 × {chunk_elements} 个元素",
        }
    return spec


def nested(t):
    """torch tensor -> nested lists, the engine's tensor encoding."""
    return t.tolist()


def buffers_state(buffers):
    return {f"buf_{r}": nested(buffers[r]) for r in range(buffers.shape[0])}


def build_graph(n, total_steps):
    nodes = [{"id": "init", "title": f"切开：每卡 {S} 个元素 → {n} 块",
              "kind": KIND_STATE, "op": "chunk", "phase": PHASE_INIT}]
    for s in range(total_steps):
        k = s % (n - 1)
        phase = PHASE_REDUCE if s < n - 1 else PHASE_GATHER
        nodes.append({
            "id": f"s{s}",
            "title": f"{phase} 第 {k + 1} / {n - 1} 步",
            "kind": KIND_COMM, "op": "sendrecv", "phase": phase,
        })
    edges = []
    prev = "init"
    for s in range(total_steps):
        edges.append({"from": prev, "to": f"s{s}", "tensor": ""})
        prev = f"s{s}"
    return {"nodes": nodes, "edges": edges}


def build_naive_entry(n, x, ref_sum):
    """The hub control schedule, its buffers, and its traffic.

    It is executed on real tensors too — `naive_allreduce` — so the trace can
    state that the strawman reaches the same values. A comparison that only
    compared byte counts would be comparing annotations.
    """
    sched = naive_schedule(n)
    buf = naive_allreduce(x, n)
    chunks = 1
    curves = cumulative_curves(n, S, sched, links=1)
    counts = counts_from_sends(n, chunks, sched)
    return {
        "label": "朴素中心化",
        "note": "所有卡发给 rank 0 求和，rank 0 再广播回去",
        "chunk_elements": S,
        "chunks": chunks,
        "steps": [
            {"id": f"n{i}", "phase": st["phase"], "k": st["k"],
             "transfers": [{"from": _step_ref(s["from"]), "to": _step_ref(s["to"]),
                            "chunk": s["chunk"], "elements": S, "mode": s["mode"]}
                           for s in st["transfers"]]}
            for i, st in enumerate(sched)
        ],
        "counts": counts,
        "buffers": {f"buf_{r}": nested(buf[r]) for r in range(n)},
        "curves": curves,
        "final_matches": bool(torch.allclose(buf[:, 0], ref_sum, rtol=TOL, atol=TOL)),
        "final_max_delta": float((buf[0, 0] - ref_sum).abs().max()),
    }


def build_tree_entry(n, x, ref_sum):
    sched = tree_schedule(n)
    buf = tree_allreduce(x, n)
    curves = cumulative_curves(n, S, sched, links=n - 1)
    counts = counts_from_sends(n, 1, sched)
    return {
        "label": "Tree AllReduce",
        "note": "树形归约到根再广播，每步一整层，延迟 O(log N)",
        "chunk_elements": S,
        "chunks": 1,
        "ranks": rank_specs(n),
        "edges": tree_edges(n),
        "depths": [tree_depth(r) for r in range(n)],
        "steps": [
            {"id": f"t{i}", "phase": st["phase"], "k": st["k"],
             "transfers": [{"from": _step_ref(s["from"]), "to": _step_ref(s["to"]),
                            "chunk": 0, "elements": S, "mode": s["mode"]}
                           for s in st["transfers"]]}
            for i, st in enumerate(sched)
        ],
        "counts": counts,
        "buffers": {f"buf_{r}": nested(buf[r]) for r in range(n)},
        "curves": curves,
        "final_matches": bool(torch.allclose(buf[:, 0], ref_sum, rtol=TOL, atol=TOL)),
        "final_max_delta": float((buf[0, 0] - ref_sum).abs().max()),
    }


# -------------------------------------------------------------- the timing model
#
# Stated as a model, because it is one. `bytes / beta` is the bandwidth term and
# `steps * alpha` is the fixed cost of issuing that many link-level transfers;
# both are what a reader would compute by hand from the two numbers. The point
# is not the absolute milliseconds — it is that Ring and Tree cross over, which
# is why NCCL ships both and picks per message size.

def time_us(steps, per_link_bytes, nominal):
    """Completion time for a schedule at a message of `nominal` bytes.

    `per_link_bytes` is the load on the busiest link for THIS message size; the
    schedule finishes when that link has finished, which is why it — and not
    the aggregate — is the bandwidth term.
    """
    scale = nominal / (S * ITEM_BYTES)
    return steps * ALPHA_US + (per_link_bytes * scale) / (BETA_GBS * 1e9) * 1e6


def timing_table(n, curves):
    rows = []
    for name, curves_for in curves.items():
        steps = curves_for["steps"]
        per_link = curves_for["link_load_total"]
        t = time_us(steps, per_link, NOMINAL_BYTES)
        # algbw is the goodput the application sees; busbw rescales it by the
        # factor a ring's rotation costs, which is the convention NCCL reports.
        algbw = NOMINAL_BYTES / (t * 1e-6) if t > 0 else 0.0
        busbw = algbw * (2 * (n - 1) / n) if name == "ring" else algbw
        rows.append({
            "schedule": name,
            "steps": steps,
            "per_link_bytes": per_link,
            "per_link_bytes_at_nominal": int(round(per_link * NOMINAL_BYTES / (S * ITEM_BYTES))),
            "links": curves_for["links"],
            "aggregate_bytes": curves_for["total"],
            "time_us": round(t, 3),
            "algbw_gbs": round(algbw / 1e9, 2),
            "busbw_gbs": round(busbw / 1e9, 2),
        })
    return rows


def crossover_bytes(n, curves):
    """The message size at which ring and tree cost the same.

    Solving `steps_r*alpha + b_r*B/(S*b) / beta = steps_t*alpha + b_t*B/(S*b)/beta`
    for B. Returned so the assumption behind it (alpha, beta) travels with it.
    """
    r, t = curves["ring"], curves["tree"]
    delta_steps = r["steps"] - t["steps"]
    delta_bytes = r["link_load_total"] - t["link_load_total"]
    if delta_bytes == 0:
        return None
    # time = steps*alpha (us) + bytes/B_scale / (beta GB/s) -> seconds
    # In units of the nominal message B: bytes(B) = bottleneck * B / (S*b)
    # delta_steps*alpha*1e-6 + delta_bytes*(B/(S*b))/(beta*1e9) = 0
    alpha_s = delta_steps * ALPHA_US * 1e-6
    per_byte = delta_bytes / (S * ITEM_BYTES) / (BETA_GBS * 1e9)
    if per_byte == 0:
        return None
    b = -alpha_s / per_byte
    if b <= 0:
        return None     # no crossover in the positive direction
    return int(round(b))


# --------------------------------------------------------------- the builder


def build_trace(n):
    x = make_input(n)
    x_int = make_input(n, integral=True)
    chunk_elements = S // n
    chunks = n
    sched = ring_schedule(n)

    # ---- the real run
    buffers, frames = run_schedule(x, n, chunk_elements, sched)
    ref_sum = reference_allreduce(x)                     # (S,)
    ref_ring = reference_ring(x, n)                      # (n, chunks, chunk_el)

    counts = counts_from_sends(n, chunks, sched)
    masks = masks_from_sends(n, chunks, sched)

    # ---- the control topologies
    naive_buf = naive_allreduce(x, n)
    tree_buf = tree_allreduce(x, n)
    int_buf, _ = run_schedule(x_int, n, chunk_elements, sched)
    seq_buf = run_schedule_sequential(x, n, chunk_elements, sched)

    # The ReduceScatter prefix, so the "exactly one complete chunk each" prop
    # can be checked on the frame where it is supposed to be true rather than on
    # the end of the whole run.
    rs_frames = n - 1
    rs_counts = counts_from_sends(n, chunks, sched[:rs_frames])

    # `links` is how many links carried the traffic, and it is what turns the
    # (identical) aggregate into the per-link load the comparison is about: the
    # ring uses all N outbound links, the hub uses one, the tree uses its N-1
    # edges.
    curves = {
        "ring": cumulative_curves(n, chunk_elements, sched, links=n),
        "naive": cumulative_curves(n, S, naive_schedule(n), links=1),
        "tree": cumulative_curves(n, S, tree_schedule(n), links=n - 1),
    }

    # ---- assertions: the schedule is right (exact)
    claims = []
    if not torch.equal(buffers, ref_ring):
        claims.append("环形跑出来的缓冲区与「按环形累加顺序」的参考实现不逐位相同 "
                      "—— 调度本身错了")
    for r in range(n):
        got = buffers[r].reshape(S)
        if not torch.allclose(got, ref_sum, rtol=TOL, atol=TOL):
            claims.append(f"rank {r} 的最终 8 个元素与 torch 的 x.sum(0) 不一致"
                          f"（最大偏差 "
                          f"{float((got - ref_sum).abs().max()):.3e}）")
    # every rank agrees with every other rank, exactly: AllReduce's contract is
    # that they end identical, not merely close.
    for r in range(1, n):
        if not torch.equal(buffers[r], buffers[0]):
            claims.append(f"rank {r} 与 rank 0 的最终缓冲区不是逐位相同 —— "
                          "AllReduce 的结果必须每张卡完全一致")

    # ---- assertions: the residual is summation order, not an error
    if not torch.equal(int_buf[0].reshape(S), x_int.sum(dim=0)):
        claims.append("整数取值下环形与 x.sum(0) 不逐位相同 —— 那么上面的容差差的"
                      "就不是「求和顺序」")
    max_delta = float((buffers[0].reshape(S) - ref_sum).abs().max())
    # The worst-case bound for reordering N fp32 addends: (N-1)·eps·Σ|x|.
    # Loose by design — it is what the tolerance below has to be argued against.
    bound = (n - 1) * EPS_FP32 * float(x.abs().sum(dim=1).max())
    if max_delta > TOL:
        claims.append(f"浮点残差 {max_delta:.3e} 超过容差 {TOL:.0e}")

    # ---- assertions: the control topologies agree
    for name, buf in (("朴素中心化", naive_buf), ("Tree", tree_buf)):
        got = buf[0].reshape(S)
        if not torch.allclose(got, ref_sum, rtol=TOL, atol=TOL):
            claims.append(f"{name} 的最终结果与 x.sum(0) 不一致"
                          f"（最大偏差 {float((got - ref_sum).abs().max()):.3e}）")
        for r in range(1, n):
            if not torch.equal(buf[r], buf[0]):
                claims.append(f"{name}：rank {r} 与 rank 0 的最终结果不是逐位相同")

    # ---- assertions: the conservation laws the lab exists to show
    for r in range(n):
        for c in range(chunks):
            if counts[r][c] != len(masks[r][c]) or counts[r][c] != n:
                claims.append(f"结束状态：rank {r} 的 chunk {c} 计数 {counts[r][c]}、"
                              f"数据集合 {sorted(masks[r][c])}（应为 {n} 份）")
                break
    for r in range(n):
        done = [c for c in range(chunks) if rs_counts[r][c] == n]
        if len(done) != 1:
            claims.append(f"ReduceScatter 结束时 rank {r} 有 {len(done)} 块完成，"
                          "应当恰好 1 块")
        elif done[0] != (r + 1) % n:
            claims.append(f"ReduceScatter 结束时 rank {r} 完成的是 chunk {done[0]}，"
                          f"应当是 chunk {(r + 1) % n}")
    # `sched` is the raw schedule (`sends`); its per-send element count is the
    # chunk size, which is the fact being asserted.
    sent_per_rank = [
        sum(chunk_elements for st in sched for s in st["transfers"] if s["from"] == r)
        for r in range(n)
    ]
    for r, sent in enumerate(sent_per_rank):
        want = 2 * (n - 1) * chunk_elements
        if sent != want:
            claims.append(f"rank {r} 总发送 {sent} 个元素，应当是 2(N-1)·S/N = {want}")

    # ---- assertions: the quantitative claim the accumulator makes.
    #
    # All three schedules move the SAME aggregate bytes — that is the identity
    # the whole comparison rests on, and asserting it first is what stops the
    # factor of N from being a comparison of three different workloads. What
    # differs is the per-link load: the ring spreads that aggregate over N
    # links, the hub puts it down one, the tree over N-1.
    aggs = {name: c["total"] for name, c in curves.items()}
    if len(set(aggs.values())) != 1:
        claims.append(f"三条调度的总搬运量不同：{aggs} —— "
                      "那么下面的倍数比的就不是「几条链路分摊同一份流量」")
    naive_link = curves["naive"]["link_load_total"]
    ring_link = curves["ring"]["link_load_total"]
    tree_link = curves["tree"]["link_load_total"]
    if naive_link != 2 * (n - 1) * S * ITEM_BYTES:
        claims.append(f"朴素方案的单链路负载 {naive_link} B，"
                      f"应当是 2(N-1)S = {2 * (n - 1) * S * ITEM_BYTES} B")
    if ring_link != 2 * (n - 1) * chunk_elements * ITEM_BYTES:
        claims.append(f"环形方案的单链路负载 {ring_link} B，"
                      f"应当是 2(N-1)S/N = {2 * (n - 1) * chunk_elements * ITEM_BYTES} B")
    if naive_link != n * ring_link:
        claims.append(f"朴素 / 环形的单链路负载比是 "
                      f"{naive_link / ring_link:.3f}，应当是 N = {n}")
    # `naive_link == 2(N-1)S` and `ring_link == 2(N-1)S/N` are the two formulas
    # the tutorial prints. Their ratio is the factor of N the lab is about, and
    # it follows from the two lines above rather than being asserted separately
    # — but it is asserted separately anyway, because it is the ONE number a
    # reader will check and an error that only showed up in the ratio would be
    # the hardest kind to notice.
    if tree_link != 2 * S * ITEM_BYTES:
        claims.append(f"Tree 的单链路负载 {tree_link} B，应当是 2S = "
                      f"{2 * S * ITEM_BYTES} B（总量 2(N-1)S 摊到 N-1 条边上）")
    step_bytes = {curves["ring"]["per_step"][si][r] // 1
                  for si in range(len(sched)) for r in range(n)
                  if curves["ring"]["per_step"][si][r]}
    if step_bytes != {chunk_elements * ITEM_BYTES}:
        claims.append(f"环形每步搬运的字节数是 {sorted(step_bytes)}，"
                      "应当只有 S/N 这一个值")

    # ---- assertions: the simultaneity property, and its control group.
    #
    # The ring's rotation never reads a slot it writes in the same step, which
    # is why the in-place loop below agrees with the snapshot version. That is
    # a property of THIS schedule, so it is asserted structurally (not by
    # comparing two runs and concluding the property from their agreement),
    # and then shown to be load-bearing: mutate the schedule into a
    # self-clashing one and the two runs must part company.
    clashes = read_write_disjoint(n, sched)
    if clashes:
        claims.append(f"环形的调度出现了同一 rank 同步读写同一个 chunk 的情况：{clashes} —— "
                      "那么就地更新会读到被本步覆盖过的值")
    if not torch.equal(seq_buf, buffers):
        claims.append("就地更新版与快照版结果不同 —— 但 read_write_disjoint 说没有冲突，"
                      "两者本该一致。说明「一步是一次同时交换」的模型在这份数据上"
                      "没有被正确实现")
    demo = hazard_demo(x, n, chunk_elements)
    if demo is None:
        claims.append("无法构造出「同步读写同一 chunk」的调度 —— "
                      "那么就无法证明快照与就地更新的差别真的存在")
    elif not demo["disagree"]:
        claims.append("构造出的自冲突调度下，快照版与就地版结果仍然相同 —— "
                      "那么「一步要一次同时交换」这个说法没有区分力")

    # ---- build the step list
    steps = []
    steps.append({
        "id": "init",
        "title": f"把每卡 {S} 个元素的梯度切成 {n} 块",
        "kind": KIND_STATE, "op": "chunk", "phase": PHASE_INIT,
        "phase_id": STEP_REDUCE, "k": 0,
        "formula": _formula_tiers(init_formula(n, chunk_elements)),
        "bindings": _formula_bindings(init_formula(n, chunk_elements)),
        "regions": init_formula(n, chunk_elements)["regions"],
        "reads": ["grad"],
        "writes": [f"buf_{r}" for r in range(n)],
        "state": {f"grad": nested(x)},
        "transfers": [],
        "narration": (
            f"这张梯度张量每卡一份，每份 {S} 个元素。AllReduce 要的结果是："
            f"每卡最终都拿到 {S} 个元素的全局和。朴素做法是把 {n} 份都塞给一张卡，"
            f"那张卡的网络链路要独自扛住 {2 * (n - 1)} 份数据的进出 —— 这就是环形要"
            f"解决的问题。先把每份切成 {n} 块，每块 {chunk_elements} 个元素；"
            f"每一块现在只含本卡自己那一份数据，计数 1。"),
    })

    for si, st in enumerate(sched):
        phase = PHASE_REDUCE if st["phase"] == STEP_REDUCE else PHASE_GATHER
        k = st["k"]
        f = rs_formula(n, chunk_elements, k) if st["phase"] == STEP_REDUCE \
            else ag_formula(n, chunk_elements, k)
        sends = [{"from": _step_ref(s["from"]), "to": _step_ref(s["to"]),
                  "chunk": s["chunk"], "elements": chunk_elements, "mode": s["mode"]}
                 for s in st["transfers"]]
        if st["phase"] == STEP_REDUCE:
            title = f"ReduceScatter 第 {k + 1} / {n - 1} 步：收一块累加、发一块"
            # Which chunk each rank touches is r-dependent, so the narration
            # names the RULE rather than one rank's index — a single number
            # here would be true of rank 0 and wrong for the other N-1.
            narr = (
                f"每张卡同时做两件事：把自己手里那个还没传过的块发给下游，"
                f"并把上游发来的那一块累加进本地。各卡操作的 chunk 下标各不相同 —— "
                f"这正是环形能同时用到所有链路的原因。本步每卡收发各 "
                f"{chunk_elements} 个元素（{chunk_elements * ITEM_BYTES} 字节）。"
                f"{n - 1 - k} 步之后，每张卡手里会恰好有 1 块是全局求和完成的。")
        else:
            title = f"AllGather 第 {k + 1} / {n - 1} 步：把已完成的那块传遍全环"
            narr = (
                f"ReduceScatter 已经让每张卡各持有一块完整的归约结果。现在沿同一个环"
                f"再转 {n - 1} 步，每步把手里那块已完成的传给下游、从上游收一块 —— "
                f"注意这一步是「覆盖」而不是「累加」：收到的块已经是全局和，"
                f"再加一次就重复计了。{n - 1 - k} 步之后，每张卡集齐 {n} 块。")
        steps.append({
            "id": f"s{si}",
            "title": title,
            "kind": KIND_COMM, "op": "sendrecv", "phase": phase,
            "phase_id": st["phase"], "k": k,
            "formula": _formula_tiers(f),
            "bindings": _formula_bindings(f),
            "regions": f["regions"],
            "reads": [f"buf_{r}" for r in range(n)],
            "writes": [f"buf_{r}" for r in range(n)],
            "state": buffers_state(frames[si + 1]),
            "transfers": sends,
            "narration": narr,
        })

    ring_link_bytes = curves["ring"]["link_load_total"]
    naive_link_bytes = curves["naive"]["link_load_total"]
    tree_link_bytes = curves["tree"]["link_load_total"]
    aggregate_bytes = curves["ring"]["total"]

    trace = {
        "meta": {
            "lab": "L13",
            "title": f"Ring AllReduce 逐帧回放 · {n} 卡",
            "source": "docs/guides/模块三-分布式训练/2.1 集合通信原语详解.md §4；"
                      "docs/guides/模块一-前置知识/communication/"
                      "collective-communication-primer.md §5–§7",
            "config": {
                "N": n,
                "S": S,
                "chunk_elements": chunk_elements,
                "chunks": chunks,
                "item_bytes": ITEM_BYTES,
                "algorithm": "ring",
                "alpha_us": ALPHA_US,
                "beta_gbs": BETA_GBS,
                "nominal_bytes": NOMINAL_BYTES,
            },
            "reference": _reference_note(n, max_delta, bound),
            "traffic": {
                "ring": {
                    "label": "Ring AllReduce",
                    "link_load_bytes": ring_link_bytes,
                    "links": n,
                    "aggregate_bytes": curves["ring"]["total"],
                    "per_rank_bytes": curve_single(curves["ring"])["per_rank_total"],
                    "steps": curves["ring"]["steps"],
                    "closed_form": f"2(N-1)·S/N = 2·{n - 1}·{S}/{n} 元素 "
                                   f"= {2 * (n - 1) * S // n} 元素",
                },
                "naive": {
                    "label": "朴素中心化",
                    "link_load_bytes": naive_link_bytes,
                    "links": 1,
                    "aggregate_bytes": curves["naive"]["total"],
                    "per_rank_bytes": curve_single(curves["naive"])["per_rank_total"],
                    "steps": curves["naive"]["steps"],
                    "closed_form": f"2(N-1)·S = 2·{n - 1}·{S} 元素 "
                                   f"= {2 * (n - 1) * S} 元素（rank 0 的链路）",
                },
                "tree": {
                    "label": "Tree AllReduce",
                    "link_load_bytes": tree_link_bytes,
                    "links": n - 1,
                    "aggregate_bytes": curves["tree"]["total"],
                    "per_rank_bytes": curve_single(curves["tree"])["per_rank_total"],
                    "steps": curves["tree"]["steps"],
                    "closed_form": f"总量 2(N-1)S 摊到 N-1 条边上 = 2S "
                                   f"= {2 * S} 元素/链路",
                },
                "aggregate_bytes": aggregate_bytes,
                "ratio_naive_over_ring": round(naive_link_bytes /
                                               ring_link_bytes, 6),
                "ratio_naive_over_ring_closed_form": n,
            },
            "timing": {
                "alpha_us": ALPHA_US,
                "beta_gbs": BETA_GBS,
                "nominal_bytes": NOMINAL_BYTES,
                "rows": timing_table(n, curves),
                "crossover_bytes": crossover_bytes(n, curves),
                "note": "α = 每步固定启动开销，β = 单链路单向带宽。数量级取自 "
                        "NVLink + NCCL 的常见值，用于展示 Ring 与 Tree 的交叉点，"
                        "不是实测。",
            },
            "metrics": {
                "ring_link_bytes": ring_link_bytes,
                "naive_link_bytes": naive_link_bytes,
                "tree_link_bytes": tree_link_bytes,
                "aggregate_bytes": aggregate_bytes,
                "ring_total_bytes": curves["ring"]["total"],
                "naive_total_bytes": curves["naive"]["total"],
                "tree_total_bytes": curves["tree"]["total"],
                "ring_steps": curves["ring"]["steps"],
                "naive_steps": curves["naive"]["steps"],
                "tree_steps": curves["tree"]["steps"],
                "naive_over_ring": round(naive_link_bytes /
                                         ring_link_bytes, 6),
            },
        },
        "tensors": tensor_specs(n, chunks, chunk_elements),
        "ring": {
            "n": n,
            "algorithm": "ring",
            "op": {
                "id": "grad",
                "label": "AllReduce(grad)",
                "tensor": "grad",
                "elements": S,
                "bytes": S * ITEM_BYTES,
                "dtype": "fp32",
            },
            "ranks": rank_specs(n),
            "chunk_elements": chunk_elements,
            "chunks": chunks,
            "seed_count": 1,
            "phases": [
                {"id": STEP_REDUCE, "label": "ReduceScatter",
                 "note": f"{n - 1} 步 · 每卡每步收 1 块累加、发 1 块"},
                {"id": STEP_GATHER, "label": "AllGather",
                 "note": f"{n - 1} 步 · 每卡每步收 1 块覆盖、发 1 块"},
            ],
            "compare": {
                "naive": build_naive_entry(n, x, ref_sum),
                "tree": build_tree_entry(n, x, ref_sum),
            },
            "curves": {
                name: {
                    "busiest_cumulative": c["busiest_cumulative"],
                    # Per rank, after each of THIS schedule's steps. The view
                    # folds the ring's own per-rank series out of the trace's
                    # steps and the acceptance harness compares the two; without
                    # it stated here the comparison would have nothing to be
                    # against. (The trace's step list begins with an `init`
                    # frame that moves nothing, so the view's series is this one
                    # with a leading all-zero row — a relationship the harness
                    # asserts rather than assumes.)
                    "per_rank_cumulative": c["per_rank_cumulative"],
                    "per_rank_total": c["per_rank_total"],
                    "busiest_total": c["busiest_total"],
                    "link_load_total": c["link_load_total"],
                    "links": c["links"],
                    "total": c["total"],
                    "steps": c["steps"],
                }
                for name, c in curves.items()
            },
        },
        "steps": steps,
        "graph": build_graph(n, len(sched)),
    }

    # `init` writes grad again so the tensor panel has a value at step 0 even
    # though it is the same array as the init — the engine reads `init` for the
    # initial frame anyway, and recording it keeps writes/state consistent.
    trace["steps"][0]["state"] = {
        "grad": nested(x),
        **{f"buf_{r}": nested(frames[0][r]) for r in range(n)},
    }
    return trace, claims, {
        "counts": counts,
        "masks": masks,
        "rs_counts": rs_counts,
        "curves": curves,
        "sent_per_rank": sent_per_rank,
        "seq_disjoint": not clashes,
        "hazard_disagree": bool(demo and demo["disagree"]),
        "max_delta": max_delta,
        "bound": bound,
        "sched": sched,
        "chunk_elements": chunk_elements,
    }


def curve_single(c):
    return c


def _formula_tiers(f):
    return {"sym": f["sym"], "idx": f["idx"], "num": f["num"]}


def _formula_bindings(f):
    return f["bindings"]


def _reference_note(n, max_delta, bound):
    return (
        f"三层对拍。① 与「按环形累加顺序」的独立 torch 参考实现逐位相同"
        f"（torch.equal，零容差）—— 断言的是调度：chunk 下标错一步、少一步、"
        f"把 add 写成 copy 都会改变累加顺序，在这里被抓到。"
        f"② 与 torch 的 x.sum(0)（与顺序无关的真值）在 fp32 下相对/绝对容差 "
        f"{TOL:.0e}。这个容差是推出来的，不是抄来的：重新排列 k 个 fp32 加数的"
        f"标准上界是 (k-1)·eps·Σ|x|，eps = 2^-24 ≈ 5.96e-8，本次 k = N = {n}、"
        f"Σ|x| ≤ {bound / ((n - 1) * EPS_FP32):.3f}，于是上界 {bound:.2e}；"
        f"实测最大偏差 {max_delta:.2e}，容差取其约 "
        f"{TOL / bound:.1f} 倍。"
        f"（L00 的 1e-12 是 float64 的值，在 fp32 下没有意义。）"
        f"③ 把输入换成整数取值的 fp32 后，每一步部分和都精确可表示，"
        f"两条路径必须逐位相同 —— 实数两种顺序与整数两种顺序只差在舍入，"
        f"所以这一跑把 ② 的残差坐实成「求和顺序」而不是别的。"
        f"对照组：朴素中心化与 Tree 两条调度在同样的容差下也到达同一个结果；"
        f"把一步内的传输按卡号顺序就地应用（而不是先取源值再写目标，"
        f"即「一步不是同时交换」的错误实现）会得到不同的答案，"
        f"所以「一步是一次同时交换」在这份数据上有区分力。"
    )


# ---------------------------------------------------------------------- main


def cfg_id(n):
    return f"ring-allreduce-N{n}"


def main():
    only_json = "--js-lint" in sys.argv
    real_stdout = sys.stdout
    if only_json:
        # Every human-readable line goes to stderr, so stdout carries the JSON
        # payload and nothing else — the parity harness pipes it straight into
        # `node`. A stray progress line would make the payload unparseable,
        # which reads as a harness bug rather than as this mode's own contract,
        # so the redirection is done once here rather than by threading `file=`
        # through a dozen print calls.
        sys.stdout = sys.stderr

    fails, written, built = [], [], {}
    for n in N_VALUES:
        name = cfg_id(n)
        trace, claims, info = build_trace(n)
        gaps, warns, infos = lint(trace)
        # Order matters: a dead mutation also reports "0 gaps" below, and the
        # two are different bugs with different fixes.
        dead = sabotage_meta_checks(trace)
        sabotages = [] if dead else sabotage_checks(trace)

        print(f"\n{name}: {len(trace['steps'])} 步 · N={n} · S={S} · "
              f"chunk={info['chunk_elements']} · "
              f"{len(json.dumps(trace, ensure_ascii=False))} bytes")
        c = info["curves"]
        print(f"  单链路负载（同一条总流量 {c['ring']['total']} B 摊到几条链路上）："
              f"环形 {c['ring']['link_load_total']} B / {c['ring']['links']} 条 · "
              f"朴素 {c['naive']['link_load_total']} B / 1 条 · "
              f"Tree {c['tree']['link_load_total']} B / {c['tree']['links']} 条 · "
              f"朴素/环形 = {c['naive']['link_load_total'] / c['ring']['link_load_total']:.2f}×")
        print(f"  步数：环形 {c['ring']['steps']} · 朴素 {c['naive']['steps']} · "
              f"Tree {c['tree']['steps']}；浮点残差 {info['max_delta']:.2e} "
              f"≤ 推导上界 {info['bound']:.2e}")
        for row in trace["meta"]["timing"]["rows"]:
            print(f"  耗时（{NOMINAL_BYTES // (1024 * 1024)} MiB 梯度）："
                  f"{row['schedule']:6s} {row['time_us']:9.1f} µs · "
                  f"algbw {row['algbw_gbs']:7.2f} GB/s · busbw {row['busbw_gbs']:7.2f} GB/s")
        cross = trace["meta"]["timing"]["crossover_bytes"]
        if cross:
            print(f"  Ring/Tree 交叉点：约 {cross / 1024 / 1024:.2f} MiB "
                  f"（更小用 Tree 省延迟，更大用 Ring 省带宽）")
        for line in infos:
            print(f"  info  {line}")
        for line in warns:
            print(f"  warn  {line}")
        for line in gaps:
            print(f"  GAP   {line}")
            fails.append(f"{name}: {line}")
        for line in claims:
            print(f"  CLAIM {line}")
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
        built[n] = trace

    if only_json:
        import copy
        # `base` names which trace the sabotages were cut from, so the JS side
        # can tell "the lint missed this" from "the mutation changed nothing" —
        # two different bugs (see `sabotage_meta_checks` here, and
        # `sabotageChecks` in the view).
        base = built[N_VALUES[1]]
        payload = {"traces": {cfg_id(n): built[n] for n in N_VALUES},
                   "base": cfg_id(N_VALUES[1]),
                   "sabotages": {}}
        for name, mutate in SABOTAGE_CASES.items():
            broken = copy.deepcopy(base)
            try:
                mutate(broken)
            except Exception as exc:      # a mutation that cannot even be applied
                payload["sabotages"][name] = {"error": repr(exc)}
                continue
            payload["sabotages"][name] = {"trace": broken}
        real_stdout.write(json.dumps(payload, ensure_ascii=False))
        return 0

    if fails:
        print("\nlint / 断言未通过，拒绝写文件：", file=sys.stderr)
        for line in fails:
            print("  - " + line, file=sys.stderr)
        return 1

    for n in N_VALUES:
        name = cfg_id(n)
        out = HERE / f"{name}.json"
        out.write_text(json.dumps(built[n], indent=1, ensure_ascii=False) + "\n")
        written.append(name)
        print(f"写 {out.name}  {out.stat().st_size} bytes")

    manifest = {
        "set": "ring-allreduce",
        "lab": "L13",
        "default": cfg_id(4),
        "params": {"N": list(N_VALUES)},
        "traces": written,
        "labels": {cfg_id(n): f"{n} 卡环形" for n in N_VALUES},
        "notes": {cfg_id(n): f"ReduceScatter {n - 1} 步 + AllGather {n - 1} 步，"
                             f"每步搬 S/{n} 个元素"
                 for n in N_VALUES},
    }
    (HERE / "ring-allreduce.manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"\nring-allreduce.manifest.json: {len(written)} 个配置，"
          f"默认 {manifest['default']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
