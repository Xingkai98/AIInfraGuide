#!/usr/bin/env python3
"""Trace generator for L10 · Continuous Batching 调度.

Emits one JSON file per configuration of `(schedule, scenario)`, plus a manifest
naming the set:

    labs/traces/continuous-batching-<schedule>-<scenario>.json
    labs/traces/continuous-batching.manifest.json

Both arms serve the *same* request table. `continuous` is the lab's subject;
`static` is its control, and the two are drawn side by side on a shared time
axis — that comparison is the whole teaching point of 2.2, so both are generated
from one set of arrivals and lengths rather than from two hand-tuned stories
that could quietly turn out not to be comparable.

WHAT IS COMPUTED HERE, AND WHY IT IS A REAL SIMULATION

Every tick is one call to a scheduler of the shape the tutorial's §6.2
pseudocode describes (vLLM V1): the token budget is handed out to the in-flight
requests first, and only what is left over — and only while nothing had to be
preempted — is offered to the waiting queue. The KV pool is a real finite
resource: a request's block count grows as it generates, an allocation that does
not fit preempts the newest in-flight peer, and recovery is **recompute**
(`num_computed_tokens` back to zero, the request returned to the front of the
waiting queue) exactly as §6.3 describes. Nothing here is a hand-drawn timeline;
the bars the page draws are a projection of the run.

The one modeling choice worth stating plainly: a tick costs one unit of time
regardless of batch size. That is not an accident of convenience — it is the
premise the whole technique rests on, and the tutorial states it in §1: decode is
memory-bound, so serving four tokens costs very nearly what serving one costs.
Under that premise the interesting question is not "how long is a tick" but "how
many of the seats are doing useful work", which is exactly what the slot
accounting below measures.

THE TWO ARMS DIFFER IN ONE PLACE ONLY

The per-tick core is shared. The arms differ in the admission policy that feeds
it, and in when a seat becomes free again:

  * **static** — build a batch, run it until *every* member has finished, then
    release the whole batch at once. A member that finished early keeps its SEAT
    for the rest of the batch, producing nothing; its `stalled` bars are that
    waste, drawn rather than described, and they are what the tutorial's §2
    "长尾请求拖死整批" costs in the unit the claim is about.
  * **continuous** — every tick, drop the members that finished and refill from
    the waiting queue. No seat is held by a finished request, so `stalled` is
    empty by construction — and that is asserted, not assumed.

Both arms respect the same `token_budget`, `max_num_seqs` and KV pool; static
simply cannot refill a seat mid-batch, which is what "batch-atomic" means.

WHAT THE NUMBERS ARE CHECKED AGAINST

There is no floating-point kernel in a scheduler, so the design doc's "与 PyTorch
参考实现对拍" has no literal analogue here, and copying a tolerance down from a
softmax would be the wrong move regardless. What replaces it is stronger for this
subject:

  * `reference_metrics()` recomputes **every published number** from the raw
    per-tick occupancy matrix with **torch tensor ops** — a path that never
    touches the scheduler's own counters and knows only about the picture the
    reader will see. The integers must match **exactly** (`np.array_equal`, zero
    tolerance) and the float64 ratios to 1e-12, with the same reasoning L00 wrote
    down: both sides evaluate the identical expression in float64, so the only
    possible difference is summation order. A tolerance would be actively worse
    here — one dropped or duplicated token is an error of 1, and any tolerance
    loose enough to be "safe" would hide exactly that.

  * A **token-conservation identity** ties the two ledgers together:
        Σ tokens computed per tick  ==  Σ(r.computed) + Σ(recompute_tokens)
        Σ(r.computed)  ==  Σ(prompt + output − 1)
    The second line is the interesting one. A seat that has finished computes
    NOTHING, so waste never appears in the token account at all — it appears in
    the seat account, which is the account §2 is about. Keeping the two apart is
    what stops a `stalled` block from being billed twice, once as a wasted seat
    and once as wasted arithmetic.

  * **Latency is seat residency**, not arrival-to-last-token: for each request,
    the number of ticks it held a seat after arriving. That is the quantity the
    queue delay and the batch-atomic hold both show up in.

  * Static's waste also has a **closed form** — for a batch that runs until its
    slowest member finishes, the wasted seat-ticks are Σ(max(output) − output)
    over the batch's members. It is computed from the batch compositions the run
    actually formed, by the definition of batch-atomic waiting rather than by
    counting the simulation's own stalls, and in the scenario where the run
    matches that description exactly the two must agree to the seat.

  * The claims that would make the lab vacuous if false are asserted outright,
    and the script refuses to write a file when one fails: every request finishes
    having generated exactly its declared length; the bars and the ledger agree
    exactly; the KV pool is never oversubscribed; continuous beats static on
    makespan, mean latency and utilization **by a margin the eye can see**; and
    the scenario switch does what it claims — `base` never preempts, `burst`
    does, and its victims really do re-prefill.

  * Both arms are asserted to have **no blocked seats** (`stalled` on a request
    that had not finished). With the pools this lab ships a live request always
    fits at least one more block, and a blocked seat would make the seat account
    and the picture disagree in a way none of the identities above would catch.

Contract notes (see docs/plans/interactive-labs.md §一 "轨迹数据模型"):

  * `tensors[].init` is mandatory for anything read but never written. Every
    tensor here is written from step 0 on, but they still carry an all-zero
    `init` so `render(trace, 0)` shows the queue tables before the first tick
    rather than "无值".
  * `step.state` is "every tensor value that changed during this step, in full"
    — never a delta. That is what makes arbitrary cursor jumps a pure function
    of (trace, cursor) with no undo log. Only the changed entries are recorded,
    which is the same rule applied per tensor.
  * Two fields are this view's own contract, linted on **both** sides — the
    Python rules below and the JS port in `labs/assets/engine/views/gantt.js`.
    (A view's contract belongs beside the view; `trace-model.js` is the engine's
    own lint and belongs to no single lab.) Both sides carry a sabotage control
    group, because a rule tested on one side only is a rule tested on neither.
      - `trace.gantt` — the rows (resources), their optional tracks, and the
        kind vocabulary.
      - `step.bars`   — what each row was doing during this step:
        `{row, kind, label?}`. The optional `label` is what L16 (micro-batch id)
        and L17 (bucket number) will put on their blocks; L10 leaves it off.

The script is deterministic (fixed scenario tables, no sampling), which is what
lets CI re-run it and require the committed JSON to be reproduced byte for byte
— see .github/workflows/trace-checks.yml.

Run: python3 labs/traces/continuous_batching.py
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ vocabulary
# The kind vocabulary the gantt draws with. Declared in trace.gantt.kinds and
# referenced by every bar, so a mistyped kind is a lint gap rather than a bar
# that silently renders with no style.
K_PREFILL = "prefill"      # admitted this tick: the whole prompt is computed
K_DECODE = "decode"        # one more token for a request already in flight
K_WAITING = "waiting"      # queued, not admitted — an empty seat, not a busy one
K_STALLED = "stalled"      # in the batch, already finished, holding a seat for nothing
K_PREEMPTED = "preempted"  # evicted this tick; its KV is released, it re-queues
KINDS = [K_PREFILL, K_DECODE, K_WAITING, K_STALLED, K_PREEMPTED]
USEFUL = (K_PREFILL, K_DECODE)

# Sentinel spellings for values JSON cannot represent. These are the contract's
# strings, not a local convention — the engine parses exactly these.
NEG_INF = r"-\infty"
POS_INF = r"+\infty"
NAN = "NaN"

# --------------------------------------------------------------- the requests
#
# The scenario is a table, not a distribution: CI re-runs this script and
# requires the committed JSON back byte for byte, so there is no sampling
# anywhere. Each row is `(arrival tick, prompt tokens, output tokens)`.
#
# The shape is the long tail the tutorial opens with: a couple of requests that
# run far longer than the rest. That is not decoration. With a flat length
# distribution static batching is merely slightly wasteful; with a tail it is
# catastrophic, and it is the tail the comparison is about.
SCENARIOS = {
    # Requests trickle in over the first few ticks, and the KV pool comfortably
    # covers four in-flight requests at these lengths. Nothing is ever preempted,
    # so the reader sees the scheduling policy on its own: the long tail, the
    # seats static holds for nothing, and the refill that continuous does every
    # tick. This is the scenario the lab opens on.
    "base": {
        "note": "请求分散到达，KV 池够用，不会被抢占 —— 看的是调度策略本身",
        "blocks": 30,
        "requests": [
            (0, 8, 1), (0, 8, 20), (0, 8, 3), (0, 8, 2),
            (1, 8, 4), (1, 6, 14), (2, 8, 2), (2, 8, 1),
            (4, 8, 6), (5, 8, 3), (6, 8, 2), (7, 6, 5),
        ],
    },
    # The same load arriving almost at once, with longer generations and a
    # smaller pool. This is the case the tutorial's §4 tip is about: a request
    # cannot get its blocks, so the scheduler preempts — and the trace is where
    # "显存耗尽时抢占" stops being a sentence in a paragraph and becomes a
    # victim, a release, a `num_computed_tokens` going back to zero, and a
    # re-prefill a few ticks later.
    #
    # The pool is small enough to force that eviction, and it is worth noticing
    # WHICH arm it forces it in. Continuous is the one that preempts, because
    # continuous is the one that keeps admitting: it refills every seat the
    # moment one frees, so it runs the pool hot by construction. Static never
    # preempts here — and not because it is better at memory, but because
    # batch-atomic admission throttles it so hard it never overcommits in the
    # first place. It pays for that safety somewhere else, and the trace shows
    # exactly where: a batch stays resident until its slowest member finishes,
    # so the short requests ride along doing nothing for the rest of the run.
    # Two ways to avoid running out of memory, and the bill lands on the seats.
    "burst": {
        "note": "请求几乎同时到达、生成更长，KV 池很小 —— continuous 会被迫抢占，"
                "static 靠「整批不放」避开抢占、代价是槽位",
        "blocks": 17,
        "requests": [
            (0, 8, 2), (0, 8, 26), (0, 8, 3), (0, 8, 2),
            (0, 8, 15), (0, 8, 2), (1, 8, 13), (1, 8, 2),
            (1, 8, 4), (2, 8, 6), (2, 8, 3), (2, 8, 5),
        ],
    },
}

SCHEDULES = ("static", "continuous")
SCHEDULE_LABEL = {"static": "Static Batching", "continuous": "Continuous Batching"}
SCENARIO_LABEL = {"base": "稳态到达", "burst": "突发到达"}

# The serving configuration. Shared by both arms, so the comparison is between
# scheduling policies and not between two differently-provisioned servers.
MAX_NUM_SEQS = 4          # seats: how many requests may be in flight at once
TOKEN_BUDGET = 32         # tokens one tick may compute, in total
BLOCK_SIZE = 4            # KV tokens per block (PagedAttention's granularity)
NUM_BLOCKS = 30           # the base scenario's KV pool; each scenario may differ
MAX_TICKS = 400           # a guard against a scheduling bug looping forever


class Req:
    """One request. `computed` is the tutorial's `num_computed_tokens`.

    The token accounting is vLLM's, and it is worth stating because the whole
    trace rests on it: a prefill computes the prompt AND samples the first
    token, so a request with prompt `p` and output `o` computes `p + o − 1`
    tokens in total and is finished once it has produced `o` tokens.
    """

    __slots__ = ("rid", "arrival", "prompt", "output", "computed", "generated",
                 "blocks", "preemptions", "recompute_tokens", "admitted_at",
                 "done_at")

    def __init__(self, rid, arrival, prompt, output):
        self.rid = rid
        self.arrival = arrival
        self.prompt = prompt
        self.output = output
        self.computed = 0          # tokens computed so far (prefix + generated)
        self.generated = 0         # output tokens produced
        self.blocks = 0            # KV blocks currently held
        self.preemptions = 0
        self.recompute_tokens = 0  # work thrown away by preemption
        self.admitted_at = None
        self.done_at = None        # the tick at whose END it left the system

    @property
    def finished(self):
        return self.generated >= self.output

    @property
    def declared_work(self):
        """Tokens this request must compute if it is never preempted."""
        return self.prompt + self.output - 1


def blocks_for(computed):
    """Blocks to hold `computed` tokens. A request that has been admitted owns
    at least one block even before it has computed anything, which is what makes
    the pool a constraint at admission time rather than only later."""
    return max(1, math.ceil(computed / BLOCK_SIZE))


def simulate(requests, schedule, n_blocks=NUM_BLOCKS):
    """Run the scheduling loop; return one frame record per tick.

    The frame record is the SINGLE encoding of the run: it says, per tick, which
    kind each request was in and how many tokens were spent. The gantt's bars,
    the state tensors and every published metric are all derived from it, so the
    picture cannot drift away from the arithmetic.
    """
    reqs = list(requests)
    waiting = []       # FIFO; preempted requests are re-queued at the FRONT (§6.3)
    running = []       # continuous: in flight. static: the current batch.
    pending = list(reqs)   # not yet arrived, in arrival order
    frames = []
    batch_groups = []  # static: each batch's membership, as formed

    def kv_used():
        return sum(r.blocks for r in reqs)

    def drop(victim, evicted, pool):
        """Preempt `victim`: release its KV, zero its progress, re-queue it.

        Recompute recovery, per §6.3 — `num_computed_tokens` goes back to zero
        and the request waits at the head of the queue, so the work it already
        did is thrown away and counted as such.
        """
        victim.recompute_tokens += victim.computed
        victim.computed = 0
        victim.generated = 0
        victim.blocks = 0
        victim.admitted_at = None
        victim.done_at = None
        victim.preemptions += 1
        pool.remove(victim)
        waiting.insert(0, victim)
        evicted.append(victim.rid)

    def make_room(r, new_computed, evicted):
        """Grow r's KV to `new_computed` tokens, preempting peers if it does not
        fit. Returns True when r ends up holding the blocks it asked for.

        The victim is the newest in-flight peer, which is what `self.running.pop()`
        does in the tutorial's FCFS pseudocode.
        """
        need = blocks_for(new_computed)
        pool = running
        while r.blocks + (n_blocks - kv_used()) < need:
            # Only a LIVE request can be a victim. A finished request's KV is
            # already gone (see phase 1), and re-running a completed generation
            # is not something any scheduler does — the seat it still occupies
            # is the waste this lab is about, not something to reclaim by
            # throwing the result away.
            victim = next((v for v in reversed(pool)
                           if v is not r and v.blocks > 0 and not v.finished), None)
            if victim is None:
                return False
            drop(victim, evicted, pool)
        r.blocks = need
        return True

    for t in range(MAX_TICKS):
        arrived = [r for r in pending if r.arrival <= t]
        for r in arrived:
            pending.remove(r)
            waiting.append(r)

        act = {}           # rid -> kind, for this tick
        evicted = []
        blocked = []       # in the batch, live, but given no KV this tick
        tokens = 0
        had_batch = bool(running)

        # ---- static: choose the candidates for the next batch when there is
        # none. "攒够一批" means either the seats are full or nothing more is
        # coming. This only PICKS them; admission below still pays the token
        # budget, exactly as the continuous arm does.
        candidates = None
        if schedule == "static" and not running and waiting and (
                len(waiting) >= MAX_NUM_SEQS or not pending):
            candidates = list(waiting[:MAX_NUM_SEQS])

        # ---- phase 1: the in-flight requests, always first (§6.2). A static
        # batch's finished members are still scheduled — the seat is computed for
        # nothing. That waste IS the lesson, so it is charged to the budget like
        # any other work rather than being quietly skipped.
        for r in list(running):
            if tokens >= TOKEN_BUDGET or r.rid in evicted:
                break
            if r.finished:
                # Done, and still holding a seat for the rest of the batch. It
                # consumes no token and no KV: there is nothing left to compute
                # and nothing left to store, which is exactly why holding the
                # seat is pure waste. `stalled` is that seat, drawn for as many
                # ticks as static keeps it — §2's "长尾请求拖死整批", counted in
                # the unit the claim is actually about, which is slots.
                r.blocks = 0
                act[r.rid] = K_STALLED
                continue
            if not make_room(r, r.computed + 1, evicted):
                # Live, in the batch, and given nothing this tick: the pool is
                # so full that not even one more block fits. Nothing is computed
                # and no token is spent, so this is a blocked seat rather than a
                # stalled one, and it is tracked separately because it is the
                # one case the pool's own accounting cannot express.
                act[r.rid] = K_STALLED
                blocked.append(r.rid)
                continue
            r.computed += 1
            tokens += 1
            r.generated += 1
            act[r.rid] = K_DECODE

        # ---- phase 2: take requests off the waiting queue, with the budget
        # that is left and only when phase 1 did not have to preempt (§6.2).
        seats_free = MAX_NUM_SEQS - len(running)
        if not evicted and tokens < TOKEN_BUDGET:
            queue = candidates if candidates is not None else list(waiting)
            for r in queue:
                if seats_free <= 0 or tokens + r.prompt > TOKEN_BUDGET:
                    break
                if not make_room(r, r.prompt, evicted):
                    break
                waiting.remove(r)
                r.computed = r.prompt
                r.generated = 1     # the prefill samples the first token
                r.admitted_at = t
                tokens += r.prompt
                seats_free -= 1
                running.append(r)
                act[r.rid] = K_PREFILL

        # The batch this tick formed, recorded as it actually ended up — a
        # preemption during admission can leave it smaller than the candidates.
        if schedule == "static" and not had_batch and running:
            batch_groups.append([r.rid for r in running])

        # ---- the end of the iteration: who leaves
        departed = []
        if schedule == "continuous":
            for r in list(running):
                if r.finished:
                    r.done_at = t + 1
                    r.blocks = 0
                    running.remove(r)
                    departed.append(r.rid)
        elif running and all(r.finished for r in running):
            # Batch-atomic release: only now does any seat come back — and the
            # members that finished early have been carrying a seat ever since.
            for r in running:
                r.done_at = t + 1
                r.blocks = 0
            departed = sorted(r.rid for r in running)
            running = []

        for r in reqs:
            if r.rid in evicted:
                act[r.rid] = K_PREEMPTED
            elif r.rid not in act and r in waiting:
                act[r.rid] = K_WAITING

        by_id = {r.rid: r for r in reqs}
        frames.append({
            "t": t,
            "act": act,
            "tokens": tokens,
            "admitted": sorted(rid for rid, k in act.items() if k == K_PREFILL),
            "decoded": sorted(rid for rid, k in act.items() if k == K_DECODE),
            "stalled": sorted(rid for rid, k in act.items() if k == K_STALLED),
            # A stalled seat that was already finished consumed a token that went
            # nowhere; one that was merely blocked consumed nothing. Both are the
            # same colour on the chart — a seat producing nothing — but only the
            # first is wasted WORK, and the conservation identity below counts it.
            "wasted": sorted(rid for rid, k in act.items()
                             if k == K_STALLED and by_id[rid].finished),
            "blocked": sorted(set(blocked)),
            "preempted": sorted(set(evicted)),
            "done": sorted(departed),
            "waiting": [r.rid for r in waiting],
            "running": sorted(r.rid for r in running),
        })

        if not (waiting or running or pending):
            break

    if waiting or running or pending:
        raise AssertionError(f"scheduler did not drain within {MAX_TICKS} ticks")
    return frames, reqs, batch_groups


# ------------------------------------------------------------------ reference
def occupancy(requests, frames, kinds):
    """The raw per-tick occupancy as a (n_requests, n_ticks) code matrix.

    Deliberately the dumbest possible encoding of the run: one small integer per
    (request, tick), built from the frame records and holding no aggregate at
    all. `reference_metrics` recomputes every published number from this matrix,
    so the two paths to those numbers are genuinely different code rather than
    the same expression written twice.
    """
    code = {k: i + 1 for i, k in enumerate(kinds)}
    M = np.zeros((len(requests), len(frames)), dtype=np.int64)
    for t, f in enumerate(frames):
        for rid, kind in f["act"].items():
            M[rid - 1, t] = code[kind]
    return M


def reference_metrics(M, kinds, requests, frames, max_num_seqs):
    """Every number the page prints, recomputed from the occupancy matrix.

    Read this as the reference implementation: it knows nothing about the
    scheduler, only about the picture the reader will see. If the two disagree,
    the picture is lying about the run.
    """
    T = torch.tensor(M, dtype=torch.float64)
    n_req, n_ticks = M.shape
    hit = {k: (T == kinds.index(k) + 1).double() for k in kinds}

    useful = hit[K_PREFILL] + hit[K_DECODE]
    in_flight = useful + hit[K_STALLED]

    slots_total = float(max_num_seqs * n_ticks)
    out = {
        "n_ticks": n_ticks,
        "makespan": n_ticks,
        "n_requests": n_req,
        "slots_total": slots_total,
        "useful_slot_ticks": float(useful.sum()),
        "stalled_slot_ticks": float(hit[K_STALLED].sum()),
        "waiting_ticks": float(hit[K_WAITING].sum()),
        "preempt_ticks": float(hit[K_PREEMPTED].sum()),
        "output_tokens": int(sum(r.output for r in requests)),
    }
    out["idle_slot_ticks"] = (slots_total - out["useful_slot_ticks"]
                              - out["stalled_slot_ticks"])
    out["utilization"] = out["useful_slot_ticks"] / slots_total if slots_total else 0.0

    # Latency is arrival-to-completion. `done_at` is the tick at whose end the
    # request left, so the reference derives it from the last column the request
    # occupied rather than reading the field the scheduler wrote.
    arrival = torch.tensor([r.arrival for r in requests], dtype=torch.float64)
    cols = torch.arange(1, n_ticks + 1, dtype=torch.float64)
    last = (in_flight * cols).max(dim=1).values
    lat = last - arrival
    out["mean_latency"] = float(lat.mean())
    out["max_latency"] = float(lat.max())
    out["latencies"] = [int(x) for x in lat.tolist()]
    out["throughput"] = out["output_tokens"] / n_ticks if n_ticks else 0.0
    return out


# ---------------------------------------------------------------- trace build
def b(slot, sym, idx, num):
    return slot, {"sym": sym, "idx": idx, "num": num}


def row_specs(requests):
    return [{
        "id": f"R{r.rid}",
        "label": f"R{r.rid}",
        "sub": f"到达 t={r.arrival} · prompt {r.prompt} · 生成 {r.output}",
    } for r in requests]


def tensor_specs(n_req, n_blocks):
    zero = [0] * n_req
    return {
        "waiting_q": {
            "shape": [n_req], "dtype": "int", "at": "调度器", "role": "queue",
            "init": zero,
            "note": "waiting 队列（优先级队列，这里用 FCFS）里有哪些请求，1 = 在排队"},
        "running_q": {
            "shape": [n_req], "dtype": "int", "at": "调度器", "role": "queue",
            "init": zero, "note": "running 列表：本 tick 结束时仍在批里的请求"},
        "done_q": {
            "shape": [n_req], "dtype": "int", "at": "调度器", "role": "state",
            "init": zero, "note": "已经生成完 output 个 token、KV 已归还的请求"},
        "computed": {
            "shape": [n_req], "dtype": "int", "at": "KV 显存", "role": "state",
            "init": zero,
            "note": "每个请求的 num_computed_tokens —— 被抢占时归零（重计算）"},
        "kv_blocks": {
            "shape": [n_req], "dtype": "int", "at": "KV 显存", "role": "state",
            "init": zero,
            "note": f"每个请求占用的 KV block 数（block_size = {BLOCK_SIZE}）"},
        "batch_tokens": {
            "shape": [], "dtype": "int", "at": "调度器", "role": "counter",
            "init": 0, "note": "本 tick 从 token budget 里花掉的 token 数"},
        "kv_free": {
            "shape": [], "dtype": "int", "at": "KV 显存", "role": "counter",
            "init": n_blocks, "note": f"还剩多少空闲 KV block（共 {n_blocks}）"},
        "stalled": {
            "shape": [], "dtype": "int", "at": "调度器", "role": "counter",
            "init": 0,
            "note": "本 tick 里已经生成完、却还占着槽位的请求数（它们不再算 token，"
                    "占的只是并发额度）"},
    }


def snapshot(requests, f, n_blocks):
    """The tensor state at the end of frame `f`, in full."""
    running = set(f["running"])
    waiting = set(f["waiting"])
    return {
        "waiting_q": [1 if r.rid in waiting else 0 for r in requests],
        "running_q": [1 if r.rid in running else 0 for r in requests],
        "done_q": [1 if r.done_at is not None and r.done_at <= f["t"] + 1 else 0
                   for r in requests],
        "computed": [r.computed for r in requests],
        "kv_blocks": [r.blocks for r in requests],
        "batch_tokens": f["tokens"],
        "kv_free": n_blocks - sum(r.blocks for r in requests),
        "stalled": len(f["stalled"]),
    }


def build_trace(scenario, schedule, requests, frames, batch_groups, metrics, n_blocks):
    n_req = len(requests)
    tensors = tensor_specs(n_req, n_blocks)
    snapshots = [snapshot(requests, f, n_blocks) for f in frames]
    inits = {name: spec["init"] for name, spec in tensors.items()}

    steps, prev = [], dict(inits)
    phases = []

    for f, snap in zip(frames, snapshots):
        changed = {k: v for k, v in snap.items() if prev[k] != v}
        prev = snap

        bars = [{"row": f"R{rid}", "kind": kind}
                for rid, kind in sorted(f["act"].items())]

        # The phase strip is derived from the run, not imposed on it: a tick is
        # "排空" once nothing is queued and nothing more is coming, and "预热"
        # until the first request has finished.
        phase = "预热"
        if metrics["_done_before"][f["t"]]:
            phase = "稳态"
        if not f["waiting"] and not any(r.arrival > f["t"] for r in requests):
            phase = "排空"
        if not phases or phases[-1] != phase:
            phases.append(phase)

        regions, formula = {}, dict(BASE_FORMULA)
        if f["done"]:
            regions["EXIT"] = (
                "本步有 " + "、".join(f"R{r}" for r in f["done"])
                + " 生成了最后一个 token，KV block 立刻归还空闲池。"
                + ("Static 下槽位还不还 —— 整批要一起结束。" if schedule == "static"
                   else "腾出的槽位下一步就能被新请求补上，这就是「随退随补」。"))
            for tier in formula:
                formula[tier] += r" \;\region{EXIT}{\text{退出}}"
        if f["preempted"]:
            regions["PREEMPT"] = (
                "KV 不够，allocate_slots 拿不到块：踢掉 running 里最后来的 "
                + "、".join(f"R{r}" for r in f["preempted"])
                + "，它的 KV 全部释放、num_computed_tokens 归零（重计算恢复），"
                  "打回 waiting 队头，等有资源了优先恢复。")
            for tier in formula:
                formula[tier] += r" \;\region{PREEMPT}{\text{抢占}}"
        if f["stalled"]:
            regions["STALL"] = (
                "本步 " + "、".join(f"R{r}" for r in f["stalled"])
                + " 已经生成完了，token 一个没算，但槽位还占着 —— 整批没结束就不释放。"
                  "这是「长尾请求拖死整批」的成本，只不过它花的不是算力而是并发额度："
                  "空位腾不出来，排队的新请求上不了车。")
            for tier in formula:
                formula[tier] += r" \;\region{STALL}{\text{空转}}"

        n_pre = sum(r.prompt for r in requests if r.rid in f["admitted"])
        step = {
            "id": f"t{f['t']}",
            "title": tick_title(f),
            "kind": "op",
            "op": "schedule",
            "phase": phase,
            "formula": formula,
            "bindings": dict([
                b("T", "t", str(f["t"]), str(f["t"])),
                b("DEC", r"|D_t|", f"|D_{{{f['t']}}}|",
                  str(len(f["decoded"]) + len(f["stalled"]))),
                b("PRE", r"\textstyle\sum_{r \in A_t} p_r",
                  f"\\textstyle\\sum_{{r \\in A_{{{f['t']}}}}} p_r", str(n_pre)),
                b("TOT", r"\mathrm{tokens}_t", f"\\mathrm{{tokens}}_{{{f['t']}}}",
                  str(f["tokens"])),
                b("BUDGET", r"\mathrm{budget}", r"\mathrm{budget}", str(TOKEN_BUDGET)),
            ]),
            "reads": ["waiting_q", "running_q", "computed", "kv_blocks"],
            "writes": sorted(changed.keys()),
            "state": changed,
            "bars": bars,
            "narration": tick_narration(f, requests, schedule),
        }
        if regions:
            step["regions"] = regions
        steps.append(step)

    edges = [{"from": f"t{frames[i]['t']}", "to": f"t{frames[i + 1]['t']}", "tensor": ""}
             for i in range(len(frames) - 1)]

    return {
        "meta": {
            "lab": "L10",
            "title": f"{SCHEDULE_LABEL[schedule]} 调度回放（{SCENARIO_LABEL[scenario]}）",
            "source": SOURCE,
            "config": {
                "schedule": schedule,
                "scenario": scenario,
                "n_requests": n_req,
                "max_num_seqs": MAX_NUM_SEQS,
                "token_budget": TOKEN_BUDGET,
                "block_size": BLOCK_SIZE,
                "num_blocks": n_blocks,
            },
            "reference": reference_note(metrics, schedule),
            "metrics": public_metrics(metrics, frames, requests, batch_groups,
                                      schedule),
        },
        "tensors": tensors,
        "graph": {"nodes": [{"id": f"t{f['t']}", "title": tick_title(f), "kind": "op",
                             "op": "schedule", "phase": steps[i]["phase"]}
                            for i, f in enumerate(frames)],
                  "edges": edges},
        "gantt": {
            "rows": row_specs(requests),
            "kinds": list(KINDS),
            "kindsLabel": {K_PREFILL: "prefill（新请求上车）",
                           K_DECODE: "decode（生成 1 个 token）",
                           K_WAITING: "waiting（排队等待）",
                           K_STALLED: "stalled（已完成 · 空转）",
                           K_PREEMPTED: "preempted（被抢占）"},
        },
        "steps": steps,
    }


BASE_FORMULA = {
    "sym": (r"\mathrm{tokens}_t = \underbrace{\slot{DEC}}_{\text{在途 · 每个 1 个}}"
            r" + \underbrace{\slot{PRE}}_{\text{新请求 · 整段 prompt}}"
            r" = \slot{TOT} \;\le\; \slot{BUDGET}"),
    "idx": (r"\mathrm{tokens}_{\slot{T}} = \slot{DEC} + \slot{PRE} = \slot{TOT}"
            r" \;\le\; \slot{BUDGET}"),
    "num": (r"\mathrm{tokens}_{\slot{T}} = \slot{DEC} + \slot{PRE} = \slot{TOT}"
            r" \;\le\; \slot{BUDGET}"),
}


def tick_title(f):
    bits = []
    if f["admitted"]:
        bits.append(f"{len(f['admitted'])} 个 prefill")
    if f["decoded"]:
        bits.append(f"{len(f['decoded'])} 个 decode")
    if f["stalled"]:
        bits.append(f"{len(f['stalled'])} 个空转")
    if f["preempted"]:
        bits.append(f"抢占 {len(f['preempted'])}")
    if f["done"]:
        bits.append(f"{len(f['done'])} 个完成")
    return f"tick {f['t']} · " + ("，".join(bits) if bits else "空闲")


def tick_narration(f, requests, schedule):
    by_id = {r.rid: r for r in requests}
    parts = [f"本 tick 花掉 {f['tokens']} / {TOKEN_BUDGET} 个 token 预算。"]
    if f["stalled"]:
        parts.append("槽位里 " + "、".join(f"R{r}" for r in f["stalled"])
                     + " 已经生成完了，却因为整批没结束还占着位置 —— "
                       "它们这一步一个 token 都没算，占的是槽位而不是算力；"
                       "槽位空不出来，后面的请求就上不了车。")
    if f["admitted"]:
        parts.append("新上车：" + "、".join(
            f"R{r}（prompt {by_id[r].prompt}）" for r in f["admitted"]) + "。")
    if f["preempted"]:
        parts.append("KV 不够，" + "、".join(f"R{r}" for r in f["preempted"])
                     + " 被抢占并打回 waiting 队头，下次调度要从头重新 prefill。")
    if f["done"] and schedule == "continuous":
        parts.append("退出的请求立刻让出槽位。")
    if f["waiting"]:
        parts.append(f"还有 {len(f['waiting'])} 个在 waiting 队列里排队。")
    return "".join(parts)


def public_metrics(metrics, frames, requests, batch_groups, schedule):
    stall_formula = static_stall_closed_form(requests, batch_groups, schedule)
    out = {
        "makespan": metrics["makespan"],
        "output_tokens": metrics["output_tokens"],
        "mean_latency": metrics["mean_latency"],
        "max_latency": metrics["max_latency"],
        "utilization": metrics["utilization"],
        "useful_slot_ticks": int(metrics["useful_slot_ticks"]),
        "stalled_slot_ticks": int(metrics["stalled_slot_ticks"]),
        "idle_slot_ticks": int(metrics["idle_slot_ticks"]),
        "slots_total": int(metrics["slots_total"]),
        "waiting_ticks": int(metrics["waiting_ticks"]),
        "throughput": metrics["throughput"],
        "preemptions": sum(r.preemptions for r in requests),
        "recompute_tokens": sum(r.recompute_tokens for r in requests),
        "kind_ticks": {k: int(sum(1 for f in frames if k in f["act"].values()))
                       for k in KINDS},
        "static_stall_closed_form": stall_formula,
    }
    return out


def static_stall_closed_form(requests, batch_groups, schedule):
    """Static batching's wasted seat-ticks, from the definition rather than from
    the simulation's own counters.

    A batch runs until its slowest member finishes, so every member is resident
    for `max(output in batch)` ticks while only `own output` of them do anything.
    The waste is the difference, summed over the batches the run formed. Returns
    None for the continuous arm, which has no batches and therefore no such
    waste — and says so rather than reporting a zero that would look like a
    measurement.
    """
    if schedule == "continuous":
        return None
    by_id = {r.rid: r for r in requests}
    waste = 0
    for group in batch_groups:
        members = [by_id[rid] for rid in group]
        span = max(m.output for m in members)
        waste += sum(span - m.output for m in members)
    return waste


def reference_note(metrics, schedule):
    return (
        f"调度循环逐步模拟：{metrics['makespan']} 步收干，"
        f"{metrics['output_tokens']} 个输出 token，"
        f"槽位利用率 {100 * metrics['utilization']:.1f}%"
        f"（{int(metrics['useful_slot_ticks'])}/{int(metrics['slots_total'])} 个 slot·tick），"
        f"空转 {int(metrics['stalled_slot_ticks'])} 个 slot·tick"
        + ("（static 独有）" if schedule == "static" else "（continuous 恒为 0）")
        + "；每个数字都用 torch 从逐 tick 占用矩阵独立重算过，"
          "整数逐位相同（np.array_equal），比值 float64 相对容差 1e-12"
    )


# -------------------------------------------------------------------- self-lint
SLOT_RE = re.compile(r"\\slot\{([A-Za-z0-9_]+)\}")
REGION_RE = re.compile(r"\\region\{([A-Za-z0-9_]+)\}")
LATEX_IN_TEXT_RE = re.compile(r"\\[a-zA-Z]+\{")
BAD_LITERAL_RE = re.compile(r"^(NaN|Infinity|-Infinity|undefined|null)$")
ROW_RE = re.compile(r"^[A-Za-z0-9_-]+$")
BAR_KEYS = {"row", "kind"}
BAR_OPTIONAL = {"label"}


def value_ok(v):
    """A trace value is a finite number, null, or one of the three sentinels."""
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
    """The gantt view's slice of the contract — the author-side half.

    The other half is the JS port in `labs/assets/engine/views/gantt.js`
    (`LabEngine.gantt.lint`), and it holds the same rules. A rule that exists on
    one side only is a rule tested on neither side, which is the failure mode
    this pair exists to prevent; when a rule changes it changes in both, and
    both sabotage tables grow a case for it.

    Only the rules that are THIS view's own are here (rows, tracks, kinds, bars).
    The general trace contract — declared tensors, graph/step agreement,
    three-tier bindings — is the engine's own lint and every generator's job
    alike, and duplicating it would just be a second copy to keep in sync.
    """
    gaps, warns, infos = [], [], []
    declared = set(trace["tensors"])
    gantt = trace.get("gantt")

    if not isinstance(gantt, dict):
        gaps.append("gantt 缺失 —— 甘特视图没有行与 kind 的定义，一帧都画不出来")
        return gaps, warns, infos

    rows = gantt.get("rows")
    if not isinstance(rows, list):
        gaps.append("gantt.rows 不是数组")
        rows = []
    elif not rows:
        gaps.append("gantt.rows 是空的 —— 甘特图一行都画不出来")

    row_ids, row_tracks = [], {}
    tracks = gantt.get("tracks")
    track_ids = []
    if tracks is not None:
        if not isinstance(tracks, list) or not tracks:
            gaps.append("gantt.tracks 存在但不是非空数组 —— 给不出任何轨名")
            tracks = []
        for i, tk in enumerate(tracks or []):
            if not isinstance(tk, dict) or not tk.get("id"):
                gaps.append(f"gantt.tracks[{i}] 没有 id")
                continue
            if tk["id"] in track_ids:
                gaps.append(f'gantt.tracks 里 id "{tk["id"]}" 重复')
            track_ids.append(tk["id"])

    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("id"):
            gaps.append(f"gantt.rows[{i}] 没有 id")
            continue
        if not ROW_RE.match(row["id"]):
            gaps.append(f'gantt.rows[{i}].id = {row["id"]!r} 含非法字符'
                        "（只允许字母数字下划线连字符）")
        if row["id"] in row_ids:
            gaps.append(f'gantt.rows 里 id "{row["id"]}" 重复 —— 两行同名，'
                        "条目分不清该画在哪一行")
        row_ids.append(row["id"])
        if not row.get("label"):
            warns.append(f'gantt.rows[{i}] ("{row["id"]}") 没有 label，只能显示 id')
        if "track" in row:
            if tracks is None:
                gaps.append(f'gantt.rows[{i}] 指定了 track = {row["track"]!r}，'
                            "但 gantt.tracks 根本没声明 —— 这一行没有轨可归")
            elif row["track"] not in track_ids:
                gaps.append(f'gantt.rows[{i}] 的 track = {row["track"]!r} '
                            "不在 gantt.tracks 里")
            else:
                row_tracks.setdefault(row["track"], []).append(row["id"])

    kinds = gantt.get("kinds")
    if not isinstance(kinds, list) or not kinds:
        gaps.append("gantt.kinds 不是非空数组 —— 条目的 kind 没有词表可查")
        kinds = []
    for k in kinds:
        if not isinstance(k, str) or not k:
            gaps.append(f"gantt.kinds 里有非字符串项 {k!r}")
    if len(set(kinds)) != len(kinds):
        gaps.append("gantt.kinds 里有重复项")

    used_rows, used_kinds, bar_counts = set(), set(), {}
    for s in trace["steps"]:
        sid = s["id"]
        bars = s.get("bars")
        if bars is None:
            gaps.append(f'步骤 "{sid}" 没有 bars —— 甘特图在这一帧没有内容可画')
            continue
        if not isinstance(bars, list):
            gaps.append(f'步骤 "{sid}" 的 bars 不是数组')
            continue
        per_row = []
        for j, bar in enumerate(bars):
            if not isinstance(bar, dict):
                gaps.append(f'步骤 "{sid}" 的 bars[{j}] 不是对象')
                continue
            # Unknown keys are rejected rather than ignored: a bar that carries a
            # field this component does not read is a field its author believed
            # was doing something.
            extra = set(bar) - BAR_KEYS - BAR_OPTIONAL
            if extra:
                gaps.append(f'步骤 "{sid}" 的 bars[{j}] 多了字段 '
                            f'{{{", ".join(sorted(extra))}}} —— 契约只认 '
                            f'{{row, kind, label?}}')
            for req in BAR_KEYS:
                if req not in bar:
                    gaps.append(f'步骤 "{sid}" 的 bars[{j}] 缺 "{req}"')
            rid, kind = bar.get("row"), bar.get("kind")
            if rid not in row_ids:
                gaps.append(f'步骤 "{sid}" 的 bars[{j}].row = {rid!r} 不在 gantt.rows 里'
                            " —— 这一条画不到任何一行上")
            else:
                used_rows.add(rid)
                per_row.append(rid)
                bar_counts[rid] = bar_counts.get(rid, 0) + 1
            if kind not in kinds:
                gaps.append(f'步骤 "{sid}" 的 bars[{j}].kind = {kind!r} 不在 gantt.kinds '
                            "词表里 —— 条目会没有任何样式")
            else:
                used_kinds.add(kind)
            if "label" in bar and not isinstance(bar["label"], str):
                gaps.append(f'步骤 "{sid}" 的 bars[{j}].label = {bar["label"]!r} 不是字符串')
        if len(set(per_row)) != len(per_row):
            dupes = sorted({r for r in per_row if per_row.count(r) > 1})
            gaps.append(f'步骤 "{sid}" 里 ' + "、".join(dupes)
                        + " 被声明了不止一个 kind —— 同一 tick 一个资源只能有一种活动")

        for key, val in (s.get("state") or {}).items():
            if key not in declared:
                gaps.append(f'步骤 "{sid}" 的 state 写了未声明张量 "{key}"')
            for item in (val if isinstance(val, list) else [val]):
                if isinstance(item, str) and item not in (NEG_INF, POS_INF, NAN):
                    gaps.append(f'步骤 "{sid}" 的 state["{key}"] 含未知哨兵字符串 "{item}"')

    for rid in row_ids:
        if rid not in used_rows:
            warns.append(f'gantt.rows 里的 "{rid}" 从头到尾没有任何条目 —— '
                         "这一行永远是空的，是不是忘了给它写 bars")
    for tid, members in row_tracks.items():
        if len(members) < 2:
            warns.append(f'轨 "{tid}" 只含一行 —— 单行成轨，和不用轨看不出区别')
    if row_ids and not used_rows:
        gaps.append("没有任何 bars 指向 gantt.rows 里的行 —— 甘特图会是空的")

    infos.append(f'{len(trace["steps"])} 步 / {len(row_ids)} 行'
                 + (f' / {len(track_ids)} 轨' if track_ids else '')
                 + f' / {len(used_kinds)} 种活动（{", ".join(sorted(used_kinds))}）')
    infos.append("活动分布：" + "、".join(f"{k}={v}" for k, v in sorted(bar_counts.items())))
    return gaps, warns, infos


def _first_bar_step(trace):
    return next(i for i, s in enumerate(trace["steps"]) if s.get("bars"))


SABOTAGE_CASES = {
    "gantt 整块缺失": lambda t: t.pop("gantt"),
    "gantt.rows 为空": lambda t: t["gantt"].update({"rows": []}),
    "gantt.rows 不是数组": lambda t: t["gantt"].update({"rows": "R1,R2"}),
    "行 id 重复": lambda t: t["gantt"]["rows"].append(dict(t["gantt"]["rows"][0])),
    "行 id 含非法字符": lambda t: t["gantt"]["rows"][0].update({"id": "R 1"}),
    "行没有 id": lambda t: t["gantt"]["rows"][0].pop("id"),
    "kinds 词表为空": lambda t: t["gantt"].update({"kinds": []}),
    "kinds 里有重复项": lambda t: t["gantt"]["kinds"].append(t["gantt"]["kinds"][0]),
    "条目指向不存在的行": lambda t: t["steps"][_first_bar_step(t)]["bars"].append(
        {"row": "GHOST", "kind": K_DECODE}),
    "条目用了词表外的 kind": lambda t: t["steps"][_first_bar_step(t)]["bars"].append(
        {"row": t["gantt"]["rows"][0]["id"], "kind": "sleeping"}),
    "步骤没有 bars": lambda t: t["steps"][_first_bar_step(t)].pop("bars"),
    "条目多了个字段": lambda t: t["steps"][_first_bar_step(t)]["bars"].append(
        {"row": t["gantt"]["rows"][0]["id"], "kind": K_DECODE, "color": "red"}),
    "条目缺 row": lambda t: t["steps"][_first_bar_step(t)]["bars"].append(
        {"kind": K_DECODE}),
    "label 不是字符串": lambda t: t["steps"][_first_bar_step(t)]["bars"].append(
        {"row": t["gantt"]["rows"][0]["id"], "kind": K_DECODE, "label": 7}),
    "同一 tick 同一资源两种活动": lambda t: t["steps"][_first_bar_step(t)]["bars"].append(
        {"row": t["steps"][_first_bar_step(t)]["bars"][0]["row"], "kind": K_WAITING}),
    "行指定了未声明的轨": lambda t: t["gantt"]["rows"][0].update({"track": "worker0"}),
    "行指定了不存在的轨": lambda t: (
        t["gantt"].update({"tracks": [{"id": "a"}, {"id": "b"}]}),
        t["gantt"]["rows"][0].update({"track": "ghost"})),
    "tracks 声明了但为空": lambda t: t["gantt"].update({"tracks": []}),
    "state 写入未声明的张量": lambda t: t["steps"][2]["state"].update({"ghost": 1}),
    "state 用未知哨兵串": lambda t: t["steps"][2]["state"].update({"batch_tokens": "-inf"}),
}


def sabotage_checks(trace):
    """Prove the lint is not a function that always returns zero.

    Every mutation breaks exactly one rule on a copy of the real trace. A rule no
    breakage can trip is a rule that is not being tested, so every one must be
    caught or the script refuses to write.
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


def sabotage_components():
    """The same control group, run against the component's own JS lint.

    The Python rules and the JS rules are two ports of one contract, and the
    failure this guards against is precisely that one of them rots. Running the
    same sabotage table through both is what makes "the two must agree" a
    checked statement instead of a comment — see the `--js-lint` mode, which the
    acceptance harness invokes.
    """
    return list(SABOTAGE_CASES)


# ------------------------------------------------------------- claims & checks
def check_claims(requests, frames, metrics, ref, M, kinds, batch_groups, schedule,
                 n_blocks):
    """The propositions that make the lab worth showing.

    These are not consistency checks; they are the statements the page makes to
    the reader. If one stops being true the page would be showing a difference
    that is not there, which is worse than showing nothing at all.
    """
    failures = []
    n_req = len(requests)

    # --- every request finishes, having generated exactly its declared length
    for r in requests:
        if r.done_at is None:
            failures.append(f"R{r.rid} 没有完成")
        elif r.generated != r.output:
            failures.append(
                f"R{r.rid} 生成了 {r.generated} 个 token，声明的 output 是 {r.output}")

    # --- token conservation. Every token the scheduler spent is either current
    # progress or work thrown away by preemption, and progress in turn is either
    # declared work or a stall tick. This is what ties the bars to the ledger.
    spent = sum(f["tokens"] for f in frames)
    progress = sum(r.computed for r in requests)
    recompute = sum(r.recompute_tokens for r in requests)
    declared = sum(r.declared_work for r in requests)
    wasted = sum(len(f["wasted"]) for f in frames)
    if spent != progress + recompute:
        failures.append(
            f"逐 tick 花掉的 {spent} token != 进度 {progress} + 重算 {recompute}")
    # A stalled seat burns no token, so the token ledger has no waste term: every
    # token spent is either progress or work thrown away by preemption. The waste
    # lives in the SEAT account instead, which is the account §2 is about.
    if progress != declared:
        failures.append(
            f"进度 {progress} != 声明工作量 {declared} —— "
            "有请求算了不该算的 token（空转的槽位应该一个 token 都不花）")
    if any(f["blocked"] for f in frames):
        # A blocked seat (in the batch, live, but given nothing this tick) would
        # make the identity above undercount: it is a wasted seat that wasted no
        # token. At the pool sizes this lab ships that never happens, and the
        # assertion is here so that a future scenario which does hit it fails
        # loudly instead of quietly producing a trace whose headline identity is
        # off by one per blocked tick.
        failures.append(
            "有请求在批里却一步都没推进（blocked）—— 当前 KV 池下不该发生，"
            "要么调大 num_blocks，要么把 blocked 计入守恒式")

    # --- the bars and the ledger are two encodings of one run. Exact equality
    # on integers: a tolerance here would hide a dropped token.
    bars_useful = np.zeros(n_req, dtype=np.int64)
    bars_stalled = np.zeros(n_req, dtype=np.int64)
    for f in frames:
        for rid, kind in f["act"].items():
            if kind in USEFUL:
                bars_useful[rid - 1] += 1
            elif kind == K_STALLED:
                bars_stalled[rid - 1] += 1
    if not np.array_equal(bars_useful.sum(), int(ref["useful_slot_ticks"])):
        failures.append("bars 里 prefill+decode 的格数与参考矩阵不一致")
    if not np.array_equal(bars_stalled.sum(), int(ref["stalled_slot_ticks"])):
        failures.append("bars 里 stalled 的格数与参考矩阵不一致")
    ref_M = occupancy(requests, frames, kinds)
    if not np.array_equal(M, ref_M):
        failures.append("占用矩阵不可复现 —— 派生出来的 picture 与原始记录不同")

    # --- static's waste has a closed form, and the run reproduces it.
    #
    # The identity is derived from batch composition and output lengths, not from
    # counting the simulation's own stall ticks, so the two agreeing is a real
    # check rather than a tautology. It presumes batch-atomic release with no
    # preemption — a victim restarts from its prompt, and the batch it re-enters
    # is not the one `span − own_output` describes. That presumption is asserted
    # rather than assumed, so a scenario that breaks it says so instead of
    # producing a trace whose headline number is quietly wrong.
    if schedule == "static":
        preemptions = sum(r.preemptions for r in requests)
        formula = static_stall_closed_form(requests, batch_groups, schedule)
        if preemptions == 0:
            if formula != wasted:
                failures.append(
                    f"static 无抢占时，空转的闭式 {formula} 与模拟出来的 {wasted} 不等")
        else:
            # A victim re-enters a batch that is not the one it left, so
            # `span − own_output` describes a batch the run never held and the
            # number is not a bound in either direction. The conservation
            # identity above already covers this arm (it holds unconditionally);
            # asserting anything about the closed form here would be asserting
            # something that is simply not true.
            print(f"  note  static 臂有抢占 {preemptions} 次，"
                  f"闭式 {formula} 不再描述实际发生的批次（实际空转 {wasted}）")
        if wasted == 0:
            failures.append("static 臂竟然没有空转 —— 那这个对照就没有意义了")
    else:
        if metrics["stalled_slot_ticks"] != 0:
            failures.append("continuous 臂出现了 stalled —— 完成的请求本该立刻走")
        if metrics["waiting_ticks"] == 0:
            failures.append("continuous 臂没有任何排队 —— 场景太松，看不出补位")

    # --- the queue is a queue, and the pool is a pool
    if any(r.blocks < 0 for r in requests):
        failures.append("有请求的 KV block 数为负")
    peak = max(sum(1 for k in f["act"].values() if k in USEFUL or k == K_STALLED)
               for f in frames)
    if peak > MAX_NUM_SEQS:
        failures.append(f"某一 tick 批里有 {peak} 个请求，超过 max_num_seqs = {MAX_NUM_SEQS}")
    for f in frames:
        if f["tokens"] > TOKEN_BUDGET:
            failures.append(f"tick {f['t']} 花了 {f['tokens']} token，超过预算")
    # The pool is checked at end-of-tick, which is when every allocation has been
    # made and every release applied.
    held_series = []
    for f in frames:
        held_series.append(sum(r.blocks for r in requests))
    if max(held_series) > n_blocks:
        failures.append(
            f"某一 tick 手里握着 {max(held_series)} 个 KV block，超过池子 {n_blocks}")
    if min(held_series) < 0:
        failures.append("KV block 数为负")
    # A preempted request must actually lose its blocks, and a finished one must
    # give them back — otherwise "释放其占用的 KV Block" is narration, not code.
    # (A finished request keeps its SEAT but not its blocks; those are two
    # different resources and the chart shows only the first.)
    for r in requests:
        if r.blocks and (r.done_at is not None or r.finished):
            failures.append(f"R{r.rid} 已经生成完却还占着 {r.blocks} 个 KV block")
    return failures


def check_arm_contrast(static_trace, cont_trace):
    """Continuous must actually beat static, on the axes the page prints."""
    failures = []
    sm, cm = static_trace["meta"]["metrics"], cont_trace["meta"]["metrics"]
    for key, better in (("makespan", "lower"), ("mean_latency", "lower"),
                        ("utilization", "higher")):
        s, c = sm[key], cm[key]
        ok = s > c if better == "lower" else s < c
        if not ok:
            failures.append(
                f"连续批处理在 {key} 上没有优于静态批处理（static={s}, continuous={c}）")
    if cm["makespan"] * 1.2 > sm["makespan"]:
        failures.append(f"makespan 的差距太小（{sm['makespan']} vs {cm['makespan']}）"
                        " —— 对照图上看不出长尾被拖累")
    if sm["stalled_slot_ticks"] == 0:
        failures.append("static 臂没有空转的槽位，对照失去教学点")
    if cm["stalled_slot_ticks"] != 0:
        failures.append("continuous 臂不该有空转")
    return failures


def check_two_paths(metrics, ref):
    """The scheduler's counters against the torch recomputation, exactly."""
    failures = []
    for key in ("n_ticks", "makespan", "output_tokens", "max_latency",
                "useful_slot_ticks", "stalled_slot_ticks", "idle_slot_ticks",
                "slots_total", "waiting_ticks"):
        if int(metrics[key]) != int(ref[key]):
            failures.append(f"{key}: 台账 {metrics[key]} != torch 参考 {ref[key]}")
    for key in ("mean_latency", "utilization", "throughput"):
        a, b = metrics[key], ref[key]
        if not np.isclose(a, b, rtol=1e-12, atol=0.0):
            failures.append(f"{key}: 台账 {a!r} != torch 参考 {b!r}")
    return failures


# ------------------------------------------------------------------------ main
SOURCE = ("docs/guides/模块四-推理优化/第2章-推理引擎核心技术/"
          "2.2-Continuous Batching.md")
DEFAULT = ("continuous", "base")


def cfg_id(schedule, scenario):
    return f"continuous-batching-{schedule}-{scenario}"


def build_one(scenario, schedule):
    table = SCENARIOS[scenario]["requests"]
    n_blocks = SCENARIOS[scenario]["blocks"]
    for (_, prompt, output) in table:
        if prompt > TOKEN_BUDGET:
            raise AssertionError(f"prompt {prompt} 超过 token budget {TOKEN_BUDGET}")
        if output < 1:
            raise AssertionError("output 至少是 1：prefill 本身会产出第一个 token")
    requests = [Req(i + 1, a, p, o) for i, (a, p, o) in enumerate(table)]
    frames, requests, batch_groups = simulate(requests, schedule, n_blocks=n_blocks)
    M = occupancy(requests, frames, KINDS)
    ref = reference_metrics(M, KINDS, requests, frames, MAX_NUM_SEQS)

    # "预热" until the first request has ever finished, then "稳态" — computed
    # from the reference's own completion columns, so the phase strip is derived
    # from the run rather than assigned by hand.
    done_col = np.zeros(len(frames), dtype=bool)
    for f in frames:
        if f["done"]:
            done_col[f["t"]] = True
    seen = np.cumsum(done_col) > 0
    done_before = np.concatenate([[False], seen[:-1]])

    # The scheduler's own counters, on the other path.
    lat = np.array([r.done_at - r.arrival for r in requests], dtype=np.int64)
    metrics = {
        "n_ticks": len(frames),
        "makespan": len(frames),
        "slots_total": float(MAX_NUM_SEQS * len(frames)),
        "output_tokens": sum(r.output for r in requests),
        "mean_latency": float(lat.mean()),
        "max_latency": float(lat.max()),
        "useful_slot_ticks": float(sum(1 for f in frames for k in f["act"].values()
                                       if k in USEFUL)),
        "stalled_slot_ticks": float(sum(len(f["stalled"]) for f in frames)),
        # Counted from the frame records' own activity codes, not from the queue
        # length: a request preempted this tick is back in the queue, but the
        # tick's activity on that row is `preempted`, not `waiting`, and the
        # reference reads the activity codes too. Counting the queue instead
        # would make the two disagree by exactly the number of preemptions.
        "waiting_ticks": float(sum(1 for f in frames for k in f["act"].values()
                                   if k == K_WAITING)),
        "_done_before": done_before,
    }
    metrics["idle_slot_ticks"] = (metrics["slots_total"] - metrics["useful_slot_ticks"]
                                  - metrics["stalled_slot_ticks"])
    metrics["utilization"] = metrics["useful_slot_ticks"] / metrics["slots_total"]
    metrics["throughput"] = metrics["output_tokens"] / metrics["makespan"]
    return requests, frames, batch_groups, metrics, ref, M, n_blocks


def main():
    only_json = "--js-lint" in sys.argv
    real_stdout = sys.stdout
    if only_json:
        # Every human-readable line this run produces goes to stderr, so stdout
        # carries the JSON payload and nothing else — the parity harness pipes it
        # straight into `node`. A stray progress line would make the payload
        # unparseable, which reads as a harness bug rather than as this mode's
        # own contract, so the redirection is done once here rather than by
        # threading `file=` through two dozen print calls.
        sys.stdout = sys.stderr
    fails, written, built = [], [], {}

    for scenario in SCENARIOS:
        for schedule in SCHEDULES:
            name = cfg_id(schedule, scenario)
            requests, frames, batch_groups, metrics, ref, M, n_blocks = build_one(scenario, schedule)
            trace = build_trace(scenario, schedule, requests, frames, batch_groups,
                               metrics, n_blocks)

            gaps, warns, infos = lint(trace)
            sabotages = sabotage_checks(trace)
            claims = check_claims(requests, frames, metrics, ref, M, KINDS,
                                  batch_groups, schedule, n_blocks)
            paths = check_two_paths(metrics, ref)

            m = trace["meta"]["metrics"]
            print(f"\n{name}: {len(trace['steps'])} 步, {len(requests)} 个请求, "
                  f"{len(json.dumps(trace, ensure_ascii=False))} bytes")
            print(f"  利用率 {100 * m['utilization']:.1f}%  makespan {m['makespan']}  "
                  f"平均延迟 {m['mean_latency']:.1f}  最大 {m['max_latency']}  "
                  f"吞吐 {m['throughput']:.2f} tok/tick  空转 {m['stalled_slot_ticks']} "
                  f"slot·tick  抢占 {m['preemptions']} 次 / 重算 {m['recompute_tokens']} token")
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
            for line in paths:
                print(f"  REF   {line}")
                fails.append(f"{name}: {line}")
            if sabotages:
                fails.append(f"{name}: lint 对照组失效")
                for line in sabotages:
                    print(f"  LINT-DEAD  {line}")
            else:
                print(f"  lint: {len(gaps)} gap / {len(warns)} warn；"
                      f"对照 {len(SABOTAGE_CASES)} 种破坏全部被抓到（Python 侧）")
            built[(schedule, scenario)] = trace

    # --- the scenario switch must actually switch something
    base_cont = built[("continuous", "base")]["meta"]["metrics"]
    burst_cont = built[("continuous", "burst")]["meta"]["metrics"]
    # The scenario switch has to actually switch something. `base` is the clean
    # read of the scheduling policy, with a pool large enough that nothing is
    # ever evicted; `burst` is the same policy under memory pressure. If either
    # stopped being true the page's scenario label would be promising a regime
    # the trace is not in.
    if base_cont["preemptions"] != 0:
        fails.append(f"base 场景不该有抢占，实际 {base_cont['preemptions']} 次")

    # The closed form is an identity only where the run is batch-atomic with no
    # preemption — which is exactly what `base`'s static arm is, and why that is
    # where the claim is checked to the seat rather than merely mentioned. In
    # `burst` the victims re-enter batches the formula never described, so it is
    # not asserted there (see check_claims).
    sb = built[("static", "base")]["meta"]["metrics"]
    if sb["preemptions"] != 0:
        fails.append(f"static/base 不该有抢占，实际 {sb['preemptions']} 次 —— "
                     "闭式断言的前提没了")
    elif sb["static_stall_closed_form"] != sb["stalled_slot_ticks"]:
        fails.append(
            f"static/base 的空转闭式 {sb['static_stall_closed_form']} "
            f"与实际 {sb['stalled_slot_ticks']} 不等")
    if burst_cont["preemptions"] == 0:
        fails.append("burst 场景本该触发抢占，实际 0 次 —— 那就没有抢占有得看")
    if burst_cont["recompute_tokens"] == 0:
        fails.append("burst 场景有抢占却没有重算 —— 恢复策略就不是 recompute 了")

    for scenario in SCENARIOS:
        contrast = check_arm_contrast(built[("static", scenario)],
                                      built[("continuous", scenario)])
        if contrast:
            fails.append(f"{scenario}: 两臂对照不成立")
            for line in contrast:
                print(f"  CONTRAST {scenario}: {line}")
        else:
            s = built[("static", scenario)]["meta"]["metrics"]
            c = built[("continuous", scenario)]["meta"]["metrics"]
            print(f"\n对照（{SCENARIO_LABEL[scenario]}）："
                  f"makespan {s['makespan']} → {c['makespan']}"
                  f"（{s['makespan'] / c['makespan']:.2f}×）· "
                  f"平均延迟 {s['mean_latency']:.1f} → {c['mean_latency']:.1f}"
                  f"（{s['mean_latency'] / c['mean_latency']:.1f}×）· "
                  f"利用率 {100 * s['utilization']:.1f}% → {100 * c['utilization']:.1f}%"
                  f"· 空转 {s['stalled_slot_ticks']} → {c['stalled_slot_ticks']} slot·tick")

    if only_json:
        # Emit the four traces plus one deliberately broken copy, on stdout, for
        # the JS side of the contract to lint. This is how the parity harness
        # (scripts/verify-gantt.py) proves that the two ports of the rules agree:
        # it runs the SAME sabotage table through `LabEngine.gantt.lint` and
        # requires the same verdict. Nothing is written into labs/traces/.
        import copy
        payload = {"traces": {cfg_id(s, sc): built[(s, sc)]
                              for sc in SCENARIOS for s in SCHEDULES},
                   "sabotages": {}}
        base = built[(DEFAULT[0], DEFAULT[1])]
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

    for scenario in SCENARIOS:
        for schedule in SCHEDULES:
            name = cfg_id(schedule, scenario)
            out = HERE / f"{name}.json"
            out.write_text(json.dumps(built[(schedule, scenario)], indent=1,
                                      ensure_ascii=False) + "\n")
            written.append(name)
            print(f"写 {out.name}  {out.stat().st_size} bytes")

    manifest = {
        "set": "continuous-batching",
        "lab": "L10",
        "default": cfg_id(*DEFAULT),
        "params": {"schedule": list(SCHEDULES), "scenario": list(SCENARIOS)},
        "traces": written,
        "labels": {cfg_id(s, sc): f"{SCHEDULE_LABEL[s]} · {SCENARIO_LABEL[sc]}"
                   for sc in SCENARIOS for s in SCHEDULES},
        "notes": {cfg_id(s, sc): SCENARIOS[sc]["note"]
                  for sc in SCENARIOS for s in SCHEDULES},
    }
    (HERE / "continuous-batching.manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"\ncontinuous-batching.manifest.json: {len(written)} 个配置，"
          f"默认 {manifest['default']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
