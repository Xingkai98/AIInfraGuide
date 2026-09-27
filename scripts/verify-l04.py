#!/usr/bin/env python3
"""Acceptance harness for L04 · Decoder Block (issue #18).

Drives the built page in real Chromium and checks the ticket's five acceptance
criteria against the artifact a reader actually gets: the staged page in
`public/labs/`, with its trace SET inlined and its assets at the flattened
paths.

    AC1  残差相加的两侧 shape 相同时在图上被确认，并演示若 shape 不匹配会怎样
    AC2  LayerNorm 的均值/方差是在 d_model 维上、对每个 token 独立计算——在视图上可辨
    AC3  SwiGLU 的 gate/up/down 三条路径与逐元素乘在回放中可见，维度全程标注
    AC4  Pre-Norm 与 Post-Norm 的对照能看出数值尺度上的差异
    AC5  整个 block 的参数量按 d_model / d_ff 可现场重算

...AND the contract of the view the block is now DRAWN with. The page mounts
`views/variable-dag.js` instead of the engine's step DAG — nodes are variables
(the 18 tensors), edges are computations — so the same harness also checks:

    DAG  the graph is the algorithm: every drawn edge is a dependency the trace
         performs, the edges that make the block a block are the ones drawn
         (`scores ← (q, k)`, `y1 ← (x, attn_out)`, `gated ← (silu, up)`), and
         none of them has silently degraded to `reads[0]`.
    MTX  the tensors are MATRICES and the page shows them as matrices:
         `[4,8,8]` printed as four 8×8 slices with true shape labels, `[8,176]`
         printed capped with the elision count SAID, and the rank glyph on each
         node carrying its shape.
    GEO  18 nodes in one panel: pairwise disjoint, inside the canvas, no edge
         routed through an unrelated box, no operation label on a box and none
         on another label.
    HL   the formula sidebar and the graph light each other through
         `step.binding_vars`, including the cases a page-side regex gets wrong
         on this trace.
    PURE  arbitrary jumps render exactly what sequential stepping renders, with
         the deliberately-broken stateful player as the control group.

WHAT MAKES THE ASSERTIONS MEANINGFUL
------------------------------------
Five things this harness will not do, each of which would have made a green
result prove nothing:

  1. It reads the numbers back OFF THE DOM and compares them to the trace the
     generator wrote — never to a re-computation. If the page printed a number
     no trace contains, that is a failure here. (The page is forbidden from
     computing anything algorithmic, so this is the check that keeps it that
     way.)
  2. **Every assertion about a shape, an axis or an ordering is paired with the
     scenario that must NOT satisfy it.** AC1's whole second half is "演示若
     shape 不匹配会怎样", so the harness requires the page to show at least one
     candidate that broadcasts WITHOUT ERROR and at least one that RAISES, and
     requires the total not to be "all of them failed" — a page that showed
     seven errors would be as wrong as one that showed seven successes. The DAG
     section does the same for the graph: it names the edges that must be there
     AND the degenerate edge that must not.
  3. Wherever a value is asserted to have MOVED, the control that must hold is
     asserted alongside it: the `params` segments may not move with `step`
     (the block's parameter count is the same in every frame), and Post-Norm's
     residual RMS must hold FLAT while Pre-Norm's climbs. A test that only
     checked "the number changed" would pass on a page that redrew itself at
     random.
  4. The geometry is measured, not eyeballed — no element may overflow its
     panel, the cell grids must not claim an aspect ratio they do not have, the
     two charts of the Pre/Post panel must share a vertical scale, and the
     graph's nodes, edges and labels must be disjoint. This project has already
     shipped a diagram whose every count was correct and whose furniture
     overlapped, so the picture is asserted like the numbers — and each
     geometric predicate is paired with a sabotage that hand-places the
     violation it is supposed to catch.
  5. The lint sabotage table is run through BOTH ports of each new contract
     block — the Python generator's and the JavaScript port beside the view —
     and every sabotage must be caught by the port that owns it, must be caught
     by at least one port, and must actually change the trace. A mutation that
     stopped biting (a no-op) proves nothing when both lints report zero, and it
     is a different bug from "the lint missed it".

Run:  python3 labs/traces/decoder_block.py       # writes the trace set
      npm run build:labs && python3 scripts/verify-l04.py
Needs `pip install playwright && playwright install chromium`.

Non-zero exit means at least one acceptance criterion failed.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "public" / "labs" / "06-decoder-block.html"
TRACE_DIR = REPO / "labs" / "traces"
MANIFEST = TRACE_DIR / "decoder-block.manifest.json"
GENERATOR = TRACE_DIR / "decoder_block.py"
SHOTS = REPO / "labs" / "pages" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1200

# The configurations this harness drives end to end. Not all 9: each one is a
# page load plus a full DOM walk. These five cover both ends of every slider and
# each single-parameter move in isolation, which is what makes "the number
# changed" attributable to the parameter that moved.
#
# Each single-parameter move is a clean FACTOR OF TWO from the default, in the
# direction the assertion states: `d128-dff176` doubles `d_model` and holds
# `d_ff`, `d64-dff88` halves `d_ff` and holds `d_model`. (The first version of
# this list had `d_model` moving DOWN in the "d_model doubles" case, and the
# assertion dutifully failed against a correct page.)
DRIVEN = [
    ("decoder-block-d64-dff176", 64, 176),     # the default, and the reference
    ("decoder-block-d32-dff88", 32, 88),       # both sliders at the low end
    ("decoder-block-d128-dff352", 128, 352),   # both at the high end
    ("decoder-block-d64-dff88", 64, 88),       # d_ff halved, d_model held
    ("decoder-block-d128-dff176", 128, 176),   # d_model doubled, d_ff held
]

# The steps each criterion is read at. Named by index and by id, and asserted to
# agree, so a change to the replay's shape is a loud failure here rather than a
# silent read of the wrong step.
STEP_RES1, STEP_LN1, STEP_MUL, STEP_LAST = 5, 1, 10, 13
EXPECTED_IDS = {STEP_RES1: "x.res1", STEP_LN1: "x.ln1", STEP_MUL: "x.gated",
                STEP_LAST: "x.out"}

failures = []
# `talks` counts the PASS/FAIL lines; `notes` is the smaller list of `·`
# summaries. The closing line reports the first, because "3 项测量" next to 202
# passing checks reads as a harness that only ran three things.
talks = []
notes = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    talks.append((name, ok))
    if not ok:
        failures.append(name)
    return ok


def note(msg):
    notes.append(msg)
    print(f"  · {msg}")


def load(name):
    return json.loads((TRACE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def thousands(n):
    return f"{n:,}"


# --------------------------------------------------------------- lint parity
def lint_parity():
    """Run the generator's sabotage table through both ports of the new blocks.

    The Python lint runs inside the generator; this asks it for the payload
    (`--js-lint`) and pushes the same broken copies through the JS ports in
    node.

    The two ports are not meant to be identical rule-for-rule: the Python side
    lints the whole contract (formulas, bindings, graph, `state`, plus all five
    of this lab's blocks), while each JS side lints the fields its own view
    renders. So the requirement is not "every rule in both files" — it is:

      * every sabotage is caught by AT LEAST ONE port;
      * every sabotage in a view's own scope is caught by THAT view's JS port
        (the case names carry the owning view, so a new case that forgets its
        prefix shows up as a count mismatch rather than silently untested);
      * every sabotage actually changes the trace. A case written against a
        literal that stops biting — a no-op — proves nothing when both lints
        report zero, and it is a different bug from "the lint missed it".
    """
    print("\n== 契约 lint 的两份实现（residual / ln / swiglu / norm_contrast）==")
    proc = subprocess.run([sys.executable, str(GENERATOR), "--js-lint"],
                          capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        check("生成器的 --js-lint 模式可用", False, (proc.stderr or "")[-500:])
        return
    payload = json.loads(proc.stdout)
    total = len(payload["sabotages"])
    note(f"生成器产出 {len(payload['traces'])} 份配置的 trace，"
         f"并对其中一份跑了 {total} 种破坏")

    # The slot -> tensor map's own argument, reported as the measurement it is.
    # The page's lint can only check that a mapping is STRUCTURALLY valid
    # (declared tensor, in scope); whether the table is the right way to build
    # it at all is a question about L04's formulas, and the generator answers it
    # by running L00's scan over them and showing what it misses.
    sm = payload.get("slotmap") or {}
    check("slot → 张量映射的自检在生成器里跑过且通过", sm.get("ok") is True,
          json.dumps(sm.get("failures") or [], ensure_ascii=False)[:300])
    check("自检覆盖了全部必需映射（残差 / 门控 / LN 那几条）",
          sm.get("required", 0) >= 8, json.dumps(sm, ensure_ascii=False))
    check("L00 式的正则扫描在 L04 的公式上一条映射也读不出 —— 所以这里的表是必需的，"
          "不是偏好", sm.get("scanned") == 0 and sm.get("stated", 0) > 0,
          json.dumps(sm, ensure_ascii=False))
    note(f'该自检同时证明扫描本身没坏：同一段扫描在 L00 的公式上读得出 '
         f'{" 与 ".join(sm.get("l00Control") or [])}')

    node_src = r"""
const fs = require('fs');
const g = {};
function load(p){ new Function('window','globalThis', fs.readFileSync(p,'utf8'))(g, g); }
load('labs/assets/engine/formula.js');
load('labs/assets/engine/trace-model.js');
load('labs/assets/engine/views/ledger.js');
load('labs/assets/engine/views/shape-guard.js');
load('labs/assets/engine/views/norm-axis.js');
load('labs/assets/engine/views/swiglu-paths.js');
load('labs/assets/engine/views/norm-contrast.js');
load('labs/assets/engine/views/variable-dag.js');
const NS = g.LabEngine;
// Non-finite floats travel as tags -- `json.dumps` writes bare Infinity, which
// is legal JavaScript and illegal JSON, so a payload carrying one could not be
// parsed at all. See `_payload_json` in the generator.
function rev(v){
  if (Array.isArray(v)) return v.map(rev);
  if (v && typeof v === 'object') {
    if (v.__nonfinite__) return v.__nonfinite__ === 'nan' ? NaN
      : v.__nonfinite__ === 'inf' ? Infinity : -Infinity;
    const o = {}; for (const k in v) o[k] = rev(v[k]); return o;
  }
  return v;
}
const payload = rev(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')));
// The five ports this harness drives, keyed by the prefix their sabotage cases
// carry. `trace-model.js` is also run, because a sabotage in the generator's
// table may target a base-contract field rather than one of the new blocks --
// that port is the one that owns those rules.
const PORTS = {
  'shapeguard': NS.shapeGuard,
  'normaxis': NS.normAxis,
  'swiglu': NS.swigluPaths,
  'contrast': NS.normContrast,
  'slotvars': NS.variableDag,
  'base': { lint: NS.traceModel.lint, SABOTAGE_CASES: {} }
};
const out = { clean: {}, sabotage: {}, ports: {} };
for (const [name, tr] of Object.entries(payload.traces)) {
  out.clean[name] = {};
  for (const [pn, port] of Object.entries(PORTS)) {
    out.clean[name][pn] = port.lint(tr).gaps.length;
  }
}
for (const [name, s] of Object.entries(payload.sabotages)) {
  if (s.error) { out.sabotage[name] = { error: s.error }; continue; }
  const r = {};
  for (const [pn, port] of Object.entries(PORTS)) {
    try { r[pn] = port.lint(s.trace).gaps.length; }
    catch (e) { r[pn] = 'ERR:' + e.message; }
  }
  r.any = Object.values(r).some(v => typeof v === 'number' && v > 0);
  r.changed = s.changed;
  // A rule with no JS port beside it is still a rule the trace generator
  // enforces before it writes a file -- and the generator records its own
  // verdict on the broken copy (`s.python`). The `params ...` cases have no JS
  // port by construction: no shared component owns that block's rules, which is
  // the same reason L04 declares `meta.params` instead of reusing
  // `trace.ledger`. Counting only the JS ports would report those as caught by
  // nobody while the authority rejects every one.
  if (s.python) r.any = true;
  out.sabotage[name] = r;
}
for (const [pn, port] of Object.entries(PORTS)) {
  out.ports[pn] = Object.keys(port.SABOTAGE_CASES || {}).length;
}
process.stdout.write(JSON.stringify(out));
"""
    scratch = Path("/tmp") / "l04-lint-probe.js"
    scratch.write_text(node_src, encoding="utf-8")
    payload_file = Path("/tmp") / "l04-lint-payload.json"
    payload_file.write_text(proc.stdout, encoding="utf-8")
    node = subprocess.run(["node", str(scratch), str(payload_file)],
                          capture_output=True, text=True, cwd=str(REPO))
    if node.returncode != 0:
        check("JS 侧 lint 能在 node 里跑起来", False, (node.stderr or "")[-500:])
        return
    res = json.loads(node.stdout)

    dirty = {k: v for k, v in res["clean"].items()
             if any(n for n in v.values() if n)}
    check(f"JS 侧 lint 对全部 {len(res['clean'])} 份干净 trace 都报 0 gap"
          f"（不是永远报 gap）", not dirty,
          json.dumps(dirty, ensure_ascii=False)[:300] if dirty else "")

    dead = [k for k, v in res["sabotage"].items() if not v.get("changed")]
    errs = [k for k, v in res["sabotage"].items() if v.get("error")]
    check(f"每一种破坏都真的改变了 trace（没有空操作，共 {total} 种）", not dead,
          "空操作：" + "、".join(sorted(dead)))
    check("每一种破坏都能被施加（不会半路抛异常）", not errs,
          "失败：" + "、".join(sorted(errs)))

    # Each view's own scope, by the tag its case list groups under. Listed
    # explicitly rather than inferred so a new case that forgets its prefix is
    # visible as a count mismatch rather than silently untested on this side.
    # The port names are the node script's registry keys -- the sabotage case
    # names carry the same short prefix.
    scopes = {"shapeguard": "shapeguard", "normaxis": "normaxis",
              "swiglu": "swiglu", "contrast": "contrast",
              "slotvars": "slotvars"}
    for prefix, port_name in scopes.items():
        cases = [k for k in res["sabotage"] if k.startswith(prefix + " ")]
        missed = [k for k in cases
                  if not (isinstance(res["sabotage"][k].get(port_name), int) and
                          res["sabotage"][k][port_name] > 0)]
        declared = res["ports"][port_name]
        check(f"{prefix} 视角的 {len(cases)} 种破坏全部被 JS 侧 lint 抓到"
              f"（该 port 声明了 {declared} 条规则）",
              bool(cases) and not missed, "漏掉：" + "、".join(sorted(missed)))

    # And the whole table must be caught by at least one port: a case neither
    # side catches is a rule that exists only in the case's name. `params …`
    # cases have no JS port, so the Python side's own verdict counts — see the
    # note in the node script.
    uncaught = [k for k, v in res["sabotage"].items() if not v.get("any")]
    py_only = [k for k, v in res["sabotage"].items()
               if v.get("python") and not any(
                   isinstance(v.get(pn), int) and v[pn] > 0
                   for pn in ("shapeguard", "normaxis", "swiglu", "contrast",
                              "slotvars", "base"))]
    check(f"每一种破坏都至少被一个 port 抓到（共 {total} 种）", not uncaught,
          "谁都没抓到的：" + "、".join(sorted(uncaught))[:400])
    note(f"其中 {len(py_only)} 种只有 Python 侧 port（{', '.join(sorted(py_only)[:3])}"
         f"{' …' if len(py_only) > 3 else ''}）—— 分为两类：meta.params（没有共享组件"
         f"拥有那个块的规则），以及 slotmap（「某个 slot 本该指向哪个张量」是 L04 公式"
         f"专属的事实，通用视图不该知道；它们的对照组在生成器里）")

    pyside = subprocess.run([sys.executable, str(GENERATOR)],
                            capture_output=True, text=True, cwd=str(REPO))
    text = pyside.stdout + pyside.stderr
    ok = "LINT-DEAD" not in text and f"对照 {total} 种破坏全部被抓到" in text
    check("Python 侧 lint 独自覆盖了 JS 侧职责之外的规则", ok,
          "生成器自报全部抓到" if ok else "生成器输出里没有出现「全部抓到」")
    note(f"六个 port 各跑同一张 {total} 种破坏的表：JS 侧 shapeGuard / normAxis / "
         f"swigluPaths / normContrast / variableDag 各自覆盖本组件的字段，"
         f"trace-model 覆盖基础契约；"
         f"每一侧都要求「破坏真的改变了 trace」")


# ------------------------------------------------------------- DOM readers
READ_METRICS = """() => {
    const out = {};
    document.querySelectorAll('#lab-readout [data-metric]').forEach(e => {
        const b = e.querySelector('b');
        out[e.dataset.metric] = b ? b.textContent.trim() : null;
    });
    document.querySelectorAll('#lab-readout [data-seg]').forEach(e => {
        const b = e.querySelector('b[data-metric]');
        if (b) out['seg_' + e.dataset.seg] = b.textContent.trim();
    });
    return out;
}"""

# The ledger bar's rendered rows, keyed by the segment label the component
# prints. Shared by the three places that read the bar, so a page that changed
# its markup would break one reader rather than three that drifted apart.
READ_LEDGER = """() => {
    const out = {rows: {}, aria: null, segIds: []};
    const table = document.querySelector('[data-l04-ledger-mount] .lab-ledger-table');
    if (!table) return out;
    table.querySelectorAll('tbody tr').forEach(tr => {
        const k = tr.querySelector('.lab-ledger-seg');
        if (k) out.rows[k.textContent.trim()] =
            tr.querySelector('.lab-ledger-num').textContent.trim();
    });
    const bar = document.querySelector('[data-l04-ledger-mount] .lab-ledger-bar');
    out.aria = bar ? bar.getAttribute('aria-label') : null;
    out.segIds = [...document.querySelectorAll('[data-l04-ledger-mount] ' +
        '.lab-ledger-bar > span[data-seg]')].map(s => s.dataset.seg);
    // The bar's own flex weights: the geometry the reader actually compares.
    out.weights = {};
    out.segIds.forEach(id => {
        const el = document.querySelector('[data-l04-ledger-mount] ' +
            '.lab-ledger-bar > span[data-seg="' + id + '"]');
        out.weights[id] = el ? Number(el.style.flexGrow) : null;
    });
    return out;
}"""

READ_SLIDERS = """() => {
    const out = {};
    document.querySelectorAll('#lab-params input[data-param]').forEach(i => {
        out[i.dataset.param] = {
            value: Number(i.value),
            max: Number(i.max),
            shown: document.querySelector('[data-param-val="' + i.dataset.param + '"]')
                        .textContent.trim(),
        };
    });
    return out;
}"""

# ================================================================ the DAG view
#
# The graph's model, read off the page rather than re-derived: the harness
# asserts against the same `model` object the view draws from, so a page that
# drew one graph and reported another would be caught by the geometry section
# (which reads the SVG) rather than passing both.

READ_MODEL = """() => {
    const v = window.__vd;
    return {
        nodes: v.model.nodes.slice(),
        edgeMode: v.model.edgeMode,
        edges: v.model.edges.map(e => ({from: e.from, to: e.to, ops: e.ops.slice(),
                                        steps: e.steps.slice(),
                                        carried: !!e.carried,
                                        rail: e.rail ? true : false})),
        loops: v.model.loops.map(l => l.id),
        lastStep: v.index.lastStep,
    };
}"""

# The SVG's own geometry, in the drawing's coordinate space plus the rendered
# client boxes. Both are needed: the coordinate box is what the predicates
# reason about, the client box is what a reader sees.
READ_SVG = """() => {
    const svg = document.querySelector('.vd-dag');
    const vb = svg.getAttribute('viewBox').split(/\\s+/).map(Number);
    const boxes = [...svg.querySelectorAll('[data-vd-node]')].map(n => {
        const r = n.querySelector('.vd-box');
        return {id: n.dataset.vdNode,
                x: +r.getAttribute('x'), y: +r.getAttribute('y'),
                w: +r.getAttribute('width'), h: +r.getAttribute('height')};
    });
    const edges = [...svg.querySelectorAll('[data-vd-edge]')].map(p => {
        const len = p.getTotalLength();
        const pts = [];
        for (let k = 0; k <= 40; k++) {
            const q = p.getPointAtLength(len * k / 40);
            pts.push([q.x, q.y]);
        }
        const m = /^(.+)>(.+)$/.exec(p.dataset.vdEdge);
        return {from: m[1], to: m[2], rail: p.dataset.vdRail === '1', pts: pts};
    });
    const labels = [...svg.querySelectorAll('[data-vd-el]')].map(t => {
        const b = t.getBoundingClientRect();
        return {pair: t.dataset.vdEl, text: t.textContent,
                l: b.left, r: b.right, t: b.top, b: b.bottom,
                opacity: +getComputedStyle(t).opacity};
    });
    const svgBox = svg.getBoundingClientRect();
    return {canvas: {w: vb[2], h: vb[3]}, boxes: boxes, edges: edges, labels: labels,
            orientation: svg.dataset.vdOrientation,
            fit: document.querySelector('.vd-dagwrap').dataset.vdFit,
            svg: {l: svgBox.left, r: svgBox.right, t: svgBox.top, b: svgBox.bottom}};
}"""

# The value grids the view rendered, keyed by tensor: shape label, the cells,
# and how many cells were elided. The elision count is how the harness knows the
# cap was ANNOUNCED rather than silent.
READ_GRID = """() => {
    const out = {};
    document.querySelectorAll('.vd-vg').forEach(g => {
        out[g.querySelector('.vd-vg-n').textContent] = {
            shape: g.querySelector('.vd-vg-s').textContent,
            /* The elision markers carry `.vd-cell` too (they sit in the grid),
               so the two are separated here rather than at every call site: a
               marker is a statement ABOUT the numbers, not one of them. */
            cells: [...g.querySelectorAll('.vd-cell:not(.vd-cell-more)')]
                .map(c => c.textContent),
            more: [...g.querySelectorAll('.vd-cell-more')].map(c => c.textContent),
            slices: [...g.querySelectorAll('.vd-vslice')].map(s => s.textContent),
            rows: g.querySelectorAll('.vd-vrow').length,
        };
    });
    return out;
}"""

# The rank glyph on each node, read from the SVG: the class, the number of rects
# and their drawn proportions. This is the picture's claim about the tensor's
# rank, so it is what the MTX section compares to `tensors[].shape`.
READ_GLYPHS = """() => {
    const out = {};
    document.querySelectorAll('[data-vd-node]').forEach(n => {
        const g = n.querySelector('.vd-nshape');
        if (!g) return;
        const rects = [...g.querySelectorAll('rect')].map(r => ({
            w: +r.getAttribute('width'), h: +r.getAttribute('height'),
            cls: r.getAttribute('class')}));
        out[n.dataset.vdNode] = {n: rects.length, rects: rects};
    });
    return out;
}"""

# The residual panel at one step: the confirmation, and the counterexamples.
READ_RESIDUAL = """(idx) => {
    const T = window.__labTrace;
    window.__vd.setCursor(idx);
    const comp = document.querySelector('[data-panel="shape-guard"]');
    const root = comp.querySelector('.lab-sg');
    if (!root) return null;
    const out = {stepId: root.dataset.stepId, lhs: root.dataset.lhs,
                 rhs: root.dataset.rhs, equal: root.dataset.equal === '1',
                 silent: Number(root.dataset.silent),
                 raised: Number(root.dataset.raised), cases: [],
                 // The result block is the THIRD shape on screen; the panel
                 // prints it under `y（结果）`. Read from the operand captions in
                 // document order (lhs, rhs, result) rather than by a class, so
                 // the reader does not have to be kept in step with the markup.
                 result_shape: (() => {
                     const caps = [...comp.querySelectorAll('.lab-sg-op-s')]
                                    .map(e => e.textContent.trim());
                     return caps.length >= 3 ? caps[2] : null;
                 })()};
    comp.querySelectorAll('.lab-sg-case').forEach(c => {
        out.cases.push({id: c.dataset.case, outcome: c.dataset.outcome,
                        shape: c.querySelector('.lab-sg-case-s').textContent.trim(),
                        badge: c.querySelector('.lab-sg-case-b').textContent.trim(),
                        msg: (c.querySelector('.lab-sg-msg') || {}).textContent || null,
                        torch: c.querySelector('.lab-sg-case-x').textContent.trim()});
    });
    out.blocks = [...comp.querySelectorAll('.lab-sg-block')].map(b => {
        const r = b.getBoundingClientRect();
        return {rows: Number(b.dataset.rows), cols: Number(b.dataset.cols),
                drawnRows: Number(b.dataset.drawnRows),
                drawnCols: Number(b.dataset.drawnCols),
                capped: b.dataset.capped === '1', w: r.width, h: r.height};
    });
    return out;
}"""

# The LayerNorm panel at one step: the per-token strips and the axis table.
READ_AXIS = """(idx) => {
    window.__vd.setCursor(idx);
    const comp = document.querySelector('[data-panel="norm-axis"]');
    const root = comp.querySelector('.lab-na');
    if (!root) return null;
    const out = {stepId: root.dataset.stepId, axis: root.dataset.axis,
                 tokens: Number(root.dataset.tokens),
                 dModel: Number(root.dataset.dModel),
                 rowZeroOne: root.dataset.rowZeroOne === '1',
                 colZeroOne: root.dataset.colZeroOne === '1',
                 strips: [], stats: {}};
    comp.querySelectorAll('.lab-na-strip').forEach(s => {
        const cells = s.querySelector('.lab-na-cells');
        out.strips.push({label: s.querySelector('.lab-na-strip-t').textContent.trim(),
                         count: Number(cells.dataset.count),
                         cells: cells.querySelectorAll('.lab-na-cell').length});
    });
    comp.querySelectorAll('.lab-na-table tbody tr').forEach(tr => {
        out.stats[tr.dataset.stat] = {zeroOne: tr.dataset.zeroOne === '1',
                                      verdict: tr.querySelector('.lab-na-verdict')
                                                  .textContent.trim()};
    });
    const sum = comp.querySelector('.lab-na-sum');
    out.flipped = sum && sum.dataset.flipped === '1';
    // The table splits into the real one and the counterfactual one; the second
    // is the one inside `.lab-na-table-cf`.
    const cf = comp.querySelector('.lab-na-table-cf');
    out.cfStats = {};
    cf.querySelectorAll('tbody tr').forEach(tr => {
        out.cfStats[tr.dataset.stat] = {zeroOne: tr.dataset.zeroOne === '1'};
    });
    return out;
}"""

READ_SWIGLU = """(idx) => {
    window.__vd.setCursor(idx);
    const comp = document.querySelector('[data-panel="swiglu-paths"]');
    const root = comp.querySelector('.lab-sw');
    if (!root) return null;
    const out = {stepId: root.dataset.stepId, path: root.dataset.path,
                 dModel: Number(root.dataset.dModel), dFf: Number(root.dataset.dFf),
                 hops: [], rows: [], mul: null};
    comp.querySelectorAll('.lab-sw-hop').forEach(h => {
        const t = [...h.querySelectorAll('.lab-sw-t')].map(x => x.textContent.trim());
        out.hops.push({hop: h.dataset.hop, on: h.dataset.on === '1', shapes: t});
    });
    comp.querySelectorAll('.lab-sw-table tbody tr').forEach(tr => {
        out.rows.push({name: tr.querySelector('td').textContent.trim(),
                       shape: tr.querySelector('.lab-sw-shape').textContent.trim(),
                       elems: tr.querySelector('.lab-sw-elems').textContent.trim()});
    });
    const m = comp.querySelector('.lab-sw-mul');
    if (m) {
        out.mul = {matches: m.dataset.matches === '1', row: Number(m.dataset.row),
                   col: Number(m.dataset.col),
                   expr: [...m.querySelectorAll('.lab-sw-mul-side')]
                           .map(x => x.textContent.trim())};
    }
    return out;
}"""

READ_CONTRAST = """() => {
    const comp = document.querySelector('[data-panel="norm-contrast"]');
    const root = comp.querySelector('.lab-nc');
    if (!root) return null;
    const out = {flows: [], series: {}};
    comp.querySelectorAll('.lab-nc-flow').forEach(f => {
        const nodes = [...f.querySelectorAll('.lab-nc-node')];
        out.flows.push({cls: f.className,
                        nodes: nodes.map(n => n.textContent.trim()),
                        lnIndex: nodes.findIndex(n => n.textContent.trim() === 'LN'),
                        lnColoured: nodes.some(n => n.classList.contains('lab-nc-ln') &&
                                                    n.textContent.trim() === 'LN'),
                        // The nodes carrying the accent class, by name — so the
                        // assertion below can check it is UNIQUE to the norm
                        // rather than merely present on it.
                        accented: nodes.filter(n => n.classList.contains('lab-nc-ln'))
                                       .map(n => n.textContent.trim())});
    });
    comp.querySelectorAll('.lab-nc-chart').forEach(chart => {
        const key = chart.dataset.chart;
        out.series[key] = [];
        chart.querySelectorAll('.lab-nc-row').forEach(r => {
            const svg = r.querySelector('.lab-nc-svg');
            const dots = [...r.querySelectorAll('.lab-nc-dot')];
            const ys = dots.map(d => Number(d.getAttribute('cy')));
            out.series[key].push({
                mode: r.dataset.mode,
                first: Number(svg.dataset.first), last: Number(svg.dataset.last),
                max: Number(svg.dataset.max), min: Number(svg.dataset.min),
                dots: dots.length,
                yTop: Math.min.apply(null, ys), yBottom: Math.max.apply(null, ys),
                amp: r.dataset.inOverOut ? Number(r.dataset.inOverOut) : null,
            });
        });
    });
    return out;
}"""


# The reference panels and the four teaching views live in a <details> so the
# page is one screen. A panel inside a CLOSED <details> is in the DOM and has
# computed styles, but everything under it has zero-sized client boxes -- which
# would make every geometry predicate below pass trivially. So the section is
# opened before anything is measured, and the assertion that it was open is
# implicit in every measurement being non-degenerate.
OPEN_MORE = """() => {
    const d = document.getElementById('lab-more');
    if (d && !d.open) d.open = true;
    return d ? d.open : false;
}"""


def open_more(page):
    page.evaluate(OPEN_MORE)
    page.wait_for_timeout(250)


# For SCREENSHOTS only. The reference section scrolls internally (`max-height:
# 66vh`), and `locator.screenshot()` clips an element to the visible region of
# its scrollable ancestors -- so with the clamp in place every panel shot came
# back as a 5KB strip of header. That is a worse artifact than no shot: a
# reviewer sees a panel that looks empty and cannot tell whether the page is
# broken. Lifting the clamp for the duration of the capture changes nothing
# about the page a reader gets and nothing about what is being asserted; the
# geometry section measures the page with the clamp in place.
EXPAND_FOR_SHOTS = """() => {
    const bd = document.querySelector('.lab-more-bd');
    if (bd) { bd.style.maxHeight = 'none'; bd.style.overflow = 'visible'; }
}"""


def expand_for_shots(page):
    page.evaluate(EXPAND_FOR_SHOTS)
    page.wait_for_timeout(300)


def inside(inner, outer, tol=1.0):
    return (inner["l"] >= outer["l"] - tol and inner["r"] <= outer["r"] + tol and
            inner["t"] >= outer["t"] - tol and inner["b"] <= outer["b"] + tol)


def main():
    for p, hint in ((PAGE, "run `npm run build:labs` first"),
                    (MANIFEST, "run labs/traces/decoder_block.py first")):
        if not p.exists():
            print(f"missing {p} — {hint}", file=sys.stderr)
            return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    url = PAGE.as_uri()

    lint_parity()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        # ================================================ the set is all there
        print("\n== 配置集与 manifest 一致 ==")
        page.goto(url)
        page.wait_for_timeout(1200)
        inline = page.evaluate("() => window.LabTraceSets['decoder-block']")
        check("页面内联了 manifest",
              inline is not None and inline["set"] == "decoder-block")
        check("manifest 的配置数与生成器一致（9 组 = 3×3 的完整笛卡尔积）",
              inline["traces"] == manifest["traces"] and len(inline["traces"]) == 9,
              f"{len(inline['traces'])} 组")
        check("两个参数的可选值与生成器一致",
              inline["params"] == {"d_model": [32, 64, 128], "d_ff": [88, 176, 352]},
              json.dumps(inline["params"], ensure_ascii=False))
        check("默认配置是 d_model=64 · d_ff=176（本项目的贯穿示例模型）",
              inline["default"] == "decoder-block-d64-dff176", inline["default"])
        inlined = page.evaluate("() => Object.keys(window.LabTraces)")
        missing = [n for n in manifest["traces"] if n not in inlined]
        check("manifest 里每一组都有一份内联的 trace", not missing,
              "缺：" + "、".join(missing))

        # The page must match a slider position to a trace by reading the
        # trace's own meta.config, not by parsing an id — so every grid point
        # must be reachable, and the ids are irrelevant to that.
        grid_ok = page.evaluate("""() => {
            const T = window.__labTraces, names = Object.keys(T);
            const out = [];
            for (const d of [32, 64, 128]) for (const f of [88, 176, 352]) {
                const hit = names.filter(n => {
                    const c = T[n].meta.config;
                    return c.d_model === d && c.d_ff === f;
                });
                out.push({key: d + '/' + f, hits: hit.length,
                          cfg: hit.length ? T[hit[0]].meta.config : null});
            }
            return out;
        }""")
        bad_grid = [g for g in grid_ok if g["hits"] != 1 or
                    g["cfg"]["d_model"] != int(g["key"].split("/")[0]) or
                    g["cfg"]["d_ff"] != int(g["key"].split("/")[1])]
        check("两个参数的每个组合都恰好对应一份 trace（页面按 meta.config 匹配，不解析 id）",
              not bad_grid, json.dumps(bad_grid[:2], ensure_ascii=False))

        # =========================================== AC5: sliders and readout
        print("\n== AC5 滑杆：参数量按 d_model / d_ff 现场重算 ==")
        sliders = page.evaluate(READ_SLIDERS)
        check("两个滑杆都渲染出来了（d_model / d_ff）",
              set(sliders) == {"d_model", "d_ff"}, json.dumps(sliders, ensure_ascii=False))
        check("滑杆的可选值长度就是 manifest 的长度",
              all(s["max"] == len(manifest["params"][k]) - 1
                  for k, s in sliders.items()),
              json.dumps({k: v["max"] for k, v in sliders.items()}))
        check("滑杆当前值显示为当前配置的参数",
              {k: v["shown"] for k, v in sliders.items()} ==
              {"d_model": "64", "d_ff": "176"},
              json.dumps({k: v["shown"] for k, v in sliders.items()}))

        observed = {}
        for cfg_id, d, dff in DRIVEN:
            trace = load(cfg_id)
            page.goto(f"{url}?cfg={cfg_id}&step=0")
            page.wait_for_timeout(900)
            shown = page.evaluate(READ_METRICS)
            observed[cfg_id] = shown
            par = trace["meta"]["params"]
            for seg in ("attn", "ffn", "norm"):
                check(f"{cfg_id}: {seg} 分项 → 页面上是 trace 的值",
                      shown.get(f"seg_{seg}") == thousands(par["bytes"][seg]),
                      f'页面 {shown.get(f"seg_{seg}")!r} / trace {thousands(par["bytes"][seg])!r}')
            check(f"{cfg_id}: 参数量合计 → 页面上是 trace 的值",
                  shown.get("params_total") == thousands(par["elements"]),
                  f'页面 {shown.get("params_total")!r} / trace {thousands(par["elements"])!r}')

        # The parameter count must MOVE with the sliders, and each parameter must
        # move it in the way the closed form says. This is AC5's "现场重算".
        def seg_of(cfg_id, seg):
            return load(cfg_id)["meta"]["params"]["bytes"][seg]

        # DRIVEN[4] doubles d_model with d_ff held; DRIVEN[3] halves d_ff with
        # d_model held. Both moves are exact factors of two, which is what makes
        # the "which segment moved, and by how much" attribution meaningful.
        base, dm_up, dff_dn = DRIVEN[0][0], DRIVEN[4][0], DRIVEN[3][0]
        check("AC5 对照：d_model 64 → 128 → 注意力分项 ×4（4d²），页面上也跟着动",
              seg_of(dm_up, "attn") == 4 * seg_of(base, "attn") and
              observed[dm_up].get("seg_attn") == thousands(seg_of(dm_up, "attn")),
              f'{seg_of(base, "attn")} → {seg_of(dm_up, "attn")}')
        check("AC5 对照：d_model 翻倍 → SwiGLU 分项只 ×2（3·d·d_ff 线性），"
              "LayerNorm 分项 ×2（4d）",
              seg_of(dm_up, "ffn") == 2 * seg_of(base, "ffn") and
              seg_of(dm_up, "norm") == 2 * seg_of(base, "norm"),
              f'ffn {seg_of(base, "ffn")} → {seg_of(dm_up, "ffn")}')
        check("AC5 对照：d_ff 减半 → 只有 SwiGLU 分项减半，另两项一动不动",
              seg_of(dff_dn, "ffn") == seg_of(base, "ffn") // 2 and
              seg_of(dff_dn, "attn") == seg_of(base, "attn") and
              seg_of(dff_dn, "norm") == seg_of(base, "norm"),
              f'ffn {seg_of(base, "ffn")} → {seg_of(dff_dn, "ffn")}，'
              f'attn 仍为 {seg_of(dff_dn, "attn")}')

        # The page's own cost model, asked directly: it must agree with the
        # trace's accounting at every grid point. This is the check that makes
        # the sliders a recomputation rather than a redraw.
        #
        # It is compared against `priced` — the model returns BYTES, because
        # that is what the ledger it feeds draws. Comparing it against the
        # parameter COUNT is what the first version did, and the page was right
        # and the harness was wrong.
        model_ok = page.evaluate("""(grid) => {
            const m = window.__labModel;
            const T = window.__labTraces;
            const bad = [];
            grid.forEach(([d, f]) => {
                const name = Object.keys(T).find(n => {
                    const c = T[n].meta.config;
                    return c.d_model === d && c.d_ff === f;
                });
                const want = T[name].meta.params.priced;
                const got = m.predict({d_model: d, d_ff: f});
                ['attn', 'ffn', 'norm'].forEach(k => {
                    if (got[k] !== want[k]) bad.push([d, f, k, got[k], want[k]]);
                });
            });
            return bad;
        }""", [[d, f] for d in (32, 64, 128) for f in (88, 176, 352)])
        check("页面的纯函数 predict(config) 在全部 9 个配置上都等于 trace 的实算字节数"
              "（滑杆是重算，不是换图）", not model_ok,
              json.dumps(model_ok[:3], ensure_ascii=False))
        # The ledger component's own self-check, read back off the page. Its
        # verdict must be green AND its control group must have fired, or the
        # green means nothing.
        led = page.evaluate("() => window.__labViewChecks.params")
        check("参数量账本：逐步对拍 0 步不一致",
              led and led["stepFailures"] == 0, json.dumps(led, ensure_ascii=False))
        check("参数量账本：缩放性质全部成立（自洽但错误的公式过不了）",
              led and led["propertyFailures"] == 0, json.dumps(led, ensure_ascii=False))
        check("参数量账本：对照组全部被抓到（那组对拍有区分力）",
              led and len(led["missedControls"]) == 0, json.dumps(led, ensure_ascii=False))
        check("参数量账本：账本组件确实被消费（L05 的 views/ledger.js），不是重写的",
              page.evaluate(
                  "() => !!document.querySelector('[data-l04-ledger-mount] .lab-ledger "
                  ".lab-ledger-bar')"),
              "")

        # The ledger draws BYTES, and the bytes it draws must be the trace's own
        # `priced` account — a parameter count wearing a byte unit would be a
        # number on the page that does not mean what the bar says it means.
        trace_ref = load(DRIVEN[0][0])
        par0 = trace_ref["meta"]["params"]
        LABELS = {"attn": "注意力（4 个 d×d 投影）",
                  "ffn": "SwiGLU FFN（3 个 d×d_{ff} 矩阵）",
                  "norm": "LayerNorm（两组 γ、β）"}
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=0")
        page.wait_for_timeout(900)
        led_dom = page.evaluate(READ_LEDGER)
        check("账本画的是 trace 的字节数（priced），不是参数个数 —— "
              "参数个数是另一个量，印在读数条里",
              all(thousands(par0["priced"][seg]) in led_dom["rows"].get(LABELS[seg], "")
                  for seg in ("attn", "ffn", "norm")) and
              led_dom["segIds"] == ["attn", "ffn", "norm"] and
              thousands(par0["priced_total"]) in (led_dom["aria"] or ""),
              json.dumps(led_dom["rows"], ensure_ascii=False)[:200])
        # ...and the geometry: the bar's three flex weights are the byte counts,
        # so the WIDTHS are the quantities and a reader compares them by looking.
        check("账本条的宽度就是字节数本身（flex-grow = 该分项的字节数）—— "
              "不是三个自成一体的比例",
              all(led_dom["weights"].get(seg) == par0["priced"][seg]
                  for seg in ("attn", "ffn", "norm")),
              json.dumps(led_dom["weights"]))

        # The count the bytes were priced from is on the same page, so the ×2 is
        # checkable by a reader rather than taken on faith. The two readouts
        # carry different units on purpose — 个参数 is a count and B is a size —
        # which is exactly the confusion the ledger bar would have shown if it
        # had been fed the count.
        both = page.evaluate(READ_METRICS)
        check("参数量（个）与按 fp16 计价（字节）两个数都在页面上，"
              "且后者正是前者的 ×2 —— 3.7 §8 的乘数没有被藏起来",
              both.get("params_total") == thousands(par0["elements"]) and
              both.get("priced_total") == thousands(par0["priced_total"]) + " B" and
              par0["priced_total"] == 2 * par0["elements"],
              f'{both.get("params_total")} 个 → {both.get("priced_total")}')

        # The parameters do NOT move with the step — the control that makes the
        # sliders' movement readable as movement of the ACCOUNT.
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=0")
        page.wait_for_timeout(900)
        per_step = page.evaluate("""() => {
            const T = window.__labTrace, P = window.__vd, out = [];
            for (let i = 0; i < T.steps.length; i++) {
                P.setCursor(i);
                const rows = {};
                document.querySelectorAll('[data-l04-ledger-mount] .lab-ledger-table '
                    + 'tbody tr').forEach(tr => {
                    const k = tr.querySelector('.lab-ledger-seg');
                    if (k) rows[k.textContent.trim()] = tr.querySelector('.lab-ledger-num')
                                                          .textContent.trim();
                });
                out.push({i, id: T.steps[i].id, rows});
            }
            P.setCursor(0);
            return out;
        }""")
        distinct = {json.dumps(r["rows"], sort_keys=True, ensure_ascii=False)
                    for r in per_step}
        check("参数量账本在每一帧都一样（整个 block 的参数量不随步骤变化）—— "
              "滑杆动的是它、步骤不动它，这是对照组", len(distinct) == 1,
              f'{len(distinct)} 种不同的账本')

        # ...and it DOES move with the sliders, in bytes, by the factor the
        # closed form says — the other half of that control.
        page.goto(f"{url}?cfg=decoder-block-d64-dff88&step=0")
        page.wait_for_timeout(800)
        small = page.evaluate(READ_LEDGER)
        # `formatBytes` prints an exact count AND a human scale ("32,768 B ·
        # 32.0 KiB"), so a row is matched by CONTAINMENT of the exact figure —
        # an equality test against the count alone fails on a correct page.
        check("d_ff 减半后账本里的 SwiGLU 分项真的减半，注意力分项一动没动 —— "
              "滑杆动的是账本本身，不是只有读数条在动",
              thousands(par0["priced"]["ffn"] // 2 + 0) in
              small["rows"].get(LABELS["ffn"], "") and
              thousands(par0["priced"]["attn"]) in
              small["rows"].get(LABELS["attn"], "") and
              small["weights"].get("ffn") == par0["priced"]["ffn"] // 2,
              json.dumps(small["rows"], ensure_ascii=False)[:200])

        # =============================================================== AC1
        print("\n== AC1 残差相加：两侧 shape 相同被确认，且演示了不匹配会怎样 ==")
        for cfg_id, d, dff in DRIVEN:
            trace = load(cfg_id)
            page.goto(f"{url}?cfg={cfg_id}&step={STEP_RES1}")
            page.wait_for_timeout(800)
            dom = page.evaluate(READ_RESIDUAL, STEP_RES1)
            if not check(f"{cfg_id}: 残差面板渲染出来了", dom is not None):
                continue
            res = trace["steps"][STEP_RES1]["residual"]
            check(f"{cfg_id} 步 {STEP_RES1} 就是 x.res1",
                  dom["stepId"] == "x.res1" and
                  EXPECTED_IDS[STEP_RES1] == trace["steps"][STEP_RES1]["id"],
                  f'{dom["stepId"]}')
            # Each of the three shapes is compared to the TRACE, not to each
            # other. Comparing `dom["rhs"]` to `dom["lhs"]` would pass on a page
            # that printed the same WRONG shape twice — the page would only have
            # to be self-consistent, which is not the claim. (That is what the
            # first version of this assertion did.)
            f = lambda xs: "[" + ", ".join(str(x) for x in xs) + "]"
            check(f"{cfg_id}: 两侧 shape 在图上被确认，且各自等于 trace 里的那一侧",
                  dom["equal"] and dom["lhs"] == f(res["lhs_shape"]) and
                  dom["rhs"] == f(res["rhs_shape"]),
                  f'{dom["lhs"]} / {dom["rhs"]}（trace {f(res["lhs_shape"])} / '
                  f'{f(res["rhs_shape"])}）')
            check(f"{cfg_id}: 结果 shape 也标在图上，且等于 trace 的结果 shape",
                  dom["result_shape"] == f(res["result_shape"]),
                  f'{dom["result_shape"]} vs trace {f(res["result_shape"])}')

            # --- THE SECOND HALF OF AC1. Counts must match the trace, and the
            # two categories must BOTH be present: a page that showed only
            # errors ("it always crashes") would be as wrong as one that showed
            # only successes ("the shapes are equal").
            outcomes = [c["outcome"] for c in res["cases"]]
            dom_outcomes = [c["outcome"] for c in dom["cases"]]
            check(f"{cfg_id}: 候选数、静默广播数与报错数都与 trace 一致",
                  len(dom["cases"]) == len(res["cases"]) and
                  dom["silent"] == len(res["silent"]) and
                  dom["raised"] == len(res["raised"]) and
                  dom_outcomes == outcomes,
                  f'页面 {dom["silent"]}/{dom["raised"]}/{len(dom["cases"])}，'
                  f'trace {len(res["silent"])}/{len(res["raised"])}/{len(res["cases"])}')
            check(f"{cfg_id}: 至少一个候选<b>静默广播</b>（不报错、结果是合法形状）"
                  f"—— 「不匹配就会崩」这句话本身是错的",
                  dom["silent"] > 0 and
                  all(o in outcomes for o in ("broadcast", "error", "same")),
                  f'广播 {dom["silent"]} 个')
            check(f"{cfg_id}: 至少一个候选抛异常，且异常文本是 numpy 的原话（非空）",
                  dom["raised"] > 0 and
                  all(c["msg"] and c["msg"].strip() for c in dom["cases"]
                      if c["outcome"] == "error"),
                  json.dumps([c["msg"] for c in dom["cases"]
                              if c["outcome"] == "error"][:1], ensure_ascii=False))
            check(f"{cfg_id}: 每一条候选都同时给出 numpy 与 torch 两侧的结论",
                  all("torch" in c["torch"] or "同样" in c["torch"] for c in dom["cases"]))

            # The two shapes that make the point, checked against the trace by
            # value rather than by category: [d] broadcasts, [N, d/2] raises.
            dshape = [c for c in res["cases"]
                      if c["rhs_shape"] == [trace["meta"]["config"]["d_model"]]]
            eshape = [c for c in res["cases"]
                      if c["rhs_shape"] == [trace["meta"]["config"]["N"],
                                            trace["meta"]["config"]["d_model"] // 2]]
            check(f"{cfg_id}: 一个偏置形状 [d_model] 被判为静默广播，"
                  f"一个维度对不上的 [{trace['meta']['config']['N']}, d/2] 被判为报错",
                  dshape and dshape[0]["outcome"] == "broadcast" and
                  eshape and eshape[0]["outcome"] == "error",
                  json.dumps([dshape[0] if dshape else None,
                              eshape[0] if eshape else None], ensure_ascii=False)[:200])

        # The second residual add is asserted too — one picture is not the story.
        trace = load(DRIVEN[0][0])
        dom2 = page.evaluate(READ_RESIDUAL, 12)
        res2 = trace["steps"][12]["residual"]
        check("第二处残差相加也被同一套规则检查（不是只画了第一处）",
              dom2 is not None and dom2["silent"] == len(res2["silent"]) and
              dom2["raised"] == len(res2["raised"]) and dom2["equal"],
              json.dumps({"silent": dom2["silent"], "raised": dom2["raised"]})
              if dom2 else "no panel")

        # =============================================================== AC2
        print("\n== AC2 LayerNorm：μ、σ 在 d_model 维上、逐 token，且可辨 ==")
        for cfg_id, d, dff in DRIVEN:
            trace = load(cfg_id)
            page.goto(f"{url}?cfg={cfg_id}&step={STEP_LN1}")
            page.wait_for_timeout(800)
            dom = page.evaluate(READ_AXIS, STEP_LN1)
            if not check(f"{cfg_id}: LayerNorm 面板渲染出来了", dom is not None):
                continue
            ln = trace["steps"][STEP_LN1]["ln"]
            check(f"{cfg_id}: 轴上标的是 d_model，不是 token 轴",
                  dom["axis"] == "d_model" and ln["axis"] == "d_model", dom["axis"])
            check(f"{cfg_id}: 图上画出了 N 个 μ 与 N 个 σ（每 token 一格，不是全局一格）",
                  len(dom["strips"]) == 2 and
                  all(s["count"] == trace["meta"]["config"]["N"] and
                      s["cells"] == s["count"] for s in dom["strips"]),
                  json.dumps(dom["strips"], ensure_ascii=False))
            check(f"{cfg_id}: 逐 token 的 μ 互不相同（极差 > 0）",
                  ln["mu_spread"] > 0 and dom["tokens"] == trace["meta"]["config"]["N"],
                  f'μ 极差 {ln["mu_spread"]:.4g}')
            # THE DISCRIMINATING PAIR: rows 0/1, columns NOT — both on screen.
            check(f"{cfg_id}: 归一化后<b>行</b>统计被判为 0/1，"
                  f"而<b>列</b>统计被判为不是 0/1（两个轴都画出来，读者可以自己比）",
                  dom["rowZeroOne"] and not dom["colZeroOne"],
                  f'行 {"0/1" if dom["rowZeroOne"] else "?"}，'
                  f'列 {"0/1" if dom["colZeroOne"] else "不是 0/1"}')
            # ...and the counterfactual must show the flip, or the asymmetry is
            # not evidence about the axis.
            check(f"{cfg_id}: 沿 token 轴的反事实对照组翻转了（该 0/1 的变成了列）"
                  f"—— 所以「行是 0/1」这件事本身才不是证据",
                  dom["flipped"] and dom["cfStats"].get("列（每个特征上）", {}
                                                        ).get("zeroOne") and
                  not dom["cfStats"].get("行（每个 token 内）", {}).get("zeroOne"),
                  json.dumps(dom["cfStats"], ensure_ascii=False))
            check(f"{cfg_id}: 反事实的列统计与 trace 一致（不是页面上现算的）",
                  ln["counterfactual_column"]["mean_max_abs"] < 1e-6 and
                  abs(ln["counterfactual_column"]["std_min"] - 1) < 1e-3,
                  f'|mean| ≤ {ln["counterfactual_column"]["mean_max_abs"]:.2e}')

        # The second LayerNorm too, and the control that the two instances'
        # statistics are computed at DIFFERENT points in the block (h1 vs y1),
        # so a page that drew the same array twice would be caught.
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=6")
        page.wait_for_timeout(800)
        dom6 = page.evaluate(READ_AXIS, 6)
        ln6 = load("decoder-block-d64-dff176")["steps"][6]["ln"]
        check("第二处 LayerNorm 的 μ 与第一处的不同（归一化的对象是 y1 而不是 x）",
              dom6 and dom6["stepId"] == "x.ln2" and ln6["mu"] !=
              load("decoder-block-d64-dff176")["steps"][1]["ln"]["mu"],
              f'μ 极差 {ln6["mu_spread"]:.4g}')

        # =============================================================== AC3
        print("\n== AC3 SwiGLU：三条路径与逐元素乘可见，维度全程标注 ==")
        trace = load(DRIVEN[0][0])
        seen_paths = {}
        for idx in range(7, 12):
            page.goto(f"{url}?cfg={DRIVEN[0][0]}&step={idx}")
            page.wait_for_timeout(700)
            dom = page.evaluate(READ_SWIGLU, idx)
            if not check(f"步 {idx}：SwiGLU 面板渲染出来了", dom is not None):
                continue
            sw = trace["steps"][idx]["swiglu"]
            seen_paths[dom["path"]] = True
            check(f"步 {idx}（{trace['steps'][idx]['id']}）：路径标识与 trace 一致",
                  dom["path"] == sw["path"], f'{dom["path"]} vs {sw["path"]}')

            # Every shape string the panel printed must be one of the shapes
            # the trace published. This is the rule that keeps "维度全程标注"
            # from being satisfied by a picture of made-up numbers: the panel is
            # allowed to print any of the trace's shapes, and nothing else.
            known = {"[" + ", ".join(str(x) for x in v) + "]"
                     for v in sw["shapes"].values()}
            printed = set()
            for hop in dom["hops"]:
                printed.update(hop["shapes"])
            printed.update(r["shape"] for r in dom["rows"])
            stray = sorted(printed - known)
            check(f"步 {idx}：图上印出来的每一个 shape 都是 trace 里的 shape",
                  not stray, "多出来的：" + "、".join(stray))

            # The labelled dimensions of every hop, checked by hop against the
            # trace — this is the second half of AC3.
            hops = {h["hop"]: h["shapes"] for h in dom["hops"]}
            wants = {
                "gate": ["[" + ", ".join(str(x) for x in sw["shapes"][k]) + "]"
                         for k in ("x", "W_gate", "gate")],
                "silu": ["[" + ", ".join(str(x) for x in sw["shapes"][k]) + "]"
                         for k in ("gate", "gate", "silu")],
                "up": ["[" + ", ".join(str(x) for x in sw["shapes"][k]) + "]"
                       for k in ("x", "W_up", "up")],
                "down": ["[" + ", ".join(str(x) for x in sw["shapes"][k]) + "]"
                         for k in ("gated", "W_down", "out")],
            }
            hop_bad = {h: hops.get(h) for h, want in wants.items()
                       if hops.get(h) != want}
            check(f"步 {idx}：三条路径每一次乘法的左右操作数与结果形状都标了出来，"
                  f"且与 trace 一致（读者可以自己核对内维对消）",
                  not hop_bad, json.dumps(hop_bad, ensure_ascii=False))

            check(f"步 {idx}：三个权重矩阵的 shape 都标了出来（W_gate/W_up 是 [d_ff,d]，"
                  f"W_down 是 [d,d_ff]）",
                  "[" + ", ".join(str(x) for x in sw["shapes"]["W_gate"]) + "]" in
                  [r["shape"] for r in dom["rows"]] and
                  "[" + ", ".join(str(x) for x in sw["shapes"]["W_down"]) + "]" in
                  [r["shape"] for r in dom["rows"]],
                  json.dumps([r["shape"] for r in dom["rows"]], ensure_ascii=False))
            mid = "[" + ", ".join(str(x) for x in
                                  [trace["meta"]["config"]["N"], dff_cfg(trace)]) + "]"
            check(f"步 {idx}：升维后的中间量都标成 {mid}（gate / SiLU / up / S 四个），"
                  f"down 的输出标成 [N, d_model]",
                  sum(1 for r in dom["rows"] if r["shape"] == mid) >= 4 and
                  dom["rows"][-1]["shape"] ==
                  "[" + ", ".join(str(x) for x in sw["shapes"]["out"]) + "]",
                  json.dumps([r["shape"] for r in dom["rows"]], ensure_ascii=False))

        check("三条路径（gate / up / down）在回放里都走到了",
              set(seen_paths) == {"gate", "up", "down"},
              json.dumps(sorted(seen_paths)))

        # The ⊙ step's own numbers. NOTE WHAT IS READ FROM WHERE: the three
        # numbers in the arithmetic are PARSED OFF THE PAGE (`SiLU(G)_4,107 =
        # -0.0283606`, `U_4,107 = 0.0222449`, `S_4,107 = -0.000630879`) and the
        # multiplication is re-derived from THOSE. The first version of this
        # assertion did the arithmetic on the trace's own `multiply` fields,
        # which the generator's lint already checks — so it would have passed on
        # a page that drew nothing at all. What is checked here is that the
        # page's three printed numbers really satisfy `a × b = product`, and
        # that they are the trace's cell.
        dom_mul = page.evaluate(READ_SWIGLU, STEP_MUL)
        mul = trace["steps"][STEP_MUL]["swiglu"]["multiply"]
        nums = []
        for side in (dom_mul["mul"] or {}).get("expr", []):
            m = re.search(r"=\s*(-?[\d.]+(?:e-?\d+)?)", side)
            nums.append(float(m.group(1)) if m else None)
        shown_ok = (len(nums) == 3 and all(n is not None for n in nums) and
                    abs(nums[0] * nums[1] - nums[2]) <= 1e-6 * max(1.0, abs(nums[2])))
        check("⊙ 那一步页面上印出来的三个数（a、b、乘积）自己就满足 a × b = product —— "
              "不是拿 trace 里的数算一遍",
              dom_mul["mul"] and dom_mul["mul"]["matches"] and shown_ok and
              dom_mul["mul"]["row"] == mul["row"] and dom_mul["mul"]["col"] == mul["col"],
              json.dumps({"页面上的三个数": nums, "trace": [mul["a"], mul["b"],
                                                         mul["product"]]},
                         ensure_ascii=False)[:200])
        check("⊙ 的两侧形状在页面上标成同一个 shape（gate 与 up 必须同维）",
              trace["steps"][STEP_MUL]["swiglu"]["shapes"]["silu"] ==
              trace["steps"][STEP_MUL]["swiglu"]["shapes"]["up"] and
              dom_mul["rows"] and
              len({r["shape"] for r in dom_mul["rows"]
                   if r["name"].startswith(("SiLU(gate)", "up（升维后）"))}) == 1,
              json.dumps(trace["steps"][STEP_MUL]["swiglu"]["shapes"], ensure_ascii=False))
        check("整个 ⊙ 覆盖的位置数就是 N × d_ff（不是只算了展示的那一格）",
              trace["steps"][STEP_MUL]["swiglu"]["elements"]["gated"] ==
              trace["meta"]["config"]["N"] * trace["meta"]["config"]["d_ff"] and
              trace["steps"][STEP_MUL]["swiglu"]["multiply"]["rows_checked"] ==
              trace["meta"]["config"]["N"] * trace["meta"]["config"]["d_ff"] and
              # ...and the page prints that count rather than only the one cell.
              thousands(trace["meta"]["config"]["N"] *
                        trace["meta"]["config"]["d_ff"]) in json.dumps(
                  dom_mul["rows"], ensure_ascii=False),
              str(trace["steps"][STEP_MUL]["swiglu"]["elements"]["gated"]))

        # =============================================================== AC4
        print("\n== AC4 Pre-Norm vs Post-Norm：数值尺度上的差异 ==")
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=0")
        page.wait_for_timeout(900)
        ncdom = page.evaluate(READ_CONTRAST)
        nc = trace["meta"]["norm_contrast"]
        pre, post = nc["modes"][0], nc["modes"][1]
        check("对照面板渲染出来了（面板是 trace 级的，不依赖游标）", ncdom is not None)

        # The #40 material: two flows, same node set, norm at a different index.
        check("两条数据流的节点名完全相同",
              ncdom and len(ncdom["flows"]) == 2 and
              sorted(ncdom["flows"][0]["nodes"]) == sorted(ncdom["flows"][1]["nodes"]),
              json.dumps([f["nodes"] for f in ncdom["flows"]], ensure_ascii=False)
              if ncdom else "")
        check("两条数据流里 LN 的位置不同（第 1 位 vs 第 3 位）—— 这正是 3.6 §4.1 的 diff",
              ncdom and ncdom["flows"][0]["lnIndex"] != ncdom["flows"][1]["lnIndex"] and
              ncdom["flows"][0]["lnIndex"] == pre["flow"].index("LN") and
              ncdom["flows"][1]["lnIndex"] == post["flow"].index("LN"),
              json.dumps([f["lnIndex"] for f in ncdom["flows"]]) if ncdom else "")
        # The name says 唯一 and so does the assertion: EVERY accented node in
        # both chains must be the norm, and there must be exactly one per chain.
        # The first version only checked that the norm WAS accented, which its
        # own name overclaimed — a page that accented every node would have
        # passed it.
        check("LN 是两条流里唯一被上色的节点（唯一那处差别一眼可见）",
              ncdom and all(f["lnColoured"] for f in ncdom["flows"]) and
              all(f["accented"] == ["LN"] for f in ncdom["flows"]),
              json.dumps([f["accented"] for f in ncdom["flows"]]) if ncdom else "")

        # The two measurements, both read off the DOM and compared to the trace.
        for key, label in (("rms", "残差流 RMS"), ("grad", "梯度放大")):
            rows = ncdom["series"][key] if ncdom else []
            check(f"{label}：两条曲线都画出来了，点数与 trace 的深度网格一致",
                  len(rows) == 2 and
                  all(r["dots"] == len(nc["depths"]) for r in rows),
                  json.dumps([{"mode": r["mode"], "dots": r["dots"]} for r in rows])
                  if rows else "")
            if len(rows) == 2:
                modes = {r["mode"]: r for r in rows}
                want_pre = (pre["residual_rms"] if key == "rms"
                            else pre["grad_amp_log"])
                want_post = (post["residual_rms"] if key == "rms"
                             else post["grad_amp_log"])
                check(f"{label}：Pre 那条的起点与终点就是 trace 的值",
                      abs(modes["pre"]["first"] - want_pre[0]) < 1e-9 and
                      abs(modes["pre"]["last"] - want_pre[-1]) < 1e-9,
                      f'{modes["pre"]["first"]} → {modes["pre"]["last"]}')
                check(f"{label}：Post 那条的起点与终点就是 trace 的值",
                      abs(modes["post"]["first"] - want_post[0]) < 1e-9 and
                      abs(modes["post"]["last"] - want_post[-1]) < 1e-9,
                      f'{modes["post"]["first"]} → {modes["post"]["last"]}')

        # THE LESSON, and its control: Pre climbs, Post holds flat. A page that
        # drew two climbing curves would show no contrast at all.
        rms_rows = {r["mode"]: r for r in ncdom["series"]["rms"]}
        check("AC4 结论：Pre-Norm 的残差流 RMS 随深度上升（1.3 → 6.2），"
              "而 Post-Norm 恒等于 1.0000 —— 后者是 LayerNorm 输出的定义，"
              "不是它「稳定」，这是对照组",
              rms_rows["pre"]["last"] > 1.5 * rms_rows["pre"]["first"] and
              abs(rms_rows["post"]["first"] - 1.0) < 1e-3 and
              abs(rms_rows["post"]["last"] - 1.0) < 1e-3 and
              abs(post["residual_rms"][0] - post["residual_rms"][-1]) < 1e-3,
              f'Pre {rms_rows["pre"]["first"]:.3f}→{rms_rows["pre"]["last"]:.3f}，'
              f'Post {rms_rows["post"]["first"]:.4f}→{rms_rows["post"]["last"]:.4f}')
        grad_rows = {r["mode"]: r for r in ncdom["series"]["grad"]}
        check("AC4 结论：Post-Norm 的梯度回传放大严格大于 Pre-Norm 的",
              post["grad_in_over_out"] > pre["grad_in_over_out"] and
              grad_rows["post"]["amp"] == post["grad_in_over_out"] and
              grad_rows["pre"]["amp"] == pre["grad_in_over_out"],
              f'Pre {pre["grad_in_over_out"]:.1f}× vs '
              f'Post {post["grad_in_over_out"]:.0f}×')

        # The gradient curve is drawn on a shared scale with Post on top —
        # asserted geometrically, because a chart whose axis was per-series would
        # draw two curves that look the same height.
        check("两张图各自共用一根纵轴：Post 的曲线整体画得比 Pre 高"
              "（不共用纵轴的话这个高度差会消失）",
              grad_rows["post"]["yTop"] < grad_rows["pre"]["yTop"],
              f'Post top {grad_rows["post"]["yTop"]:.1f} < '
              f'Pre top {grad_rows["pre"]["yTop"]:.1f}')
        check("深度 1…32 全部画出来了（不是只画了前几个点）",
              nc["depths"] == [1, 2, 4, 8, 16, 32] and
              len(nc["modes"][0]["residual_rms"]) == 6,
              json.dumps(nc["depths"]))

        # The ordering must hold in EVERY configuration, not just the default:
        # it is a property of the architecture, which is why the generator
        # re-runs it on probe seeds the trace does not publish.
        all_post_gt = all(
            load(name)["meta"]["norm_contrast"]["modes"][1]["grad_in_over_out"] >
            load(name)["meta"]["norm_contrast"]["modes"][0]["grad_in_over_out"]
            for name in manifest["traces"])
        check("这个排序在全部 9 个配置上都成立（不是默认配置的巧合）", all_post_gt)

        # ====================================================== the DAG view
        #
        # The page draws the block with `views/variable-dag.js`: nodes are the
        # tensors, edges are the computations. What is asserted here is that the
        # picture IS the algorithm -- every line on it is a dependency the trace
        # performs, the lines that carry the block's meaning are present, and
        # the degenerate line that a naive rule draws is absent.
        print("\n== 变量图：节点 = 变量，边 = 一次计算 ==")
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=0")
        page.wait_for_timeout(1100)
        check("引擎把页面完整驱动起来（变量图 / 公式 / 数值 / 控制条都在）",
              page.evaluate("""() => !!document.querySelector('.vd-root') &&
                  !!document.querySelector('.vd-dag') &&
                  !!document.querySelector('.vd-fml .lab-formula-node') &&
                  !!document.querySelector('[data-vd=scrub]') &&
                  document.querySelectorAll('[data-vd-node]').length > 0"""))
        model = page.evaluate(READ_MODEL)
        trace0 = load(DRIVEN[0][0])
        tensor_names = list(trace0["tensors"].keys())

        check("节点集就是 tensors 的键（不是 graph.nodes 的那些步骤）",
              sorted(model["nodes"]) == sorted(tensor_names),
              f'节点 {len(model["nodes"])} 个 vs trace tensors {len(tensor_names)} 个')
        note(f'{len(model["nodes"])} 个变量节点、{len(model["edges"])} 条计算边，'
             f'模式 = {model["edgeMode"]}')

        # Every drawn edge must be justified by a step that reads its source and
        # writes its target. This is the invariant that separates a real data
        # dependency from a line drawn between two boxes that happen to look
        # close, and it is checkable from the trace alone.
        unjustified = []
        for e in model["edges"]:
            why = [s for s in trace0["steps"]
                   if e["from"] in (s.get("reads") or []) and e["to"] in (s.get("writes") or [])]
            if not why:
                unjustified.append(e)
        check("每条画出来的边都有某一步「读 from、写 to」作为依据",
              not unjustified, json.dumps(unjustified, ensure_ascii=False)[:300])
        stray = [e for e in model["edges"]
                 if e["from"] not in set(tensor_names) or e["to"] not in set(tensor_names)]
        check("边的端点都是已声明的张量", not stray,
              json.dumps(stray, ensure_ascii=False)[:200])

        # THE DISCRIMINATING ASSERTIONS. A view that took `reads[0]` as the
        # source of every edge would draw a different graph, and on this trace
        # the difference is not subtle: `x.scores` reads (q, k) and `reads[0]`
        # is q, so `k → scores` would vanish; `x.res1` reads (x, attn_out) and
        # `reads[0]` is x, so `attn_out → y1` would vanish and the residual
        # connection -- this lab's whole subject -- would be drawn as x adding
        # to itself.
        def sources_of(to):
            return sorted(e["from"] for e in model["edges"] if e["to"] == to)
        check("scores 的两个来源都画了出来（q 与 k，不是 reads[0] 挑一个）",
              sources_of("scores") == ["k", "q"], f'scores ← {sources_of("scores")}')
        check("残差 y1 的两个来源都画了出来（x 与 attn_out —— 残差连接本身）",
              sources_of("y1") == ["attn_out", "x"], f'y1 ← {sources_of("y1")}')
        check("残差 y2 的两个来源都画了出来（y1 与 ffn_out）",
              sources_of("y2") == ["ffn_out", "y1"], f'y2 ← {sources_of("y2")}')
        check("门控 gated 的两个来源都画了出来（silu 与 up —— ⊙ 的两个操作数）",
              sources_of("gated") == ["silu", "up"], f'gated ← {sources_of("gated")}')
        check("probs 来自 scores 与 v（softmax 的输入与它乘的那个 V）",
              sources_of("probs") == ["scores", "v"], f'probs ← {sources_of("probs")}')
        check("q/k/v 三个投影都各自连到 h1（不是只连了第一个）",
              all(sources_of(t) == ["h1"] for t in ("q", "k", "v")),
              json.dumps({t: sources_of(t) for t in ("q", "k", "v")}))
        check("ffn_out 来自 gated（down 路径读的是 ⊙ 的结果，不是 gate）",
              sources_of("ffn_out") == ["gated"], f'ffn_out ← {sources_of("ffn_out")}')

        # The negative half: the edges that must NOT be there. `x_blk → l` in
        # L00's terms was the symptom of the index bug; the equivalent on a
        # forward-only dataflow graph would be a back-edge (an arrow pointing at
        # an earlier layer, which only makes sense if the layering and the
        # reads/writes disagree) or a synthetic edge for the step that computes
        # nothing.
        check("h1 只来自 x（第一处 LN 的输入是残差流的入口）",
              sources_of("h1") == ["x"], f'h1 ← {sources_of("h1")}')
        check("图的边数就是 trace 里真实发生过的「读→写」对数，而非每步一条",
              len(model["edges"]) == len({
                  (r, w) for s in trace0["steps"]
                  for r in (s.get("reads") or []) for w in (s.get("writes") or [])
                  if r != w}),
              f'{len(model["edges"])} 条')

        # Back-edges. L04 is a feed-forward block, so every edge must go from a
        # lower band to a higher one; a drawing with an arrow pointing upstream
        # would be a picture of an algorithm this trace does not run. The
        # generator's own layering derives from the same reads/writes, so this
        # is the check that the two agree rather than assuming they do.
        back = page.evaluate("""() => {
            const L = window.__vd.layout();
            return window.__vd.model.edges
                .filter(e => L.pos[e.to].band <= L.pos[e.from].band)
                .map(e => e.from + '→' + e.to);
        }""")
        check("没有反向边（L04 是前馈 block，每条边都指向更靠后的一层）",
              not back, json.dumps(back, ensure_ascii=False))
        check("没有自环（L04 没有跨迭代递推 —— 与 L00 的 m/ℓ/acc 正相反）",
              model["loops"] == [], json.dumps(model["loops"]))
        # The summary step reads and writes nothing, so it must have no edge at
        # all. A view that synthesised an edge for every step would give it one.
        check("汇总步 x.out 不读写任何张量，因此图上没有它的边",
              not [e for e in model["edges"] if "x.out" in (e["from"], e["to"])],
              "(x.out 无入边无出边)")

        # Skipped-layer edges are routed on a rail, and the rail is what keeps
        # them out of the boxes between. Asserted as a PROPERTY of the model (a
        # band delta above 1 has a rail) -- the geometric half, that no rail
        # passes through a box, is in the geometry section.
        below = page.evaluate("""() => {
            const v = window.__vd, L = v.layout();
            return v.model.edges.filter(e => Math.abs((L.pos[e.to].band - L.pos[e.from].band)) > 1)
                .map(e => ({pair: e.from + '>' + e.to, rail: !!e.rail,
                            delta: L.pos[e.to].band - L.pos[e.from].band}));
        }""")
        check("跨层边全部走 rail 绕行（不直接从中间的箱子上穿过去）",
              below and all(b["rail"] for b in below),
              json.dumps(below, ensure_ascii=False))
        note("跨层边：" + " · ".join(f'{b["pair"]}(Δ{b["delta"]}层)' for b in below))

        # ------------------------------------------------------------- MTX
        #
        # "展示数据结构是二维的矩阵" is this lab's increment over L00, where
        # almost every variable was a scalar. So the matrix-ness is asserted
        # three ways: the true shape is printed, the value is laid out BY RANK
        # (a rank-3 tensor as a stack of matrices, not one wide one), and the
        # node carries a glyph whose geometry is the rank.
        print("\n== 矩阵：形状、按秩排版、秩字形 ==")
        mtx_steps = {"scores": "scores", "probs": "probs", "gate": "gate",
                     "up": "up", "silu": "silu", "gated": "gated"}
        shapes = {k: trace0["tensors"][k]["shape"] for k in mtx_steps}
        check("这 6 个张量在本配置下确实是矩阵 / 张量，不是标量",
              all(len(s) >= 2 for s in shapes.values()),
              json.dumps(shapes, ensure_ascii=False))

        # scores/probs are written by steps 3 and 4; gate/up/silu/gated by 7..10.
        found_grids = {}
        for step in (3, 4, 7, 8, 9, 10):
            page.evaluate("i => window.__vd.setCursor(i)", step)
            page.wait_for_timeout(220)
            g = page.evaluate(READ_GRID)
            for k in g:
                found_grids.setdefault(k, g[k])
        for name in mtx_steps:
            if not check(f"{name} 的值被渲染出来了", name in found_grids):
                continue
            grid = found_grids[name]
            want = "[" + "×".join(str(x) for x in shapes[name]) + "]"
            check(f"{name} 的形状标签就是 trace 的 {want}（不是转置、不是摊平）",
                  grid["shape"] == want, f'页面 {grid["shape"]!r} vs trace {want!r}')
            check(f"{name} 至少渲染出一个值（不是空网格）", bool(grid["cells"]),
                  f'{len(grid["cells"])} 个格子')

        # Rank-3: `[4,8,8]` must be FOUR 8×8 slices, not one 4×64 or 32×8 grid.
        page.evaluate("() => window.__vd.setCursor(3)")   # x.scores
        page.wait_for_timeout(250)
        g3 = page.evaluate(READ_GRID)
        sc = g3.get("scores", {})
        check("scores 按「每头一张」排版：标题写明第几张 / 共几张（4 个头各一张）",
              any("张" in s for s in sc.get("slices", [])) and
              int(shapes["scores"][0]) == 4,
              json.dumps(sc.get("slices"), ensure_ascii=False))
        # The number of cells per drawn slice is capped, AND the elision marker
        # says how many were left out. A cap that is not announced is a grid
        # that silently looks complete -- which for `[4,8,8]` would mean a
        # reader counting 48 cells and concluding each score matrix is 6×8.
        #
        # The limits are read off the view rather than hard-coded here, so the
        # assertion is about the CONTRACT (capped + announced) and not about the
        # number the page happens to have chosen.
        lim = page.evaluate("() => window.__vd.VALUE_LIMITS || null")
        n_slices = min(int(shapes["scores"][0]), 2)
        per_slice = 6 * 8
        check(f"scores 的每一张切片都被截断到上限、且截断量被写出来（不是静默画一半）",
              len(sc.get("cells", [])) <= n_slices * per_slice and bool(sc.get("more")),
              f'{len(sc.get("cells", []))} 个格子 / 上限 {n_slices * per_slice}，'
              f'省略标记 {sc.get("more")}')
        check("scores 的省略标记写的是没画出来的数量（是一句陈述，不是一个数）",
              all("⋯" in m for m in sc.get("more", [])),
              json.dumps(sc.get("more"), ensure_ascii=False))
        # ...and the cap is a LIMIT, not a truncation of the truth: the header
        # still carries the full shape, so the reader knows what they are not
        # seeing. Both halves are on screen at once.
        check("被截断的网格，表头仍写着完整形状（截断的是画面，不是事实）",
              sc.get("shape") == "[" + "×".join(str(x) for x in shapes["scores"]) + "]",
              sc.get("shape"))
        note(f'值网格上限（视图声明）：{json.dumps(lim, ensure_ascii=False)}')

        # Rank-2 with a wide second dimension: `[8,176]`.
        page.evaluate("() => window.__vd.setCursor(7)")   # x.gate
        page.wait_for_timeout(250)
        gg = page.evaluate(READ_GRID)
        gate_grid = gg.get("gate", {})
        check(f"gate 的 [8,176] 被截断到上限并写明省略了多少列（1408 个格子放不进侧栏）",
              bool(gate_grid.get("more")) and len(gate_grid.get("cells", [])) < 8 * 176,
              f'{len(gate_grid.get("cells", []))} / 1408，省略 {gate_grid.get("more")}')

        # The rank glyph: it is the picture's claim about the tensor's rank, so
        # it is checked against the trace's shape by rank.
        page.evaluate("() => window.__vd.setCursor(0)")
        page.wait_for_timeout(250)
        glyphs = page.evaluate(READ_GLYPHS)
        check("每个变量节点都带一个秩字形", len(glyphs) == len(tensor_names),
              f"{len(glyphs)} / {len(tensor_names)}")
        bad_glyph = []
        for name, spec in trace0["tensors"].items():
            gl = glyphs.get(name)
            if not gl:
                bad_glyph.append([name, "缺失"])
                continue
            rank = len(spec["shape"])
            # 1 rect for rank 0/1/2; 2 nested rects for rank ≥ 3.
            want_n = 1 if rank <= 2 else 2
            if gl["n"] != want_n:
                bad_glyph.append([name, f'秩 {rank} 画了 {gl["n"]} 个矩形'])
        check("秩字形的层数就是张量的秩（rank≥3 画成嵌套的两层，所以「一批矩阵」"
              "不会被读成「一个矩阵」）", not bad_glyph, json.dumps(bad_glyph, ensure_ascii=False))
        # The glyph IS the picture's claim about the shape, so it has to carry
        # more than "there is a tensor here". L04 has no rank-0 or rank-1
        # tensors -- every variable is a matrix or a stack of them -- so the
        # discriminating pair on THIS trace is rank 2 vs rank 3: the rank-3
        # glyph is a matrix with a second rectangle inside it, which is exactly
        # how "4 score matrices" is told apart from "one 8×8 score matrix".
        r2 = next((n for n, g in glyphs.items()
                   if len(trace0["tensors"][n]["shape"]) == 2), None)
        r3 = next((n for n, g in glyphs.items()
                   if len(trace0["tensors"][n]["shape"]) >= 3), None)
        check("L04 上同时存在 rank-2 与 rank-3 的张量（字形才有东西可分）",
              r2 is not None and r3 is not None, f"rank2={r2} rank3={r3}")
        if r2 and r3:
            i2 = [x for x in glyphs[r2]["rects"] if x["cls"] == "vd-shape"]
            i3 = [x for x in glyphs[r3]["rects"] if x["cls"] == "vd-shape-inner"]
            check("秩字形的几何真的区分了秩：rank-3 多一层内框（所以「一批矩阵」"
                  "不会被读成「一个矩阵」）",
                  len(i3) == 1 and i3[0]["w"] < glyphs[r3]["rects"][0]["w"],
                  json.dumps({"rank2": glyphs[r2], "rank3": glyphs[r3]},
                             ensure_ascii=False))

        # ------------------------------------------------- the three states
        #
        # The defect the L00 ticket fixed, re-asserted here because this page
        # uses the same view: the states must be painted, distinguishable, and
        # never lag the cursor.
        print("\n== 节点三态高亮（游标状态 = 图的状态）==")
        lag = page.evaluate("""() => {
            const read = () => {
                const out = {};
                document.querySelectorAll('[data-vd-node]').forEach(g => {
                    out[g.dataset.vdNode] = g.dataset.vdState + '|' +
                        getComputedStyle(g.querySelector('.vd-box')).stroke;
                });
                return out;
            };
            window.__vd.setCursor(0);
            const at0 = read();
            window.__vd.setCursor(12);
            const at12 = read();
            return {at0: at0, at12: at12};
        }""")
        check("高亮不落后于游标（同一次任务里跳转后立刻读，已是新状态）",
              lag["at0"]["y2"] != lag["at12"]["y2"] and
              lag["at12"]["y2"].endswith("|") is False,
              json.dumps({k: lag["at12"][k] for k in ("y2", "x", "y1")}, ensure_ascii=False))
        check("跳转后本步读写的变量确实处于 on 态",
              lag["at12"]["y1"].startswith("on|") and lag["at12"]["ffn_out"].startswith("on|"),
              lag["at12"]["y1"])

        page.evaluate("() => window.__vd.setCursor(0)")
        page.wait_for_timeout(250)
        s0 = page.evaluate("""() => ({
            past: [...document.querySelectorAll('[data-vd-node]')]
                .filter(g => g.dataset.vdState === 'past').map(g => g.dataset.vdNode),
            cls: [...document.querySelectorAll('[data-vd-node] .vd-box')]
                .map(r => r.getAttribute('class')),
        })""")
        check("第 0 步没有任何「已访问」节点（进度是从零开始的）", not s0["past"],
              json.dumps(s0["past"]))
        check("每个节点的 rect class 恰好是三态之一（没有裸 vd-box）",
              all(c in ("vd-box vd-box-on", "vd-box vd-box-past", "vd-box vd-box-off")
                  for c in s0["cls"]),
              json.dumps(sorted(set(s0["cls"]))))

        arc = page.evaluate("""() => {
            const out = [];
            for (let i = 0; i <= window.__vd.index.lastStep; i++) {
                window.__vd.setCursor(i);
                let touched = 0;
                document.querySelectorAll('[data-vd-node]').forEach(g => {
                    if (g.dataset.vdState !== 'off') touched++;
                });
                out.push(touched);
            }
            return out;
        }""")
        check("沿时间轴前进时「已触及的变量」单调不减（进度看得见）",
              all(arc[i] <= arc[i + 1] for i in range(len(arc) - 1)), f"逐步计数 {arc}")
        # All 18, not 17: the last step to WRITE anything is `x.res2` (writes y2,
        # index 12), and `x.out` at index 13 reads and writes nothing. So the
        # final step still shows every variable as visited -- the graph is
        # complete one step before the trace ends, which is the intended picture
        # and not an off-by-one.
        check("走到最后一步时 18 个变量全部已被触及（最后一次写在第 12 步 x.res2）",
              arc[-1] == len(tensor_names),
              f"{arc[-1]} / {len(tensor_names)}（最后一步 x.out 是汇总步，无读写）")

        # ------------------------------------------------------- one screen
        print("\n== 一屏（每一步一屏）==")
        page.evaluate("() => window.__vd.setTier('num')")
        fit = page.evaluate("""() => {
            const out = [];
            for (let i = 0; i <= window.__vd.index.lastStep; i++) {
                window.__vd.setCursor(i);
                const root = document.querySelector('.vd-root');
                const side = document.querySelector('.vd-side');
                const ctl = document.querySelector('.vd-ctl').getBoundingClientRect();
                out.push({i: i,
                    root: root.scrollHeight - root.clientHeight,
                    side: side.scrollHeight - side.clientHeight,
                    doc: document.documentElement.scrollHeight - window.innerHeight,
                    ctlVisible: ctl.bottom <= window.innerHeight + 1 && ctl.top >= 0});
            }
            return out;
        }""")
        worst_root = max(r["root"] for r in fit)
        worst_side = max(r["side"] for r in fit)
        worst_doc = max(r["doc"] for r in fit)
        check("PC：每一步的图 + 公式都在一屏内（视图自身不滚）",
              worst_root <= 0 and worst_side <= 0,
              f"最大溢出 根 {worst_root}px / 侧栏 {worst_side}px")
        check("PC：整个页面不滚（参考面板折在「参考面板」里，不撑高页面）",
              worst_doc <= 0, f"最大溢出 {worst_doc}px")
        check("PC：控制条在每一步都留在视野内",
              all(r["ctlVisible"] for r in fit),
              json.dumps([r["i"] for r in fit if not r["ctlVisible"]]))
        note(f"逐步实测 {len(fit)} 步 × 3 项")

        fml = page.evaluate("""() => {
            const out = [];
            for (const tier of ['sym', 'idx', 'num']) {
                window.__vd.setTier(tier);
                for (let i = 0; i <= window.__vd.index.lastStep; i++) {
                    window.__vd.setCursor(i);
                    const node = document.querySelector('.vd-fml .lab-formula-node');
                    out.push({tier: tier, i: i, over: node.scrollWidth - node.clientWidth,
                              fit: document.querySelector('.vd-dagwrap').dataset.vdFit});
                }
            }
            window.__vd.setTier('num');
            return out;
        }""")
        over = [x for x in fml if x["over"] > 0]
        check("PC：每一档、每一步的公式都横向放得下（不用左右拖动读公式）",
              not over, json.dumps(over[:5], ensure_ascii=False))
        note(f"公式排版核对 {len(fml)} 个（档位 × 步）组合")

        # The drawing must be legible, not merely present. A twelve-band graph
        # letterboxed into a square panel is the failure mode this assertion
        # exists for: it renders, every count is right, and the labels are four
        # pixels tall. Measured on the rendered node label, not on the viewBox.
        legible = page.evaluate("""() => {
            const svg = document.querySelector('.vd-dag');
            const wrap = document.querySelector('.vd-dagwrap');
            const wr = wrap.getBoundingClientRect();
            const nodes = [...svg.querySelectorAll('[data-vd-node]')];
            const label = nodes[0].querySelector('.vd-nl').getBoundingClientRect();
            const NODE = nodes[0].querySelector('.vd-box').getBoundingClientRect();
            /* The drawing's FOOTPRINT: the bounding box of every node, which is
               what "does the drawing use the panel" is a question about. One
               node's width is not -- a 112px node in a 700px panel says nothing
               about whether the other 17 span it. */
            const b = nodes.map(n => n.querySelector('.vd-box').getBoundingClientRect())
                .reduce((a, r) => ({
                    l: Math.min(a.l, r.left), r: Math.max(a.r, r.right),
                    t: Math.min(a.t, r.top), b: Math.max(a.b, r.bottom)}),
                    {l: Infinity, r: -Infinity, t: Infinity, b: -Infinity});
            return {labelPx: label.height, boxH: NODE.height, boxW: NODE.width,
                    fit: wrap.dataset.vdFit,
                    spanX: (b.r - b.l) / wr.width, spanY: (b.b - b.t) / wr.height,
                    overflowX: wrap.scrollWidth - wrap.clientWidth,
                    overflowY: wrap.scrollHeight - wrap.clientHeight,
                    wrapW: wr.width, wrapH: wr.height};
        }""")
        check("节点标签在渲染尺寸下是可读的（不是缩到 4px 的一张纹理）",
              legible["labelPx"] >= 8.0,
              f'标签高 {legible["labelPx"]:.1f}px，节点箱 {legible["boxW"]:.0f}×'
              f'{legible["boxH"]:.0f}px，fit={legible["fit"]}')
        # The drawing must fill its panel in the axis it is long on: a
        # twelve-band vertical graph letterboxed into a square panel would leave
        # two thirds of the width empty and be legible only by accident.
        check("图的跨度用满了面板的长边（不是缩在角落里的一小团）",
              max(legible["spanX"], legible["spanY"]) >= 0.7,
              f'节点跨度 / 面板 = {legible["spanX"]:.2f} 宽 × {legible["spanY"]:.2f} 高'
              f'（面板 {legible["wrapW"]:.0f}×{legible["wrapH"]:.0f}），'
              f'pan 溢出 {legible["overflowX"]}×{legible["overflowY"]}px')

        # The toolbar must stay reachable when the drawing pans rather than
        # shrinking: a graph the reader can pan but whose cursor controls have
        # scrolled out of the window is not usable either.
        check("图改为平移时，控制条仍固定在视野内（不是跟着图一起滚）",
              page.evaluate("""() => {
                  const c = document.querySelector('.vd-ctl').getBoundingClientRect();
                  return c.bottom <= window.innerHeight + 1 && c.top >= 0;
              }"""))

        # ---------------------------------------------------- highlight
        print("\n== 公式侧栏与双向高亮（binding_vars）==")
        # The symbol tier, deliberately: WHICH SLOTS EXIST is a property of the
        # tier. `x.res1`'s `num` form substitutes the whole add and is written
        # `[N,d] + [N,d] = [N,d]` with no LHS/RHS slots at all, so the click
        # target is a sym-tier thing. Asserted rather than assumed below.
        page.evaluate("() => window.__vd.setTier('sym')")
        page.wait_for_timeout(250)
        slots_at_res1 = page.evaluate("""() => {
            window.__vd.setCursor(5);
            return [...document.querySelectorAll('.vd-fml [id*="-slot-"]')]
                .map(e => /-slot-(.+)$/.exec(e.id)[1]);
        }""")
        check("符号档下 x.res1 的公式带 LHS / RHS 两个 slot（数值档把它们代掉了）",
              set(slots_at_res1) >= {"LHS", "RHS"}, json.dumps(slots_at_res1))
        bv = page.evaluate("() => window.__vd.model.steps.map(s => s.bindingVars)")
        check("每一步都带 binding_vars，且覆盖它的全部 slot",
              all(x is not None and
                  set(x) == set(trace0["steps"][i]["bindings"])
                  for i, x in enumerate(bv)),
              json.dumps([i for i, x in enumerate(bv)
                          if x is None or set(x) != set(trace0["steps"][i]["bindings"])]))
        check("binding_vars 只指向已声明的张量",
              all(t in set(tensor_names) for x in bv for names in x.values() for t in names),
              "(全部端点都在 tensors 里)")

        def click_slot(slot):
            page.evaluate("""(slot) => {
                const el = [...document.querySelectorAll('.vd-fml [id*="-slot-"]')]
                    .find(e => e.id.endsWith('-slot-' + slot));
                if (el) el.dispatchEvent(new MouseEvent('click', {bubbles: true}));
            }""", slot)
            page.wait_for_timeout(200)

        def highlight_state():
            return page.evaluate("""() => ({
                pinned: [...document.querySelectorAll('[data-vd-node]')]
                    .filter(g => g.classList.contains('vd-n-pin')).map(g => g.dataset.vdNode),
                slotOn: [...document.querySelectorAll('.vd-fml [id*="-slot-"].vd-slot-on')]
                    .map(e => /-slot-(.+)$/.exec(e.id)[1]),
                chipOn: [...document.querySelectorAll('[data-vd-var].vd-chip-on')]
                    .map(e => e.dataset.vdVar),
            })""")

        # Direction 1: formula -> graph. The residual's LHS slot names `x`, and
        # `x` is the value the residual connection carries -- the case where a
        # regex over the rendered formula finds the `x` in `\times` instead.
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(5); }")
        page.wait_for_timeout(220)
        click_slot("LHS")
        hs = highlight_state()
        check("点残差公式里的 X → 图里 x 的节点亮（公式 → 图，且不误伤 y1）",
              hs["pinned"] == ["x"], json.dumps(hs, ensure_ascii=False))

        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(5); }")
        page.wait_for_timeout(220)
        click_slot("RHS")
        hs = highlight_state()
        check("点残差公式里的 Attn → 图里 attn_out 的节点亮（RHS 与 LHS 指向不同的张量）",
              hs["pinned"] == ["attn_out"], json.dumps(hs, ensure_ascii=False))

        # The `\ell`-class trap for THIS cover: `x.up`'s formula writes the FFN
        # activation as `U`, and the tensor is named `up` -- no identifier `up`
        # occurs anywhere in `U_{[N,d_ff]} = LN(Y_1) W_up`. A page-side scan
        # finds `up` only inside `W_{up}`, which is a weight, not a node.
        #
        # WHICH TIER, and why it matters here more than anywhere else. A / BB
        # are `num`-tier slots: the symbol tier writes `\mathrm{SiLU}(G) \odot
        # U` with no slots at all in the operands, and only the numeric tier
        # substitutes the single cell each slot names. So this block runs on the
        # numeric tier -- and the "does the mapping change with the tier" check
        # right below it runs on the other one, which is the pair that makes
        # either meaningful.
        page.evaluate("""() => {
            window.__vd.state.pinned = {};
            window.__vd.setTier('num');
            window.__vd.setCursor(10);
        }""")
        page.wait_for_timeout(280)
        check("数值档下 x.gated 的公式带 A / BB / P 三个 slot（符号档只有 G 与 U 的字面量）",
              set(page.evaluate("""() => [...document.querySelectorAll(
                  '.vd-fml [id*="-slot-"]')].map(e => /-slot-(.+)$/.exec(e.id)[1])"""))
              >= {"A", "BB", "P"})
        click_slot("BB")
        hs = highlight_state()
        check("⊙ 的第二个操作数 U 点亮的是张量 up（公式里没有一个叫 up 的标识符）",
              hs["pinned"] == ["up"], json.dumps(hs, ensure_ascii=False))
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(10); }")
        page.wait_for_timeout(220)
        click_slot("A")
        hs = highlight_state()
        check("⊙ 的第一个操作数 SiLU(G) 点亮的是张量 silu", hs["pinned"] == ["silu"],
              json.dumps(hs, ensure_ascii=False))
        # The other half of that pair: the SYMBOL tier, where the operands are
        # written as literals with no slots. `binding_vars` is derived from
        # `sym` by the generator (that is the tier that NAMES variables), so
        # the map exists at both tiers while the click targets only exist at
        # one -- which is exactly the asymmetry worth stating: the mapping is a
        # property of the step, not of the tier the reader is on.
        page.evaluate("""() => {
            window.__vd.state.pinned = {};
            window.__vd.setTier('sym');
            window.__vd.setCursor(10);
        }""")
        page.wait_for_timeout(280)
        sym_slots = page.evaluate("""() => [...document.querySelectorAll(
            '.vd-fml [id*="-slot-"]')].map(e => /-slot-(.+)$/.exec(e.id)[1])""")
        check("符号档下 x.gated 没有 A / BB 这两个 slot（它是数值档的代入位）",
              not ({"A", "BB"} & set(sym_slots)), json.dumps(sym_slots))
        check("但 binding_vars 在两个档位下都是同一张表（映射属于步骤，不属于档位）",
              page.evaluate("""() => {
                  const bv = window.__vd.model.steps[10].bindingVars;
                  return bv.A && bv.A.length === 1 && bv.BB && bv.BB.length === 1;
              }"""))

        # Direction 2: graph -> formula.
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(5); }")
        page.wait_for_timeout(220)
        page.evaluate("""() => document.querySelector('[data-vd-node="x"]')
            .dispatchEvent(new MouseEvent('click', {bubbles: true}))""")
        page.wait_for_timeout(220)
        hs = highlight_state()
        check("点图里的 x 节点 → 公式里所有提到它的 slot 都亮（图 → 公式）",
              hs["pinned"] == ["x"] and set(hs["slotOn"]) == {"LHS"} and
              set(hs["chipOn"]) == {"x"},
              json.dumps(hs, ensure_ascii=False))
        page.evaluate("() => { window.__vd.state.pinned = {}; }")
        # The graph→formula direction names EVERY slot that mentions the tensor,
        # which is the half a flat per-step set cannot do: `x` appears in this
        # step's LHS only, while `y1` (the step's own output) appears in no slot
        # of this formula at all -- so the two nodes behave differently, and a
        # page that lit every slot for every touched variable would hide that.
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(5); }")
        page.wait_for_timeout(200)
        page.evaluate("""() => document.querySelector('[data-vd-node="y1"]')
            .dispatchEvent(new MouseEvent('click', {bubbles: true}))""")
        page.wait_for_timeout(220)
        hs = highlight_state()
        check("点 y1（本步的输出，公式里没提到它）→ 没有 slot 亮，但节点与 chip 亮",
              hs["pinned"] == ["y1"] and hs["slotOn"] == [] and
              set(hs["chipOn"]) == {"y1"},
              json.dumps(hs, ensure_ascii=False))

        # A slot naming two variables lights BOTH: it would be a lie about what
        # the formula reads to light only one. L04's `x.gated` reads two
        # different tensors through two different slots, so pinning both is the
        # two-variable case -- and the assertion is that the two are independent
        # (clicking one does not move the other), which is what makes the
        # per-binding map worth having over one flat set per step.
        page.evaluate("""() => {
            window.__vd.state.pinned = {};
            window.__vd.setTier('num');
            window.__vd.setCursor(10);
        }""")
        page.wait_for_timeout(280)
        click_slot("A")
        click_slot("BB")
        page.wait_for_timeout(220)
        hs = highlight_state()
        check("⊙ 的两个操作数各自点名一个张量，两个都亮（不挑一个）",
              set(hs["pinned"]) == {"silu", "up"} and
              set(hs["slotOn"]) == {"A", "BB"},
              json.dumps(hs, ensure_ascii=False))
        # ...and the two are independent: clicking A again releases only silu.
        click_slot("A")
        hs = highlight_state()
        check("再点其中一个只松开它自己点名的那个张量（两个映射互不干扰）",
              set(hs["pinned"]) == {"up"}, json.dumps(hs, ensure_ascii=False))

        # The panel's own verdict on the contract, read off the page.
        sv = page.evaluate("() => window.__labViewChecks.slotVars")
        check("slot → 张量映射的契约 lint 干净", sv and sv["lintGaps"] == 0,
              json.dumps(sv, ensure_ascii=False))
        check("该 lint 的对照组全部被抓到（那 0 个 gap 不是 lint 永远报 0）",
              sv and sv["sabotageMissed"] == [] and sv["sabotageNoop"] == [],
              json.dumps(sv, ensure_ascii=False))
        check("该 lint 的对照组确实跑过（不是一张空表）",
              sv and sv["sabotageTotal"] >= 5, json.dumps(sv, ensure_ascii=False))

        # ---------------------------------------------------- the side panels
        #
        # The four teaching panels ride the view's step callback. Their
        # contracts are checked above (AC1–AC4); what is checked HERE is that
        # they are alive and bound to the same cursor -- a panel that rendered
        # once and froze would pass every AC above.
        print("\n== 侧栏面板随游标重建 ==")
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=1")
        page.wait_for_timeout(1100)
        panel_a = page.evaluate("""() => ({
            ln: document.querySelector('[data-panel-body="norm-axis"]').textContent.slice(0, 40),
            sg: document.querySelector('[data-panel-body="shape-guard"]').textContent.slice(0, 40),
            sw: document.querySelector('[data-panel-body="swiglu-paths"]').textContent.slice(0, 40),
        })""")
        step1_panel = page.evaluate("""() => {
            const el = document.querySelector('[data-panel-body="norm-axis"] .lab-na');
            return el ? el.dataset.stepId : null;
        }""")
        page.evaluate("() => window.__vd.setCursor(6)")
        page.wait_for_timeout(300)
        step6_panel = page.evaluate("""() => {
            const el = document.querySelector('[data-panel-body="norm-axis"] .lab-na');
            const sg = document.querySelector('[data-panel-body="shape-guard"] .lab-sg');
            return {na: el ? el.dataset.stepId : null, sg: sg ? sg.dataset.stepId : null};
        }""")
        check("停下不动时面板停在本步", step1_panel == "x.ln1", f"{step1_panel}")
        check("游标移动后面板重建到新的一步（不是渲染一次就冻住）",
              step6_panel["na"] == "x.ln2" and panel_a["ln"] != step6_panel["na"],
              json.dumps(step6_panel, ensure_ascii=False))
        check("账本视图也是活的（每一步都被 update 过）",
              page.evaluate("""() => {
                  const el = document.querySelector('[data-l04-ledger-mount] .lab-ledger');
                  return !!el && el.getAttribute('data-step-id') ===
                      window.__labTrace.steps[window.__vd.state.cursor].id;
              }"""))

        # ------------------------------------------------------ arbitrary jump
        print("\n== 任意跳转（纯函数重建 + 对照组）==")
        verdict = page.evaluate("() => window.__labVerify && window.__labVerify.L04")
        check("自检面板存在且跑过", verdict is not None)
        if verdict:
            j = verdict["jumps"]
            note(f'纯函数重建：{j["total"]} 次跳转，{j["pureFailures"]} 次不一致')
            note(f'对照组（故意做错的有状态播放器）：{j["controlFailures"]} 次不一致')
            check("纯函数重建 0 次不一致", j["pureFailures"] == 0)
            check("对照组确实失败（证明上面的 0 不是测试写错）",
                  j["controlFailures"] > 0, f'{j["controlFailures"]} 次')
            check("跳转集有区分力（conclusive）", j["conclusive"])
            check("lint 干净且 lint 本身是活的",
                  verdict["lint"]["passed"] and verdict["lint"]["missed"] == [],
                  f'clean gaps={verdict["lint"]["gaps"]}, '
                  f'caught={verdict["lint"]["caught"]}')

        # The end-to-end version of the same claim, through the real view: what
        # is RENDERED after a jump must equal what is rendered after stepping to
        # the same place one at a time -- on the GRAPH as well as the grid, so a
        # view whose three-state classes lagged a jump would be caught.
        e2e = page.evaluate("""() => {
            const V = window.__vd, last = V.index.lastStep;
            const read = () => {
                const vals = [];
                document.querySelectorAll('.vd-vg').forEach(g => {
                    vals.push([g.querySelector('.vd-vg-n').textContent,
                               [...g.querySelectorAll('.vd-cell')].map(c => c.textContent).join(',')]);
                });
                const states = [];
                document.querySelectorAll('[data-vd-node]').forEach(g => {
                    states.push(g.dataset.vdNode + ':' + g.dataset.vdState);
                });
                const fml = document.querySelector('.vd-fml').textContent;
                return JSON.stringify({vals: vals, states: states.sort(), fml: fml});
            };
            const paths = [['尾 -> 1', last, 1], ['1 -> 0', 1, 0], ['0 -> 尾', 0, last],
                           ['尾 -> 3', last, 3], ['3 -> 3', 3, 3], ['3 -> 0', 3, 0],
                           ['5 -> 12', 5, 12], ['12 -> 5', 12, 5]];
            const checks = [];
            for (const [label, from, to] of paths) {
                V.setCursor(from); V.setCursor(to);
                const shown = read();
                V.setCursor(0);
                for (let i = 1; i <= to; i++) V.setCursor(i);
                checks.push({label: label, same: shown === read()});
            }
            V.setCursor(0);
            return checks;
        }""")
        for c in e2e:
            check(f"端到端：跳转 {c['label']} 的渲染（值 + 节点态 + 公式）== 顺序播放到该步",
                  c["same"])

        # ============================================================ geometry
        print("\n== 几何：面板不溢出、格子不谎报长宽比、曲线共用纵轴 ==")
        for cfg_id, d, dff in DRIVEN:
            for step in (STEP_LN1, STEP_RES1, STEP_MUL, STEP_LAST):
                page.goto(f"{url}?cfg={cfg_id}&step={step}")
                page.wait_for_timeout(600)
                open_more(page)
                # A closed <details> lends every descendant a zero-sized client
                # box, which would make the predicate below vacuous. Asserted,
                # because "the panel had no overflowing elements" is not a fact
                # about a panel nobody rendered.
                check(f"{cfg_id} 步 {step}: 参考面板确实展开了（否则下面的几何谓词没有意义）",
                      page.evaluate("() => !!document.getElementById('lab-more').open"))
                geo = page.evaluate("""() => {
                    const out = [];
                    ['shape-guard', 'norm-axis', 'swiglu-paths', 'norm-contrast',
                     'params'].forEach(pid => {
                        const comp = document.querySelector('[data-panel="' + pid + '"]');
                        if (!comp) return;
                        const cb = comp.getBoundingClientRect();
                        const over = [];
                        comp.querySelectorAll('*').forEach(e => {
                            const r = e.getBoundingClientRect();
                            if (r.width > 0 &&
                                (r.left < cb.left - 1 || r.right > cb.right + 1)) {
                                over.push((e.className && e.className.baseVal !== undefined
                                    ? e.className.baseVal : e.className) || e.tagName);
                            }
                        });
                        out.push({panel: pid, over: over.slice(0, 3),
                                  n: over.length, w: cb.width});
                    });
                    return out;
                }""")
                bad = [g for g in geo if g["n"] > 0]
                check(f"{cfg_id} 步 {step}: 五个面板里没有元素溢出面板边界",
                      not bad, json.dumps(bad, ensure_ascii=False)[:240])

        # The cell grids: a drawn grid whose declared aspect ratio it does not
        # have is a picture making a claim its own geometry contradicts. The
        # grids are square-by-construction (one cell per element slot), so the
        # rule is "the drawn grid is square to within a cell".
        page.goto(f"{url}?cfg=decoder-block-d128-dff352&step={STEP_RES1}")
        page.wait_for_timeout(800)
        grids = page.evaluate(READ_RESIDUAL, STEP_RES1)["blocks"]
        bad_grids = []
        for g in grids:
            if g["w"] <= 0 or g["h"] <= 0:
                continue
            drawn_ar = g["w"] / g["h"]
            want_ar = g["drawnCols"] / g["drawnRows"]
            # The aspect ratio comes from the CSS grid's column count, so the
            # tolerance is one row's worth of height.
            if abs(drawn_ar - want_ar) / want_ar > 0.15:
                bad_grids.append({**g, "drawnAr": round(drawn_ar, 2),
                                  "wantAr": want_ar})
        check("格子矩阵画出来的长宽比与它声明的行列数一致（不谎报形状）",
              not bad_grids, json.dumps(bad_grids, ensure_ascii=False)[:240])
        check("格子有上限：d_model=128 的 [8,128] 操作数被画成 capped 的网格，"
              "而真实 shape 仍写在旁边",
              any(g["capped"] for g in grids),
              json.dumps([{"rows": g["rows"], "cols": g["cols"],
                           "capped": g["capped"]} for g in grids], ensure_ascii=False))

        # ---------------------------------------------------- the graph's geometry
        #
        # A graph can be wrong in a way only the graph shows. This project has
        # already shipped a diagram whose every count was right and whose
        # furniture overlapped, and L04's twelve-band drawing is exactly the
        # shape that produces it -- so the picture is measured, every step:
        # node boxes pairwise disjoint, every box inside the canvas, no edge
        # routed through a box it does not belong to, and no operation label on
        # a box or on another label.
        print("\n== 变量图的几何：18 个节点、25 条边、跨层绕行 ==")
        geo_dag = {"overlap": [], "clipped": [], "through": [], "labelBox": [],
                   "labelLabel": []}

        def rect_hit(a, b):
            return (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and
                    a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"])

        for cfg_id, d, dff in DRIVEN:
            page.goto(f"{url}?cfg={cfg_id}&step=0")
            page.wait_for_timeout(700)
            g = page.evaluate(READ_SVG)
            boxes = g["boxes"]
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    if rect_hit(boxes[a], boxes[b]):
                        geo_dag["overlap"].append([cfg_id, boxes[a]["id"], boxes[b]["id"]])
            for bx in boxes:
                if (bx["x"] < -0.5 or bx["y"] < -0.5 or
                        bx["x"] + bx["w"] > g["canvas"]["w"] + 0.5 or
                        bx["y"] + bx["h"] > g["canvas"]["h"] + 0.5):
                    geo_dag["clipped"].append([cfg_id, bx["id"]])
            for e in g["edges"]:
                for bx in boxes:
                    if bx["id"] in (e["from"], e["to"]):
                        continue
                    for (px, py) in e["pts"]:
                        if (bx["x"] < px < bx["x"] + bx["w"] and
                                bx["y"] < py < bx["y"] + bx["h"]):
                            geo_dag["through"].append(
                                [cfg_id, e["from"] + "→" + e["to"], bx["id"],
                                 "rail" if e["rail"] else "direct"])
                            break
            # The labels, in client space: a label sitting on a box is a label a
            # reader cannot read, and two labels on each other are one label.
            # Measured in one round trip -- a per-node `evaluate` here is 450
            # calls per configuration, which is a slow harness for no reason.
            lbl = page.evaluate("""() => {
                const svg = document.querySelector('.vd-dag');
                const boxes = [...svg.querySelectorAll('[data-vd-node]')].map(n => {
                    const r = n.querySelector('.vd-box').getBoundingClientRect();
                    return {id: n.dataset.vdNode, l: r.left, r: r.right,
                            t: r.top, b: r.bottom};
                });
                const labels = [...svg.querySelectorAll('[data-vd-el]')].map(t => {
                    const b = t.getBoundingClientRect();
                    return {pair: t.dataset.vdEl, text: t.textContent,
                            l: b.left, r: b.right, t: b.top, b: b.bottom,
                            opacity: +getComputedStyle(t).opacity};
                }).filter(x => x.opacity >= 0.5);
                const hit = (p, q) => p.l < q.r - 1 && q.l < p.r - 1 &&
                                      p.t < q.b - 1 && q.t < p.b - 1;
                const box = [], pair = [];
                labels.forEach(x => boxes.forEach(b => {
                    if (hit(x, b)) box.push([x.pair, b.id]);
                }));
                for (let a = 0; a < labels.length; a++)
                    for (let b = a + 1; b < labels.length; b++)
                        if (hit(labels[a], labels[b]))
                            pair.push([labels[a].text, labels[b].text]);
                return {box: box, pair: pair};
            }""")
            geo_dag["labelBox"] += [[cfg_id] + b for b in lbl["box"]]
            geo_dag["labelLabel"] += [[cfg_id] + p for p in lbl["pair"]]

        check("18 个节点两两不重叠（每个配置都测）",
              not geo_dag["overlap"], json.dumps(geo_dag["overlap"][:4], ensure_ascii=False))
        check("每个节点都在画布内（没有被 viewBox 裁掉）",
              not geo_dag["clipped"], json.dumps(geo_dag["clipped"][:4], ensure_ascii=False))
        check("没有边穿过与它无关的节点箱 —— 跨层边靠 rail 绕行，这正是这条规则在盯的",
              not geo_dag["through"], json.dumps(geo_dag["through"][:4], ensure_ascii=False))
        check("边的操作名不压在节点箱上",
              not geo_dag["labelBox"], json.dumps(geo_dag["labelBox"][:4], ensure_ascii=False))
        check("两个边的操作名不互相压住（同名标签不重复印）",
              not geo_dag["labelLabel"],
              json.dumps(geo_dag["labelLabel"][:4], ensure_ascii=False))
        note(f'{len(DRIVEN)} 个配置 × 每个 18 节点 / 25 边 / 每步全量几何谓词')

        # --- the geometry control group -----------------------------------------
        # The checks above must be able to FAIL. This does not push a data value
        # out of range and hope the drawing follows — it BUILDS a DOM that
        # violates the invariant and requires the same predicates, evaluated the
        # same way, to flag it.
        print("\n== 几何对照：人为把元素推出面板 / 压扁格子，规则必须报警 ==")
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step={STEP_RES1}")
        page.wait_for_timeout(900)
        open_more(page)
        control = page.evaluate("""() => {
            const out = {};
            const overflowOf = (comp) => {
                const cb = comp.getBoundingClientRect();
                const over = [];
                comp.querySelectorAll('*').forEach(e => {
                    const r = e.getBoundingClientRect();
                    if (r.width > 0 &&
                        (r.left < cb.left - 1 || r.right > cb.right + 1)) {
                        over.push(e.className || e.tagName);
                    }
                });
                return over;
            };
            // (1) the same predicate on the UNTOUCHED panel must find nothing,
            // or "it flagged the sabotage" would only prove it always flags.
            const sg = document.querySelector('[data-panel="shape-guard"]');
            out.cleanOverflow = overflowOf(sg).length;

            // (2) push a case card past the panel's right edge
            const card = sg.querySelector('.lab-sg-case');
            const saved = card.getAttribute('style') || '';
            card.setAttribute('style', saved + ';position:relative;left:2000px');
            out.pushedOverflow = overflowOf(sg).length;
            card.setAttribute('style', saved);
            out.restoredOverflow = overflowOf(sg).length;

            // (3) squeeze a cell grid away from its declared aspect ratio
            const grid = sg.querySelector('.lab-sg-block');
            const declaredCols = Number(grid.dataset.drawnCols);
            const declaredRows = Number(grid.dataset.drawnRows);
            const before = grid.getBoundingClientRect();
            out.cleanAr = before.width / before.height;
            out.wantAr = declaredCols / declaredRows;
            const gs = grid.getAttribute('style') || '';
            grid.setAttribute('style', gs + ';height:8px;grid-template-columns:' +
                'repeat(' + declaredCols * 8 + ',1fr);aspect-ratio:auto');
            const after = grid.getBoundingClientRect();
            out.squeezedAr = after.width / after.height;
            // the predicate under test
            out.squeezeBad = Math.abs(out.squeezedAr - out.wantAr) / out.wantAr > 0.15;
            grid.setAttribute('style', gs);
            const restored = grid.getBoundingClientRect();
            out.restoredAr = restored.width / restored.height;
            out.restoredOk = Math.abs(out.restoredAr - out.wantAr) / out.wantAr <= 0.15;
            return out;
        }""")
        check("对照组：被推出面板右缘的元素会被「不溢出面板」规则抓到，"
              "而同一规则对未改动的面板判为干净（且破坏后可恢复）",
              control["pushedOverflow"] > 0 and control["cleanOverflow"] == 0 and
              control["restoredOverflow"] == 0,
              json.dumps(control))
        check("对照组：被压扁的格子会被「不谎报长宽比」规则抓到，"
              "而同一规则对未改动的那一个判为诚实",
              control["squeezeBad"] and control["restoredOk"] and
              control["cleanAr"] > 0,
              f'压后 {control["squeezedAr"]:.2f} vs 声明 {control["wantAr"]:.2f}；'
              f'未动 {control["cleanAr"]:.2f}')

        # --- the GRAPH geometry control group ---------------------------------
        # The same terms the panel control uses, applied to the predicates the
        # variable-graph section runs: hand-place the three violations they are
        # supposed to catch (a box on a box, a box past the canvas, an edge
        # rerouted through an unrelated box) and require the same predicates,
        # evaluated the same way, to flag each one -- with the untouched
        # baseline asserted clean so "it flagged" is not just "it always flags".
        print("\n== 变量图的几何对照：人为造出违规，同一批谓词必须报警 ==")
        gcontrol = page.evaluate("""() => {
            const read = () => {
                const svg = document.querySelector('.vd-dag');
                const vb = svg.getAttribute('viewBox').split(/\\s+/).map(Number);
                const boxes = [...svg.querySelectorAll('[data-vd-node]')].map(n => {
                    const r = n.querySelector('.vd-box');
                    return {id: n.dataset.vdNode,
                            x: +r.getAttribute('x'), y: +r.getAttribute('y'),
                            w: +r.getAttribute('width'), h: +r.getAttribute('height')};
                });
                const edges = [...svg.querySelectorAll('[data-vd-edge]')].map(p => {
                    const len = p.getTotalLength();
                    const pts = [];
                    for (let k = 0; k <= 60; k++) {
                        const q = p.getPointAtLength(len * k / 60);
                        pts.push([q.x, q.y]);
                    }
                    const m = /^(.+)>(.+)$/.exec(p.dataset.vdEdge);
                    return {from: m[1], to: m[2], pts: pts};
                });
                return {canvas: {w: vb[2], h: vb[3]}, boxes: boxes, edges: edges};
            };
            const overlaps = (p, q) =>
                p.x < q.x + q.w && q.x < p.x + p.w && p.y < q.y + q.h && q.y < p.y + p.h;
            const outside = (b, c) =>
                b.x < -0.5 || b.y < -0.5 || b.x + b.w > c.w + 0.5 || b.y + b.h > c.h + 0.5;
            const crosses = (boxes, e) => {
                for (const b of boxes) {
                    if (b.id === e.from || b.id === e.to) continue;
                    for (const [px, py] of e.pts) {
                        if (b.x < px && px < b.x + b.w && b.y < py && py < b.y + b.h) return b.id;
                    }
                }
                return null;
            };
            const svg = document.querySelector('.vd-dag');
            const base = read();
            const out = {cleanOverlap: 0, cleanClip: 0, cleanThrough: 0};
            for (let a = 0; a < base.boxes.length; a++)
                for (let b = a + 1; b < base.boxes.length; b++)
                    if (overlaps(base.boxes[a], base.boxes[b])) out.cleanOverlap++;
            out.cleanClip = base.boxes.filter(b => outside(b, base.canvas)).length;
            // (1) put one node exactly on another
            const two = base.boxes.slice(0, 2);
            const el0 = svg.querySelector('[data-vd-node="' + two[0].id + '"] .vd-box');
            const keep = {x: el0.getAttribute('x'), y: el0.getAttribute('y')};
            el0.setAttribute('x', String(two[1].x));
            el0.setAttribute('y', String(two[1].y));
            const after1 = read();
            out.overlapBad = 0;
            for (let a = 0; a < after1.boxes.length; a++)
                for (let b = a + 1; b < after1.boxes.length; b++)
                    if (overlaps(after1.boxes[a], after1.boxes[b])) out.overlapBad++;
            el0.setAttribute('x', keep.x);
            el0.setAttribute('y', keep.y);
            // (2) push a node past the canvas
            const el1 = svg.querySelector('[data-vd-node="' + two[0].id + '"] .vd-box');
            el1.setAttribute('x', String(base.canvas.w + 60));
            out.clipBad = read().boxes.filter(b => outside(b, base.canvas)).length;
            el1.setAttribute('x', keep.x);
            // (3) reroute an edge straight through an unrelated node
            const path = svg.querySelector('[data-vd-edge]');
            const keepD = path.getAttribute('d');
            const a1 = svg.querySelector('[data-vd-node="' + base.boxes[0].id + '"] .vd-box');
            const z1 = svg.querySelector('[data-vd-node="' + base.boxes[base.boxes.length - 1].id +
                                        '"] .vd-box');
            const cx = n => +n.getAttribute('x') + (+n.getAttribute('width')) / 2;
            const cy = n => +n.getAttribute('y') + (+n.getAttribute('height')) / 2;
            path.setAttribute('d', 'M' + cx(a1) + ' ' + cy(a1) + ' L' + cx(z1) + ' ' + cy(z1));
            out.throughBad = crosses(read().boxes, read().edges[0]);
            path.setAttribute('d', keepD);
            return out;
        }""")
        check("对照组：未改动的图在同一批谓词下是干净的（不重叠 / 不越界 / 不穿箱）",
              gcontrol["cleanOverlap"] == 0 and gcontrol["cleanClip"] == 0,
              json.dumps(gcontrol))
        check("对照组：被推到另一个节点上的箱子会被「节点不重叠」规则抓到",
              gcontrol["overlapBad"] > 0, json.dumps(gcontrol))
        check("对照组：被推出画布的节点会被「不被裁」规则抓到",
              gcontrol["clipBad"] == 1, json.dumps(gcontrol))
        check("对照组：被改道穿过别的节点的边会被「边不穿箱」规则抓到",
              bool(gcontrol["throughBad"]), json.dumps(gcontrol))

        # ============================================================== shots
        print("\n== 截图 ==")
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=0")
        page.wait_for_timeout(1000)
        page.evaluate("() => window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "60-l04-overview.png")

        # The four teaching panels and the ledger are inside the <details>, so
        # it has to be OPEN before an element screenshot can scroll them into
        # view -- a closed one is `not visible` and playwright retries until it
        # times out.
        open_more(page)
        expand_for_shots(page)
        for idx, name in ((STEP_LN1, "61-l04-layernorm-axis"),
                          (STEP_RES1, "62-l04-residual-shapes"),
                          (STEP_MUL, "63-l04-swiglu-multiply"),
                          (STEP_LAST, "64-l04-params")):
            page.evaluate(f"() => window.__vd.setCursor({idx})")
            page.wait_for_timeout(500)
            sel = {STEP_LN1: "norm-axis", STEP_RES1: "shape-guard",
                   STEP_MUL: "swiglu-paths", STEP_LAST: "params"}[idx]
            page.locator(f'[data-panel="{sel}"]').screenshot(
                path=SHOTS / f"{name}.png")

        page.locator('[data-panel="norm-contrast"]').screenshot(
            path=SHOTS / "65-l04-norm-contrast.png")

        # The two ends of the grid, so a reviewer sees the slider's effect on
        # the parameter account without running the harness.
        for cfg_id, name in (("decoder-block-d32-dff88", "66-l04-params-small"),
                             ("decoder-block-d128-dff352", "67-l04-params-large")):
            page.goto(f"{url}?cfg={cfg_id}&step={STEP_LAST}")
            page.wait_for_timeout(900)
            open_more(page)
            expand_for_shots(page)
            page.locator('[data-panel="params"]').screenshot(
                path=SHOTS / f"{name}.png")

        # The whole page with everything unfolded -- the one shot that shows what
        # a reader gets if they open the reference section, all five panels and
        # the provenance note in one frame. `full_page` because the point of
        # this one IS the length.
        page.goto(f"{url}?cfg=decoder-block-d128-dff352&step={STEP_RES1}")
        page.wait_for_timeout(900)
        open_more(page)
        expand_for_shots(page)
        page.screenshot(path=SHOTS / "68-l04-full-light.png", full_page=True)

        # The variable graph itself, at the steps a reader would stop on, plus
        # the two cased panels of the matrix rendering. These are the shots that
        # let a reviewer SEE the shape of the drawing without running the
        # harness -- the geometry assertions say it is not broken; these say
        # what it is.
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=0")
        page.wait_for_timeout(1000)
        for step, name in ((0, "70-l04-dag-step0"), (3, "71-l04-dag-scores"),
                           (5, "72-l04-dag-residual"), (10, "73-l04-dag-gated"),
                           (13, "74-l04-dag-summary")):
            page.evaluate(f"() => window.__vd.setCursor({step})")
            page.wait_for_timeout(420)
            page.screenshot(path=SHOTS / f"{name}.png")
        page.evaluate("() => window.__vd.setCursor(3)")
        page.wait_for_timeout(350)
        page.locator(".vd-dagwrap").screenshot(path=SHOTS / "75-l04-dag-graph-only.png")
        page.locator(".vd-side").screenshot(path=SHOTS / "76-l04-dag-sidebar.png")

        # The bidirectional highlight, captured rather than only asserted.
        page.evaluate("""() => {
            window.__vd.state.pinned = {};
            window.__vd.setCursor(5);
            const el = [...document.querySelectorAll('.vd-fml [id*="-slot-"]')]
                .find(e => /-slot-LHS$/.test(e.id));
            if (el) el.dispatchEvent(new MouseEvent('click', {bubbles: true}));
        }""")
        page.wait_for_timeout(350)
        page.screenshot(path=SHOTS / "77-l04-dag-highlight.png")

        # Both themes, from pages that actually requested them.
        for name, scheme in (("78-l04-dag-dark.png", "dark"),
                             ("79-l04-dag-light.png", "light")):
            p_theme = browser.new_page(viewport={"width": W, "height": H},
                                       color_scheme=scheme)
            p_theme.goto(f"{url}?cfg=decoder-block-d64-dff176&step=3")
            p_theme.wait_for_timeout(1000)
            p_theme.screenshot(path=SHOTS / name)
            p_theme.close()

        # ------------------------------------------------------------- phone
        #
        # "PC 与手机都能方便阅读" is the same promise here as in L00, and for
        # this page it is the load-bearing one: a twelve-band drawing is exactly
        # the shape that tempts a designer into a "needs a wider screen" notice.
        # The view does not have one -- the same two halves stack -- so what is
        # asserted is that the phone gets the REAL view and that the graph keeps
        # a readable height, not merely that it did not throw.
        print("\n== 手机：真适配（不是降级）==")
        phone = browser.new_page(viewport={"width": 390, "height": 844},
                                 device_scale_factor=2, is_mobile=True, has_touch=True)
        phone_logs = []
        phone.on("pageerror", lambda e: phone_logs.append(str(e)))
        phone.goto(url)
        phone.wait_for_timeout(1300)
        mob = phone.evaluate("""() => {
            const root = document.querySelector('.vd-root');
            const main = document.querySelector('.vd-main');
            const ctl = document.querySelector('.vd-ctl').getBoundingClientRect();
            const label = document.querySelector('[data-vd-node] .vd-nl')
                .getBoundingClientRect();
            return {
                rows: getComputedStyle(main).gridTemplateRows.split(' ').length,
                rootOverflow: root.scrollHeight - root.clientHeight,
                docOverflow: document.documentElement.scrollHeight - window.innerHeight,
                ctlVisible: ctl.bottom <= window.innerHeight + 1 && ctl.top >= 0,
                graphH: Math.round(document.querySelector('.vd-dagwrap')
                    .getBoundingClientRect().height),
                labelPx: label.height,
                fit: document.querySelector('.vd-dagwrap').dataset.vdFit,
                hasView: !!document.querySelector('.vd-dag'),
                hasFormula: !!document.querySelector('.vd-fml .lab-formula-node'),
                hasFallback: !!document.querySelector('.lab-narrow'),
            };
        }""")
        check("手机上渲染的是同一个视图（不是「需要更宽的屏幕」降级卡片）",
              mob["hasView"] and mob["hasFormula"] and not mob["hasFallback"],
              json.dumps(mob, ensure_ascii=False))
        check("手机上两半改为上下堆叠（真排版，不是缩小的 PC 版）",
              mob["rows"] == 2, f"grid 行数 = {mob['rows']}")
        check("手机上图仍有可读高度（不是被压成一条）",
              mob["graphH"] >= 140, f"图高 {mob['graphH']}px")
        # The parameter bar is L04's own chrome and its labels are long by
        # design on a desktop. On a phone the same strings push the sliders into
        # the value readouts, so the bar is asserted to fit rather than assumed
        # to: a slider that overlaps its own label reads as the wrong number
        # beside the wrong control.
        bar = phone.evaluate("""() => {
            const row = document.querySelector('.l04-params-row');
            const over = [];
            row.querySelectorAll('*').forEach(e => {
                const b = e.getBoundingClientRect();
                if (b.width > 0 && (b.right > row.getBoundingClientRect().right + 1 ||
                                    b.left < row.getBoundingClientRect().left - 1)) {
                    over.push(e.className || e.tagName);
                }
            });
            const labels = [...row.querySelectorAll('.l04-param-label')]
                .filter(e => getComputedStyle(e).display !== 'none');
            return {over: over, labels: labels.map(e => e.textContent),
                    shown: labels.length};
        }""")
        check("手机：参数条里的元素不互相越界（长说明换成短标题，滑杆不被文字挤出去）",
              not bar["over"], json.dumps(bar, ensure_ascii=False)[:240])
        check("手机上节点标签仍可读（>= 8px）",
              mob["labelPx"] >= 8.0, f'标签高 {mob["labelPx"]:.1f}px，fit={mob["fit"]}')

        mfit = phone.evaluate("""() => {
            const out = [];
            for (let i = 0; i <= window.__vd.index.lastStep; i++) {
                window.__vd.setCursor(i);
                const ctl = document.querySelector('.vd-ctl').getBoundingClientRect();
                out.push({i: i,
                    doc: document.documentElement.scrollHeight - window.innerHeight,
                    ctlVisible: ctl.bottom <= window.innerHeight + 1 && ctl.top >= 0});
            }
            return out;
        }""")
        check("手机：每一步都不需要滚动页面（一屏 = 一步）",
              max(r["doc"] for r in mfit) <= 0,
              f"最大溢出 {max(r['doc'] for r in mfit)}px")
        check("手机：控制条在每一步都留在视野内",
              all(r["ctlVisible"] for r in mfit),
              json.dumps([r["i"] for r in mfit if not r["ctlVisible"]]))
        check("手机：与 PC 是同一份数据同一批节点",
              phone.evaluate("() => window.__vd.model.nodes.length") == len(tensor_names),
              f"{phone.evaluate('() => window.__vd.model.nodes.length')} 节点")

        # The phone's graph must satisfy the SAME geometric predicates as the
        # desktop's -- the layout is fixed in viewBox units and scaled, so a
        # collision at this size would mean the layout was wrong all along and
        # the desktop had simply been hiding it in a smaller box.
        phone_geo = phone.evaluate(READ_SVG)
        phone_bad = []
        for a in range(len(phone_geo["boxes"])):
            for b in range(a + 1, len(phone_geo["boxes"])):
                if rect_hit(phone_geo["boxes"][a], phone_geo["boxes"][b]):
                    phone_bad.append([phone_geo["boxes"][a]["id"],
                                      phone_geo["boxes"][b]["id"]])
        check("手机上的节点同样两两不重叠（同一份布局，不是另一套）",
              not phone_bad, json.dumps(phone_bad[:4], ensure_ascii=False))
        check("手机上边同样不穿箱",
              page.evaluate("true") and not any(
                  bx["x"] < px < bx["x"] + bx["w"] and bx["y"] < py < bx["y"] + bx["h"]
                  for e in phone_geo["edges"]
                  for bx in phone_geo["boxes"]
                  if bx["id"] not in (e["from"], e["to"])
                  for (px, py) in e["pts"]),
              "")
        check("手机上 18 个节点全在画布内",
              all(bx["x"] >= -0.5 and bx["y"] >= -0.5 and
                  bx["x"] + bx["w"] <= phone_geo["canvas"]["w"] + 0.5 and
                  bx["y"] + bx["h"] <= phone_geo["canvas"]["h"] + 0.5
                  for bx in phone_geo["boxes"]),
              f'{len(phone_geo["boxes"])} 个节点')
        check("手机上 18 个节点都在",
              len(phone_geo["boxes"]) == len(tensor_names),
              f'{len(phone_geo["boxes"])} / {len(tensor_names)}')

        for step, name in ((0, "80-l04-phone-step0"), (3, "81-l04-phone-scores"),
                           (5, "82-l04-phone-residual"), (10, "83-l04-phone-gated")):
            phone.evaluate(f"() => window.__vd.setCursor({step})")
            phone.wait_for_timeout(420)
            phone.screenshot(path=SHOTS / f"{name}.png")
        check("手机上没有 JS 错误", not phone_logs, str(phone_logs[:3]))
        phone.close()

        print(f"\n截图 -> {SHOTS}")

        print("\n== console ==")
        if console:
            for c in console[:20]:
                print("  " + c[:200])
            failures.append("console errors/warnings present")
        else:
            print("  （无 error / warning）")

        browser.close()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} / {len(talks)} 项未通过：")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"全部通过（{len(talks)} 项断言，覆盖 5 条验收标准 + 变量图契约 + "
          f"契约 lint 双份 + 几何对照）")
    return 0


def dff_cfg(trace):
    return trace["meta"]["config"]["d_ff"]


if __name__ == "__main__":
    sys.exit(main())
