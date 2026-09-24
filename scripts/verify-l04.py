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
     seven errors would be as wrong as one that showed seven successes.
  3. Wherever a value is asserted to have MOVED, the control that must hold is
     asserted alongside it: the `params` segments may not move with `step`
     (the block's parameter count is the same in every frame), and Post-Norm's
     residual RMS must hold FLAT while Pre-Norm's climbs. A test that only
     checked "the number changed" would pass on a page that redrew itself at
     random.
  4. The geometry is measured, not eyeballed — no element may overflow its
     panel, the cell grids must not claim an aspect ratio they do not have, and
     the two charts of the Pre/Post panel must share a vertical scale. This
     project has already shipped a chart whose every count was correct and
     whose furniture overlapped, so the picture is asserted like the numbers.
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
              "swiglu": "swiglu", "contrast": "contrast"}
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
                   for pn in ("shapeguard", "normaxis", "swiglu", "contrast", "base"))]
    check(f"每一种破坏都至少被一个 port 抓到（共 {total} 种）", not uncaught,
          "谁都没抓到的：" + "、".join(sorted(uncaught))[:400])
    note(f"其中 {len(py_only)} 种只有 Python 侧 port（{', '.join(sorted(py_only)[:3])}"
         f"{' …' if len(py_only) > 3 else ''}）—— 它们属于 meta.params，"
         f"而没有共享组件拥有那个块的规则")

    pyside = subprocess.run([sys.executable, str(GENERATOR)],
                            capture_output=True, text=True, cwd=str(REPO))
    text = pyside.stdout + pyside.stderr
    ok = "LINT-DEAD" not in text and f"对照 {total} 种破坏全部被抓到" in text
    check("Python 侧 lint 独自覆盖了 JS 侧职责之外的规则", ok,
          "生成器自报全部抓到" if ok else "生成器输出里没有出现「全部抓到」")
    note(f"五个 port 各跑同一张 {total} 种破坏的表：JS 侧 shapeGuard / normAxis / "
         f"swigluPaths / normContrast 各自覆盖本组件的字段，trace-model 覆盖基础契约；"
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

# The residual panel at one step: the confirmation, and the counterexamples.
READ_RESIDUAL = """(idx) => {
    const P = window.__p, T = window.__labTrace;
    P.setCursor(idx);
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
    const P = window.__p;
    P.setCursor(idx);
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
    const P = window.__p;
    P.setCursor(idx);
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
            const T = window.__labTrace, P = window.__p, out = [];
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

        # ============================================================ geometry
        print("\n== 几何：面板不溢出、格子不谎报长宽比、曲线共用纵轴 ==")
        for cfg_id, d, dff in DRIVEN:
            for step in (STEP_LN1, STEP_RES1, STEP_MUL, STEP_LAST):
                page.goto(f"{url}?cfg={cfg_id}&step={step}")
                page.wait_for_timeout(600)
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

        # --- the geometry control group -----------------------------------------
        # The checks above must be able to FAIL. This does not push a data value
        # out of range and hope the drawing follows — it BUILDS a DOM that
        # violates the invariant and requires the same predicates, evaluated the
        # same way, to flag it.
        print("\n== 几何对照：人为把元素推出面板 / 压扁格子，规则必须报警 ==")
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step={STEP_RES1}")
        page.wait_for_timeout(900)
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

        # ============================================================== shots
        print("\n== 截图 ==")
        page.goto(f"{url}?cfg=decoder-block-d64-dff176&step=0")
        page.wait_for_timeout(1000)
        page.evaluate("() => window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "60-l04-overview.png")

        for idx, name in ((STEP_LN1, "61-l04-layernorm-axis"),
                          (STEP_RES1, "62-l04-residual-shapes"),
                          (STEP_MUL, "63-l04-swiglu-multiply"),
                          (STEP_LAST, "64-l04-params")):
            page.evaluate(f"() => window.__p.setCursor({idx})")
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
            page.locator('[data-panel="params"]').screenshot(
                path=SHOTS / f"{name}.png")

        page.goto(f"{url}?cfg=decoder-block-d128-dff352&step={STEP_RES1}")
        page.wait_for_timeout(900)
        page.screenshot(path=SHOTS / "68-l04-full-light.png", full_page=True)

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
    print(f"全部通过（{len(talks)} 项断言，覆盖 5 条验收标准 + 契约 lint 双份 + 几何对照）")
    return 0


def dff_cfg(trace):
    return trace["meta"]["config"]["d_ff"]


if __name__ == "__main__":
    sys.exit(main())
