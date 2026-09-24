#!/usr/bin/env python3
"""Acceptance harness for L05 · KV Cache and autoregressive decoding (issue #19).

Drives the built page in real Chromium and checks the ticket's five acceptance
criteria against the artifact a reader actually gets: the staged page in
`public/labs/`, with its trace SET inlined and its assets at the flattened
paths.

    AC1  逐 token 生成过程可回放，KV Cache 的增长在视图上可见
    AC2  有/无 KV Cache 两种模式下，attention 的输入 shape 对比一目了然
    AC3  显存账本是可复用的独立视图组件，而非写死在本页
    AC4  改变上下文长度/层数/精度后，账本各分项数值正确重算
    AC5  Prefill 与 Decode 在 Roofline 图上落点不同，且能解释原因

AC3 is `#45`'s deliverable and is asserted by `scripts/verify-ledger.py` against
the component's own acceptance page. What is asserted HERE is the consumer's
half of it: that this page mounts that component rather than re-implementing a
stacked bar, that the segments arrive from the trace rather than from a literal,
and that the same component renders a configuration the trace never ran.

WHAT MAKES THE ASSERTIONS MEANINGFUL
------------------------------------
Four things this harness will not do, each of which would have made a green
result prove nothing:

  1. It reads the numbers back OFF THE DOM and compares them to the trace the
     generator wrote — never to a re-computation. If the page printed a number
     no trace contains, that is a failure here. (The page is forbidden from
     computing anything except in the block that is explicitly labelled a
     prediction, and the self-check panel is what makes that block legitimate.)
  2. Wherever a value is asserted to have MOVED, the control that must hold is
     asserted alongside it: the parameters segment may NOT move with the
     context, the Roofline's reference point must be the same dot in every
     configuration, and the two attention modes must be IDENTICAL at prefill.
     A test that only checked "the number changed" would pass on a page that
     redrew itself at random.
  3. The geometry is measured, not eyeballed — the Roofline's dots must land
     inside the plot frame, no dot's label may sit on a neighbouring dot, no
     text may be clipped, and the shape comparison's rectangles must not
     overflow their panel and must not claim an aspect ratio they do not have.
     This project has already shipped a chart whose every count was correct and
     whose furniture overlapped, so the picture is asserted like the numbers.
  4. The same lint sabotage table is run through BOTH ports of each new
     contract block — the Python generator's and the JavaScript port beside the
     view — and every sabotage must be caught by the port that owns it, must be
     caught by at least one port, and must actually change the trace. A
     mutation that stopped biting (a no-op) proves nothing when both lints
     report zero, and it is a different bug from "the lint missed it".

Run:  python3 labs/traces/kv_cache.py          # writes the trace set
      npm run build:labs && python3 scripts/verify-l05.py
Needs `pip install playwright && playwright install chromium`.

Non-zero exit means at least one acceptance criterion failed.
"""

import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "public" / "labs" / "05-kv-cache.html"
TRACE_DIR = REPO / "labs" / "traces"
MANIFEST = TRACE_DIR / "kv-cache.manifest.json"
GENERATOR = TRACE_DIR / "kv_cache.py"
SHOTS = REPO / "labs" / "pages" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1200

# The configurations this harness drives end to end. Not all 18: each one is a
# page load plus a full DOM walk, and a 1MB trace. These five cover both ends of
# every slider and each single-parameter move in isolation, which is what makes
# "the number changed" attributable to the parameter that moved.
DRIVEN = [
    ("kv-cache", 8, 2, "fp16"),                   # the default, and the reference
    ("kv-cache-N128-L2-fp16", 128, 2, "fp16"),    # context ×16, everything else held
    ("kv-cache-N8-L4-fp16", 8, 4, "fp16"),        # layers ×2, everything else held
    ("kv-cache-N8-L2-fp8", 8, 2, "fp8"),          # precision halved, rest held
    ("kv-cache-N128-L4-int8", 128, 4, "int8"),    # every slider at its far end
]

failures = []
notes = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def note(msg):
    notes.append(msg)
    print(f"  · {msg}")


def load(name):
    return json.loads((TRACE_DIR / f"{name}.json").read_text(encoding="utf-8"))


# The page's formatting rule, evaluated IN THE BROWSER.
#
# The harness needs the string the page WOULD print for a trace's value, so it
# can compare it against the string the page DID print. Porting that rule to
# Python was the obvious move and it was wrong: the page does
# `v.toPrecision(4)` and strips trailing zeros, and `toPrecision` breaks ties
# away from zero while Python's `%g` breaks them to even on the exact binary
# value. At this trace's 4-layer crossover of 402.25 that is `402.3` versus
# `402.2` — the two renderers genuinely disagree, so a "faithful port" would be
# a third implementation to keep in step, and every future disagreement would
# read as a page bug.
#
# So the rule lives in one place: the engine the page runs on. `fmtExpected`
# asks the browser to format the trace's own values, and the comparison is then
# about the NUMBER (did the page print the trace's value?) rather than about
# string formatting, which is not what any of these assertions are testing.
# Playwright passes ONE argument, so the batch and the digit count travel
# together as an object rather than as two positional params.
FMT_JS = """(args) => args.values.map(v => {
    if (v === null || v === undefined) return '—';
    if (typeof v !== 'number') return String(v);
    const s = v.toPrecision(args.digits || 4);
    return s.replace(/(\\.\\d*?)0+$/, '$1').replace(/\\.$/, '');
})"""


# The page object, set by `main()` once the browser is up. Module-level so the
# formatting helper can reach it without threading a handle through every
# function that renders an expected string.
_PAGE = None


def fmt(values, digits=4):
    """Format trace values with the page's own rule, in the browser.

    `values` is a list; returns the list of strings the page would print. One
    call formats one batch, so a check that needs a dozen strings does not open
    a dozen round trips.
    """
    return _PAGE.evaluate(FMT_JS, {"values": values, "digits": digits})




def thousands(n):
    return f"{n:,}"


# ------------------------------------------------------------- lint parity
def lint_parity():
    """Run the generator's sabotage table through both ports of the new blocks.

    The Python lint runs inside the generator; this asks it for the payload
    (`--js-lint`) and pushes the same broken copies through the JS ports in
    node.

    The two ports are not meant to be identical rule-for-rule: the Python side
    lints the whole contract (formulas, bindings, graph, `state`, the ledger,
    the crossover), while each JS side lints the fields its view renders. So the
    requirement is not "every rule in both files" — it is:

      * every sabotage is caught by AT LEAST ONE port;
      * every sabotage in a view's own scope is caught by THAT view's JS port
        (the case names carry the owning view, so a new case that forgets its
        prefix shows up as a count mismatch rather than silently untested);
      * every sabotage actually changes the trace. A case written against a
        literal that stops biting — a no-op — proves nothing when both lints
        report zero, and it is a different bug from "the lint missed it".
    """
    print("\n== 契约 lint 的两份实现（attn_cost / phase_roofline / kv_crossover）==")
    proc = subprocess.run([sys.executable, str(GENERATOR), "--js-lint"],
                          capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        check("生成器的 --js-lint 模式可用", False, (proc.stderr or "")[-400:])
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
load('labs/assets/engine/views/roofline.js');
load('labs/assets/engine/views/ledger.js');
load('labs/assets/engine/views/attn-shape.js');
load('labs/assets/engine/views/phase-roofline.js');
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
// The three ports this harness drives, keyed by the prefix their sabotage
// cases carry. `trace-model.js` is also run, because a sabotage in the
// generator's table may target a base-contract field rather than one of the
// new blocks -- that port is the one that owns those rules.
const PORTS = {
  'attn': NS.attnShape,
  'roofline': NS.phaseRoofline,
  'ledger': NS.ledger,
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
  // The whole-contract port: any gap from any of the four is the Python side's
  // rule set being covered on this side too.
  r.any = Object.values(r).some(v => typeof v === 'number' && v > 0);
  r.changed = s.changed;
  out.sabotage[name] = r;
}
for (const [pn, port] of Object.entries(PORTS)) {
  out.ports[pn] = Object.keys(port.SABOTAGE_CASES || {}).length;
}
process.stdout.write(JSON.stringify(out));
"""
    scratch = Path("/tmp") / "l05-lint-probe.js"
    scratch.write_text(node_src, encoding="utf-8")
    payload_file = Path("/tmp") / "l05-lint-payload.json"
    payload_file.write_text(proc.stdout, encoding="utf-8")
    node = subprocess.run(["node", str(scratch), str(payload_file)],
                          capture_output=True, text=True, cwd=str(REPO))
    if node.returncode != 0:
        check("JS 侧 lint 能在 node 里跑起来", False, (node.stderr or "")[-400:])
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
    # The port names are the node script's registry keys — the sabotage
    # case names carry the same short prefix.
    for prefix, port_name in {"attn": "attn", "roofline": "roofline"}.items():
        cases = [k for k in res["sabotage"] if k.startswith(prefix + " ")]
        missed = [k for k in cases
                  if not (isinstance(res["sabotage"][k].get(port_name), int) and
                          res["sabotage"][k][port_name] > 0)]
        declared = res["ports"][port_name]
        check(f"{prefix} 视角的 {len(cases)} 种破坏全部被 JS 侧 lint 抓到"
              f"（该 port 声明了 {declared} 条规则）",
              bool(cases) and not missed, "漏掉：" + "、".join(sorted(missed)))

    # And the whole table must be caught by at least one of the ports: a case
    # neither side catches is a rule that exists only in the case's name.
    uncaught = [k for k, v in res["sabotage"].items() if not v.get("any")]
    check(f"每一种破坏都至少被四个 port 之一抓到（共 {total} 种）", not uncaught,
          "谁都没抓到的：" + "、".join(sorted(uncaught))[:300])

    pyside = subprocess.run([sys.executable, str(GENERATOR)],
                            capture_output=True, text=True, cwd=str(REPO))
    text = pyside.stdout + pyside.stderr
    ok = "LINT-DEAD" not in text and f"对照 {total} 种破坏全部被抓到" in text
    check("Python 侧 lint 独自覆盖了 JS 侧职责之外的规则", ok,
          "生成器自报全部抓到" if ok else "生成器输出里没有出现「全部抓到」")
    note(f"四个 port 各跑同一张 {total} 种破坏的表：JS 侧 attnShape / phaseRoofline "
         f"各自覆盖本组件的字段，ledger 覆盖账本，trace-model 覆盖基础契约；"
         f"每一侧都要求「破坏真的改变了 trace」")


# ------------------------------------------------------------- DOM readers
READ_METRICS = """() => {
    const out = {};
    document.querySelectorAll('#lab-readout [data-metric]').forEach(e => {
        const b = e.querySelector('b');
        out[e.dataset.metric] = b ? b.textContent.trim() : null;
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

# The replay's ledger, read frame by frame: one record per step with the three
# segment rows and the derived total.
READ_LEDGER_SERIES = """() => {
    const P = window.__p, T = window.__labTrace, out = [];
    for (let i = 0; i < T.steps.length; i++) {
        P.setCursor(i);
        const host = document.querySelector('[data-ledger-mount] .lab-ledger');
        const segs = {};
        [...host.querySelectorAll('.lab-ledger-table tbody tr')]
            .filter(tr => tr.querySelector('.lab-ledger-seg'))
            .forEach(tr => {
                const id = tr.querySelector('.lab-ledger-seg').textContent.trim();
                segs[id] = tr.querySelector('.lab-ledger-num').textContent.trim();
            });
        const total = host.querySelector('.lab-ledger-total .lab-ledger-num');
        out.push({
            i, id: T.steps[i].id,
            stepId: host.getAttribute('data-step-id'),
            phase: T.steps[i].ledger.config.phase,
            seq: T.steps[i].ledger.config.seq_len,
            segs, total: total ? total.textContent.trim() : null,
            kvWidth: (() => {
                const bar = host.querySelector('.lab-ledger-bar > span[data-seg="kv_cache"]');
                return bar ? Number(bar.style.flexGrow) : null;
            })(),
        });
    }
    P.setCursor(0);
    return out;
}"""

READ_SHAPE = """() => {
    const comp = document.querySelector('[data-panel="attn-shape"]');
    if (!comp) return null;
    const root = comp.querySelector('.lab-as');
    if (!root) return null;
    const out = {fromStep: root.dataset.fromStep, exact: root.dataset.exact,
                 ratio: Number(root.dataset.ratio),
                 cols: [], grids: [], facts: {}};
    comp.querySelectorAll('.lab-as-col').forEach(col => {
        const facts = {};
        col.querySelectorAll('.lab-as-facts dd').forEach(dd => {
            facts[dd.dataset.fact] = dd.textContent.trim();
        });
        out.cols.push({mode: col.dataset.mode, queries: Number(col.dataset.queries),
                       scoreElems: Number(col.dataset.scoreElems), facts,
                       ops: [...col.querySelectorAll('.lab-as-op')].map(o => ({
                           op: o.dataset.op, rows: Number(o.dataset.rows),
                           elems: Number(o.dataset.elems),
                           widthPct: Number(o.dataset.widthPct)}))});
    });
    const colsEl = comp.querySelector('.lab-as-cols');
    out.ncShare = Number(colsEl.dataset.ncShare);
    out.wcShare = Number(colsEl.dataset.wcShare);
    comp.querySelectorAll('.lab-as-grid').forEach(x => {
        const r = x.getBoundingClientRect();
        out.grids.push({rows: Number(x.dataset.rows), cols: Number(x.dataset.cols),
                        trueRows: Number(x.dataset.trueRows),
                        trueCols: Number(x.dataset.trueCols),
                        capped: x.dataset.capped === '1',
                        compressed: x.dataset.compressed === '1',
                        drawnAr: Number(x.dataset.drawnAr),
                        w: r.width, h: r.height,
                        labelCompressed: (x.parentElement.textContent || '').includes('压缩')});
    });
    out.summary = (comp.querySelector('.lab-as-sum') || {}).textContent || '';
    return out;
}"""

READ_ROOF = """() => {
    const pr = document.querySelector('[data-panel="phase-roofline"]');
    if (!pr) return null;
    const svg = pr.querySelector('.lab-roof-svg');
    if (!svg) return {error: 'no chart'};
    const sb = svg.getBoundingClientRect();
    const pb = pr.querySelector('.lab-roof-plot').getBoundingClientRect();
    const pts = [...pr.querySelectorAll('.lab-roof-pt')].map(g => {
        const c = g.querySelector('.lab-roof-dot').getBoundingClientRect();
        const t = g.querySelector('.lab-roof-pt-l').getBoundingClientRect();
        return {id: g.dataset.id, ai: Number(g.dataset.ai),
                ceiling: Number(g.dataset.ceiling), bound: g.dataset.bound,
                dot: {l: c.left, r: c.right, t: c.top, b: c.bottom},
                lbl: {l: t.left, r: t.right, t: t.top, b: t.bottom}};
    });
    const clipped = [];
    svg.querySelectorAll('text').forEach(t => {
        const b = t.getBoundingClientRect();
        if (b.left < sb.left - 0.5 || b.right > sb.right + 0.5 ||
            b.top < sb.top - 0.5 || b.bottom > sb.bottom + 0.5) {
            clipped.push(t.textContent);
        }
    });
    return {svg: {l: sb.left, r: sb.right, t: sb.top, b: sb.bottom},
            plot: {l: pb.left, r: pb.right, t: pb.top, b: pb.bottom},
            pts, clipped,
            rows: [...pr.querySelectorAll('.lab-pr-table tbody tr')].map(tr => ({
                phase: tr.dataset.phase,
                cells: [...tr.querySelectorAll('td')].map(td => td.textContent.trim())}))};
}"""


def inside(inner, outer, tol=0.5):
    return (inner["l"] >= outer["l"] - tol and inner["r"] <= outer["r"] + tol and
            inner["t"] >= outer["t"] - tol and inner["b"] <= outer["b"] + tol)


def overlaps(a, b):
    return a["l"] < b["r"] and b["l"] < a["r"] and a["t"] < b["b"] and b["t"] < a["b"]


def main():
    for p, hint in ((PAGE, "run `npm run build:labs` first"),
                    (MANIFEST, "run labs/traces/kv_cache.py first")):
        if not p.exists():
            print(f"missing {p} — {hint}", file=sys.stderr)
            return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    default = load(manifest["default"])
    url = PAGE.as_uri()
    global _PAGE

    lint_parity()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        _PAGE = page
        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        # ================================================ the set is all there
        print("\n== 配置集与 manifest 一致 ==")
        page.goto(url)
        page.wait_for_timeout(1200)
        inline = page.evaluate("() => window.LabTraceSets['kv-cache']")
        check("页面内联了 manifest", inline is not None and inline["set"] == "kv-cache")
        check("manifest 的配置数与生成器一致（18 组 = 3×2×3 的完整笛卡尔积）",
              inline["traces"] == manifest["traces"] and len(inline["traces"]) == 18,
              f"{len(inline['traces'])} 组")
        check("三个参数的可选值与生成器一致",
              inline["params"] == {"N": [8, 64, 128], "L": [2, 4],
                                   "prec": ["fp16", "fp8", "int8"]},
              json.dumps(inline["params"], ensure_ascii=False))
        check("默认配置是 N=8 · L=2 · fp16", inline["default"] == "kv-cache",
              inline["default"])
        inlined = page.evaluate("() => Object.keys(window.LabTraces)")
        missing = [n for n in manifest["traces"] if n not in inlined]
        check("manifest 里每一组都有一份内联的 trace", not missing,
              "缺：" + "、".join(missing))

        # The page must match a slider position to a trace by reading the
        # trace's own meta.config, not by parsing an id — so every grid point
        # must be reachable, and the ids are irrelevant to that.
        page.evaluate("""() => {
            const out = [];
            for (const N of [8, 64, 128])
              for (const L of [2, 4])
                for (const prec of ['fp16','fp8','int8']) out.push([N,L,prec]);
            window.__probeGrid = out;
        }""")
        grid_ok = page.evaluate("""() => {
            const T = window.__labTraces, names = Object.keys(T);
            return window.__probeGrid.map(([N,L,prec]) => {
                const hit = names.filter(n => {
                    const c = T[n].meta.config;
                    return c.N === N && c.L === L && c.dtype === prec;
                });
                return {key: N+'/'+L+'/'+prec, hits: hit.length,
                        cfg: hit.length ? T[hit[0]].meta.config : null};
            });
        }""")
        bad_grid = [g for g in grid_ok if g["hits"] != 1 or
                    g["cfg"]["N"] != int(g["key"].split("/")[0])]
        check("三个参数的每个组合都恰好对应一份 trace（页面按 meta.config 匹配，不解析 id）",
              not bad_grid, json.dumps(bad_grid[:2], ensure_ascii=False))

        # ============================================ AC4: sliders and readout
        print("\n== AC4 滑杆：三个参数可调，位置来自 manifest ==")
        sliders = page.evaluate(READ_SLIDERS)
        check("三个滑杆都渲染出来了（N / L / prec）",
              set(sliders) == {"N", "L", "prec"}, json.dumps(sliders, ensure_ascii=False))
        check("滑杆的可选值长度就是 manifest 的长度",
              all(s["max"] == len(manifest["params"][k]) - 1
                  for k, s in sliders.items()),
              json.dumps({k: v["max"] for k, v in sliders.items()}))
        check("滑杆当前值显示为当前配置的参数",
              {k: v["shown"] for k, v in sliders.items()} ==
              {"N": "8", "L": "2", "prec": "fp16"},
              json.dumps({k: v["shown"] for k, v in sliders.items()}))

        observed = {}
        for cfg_id, N, L, prec in DRIVEN:
            trace = load(cfg_id)
            page.goto(f"{url}?cfg={cfg_id}&step=0")
            page.wait_for_timeout(900)
            observed[cfg_id] = {
                "trace": trace,
                "metrics": page.evaluate(READ_METRICS),
                "sliders": page.evaluate(READ_SLIDERS),
                "config": page.evaluate("() => window.__labConfig"),
            }

        print("\n== AC4 数字来自 trace，不是页面算的 ==")
        for cfg_id, N, L, prec in DRIVEN:
            got = observed[cfg_id]
            trace, m = got["trace"], got["metrics"]
            led = trace["steps"][-1]["ledger"]["bytes"]
            cfg = trace["meta"]["config"]
            tot = trace["meta"]["attn_cost"]["totals"]
            pts = {x["id"]: x for x in trace["meta"]["phase_roofline"]["points"]}
            cross = trace["meta"]["kv_crossover"]
            # One browser round trip formats every value this row needs, with
            # the page's own rule — see the note on `fmt`.
            vals = [
                tot["no_cache"]["score_elems"] / max(tot["with_cache"]["score_elems"], 1),
                pts["prefill"]["ai"], pts["decode"]["ai"],
                pts["prefill"]["ai"] / pts["decode"]["ai"], cross["context"],
            ]
            ratio_s, ai_pre_s, ai_dec_s, ai_ratio_s, cross_s = fmt(vals)
            ratio_s = fmt([tot["no_cache"]["score_elems"] /
                           max(tot["with_cache"]["score_elems"], 1)], 3)[0]
            ai_ratio_s = fmt([pts["prefill"]["ai"] / pts["decode"]["ai"]], 3)[0]
            want = {
                "params": thousands(led["params"]) + " B",
                "kv_cache": thousands(led["kv_cache"]) + " B",
                "activations": thousands(led["activations"]) + " B",
                "total": thousands(sum(led.values())) + " B",
                "nc_score_elems": thousands(tot["no_cache"]["score_elems"]),
                "wc_score_elems": thousands(tot["with_cache"]["score_elems"]),
                "score_ratio": ratio_s + "×",
                "ai_prefill": ai_pre_s,
                "ai_decode": ai_dec_s,
                "ai_ratio": ai_ratio_s + "×",
                "per_token": thousands(cross["per_token_bytes"]) + " B",
                "crossover": cross_s,
            }
            bad = [f'{k}: {m.get(k)!r} != {v!r}' for k, v in want.items() if m.get(k) != v]
            check(f"{cfg_id}: 页面的 12 个读数与 trace 的 meta 块逐一一致", not bad,
                  "；".join(bad[:3]))
            slid = {k: v["shown"] for k, v in got["sliders"].items()}
            check(f"{cfg_id}: 滑杆显示的是本配置的 (N, L, prec)",
                  slid == {"N": str(N), "L": str(L), "prec": prec} and
                  got["config"] == cfg_id,
                  f"{json.dumps(slid, ensure_ascii=False)} · {got['config']}")

        # --- AC4's actual claim: the ledger's parts RECOMPUTE --------------------
        print("\n== AC4 改上下文 / 层数 / 精度 → 账本各分项正确重算 ==")
        base = observed["kv-cache"]
        ctx = observed["kv-cache-N128-L2-fp16"]
        lay = observed["kv-cache-N8-L4-fp16"]
        pre = observed["kv-cache-N8-L2-fp8"]
        far = observed["kv-cache-N128-L4-int8"]

        def parts(o):
            led = o["trace"]["steps"][-1]["ledger"]["bytes"]
            return led["params"], led["kv_cache"], led["activations"]

        b_par, b_kv, b_act = parts(base)
        c_par, c_kv, c_act = parts(ctx)
        l_par, l_kv, l_act = parts(lay)
        p_par, p_kv, p_act = parts(pre)

        check("上下文 ×16 → KV Cache 精确 ×16，参数与激活不动（对照组）",
              c_kv == 16 * b_kv and c_par == b_par and c_act == b_act,
              f"KV {b_kv:,} → {c_kv:,} B；参数 {b_par:,} → {c_par:,} B")
        check("层数 ×2 → KV Cache 精确 ×2，参数按公式增加而不是翻倍",
              l_kv == 2 * b_kv and l_par > b_par,
              f"KV {b_kv:,} → {l_kv:,} B；参数 {b_par:,} → {l_par:,} B")
        check("精度 fp16 → fp8 → KV Cache 精确减半",
              p_kv * 2 == b_kv, f"{b_kv:,} → {p_kv:,} B")
        # The rendered numbers, not just the trace's: the criterion is about what
        # the LEDGER shows, so the DOM is read back and compared to the trace.
        for label, o in (("上下文 ×16", ctx), ("层数 ×2", lay), ("fp16→fp8", pre)):
            led = o["trace"]["steps"][-1]["ledger"]["bytes"]
            m = o["metrics"]
            check(f"{label}: 页面上渲染的 KV/参数就是新 trace 的字节数",
                  m["kv_cache"] == thousands(led["kv_cache"]) + " B" and
                  m["params"] == thousands(led["params"]) + " B",
                  f'{m["kv_cache"]} / {m["params"]}')
        # And the far corner is the product of all three moves, so no slider can
        # be inert while the others carry the change.
        f_par, f_kv, f_act = parts(far)
        # Read the expectation off the trace's own formula — 2·L·H_kv·d_h·N·b —
        # rather than a hand-composed product of the three moves, so a mistake in
        # the composition cannot pass as a page bug (or hide one). Note what is
        # NOT asserted: that activations are unchanged. They are NOT, because the
        # precision slider moves `act_bytes` too — each single-move check above
        # isolates which of the three numbers a slider is allowed to move.
        f_cfg = far["trace"]["meta"]["config"]
        want_far = (2 * f_cfg["L"] * f_cfg["H"] * f_cfg["d_head"] * f_cfg["N"] *
                    f_cfg["kv_bytes"])
        check("三个滑杆一起推到另一端 → KV 分项等于 2·L·H_kv·d_h·N·b（每个滑杆都起了作用）",
              f_kv == want_far and f_kv != b_kv,
              f"KV {b_kv:,} → {f_kv:,} B（公式 {want_far:,}）")
        note(f"KV 分项：fp16/8/2 层 {b_kv:,} B → 上下文 128 {c_kv:,} B → "
             f"4 层 {l_kv:,} B → fp8 {p_kv:,} B → int8/256/4 层 {f_kv:,} B")

        # --- the sliders' own event wiring, which the deep links bypass ---------
        print("\n== 滑杆事件：拖动 N 真的会换配置 ==")
        page.goto(f"{url}?cfg=kv-cache&step=0")
        page.wait_for_timeout(700)
        before = page.evaluate("() => window.__labConfig")
        before_kv = page.evaluate(READ_METRICS)["kv_cache"]
        page.evaluate("""() => {
            const i = document.querySelector('#lab-params input[data-param="N"]');
            i.value = String(Number(i.max));       // 8 -> 256
            i.dispatchEvent(new Event('input', {bubbles: true}));
        }""")
        page.wait_for_timeout(900)
        after = page.evaluate("() => window.__labConfig")
        after_kv = page.evaluate(READ_METRICS)["kv_cache"]
        check("拖动 N 滑杆会载入新配置（8 → 256）",
              before == "kv-cache" and after == "kv-cache-N128-L2-fp16",
              f"{before} → {after}")
        check("换配置后页面的 KV Cache 随之重算（读的是新 trace 的数）",
              after_kv == thousands(load("kv-cache-N128-L2-fp16")
                                    ["steps"][-1]["ledger"]["bytes"]["kv_cache"]) + " B" and
              after_kv != before_kv,
              f"{before_kv} → {after_kv}")
        check("拖动滑杆后深链跟着更新（?cfg= 指向新配置）",
              "cfg=kv-cache-N128-L2-fp16" in page.url, page.url.split("?")[-1])

        # The OTHER 精度 position, to prove the third slider is wired too and
        # that its two byte-identical positions are both reachable.
        page.evaluate("""() => {
            const i = document.querySelector('#lab-params input[data-param="prec"]');
            i.value = '1';                          // fp16 -> fp8
            i.dispatchEvent(new Event('input', {bubbles: true}));
        }""")
        page.wait_for_timeout(900)
        check("拖动精度滑杆也会换配置（fp16 → fp8）",
              page.evaluate("() => window.__labConfig") == "kv-cache-N128-L2-fp8",
              page.evaluate("() => window.__labConfig"))

        # ==================================================== AC1: the replay
        print("\n== AC1 逐 token 生成可回放，KV Cache 的增长在视图上可见 ==")
        page.goto(f"{url}?cfg=kv-cache-N128-L2-fp16&step=0")
        page.wait_for_timeout(1000)
        series = page.evaluate(READ_LEDGER_SERIES)
        trace = load("kv-cache-N128-L2-fp16")
        check("回放的每一步都把账本重画了一次（data-step-id 与步骤一一对应）",
              len(series) == len(trace["steps"]) == 20 and
              all(r["stepId"] == r["id"] for r in series),
              f'{len(series)} 帧')

        def leading_int(s):
            return int(s.split()[0].replace(",", ""))

        # The cell reads "210,176 B · 205.3 KiB" — the exact count first, then a
        # human scale — so the comparison takes the leading integer and ignores
        # the suffix, which is a display convenience rather than a second fact.
        bad = []
        for r in series:
            s = trace["steps"][r["i"]]
            want = {seg["label"]: s["ledger"]["bytes"][seg["id"]]
                    for seg in trace["ledger"]["segments"]}
            got = {k: leading_int(v) for k, v in r["segs"].items()}
            if got != want:
                bad.append(f'{r["id"]}: 渲染={got} trace={want}')
        check("每一步渲染出的分项字节数 == trace 里发布的字节数", not bad,
              "；".join(bad[:2]) if bad else f"{len(series)} 步全部一致")

        kv_series = [leading_int(r["segs"]["KV Cache"]) for r in series]
        par_series = [leading_int(r["segs"]["参数"]) for r in series]
        check("KV Cache 分项在回放中单调不减", all(b >= a for a, b in zip(kv_series, kv_series[1:])),
              f"{kv_series[0]:,} → {kv_series[-1]:,} B")
        # The growth is at the KV append steps and NOWHERE else: a page whose
        # cache grew on the attention frame would be showing a cache that grows
        # when nothing was appended to it.
        #
        # The diff sees FOUR growths, not five. The prefill's append is the one
        # that starts the cache from nothing, so there is no preceding frame for
        # it to differ from; it shows up instead as step 0 already carrying the
        # prompt's rows, which is asserted separately below.
        grew_at = [r["stepId"] for a, b, r in zip(kv_series, kv_series[1:], series[1:])
                   if b > a]
        trace_grew = [b["id"] for a, b in zip(trace["steps"], trace["steps"][1:])
                      if b["ledger"]["bytes"]["kv_cache"] >
                         a["ledger"]["bytes"]["kv_cache"]]
        check("KV Cache 只在 K/V 追加步增长，且每一个 decode 追加步都增长",
              grew_at == trace_grew and
              all(i.endswith(".kv") for i in grew_at) and
              grew_at == ["d1.kv", "d2.kv", "d3.kv", "d4.kv"],
              f"增长发生在 {grew_at}")
        # ...and the cache did start non-empty: the prefill appended the whole
        # prompt, which is the part the diff above cannot see.
        check("Prefill 的追加也在视图上：第 0 步的 KV 分项就是整段 prompt 的行数",
              kv_series[0] > 0 and
              kv_series[0] == trace["steps"][0]["ledger"]["bytes"]["kv_cache"] and
              trace["steps"][0]["ledger"]["config"]["kv_rows_total"] ==
              trace["meta"]["config"]["L"] * trace["meta"]["config"]["prompt_len"],
              f'第 0 步 KV {kv_series[0]:,} B = '
              f'{trace["meta"]["config"]["L"]} 层 × {trace["meta"]["config"]["prompt_len"]} 行')
        check("参数分项在回放中恒定（它不随上下文变 —— 对照组）",
              len(set(par_series)) == 1, f"{par_series[0]:,} B")

        # The rendered bar's widths are driven by the bytes, so the KV block
        # visibly widens: this is the "在视图上可见" half of the criterion.
        widths = [r["kvWidth"] for r in series if r["kvWidth"] is not None]
        check("堆叠条上 KV 色块的宽度由字节数驱动（读的是真实 flex-grow）",
              len(widths) == len(series) and widths[-1] > widths[0] and
              widths == [s["ledger"]["bytes"]["kv_cache"] for s in trace["steps"]],
              f"首步 {widths[0]:,} → 末步 {widths[-1]:,}")
        # The sequence really does grow, one sampled token at a time: every head
        # step (one prefill + four decode) publishes the token it sampled, and
        # the last one's `tokens` array is the full sequence.
        sampled = [s["state"]["token_new"] for s in trace["steps"]
                   if "token_new" in s["state"]]
        last_len = len(trace["steps"][-1]["state"]["tokens"])
        # The sequence ends at N + 1, not N: the token sampled by the LAST head
        # step is appended to `tokens` but never fed back into the model, which
        # is the same asymmetry the trace's own notes call out for the cache
        # ("缓存比序列少一行"). Asserting == N would have failed on a trace that
        # is right, so the expectation comes from the trace's config plus that
        # stated off-by-one rather than from a guess.
        cfgT = trace["meta"]["config"]
        check("序列真的在逐 token 增长（5 个 head 步各采出一个新 token）",
              len(sampled) == 5 and len(set(sampled)) >= 3 and
              last_len == cfgT["prompt_len"] + cfgT["gen_len"] + 1,
              f"采出 {sampled}，末态序列 {last_len} 个 token"
              f"（prompt {cfgT['prompt_len']} + 5 步）")

        # The player's own replay controls must actually move the cursor, and
        # the ledger with it — otherwise "回放" is a static page.
        first = page.evaluate("""() => {
            window.__p.setCursor(0);
            const h = document.querySelector('[data-ledger-mount] .lab-ledger');
            return {step: h.getAttribute('data-step-id')};
        }""")
        page.evaluate("() => document.querySelector('[data-ctl=\"next\"]').click()")
        page.wait_for_timeout(350)
        second = page.evaluate("""() => {
            const h = document.querySelector('[data-ledger-mount] .lab-ledger');
            return {step: h.getAttribute('data-step-id')};
        }""")
        check("播放器的「下一步」按钮会推进回放并重画账本",
              first["step"] != second["step"], f'{first["step"]} → {second["step"]}')

        # ============================================ AC2: the shape comparison
        print("\n== AC2 有/无 KV Cache：attention 的输入 shape 对比 ==")
        # At each phase's attention step, both modes must be on screen with the
        # trace's own numbers. Driven by the cursor, so this is the replay's
        # comparison and not a static rendering of the default.
        for step_idx, label in ((1, "prefill"), (5, "decode 1"), (17, "decode 4")):
            page.goto(f"{url}?cfg=kv-cache-N128-L2-fp16&step={step_idx}")
            page.wait_for_timeout(800)
            shape = page.evaluate(READ_SHAPE)
            cost = trace["steps"][step_idx]["attn_cost"]
            nc, wc = cost["no_cache"], cost["with_cache"]
            check(f"{label}（{trace['steps'][step_idx]['id']}）：两栏都在，且是 trace 的两种模式",
                  shape is not None and
                  {c["mode"] for c in shape["cols"]} == {"no_cache", "with_cache"} and
                  shape["fromStep"] == trace["steps"][step_idx]["id"] and
                  shape["exact"] == "1",
                  json.dumps(shape["cols"] and {c["mode"]: c["scoreElems"]
                                                for c in shape["cols"]}))
            by_mode = {c["mode"]: c for c in shape["cols"]}
            check(f"{label}：两栏的查询数与分数矩阵元素数 == trace 的 attn_cost",
                  by_mode["no_cache"]["queries"] == nc["queries"] and
                  by_mode["no_cache"]["scoreElems"] == nc["score_elems"] and
                  by_mode["with_cache"]["queries"] == wc["queries"] and
                  by_mode["with_cache"]["scoreElems"] == wc["score_elems"],
                  f'无 {nc["queries"]}/{nc["score_elems"]}，有 {wc["queries"]}/{wc["score_elems"]}')
            # The operand boxes carry the shapes the tutorial draws: [t,d] vs
            # [n_q,d]. Their ROWS are the query counts, and the widths on screen
            # are proportional to them — which is what makes the collapse visible.
            for mode in ("no_cache", "with_cache"):
                ops = {o["op"]: o for o in by_mode[mode]["ops"]}
                want_q = cost[mode]["q_shape"][0]
                want_k = cost[mode]["k_shape"][0]
                if ops["Q"]["rows"] != want_q or ops["K"]["rows"] != want_k:
                    check(f"{label} {mode}：Q/K 形状框的行数 == trace 的 q_shape/k_shape",
                          False, f'{ops["Q"]["rows"]}/{ops["K"]["rows"]} != {want_q}/{want_k}')
                    break
            else:
                check(f"{label}：Q/K 形状框的行数 == trace 的 q_shape/k_shape", True,
                      f'无 Q{cost["no_cache"]["q_shape"][0]} K{cost["no_cache"]["k_shape"][0]}，'
                      f'有 Q{cost["with_cache"]["q_shape"][0]} K{cost["with_cache"]["k_shape"][0]}')
            # The operand bars are drawn against ONE shared scale, so the cached
            # Q bar must be exactly n_q/t of the uncached one. This is the
            # picture's central claim, asserted as a ratio rather than eyeballed.
            q_nc = {o["op"]: o for o in by_mode["no_cache"]["ops"]}["Q"]
            q_wc = {o["op"]: o for o in by_mode["with_cache"]["ops"]}["Q"]
            want_pct = 100 * wc["queries"] / max(nc["queries"], 1)
            # The tolerance is for the ROUND TRIP, not for the rule: the
            # component writes the width as a 3-decimal percent and the harness
            # parses the attribute back, so 0.78125% arrives as 0.781%. A
            # tolerance that demanded more would be testing `toFixed(3)`.
            check(f"{label}：两栏的 Q 框按同一把尺子画（有缓存 = 无缓存的 n_q/t）",
                  abs(q_wc["widthPct"] - want_pct) <= 0.001,
                  f'{q_wc["widthPct"]}% vs 期望 {want_pct}%')
            if label == "prefill":
                check("对照组：Prefill 时两种模式的工作量相同（缓存是空的）",
                      nc["queries"] == wc["queries"] and
                      nc["score_elems"] == wc["score_elems"] and
                      shape["ratio"] == 1.0 and "完全相同" in shape["summary"],
                      f'两者都是 {nc["score_elems"]} 个元素')
            else:
                check(f"{label}：Decode 时无缓存要重算整段前缀，比值 = t",
                      nc["queries"] == cost["t"] and wc["queries"] == 1 and
                      abs(shape["ratio"] - nc["score_elems"] / wc["score_elems"]) < 1e-6 and
                      shape["ratio"] > 1,
                      f'比值 {shape["ratio"]}，查询 {nc["queries"]} → {wc["queries"]}')
        # The control that makes the divergence attributable: the two modes
        # must be IDENTICAL at prefill and must differ at decode. A harness that
        # only checked "the cached one is smaller" would pass on a page that
        # understated both.
        check("对照组：同一配置里 prefill 两栏相同、decode 两栏不同"
              "（差别是 decode 独有的，不是全程的）",
              trace["steps"][1]["attn_cost"]["no_cache"]["score_elems"] ==
              trace["steps"][1]["attn_cost"]["with_cache"]["score_elems"] and
              all(trace["steps"][i]["attn_cost"]["no_cache"]["score_elems"] >
                  trace["steps"][i]["attn_cost"]["with_cache"]["score_elems"]
                  for i in (5, 9, 13, 17)))
        # And it must be on screen at every cursor: the comparison panel is
        # never blank, and when it shows a neighbouring step's numbers it says
        # which step they are from.
        blanks = []
        for i in range(len(trace["steps"])):
            page.evaluate(f"() => window.__p.setCursor({i})")
            page.wait_for_timeout(60)
            shape = page.evaluate(READ_SHAPE)
            if shape is None or len(shape["cols"]) != 2:
                blanks.append(trace["steps"][i]["id"])
        check("回放的每一帧都不留空：对比面板始终有两栏（非注意力帧显示最近一次注意力）",
              not blanks, "空帧：" + "、".join(blanks))
        page.evaluate("() => window.__p.setCursor(0)")
        page.wait_for_timeout(200)
        near = page.evaluate(READ_SHAPE)
        check("非注意力帧上明说了数字取自哪一步（不假装它就是当前步）",
              near["exact"] == "0" and near["fromStep"].endswith(".attn") and
              "最近一次注意力" in page.evaluate(
                  "() => document.querySelector('[data-panel=attn-shape]').textContent"),
              f'第 0 步显示的是 {near["fromStep"]}')

        # ============================================ AC3: the shared component
        print("\n== AC3 显存账本是可复用的独立组件（#45 已交付），本页只是消费它 ==")
        reuse = page.evaluate("""() => {
            const host = document.querySelector('[data-ledger-mount]');
            const r = host && host.querySelector('.lab-ledger');
            const pr = document.querySelector('#l05-predict-ledger .lab-ledger');
            return {
                mounted: !!r,
                segs: [...r.querySelectorAll('.lab-ledger-seg')].map(e => e.textContent.trim()),
                formulas: r.querySelectorAll('.lab-ledger-fml .katex').length,
                ticks: !!r.querySelector('.lab-ledger-tick'),
                predictMounted: !!pr,
                predictSegs: pr ? [...pr.querySelectorAll('.lab-ledger-seg')]
                                    .map(e => e.textContent.trim()) : [],
                paintFns: Object.keys(LabEngine.ledger).sort(),
                cols: Object.keys(LabEngine.ledger).length,
            };
        }""")
        check("账本是 LabEngine.ledger 组件渲染的，本节没有自绘堆叠条",
              reuse["mounted"] and reuse["predictMounted"] and reuse["ticks"])
        # The segments come from the trace's own declaration, not a literal in
        # the page: changing the trace's segments changes the legend.
        declared = {seg["label"] for seg in default["ledger"]["segments"]}
        check("分项名称来自 trace 的 ledger.segments，不是写死在页面里",
              set(reuse["segs"]) == declared == {"参数", "KV Cache", "激活"},
              json.dumps(reuse["segs"], ensure_ascii=False))
        check("每个分项的显存公式由 KaTeX 渲染（可追溯到公式，而不是一串无出处的数字）",
              reuse["formulas"] == 3, f'{reuse["formulas"]} 条公式')
        # Reuse across configurations, and across a configuration no trace ran:
        # that is the "组件" claim, and it is what L14 will lean on.
        check("同一个组件也在滑杆驱动的「预测值」面板上渲染（同一份组件，不同数据源）",
              reuse["predictSegs"] == reuse["segs"], json.dumps(reuse["predictSegs"]))
        check("组件暴露的接口是通用的（bar / matrix / stack / lint / check / panel）",
              {"bar", "matrix", "stack", "lint", "check", "panel", "SABOTAGE_CASES"}
              <= set(reuse["paintFns"]), ", ".join(reuse["paintFns"]))

        # The self-check verdicts the page computed, read back rather than
        # re-derived: a green panel that the page rendered is the thing a reader
        # sees, so it is the thing that gets asserted.
        checks = page.evaluate("() => window.__labViewChecks")
        check("页面上三个自检面板都跑过并给出结论",
              checks is not None and set(checks) == {"ledger", "attnShape", "phaseRoofline"},
              json.dumps(list(checks or {})))
        check("账本自检：20 步逐步对拍 0 不一致、4 条缩放性质成立、4 种错模型全被抓到",
              checks["ledger"]["stepFailures"] == 0 and
              checks["ledger"]["propertyFailures"] == 0 and
              not checks["ledger"]["missedControls"] and len(checks["ledger"]["controls"]) == 4,
              f'步 {checks["ledger"]["stepFailures"]}，性质 {checks["ledger"]["propertyFailures"]}，'
              f'对照组漏 {checks["ledger"]["missedControls"]}')
        for key, label in (("attnShape", "有/无 cache 对比"), ("phaseRoofline", "Roofline")):
            check(f"{label}自检：0 gap、{checks[key]['sabotageTotal']} 种破坏全被抓到",
                  checks[key]["lintGaps"] == 0 and not checks[key]["sabotageMissed"],
                  f'gaps={checks[key]["lintGaps"]}, 漏={checks[key]["sabotageMissed"]}')

        # The engine's own jump check, on this trace. It rides the same paint
        # path the ledger does, so a broken jump would render the ledger against
        # a state the trace cannot reproduce.
        jump = page.evaluate("() => window.__labVerify && window.__labVerify.L05")
        check("引擎的任意跳转纯函数重建在此 trace 上成立（且对照组确实翻车）",
              jump is not None and jump["jumps"]["pureFailures"] == 0 and
              jump["jumps"]["controlFailures"] > 0 and jump["jumps"]["conclusive"] and
              jump["jumps"]["passed"],
              json.dumps(jump["jumps"], ensure_ascii=False) if jump else "missing")
        check("引擎的基础契约 lint（trace-model）在此 trace 上干净、且对照组适用本份 trace",
              jump["lint"]["gaps"] == 0 and jump["lint"]["passed"] and
              jump["lint"]["conclusive"],
              json.dumps(jump["lint"], ensure_ascii=False))

        # ================================================ AC5: the phase Roofline
        print("\n== AC5 Prefill 与 Decode 在 Roofline 上落点不同，且能解释原因 ==")
        roof_rows = {}
        for cfg_id, N, L, prec in DRIVEN:
            trace = load(cfg_id)
            page.goto(f"{url}?cfg={cfg_id}&step=0")
            page.wait_for_timeout(800)
            roof = page.evaluate(READ_ROOF)
            roof_rows[cfg_id] = roof
            by_id = {x["id"]: x for x in trace["meta"]["phase_roofline"]["points"]}
            bad = []
            if set(x["id"] for x in roof["pts"]) != {"prefill", "decode", "doc"}:
                bad.append(f'点集 {[x["id"] for x in roof["pts"]]}')
            else:
                for pt in roof["pts"]:
                    want = by_id[pt["id"]]
                    if abs(pt["ai"] - want["ai"]) > 1e-9:
                        bad.append(f'{pt["id"]}.ai {pt["ai"]} != {want["ai"]}')
                    if abs(pt["ceiling"] - want["ceiling"]) > 1e-6:
                        bad.append(f'{pt["id"]}.ceiling 不一致')
                    if pt["bound"] != want["bound"]:
                        bad.append(f'{pt["id"]}.bound {pt["bound"]} != {want["bound"]}')
            check(f"{cfg_id}: 图上三个点的坐标/归属与 meta.phase_roofline 一致", not bad,
                  "；".join(bad[:2]))
            # The breakdown table is the "能解释原因" half: the terms that
            # produced each point's intensity are on screen beside the chart.
            rows = {r["phase"]: r["cells"] for r in roof["rows"]}
            # The table is rendered by the VIEW, so the expected strings come
            # from the view's own formatter (`phaseRoofline.format`) rather than
            # from the page's — two different renderers, and the harness must
            # ask the one that produced the DOM. This is the same lesson as
            # `fmt`, one level down.
            pts_list = trace["meta"]["phase_roofline"]["points"]
            flat = [x for p in pts_list for x in
                    (p["flops"], p["parts"]["weights_bytes"],
                     p["parts"]["kv_read_bytes"] + p["parts"]["kv_write_bytes"], p["ai"])]
            strings = page.evaluate(
                "vs => vs.map(v => LabEngine.phaseRoofline.format(v))", flat)
            want_rows = {}
            for k, p in enumerate(pts_list):
                want_rows[p["id"]] = strings[k * 4:(k + 1) * 4] + [
                    "算力" if p["bound"] == "compute" else "带宽"]
            check(f"{cfg_id}: 三行明细表的每个数与 trace 的分项一致（图上每个点都能追溯到项）",
                  rows == want_rows,
                  json.dumps({k: rows.get(k) for k in ("prefill", "decode")}, ensure_ascii=False))

        # The mounting evidence: the two live points are in DIFFERENT places,
        # and the direction is the one the arithmetic says.
        base = roof_rows["kv-cache"]
        pts = {x["id"]: x for x in base["pts"]}
        check("Prefill 与 Decode 在图上确实是两个不同的落点",
              (pts["prefill"]["dot"]["l"], pts["prefill"]["dot"]["t"]) !=
              (pts["decode"]["dot"]["l"], pts["decode"]["dot"]["t"]),
              f'x: {pts["prefill"]["dot"]["l"]:.0f} vs {pts["decode"]["dot"]["l"]:.0f}')
        check("Prefill 的算术强度远高于 Decode → 点明显更靠右",
              pts["prefill"]["dot"]["l"] > pts["decode"]["dot"]["l"],
              f'x 差 {pts["prefill"]["dot"]["l"] - pts["decode"]["dot"]["l"]:.0f} px')
        # Which roof binds, which is the criterion's "能解释原因". Both live
        # points are bandwidth-bound — that is the lab's claim — and being ON the
        # diagonal means their y's DIFFER, because the diagonal rises with
        # intensity. The reference point is the only one that reaches the flat
        # compute roof, so it is the only one whose y stops tracking its x.
        check("Prefill 与 Decode 都落在带宽屋顶（同一条斜线上），参考点才够到算力屋顶",
              pts["prefill"]["bound"] == pts["decode"]["bound"] == "bandwidth" and
              pts["doc"]["bound"] == "compute" and
              # The compute roof is a ceiling in y: the doc dot sits no lower
              # than any bandwidth-bound point can reach, and strictly higher on
              # screen than the prefill dot whose intensity is far lower.
              pts["doc"]["dot"]["t"] < pts["prefill"]["dot"]["t"] and
              pts["doc"]["dot"]["t"] < pts["decode"]["dot"]["t"],
              f'prefill/decode bound={pts["prefill"]["bound"]}/{pts["decode"]["bound"]}，'
              f'doc bound={pts["doc"]["bound"]}；'
              f'doc y {pts["doc"]["dot"]["t"]:.0f} < '
              f'prefill {pts["prefill"]["dot"]["t"]:.0f}')
        def px_of(cfg_id, pid, field):
            return {x["id"]: x[field] for x in roof_rows[cfg_id]["pts"]}[pid]

        check("上下文越长 Prefill 点越靠右（它随滑杆移动）",
              px_of("kv-cache-N128-L2-fp16", "prefill", "dot")["l"] >
              px_of("kv-cache", "prefill", "dot")["l"],
              f'N=8 x={px_of("kv-cache", "prefill", "dot")["l"]:.0f} → '
              f'N=128 x={px_of("kv-cache-N128-L2-fp16", "prefill", "dot")["l"]:.0f}')
        # ...and it moves for the RIGHT reason: the intensity, not the layout.
        ai_ctx = {c: px_of(c, "prefill", "ai") for c, _, _, _ in DRIVEN}
        check("Prefill 点的移动方向与算术强度一致（上下文 8 → 128，AI 增大）",
              ai_ctx["kv-cache-N128-L2-fp16"] > ai_ctx["kv-cache"],
              f'{ai_ctx["kv-cache"]:.2f} → {ai_ctx["kv-cache-N128-L2-fp16"]:.2f} FLOP/Byte')

        # --- the CONTROL: the reference point is the same dot everywhere -------
        # The control: the reference point is the same MODEL everywhere, so its
        # intensity and its traffic must be identical across configurations.
        # What is deliberately NOT asserted is its pixel position — the chart
        # derives its axis domain from the points it is handed, so the same dot
        # legitimately lands elsewhere once the live points move. Asserting the
        # pixel would fail on a chart that is right, which is the bug the first
        # version of this check had.
        doc_data = {c: (px_of(c, "doc", "ai"), px_of(c, "doc", "ceiling"))
                    for c, _, _, _ in DRIVEN}
        # The right-edge rule: a label whose dot crowds the frame must have been
        # flipped to its left, and NO label may end up outside the viewBox —
        # which is what the geometry check above measures, per configuration.
        #
        # What this asserts is the MECHANISM, not a specific point. The flip is
        # a function of where the dots land, and the dots move with the sliders:
        # at the default the doc reference is the one near the edge, but a long
        # prompt at 128 tokens pushes prefill out there too (`kv-cache-N128-L4-
        # int8` is the configuration that caught the original clip). Naming a
        # point here would be asserting today's layout rather than the rule.
        flipped_all = {}
        for cfg_id, N, L, prec in DRIVEN:
            page.goto(f"{url}?cfg={cfg_id}&step=0")
            page.wait_for_timeout(700)
            got = page.evaluate("""() => {
                const el = document.querySelector(
                    '[data-panel="phase-roofline"] .lab-pr-chart');
                return el ? el.dataset.flipped : null;
            }""")
            flipped_all[cfg_id] = (got or '').split()
        # The doc point is ALWAYS anchored left — that is `rightmostId`'s doing
        # in the shared view, not the wrapper's, so it never appears in this
        # list. What the wrapper adds is the same treatment for the OTHER points
        # when the sliders push them out there. The union must therefore be
        # non-empty (the reference point is always near the edge) and must
        # include a live point on at least one configuration — the long-prompt
        # one, which is exactly where the original clip was found.
        check("右边缘规则在起作用：每个配置都有一个点被左锚定，且滑杆推远时活点也会被接管",
              any(cid == "kv-cache-N128-L4-int8" and v for cid, v in flipped_all.items()) and
              "prefill" in flipped_all["kv-cache-N128-L4-int8"],
              json.dumps(flipped_all, ensure_ascii=False))

        check("对照组：参考模型的 AI 与上界在所有配置里完全相同（它是同一个模型）",
              len(set(doc_data.values())) == 1,
              json.dumps({k: [round(a, 4), round(b, 2)] for k, (a, b) in doc_data.items()}))
        doc_ai = {c: px_of(c, "doc", "ai") for c, _, _, _ in DRIVEN}
        check("对照组：参考点的 AI 恒为 493（7B / 512 prompt），与三个滑杆无关",
              all(abs(v - 492.6) < 1.0 for v in doc_ai.values()),
              json.dumps({k: round(v, 2) for k, v in doc_ai.items()}))
        note(f"三个点：decode AI {pts['decode']['ai']:.3f}（带宽受限，最左）→ "
             f"prefill AI {pts['prefill']['ai']:.2f}（带宽受限）→ "
             f"参考 7B AI {pts['doc']['ai']:.1f}（唯一够到算力屋顶的点）；"
             f"ridge {base and load('kv-cache')['meta']['phase_roofline']['ridge']:.1f}")

        # ============================================================ geometry
        print("\n== 几何：Roofline 的点在图框内、标签不压别的点、不被裁 ==")
        for cfg_id, N, L, prec in DRIVEN:
            roof = roof_rows[cfg_id]
            if roof.get("error"):
                check(f"{cfg_id}: Roofline 图画出来了", False, roof["error"])
                continue
            plot, bad_inside, bad_lbl, bad_plot = roof["plot"], [], [], []
            for d in roof["pts"]:
                if not inside(d["dot"], plot):
                    bad_inside.append(d["id"])
                for e in roof["pts"]:
                    if e["id"] != d["id"] and overlaps(d["lbl"], e["dot"]):
                        bad_lbl.append([d["id"], e["id"]])
                # Inside the PLOT, not merely inside the SVG: the axes and titles
                # are drawn in the margins, and a dot there sits on a tick label.
                if not (d["dot"]["r"] >= plot["l"] and d["dot"]["l"] <= plot["r"] and
                        d["dot"]["b"] >= plot["t"] and d["dot"]["t"] <= plot["b"]):
                    bad_plot.append(d["id"])
            check(f"{cfg_id}: 三个点都落在图框内（不压轴线/刻度）",
                  not bad_inside and not bad_plot, f"越界 {bad_inside or bad_plot}")
            # Each dot's own label may not run outside the frame either: a label
            # half outside the viewBox is silently cut, which reads as a glitch.
            check(f"{cfg_id}: 点的标签不压在别的点上、也没有文字被 viewBox 裁掉",
                  not bad_lbl and not roof["clipped"],
                  json.dumps(bad_lbl) + " " + json.dumps(roof["clipped"], ensure_ascii=False))

        print("\n== 几何：shape 对比的矩形不溢出面板、长宽比不说谎 ==")
        for cfg_id, N, L, prec in DRIVEN:
            for step_idx in (1, 5):
                page.goto(f"{url}?cfg={cfg_id}&step={step_idx}")
                page.wait_for_timeout(700)
                geo = page.evaluate("""() => {
                    const comp = document.querySelector('[data-panel="attn-shape"]');
                    const cb = comp.getBoundingClientRect();
                    const over = [];
                    comp.querySelectorAll('*').forEach(e => {
                        const r = e.getBoundingClientRect();
                        if (r.width > 0 && (r.left < cb.left - 1 || r.right > cb.right + 1)) {
                            over.push(e.className || e.tagName);
                        }
                    });
                    const grid = comp.querySelector('.lab-as-grid-wrap');
                    const gb = grid ? grid.getBoundingClientRect() : null;
                    return {over, gridRight: gb ? gb.right : null, compRight: cb.right};
                }""")
                check(f"{cfg_id} 步 {step_idx}: 对比面板里没有元素溢出面板边界",
                      not geo["over"], f'{len(geo["over"])} 个：{geo["over"][:3]}')
                check(f"{cfg_id} 步 {step_idx}: 分数矩阵不超出它所在的栏",
                      geo["gridRight"] is None or geo["gridRight"] <= geo["compRight"] + 1,
                      f'{geo["gridRight"]} vs {geo["compRight"]}')
            # The aspect-ratio honesty check, at the decode step where the two
            # modes differ: a rectangle that COMPRESSED must say so, and one that
            # did not must not.
            page.goto(f"{url}?cfg={cfg_id}&step=5")
            page.wait_for_timeout(700)
            shape = page.evaluate(READ_SHAPE)
            for g in shape["grids"]:
                drawn = g["w"] / g["h"] if g["h"] else 0
                agrees = abs(drawn - g["drawnAr"]) / g["drawnAr"] < 0.06
                if not agrees:
                    check(f'{cfg_id} 步 5: {g["trueRows"]}x{g["trueCols"]} 的矩形'
                          f'(画成 {drawn:.2f}:1，真实 {g["drawnAr"]:.1f}:1) 必须标注「压缩」',
                          g["compressed"] and g["labelCompressed"], json.dumps(g))
                else:
                    check(f'{cfg_id} 步 5: {g["trueRows"]}x{g["trueCols"]} 的矩形'
                          f'长宽比忠实（{drawn:.2f}:1）且没有多标「压缩」',
                          not g["labelCompressed"], json.dumps(g))

        # --- the geometry control group -----------------------------------------
        # The checks above must be able to FAIL. Note what this does NOT do: it
        # does not push a point's data out of range and hope the dot follows —
        # the view derives its domain from the points, so a tampered intensity
        # simply widens the axis and the dot stays inside. (That was the first
        # version of this control, and it proved nothing: `bad` was empty,
        # correctly, because the view had re-scaled around the sabotage.)
        #
        # What a measurement needs to be able to see is a DOM that VIOLATES the
        # invariant, so the control BUILDS one: it renders the real view, moves
        # a dot where the rules forbid, squeezes a grid, and requires the same
        # predicates, evaluated the same way, to flag both.
        print("\n== 几何对照：人为把点推出图框 / 把标签压到点上 / 压缩矩形不标注，规则必须报警 ==")
        page.goto(f"{url}?cfg=kv-cache-N128-L2-fp16&step=5")
        page.wait_for_timeout(900)
        control = page.evaluate("""() => {
            const T = window.__labTrace;
            const out = {};
            // (1) the Roofline: a dot pushed past the frame's left edge
            const rv = LabEngine.roofline.makeView({trace: {meta: {roofline:
                LabEngine.phaseRoofline.asRoofline(T.meta.phase_roofline)}}});
            // (1a) the phase view's shim, so the control exercises the same
            // object the page renders rather than a hand-built chart.
            const pv = LabEngine.phaseRoofline.makeView({trace: T});
            const host = document.createElement('div');
            host.style.cssText = 'position:absolute;left:-9999px;top:0';
            document.body.appendChild(host);
            host.innerHTML = pv.render();
            const readDots = () => {
                const sb = host.querySelector('.lab-roof-svg').getBoundingClientRect();
                const pb = host.querySelector('.lab-roof-plot').getBoundingClientRect();
                const dots = [...host.querySelectorAll('.lab-roof-pt')].map(g => {
                    const c = g.querySelector('.lab-roof-dot').getBoundingClientRect();
                    const t = g.querySelector('.lab-roof-pt-l').getBoundingClientRect();
                    return {id: g.dataset.id,
                            dot: {l: c.left, r: c.right, t: c.top, b: c.bottom},
                            lbl: {l: t.left, r: t.right, t: t.top, b: t.bottom}};
                });
                const clipped = [];
                host.querySelectorAll('.lab-roof-svg text').forEach(t => {
                    const b = t.getBoundingClientRect();
                    if (b.left < sb.left - 0.5 || b.right > sb.right + 0.5 ||
                        b.top < sb.top - 0.5 || b.bottom > sb.bottom + 0.5) {
                        clipped.push(t.textContent);
                    }
                });
                return {pb, dots, clipped};
            };
            const outside = (m, pb) =>
                !(m.l >= pb.left - 0.5 && m.r <= pb.right + 0.5 &&
                  m.t >= pb.top - 0.5 && m.b <= pb.bottom + 0.5);
            const overlaps = (a, b) =>
                a.l < b.r && b.l < a.r && a.t < b.b && b.t < a.b;

            const tiled = host.querySelector('.lab-roof-pt-prefill .lab-roof-dot');
            const keep = tiled.getAttribute('cx');
            tiled.setAttribute('cx', '-60');
            out.frameBad = readDots().dots.filter(d => outside(d.dot, readDots().pb))
                                         .map(d => d.id);
            tiled.setAttribute('cx', keep);

            const a = host.querySelector('.lab-roof-pt-decode .lab-roof-pt-l');
            const docDot = host.querySelector('.lab-roof-pt-doc .lab-roof-dot')
                                    .getBoundingClientRect();
            const sx = a.getAttribute('x'), sy = a.getAttribute('y');
            const svgEl = host.querySelector('.lab-roof-svg');
            const vb = svgEl.getBoundingClientRect();
            const pbNow = host.querySelector('.lab-roof-plot').getBoundingClientRect();
            // The offsets are measured in RENDERED pixels, but `x`/`y` are in
            // viewBox units — and the SVG is scaled to its container, so on a
            // 1530px panel a 720-unit viewBox scales by ~2.1. Dividing by the
            // scale is what makes the label land where the dot is; without it
            // the label moves a fifth of the way and no overlap is produced,
            // which is how the first version of this control managed to pass
            // nothing while looking like it proved something.
            const sxScale = vb.width / Number(svgEl.getAttribute('data-w'));
            const syScale = vb.height / Number(svgEl.getAttribute('data-h'));
            const dx = (docDot.left - a.getBoundingClientRect().left) / sxScale;
            const dy = (docDot.top - a.getBoundingClientRect().top) / syScale;
            a.setAttribute('x', String(Number(sx) + dx));
            a.setAttribute('y', String(Number(sy) + dy));
            const after = readDots();
            const dp = after.dots.find(d => d.id === 'decode');
            const doc = after.dots.find(d => d.id === 'doc');
            out.labelBad = !!(dp && doc && overlaps(dp.lbl, doc.dot));
            out.labelMoved = Math.round(a.getBoundingClientRect().left - docDot.left);
            a.setAttribute('x', sx); a.setAttribute('y', sy);

            // (1b) a text node pushed outside the viewBox must be reported as
            // clipped
            const axis = host.querySelector('.lab-roof-ax');
            const ax0 = axis.getAttribute('x');
            axis.setAttribute('x', '-400');
            out.clipBad = readDots().clipped.length > 0;
            axis.setAttribute('x', ax0);

            host.remove();

            // (2) the shape comparison: a grid forced away from its declared
            // aspect ratio must be reported as disagreeing with it.
            //
            // Two details this got wrong first. The sabotage must PIN a
            // dimension — squeezing a square by width alone leaves it square,
            // because its height follows — so the height is pinned and the
            // width clamped. And the restore must put the whole style ATTRIBUTE
            // back: `grid.style.maxWidth = ''` removes the property that the
            // style attribute itself defined, so the "unmodified" grid measured
            // afterwards was not unmodified at all and read as dishonest.
            const sh = document.querySelector('[data-panel="attn-shape"]');
            const grid = sh.querySelector('.lab-as-grid');
            const declared = Number(grid.dataset.drawnAr);

            // ...first the predicate on the UNTOUCHED grid: it must call it
            // honest, or "it flagged the sabotage" would only prove it always
            // flags.
            const ok = grid.getBoundingClientRect();
            out.cleanAr = ok.width / ok.height;
            out.cleanOk = Math.abs(out.cleanAr - declared) / declared < 0.06;

            const savedStyle = grid.getAttribute('style');
            grid.setAttribute('style', savedStyle +
                ';max-width:40px;height:' + Math.round(ok.height) + 'px;' +
                'aspect-ratio:auto');
            const squeezed = grid.getBoundingClientRect();
            const ar = squeezed.width / squeezed.height;
            out.squeezeBad = !(Math.abs(ar - declared) / declared < 0.06);
            out.squeezedAr = ar;
            out.declaredAr = declared;
            grid.setAttribute('style', savedStyle);

            // ...and after restoring, the predicate must call it honest again —
            // which is what the broken restore above silently failed to do.
            const restored = grid.getBoundingClientRect();
            out.restoredAr = restored.width / restored.height;
            out.restoredOk = Math.abs(out.restoredAr - declared) / declared < 0.06;
            return out;
        }""")
        check("对照组：被推出图框的点会被「点在框内」规则抓到",
              control["frameBad"] == ["prefill"], json.dumps(control))
        check("对照组：压到别的点上的标签会被「标签不压点」规则抓到",
              control["labelBad"], json.dumps(control))
        check("对照组：被移出 viewBox 的文字会被「不被裁」规则抓到",
              control["clipBad"], json.dumps(control))
        check("对照组：被压扁的分数矩阵会被「长宽比不说谎」规则抓到，"
              "而同一规则对未改动的那一个判为诚实（且破坏后可恢复）",
              control["squeezeBad"] and control["cleanOk"] and control["restoredOk"],
              f'压后 {control["squeezedAr"]:.2f} vs 声明 {control["declaredAr"]:.2f}；'
              f'未动 {control["cleanAr"]:.2f}；还原后 {control["restoredAr"]:.2f}')

        # ============================================================== shots
        print("\n== 截图 ==")
        page.goto(f"{url}?cfg=kv-cache-N128-L2-fp16&step=5")
        page.wait_for_timeout(900)
        page.evaluate("() => window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "50-l05-overview.png")

        page.locator('[data-panel="attn-shape"]').screenshot(
            path=SHOTS / "51-l05-shape-compare.png")
        page.locator('[data-panel="phase-roofline"]').screenshot(
            path=SHOTS / "52-l05-phase-roofline.png")
        page.locator('[data-panel="ledger"]').screenshot(
            path=SHOTS / "53-l05-ledger-mid.png")

        # The ledger at its two ends, so a reviewer sees the growth the harness
        # asserts without running it.
        for idx, name in ((0, "54-l05-ledger-first"), (19, "55-l05-ledger-last")):
            page.evaluate(f"() => window.__p.setCursor({idx})")
            page.wait_for_timeout(400)
            page.locator('[data-panel="ledger"]').screenshot(path=SHOTS / f"{name}.png")

        # The smallest and largest context, for the shape comparison's two ends.
        for cfg_id, name in (("kv-cache", "56-l05-shape-small"),
                             ("kv-cache-N128-L4-int8", "57-l05-shape-large")):
            page.goto(f"{url}?cfg={cfg_id}&step=5")
            page.wait_for_timeout(900)
            page.locator('[data-panel="attn-shape"]').screenshot(path=SHOTS / f"{name}.png")

        # The predicted-configuration ledger, driven to a context where the
        # cache dominates — the extrapolation the sliders exist for, and one no
        # trace covers. Dragged with real `input` events, so the wiring is
        # exercised rather than just the initial render.
        page.goto(f"{url}?cfg=kv-cache-N128-L4-fp16&step=0")
        page.wait_for_timeout(800)
        page.evaluate("""() => {
            const set = (idx, v) => {
                const el = document.querySelectorAll('#l05-predict-row input')[idx];
                el.value = String(v);
                el.dispatchEvent(new Event('input', {bubbles: true}));
            };
            set(0, 4096);   // context
            set(1, 32);     // layers
            set(2, 2);      // fp16
        }""")
        page.wait_for_timeout(400)
        pred = page.evaluate("""() => ({
            bytes: window.__ledgerPredict,
            cfg: window.__ledgerPredictCfg,
            caption: document.querySelector('#l05-predict-ledger .lab-ledger-caption')
                             .textContent,
        })""")
        check("预测面板：滑杆推到「上下文已反超参数」的配置，KV 分项成为主体",
              pred["cfg"]["seq_len"] == 4096 and pred["cfg"]["layers"] == 32 and
              pred["bytes"]["kv_cache"] > pred["bytes"]["params"] and
              "已经是主体" in pred["caption"],
              f'KV {pred["bytes"]["kv_cache"]:,} B vs 参数 {pred["bytes"]["params"]:,} B')
        check("预测面板明说这是预测值（不是 trace 跑过的配置）",
              "预测值" in pred["caption"], pred["caption"][:60])
        page.locator(".l05-predict").screenshot(path=SHOTS / "58-l05-predict-ledger.png")

        page.goto(f"{url}?cfg=kv-cache-N128-L4-fp16&step=5")
        page.wait_for_timeout(800)
        page.screenshot(path=SHOTS / "59-l05-full-light.png", full_page=True)

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
        print(f"{len(failures)} 项未通过：")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"全部通过（{len(notes)} 项测量）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
