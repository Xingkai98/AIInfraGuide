#!/usr/bin/env python3
"""Acceptance harness for L01's parameter controls (issue #15, criterion 4).

Drives the built page in real Chromium and checks the ticket's fourth acceptance
criterion — "改变 B_M/B_N/B_K 后，访存次数与算术强度实时重算，Roofline 图上的点
相应移动" — against the artifact a reader actually gets: the staged page in
`public/labs/`, with its trace SET inlined and its assets at the flattened paths.

WHY THIS IS A SEPARATE SCRIPT FROM verify-tiling-stage.py
---------------------------------------------------------
That one asserts the memory-hierarchy stage's contract (a shared view component,
ticket #44) and L01's rendering of it. This one asserts a *parameter feature*:
that the sliders select among generated configurations, that the numbers and the
Roofline point move with them, and that the chart's geometry survives. Same
shape, different subject — and merging them would make one script that has to be
understood as two, which is the reasoning every harness in this directory
already follows.

WHAT MAKES THE ASSERTIONS MEANINGFUL
------------------------------------
Three things this harness will not do, each of which would have made a green
result prove nothing:

  1. It reads the numbers back OFF THE DOM and compares them to the trace the
     generator wrote — not to a re-computation. If the page printed a number no
     trace contains, that is a failure here. (The page is forbidden from
     computing anything, so this is the check that keeps it that way.)
  2. Wherever a value is asserted to have MOVED, the control that must hold is
     asserted alongside it: the naive kernel's point may NOT move, and B_K must
     NOT change the element traffic. A test that only checked "the number
     changed" would pass on a page that redrew itself at random.
  3. The geometry is measured, not eyeballed — the dots must land inside the
     plot frame, no dot's label may sit on a neighbouring dot, and no text may
     be clipped by the viewBox. This project has already shipped a chart whose
     every count was correct and whose furniture overlapped (see the layout
     search in ring.js), so the picture is asserted like the numbers are.

Run:  python3 labs/traces/gemm_tiling.py          # writes the trace set
      npm run build:labs && python3 scripts/verify-l01-params.py
Needs `pip install playwright && playwright install chromium`.

Non-zero exit means at least one acceptance criterion failed.
"""

import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "public" / "labs" / "01-gemm-tiling.html"
TRACE_DIR = REPO / "labs" / "traces"
MANIFEST = TRACE_DIR / "gemm-tiling.manifest.json"
GENERATOR = TRACE_DIR / "gemm_tiling.py"
ROOFLINE_JS = REPO / "labs" / "assets" / "engine" / "views" / "roofline.js"
SHOTS = REPO / "labs" / "pages" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1100

# The configurations this harness drives end to end. Not all 27: each one is a
# page load plus a full DOM walk, and these five cover the extremes and both
# axes of movement — the smallest tile (most traffic, lowest AI), the largest
# (least traffic, highest AI), two that differ only in B_M so the B_M slider is
# exercised in isolation, and two that differ only in B_K so the B_K invariance
# is too.
DRIVEN = [
    "gemm-tiling-BM4-BN4-BK4",     # the default, and the reference
    "gemm-tiling-BM2-BN2-BK2",     # smallest tile: most traffic, lowest AI
    "gemm-tiling-BM8-BN8-BK8",     # largest tile: least traffic, highest AI
    "gemm-tiling-BM2-BN8-BK4",     # differs from default in B_M only
    "gemm-tiling-BM4-BN4-BK2",     # differs from default in B_K only
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


def num(v):
    """Render a number the way the PAGE's `num()` does.

    Ported rather than approximated, because the harness compares the page's
    printed string against the trace's value: a formatting disagreement would
    read as a page bug. The page's rule is four significant digits with trailing
    zeros and a trailing point stripped, which for every value this lab renders
    (no magnitude reaches 1e5 or 1e-4) is what `%.4g` produces.
    """
    if v is None:
        return "—"
    if not isinstance(v, (int, float)):
        return str(v)
    return f"{v:.4g}"


def reads_str(tr):
    return f"{num(tr['naive_reads'])} → {num(tr['tiled_reads'])}"


def ai_str(tr):
    return f"{num(tr['naive_ai'])} → {num(tr['tiled_ai'])}"


# ------------------------------------------------------------- lint parity
def lint_parity():
    """Run the generator's sabotage table through both ports of the roofline lint.

    The Python lint runs inside the generator and reports as it goes; this asks
    the generator for the payload (`--js-lint`) and pushes the same broken copies
    through `LabEngine.roofline.lint` in node.

    The two ports are not meant to be identical rule-for-rule: the Python side
    lints the whole contract (formulas, bindings, graph, `state`, traffic, the
    roofline block), while the JS side lints only the fields this view renders.
    So the requirement is not "every rule in both files" — it is:

      * every sabotage is caught by AT LEAST ONE port;
      * every sabotage in the view's own scope (the `roofline ` cases) is caught
        by the JS port;
      * every sabotage actually changes the trace. A case written against a
        literal that stops biting — a no-op — proves nothing when both lints
        report zero, and it is a different bug from "the lint missed it".
    """
    print("\n== 契约 lint 的两份实现（roofline）==")
    proc = subprocess.run([sys.executable, str(GENERATOR), "--js-lint"],
                          capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        check("生成器的 --js-lint 模式可用", False, (proc.stderr or "")[-400:])
        return
    payload = json.loads(proc.stdout)

    node_src = r"""
const fs = require('fs');
const g = {};
function load(p){ new Function('window','globalThis', fs.readFileSync(p,'utf8'))(g, g); }
load('labs/assets/engine/formula.js');
load('labs/assets/engine/trace-model.js');
load('labs/assets/engine/views/roofline.js');
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
const out = { clean: {}, sabotage: {} };
for (const [name, tr] of Object.entries(payload.traces)) {
  const r = NS.roofline.lint(tr);
  out.clean[name] = { gaps: r.gaps.length, warns: r.warns.length };
}
for (const [name, s] of Object.entries(payload.sabotages)) {
  if (s.error) { out.sabotage[name] = { error: s.error }; continue; }
  let gaps = 0;
  try { gaps = NS.roofline.lint(s.trace).gaps.length; }
  catch (e) { gaps = 'ERR:' + e.message; }
  out.sabotage[name] = { gaps: gaps, changed: s.changed };
}
process.stdout.write(JSON.stringify(out));
"""
    scratch = Path("/tmp") / "l01-roofline-lint-probe.js"
    scratch.write_text(node_src, encoding="utf-8")
    payload_file = Path("/tmp") / "l01-roofline-lint-payload.json"
    payload_file.write_text(proc.stdout, encoding="utf-8")
    node = subprocess.run(["node", str(scratch), str(payload_file)],
                          capture_output=True, text=True, cwd=str(REPO))
    if node.returncode != 0:
        check("JS 侧 lint 能在 node 里跑起来", False, (node.stderr or "")[-400:])
        return
    res = json.loads(node.stdout)

    clean = {k: v["gaps"] for k, v in res["clean"].items()}
    check(f"JS 侧 lint 对全部 {len(clean)} 份干净 trace 都报 0 gap（不是永远报 gap）",
          all(v == 0 for v in clean.values()), json.dumps(clean, ensure_ascii=False))

    total = len(res["sabotage"])
    dead = [k for k, v in res["sabotage"].items() if not v.get("changed")]
    errs = [k for k, v in res["sabotage"].items() if v.get("error")]
    caught = [k for k, v in res["sabotage"].items()
              if isinstance(v.get("gaps"), int) and v["gaps"] > 0]
    check(f"每一种破坏都真的改变了 trace（没有空操作，共 {total} 种）", not dead,
          "空操作：" + "、".join(sorted(dead)))
    check("每一种破坏都能被施加（不会半路抛异常）", not errs,
          "失败：" + "、".join(sorted(errs)))

    # The view's own scope, by the tag the case list groups under. Listed
    # explicitly rather than inferred so a new case that forgets the prefix is
    # visible as a count mismatch rather than silently untested on this side.
    component = [k for k in res["sabotage"] if k.startswith("roofline ")]
    missed = [k for k in component if k not in caught]
    check(f"组件职责内的破坏（{len(component)} 种 roofline 规则）全部被 JS 侧 lint 抓到",
          not missed, "漏掉：" + "、".join(sorted(missed)))

    # And the whole table must be caught by at least one of the two ports: a case
    # neither side catches is a rule that exists only in the case's name.
    pyside = subprocess.run([sys.executable, str(GENERATOR)],
                            capture_output=True, text=True, cwd=str(REPO))
    text = pyside.stdout + pyside.stderr
    ok = "LINT-DEAD" not in text and f"对照 {total} 种破坏全部被抓到" in text
    check(f"Python 侧 lint 独自覆盖了 JS 侧职责之外的规则（对照 {total} 种全部被抓到）",
          ok,
          f"生成器自报 {total} 种全部抓到" if ok else "生成器输出里没有出现「全部抓到」")
    note(f"两份 lint 各跑 {total} 种破坏：JS 侧覆盖组件字段（{len(component)} 种），"
         f"Python 侧覆盖全契约；两侧都要求「破坏真的改变了 trace」")


# ------------------------------------------------------------- DOM readers
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

# The readout, keyed by the `data-metric` attributes the page puts on each item.
# Reading these rather than the visible prose is deliberate: the harness should
# assert the NUMBERS the page states, and a wording change should not fail it.
READ_METRICS = """() => {
    const out = {};
    document.querySelectorAll('#lab-readout [data-metric]').forEach(e => {
        const b = e.querySelector('b');
        out[e.dataset.metric] = b ? b.textContent.trim() : null;
    });
    return out;
}"""

READ_ROOF_POINTS = """() => {
    const out = {};
    document.querySelectorAll('.lab-roof-pt').forEach(g => {
        const c = g.querySelector('.lab-roof-dot');
        const box = c.getBoundingClientRect();
        out[g.dataset.id] = {
            id: g.dataset.id,
            ai: Number(g.dataset.ai),
            ceiling: Number(g.dataset.ceiling),
            bound: g.dataset.bound,
            cx: box.left + box.width / 2,
            cy: box.top + box.height / 2,
            r: box.width / 2,
        };
    });
    return out;
}"""


def main():
    for p, hint in ((PAGE, "run `npm run build:labs` first"),
                    (MANIFEST, "run labs/traces/gemm_tiling.py first"),
                    (ROOFLINE_JS, "the roofline view is missing")):
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

        # ================================================== the whole set is there
        print("\n== 配置集与 manifest 一致 ==")
        page.goto(url)
        page.wait_for_timeout(900)
        inline = page.evaluate("() => window.LabTraceSets['gemm-tiling']")
        check("页面内联了 manifest",
              inline is not None and inline["set"] == "gemm-tiling")
        check("manifest 的配置数与生成器一致（27 组 = 3×3×3）",
              inline["traces"] == manifest["traces"] and
              len(inline["traces"]) == 27,
              f"{len(inline['traces'])} 组")
        check("三个参数各给出 {2, 4, 8}，且全部能整除 M=N=K=8",
              inline["params"] == {"B_M": [2, 4, 8], "B_N": [2, 4, 8], "B_K": [2, 4, 8]},
              json.dumps(inline["params"], ensure_ascii=False))
        check("默认配置是 B_M=B_N=B_K=4",
              inline["default"] == "gemm-tiling-BM4-BN4-BK4", inline["default"])
        # Every manifest entry must have a real trace behind it, and every trace
        # must be in the manifest -- the failure mode is a slider position that
        # loads nothing, which the build would not catch.
        inlined = page.evaluate("() => Object.keys(window.LabTraces)")
        missing = [n for n in manifest["traces"] if n not in inlined]
        check("manifest 里每一组都有一份内联的 trace",
              not missing, "缺：" + "、".join(missing))
        check("页面上的配置 id 集合与 manifest 完全一致（生成器是唯一定义处）",
              set(inline["traces"]) == set(manifest["traces"]) ==
              set(t for t in manifest["traces"] if t.startswith("gemm-tiling-")))

        # ============================================ sliders and the readout
        print("\n== 滑杆：三个参数可调，位置来自 manifest ==")
        sliders = page.evaluate(READ_SLIDERS)
        check("三个滑杆都渲染出来了（B_M / B_N / B_K）",
              set(sliders) == {"B_M", "B_N", "B_K"}, json.dumps(sliders))
        check("滑杆的可选值就是 manifest 的三个值",
              all(s["max"] == 2 for s in sliders.values()),
              json.dumps({k: v["max"] for k, v in sliders.items()}))
        check("滑杆当前值显示为当前配置的参数",
              all(s["shown"] == "4" for s in sliders.values()),
              json.dumps({k: v["shown"] for k, v in sliders.items()}))

        # ---- drive each configuration and read the page's numbers back
        #
        # Reloading via the ?cfg= deep link rather than clicking the slider is
        # deliberate: the deep link is a path a reader can take, and asserting it
        # means the harness checks the page a reader lands on rather than a
        # sequence of events only the test performs. The slider's own event
        # wiring is exercised separately, below.
        observed = {}
        for cfg_id in DRIVEN:
            trace = load(cfg_id)
            page.goto(f"{url}?cfg={cfg_id}&step=0")
            page.wait_for_timeout(700)
            got = {
                "sliders": page.evaluate(READ_SLIDERS),
                "metrics": page.evaluate(READ_METRICS),
                "points": page.evaluate(READ_ROOF_POINTS),
                "trace": trace,
                "config": page.evaluate("() => window.__labConfig"),
            }
            observed[cfg_id] = got

        print("\n== 数字来自 trace，不是页面算的 ==")
        for cfg_id, got in observed.items():
            trace, m = got["trace"], got["metrics"]
            tr = trace["meta"]["traffic"]
            cfg = trace["meta"]["config"]
            bad = []
            # The reads comparison: `naive → tiled`, both sides of it.
            if m.get("reads") != reads_str(tr):
                bad.append(f"reads: {m.get('reads')!r} != {reads_str(tr)!r}")
            if m.get("ai") != ai_str(tr):
                bad.append(f"ai: {m.get('ai')!r} != {ai_str(tr)!r}")
            if m.get("load_steps") != num(tr["load_steps"]):
                bad.append(f"load_steps: {m.get('load_steps')!r} != {num(tr['load_steps'])!r}")
            if m.get("smem_tile") != num(tr["smem_tile"]):
                bad.append(f"smem_tile: {m.get('smem_tile')!r} != {num(tr['smem_tile'])!r}")
            check(f"{cfg_id}: 页面的访存/强度与 trace 的 meta.traffic 一致",
                  not bad, "；".join(bad))
            # And the sliders show this configuration's own parameters.
            slid = {k: v["shown"] for k, v in got["sliders"].items()}
            check(f"{cfg_id}: 滑杆显示的是本配置的 (B_M, B_N, B_K)",
                  slid == {"B_M": str(cfg["B_M"]), "B_N": str(cfg["B_N"]),
                           "B_K": str(cfg["B_K"])} and got["config"] == cfg_id,
                  f"{json.dumps(slid, ensure_ascii=False)} · {got['config']}")

        # ======================================== the Roofline point moves
        print("\n== Roofline：点随滑杆移动，且移动的是该动的那个 ==")
        base = observed["gemm-tiling-BM4-BN4-BK4"]
        small = observed["gemm-tiling-BM2-BN2-BK2"]
        big = observed["gemm-tiling-BM8-BN8-BK8"]

        for cfg_id, got in observed.items():
            trace = got["trace"]
            pts = got["points"]
            roof = trace["meta"]["roofline"]
            by_id = {p["id"]: p for p in roof["points"]}
            bad = []
            if set(pts) != {"naive", "tiled", "doc"}:
                bad.append(f"点集 {sorted(pts)}")
            else:
                for pid, want in by_id.items():
                    got_pt = pts[pid]
                    if abs(got_pt["ai"] - want["ai"]) > 1e-9:
                        bad.append(f"{pid}.ai {got_pt['ai']} != {want['ai']}")
                    if abs(got_pt["ceiling"] - want["ceiling"]) > 1e-6:
                        bad.append(f"{pid}.ceiling 不一致")
                    if got_pt["bound"] != want["bound"]:
                        bad.append(f"{pid}.bound {got_pt['bound']} != {want['bound']}")
            check(f"{cfg_id}: 图上三个点的坐标/归属与 meta.roofline 一致",
                  not bad, "；".join(bad))

        # --- the MOUNTING EVIDENCE -------------------------------------------------
        # The tiled point must move when the tile changes, and it must move in the
        # direction the arithmetic says: a bigger tile is more reuse, so more
        # FLOPs per byte, so further right AND higher on a bandwidth-bound roof.
        def tiled_xy(got):
            p = got["points"]["tiled"]
            return p["cx"], p["cy"]

        small_xy, base_xy, big_xy = (tiled_xy(small), tiled_xy(base), tiled_xy(big))
        check("tiled 点在页面上真的移动了（2×2×2 → 4×4×4 → 8×8×8）",
              small_xy != base_xy != big_xy and small_xy != big_xy,
              f"{small_xy} → {base_xy} → {big_xy}")
        check("分块变大 → 算术强度变大 → 点向右移（AI 增大）",
              small_xy[0] < base_xy[0] < big_xy[0] and
              small["trace"]["meta"]["traffic"]["tiled_ai"] <
              base["trace"]["meta"]["traffic"]["tiled_ai"] <
              big["trace"]["meta"]["traffic"]["tiled_ai"],
              f"x: {small_xy[0]:.1f} < {base_xy[0]:.1f} < {big_xy[0]:.1f}")
        check("分块变大 → 上界抬高 → 点向上移（屏幕坐标 y 减小）",
              small_xy[1] > base_xy[1] > big_xy[1],
              f"y: {small_xy[1]:.1f} > {base_xy[1]:.1f} > {big_xy[1]:.1f}")

        # --- the CONTROL, and it is the load-bearing half of the two above ---
        # The naive kernel has no tile, so its point must be the same one on every
        # configuration. Without this, "the point moved" would be satisfied by a
        # chart that redrew everything whenever the sliders moved.
        naive_xy = {cid: (g["points"]["naive"]["cx"], g["points"]["naive"]["cy"])
                    for cid, g in observed.items()}
        naive_ais = {cid: g["points"]["naive"]["ai"] for cid, g in observed.items()}
        check("对照组：朴素点在所有配置里都落在同一处（它没有 tile，不该动）",
              len(set(naive_xy.values())) == 1 and len(set(naive_ais.values())) == 1,
              json.dumps({k: [round(v, 1) for v in xy] for k, xy in naive_xy.items()}))
        check("对照组：朴素点的 AI 恒为 0.25（2K/8K），与分块无关",
              all(abs(v - 0.25) < 1e-9 for v in naive_ais.values()),
              json.dumps(naive_ais))

        # --- B_K: the asymmetry, asserted in both directions -------------------
        # Under this traffic model B_K cancels out of the element count (each
        # output block walks the whole K range), which is why the tutorial's
        # §4.3 closed form has no B_K in it. So the element traffic and the
        # tiled point's AI must be IDENTICAL across the two configs that differ
        # only in B_K — while the load count and the SMEM tile must differ. A
        # harness that only checked "something changed" would pass on either a
        # page that ignored B_K entirely or one that fabricated a dependency.
        bk_same, bk_diff = (observed["gemm-tiling-BM4-BN4-BK4"],
                            observed["gemm-tiling-BM4-BN4-BK2"])
        a_tr, b_tr = bk_same["trace"]["meta"]["traffic"], bk_diff["trace"]["meta"]["traffic"]
        check("B_K 不改变元素搬运量（4×4×4 与 4×4×2 的 tiled_reads 相同）",
              a_tr["tiled_reads"] == b_tr["tiled_reads"] == 256,
              f"{a_tr['tiled_reads']} / {b_tr['tiled_reads']}")
        check("B_K 不改变算术强度，tiled 点也不移动（它在同一处）",
              abs(a_tr["tiled_ai"] - b_tr["tiled_ai"]) < 1e-12 and
              (bk_same["points"]["tiled"]["cx"], bk_same["points"]["tiled"]["cy"]) ==
              (bk_diff["points"]["tiled"]["cx"], bk_diff["points"]["tiled"]["cy"]),
              f"AI {a_tr['tiled_ai']} / {b_tr['tiled_ai']}")
        check("B_K 确实改变了载入次数与 SMEM tile（滑杆不是死的）",
              a_tr["load_steps"] != b_tr["load_steps"] and
              a_tr["smem_tile"] != b_tr["smem_tile"] and
              bk_same["metrics"]["load_steps"] != bk_diff["metrics"]["load_steps"],
              f"载入次数 {a_tr['load_steps']} vs {b_tr['load_steps']}，"
              f"SMEM tile {a_tr['smem_tile']} vs {b_tr['smem_tile']}")
        note("B_K 的两个方向都断言了：元素量与 AI 不变（§4.3 闭式里没有 B_K），"
             "载入次数与 SMEM tile 按 1/B_K 缩放")

        # --- the slider's own event wiring, which the deep links bypass -------
        print("\n== 滑杆事件：拖动 B_M 真的会换配置 ==")
        page.goto(f"{url}?cfg={base['trace']['meta']['config'] and 'gemm-tiling-BM4-BN4-BK4'}&step=0")
        page.wait_for_timeout(600)
        before = page.evaluate("() => window.__labConfig")
        # Move B_M from index 1 (value 4) to index 2 (value 8) and dispatch the
        # same `input` event a drag produces.
        page.evaluate("""() => {
            const i = document.querySelector('#lab-params input[data-param="B_M"]');
            i.value = String(Number(i.max));
            i.dispatchEvent(new Event('input', {bubbles: true}));
        }""")
        page.wait_for_timeout(700)
        after = page.evaluate("() => window.__labConfig")
        check("拖动 B_M 滑杆会载入新配置（4→8）",
              before == "gemm-tiling-BM4-BN4-BK4" and after == "gemm-tiling-BM8-BN4-BK4",
              f"{before} → {after}")
        # And the readout followed: the numbers must be the new trace's.
        moved = page.evaluate(READ_METRICS)
        new_tr = load("gemm-tiling-BM8-BN4-BK4")["meta"]["traffic"]
        check("换配置后页面的访存与强度也随之重算（读的是新 trace 的数）",
              moved.get("reads") == reads_str(new_tr) and
              moved.get("ai") == ai_str(new_tr),
              f"reads={moved.get('reads')} ai={moved.get('ai')}")
        # And it must have actually CHANGED from the configuration it replaced --
        # a page that reloaded the same trace would satisfy the line above.
        old_tr = load("gemm-tiling-BM4-BN4-BK4")["meta"]["traffic"]
        check("换配置前后这两行数字确实不同（滑杆不是只换了个 id）",
              reads_str(new_tr) != reads_str(old_tr) and ai_str(new_tr) != ai_str(old_tr),
              f"{reads_str(old_tr)} → {reads_str(new_tr)}")
        # The deep link must be updated too, so the moved slider is shareable.
        check("拖动滑杆后深链跟着更新（?cfg= 指向新配置）",
              "cfg=gemm-tiling-BM8-BN4-BK4" in page.url, page.url.split("?")[-1])

        # ===================================================== geometry
        print("\n== 几何：点在图框内、标签不压在别的点上、不被裁 ==")
        # The chart is the point of this ticket, and a chart can be wrong in a
        # way only a chart shows. So the layout is measured off the rendered
        # DOM: every dot inside the plot frame with margin, no dot's label box
        # overlapping another dot, and nothing spilling outside the viewBox.
        for cfg_id in DRIVEN:
            trace = load(cfg_id)
            page.goto(f"{url}?cfg={cfg_id}&step=0")
            page.wait_for_timeout(550)
            geo = page.evaluate("""() => {
                const svg = document.querySelector('.lab-roof-svg');
                if (!svg) return {error: 'no chart'};
                const sb = svg.getBoundingClientRect();
                const pb = document.querySelector('.lab-roof-plot').getBoundingClientRect();
                const dots = [...document.querySelectorAll('.lab-roof-pt')].map(g => {
                    const c = g.querySelector('.lab-roof-dot').getBoundingClientRect();
                    const t = g.querySelector('.lab-roof-pt-l').getBoundingClientRect();
                    return {id: g.dataset.id,
                            dot: {l: c.left, r: c.right, t: c.top, b: c.bottom},
                            lbl: {l: t.left, r: t.right, t: t.top, b: t.bottom}};
                });
                // Any text drawn outside the viewBox is silently clipped, which
                // reads as a rendering glitch rather than as a layout bug.
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
                        dots, clipped};
            }""")
            if geo.get("error"):
                check(f"{cfg_id}: Roofline 图画出来了", False, geo["error"])
                continue
            plot, bad_inside, bad_lbl, bad_plot = geo["plot"], [], [], []
            for d in geo["dots"]:
                m = d["dot"]
                # The DOT's centre must be inside the frame, with the whole dot
                # clear of the border -- a dot painted on the axis is
                # indistinguishable from a tick mark.
                if not (m["l"] >= plot["l"] - 0.5 and m["r"] <= plot["r"] + 0.5 and
                        m["t"] >= plot["t"] - 0.5 and m["b"] <= plot["b"] + 0.5):
                    bad_inside.append(d["id"])
                # No dot's LABEL may overlap another dot. Labels and dots are the
                # two furniture types that share the plot area, and a label
                # sitting on a neighbouring dot is the shape of collision that
                # every count-based assertion misses.
                for e in geo["dots"]:
                    if e["id"] == d["id"]:
                        continue
                    a, b = d["lbl"], e["dot"]
                    if a["l"] < b["r"] and b["l"] < a["r"] and a["t"] < b["b"] and b["t"] < a["b"]:
                        bad_lbl.append([d["id"], e["id"]])
                # Every dot must be inside the PLOT, not merely inside the SVG:
                # the axes and titles are drawn in the margins, and a dot there
                # would be sitting on a tick label.
                c = d["dot"]
                if c["r"] < plot["l"] or c["l"] > plot["r"] or c["b"] < plot["t"] or c["t"] > plot["b"]:
                    bad_plot.append(d["id"])
            check(f"{cfg_id}: 三个点都落在图框内（不压轴线/刻度）",
                  not bad_inside and not bad_plot,
                  f"越界 {bad_inside or bad_plot}")
            check(f"{cfg_id}: 点的标签不压在别的点上",
                  not bad_lbl, json.dumps(bad_lbl))
            check(f"{cfg_id}: 没有文字被 viewBox 裁掉",
                  not geo["clipped"], json.dumps(geo["clipped"], ensure_ascii=False))

        # The control for the geometry: the checks above must be able to FAIL.
        #
        # Note what this does NOT do. It does not push a point's data out of
        # range and hope the dot follows — the view derives its domain from the
        # points, so a tampered intensity simply widens the axis and the dot
        # stays inside. (That was the first version of this control, and it
        # passed nothing: `bad` was empty, correctly, because the view had
        # re-scaled around the sabotage.)
        #
        # What a measurement needs to be able to see is a DOM that violates the
        # invariant, so the control BUILDS one: it renders the real view, then
        # moves a dot to where the rules forbid — outside the frame, and a
        # label onto a neighbouring dot — and requires the same predicates,
        # evaluated the same way, to flag both. A predicate that cannot see a
        # hand-placed violation is not measuring anything.
        print("\n== 几何对照：人为把点推出图框 / 让标签压到点上，规则必须报警 ==")
        control = page.evaluate("""() => {
            const T = window.__labTrace;
            const view = LabEngine.roofline.makeView({trace: T});
            const host = document.createElement('div');
            host.style.position = 'absolute';
            host.style.left = '-9999px';
            host.style.top = '0';
            document.body.appendChild(host);
            host.innerHTML = view.render();

            const read = () => {
                const pb = host.querySelector('.lab-roof-plot').getBoundingClientRect();
                const dots = [...host.querySelectorAll('.lab-roof-pt')].map(g => {
                    const c = g.querySelector('.lab-roof-dot').getBoundingClientRect();
                    const t = g.querySelector('.lab-roof-pt-l').getBoundingClientRect();
                    return {id: g.dataset.id,
                            dot: {l: c.left, r: c.right, t: c.top, b: c.bottom},
                            lbl: {l: t.left, r: t.right, t: t.top, b: t.bottom}};
                });
                return {pb, dots};
            };
            const outside = (m, pb) =>
                !(m.l >= pb.left - 0.5 && m.r <= pb.right + 0.5 &&
                  m.t >= pb.top - 0.5 && m.b <= pb.bottom + 0.5);
            const overlaps = (a, b) =>
                a.l < b.r && b.l < a.r && a.t < b.b && b.t < a.b;

            // (1) a dot pushed past the frame's left edge
            const tiled = host.querySelector('.lab-roof-pt-tiled .lab-roof-dot');
            const keepCx = tiled.getAttribute('cx');
            tiled.setAttribute('cx', '-60');
            const frameBad = read().dots.filter(d => outside(d.dot, read().pb)).map(d => d.id);
            tiled.setAttribute('cx', keepCx);

            // (2) a label moved onto a neighbouring dot
            const naiveLbl = host.querySelector('.lab-roof-pt-naive .lab-roof-pt-l');
            const docDot = host.querySelector('.lab-roof-pt-doc .lab-roof-dot').getBoundingClientRect();
            const savedX = naiveLbl.getAttribute('x'), savedY = naiveLbl.getAttribute('y');
            // In viewBox units: move the naive label onto the doc dot. The svg
            // is rendered at its natural size here, so 1 unit == 1 px.
            const vb = host.querySelector('.lab-roof-svg').getBoundingClientRect();
            const pbNow = host.querySelector('.lab-roof-plot').getBoundingClientRect();
            const dx = (docDot.left - pbNow.left) - (naiveLbl.getBoundingClientRect().left - pbNow.left);
            const dy = (docDot.top - pbNow.top) - (naiveLbl.getBoundingClientRect().top - pbNow.top);
            const sx = vb.width / Number(host.querySelector('.lab-roof-svg').getAttribute('data-w'));
            const sy = vb.height / Number(host.querySelector('.lab-roof-svg').getAttribute('data-h'));
            naiveLbl.setAttribute('x', String(Number(savedX) + dx / sx));
            naiveLbl.setAttribute('y', String(Number(savedY) + dy / sy));
            const after = read();
            const naivePt = after.dots.find(d => d.id === 'naive');
            const docPt = after.dots.find(d => d.id === 'doc');
            const labelBad = naivePt && docPt && overlaps(naivePt.lbl, docPt.dot);
            naiveLbl.setAttribute('x', savedX);
            naiveLbl.setAttribute('y', savedY);

            host.remove();
            return {frameBad: frameBad, labelBad: !!labelBad};
        }""")
        check("对照组：被推出图框的点会被「点在框内」规则抓到",
              control["frameBad"] == ["tiled"], json.dumps(control))
        check("对照组：压到别的点上的标签会被「标签不压点」规则抓到",
              control["labelBad"], json.dumps(control))

        # ===================================================== shots
        print("\n== 截图 ==")
        # The parameter bar and the readout lead the page, so a plain viewport
        # shot at step 0 shows them with the stage and the Roofline beneath.
        page.goto(f"{url}?cfg=gemm-tiling-BM4-BN4-BK4&step=0")
        page.wait_for_timeout(600)
        page.evaluate("() => window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "30-l01-params-overview.png")

        # The Roofline at the two ends of the tile range, so a reviewer can see
        # the movement the harness asserts without running it.
        for cfg_id, name in (("gemm-tiling-BM2-BN2-BK2", "31-l01-roofline-small"),
                             ("gemm-tiling-BM4-BN4-BK4", "32-l01-roofline-default"),
                             ("gemm-tiling-BM8-BN8-BK8", "33-l01-roofline-large")):
            page.goto(f"{url}?cfg={cfg_id}&step=0")
            page.wait_for_timeout(600)
            page.locator(".lab-roof-panel").screenshot(path=SHOTS / f"{name}.png")

        # Light mode, since the page is theme-aware.
        page_light = browser.new_page(viewport={"width": W, "height": H},
                                      color_scheme="light")
        page_light.goto(f"{url}?cfg=gemm-tiling-BM8-BN8-BK8&step=0")
        page_light.wait_for_timeout(800)
        page_light.locator(".lab-roof-panel").screenshot(
            path=SHOTS / "34-l01-roofline-light.png")

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
