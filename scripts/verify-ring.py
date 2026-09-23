#!/usr/bin/env python3
"""Acceptance harness for the ring view (issue #47) and L13.

Drives the built page in real Chromium and checks the ticket's acceptance
criteria one by one against the artifact a reader actually gets — the staged
page in `public/labs/`, with its trace set inlined and its assets at the
flattened paths. Then it takes the screenshots, which are committed so a
reviewer can see the result without running this.

It is deliberately a separate script from verify-labs.py (the engine's harness),
verify-tiling-stage.py and verify-ledger.py (the other two views') and
verify-gantt.py (L10's), for the same reason those are separate from each
other: this one asserts a *view component*'s contract plus L13's rendering, and
merging them would make one script that has to be understood as five.

It has one job the others do not, beyond running the sabotage table through both
lint ports (which verify-gantt.py established): this component's honesty check
runs at TWO levels, and the levels disagree by design. The engine's generic
`verify.runJumpCheck` folds TENSOR STATE, and every step of a ring rewrites
every rank's buffer in full — so its stateful control group lands on the right
answer and has no discriminating power here. This harness asserts that claim
directly (section "两层的对照组"), rather than accepting it as a comment: if the
engine's control group ever DOES start catching things on this trace, the
reasoning behind the view's own check would need revisiting and this is where
that shows up.

Run:  python3 labs/traces/ring_allreduce.py          # writes the trace set
      npm run build:labs && python3 scripts/verify-ring.py
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
PAGE = REPO / "public" / "labs" / "03-ring-allreduce.html"
TRACE_DIR = REPO / "labs" / "traces"
MANIFEST = TRACE_DIR / "ring-allreduce.manifest.json"
GENERATOR = TRACE_DIR / "ring_allreduce.py"
RING_JS = REPO / "labs" / "assets" / "engine" / "views" / "ring.js"
SHOTS = REPO / "labs" / "pages" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1050
N_VALUES = [2, 4, 8]

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


def load(n):
    return json.loads((TRACE_DIR / f"ring-allreduce-N{n}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- lint parity
def lint_parity():
    """Run the generator's sabotage table through both contract lints.

    The Python lint runs inside the generator and reports as it goes; this asks
    the generator for the payload (--js-lint) and pushes the same broken copies
    through `LabEngine.ring.lint` in node. The verdicts must agree.

    The two ports are not identical by accident and are not meant to be: the
    Python side lints the whole ring contract, the JS side the same rules. So
    the requirement is "every sabotage is caught by the JS port, and every
    sabotage actually changes the trace on both sides" — the second half is what
    catches a case written against a literal that stops biting at some N.
    """
    print("\n== 契约 lint 的两份实现 ==")
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
load('labs/assets/engine/trace-model.js');
load('labs/assets/engine/formula.js');
load('labs/assets/engine/views/ring.js');
const NS = g.LabEngine;
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = { clean: {}, sabotage: {} };
for (const [name, tr] of Object.entries(payload.traces)) {
  const r = NS.ring.lint(tr);
  out.clean[name] = { gaps: r.gaps.length, warns: r.warns.length };
}
for (const [name, s] of Object.entries(payload.sabotages)) {
  if (s.error) { out.sabotage[name] = { error: s.error }; continue; }
  const r = NS.ring.lint(s.trace);
  // A mutation that left the trace byte-identical is a no-op, and a no-op
  // proves nothing on either side. Reported separately from "the lint missed
  // it", because they are different bugs.
  out.sabotage[name] = {
    gaps: r.gaps.length,
    changed: JSON.stringify(s.trace) !== JSON.stringify(payload.traces[payload.base])
  };
}
process.stdout.write(JSON.stringify(out));
"""
    scratch = Path("/tmp") / "ring-lint-probe.js"
    scratch.write_text(node_src, encoding="utf-8")
    payload_file = Path("/tmp") / "ring-lint-payload.json"
    payload_file.write_text(proc.stdout, encoding="utf-8")
    node = subprocess.run(["node", str(scratch), str(payload_file)],
                          capture_output=True, text=True, cwd=str(REPO))
    if node.returncode != 0:
        check("JS 侧 lint 能在 node 里跑起来", False, node.stderr[-400:])
        return
    res = json.loads(node.stdout)

    clean_gaps = {k: v["gaps"] for k, v in res["clean"].items()}
    check("JS 侧 lint 对三份干净 trace 都报 0 gap（不是永远报 gap）",
          all(v == 0 for v in clean_gaps.values()),
          json.dumps(clean_gaps, ensure_ascii=False))

    total = len(res["sabotage"])
    dead = [k for k, v in res["sabotage"].items()
            if "gaps" in v and not v["changed"]]
    caught = [k for k, v in res["sabotage"].items() if v.get("gaps", 0) > 0]
    check(f"JS 侧每一种破坏都真的改变了 trace（没有空操作，共 {total} 种）",
          not dead, "空操作：" + "、".join(sorted(dead)))
    check(f"JS 侧 lint 抓到全部 {total} 种破坏",
          len(caught) == total,
          "漏掉：" + "、".join(sorted(set(res["sabotage"]) - set(caught))))

    py = subprocess.run([sys.executable, str(GENERATOR)],
                        capture_output=True, text=True, cwd=str(REPO))
    text = py.stdout + py.stderr
    # `check`'s detail prints unconditionally, so it has to describe the
    # outcome. The failure text is passed only when the check fails — an
    # always-failure-phrased detail reads as a contradiction on a PASS line.
    ok = ("LINT-DEAD" not in text and "SABOTAGE-DEAD" not in text and
          f"对照 {total} 种破坏" in text)
    check(f"Python 侧 lint 也抓到全部 {total} 种破坏（生成器自己报的）", ok,
          f"生成器自报 {total} 种全部抓到" if ok else "生成器输出里没有出现「全部被抓到」")
    note(f"两份 lint 各跑 {total} 种破坏，两侧都要求「破坏真的改变了 trace」且被抓到")


def main():
    for p, hint in ((PAGE, "run `npm run build:labs` first"),
                    (MANIFEST, "run labs/traces/ring_allreduce.py first")):
        if not p.exists():
            print(f"missing {p} — {hint}", file=sys.stderr)
            return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    traces = {n: load(n) for n in N_VALUES}
    url = PAGE.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        page.goto(url)
        page.wait_for_timeout(1400)

        # ================================================ configurable N
        print("\n== 卡数可配置（2 / 4 / 8）==")
        default_n = page.evaluate("() => +document.querySelector('.lab-ring').dataset.n")
        check("默认配置是 4 卡（与 manifest 的 default 一致）",
              default_n == 4 and manifest["default"] == "ring-allreduce-N4",
              f"n={default_n}, default={manifest['default']}")
        check("拓扑节点数 == 卡数，弧数 == 卡数（一个 N 节点的有向环）",
              page.evaluate("() => document.querySelectorAll('.lab-ring-node').length") == 4
              and page.evaluate("() => document.querySelectorAll('.lab-ring-arc').length") == 4)

        switched = page.evaluate("""() => {
            const P = window.__p;
            document.querySelector('[data-n="8"]').click();
            return null;
        }""")
        page.wait_for_timeout(900)
        n8 = page.evaluate("""() => ({
            n: +document.querySelector('.lab-ring').dataset.n,
            nodes: document.querySelectorAll('.lab-ring-node').length,
            arcs: document.querySelectorAll('.lab-ring-arc').length,
            cells: document.querySelectorAll('.lab-ring-cell').length,
            ticks: document.querySelectorAll('.lab-ring-tl-tick').length,
            last: window.__p.index.lastStep
        })""")
        check("切到 8 卡后换的是另一份 trace（节点 / 弧 / 格子数都变了）",
              n8["n"] == 8 and n8["nodes"] == 8 and n8["arcs"] == 8
              and n8["cells"] == 64 and n8["last"] == len(traces[8]["steps"]) - 1,
              json.dumps(n8, ensure_ascii=False))
        check("时间轴步数 == 2(N−1)：8 卡 14 步、4 卡 6 步、2 卡 2 步",
              n8["ticks"] == 14 and
              len(traces[4]["steps"]) == 7 and len(traces[2]["steps"]) == 3)
        void = switched

        # ============================================ ring <-> buffer sync
        print("\n== 环形拓扑与每卡缓冲区状态同步更新 ==")
        sync = page.evaluate("""() => {
            const P = window.__p;
            const T = window.__labTrace;
            const ranks = T.ring.ranks.map(r => r.id);
            const out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                // Read the DOM, not the component's own fold: this is the check
                // that what was PAINTED is right, not that the fold is
                // self-consistent.
                const cells = {};
                document.querySelectorAll('.lab-ring-cell').forEach(g => {
                    const rk = g.dataset.rank, ch = +g.dataset.chunk;
                    cells[rk + ':' + ch] = +g.dataset.count;
                });
                const table = {};
                document.querySelectorAll('.lab-ring-table tbody tr').forEach(tr => {
                    const rk = tr.dataset.rank;
                    tr.querySelectorAll('.lab-ring-td-cell').forEach((td, j) => {
                        table[rk + ':' + j] = +td.dataset.count;
                    });
                });
                const lit = document.querySelector('.lab-ring-tl-on');
                out.push({i, cells, table,
                          live: document.querySelectorAll('.lab-ring-arc-live').length,
                          nodes: ranks.length,
                          tlOn: lit ? +lit.dataset.step : null});
            }
            P.setCursor(0);
            return out;
        }""")
        t8 = traces[8]
        # Recompute the expected counts here, in Python, from the trace's own
        # transfer list — a third independent derivation, in a third language.
        n = 8
        rids = [r["id"] for r in t8["ring"]["ranks"]]
        idx_of = {r: i for i, r in enumerate(rids)}
        chunks = t8["ring"]["chunks"]
        want_counts = []
        grid = [[1] * chunks for _ in range(n)]
        for s in t8["steps"]:
            for x in s.get("transfers", []):
                a, b = idx_of[x["from"]], idx_of[x["to"]]
                grid[b][x["chunk"]] = (grid[a][x["chunk"]]
                                       if x.get("mode") == "copy"
                                       else grid[b][x["chunk"]] + grid[a][x["chunk"]])
            want_counts.append([row[:] for row in grid])

        mismatch = []
        for row in sync:
            w = want_counts[row["i"]]
            for k in range(n):
                for c in range(chunks):
                    key = f"{rids[k]}:{c}"
                    if row["cells"].get(key) != w[k][c]:
                        mismatch.append((row["i"], key, row["cells"].get(key), w[k][c]))
                    if row["table"].get(key) != w[k][c]:
                        mismatch.append((row["i"], key, row["table"].get(key), w[k][c]))
        check("每一步：拓扑上的格子、缓冲区表、以及从传输列表独立重算的计数三者完全一致",
              not mismatch, f"不一致 {len(mismatch)} 处：" + str(mismatch[:4]))
        check("每一步所有 N 条链路都在传（环形最核心的那句话）",
              all(r["live"] == 8 for r in sync[1:]),
              json.dumps([r["live"] for r in sync]))
        # The prepare frame (step 0) moves nothing, so no tick is lit — which is
        # the honest reading, not a missing highlight. Every later step lights
        # exactly the tick whose data-step IS the cursor, so the strip and the
        # player cannot drift apart by an off-by-one.
        check("时间轴：通信步点亮的那一格 data-step 恒等于游标；准备帧不点亮任何格",
              sync[0]["tlOn"] is None and
              all(r["tlOn"] == r["i"] for r in sync[1:]),
              json.dumps([(r["i"], r["tlOn"]) for r in sync
                          if r["i"] > 0 and r["tlOn"] != r["i"]][:5]))

        # The end state, asserted on the DOM: every cell complete.
        end = sync[-1]
        check("末步每张卡的每一块都是 N/N（AllReduce 的契约）",
              all(end["cells"].get(f"{rids[k]}:{c}") == 8
                  for k in range(8) for c in range(chunks)))
        # And the midpoint: after ReduceScatter each rank owns exactly one
        # complete chunk, and it is a DIFFERENT one per rank.
        rs_last = sync[n - 1]
        done = {}
        for k in range(8):
            done[rids[k]] = [c for c in range(chunks)
                             if rs_last["cells"].get(f"{rids[k]}:{c}") == 8]
        check("ReduceScatter 结束（第 N−1 步）时每张卡恰好有 1 块完成，且各卡不同",
              all(len(v) == 1 for v in done.values()) and
              len({v[0] for v in done.values()}) == 8,
              json.dumps(done, ensure_ascii=False))

        # The topology is a picture, and a picture can be WRONG in a way no
        # count can detect: overlapping furniture. The first version of this
        # view drew every rank's buffer strip to the right of its node, so at
        # N=8 the left-hand ranks drew their strips straight through the ones
        # beside them — obvious in a screenshot, invisible to every assertion
        # above. So the geometry is asserted, not eyeballed: no two buffer
        # strips may intersect, and none may cover a rank's disc.
        overlap = page.evaluate("""() => {
            const boxes = [...document.querySelectorAll('.lab-ring-buf')].map(g => {
                const b = g.getBoundingClientRect();
                return {rank: g.dataset.rank, l: b.left, t: b.top,
                        r: b.right, b: b.bottom};
            });
            // The WHOLE node group, not just the inner disc: the group includes
            // the progress ring a few px outside it, and measuring only the
            // disc is exactly the looseness that let the first overlap through.
            const nodes = [...document.querySelectorAll('.lab-ring-node')].map(g => {
                const b = g.getBoundingClientRect();
                return {rank: g.dataset.rank, l: b.left, t: b.top,
                        r: b.right, b: b.bottom};
            });
            const svgB = document.querySelector('.lab-ring-svg')
                .getBoundingClientRect();
            const hits = [];
            for (let i = 0; i < boxes.length; i++) {
                for (let j = i + 1; j < boxes.length; j++) {
                    const a = boxes[i], c = boxes[j];
                    if (a.l < c.r && c.l < a.r && a.t < c.b && c.t < a.b) {
                        hits.push([a.rank, c.rank]);
                    }
                }
            }
            const onNodes = [];
            boxes.forEach(a => nodes.forEach(nd => {
                if (a.l < nd.r && nd.l < a.r && a.t < nd.b && nd.t < a.b) {
                    onNodes.push([a.rank, nd.rank]);
                }
            }));
            // And nothing may be cut off by the canvas: a strip drawn past the
            // viewBox is silently clipped, which reads as a rendering glitch
            // rather than as a layout bug.
            const clipped = boxes.filter(a =>
                a.l < svgB.left - 0.5 || a.r > svgB.right + 0.5 ||
                a.t < svgB.top - 0.5 || a.b > svgB.bottom + 0.5).map(a => a.rank);
            // No SVG text may sit on a strip UNLESS it belongs to it. The cell
            // digits are inside their own strip by construction; anything else
            // is a label that drifted. This is the check that caught the
            // per-rank caption this view used to draw — its box landed on the
            // strips two ranks over at N=8, and an SVG text box is sized by
            // whatever font the browser resolves, which no layout search can
            // predict reliably.
            const textHits = [];
            [...document.querySelectorAll('.lab-ring-svg text')].forEach(t => {
                if (t.closest('.lab-ring-buf')) return;      // its own strip
                const b = t.getBoundingClientRect();
                boxes.forEach(a => {
                    if (b.left < a.r && a.l < b.right && b.top < a.b && a.t < b.bottom) {
                        textHits.push([t.textContent, a.rank]);
                    }
                });
            });
            return {n: boxes.length, overlaps: hits, covers: onNodes,
                    clipped: clipped, textHits: textHits};
        }""")
        check(f"N=8 的 {overlap['n']} 条缓冲区条带两两不重叠、不压住任何一张卡、"
              "不被画布裁掉、也没有外来文字压在条带上",
              not overlap["overlaps"] and not overlap["covers"]
              and not overlap["clipped"] and not overlap["textHits"],
              json.dumps(overlap, ensure_ascii=False))

        # ======================================== two phases, single-steppable
        print("\n== ReduceScatter 与 AllGather 两阶段逐步可看 ==")
        phases = page.evaluate("""() => {
            const P = window.__p;
            const out = [];
            for (let i = 1; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                const on = document.querySelector('.lab-ring-tl-on');
                if (!on) continue;
                out.push({i,
                    phase: on.closest('.lab-ring-tl-phase').dataset.phase,
                    label: on.closest('.lab-ring-tl-phase')
                             .querySelector('.lab-ring-tl-hd').textContent});
            }
            P.setCursor(0);
            return out;
        }""")
        check("时间轴把 2(N−1) 步分成两个阶段，各 N−1 步",
              len(phases) == 2 * (n - 1) and
              sum(1 for x in phases if x["phase"] == "reduce") == n - 1 and
              sum(1 for x in phases if x["phase"] == "gather") == n - 1,
              json.dumps([x["phase"] for x in phases]))
        check("两阶段的标签分别写着 ReduceScatter / AllGather",
              all("ReduceScatter" in x["label"] for x in phases if x["phase"] == "reduce")
              and all("AllGather" in x["label"] for x in phases
                      if x["phase"] == "gather"))

        # Each step's own narration and formula must be that step's, and the
        # formula's slots must be filled from that step's bindings — a panel
        # that showed step 3's formula at step 9 would be a silent lie.
        formulas = page.evaluate("""() => {
            const P = window.__p, out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                const num = document.querySelector('.lab-formula-host .katex');
                out.push({i, has: !!num,
                          text: num ? num.textContent.slice(0, 40) : null,
                          title: document.querySelector('.lab-narration-h').textContent});
            }
            P.setCursor(0);
            return out;
        }""")
        check("每一步都渲染出公式（不是空面板）",
              all(x["has"] for x in formulas))
        check("每一步的标题都是那一步自己的（逐步不同）",
              len({x["title"] for x in formulas}) >= n,
              f"{len({x['title'] for x in formulas})} 个不同标题 / {len(formulas)} 步")

        # ================================================ traffic accumulator
        print("\n== 通信量累计器：Ring 与朴素两条对比曲线，差距可量化 ==")
        acc = page.evaluate("""() => {
            const P = window.__p;
            const at = (i) => {
                P.setCursor(i);
                const el = document.querySelector('.lab-ring-acc');
                const lines = [...document.querySelectorAll('.lab-ring-acc-line')]
                    .map(l => l.getAttribute('class'));
                return {i, max: +el.dataset.max, at: +el.dataset.at,
                        note: document.querySelector('.lab-ring-acc-note').textContent,
                        keys: [...document.querySelectorAll('.lab-ring-acc-key')]
                              .map(k => k.textContent)};
            };
            const out = {start: at(0), mid: at(7), end: at(P.index.lastStep)};
            P.setCursor(0);
            return out;
        }""")
        check("累计器画出三条曲线：Ring / 朴素中心化 / Tree",
              len(acc["end"]["keys"]) == 3 and
              any("Ring" in k for k in acc["end"]["keys"]) and
              any("朴素" in k for k in acc["end"]["keys"]) and
              any("Tree" in k for k in acc["end"]["keys"]),
              json.dumps(acc["end"]["keys"], ensure_ascii=False))
        t = traces[8]
        check("曲线随游标增长（终点的读数大于起点）",
              acc["end"]["max"] >= acc["mid"]["max"] > 0)
        # The quantified gap, read off the rendered note and checked against the
        # trace's own arithmetic.
        ratio = t["meta"]["metrics"]["naive_over_ring"]
        check(f"图上写出的倍数 == trace 里的朴素/环形比（{ratio}×，即 N）",
              f"{ratio:.2f}×" in acc["end"]["note"],
              acc["end"]["note"][:120])
        cfg = t["meta"]["config"]
        tr = t["meta"]["traffic"]
        # The premise the whole comparison rests on, asserted before the ratio:
        # all three topologies move the SAME total. What differs is only how
        # many links that total is spread over, and `load × links == total` is
        # the identity that says so for each of them.
        agg = 2 * (8 - 1) * cfg["S"] * cfg["item_bytes"]
        same_total = all(tr[k]["link_load_bytes"] * tr[k]["links"] ==
                         tr["aggregate_bytes"] for k in ("ring", "naive", "tree"))
        check("三条调度的总搬运量相同（对照的前提：比的是摊薄，不是工作量）",
              tr["aggregate_bytes"] == agg and same_total,
              f'总量 {tr["aggregate_bytes"]} B = 2(N−1)S；'
              + "；".join(f'{k}: {tr[k]["link_load_bytes"]}×{tr[k]["links"]}'
                          for k in ("ring", "naive", "tree")))
        check("单链路负载差 N 倍：朴素 == 环形 × N，Tree == 2S",
              tr["naive"]["link_load_bytes"] == tr["ring"]["link_load_bytes"] * 8 and
              tr["tree"]["link_load_bytes"] == 2 * cfg["S"] * cfg["item_bytes"],
              f'ring {tr["ring"]["link_load_bytes"]} / naive '
              f'{tr["naive"]["link_load_bytes"]} / tree {tr["tree"]["link_load_bytes"]}')
        check("树拓扑确实画出来了（对照拓扑的节点与边）",
              page.evaluate("() => document.querySelectorAll('.lab-ring-tnode').length") == 8
              and page.evaluate("() => document.querySelectorAll('.lab-ring-tedge').length") == 7)

        # ================================================ jump purity + control
        print("\n== 任意跳转：纯函数重建，且对照组确实会翻车（组件层）==")
        self_check = page.evaluate("""() => {
            const T = window.__labTrace;
            const r = LabEngine.ring.check(T);
            return {jumps: r.jumps.total, pure: r.jumps.pureFailures,
                    ctl: r.jumps.controlFailures, conclusive: r.jumps.conclusive,
                    passed: r.passed,
                    lintGaps: r.lint.gaps.length,
                    missed: r.sabotage.missed, dead: r.sabotage.dead};
        }""")
        check("组件自检：纯函数折叠在每一次跳转上都与顺序播放一致",
              self_check["pure"] == 0, f"{self_check['jumps']} 次跳转")
        check("组件自检：对照组（有状态、只往后累积的折叠）确实在倒退跳转上翻车",
              self_check["ctl"] > 0 and self_check["conclusive"],
              f"{self_check['ctl']} / {self_check['jumps']} 次不一致")
        check("组件自检整体通过（且 lint 干净、破坏表无人漏网、无空操作）",
              self_check["passed"] and self_check["lintGaps"] == 0 and
              not self_check["missed"] and not self_check["dead"],
              json.dumps(self_check, ensure_ascii=False))

        # The same claim, driven through the real player rather than the API:
        # jump by jump, compare the rendered panel against having played there.
        purity = page.evaluate("""() => {
            const P = window.__p, last = P.index.lastStep;
            const read = () => document.querySelector('[data-panel="ring"]').innerHTML;
            const shapes = [
                ['尾 -> 1', last, 1], ['1 -> 0', 1, 0], ['0 -> 尾', 0, last],
                ['尾 -> 7', last, 7], ['7 -> 3', 7, 3], ['3 -> 11', 3, 11],
                ['11 -> 11', 11, 11], ['11 -> 0', 11, 0], ['0 -> 5', 0, 5],
                ['5 -> 1', 5, 1], ['1 -> 9', 1, 9], ['9 -> 4', 9, 4],
            ];
            const out = [];
            for (const [label, from, to] of shapes) {
                P.setCursor(from); P.setCursor(to);
                const jumped = read();
                P.setCursor(0);
                for (let i = 1; i <= to; i++) P.setCursor(i);
                const played = read();
                out.push({label, same: jumped === played});
            }
            P.setCursor(0);
            return out;
        }""")
        for c in purity:
            check(f"跳转 {c['label']} 后的面板 == 顺序播放到该步", c["same"])

        control = page.evaluate("""() => {
            const T = window.__labTrace;
            const ranks = T.ring.ranks.map(r => r.id);
            const R = {}; ranks.forEach((r, i) => R[r] = i);
            const chunks = T.ring.chunks;
            const blank = () => ranks.map(() => new Array(chunks).fill(1));
            const step = (g, i) => (T.steps[i].transfers || []).forEach(x => {
                const a = R[x.from], b = R[x.to];
                g[b][x.chunk] = x.mode === 'copy' ? g[a][x.chunk]
                                                  : g[b][x.chunk] + g[a][x.chunk];
            });
            // ground truth at step 7, played forward
            let g = blank(); for (let i = 0; i <= 7; i++) step(g, i);
            const truth = JSON.stringify(g);
            // the stateful bug: play the whole run, then "jump back" to 7 by
            // applying only step 7's transfers on top of the final state
            g = blank(); for (let i = 0; i < T.steps.length; i++) step(g, i);
            step(g, 7);
            return {agree: JSON.stringify(g) === truth};
        }""")
        check("对照组（有状态折叠）确实会翻车 —— 上面的『一致』不是测试写错",
              not control["agree"],
              "对照组跳转后与基准不一致" if not control["agree"]
              else "对照组也一致，无区分力")

        # ========================================= two levels of control group
        print("\n== 两层的对照组（引擎张量层 vs 组件折叠层）==")
        engine_jump = page.evaluate("""() => {
            const T = window.__labTrace;
            const idx = LabEngine.traceModel.index(T);
            const r = LabEngine.verify.runJumpCheck(T, idx);
            return {total: r.total, pure: r.pureFailures, ctl: r.controlFailures,
                    conclusive: r.conclusive};
        }""")
        note(f"引擎张量层：{engine_jump['total']} 次跳转，纯函数 0 次不一致，"
             f"对照组 {engine_jump['ctl']} 次不一致")
        # This is the assertion behind the view's design, not a restatement of
        # it: the engine's tensor-state control group is INCONCLUSIVE here, and
        # the reason is a property of the ring — every step rewrites every
        # buffer in full, so a stateful player lands on the right answer.
        check("引擎张量层的对照组在这份 trace 上没有区分力（这正是组件自检必须存在的理由）",
              engine_jump["pure"] == 0 and engine_jump["ctl"] == 0
              and not engine_jump["conclusive"])
        # The reason, asserted rather than asserted-about: the engine's control
        # group folds tensor state, and a ring's state is COMPLETE every step —
        # all N buffers are rewritten in full. (Not every tensor: `grad` is the
        # input and is written once. The buffers are what a stateful player
        # would have to get right, and it does, by luck of the ring's
        # completeness.)
        rewrite = page.evaluate("""() => {
            const T = window.__labTrace;
            const bufs = Object.keys(T.tensors).filter(k => k.indexOf('buf_') === 0);
            return T.steps.map(s => {
                const st = s.state || {};
                return bufs.every(b => Object.prototype.hasOwnProperty.call(st, b));
            });
        }""")
        check("而它没有区分力的原因确实是「每一步都整体重写每张卡的缓冲区」",
              all(rewrite), f"整体重写的步：{sum(rewrite)} / {len(rewrite)}")
        check("组件折叠层的对照组相反：有区分力（两块的控制组不是一回事）",
              self_check["ctl"] > 0)

        # ==================================================== trace 真实性
        print("\n== trace 真实性（本票自己产出的那份）==")
        for n in N_VALUES:
            ref = traces[n]["meta"]["reference"]
            check(f"N={n} 的对拍结论写进了 meta.reference（可追溯到生成器）",
                  "torch.equal" in ref and "容差" in ref and "整数" in ref)
        m8 = traces[8]["meta"]["metrics"]
        check("守恒：环形单链路负载 == 2(N−1)·S/N 字节",
              m8["ring_link_bytes"] == 2 * 7 * 8 // 8 * 4,
              f"{m8['ring_link_bytes']} B")
        check("守恒：朴素单链路负载 == 环形 × N",
              m8["naive_link_bytes"] == m8["ring_link_bytes"] * 8)
        # Reproducibility, checked the strong way: the files were read into
        # memory above, then the generator is re-run now, so a second read must
        # return exactly what the first did. That is what CI relies on.
        before = {n: traces[n] for n in N_VALUES}
        proc = subprocess.run([sys.executable, str(GENERATOR)],
                              capture_output=True, text=True, cwd=str(REPO))
        check("生成器重跑成功（断言与 lint 对照组都过，否则它会拒绝写文件）",
              proc.returncode == 0, (proc.stderr or "")[-300:])
        check("生成的 trace 可复现（重跑后逐字段相同）",
              all(load(n) == before[n] for n in N_VALUES))

        # ==================================================== deep links
        print("\n== 深链（入口卡片指向的正是它声称的那一刻）==")
        page.goto(url + "?n=4&step=3")
        page.wait_for_timeout(1000)
        dl = page.evaluate("""() => ({
            cursor: window.__p.state.cursor,
            n: +document.querySelector('.lab-ring').dataset.n,
            // step 3 is the last ReduceScatter step: every rank has exactly one
            // complete chunk, which is the moment the second phase exists for.
            full: document.querySelectorAll('.lab-ring-cell-full').length,
            banner: !document.getElementById('lab-banner').hidden})""")
        check("深链 ?n=4&step=3 落在 ReduceScatter 收尾那一刻（每卡各一块完成）",
              dl["cursor"] == 3 and dl["n"] == 4 and dl["full"] == 4 and not dl["banner"],
              json.dumps(dl, ensure_ascii=False))

        page.goto(url + "?n=8&step=9999")
        page.wait_for_timeout(1000)
        bad = page.evaluate("""() => ({
            cursor: window.__p.state.cursor,
            text: document.getElementById('lab-banner').textContent,
            shown: !document.getElementById('lab-banner').hidden})""")
        check("越界的 ?step 停在最后一步，且横幅说的就是「最后一步」",
              bad["shown"] and bad["cursor"] == len(traces[8]["steps"]) - 1
              and "最后一步" in bad["text"],
              json.dumps(bad, ensure_ascii=False))

        page.goto(url + "?n=7")
        page.wait_for_timeout(1000)
        badn = page.evaluate("""() => ({
            n: +document.querySelector('.lab-ring').dataset.n,
            text: document.getElementById('lab-banner').textContent,
            shown: !document.getElementById('lab-banner').hidden})""")
        check("不存在的 ?n=7 退回默认配置，并说明原因",
              badn["n"] == 4 and badn["shown"] and "7" in badn["text"],
              json.dumps(badn, ensure_ascii=False))

        page.goto(url)
        page.wait_for_timeout(1100)

        # ================================================================= shots
        print("\n== 截图 ==")
        page.click('[data-n="4"]')
        page.wait_for_timeout(700)
        page.evaluate("() => { window.__p.setCursor(0); window.scrollTo(0, 0); }")
        page.wait_for_timeout(400)
        page.screenshot(path=SHOTS / "50-l13-overview.png")

        for step, name in ((3, "51-l13-reducescatter-done"),
                           (6, "52-l13-allgather-done")):
            page.evaluate(f"() => window.__p.setCursor({step})")
            page.wait_for_timeout(350)
            page.locator(".lab-ring-panel").screenshot(path=SHOTS / f"{name}.png")

        page.click('[data-n="8"]')
        page.wait_for_timeout(700)
        page.evaluate("() => window.__p.setCursor(7)")
        page.wait_for_timeout(350)
        page.locator(".lab-ring-svg").screenshot(path=SHOTS / "53-l13-ring8.png")
        page.locator('[data-panel="ring"]').screenshot(path=SHOTS / "54-l13-acc-8.png")

        page.click('[data-n="2"]')
        page.wait_for_timeout(700)
        page.evaluate("() => window.__p.setCursor(2)")
        page.wait_for_timeout(350)
        page.locator('[data-panel="ring"]').screenshot(path=SHOTS / "55-l13-ring2.png")

        page_light = browser.new_page(viewport={"width": W, "height": H},
                                      color_scheme="light")
        page_light.goto(url)
        page_light.wait_for_timeout(1300)
        page_light.evaluate("() => window.__p.setCursor(5)")
        page_light.wait_for_timeout(350)
        page_light.locator(".lab-ring-panel").screenshot(
            path=SHOTS / "56-l13-light.png")

        print(f"\n截图 -> {SHOTS}")
        print("\n== console ==")
        if console:
            for c in console[:20]:
                print("  " + c[:200])
            failures.append("console errors/warnings present")
        else:
            print("  （无 error / warning）")

        browser.close()

    lint_parity()

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
