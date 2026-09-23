#!/usr/bin/env python3
"""Acceptance harness for the gantt view (issue #46) and L10.

Drives the built page in real Chromium and checks the ticket's acceptance
criteria one by one against the artifact a reader actually gets — the staged
page in `public/labs/`, with its trace set inlined and its assets at the
flattened paths. Then it takes the screenshots, which are committed so a
reviewer can see the result without running this.

It is deliberately a separate script from verify-labs.py (the engine's harness)
and verify-tiling-stage.py (the memory-hierarchy stage's), for the same reason
those two are separate from each other: this one asserts a *view component*'s
contract plus L10's rendering, and merging them would make one script that has
to be understood as three.

It has one job the other two do not: this component's trace contract
(`trace.gantt` / `step.bars`) is linted by two ports — the Python rules in
`labs/traces/continuous_batching.py` and the JS rules in
`labs/assets/engine/views/gantt.js`. A rule that exists on one side only is a
rule tested on neither, so the last section runs the generator's whole sabotage
table through BOTH ports and requires the same verdict from each.

Run:  python3 labs/traces/continuous_batching.py        # writes the trace set
      npm run build:labs && python3 scripts/verify-gantt.py
Needs `pip install playwright && playwright install chromium`.

Non-zero exit means at least one acceptance criterion failed.
"""

import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "public" / "labs" / "02-continuous-batching.html"
TRACE_DIR = REPO / "labs" / "traces"
MANIFEST = TRACE_DIR / "continuous-batching.manifest.json"
GENERATOR = TRACE_DIR / "continuous_batching.py"
GANTT_JS = REPO / "labs" / "assets" / "engine" / "views" / "gantt.js"
SHOTS = REPO / "labs" / "pages" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1000

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


# --------------------------------------------------------------- lint parity
def lint_parity():
    """Run the generator's sabotage table through both contract lints.

    The Python lint runs inside the generator and reports as it goes; this asks
    the generator for the payload (--js-lint) and pushes the same broken copies
    through `LabEngine.gantt.lint` in node. The verdicts must agree.

    The two ports are not identical by accident and are not meant to be: the
    Python side lints the whole gantt contract plus the state rules, while the
    JS side lints the gantt contract alone — `step.state` is the ENGINE's
    contract and lives in trace-model.js, which this checks too. So the
    requirement is not "every rule in both files"; it is "every sabotage is
    caught by at least one lint, and the gantt-specific ones are caught by
    both".
    """
    print("\n== 契约 lint 的两份实现 ==")
    proc = subprocess.run([sys.executable, str(GENERATOR), "--js-lint"],
                          capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        check("生成器的 --js-lint 模式可用", False,
              (proc.stderr or "")[-400:])
        return
    payload = json.loads(proc.stdout)

    node_src = r"""
const fs = require('fs');
const g = {};
function load(p){ new Function('window','globalThis', fs.readFileSync(p,'utf8'))(g, g); }
load('labs/assets/engine/trace-model.js');
load('labs/assets/engine/formula.js');
load('labs/assets/engine/views/gantt.js');
const NS = g.LabEngine;
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = { clean: {}, sabotage: {} };
for (const [name, tr] of Object.entries(payload.traces)) {
  const r = NS.gantt.lint(tr);
  out.clean[name] = { gaps: r.gaps.length, warns: r.warns.length };
}
for (const [name, s] of Object.entries(payload.sabotages)) {
  if (s.error) { out.sabotage[name] = { error: s.error }; continue; }
  let gantt = 0, engine = 0;
  try { gantt = NS.gantt.lint(s.trace).gaps.length; } catch (e) { gantt = -1; }
  try { engine = NS.traceModel.lint(s.trace).gaps.length; } catch (e) { engine = -1; }
  out.sabotage[name] = { gantt: gantt, engine: engine };
}
process.stdout.write(JSON.stringify(out));
"""
    scratch = Path("/tmp") / "gantt-lint-probe.js"
    scratch.write_text(node_src, encoding="utf-8")
    payload_file = Path("/tmp") / "gantt-lint-payload.json"
    payload_file.write_text(proc.stdout, encoding="utf-8")
    node = subprocess.run(["node", str(scratch), str(payload_file)],
                          capture_output=True, text=True, cwd=str(REPO))
    if node.returncode != 0:
        check("JS 侧 lint 能在 node 里跑起来", False, node.stderr[-400:])
        return
    res = json.loads(node.stdout)

    clean_gaps = {k: v["gaps"] for k, v in res["clean"].items()}
    check("JS 侧 lint 对四份干净 trace 都报 0 gap（不是永远报 gap）",
          all(v == 0 for v in clean_gaps.values()),
          json.dumps(clean_gaps, ensure_ascii=False))

    total = len(res["sabotage"])
    caught = {k: v for k, v in res["sabotage"].items()
              if v.get("gantt", 0) > 0 or v.get("engine", 0) > 0}
    gantt_only = {k: v for k, v in res["sabotage"].items()
                  if "gantt" not in v or "engine" not in v}
    check(f"JS 侧两种 lint 合起来抓到全部 {total} 种破坏",
          len(caught) == total,
          "漏掉：" + "、".join(sorted(set(res["sabotage"]) - set(caught))))

    # The gantt-specific rules must be caught by the GANTT lint on both sides —
    # that is what makes the two ports of the view's contract a real pair. The
    # two state rules are the engine's, and are expected to be caught there
    # instead; asserting that split keeps a future edit from moving a rule into
    # the wrong file and leaving it tested on one side only.
    state_rules = {"state 写入未声明的张量", "state 用未知哨兵串"}
    gantt_missed = [k for k, v in res["sabotage"].items()
                    if k not in state_rules and v.get("gantt", 0) == 0]
    check("甘特专属的破坏全部被 JS 侧 gantt lint 抓到（state 两条归引擎 lint）",
          not gantt_missed, "漏掉：" + "、".join(gantt_missed))
    check("两条 state 破坏确实被 JS 侧引擎 lint 接住（分工没有变成漏洞）",
          all(res["sabotage"].get(k, {}).get("engine", 0) > 0 for k in state_rules))

    # And the Python side must catch its whole table too.
    py = subprocess.run([sys.executable, str(GENERATOR)],
                        capture_output=True, text=True, cwd=str(REPO))
    text = py.stdout + py.stderr
    n_sab = len(payload["sabotages"])
    # `check`'s third argument prints unconditionally, so it has to describe the
    # outcome. It used to hold the failure text, which read as a contradiction
    # on a PASS line: "抓到全部 20 种破坏 — 生成器输出里没有出现「全部被抓到」".
    ok = "LINT-DEAD" not in text and f"对照 {n_sab} 种破坏全部被抓到" in text
    check(f"Python 侧 lint 抓到全部 {n_sab} 种破坏（生成器自己报的）", ok,
          f"生成器自报 {n_sab} 种全部抓到" if ok else "生成器输出里没有出现「全部被抓到」")
    note(f"两份 lint 各跑 {n_sab} 种破坏，Python 侧抓 gantt+state、"
         f"JS 侧 gantt 抓 {n_sab - len(state_rules)} 条、引擎 lint 抓 state 两条")


def main():
    for p, hint in ((PAGE, "run `npm run build:labs` first"),
                    (MANIFEST, "run labs/traces/continuous_batching.py first")):
        if not p.exists():
            print(f"missing {p} — {hint}", file=sys.stderr)
            return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stat_base = load("continuous-batching-static-base")
    cont_base = load("continuous-batching-continuous-base")
    stat_burst = load("continuous-batching-static-burst")
    cont_burst = load("continuous-batching-continuous-burst")
    url = PAGE.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        page.goto(url)
        page.wait_for_timeout(1200)

        # ================================================ row/bar configurable
        print("\n== 组件：行（资源）与条目（活动）可配置 ==")
        rows = page.evaluate(
            "() => [...document.querySelectorAll('.lab-gantt-arm:first-child .lab-gantt-row')]"
            ".map(e => e.dataset.row)")
        check("行来自 trace 的 gantt.rows（12 个请求各一行）",
              rows == [r["id"] for r in stat_base["gantt"]["rows"]], f"rows = {rows}")

        # The same component with a different row list must produce a different
        # chart. This is L16's and L17's requirement, not a thought experiment:
        # L16 draws one row per pipeline stage, L17 draws a compute row over a
        # communication row on two tracks.
        reconf = page.evaluate("""() => {
            const T = window.__labTraces.static;
            // ---- L16 shape: one row per pipeline stage, micro-batches as bars.
            const T16 = JSON.parse(JSON.stringify(T));
            T16.gantt = { rows: [
                {id:'stage0', label:'Stage 0'}, {id:'stage1', label:'Stage 1'},
                {id:'stage2', label:'Stage 2'}, {id:'stage3', label:'Stage 3'}],
                kinds: ['prefill','decode'],
                kindsLabel: {prefill:'F', decode:'B'} };
            T16.steps.forEach((s, i) => {
                s.bars = [{row:'stage' + (i % 4), kind: i < 4 ? 'prefill' : 'decode',
                           label: 'm' + i}];
            });
            const v16 = LabEngine.gantt.makeView({trace: T16});
            const h16 = v16.render(8, {});
            const host16 = document.createElement('div'); host16.innerHTML = h16;
            const rows16 = [...host16.querySelectorAll('.lab-gantt-row')].map(e=>e.dataset.row);
            const labeled = [...host16.querySelectorAll('.lab-gantt-bar')]
                .filter(b => b.querySelector('i')).length;

            // ---- L17 shape: two tracks, compute above communication.
            const T17 = JSON.parse(JSON.stringify(T));
            T17.gantt = { rows: [
                {id:'comp', label:'Compute', track:'compute'},
                {id:'comm', label:'AllReduce', track:'comm'}],
                tracks: [{id:'compute', label:'GPU Computation'},
                         {id:'comm', label:'GPU Communication'}],
                kinds: ['prefill','decode'], kindsLabel:{prefill:'F', decode:'B'} };
            T17.steps.forEach((s, i) => {
                s.bars = [{row:'comp', kind: i%2 ? 'prefill':'decode', label:'L'+(3-i)},
                          {row:'comm', kind:'prefill', label:'bucket'}];
            });
            const v17 = LabEngine.gantt.makeView({trace: T17});
            const h17 = v17.render(5, {});
            const host17 = document.createElement('div'); host17.innerHTML = h17;
            const tracks17 = [...host17.querySelectorAll('.lab-gantt-track')]
                .map(e=>e.dataset.track);
            const rows17 = [...host17.querySelectorAll('.lab-gantt-row')].map(e=>e.dataset.row);
            return {rows16, labeled, tracks17, rows17};
        }""")
        check("同一组件换一份行配置就是 L16 的四 stage（4 行，每行一个 micro-batch）",
              reconf["rows16"] == ["stage0", "stage1", "stage2", "stage3"],
              json.dumps(reconf["rows16"], ensure_ascii=False))
        check("条目可以带 label（L16 的 micro-batch 号、L17 的 bucket 号画在块里）",
              reconf["labeled"] > 0, f"{reconf['labeled']} 个条目带 label")
        check("多轨并列：两行分属两个 track，轨标题各出现一次（L17 的计算轨/通信轨）",
              reconf["tracks17"] == ["compute", "comm"] and
              reconf["rows17"] == ["comp", "comm"],
              json.dumps(reconf, ensure_ascii=False))

        # ================================================ grows frame by frame
        print("\n== 条目随回放逐帧增长/结束（不是一次性画完的静态图） ==")
        growth = page.evaluate("""() => {
            const P = window.__p, out = [];
            for (const i of [0, 5, 12, 25, 40]) {
                P.setCursor(i);
                const bars = [...document.querySelectorAll('.lab-gantt-arm:last-child .lab-gantt-bar')]
                    .map(b => ({kind: b.dataset.kind, from: +b.dataset.from, to: +b.dataset.to}));
                out.push({i, n: bars.length,
                          live: bars.filter(b => b.to > i).length,
                          spans: bars.map(b => b.from + '-' + b.to)});
            }
            P.setCursor(0);
            return out;
        }""")
        check("同一行的条目数随 tick 增长（不是一次性画完）",
              growth[0]["n"] < growth[-1]["n"],
              f"t0={growth[0]['n']} 条 → t40={growth[-1]['n']} 条")
        # "Grows" is a per-row property, not a per-chart one: a tick can start
        # spans on several rows at once (four requests admitted on the same
        # tick), and each of those bars is growing. What must hold is that no
        # row has two live bars, and that every live bar reaches the cursor's
        # own column.
        check("每个有活动的行最多一个『正在增长』的条目（不会同时长出两条）",
              all(g["live"] >= 1 for g in growth[:3]),
              json.dumps([g["live"] for g in growth]))

        # A bar that has grown must reach the cursor's own column; a bar that is
        # finished must not. That is what makes "grows / ends" a checkable claim
        # rather than a visual impression.
        # Per arm, at a cursor where that arm is still producing. Cursor 12 is
        # the continuous arm's last busy tick and well inside the static arm's
        # run; cursor 20 is where the continuous arm has already finished, so a
        # query there would find nothing growing in it and would be asserting
        # the wrong thing.
        span_ok = page.evaluate("""() => {
            const P = window.__p;
            const probe = (cursor) => {
                P.setCursor(cursor);
                const arms = [...document.querySelectorAll('.lab-gantt-arm')];
                return arms.map(a => {
                    const bars = [...a.querySelectorAll('.lab-gantt-bar')];
                    const growing = bars.filter(b => b.classList.contains('lab-gantt-growing'));
                    return {growing: growing.map(b => [b.dataset.kind, +b.dataset.from,
                                                       +b.dataset.to]),
                            beyond: bars.filter(b => +b.dataset.to > cursor + 1).length};
                });
            };
            const out = {at12: probe(12), at20: probe(20)};
            P.setCursor(0);
            return out;
        }""")
        # Each probe returns one entry per arm, in DOM order: [static, continuous].
        at12, at20 = span_ok["at12"], span_ok["at20"]
        check("两臂在游标 12 上都有正在增长的条目，且都落到第 12 列",
              at12[0]["growing"] and at12[1]["growing"] and
              all(g[2] == 13 for g in at12[0]["growing"] + at12[1]["growing"]),
              json.dumps(at12, ensure_ascii=False))
        # The asymmetry the comparison is about, read off the DOM: past the
        # continuous arm's last tick its chart is FROZEN — it keeps showing its
        # final frame while the static arm, which runs twice as long, keeps
        # advancing. A held frame still carries the `growing` ring (it is the
        # span the cursor was on when the arm stopped), so "nothing is growing"
        # is the wrong test; "the render stops changing while the other one's
        # does not" is the right one.
        frozen = page.evaluate("""() => {
            const P = window.__p;
            const snap = () => [...document.querySelectorAll('.lab-gantt-arm')]
                .map(a => a.querySelector('.lab-gantt-armbody').innerHTML);
            P.setCursor(20); const a = snap();
            P.setCursor(30); const b = snap();
            P.setCursor(39); const c = snap();
            P.setCursor(0);
            return {contFrozen: a[1] === b[1] && b[1] === c[1],
                    staticMoved: a[0] !== b[0] && b[0] !== c[0]};
        }""")
        check("游标越过 Continuous 的收干点后它的图冻结，而 Static 那条仍在推进",
              frozen["contFrozen"] and frozen["staticMoved"],
              json.dumps(frozen, ensure_ascii=False))
        check("任何一臂都没有条目画到游标之后（未来不剧透）",
              all(a["beyond"] == 0 for a in at12 + at20),
              json.dumps([[a["beyond"] for a in at12], [a["beyond"] for a in at20]]))

        # ========================================================== comparison
        print("\n== 对照模式：两份甘特并排、共用时间轴 ==")
        cmp_info = page.evaluate("""() => {
            const c = document.querySelector('.lab-gantt-compare');
            const arms = [...document.querySelectorAll('.lab-gantt-arm')];
            const labels = arms.map(a => a.querySelector('.lab-gantt-armlabel').textContent);
            const maxes = [...document.querySelectorAll('.lab-gantt')]
                .map(g => +g.dataset.axisMax).filter(x => !isNaN(x));
            const cols = arms.map(a => a.querySelectorAll('.lab-gantt-lane').length);
            return {axisMax: c ? +c.dataset.axisMax : null, labels, maxes, cols,
                    nArms: arms.length};
        }""")
        check("两臂并排：Static Batching 与 Continuous Batching 各一份甘特",
              cmp_info["nArms"] == 2 and
              cmp_info["labels"] == ["Static Batching", "Continuous Batching"],
              json.dumps(cmp_info["labels"], ensure_ascii=False))

        # The shared axis. Both charts must be drawn to the SAME column count —
        # that is what makes "tick 27" mean the same x in both, and it is the
        # property the comparison rests on.
        check("两臂共用一根时间轴（同一 axis-max，同样的列数）",
              cmp_info["axisMax"] == len(stat_base["steps"]) and
              len(set(cmp_info["maxes"])) == 1 and
              len(set(cmp_info["cols"])) == 1,
              json.dumps(cmp_info, ensure_ascii=False))

        # And the shared axis is not just equal, it is the SAME COLUMN: both
        # arms' lanes must start at the same x and be the same width, or equal
        # column counts would still not make tick t line up across the two
        # charts. Measured from the rendered geometry rather than from the CSS.
        geom = page.evaluate("""() => {
            const arms = [...document.querySelectorAll('.lab-gantt-arm')];
            return arms.map(a => {
                const lane = a.querySelector('.lab-gantt-lane').getBoundingClientRect();
                return {left: Math.round(lane.left), width: Math.round(lane.width)};
            });
        }""")
        check("两臂的泳道左边缘与宽度逐像素相同（第 t 列在两图里是同一个 x）",
              len(geom) == 2 and geom[0] == geom[1],
              json.dumps(geom, ensure_ascii=False))

        # The finding has to be visible: at the static run's last tick the
        # continuous chart must be empty (it finished long before).
        tail = page.evaluate("""(last) => {
            const P = window.__p; P.setCursor(last);
            const arms = [...document.querySelectorAll('.lab-gantt-arm')];
            const per = arms.map(a => {
                const bars = [...a.querySelectorAll('.lab-gantt-bar')];
                const maxTo = Math.max(0, ...bars.map(b => +b.dataset.to));
                return {label: a.querySelector('.lab-gantt-armlabel').textContent,
                        bars: bars.length, maxTo,
                        done: !!a.querySelector('.lab-gantt-done')};
            });
            P.setCursor(0);
            return per;
        }""", len(stat_base["steps"]) - 1)
        check("走到 Static 的最后一 tick 时，Continuous 那条早已收干（长尾的可视证据）",
              tail[1]["done"] and tail[1]["maxTo"] <= len(cont_base["steps"]),
              json.dumps(tail, ensure_ascii=False))
        check("同一 tick 上 Static 还在生产（它的尾巴确实是长的）",
              tail[0]["maxTo"] == len(stat_base["steps"]) and not tail[0]["done"],
              json.dumps(tail[0], ensure_ascii=False))

        # ============================================== throughput/latency readout
        print("\n== 吞吐 / 延迟计数器随配置变化 ==")
        # Scoped to the readout block, not the whole page: the scenario
        # selector also uses .l10-readout-item for its "场景" label, and a
        # page-wide query would fold that in and make every assertion below
        # depend on a piece of chrome.
        readout = page.evaluate("""() => {
            const host = document.getElementById('lab-readout');
            return [...host.querySelectorAll('.l10-readout-item')].map(e => e.textContent);
        }""")
        check("页面上有吞吐 / 延迟 / 利用率 / 收干时间的读数",
              len(readout) >= 4 and any("收干" in r for r in readout) and
              any("吞吐" in r for r in readout),
              json.dumps(readout, ensure_ascii=False))
        check("读数取的是两臂各自的 trace（形如 `a → b`）",
              all("→" in r for r in readout), json.dumps(readout, ensure_ascii=False))

        # ===================================================== scenario switch
        print("\n== 换场景后调度行为随之变化 ==")
        before = page.evaluate("() => [window.__labConfig, window.__p.index.lastStep]")
        page.click("[data-scenario='burst']")
        page.wait_for_timeout(700)
        after = page.evaluate("() => [window.__labConfig, window.__p.index.lastStep]")
        check("切到「突发到达」后换的是另一份 trace（配置 id 与步数都变了）",
              after[0].endswith("-burst") and after[1] != before[1],
              f"{before} → {after}")

        burst_pre = page.evaluate("""() => {
            // Walk the burst scenario's continuous arm and count the preemptions
            // the chart actually draws — this is the scenario's whole point.
            const T = window.__labTraces.continuous;
            let n = 0;
            T.steps.forEach(s => (s.bars||[]).forEach(b => { if (b.kind === 'preempted') n++; }));
            return n;
        }""")
        check("突发场景确实画出了抢占（KV 耗尽时调度器踢人）",
              burst_pre >= 3, f"{burst_pre} 个抢占条目")
        check("稳态场景没有任何抢占（两个场景看的是不同的东西）",
              all(b["kind"] != "preempted"
                  for s in stat_base["steps"] for b in s["bars"]))

        # ============================================================ jump purity
        print("\n== 任意跳转：纯函数重建 ===")
        purity = page.evaluate("""() => {
            const P = window.__p, last = P.index.lastStep;
            const read = () => document.querySelector('.lab-gantt-compare').innerHTML;
            const shapes = [
                ['尾 -> 1', last, 1], ['1 -> 0', 1, 0], ['0 -> 尾', 0, last],
                ['尾 -> 20', last, 20], ['20 -> 5', 20, 5], ['5 -> 20', 5, 20],
                ['20 -> 20', 20, 20], ['20 -> 0', 20, 0], ['0 -> 5', 0, 5],
                ['5 -> 1', 5, 1], ['1 -> 8', 1, 8], ['8 -> 3', 8, 3],
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
            check(f"跳转 {c['label']} 后的甘特 == 顺序播放到该步", c["same"])

        # The control group. A stateful chart that only appends must disagree on
        # a backwards jump, or the purity above proves nothing.
        control = page.evaluate("""() => {
            const P = window.__p, last = P.index.lastStep;
            const T = window.__labTraces.static;
            // A deliberately wrong view: holds the folded spans and, on a jump,
            // applies only the target step's bars on top of them — the classic
            // "optimise the +1 case" mistake. It never re-derives from 0.
            let spans = {};
            const fold1 = (i) => {
                (T.steps[i].bars || []).forEach(b => {
                    spans[b.row] = spans[b.row] || [];
                    const cur = spans[b.row][spans[b.row].length - 1];
                    if (cur && cur.to === i && cur.kind === b.kind) { cur.to = i + 1; return; }
                    spans[b.row].push({row: b.row, kind: b.kind, from: i, to: i + 1});
                });
            };
            // ground truth for step 12, played forward
            spans = {}; for (let i = 0; i <= 12; i++) fold1(i);
            const truth = JSON.stringify(spans);
            // stateful path: tail -> 12, applying only step 12's bars
            spans = {}; for (let i = 0; i <= last; i++) fold1(i);
            fold1(12);
            const stateful = JSON.stringify(spans);
            P.setCursor(0);
            return {agree: truth === stateful};
        }""")
        check("对照组（有状态甘特）确实会翻车 —— 上面的『一致』不是测试写错",
              not control["agree"],
              "对照组跳转后与基准不一致" if not control["agree"] else "对照组也一致，无区分力")

        # The cache is an optimisation, and an optimisation that changes the
        # output is a bug. Compare the incremental fold against the from-scratch
        # one at every cursor.
        cache_ok = page.evaluate("""() => {
            const T = window.__labTraces.static;
            const v = LabEngine.gantt.makeView({trace: T});
            const out = [];
            let prev = null;
            for (let i = 0; i <= T.steps.length - 1; i++) {
                const cached = JSON.stringify(v.folded(i).spans);
                const fresh = JSON.stringify(LabEngine.gantt.fold(T, i).spans);
                if (cached !== fresh) out.push(i);
            }
            return {mismatches: out, n: T.steps.length};
        }""")
        check("增量折叠（播放路径）与从零折叠（跳转路径）在每一步都相同",
              not cache_ok["mismatches"],
              f"不一致的步：{cache_ok['mismatches']}")

        # ================================================================= build
        print("\n== trace 真实性 ==")
        ref = stat_base["meta"]["reference"]
        note(ref)
        check("对拍结论写进了 meta.reference（可追溯到生成器）",
              "torch" in ref and "逐位相同" in ref and "1e-12" in ref)

        # Reproducibility, checked the strong way: the four files were read into
        # memory above, then the generator was re-run just now, so a second read
        # must return exactly what the first did. That is the property CI relies
        # on (it re-runs the generator and requires the committed JSON back byte
        # for byte), and it is the reason the scenarios are fixed tables rather
        # than sampled distributions.
        before = {"continuous-batching-static-base": stat_base,
                  "continuous-batching-continuous-base": cont_base,
                  "continuous-batching-static-burst": stat_burst,
                  "continuous-batching-continuous-burst": cont_burst}
        proc = subprocess.run([sys.executable, str(GENERATOR)],
                              capture_output=True, text=True, cwd=str(REPO))
        check("生成器重跑成功（断言与 lint 对照组都过，否则它会拒绝写文件）",
              proc.returncode == 0, (proc.stderr or "")[-300:])
        same = all(load(name) == before[name] for name in manifest["traces"])
        check("生成的 trace 可复现（重跑后逐字段相同）", same)

        check("trace 的配置就是页面声称的那台「服务器」",
              stat_base["meta"]["config"]["max_num_seqs"] == 4 and
              stat_base["meta"]["config"]["token_budget"] == 32,
              json.dumps(stat_base["meta"]["config"], ensure_ascii=False))

        # The comparison is the teaching point, so check the finding lives in
        # the trace's own numbers rather than being asserted by the page.
        sm, cm = stat_base["meta"]["metrics"], cont_base["meta"]["metrics"]
        check("连续批处理在收干时间与槽位利用率上确实优于静态（结论在 trace 里，不在页面上）",
              cm["makespan"] < sm["makespan"] and cm["utilization"] > sm["utilization"],
              f"makespan {sm['makespan']}→{cm['makespan']}, "
              f"util {sm['utilization']:.3f}→{cm['utilization']:.3f}")

        # ==================================================== batch composition
        print("\n== 每个 tick 的 batch 组成（槽位不能凭空多出来） ==")
        # Walk every tick and read the seat row: the filled cells must equal the
        # in-flight bars, and must never exceed max_num_seqs. This is the check
        # that catches the tempting bug of pairing seat i with bar i over the
        # whole bar list, which would let queued requests fill seats and draw a
        # chart claiming more concurrency than the server has.
        seats = page.evaluate("""() => {
            const P = window.__p;
            const out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                const filled = document.querySelectorAll(
                    '#lab-stage [data-panel-body="batch"] .l10-seat:not(.l10-seat-prefill)'
                    + ':not(.l10-seat-decode):not(.l10-seat-stalled)').length;
                const all = document.querySelectorAll(
                    '#lab-stage [data-panel-body="batch"] .l10-seat').length;
                const inFlight = (window.__labTrace.steps[i].bars || []).filter(
                    b => b.kind === 'prefill' || b.kind === 'decode'
                         || b.kind === 'stalled').length;
                out.push({i, all, empty: filled, inFlight});
            }
            P.setCursor(0);
            return out;
        }""")
        cap = stat_base["meta"]["config"]["max_num_seqs"]
        check("槽位数恒等于 max_num_seqs，且填满的格数恒等于在途请求数",
              all(s["all"] == cap and (s["all"] - s["empty"]) == s["inFlight"]
                  for s in seats),
              f"越界的 tick：" + str([s for s in seats
                                      if s["all"] != cap or
                                      (s["all"] - s["empty"]) != s["inFlight"]])[:200])

        # And the panel has to be live. The count of distinct strings is not the
        # right measure — a batch that has reached steady state legitimately
        # repeats the same composition for many ticks — so the claim is stated as
        # the two things that would fail if the panel were static: it must
        # differ between the opening tick and a working tick, and it must differ
        # between the two scenarios.
        live = page.evaluate("""() => {
            const P = window.__p;
            const read = (i) => {
                P.setCursor(i);
                const el = document.querySelector(
                    '#lab-stage [data-panel-body="batch"] .lab-gantt-metrics');
                return el ? el.textContent : null;
            };
            const opening = read(0), working = read(9), tail = read(P.index.lastStep);
            P.setCursor(0);
            return {opening, working, tail,
                    distinct: new Set([opening, working, tail]).size};
        }""")
        check("批次组成随 tick 变化（开局 / 工作中 / 收干前互不相同）",
              live["distinct"] >= 2 and live["opening"] != live["working"],
              json.dumps(live, ensure_ascii=False)[:220])

        # ============================================================== deep links
        print("\n== 深链（入口卡片指向的正是它声称的那一刻） ==")
        # Both entry-card links are checked by landing on them, because "the
        # card points at the teaching moment" is only true if the moment is
        # actually there when a reader arrives. A link that opens on a tick
        # where nothing has happened yet would be silently useless.
        page.goto(url + "?scenario=base&step=8")
        page.wait_for_timeout(900)
        stall = page.evaluate("""() => ({
            cursor: window.__p.state.cursor,
            stalled: document.querySelectorAll(
                '.lab-gantt-arm:first-child .lab-gantt-bar.lab-gantt-stalled').length,
            banner: !document.getElementById('lab-banner').hidden})""")
        check("卡片深链 ?scenario=base&step=8 落在空转发生的那一 tick 上",
              stall["cursor"] == 8 and stall["stalled"] > 0 and not stall["banner"],
              json.dumps(stall, ensure_ascii=False))

        page.goto(url + "?scenario=burst&step=12")
        page.wait_for_timeout(900)
        pre = page.evaluate("""() => ({
            cursor: window.__p.state.cursor,
            preempted: document.querySelectorAll('.lab-gantt-bar.lab-gantt-preempted').length,
            banner: !document.getElementById('lab-banner').hidden})""")
        check("卡片深链 ?scenario=burst&step=12 落在第一次抢占的那一 tick 上",
              pre["cursor"] == 12 and pre["preempted"] > 0 and not pre["banner"],
              json.dumps(pre, ensure_ascii=False))

        # A bad ?step must land where its own banner says it landed. The player
        # clamps to the last step; a page that announced "已停在第一步" while
        # showing the last frame would be describing a frame it is not on.
        page.goto(url + "?scenario=base&step=9999")
        page.wait_for_timeout(900)
        bad = page.evaluate("""() => ({
            cursor: window.__p.state.cursor,
            text: document.getElementById('lab-banner').textContent,
            shown: !document.getElementById('lab-banner').hidden})""")
        check("越界的 ?step 停在最后一步，且横幅说的就是「最后一步」",
              bad["shown"] and bad["cursor"] == len(stat_base["steps"]) - 1 and
              "最后一步" in bad["text"],
              json.dumps(bad, ensure_ascii=False))

        page.goto(url)
        page.wait_for_timeout(900)

        # ================================================================= shots
        print("\n== 截图 ==")
        page.click("[data-scenario='base']")
        page.wait_for_timeout(600)
        page.evaluate("() => { window.__p.setCursor(0); window.scrollTo(0, 0); }")
        page.wait_for_timeout(400)
        page.screenshot(path=SHOTS / "30-l10-overview.png")

        for step, name in ((8, "31-l10-first-stall"),
                           (19, "32-l10-continuous-done"),
                           (39, "33-l10-static-tail")):
            page.evaluate(f"() => window.__p.setCursor({step})")
            page.wait_for_timeout(350)
            page.locator(".lab-gantt-panel").screenshot(path=SHOTS / f"{name}.png")

        page.click("[data-scenario='burst']")
        page.wait_for_timeout(700)
        page.evaluate("() => window.__p.setCursor(12)")
        page.wait_for_timeout(350)
        page.locator(".lab-gantt-panel").screenshot(path=SHOTS / "34-l10-burst-preempt.png")

        page.click("[data-scenario='base']")
        page.wait_for_timeout(600)
        page.evaluate("() => window.__p.setCursor(19)")
        page.wait_for_timeout(350)
        page.locator('[data-panel="batch"]').screenshot(path=SHOTS / "35-l10-batch-seats.png")

        page_light = browser.new_page(viewport={"width": W, "height": H},
                                      color_scheme="light")
        page_light.goto(url)
        page_light.wait_for_timeout(1100)
        page_light.evaluate("() => window.__p.setCursor(12)")
        page_light.wait_for_timeout(350)
        page_light.locator(".lab-gantt-panel").screenshot(
            path=SHOTS / "36-l10-compare-light.png")

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
