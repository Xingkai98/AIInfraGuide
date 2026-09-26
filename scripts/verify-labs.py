#!/usr/bin/env python3
"""Drive the staged lab page in real Chromium: assert behaviour, capture shots.

This is the acceptance harness for the engine, not a demo. It runs against the
built artifact in `public/labs/` (so it exercises the same inlined trace and
flattened asset paths a reader gets), asserts the acceptance criteria one by
one, and only then takes screenshots.

L00 draws its graph with `views/variable-dag.js` instead of the engine's step
DAG, so what is asserted here is the contract of THAT view: nodes are variables,
edges are computations, every step fits one screen, and the three node states
are actually painted. The engine-side checks the page still owes (arbitrary
jumps, the trace contract lint, the two reference panels, the parameter
sliders) are asserted too, unchanged.

Run:  npm run build:labs && python3 scripts/verify-labs.py
Needs `pip install playwright && playwright install chromium`.

Everything it checks is printed; a non-zero exit means at least one acceptance
criterion failed. Screenshots land in `labs/pages/shots/` (committed, so a
reviewer can see the rendered result without running this).
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# This lives in scripts/ but verifies labs/, so paths are stated relative to the
# repo root rather than to this file's directory — the two differ, and getting
# it wrong fails silently as "page not found".
REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "public" / "labs" / "00-online-softmax.html"
SHOTS = REPO / "labs" / "pages" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1000
PHONE_W, PHONE_H = 390, 844

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


# Read the value grids the view rendered, keyed by tensor. This is what a reader
# sees, which is the only thing any assertion below is allowed to compare.
READ_GRID = """() => {
    const out = {};
    document.querySelectorAll('.vd-vg').forEach(g => {
        const name = g.querySelector('.vd-vg-n').textContent;
        out[name] = {
            shape: g.querySelector('.vd-vg-s').textContent,
            cells: [...g.querySelectorAll('.vd-cell')].map(c => c.textContent),
        };
    });
    return out;
}"""

# The rectangular furniture a node occupies, in the SVG's own coordinate space,
# plus its rendered client box. Both are needed: the coordinate-space box is what
# the geometry predicates reason about, the client box is what a reader sees.
READ_NODES = """() => {
    const svg = document.querySelector('.vd-dag');
    const sb = svg.getBoundingClientRect();
    const vb = svg.getAttribute('viewBox').split(/\\s+/).map(Number);
    const sx = sb.width / vb[2], sy = sb.height / vb[3];
    return {
        svg: {l: sb.left, r: sb.right, t: sb.top, b: sb.bottom, w: vb[2], h: vb[3],
              sx: sx, sy: sy, ox: sb.left - vb[0] * sx, oy: sb.top - vb[1] * sy},
        nodes: [...svg.querySelectorAll('[data-vd-node]')].map(g => {
            const r = g.querySelector('.vd-box');
            const lab = g.querySelector('.vd-nl').getBoundingClientRect();
            return {
                id: g.dataset.vdNode,
                state: g.dataset.vdState,
                cls: r.getAttribute('class'),
                svgBox: {x: +r.getAttribute('x'), y: +r.getAttribute('y'),
                         w: +r.getAttribute('width'), h: +r.getAttribute('height')},
                client: (b => ({l: b.left, r: b.right, t: b.top, b: b.bottom}))(
                    r.getBoundingClientRect()),
                labelClient: {l: lab.left, r: lab.right, t: lab.top, b: lab.bottom},
            };
        }),
    };
}"""


def main():
    if not PAGE.exists():
        print(f"missing {PAGE} — run `npm run build:labs` first", file=sys.stderr)
        return 2

    url = PAGE.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        page.goto(url)
        page.wait_for_timeout(1000)

        # ---------------------------------------------------------- headline
        print("\n== 契约与渲染 ==")
        check("引擎把页面完整驱动起来（变量图/公式/数值/控制条都在）",
              page.evaluate("""() => !!document.querySelector('.vd-root') &&
                  !!document.querySelector('.vd-dag') &&
                  !!document.querySelector('.vd-fml .lab-formula-node') &&
                  !!document.querySelector('[data-vd=scrub]') &&
                  !!document.querySelectorAll('[data-vd-node]').length"""))

        trace = page.evaluate("() => window.__labTrace")
        tensor_names = list(trace["tensors"].keys())

        # The input's `init` must be consumed. x is never written by any step,
        # so if the view only showed what a step writes, the lab's own input
        # would never appear anywhere on the page. It appears at the load step,
        # which is where x is first read — that is the step that makes the
        # input's value the subject.
        page.evaluate("""() => {
            const T = window.__labTrace;
            for (let i = 0; i < T.steps.length; i++) {
                if ((T.steps[i].reads || []).indexOf('x') !== -1) { window.__vd.setCursor(i); return; }
            }
        }""")
        page.wait_for_timeout(250)
        load_grid = page.evaluate(READ_GRID)
        want_x = page.evaluate("() => window.__labTrace.tensors.x.init")
        check("输入张量 x 的 init 被消费（在第一次读它的那一步显示出来）",
              "x" in load_grid and
              len(load_grid["x"]["cells"]) == len(want_x) and
              load_grid["x"]["cells"][0] ==
              (str(want_x[0]) if float(want_x[0]).is_integer()
               else f"{want_x[0]:.4g}"),
              f"x = {load_grid.get('x')} vs trace init[0] = {want_x[0]}")
        check("x 的形状按 2D/1D 决定排版（8 个数排成一行，不是一列）",
              load_grid["x"]["shape"] == "[8]" and
              len(load_grid["x"]["cells"]) == 8,
              json.dumps(load_grid.get("x"), ensure_ascii=False))

        # sentinel rendering: step 0 has m = -\infty
        page.evaluate("() => window.__vd.setCursor(0)")
        page.wait_for_timeout(200)
        m0 = page.evaluate("""() => {
            const g = document.querySelector('.vd-vg');
            return g ? g.textContent : '';
        }""")
        check("±∞ 哨兵解析为 −∞（不是字面 -\\infty）",
              "−∞" in m0 and "\\infty" not in m0, f"step0 值 = {m0!r}")

        mformula = page.evaluate("() => document.querySelector('.vd-fml').textContent")
        check("哨兵在公式面板由 KaTeX 正确渲染（无字面反斜杠）",
              "\\infty" not in mformula, f"formula = {mformula!r}")

        # Every number on the page comes from the trace. Asserted across EVERY
        # step rather than at a sample: the page's core promise is about all of
        # them, and a spot check would pass on a page that renders the right
        # grid at step 4 and a stale one everywhere else.
        print("\n== 页面上的每个数字都来自 trace ==")
        every = page.evaluate("""() => {
            const out = [];
            const fmt = v => {
                if (typeof v === 'string') return v === '-\\\\infty' ? '−∞' : v;
                if (typeof v !== 'number') return String(v);
                if (!isFinite(v)) return v > 0 ? '+∞' : '−∞';
                if (Number.isInteger(v)) return String(v);
                return v.toPrecision(4).replace(/(\\.\\d*?)0+$/, '$1').replace(/\\.$/, '');
            };
            for (let i = 0; i <= window.__vd.index.lastStep; i++) {
                window.__vd.setCursor(i);
                const shown = {};
                document.querySelectorAll('.vd-vg').forEach(g => {
                    shown[g.querySelector('.vd-vg-n').textContent] =
                        [...g.querySelectorAll('.vd-cell')].map(c => c.textContent);
                });
                const step = window.__labTrace.steps[i];
                const want = {};
                Object.keys(step.state || {}).forEach(k => {
                    const v = step.state[k];
                    want[k] = (Array.isArray(v) ? v : [v]).map(fmt);
                });
                out.push({i: i, id: step.id, shown: shown, want: want});
            }
            return out;
        }""")
        bad = []
        for row in every:
            for name, want in row["want"].items():
                got = row["shown"].get(name)
                if got is None:
                    bad.append(f"step {row['i']}({row['id']}): {name} 没有渲染")
                elif got != want:
                    bad.append(f"step {row['i']}({row['id']}): {name} 显示 {got} vs trace {want}")
        check("每一步渲染出的数值都逐位等于 trace 的 state（非抽样，全步覆盖）",
              not bad, "; ".join(bad[:4]))
        note(f"核对 {len(every)} 步 × 每步写入的张量")

        # ------------------------------------------------- DAG = variables
        print("\n== 图：节点 = 变量，边 = 一次计算 ==")
        model = page.evaluate("() => window.__vd.model")
        check("节点集就是 tensors 的键（不是 graph.nodes 的那些步骤）",
              [n for n in model["nodes"]] == tensor_names or
              set(model["nodes"]) == set(tensor_names),
              f"节点 {model['nodes']} vs trace tensors {tensor_names}")

        # Every drawn edge must be justified by a step that reads its source and
        # writes its target. This is the invariant that separates a real data
        # dependency from a line drawn between two boxes that happen to look
        # close, and it is checkable from the trace alone.
        justified, unjustified = [], []
        for e in model["edges"]:
            why = [s for s in trace["steps"]
                   if e["from"] in (s.get("reads") or []) and e["to"] in (s.get("writes") or [])]
            (justified if why else unjustified).append(e)
        check("每条画出来的边都有某一步「读 from、写 to」作为依据",
              not unjustified,
              json.dumps(unjustified, ensure_ascii=False))
        note("边：" + " · ".join(f"{e['from']}→{e['to']}({'+'.join(e['ops'])})"
                                 for e in model["edges"]))

        # The discriminating assertion. `reads[0]` is x_blk on every op step of
        # this trace, so a view that took the first read would draw x_blk→l and
        # x_blk→acc and would never show that m gates the rest. Naming m as the
        # source of l and of acc is what makes this picture the algorithm.
        def source_of(target):
            m = [e for e in model["edges"] if e["to"] == target]
            return m[0]["from"] if m else None
        check("ℓ 的更新来自 m（而不是 x_blk）—— 数据流画的是算法不是扇出",
              source_of("l") == "m", f"l ← {source_of('l')}")
        check("acc 的更新来自 m", source_of("acc") == "m", f"acc ← {source_of('acc')}")
        check("输出的归一化来自 acc", source_of("O") == "acc", f"O ← {source_of('O')}")
        check("暂存区由输入装载而来", source_of("x_blk") == "x", f"x_blk ← {source_of('x_blk')}")

        # The recurrences. `m ← max(m, ·)` is the shape of online softmax, so the
        # self-loops have to be drawn rather than silently dropped.
        loops = sorted(lp["id"] for lp in model["loops"])
        check("跨迭代递推画成自环（m / ℓ / acc 各自被反复更新）",
              loops == ["acc", "l", "m"], f"自环 = {loops}")

        # A step whose only reads are the staging buffer and whose write is a
        # scalar state would be a step the graph cannot place, so the trace's
        # own reads/writes are the check that nothing was invented.
        declared = set(tensor_names)
        stray = [e for e in model["edges"]
                 if e["from"] not in declared or e["to"] not in declared]
        check("边的端点都是已声明的张量", not stray, json.dumps(stray, ensure_ascii=False))

        # ------------------------------------------- the three node states
        # The defect this ticket fixes. The states were computed and then not
        # written into the SVG: every rect came out with a bare `n-box` class,
        # so the whole graph rendered at the same faint weight and a reader who
        # had walked five steps saw the same picture as at step zero.
        print("\n== 节点三态高亮（本票修掉的缺陷）==")
        page.evaluate("() => window.__vd.setCursor(4)")
        page.wait_for_timeout(250)
        nodes = page.evaluate(READ_NODES)
        step4 = trace["steps"][4]
        hot = set(step4["reads"]) | set(step4["writes"])

        cls_bad = [n["id"] for n in nodes["nodes"]
                   if n["cls"] not in ("vd-box vd-box-on", "vd-box vd-box-past",
                                       "vd-box vd-box-off")]
        check("每个节点的 rect class 恰好是三态之一（没有裸 vd-box）",
              not cls_bad, f"裸/异常的 class：{cls_bad}")

        states = {}
        for n in nodes["nodes"]:
            states.setdefault(n["state"], []).append(n["id"])
        check("三态同时出现在同一帧（当前 / 已访问 / 未到）",
              all(states.get(k) for k in ("on", "past", "off")),
              json.dumps({k: sorted(v) for k, v in states.items()}, ensure_ascii=False))

        check("「当前」态就是本步读写的变量",
              sorted(states.get("on", [])) == sorted(hot),
              f"on = {sorted(states.get('on', []))} vs 本步 reads∪writes = {sorted(hot)}")

        # The three states have to be DISTINGUISHABLE, not merely present. A
        # class that sets no colour is three names for one appearance.
        #
        # Read by STATE, one representative node each: keying by node id and
        # then indexing by state name happens to work only when a node with
        # that id exists, and silently mixes two nodes' properties otherwise.
        look = page.evaluate("""() => {
            const seen = {};
            document.querySelectorAll('[data-vd-node]').forEach(g => {
                const st = g.dataset.vdState;
                if (seen[st]) return;
                const cs = getComputedStyle(g.querySelector('.vd-box'));
                seen[st] = {node: g.dataset.vdNode, stroke: cs.stroke,
                            width: parseFloat(cs.strokeWidth), opacity: parseFloat(cs.opacity),
                            fill: cs.fill};
            });
            return seen;
        }""")
        check("三态的描边颜色两两不同（不是三个名字一种外观）",
              len({look[k]["stroke"] for k in ("on", "past", "off")}) == 3,
              json.dumps(look, ensure_ascii=False))
        check("未到的节点被压暗、当前的最重（强度有序）",
              look["off"]["opacity"] < look["past"]["opacity"] <= 1.0 and
              look["on"]["width"] > look["past"]["width"] > look["off"]["width"],
              f"opacity off={look['off']['opacity']} past={look['past']['opacity']}；"
              f"描边宽 on={look['on']['width']} past={look['past']['width']} "
              f"off={look['off']['width']}")

        # The highlight must not lag the cursor. A CSS transition on a
        # state-bearing property looks pleasant and is a lie for its duration:
        # the sidebar has moved and the graph has not. Asserted by stepping in
        # ONE task and reading back immediately -- if any transition is in play
        # the computed colour is still the previous step's at this instant.
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
            window.__vd.setCursor(9);
            const at9 = read();
            return {at0: at0, at9: at9};
        }""")
        check("高亮不落后于游标（同一次任务里跳转后立刻读，已是新状态）",
              lag["at0"]["m"] != lag["at9"]["m"] and
              lag["at9"]["acc"].startswith("on|"),
              json.dumps(lag, ensure_ascii=False))

        # Step 0 has no history, so nothing may be marked as visited.
        page.evaluate("() => window.__vd.setCursor(0)")
        page.wait_for_timeout(200)
        s0 = page.evaluate(READ_NODES)
        check("第 0 步没有任何「已访问」节点（进度是从零开始的）",
              not [n for n in s0["nodes"] if n["state"] == "past"],
              json.dumps([n["id"] for n in s0["nodes"] if n["state"] == "past"]))

        # Walking forward, the visited set grows monotonically: that is what
        # makes the picture say "you have moved".
        arc = page.evaluate("""() => {
            const out = [];
            for (let i = 0; i <= window.__vd.index.lastStep; i++) {
                window.__vd.setCursor(i);
                const s = {on: 0, past: 0, off: 0};
                document.querySelectorAll('[data-vd-node]').forEach(g => { s[g.dataset.vdState]++; });
                out.push(s);
            }
            return out;
        }""")
        grown = [r["past"] + r["on"] for r in arc]
        check("沿时间轴前进时「已触及的变量」单调不减（进度看得见）",
              all(grown[i] <= grown[i + 1] for i in range(len(grown) - 1)),
              f"逐步计数 {grown}")
        check("走到最后一步时所有变量都已被触及",
              grown[-1] == len(tensor_names), f"{grown[-1]} / {len(tensor_names)}")

        check("重绘走增量 class 而非 DOM 重建（节点身份保持不变）",
              page.evaluate("""() => {
                  window.__vd.setCursor(2);
                  const a = document.querySelector('.vd-dag [data-vd-node]');
                  const id = a.dataset.vdNode;
                  window.__vd.setCursor(5);
                  const b = document.querySelector('.vd-dag [data-vd-node]');
                  return a === b && b.dataset.vdNode === id;
              }"""))

        # ----------------------------------------- formula <-> graph highlight
        print("\n== 公式侧栏与双向高亮 ==")
        page.evaluate("() => window.__vd.setCursor(2)")
        page.wait_for_timeout(200)
        tiers = {}
        for tier in ("sym", "idx", "num"):
            page.click(f"[data-vd-tier='{tier}']")
            page.wait_for_timeout(200)
            tiers[tier] = page.evaluate("() => document.querySelector('.vd-fml').textContent")
        check("三档切换对同一份绑定表生效（三档文本互不相同且都非空）",
              all(tiers.values()) and len(set(tiers.values())) == 3,
              f"sym={tiers['sym']!r} idx={tiers['idx']!r} num={tiers['num']!r}")
        check("符号档渲染的是 LaTeX 片段而非字面源码",
              "\\" not in tiers["sym"], f"sym = {tiers['sym']!r}")

        step2_vals = page.evaluate("""() => Object.values(window.__labTrace.steps[2].bindings)
            .map(b => String(b.num))
            .filter(v => /^-?\\d+\\.\\d/.test(v))
            .map(v => v.replace(/^-/, ''))""")
        check("代入数值档确实代入了 trace 里的数值（而符号档没有）",
              bool(step2_vals) and any(v in tiers["num"] for v in step2_vals)
              and not any(v in tiers["sym"] for v in step2_vals),
              f"trace 值 {step2_vals} · num={tiers['num']!r}")

        page.evaluate("() => window.__vd.setCursor(3)")
        page.wait_for_timeout(250)
        region = page.evaluate("""() => {
            const el = document.querySelector('.vd-fml .lab-region');
            if (!el) return null;
            const cs = getComputedStyle(el);
            return {tag: el.tagName, bg: cs.backgroundColor, cls: el.className};
        }""")
        check("修正因子的公式片段被 \\region 真正框住（KaTeX 生成的真实元素）",
              region is not None and "region-CORR" in region["cls"],
              json.dumps(region, ensure_ascii=False))

        # The slot -> variable map must come from the trace, and it must name
        # the right tensors. The two cases a page-side guess cannot get right
        # are asserted by name below.
        bv = page.evaluate("() => window.__vd.model.steps.map(s => s.bindingVars)")
        check("每一步都带 binding_vars，且覆盖它的全部 slot",
              all(x is not None and set(x) == set(trace["steps"][i]["bindings"])
                  for i, x in enumerate(bv)),
              json.dumps([i for i, x in enumerate(bv)
                          if x is None or set(x) != set(trace["steps"][i]["bindings"])]))
        check("binding_vars 只指向已声明的张量",
              all(t in set(tensor_names) for x in bv for names in x.values() for t in names),
              "(全部端点都在 tensors 里)")

        def click_slot(slot):
            page.evaluate("""(slot) => {
                const el = [...document.querySelectorAll('.vd-fml [id*="-slot-"]')]
                    .find(e => e.id.endsWith('-slot-' + slot));
                if (el) el.dispatchEvent(new MouseEvent('click', {bubbles: true}));
            }""", slot)
            page.wait_for_timeout(220)

        def highlight_state():
            return page.evaluate("""() => ({
                pinned: [...document.querySelectorAll('[data-vd-node]')]
                    .filter(g => g.classList.contains('vd-n-pin')).map(g => g.dataset.vdNode),
                slotOn: [...document.querySelectorAll('.vd-fml [id*="-slot-"].vd-slot-on')]
                    .map(e => /-slot-(.+)$/.exec(e.id)[1]),
                chipOn: [...document.querySelectorAll('[data-vd-var].vd-chip-on')]
                    .map(e => e.dataset.vdVar),
            })""")

        # Direction 1: formula -> graph. Step 4 (a1) reads acc; its AOLD slot
        # names the accumulator.
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(4); }")
        page.wait_for_timeout(200)
        click_slot("AOLD")
        hs = highlight_state()
        check("点公式里的变量 → 图里对应节点亮（公式 → 图）",
              hs["pinned"] == ["acc"], json.dumps(hs, ensure_ascii=False))
        check("同一个变量的读写 chip 也一起亮（三处同亮）",
              set(hs["chipOn"]) == {"acc"}, json.dumps(hs, ensure_ascii=False))

        # Direction 2: graph -> formula.
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(4); }")
        page.wait_for_timeout(200)
        page.evaluate("""() => document.querySelector('[data-vd-node="m"]')
            .dispatchEvent(new MouseEvent('click', {bubbles: true}))""")
        page.wait_for_timeout(220)
        hs = highlight_state()
        check("点图里的节点 → 公式里所有提到它的变量都亮（图 → 公式）",
              hs["pinned"] == ["m"] and set(hs["slotOn"]) == {"MOLD", "MNEW", "WSUM"},
              json.dumps(hs, ensure_ascii=False))

        # The case a page-side regex cannot get right. The tensor is named `l`
        # but the formula writes it \ell, which contains no identifier `l`; a
        # scan of the rendered text finds nothing and the node never lights.
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(3); }")
        page.wait_for_timeout(200)
        click_slot("LOLD")
        hs = highlight_state()
        check("公式里的 \\ell 能点亮图里名为 l 的节点（正则做不到的那个情形）",
              hs["pinned"] == ["l"], json.dumps(hs, ensure_ascii=False))

        # And the near-miss the other way: `x` must not match inside `x_blk`.
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(2); }")
        page.wait_for_timeout(200)
        click_slot("BLKMAX")
        hs = highlight_state()
        check("公式里的 x 只点亮 x，不会误伤 x_blk（词边界）",
              hs["pinned"] == ["x"], json.dumps(hs, ensure_ascii=False))

        # A slot naming two variables lights both: it would be a lie about what
        # the formula reads to light only one.
        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(4); }")
        page.wait_for_timeout(200)
        click_slot("WSUM")
        hs = highlight_state()
        check("一个 slot 提到两个变量时两个都亮（不挑一个）",
              set(hs["pinned"]) == {"m", "x"}, json.dumps(hs, ensure_ascii=False))

        page.evaluate("() => { window.__vd.state.pinned = {}; window.__vd.setCursor(0); }")

        # ------------------------------------------------------- one screen
        # "每个知识点基本一屏" is a per-STEP promise: whatever step the cursor is
        # on — its graph, its formula, its values — fits without scrolling. The
        # engine's four-panel instrument layout cannot make that promise, which
        # is why the page is built around this view instead.
        print("\n== 一屏（每一步一屏，不是整个 lab 一屏）==")
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
        check("PC：每一步的图 + 公式都在一屏内（播放器自身不滚）",
              worst_root <= 0 and worst_side <= 0,
              f"最大溢出 根 {worst_root}px / 侧栏 {worst_side}px")
        check("PC：整个页面不滚（面板折在「参考面板」里，不撑高页面）",
              worst_doc <= 0, f"最大溢出 {worst_doc}px")
        check("PC：控制条在每一步都留在视野内",
              all(r["ctlVisible"] for r in fit),
              json.dumps([r["i"] for r in fit if not r["ctlVisible"]]))
        note(f"逐步实测 {len(fit)} 步 × 3 项")

        # "One screen" is not only about height. The num tier of the acc-update
        # steps substitutes the entire weighted sum and is the widest formula in
        # the lab; if it does not fit, the reader gets a formula that runs off
        # the edge and has to scroll the sidebar sideways to finish reading an
        # equation. Measured across every (tier, step) pair, not sampled.
        fml = page.evaluate("""() => {
            const out = [];
            for (const tier of ['sym', 'idx', 'num']) {
                window.__vd.setTier(tier);
                for (let i = 0; i <= window.__vd.index.lastStep; i++) {
                    window.__vd.setCursor(i);
                    const node = document.querySelector('.vd-fml .lab-formula-node');
                    out.push({tier: tier, i: i, over: node.scrollWidth - node.clientWidth});
                }
            }
            window.__vd.setTier('num');
            return out;
        }""")
        over = [x for x in fml if x["over"] > 0]
        check("PC：每一档、每一步的公式都横向放得下（不用左右拖动读公式）",
              not over, json.dumps(over[:5], ensure_ascii=False))
        note(f"公式排版核对 {len(fml)} 个（档位 × 步）组合")

        # ------------------------------------------------------- geometry
        print("\n== 几何：节点不重叠、不被裁、边不穿箱 ==")
        # A graph can be wrong in a way only the graph shows. The layout is
        # measured off the rendered SVG: node boxes pairwise disjoint, every
        # node inside the canvas, and no edge routed through a node it does not
        # belong to. This project has already shipped a diagram whose every
        # count was right and whose furniture overlapped.
        geo_bad = {"overlap": [], "clipped": [], "through": []}
        for i in range(len(trace["steps"])):
            page.evaluate("i => window.__vd.setCursor(i)", i)
            g = page.evaluate("""() => {
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
                    return {from: m[1], to: m[2], pts: pts};
                });
                const svgBox = svg.getBoundingClientRect();
                const texts = [...svg.querySelectorAll('text')].map(t => {
                    const b = t.getBoundingClientRect();
                    return {text: t.textContent, l: b.left, r: b.right, t: b.top, b: b.bottom};
                });
                return {canvas: {w: vb[2], h: vb[3]}, boxes: boxes, edges: edges,
                        texts: texts,
                        svg: {l: svgBox.left, r: svgBox.right, t: svgBox.top, b: svgBox.bottom}};
            }""")
            boxes = g["boxes"]
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    p, q = boxes[a], boxes[b]
                    if (p["x"] < q["x"] + q["w"] and q["x"] < p["x"] + p["w"] and
                            p["y"] < q["y"] + q["h"] and q["y"] < p["y"] + p["h"]):
                        geo_bad["overlap"].append([i, p["id"], q["id"]])
            for bx in boxes:
                if (bx["x"] < -0.5 or bx["y"] < -0.5 or
                        bx["x"] + bx["w"] > g["canvas"]["w"] + 0.5 or
                        bx["y"] + bx["h"] > g["canvas"]["h"] + 0.5):
                    geo_bad["clipped"].append([i, bx["id"]])
            # A path point inside a node other than its own endpoints is the
            # collision that a count-based assertion can never see.
            for e in g["edges"]:
                for bx in boxes:
                    if bx["id"] in (e["from"], e["to"]):
                        continue
                    for (px, py) in e["pts"]:
                        if (bx["x"] < px < bx["x"] + bx["w"] and
                                bx["y"] < py < bx["y"] + bx["h"]):
                            geo_bad["through"].append([i, e["from"] + "→" + e["to"], bx["id"]])
                            break
            # Text drawn outside the SVG is silently clipped, which reads as a
            # rendering glitch rather than as a layout bug.
            for t in g["texts"]:
                if (t["l"] < g["svg"]["l"] - 0.5 or t["r"] > g["svg"]["r"] + 0.5 or
                        t["t"] < g["svg"]["t"] - 0.5 or t["b"] > g["svg"]["b"] + 0.5):
                    geo_bad["clipped"].append([i, "text:" + t["text"]])
        check("节点矩形两两不重叠", not geo_bad["overlap"],
              json.dumps(geo_bad["overlap"][:4], ensure_ascii=False))
        check("每个节点与每段文字都在画布内（没有被 viewBox 裁掉）",
              not geo_bad["clipped"], json.dumps(geo_bad["clipped"][:4], ensure_ascii=False))
        check("没有边穿过与它无关的节点箱（不是「看着差不多」就连）",
              not geo_bad["through"], json.dumps(geo_bad["through"][:4], ensure_ascii=False))

        # Edge labels must not overlap a node box either: the label sits in the
        # gap and a mis-sized box puts it inside one.
        lbl_bad = page.evaluate("""() => {
            const svg = document.querySelector('.vd-dag');
            const boxes = [...svg.querySelectorAll('[data-vd-node]')].map(n =>
                n.querySelector('.vd-box').getBoundingClientRect());
            const out = [];
            svg.querySelectorAll('[data-vd-el]').forEach(t => {
                if (+getComputedStyle(t).opacity < 0.5) return;
                const b = t.getBoundingClientRect();
                boxes.forEach((q, i) => {
                    if (b.left < q.right && q.left < b.right &&
                        b.top < q.bottom && q.top < b.bottom) out.push(t.textContent);
                });
            });
            return out;
        }""")
        check("边的操作名不压在节点箱上", not lbl_bad, json.dumps(lbl_bad, ensure_ascii=False))

        # --- the geometry control group -----------------------------------
        # The checks above must be able to FAIL. A measurement that cannot see a
        # hand-placed violation is not measuring anything, so the control builds
        # one: it renders the real view off-screen, moves a node onto another,
        # pushes one past the canvas edge, and routes an edge through a node --
        # then requires the SAME predicates, evaluated the same way, to flag
        # each one.
        print("\n== 几何对照：人为造出违规，同一批谓词必须报警 ==")
        control = page.evaluate("""() => {
            const host = document.createElement('div');
            host.style.position = 'absolute';
            host.style.left = '-9999px';
            host.style.top = '0';
            document.body.appendChild(host);
            const v = LabEngine.variableDag.mount(host, window.__labTrace, {tier: 'num'});
            v.setCursor(4);
            const svg = host.querySelector('.vd-dag');

            const read = () => {
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

            const base = read();
            for (let a = 0; a < base.boxes.length; a++)
                for (let b = a + 1; b < base.boxes.length; b++)
                    if (overlaps(base.boxes[a], base.boxes[b]))
                        return {error: 'baseline already overlapping: ' +
                                base.boxes[a].id + '/' + base.boxes[b].id};

            // (1) a node pushed onto another
            const acc = svg.querySelector('[data-vd-node="acc"] .vd-box');
            const lBox = svg.querySelector('[data-vd-node="l"] .vd-box');
            const keep = {x: acc.getAttribute('x'), y: acc.getAttribute('y')};
            acc.setAttribute('x', lBox.getAttribute('x'));
            acc.setAttribute('y', lBox.getAttribute('y'));
            const after1 = read();
            const overlapBad = [];
            for (let a = 0; a < after1.boxes.length; a++)
                for (let b = a + 1; b < after1.boxes.length; b++)
                    if (overlaps(after1.boxes[a], after1.boxes[b]))
                        overlapBad.push(after1.boxes[a].id + '/' + after1.boxes[b].id);
            acc.setAttribute('x', keep.x);
            acc.setAttribute('y', keep.y);

            // (2) a node pushed past the canvas
            const xBox = svg.querySelector('[data-vd-node="O"] .vd-box');
            const keepOx = xBox.getAttribute('x');
            xBox.setAttribute('x', String(base.canvas.w + 40));
            const clipBad = read().boxes.filter(b => outside(b, base.canvas)).map(b => b.id);
            xBox.setAttribute('x', keepOx);

            // (3) an edge rerouted through an unrelated node
            const path = svg.querySelector('[data-vd-edge]');
            const keepD = path.getAttribute('d');
            const a = svg.querySelector('[data-vd-node="x"] .vd-box');
            const z = svg.querySelector('[data-vd-node="O"] .vd-box');
            const cx = n => +n.getAttribute('x') + (+n.getAttribute('width')) / 2;
            const cy = n => +n.getAttribute('y') + (+n.getAttribute('height')) / 2;
            path.setAttribute('d', 'M' + cx(a) + ' ' + cy(a) + ' L' + cx(z) + ' ' + cy(z));
            const throughBad = crosses(read().boxes, read().edges[0]);
            path.setAttribute('d', keepD);

            v.destroy();
            host.remove();
            return {overlapBad: overlapBad, clipBad: clipBad, throughBad: throughBad};
        }""")
        check("对照组：被推到另一个节点上的箱子会被「节点不重叠」规则抓到",
              bool(control.get("overlapBad")), json.dumps(control, ensure_ascii=False))
        check("对照组：被推出画布的节点会被「不被裁」规则抓到",
              control.get("clipBad") == ["O"], json.dumps(control, ensure_ascii=False))
        check("对照组：被改道穿过别的节点的边会被「边不穿箱」规则抓到",
              bool(control.get("throughBad")), json.dumps(control, ensure_ascii=False))

        # ------------------------------------------------------ arbitrary jump
        print("\n== 任意跳转（引擎契约，L00 仍然遵循）==")
        verdict = page.evaluate("() => window.__labVerify && window.__labVerify.L00")
        check("自检面板存在且跑过", verdict is not None)
        if verdict:
            j = verdict["jumps"]
            note(f"纯函数重建：{j['total']} 次跳转，{j['pureFailures']} 次不一致")
            note(f"对照组（故意做错的有状态播放器）：{j['controlFailures']} 次不一致")
            check("纯函数重建 0 次不一致", j["pureFailures"] == 0)
            check("对照组确实失败（证明上面的 0 不是测试写错）",
                  j["controlFailures"] > 0, f"{j['controlFailures']} 次")
            check("跳转集有区分力（conclusive）", j["conclusive"])
            check("lint 干净且 lint 本身是活的",
                  verdict["lint"]["passed"] and verdict["lint"]["missed"] == [],
                  f"clean gaps={verdict['lint']['gaps']}, sabotages caught={verdict['lint']['caught']}")

            plan = page.evaluate("() => LabEngine.verify.jumpPlan(%d).map(x => x.label)"
                                 % (page.evaluate("() => window.__vd.index.lastStep")))
            joined = " ".join(plan)
            check("跳转路径覆盖：正向", "正向" in joined)
            check("跳转路径覆盖：完全逆序", "完全逆序" in joined)
            check("跳转路径覆盖：跳回", "跳回" in joined)
            check("跳转路径覆盖：原地", "原地" in joined)
            check("跳转路径覆盖：乱序回环", "乱序" in joined and "回环" in joined)

        # The end-to-end version of the same claim, through the real view: what
        # is RENDERED after a jump must equal what is rendered after stepping to
        # the same place one at a time.
        e2e = page.evaluate("""() => {
            const V = window.__vd, last = V.index.lastStep;
            const read = () => {
                const out = [];
                document.querySelectorAll('.vd-vg').forEach(g => {
                    out.push([g.querySelector('.vd-vg-n').textContent,
                              [...g.querySelectorAll('.vd-cell')].map(c => c.textContent).join(',')]);
                });
                return JSON.stringify(out);
            };
            const paths = [['尾 -> 1', last, 1], ['1 -> 0', 1, 0], ['0 -> 尾', 0, last],
                           ['尾 -> 3', last, 3], ['3 -> 3', 3, 3], ['3 -> 0', 3, 0]];
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
            check(f"端到端：跳转 {c['label']} 的渲染 == 顺序播放到该步", c["same"])

        # ---------------------------------------------------------- player UX
        print("\n== 播放器交互 ==")
        page2 = browser.new_page(viewport={"width": W, "height": H})
        page2.goto(url)
        page2.wait_for_timeout(900)
        start = page2.evaluate("() => window.__vd.state.cursor")
        check("加载时停在第一步（不自动播放）",
              start == 0 and page2.evaluate("() => !window.__vd.state.playing"),
              f"cursor={start}")

        # deep link
        page2.goto(url + "?step=7")
        page2.wait_for_timeout(900)
        deep = page2.evaluate("() => window.__vd.state.cursor")
        check("?step=N 深链生效", deep == 7, f"cursor={deep}")
        deep_grid = page2.evaluate(READ_GRID)
        page2.evaluate("() => window.__vd.setCursor(0)")
        page2.wait_for_timeout(150)
        for _ in range(7):
            page2.evaluate("() => window.__vd.setCursor(window.__vd.state.cursor + 1)")
        page2.wait_for_timeout(200)
        seq_grid = page2.evaluate(READ_GRID)
        check("深链渲染与顺序播放到该步完全一致",
              json.dumps(deep_grid, sort_keys=True) == json.dumps(seq_grid, sort_keys=True))

        # replaceState, not pushState
        hist = page2.evaluate("""() => {
            const before = history.length;
            window.__vd.setCursor(2); window.__vd.setCursor(5); window.__vd.setCursor(8);
            return {before: before, after: history.length, url: location.search};
        }""")
        check("拖动游标用 replaceState，不污染历史栈",
              hist["before"] == hist["after"], json.dumps(hist))

        # keyboard
        page2.evaluate("() => { window.__vd.setCursor(0); document.body.focus(); }")
        page2.keyboard.press("ArrowRight")
        page2.wait_for_timeout(150)
        right = page2.evaluate("() => window.__vd.state.cursor")
        page2.keyboard.press("ArrowLeft")
        page2.wait_for_timeout(150)
        left = page2.evaluate("() => window.__vd.state.cursor")
        check("左右箭头单步", right == 1 and left == 0, f"right={right} left={left}")

        scroll_before = page2.evaluate("() => window.scrollY")
        page2.keyboard.press("Space")
        page2.wait_for_timeout(150)
        space_playing = page2.evaluate("() => window.__vd.state.playing")
        scroll_after = page2.evaluate("() => window.scrollY")
        check("空格播放/暂停且不触发页面滚动",
              space_playing and scroll_before == scroll_after,
              f"playing={space_playing} scroll {scroll_before}->{scroll_after}")
        page2.keyboard.press("Space")
        page2.wait_for_timeout(150)
        check("再按空格暂停", not page2.evaluate("() => window.__vd.state.playing"))

        page2.keyboard.press("End")
        page2.wait_for_timeout(200)
        end = page2.evaluate("() => window.__vd.state.cursor")
        page2.keyboard.press("Home")
        page2.wait_for_timeout(200)
        home = page2.evaluate("() => window.__vd.state.cursor")
        last = page2.evaluate("() => window.__vd.index.lastStep")
        check("Home/End 到首尾", home == 0 and end == last, f"home={home} end={end} last={last}")

        # -------------------------------------------- correction amplifier
        print("\n== 修正因子放大器（#14 专用视图）==")
        rescale = page.evaluate("""() => {
            const T = window.__labTrace;
            for (let i = 0; i < T.steps.length; i++) {
                const c = T.steps[i].corr;
                if (c && c.kind === 'rescale') return {i: i, corr: c};
            }
            return null;
        }""")
        check("默认配置里存在 m 被抬高、必须缩放的步",
              rescale is not None,
              f"第一个 rescale 步 = {rescale['i'] if rescale else None}")

        if rescale:
            page.evaluate("i => window.__vd.setCursor(i)", rescale["i"])
            page.wait_for_timeout(300)
            amp = page.evaluate("""() => {
                const host = document.querySelector('[data-panel-body=amplifier]');
                if (!host) return null;
                return {
                    text: host.textContent.replace(/\\s+/g, ' '),
                    factor: (host.querySelector('.l00-corr-factor') || {}).textContent || '',
                    tag: (host.querySelector('.l00-tag') || {}).textContent || '',
                    cells: [...host.querySelectorAll('.l00-vs-cell')].map(c => ({
                        k: (c.querySelector('.l00-vs-k') || {}).textContent || '',
                        v: (c.querySelector('.l00-vs-v') || {}).textContent || '',
                    })),
                };
            }""")
            check("放大器面板渲染（因子 / 标签 / 对比格都在）",
                  amp is not None and amp["factor"] and amp["cells"],
                  (amp or {}).get("tag", ""))

            got_factor = float(amp["factor"])
            want_factor = rescale["corr"]["factor"]
            check("高亮显示的因子就是 trace 里的 e^(m_old − m_new)",
                  abs(got_factor - want_factor) <= 5e-4 * max(1e-9, abs(want_factor)),
                  f"页面 {got_factor} vs trace {want_factor!r}")
            check("该步被标记为 rescale（m 确实被抬高了）",
                  "缩放" in amp["tag"], amp["tag"])

            def disp4(v):
                if isinstance(v, float):
                    return float(f"{v:.4g}")
                return v

            cells = {c["k"]: c["v"] for c in amp["cells"]}
            corrected_key = next((k for k in cells if "带上修正因子" in k), None)
            uncorrected_key = next((k for k in cells if "漏掉因子" in k), None)
            bias_key = next((k for k in cells if k.strip() == "偏差"), None)
            check("放大器同时给出「修正后」与「不修正」两个数值",
                  corrected_key and uncorrected_key and bias_key,
                  f"格 = {list(cells)}")
            if corrected_key and uncorrected_key:
                check("两个数值与 trace 的 corr.o_correct / o_uncorrected 一致",
                      abs(float(cells[corrected_key]) -
                          disp4(rescale["corr"]["o_correct"])) <= 5e-4 *
                          max(1e-9, abs(rescale["corr"]["o_correct"])) and
                      abs(float(cells[uncorrected_key]) -
                          disp4(rescale["corr"]["o_uncorrected"])) <= 5e-4 *
                          max(1e-9, abs(rescale["corr"]["o_uncorrected"])),
                      f"页面 {cells[corrected_key]} / {cells[uncorrected_key]} vs trace "
                      f"{rescale['corr']['o_correct']:.4g} / "
                      f"{rescale['corr']['o_uncorrected']:.4g}")
            if bias_key:
                check("偏差数值 = 两个数值之差的真实结果（非零且非手编）",
                      abs(float(cells[bias_key]) - disp4(rescale["corr"]["bias_abs"])) <=
                      5e-4 * max(1e-9, abs(rescale["corr"]["bias_abs"])) and
                      rescale["corr"]["bias_abs"] > 0,
                      f"页面 {cells[bias_key]} vs trace {rescale['corr']['bias_abs']:.4g}")

        # -------------------------------------------- correction aggregate
        agg = page.evaluate("""() => {
            const c = window.__labTrace.meta.correction;
            const host = document.querySelector('[data-panel-body=amplifier]').textContent
                .replace(/\\s+/g, ' ');
            return {rescales: c.rescales, nblocks: window.__labTrace.meta.config.n_blocks,
                    final_bias: c.final_bias_abs, host: host};
        }""")
        check("放大器给出整条 trace 的最终偏差（不修正的累计代价）",
              f"{agg['final_bias']:.4g}" in agg["host"] or
              f"{agg['final_bias']}" in agg["host"],
              f"final_bias = {agg['final_bias']!r}")
        check("「几块真的做了缩放」与 trace 一致",
              f"{agg['rescales']} / {agg['nblocks']}" in agg["host"],
              f"{agg['rescales']} / {agg['nblocks']}")

        # -------------------------------------------------- comparison mode
        print("\n== 对照模式（三轨迹并排）==")
        page.evaluate("() => window.__vd.setCursor(window.__vd.index.lastStep)")
        page.wait_for_timeout(350)
        tracks = page.evaluate("""() => {
            const host = document.querySelector('[data-panel-body=compare]');
            if (!host) return null;
            return [...host.querySelectorAll('.l00-track')].map(t => ({
                cls: t.className,
                title: (t.querySelector('.l00-track-t') || {}).textContent || '',
                running: (t.querySelector('.l00-track-v') || {}).textContent || '',
                final: ((t.querySelector('.l00-track-final b') || {}).textContent) || '',
                badge: (t.querySelector('.l00-badge-ok, .l00-badge-bad') || {}).textContent || '',
                sub: (t.querySelector('.l00-track-sub') || {}).textContent || '',
            }));
        }""")
        check("三条轨迹并排渲染（朴素 / 安全 / online）",
              tracks is not None and len(tracks) == 3, f"{len(tracks) if tracks else 0} 条")
        if tracks:
            note(" · ".join(f"{t['title'].split('（')[0]}: 跑动={t['running']} 终值={t['final']}"
                            for t in tracks))
            naive, safe, online = tracks
            check("朴素版确实展示了数值溢出（NaN 或 ∞，不是有限值）",
                  any(s in naive["final"] for s in ("NaN", "∞", "nan", "inf")),
                  f"朴素版终值 {naive['final']!r}")
            check("朴素版被打上「数值溢出」标记（不是静默显示一个数）",
                  "溢出" in naive["badge"], naive["badge"])
            check("安全版与 online 版给出有限、相同的正确终值",
                  safe["final"] == online["final"] and
                  not any(s in safe["final"] for s in ("NaN", "∞")),
                  f"安全 {safe['final']!r} vs online {online['final']!r}")
            check("读数代价可见：安全版 3 遍、其余各 1 遍",
                  "3 遍" in safe["sub"] and "1 遍" in naive["sub"] and "1 遍" in online["sub"],
                  f"{naive['sub']} | {safe['sub']} | {online['sub']}")

            check("三行终值与 trace 里 compare.methods[].final 逐条对应",
                  page.evaluate("""() => {
                      const T = window.__labTrace;
                      const shown = [...document.querySelectorAll(
                          '[data-panel-body=compare] .l00-track-final b')]
                          .map(e => e.textContent);
                      const fmt = v => {
                          if (typeof v !== 'number') return v === 'NaN' ? 'NaN' : '∞';
                          return v.toPrecision(6)
                              .replace(/(\\.\\d*?)0+$/, '$1').replace(/\\.$/, '');
                      };
                      return T.compare.methods.every((m, i) => shown[i] === fmt(m.final));
                  }"""),
                  f"页面终值 = {[t['final'] for t in tracks]}")

        ctrl2 = page.evaluate("""() => {
            const host = document.querySelector('[data-panel-body=compare]');
            const si = window.__labTrace.compare.shift_invariance;
            const T = window.__labTrace;
            const method = T.compare.methods.find(m => m.id === 'naive');
            const text = host.textContent.replace(/\\s+/g, ' ');
            return {unshifted: si.unshifted_final, shifted: si.shifted_final,
                    reference: si.reference, text: text,
                    naive_final: method.final};
        }""")
        check("对照组存在：同一份朴素实现不加偏移时算对（证明溢出是量级问题）",
              isinstance(ctrl2["unshifted"], float) and
              abs(ctrl2["unshifted"] - ctrl2["reference"]) < 1e-9,
              f"不加偏移 {ctrl2['unshifted']!r} vs 参考 {ctrl2['reference']!r}")
        check("对照组被写进面板（读者能看到「代码一个字没改」）",
              "代码一个字没改" in ctrl2["text"] or "只改量级" in ctrl2["text"])
        check("对照组的偏移结果与朴素行一致（都是 NaN）",
              ctrl2["shifted"] == ctrl2["naive_final"] == "NaN",
              f"shifted={ctrl2['shifted']!r} naive={ctrl2['naive_final']!r}")

        page.evaluate("() => window.__vd.setCursor(0)")
        page.wait_for_timeout(300)
        early = page.evaluate("""() => {
            const ts = [...document.querySelectorAll('[data-panel-body=compare] .l00-track')];
            return {sub: ts[2].querySelector('.l00-track-sub').textContent,
                    w: ts[2].querySelector('.l00-progress > span').style.width};
        }""")
        page.evaluate("() => window.__vd.setCursor(window.__vd.index.lastStep)")
        page.wait_for_timeout(300)
        late = page.evaluate("""() => {
            const ts = [...document.querySelectorAll('[data-panel-body=compare] .l00-track')];
            return {sub: ts[2].querySelector('.l00-track-sub').textContent,
                    w: ts[2].querySelector('.l00-progress > span').style.width};
        }""")
        check("对照面板随游标推进（online 行的进度条与文字跟着 k 变）",
              early["w"] != late["w"] and "尚在扫描中" in early["sub"] and
              "已跑完" in late["sub"],
              f"{early['w']} -> {late['w']}")

        # -------------------------------------------------- parameter sliders
        print("\n== 参数滑杆（N / Bc / T）==")
        sliders = page.evaluate("""() => [...document.querySelectorAll('input[data-param]')]
            .map(i => ({p: i.dataset.param, min: +i.min, max: +i.max, v: +i.value}))""")
        check("三根滑杆都在（N / Bc / T）",
              {s["p"] for s in sliders} == {"N", "Bc", "T"}, json.dumps(sliders))
        check("滑杆量程来自 manifest（不是写死的）",
              page.evaluate("""() => {
                  const m = window.LabTraceSets['online-softmax'];
                  return [...document.querySelectorAll('input[data-param]')].every(i => {
                      const opts = m.params[i.dataset.param];
                      return +i.max === opts.length - 1;
                  });
              }"""))

        def set_slider(param, index):
            page.evaluate("""([p, i]) => {
                const el = document.querySelector(`input[data-param=${p}]`);
                el.value = i;
                el.dispatchEvent(new Event('input', {bubbles: true}));
            }""", [param, index])
            page.wait_for_timeout(550)

        def page_config():
            # `x` is read off the DOM at the step that first reads it, not off
            # the first grid on screen: at step 0 the first grid is `m`, whose
            # value is −∞ for every configuration, so comparing it would compare
            # a constant and pass on a page that never reloaded anything.
            return page.evaluate("""() => {
                const T = window.__labTrace;
                let li = 0;
                for (let i = 0; i < T.steps.length; i++) {
                    if ((T.steps[i].reads || []).indexOf('x') !== -1) { li = i; break; }
                }
                window.__vd.setCursor(li);
                const g = [...document.querySelectorAll('.vd-vg')]
                    .find(e => e.querySelector('.vd-vg-n').textContent === 'x');
                return {
                    cfg: window.__labConfig,
                    config: T.meta.config,
                    steps: T.steps.length,
                    nodes: Object.keys(T.tensors).length,
                    x: g ? [...g.querySelectorAll('.vd-cell')].map(c => c.textContent).join(',') : '',
                };
            }""")

        before = page_config()
        note(f"默认 = {before['cfg']}（{before['steps']} 步）")

        n_idx = page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const cur = window.__labTrace.meta.config.N;
            return m.params.N.findIndex(v => v !== cur);
        }""")
        set_slider("N", n_idx)
        after_n = page_config()
        check("改 N 后换到了另一份 trace（配置、步数都变了）",
              after_n["config"]["N"] != before["config"]["N"] and
              after_n["cfg"] != before["cfg"] and
              after_n["steps"] != before["steps"],
              f"{before['cfg']} -> {after_n['cfg']}（{after_n['steps']} 步）")

        bc_idx = page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const c = window.__labTrace.meta.config;
            return m.params.Bc.findIndex(v => v !== c.Bc && c.N % v === 0);
        }""")
        set_slider("Bc", bc_idx)
        after_bc = page_config()
        check("改块大小 Bc 后同样重新载入（块数随之改变）",
              after_bc["config"]["Bc"] != after_n["config"]["Bc"] and
              after_bc["config"]["n_blocks"] != after_n["config"]["n_blocks"],
              f"Bc {after_n['config']['Bc']} -> {after_bc['config']['Bc']}，"
              f"块数 {after_n['config']['n_blocks']} -> {after_bc['config']['n_blocks']}")

        t_idx = page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const cur = window.__labTrace.meta.config.T;
            return m.params.T.findIndex(v => v !== cur);
        }""")
        set_slider("T", t_idx)
        after_t = page_config()
        check("改温度 T 后 trace 换了，且输入按 x/T 缩放",
              after_t["config"]["T"] != after_bc["config"]["T"] and
              after_t["x"] != after_bc["x"],
              f"T {after_bc['config']['T']} -> {after_t['config']['T']}")

        # Whatever configuration is loaded, the VIEW must have been rebuilt
        # around it: the graph is derived from that trace's tensors and steps,
        # and the panels are the ones for that trace.
        rebuilt = page.evaluate("""() => {
            const T = window.__labTrace, c = T.meta.correction;
            const host = document.querySelector('[data-panel-body=compare]');
            const fmt = (v, d) => typeof v === 'number'
                ? v.toPrecision(d || 4).replace(/(\\.\\d*?)0+$/, '$1').replace(/\\.$/, '')
                : (v === 'NaN' ? 'NaN' : '∞');
            const amp = document.querySelector('[data-panel-body=amplifier]').textContent
                .replace(/\\s+/g, ' ');
            const cmp = host.textContent.replace(/\\s+/g, ' ');
            const si = T.compare.shift_invariance;
            const finals = [...host.querySelectorAll('.l00-track-final b')].map(e => e.textContent);
            return {
                cfg: window.__labConfig,
                ampHas: amp.includes(fmt(c.final_bias_abs)),
                cmpHas: cmp.includes(fmt(si.unshifted_final, 6)),
                cmpFinals: finals,
                wantFinals: T.compare.methods.map(m => fmt(m.final, 6)),
                viewNodes: window.__vd.model.nodes.slice().sort(),
                traceTensors: Object.keys(T.tensors).sort(),
                graphNodes: document.querySelectorAll('.vd-dag [data-vd-node]').length,
                panels: document.querySelectorAll('[data-panel-body]').length,
                roots: document.querySelectorAll('.vd-root').length,
                hosts: document.querySelectorAll('#lab-stage > .vd-host').length,
            };
        }""")
        check("换配置后两个参考面板跟着重建（数值来自新 trace，不是旧的那份）",
              rebuilt["ampHas"] and rebuilt["cmpHas"] and
              rebuilt["cmpFinals"] == rebuilt["wantFinals"],
              json.dumps({k: rebuilt[k] for k in ("cfg", "ampHas", "cmpHas")}, ensure_ascii=False))
        check("换配置后变量图重建在新 trace 的张量上",
              rebuilt["viewNodes"] == rebuilt["traceTensors"] and
              rebuilt["graphNodes"] == len(rebuilt["traceTensors"]),
              f"{rebuilt['graphNodes']} 节点 vs {len(rebuilt['traceTensors'])} 张量")
        check("换配置后没有累积出多余的播放器实例（DOM 每次重建）",
              rebuilt["panels"] == 3 and rebuilt["roots"] == 1 and rebuilt["hosts"] == 1,
              json.dumps({k: rebuilt[k] for k in ("panels", "roots", "hosts")}))

        coverage = page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const missing = m.traces.filter(t => !window.LabTraces[t]);
            return {n: m.traces.length, missing: missing, default: m.default,
                    hasDefault: !!window.LabTraces[m.default]};
        }""")
        check("manifest 列出的每个配置都真的被内联进了页面",
              not coverage["missing"] and coverage["hasDefault"],
              f"{coverage['n']} 个配置，缺 {coverage['missing']}")

        page3cfg = browser.new_page(viewport={"width": W, "height": H})
        page3cfg.goto(url + "?cfg=online-softmax-N4-B2-T0p5")
        page3cfg.wait_for_timeout(900)
        dl = page3cfg.evaluate("""() => ({
            cfg: window.__labConfig,
            config: window.__labTrace.meta.config,
            slider: [...document.querySelectorAll('input[data-param]')]
                .map(i => i.dataset.param + '=' + i.value).join(','),
            nodes: document.querySelectorAll('.vd-dag [data-vd-node]').length,
        })""")
        check("?cfg= 深链载入指定配置，滑杆位置同步",
              dl["cfg"] == "online-softmax-N4-B2-T0p5" and
              dl["config"]["N"] == 4 and dl["config"]["Bc"] == 2 and dl["config"]["T"] == 0.5,
              json.dumps(dl, ensure_ascii=False))

        page3cfg.goto(url + "?cfg=online-softmax-N999-B1-T1")
        page3cfg.wait_for_timeout(800)
        bad = page3cfg.evaluate("""() => ({
            cfg: window.__labConfig,
            banner: document.getElementById('lab-banner').textContent,
            hidden: document.getElementById('lab-banner').hidden,
        })""")
        check("非法的 ?cfg= 会在页面上说明并回落到默认配置",
              bad["cfg"] == coverage["default"] and not bad["hidden"] and
              "N999" in bad["banner"],
              json.dumps(bad, ensure_ascii=False))
        page3cfg.close()

        page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const parsed = /^online-softmax-N(\\d+)-B(\\d+)-T(.+)$/.exec(m.default);
            const want = {N: +parsed[1], Bc: +parsed[2], T: +parsed[3].replace('p', '.')};
            ['N', 'Bc', 'T'].forEach(k => {
                const el = document.querySelector(`input[data-param=${k}]`);
                el.value = m.params[k].indexOf(want[k]);
                el.dispatchEvent(new Event('input', {bubbles: true}));
            });
        }""")
        page.wait_for_timeout(700)
        check("回到 manifest 的默认配置（截图基准）",
              page.evaluate("() => window.__labConfig") == coverage["default"],
              page.evaluate("() => window.__labConfig"))

        # ------------------------------------------------------------ phone
        print("\n== 手机：真适配（不是降级）==")
        phone = browser.new_page(viewport={"width": PHONE_W, "height": PHONE_H},
                                 device_scale_factor=2, is_mobile=True, has_touch=True)
        phone_console = []
        phone.on("pageerror", lambda e: phone_console.append(str(e)))
        phone.goto(url)
        phone.wait_for_timeout(1100)
        mob = phone.evaluate("""() => {
            const root = document.querySelector('.vd-root');
            const main = document.querySelector('.vd-main');
            const ctl = document.querySelector('.vd-ctl').getBoundingClientRect();
            return {
                rows: getComputedStyle(main).gridTemplateRows.split(' ').length,
                rootOverflow: root.scrollHeight - root.clientHeight,
                docOverflow: document.documentElement.scrollHeight - window.innerHeight,
                ctlVisible: ctl.bottom <= window.innerHeight + 1 && ctl.top >= 0,
                graphH: Math.round(document.querySelector('.vd-dagwrap').getBoundingClientRect().height),
                hasView: !!document.querySelector('.vd-dag'),
                hasFormula: !!document.querySelector('.vd-fml .lab-formula-node'),
                hasFallback: !!document.querySelector('.lab-narrow'),
                orientation: document.querySelector('.vd-dag').dataset.vdOrientation,
            };
        }""")
        check("手机上渲染的是同一个视图（不是「需要更宽的屏幕」降级卡片）",
              mob["hasView"] and mob["hasFormula"] and not mob["hasFallback"],
              json.dumps(mob, ensure_ascii=False))
        check("手机上两半改为上下堆叠（真排版，不是缩小的 PC 版）",
              mob["rows"] == 2, f"grid 行数 = {mob['rows']}")
        check("手机上图仍有可读高度（不是被压成一条）",
              mob["graphH"] >= 140, f"图高 {mob['graphH']}px")

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
        check("手机上没有 JS 错误", not phone_console, str(phone_console[:3]))

        phone.evaluate("() => window.__vd.setCursor(2)")
        phone.wait_for_timeout(300)

        # ---------------------------------------------------------- site bits
        print("\n== 站点 ==")
        pf = page.evaluate("""() => {
            const b = document.body;
            return {onBody: b.hasAttribute('data-pagefind-ignore'),
                    onMeta: !!document.querySelector('meta[data-pagefind-ignore]')};
        }""")
        check("data-pagefind-ignore 包在 <body> 上", pf["onBody"], json.dumps(pf))

        assets = page.evaluate("""() => {
            const out = [];
            document.querySelectorAll('link[rel=stylesheet][href], script[src]').forEach(el => {
                const u = el.getAttribute('href') || el.getAttribute('src');
                if (u && !/^[a-z]+:|^\\/\\/|^#/.test(u)) out.push(u);
            });
            return out;
        }""")
        check("所有本地资源用 ./assets/ 引用（暂存后会拍平目录，../ 会 404）",
              len(assets) >= 5 and all(p.startswith("./assets/") for p in assets),
              f"{len(assets)} 个资源引用，全部 ./assets/")
        check("变量图视图的文件名遵循「一个 .js + 一个同名 .css」",
              "./assets/engine/views/variable-dag.js" in assets and
              "./assets/engine/views/variable-dag.css" in assets,
              json.dumps(assets, ensure_ascii=False))

        # ------------------------------------------------------------- shots
        print("\n== 截图 ==")
        for step, name in ((0, "70-l00-dag-step0"), (4, "71-l00-dag-step4"),
                           (7, "72-l00-dag-step7"), (9, "73-l00-dag-step9")):
            page.evaluate("i => window.__vd.setCursor(i)", step)
            page.wait_for_timeout(320)
            page.screenshot(path=SHOTS / f"{name}.png")
        page.locator(".vd-dagwrap").screenshot(path=SHOTS / "74-l00-graph-only.png")
        page.evaluate("() => window.__vd.setCursor(4)")
        page.wait_for_timeout(250)
        page.locator(".vd-side").screenshot(path=SHOTS / "75-l00-formula-sidebar.png")

        # The highlight, captured rather than only asserted: a slot clicked in
        # the formula and the node it names.
        page.evaluate("""() => {
            window.__vd.state.pinned = {};
            const el = [...document.querySelectorAll('.vd-fml [id*="-slot-"]')]
                .find(e => /-slot-AOLD$/.test(e.id));
            if (el) el.dispatchEvent(new MouseEvent('click', {bubbles: true}));
        }""")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "76-l00-highlight-linked.png")

        # Both themes, from pages that actually requested them.
        for name, scheme in (("77-l00-full-dark.png", "dark"),
                             ("78-l00-full-light.png", "light")):
            p_theme = browser.new_page(viewport={"width": W, "height": H},
                                       color_scheme=scheme)
            p_theme.goto(url)
            p_theme.wait_for_timeout(900)
            p_theme.evaluate("() => window.__vd.setCursor(4)")
            p_theme.wait_for_timeout(300)
            p_theme.screenshot(path=SHOTS / name)
            p_theme.close()

        # Phone shots, several steps, so a reviewer can see the responsive form
        # without running the harness.
        for step, name in ((0, "80-l00-phone-step0"), (4, "81-l00-phone-step4"),
                           (8, "82-l00-phone-step8")):
            phone.evaluate("i => window.__vd.setCursor(i)", step)
            phone.wait_for_timeout(320)
            phone.screenshot(path=SHOTS / f"{name}.png")
        phone.evaluate("""() => {
            window.__vd.state.pinned = {};
            window.__vd.setCursor(3);
            const el = [...document.querySelectorAll('.vd-fml [id*="-slot-"]')]
                .find(e => /-slot-LOLD$/.test(e.id));
            if (el) el.dispatchEvent(new MouseEvent('click', {bubbles: true}));
        }""")
        phone.wait_for_timeout(320)
        phone.screenshot(path=SHOTS / "83-l00-phone-highlight.png")
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
        print(f"{len(failures)} 项未通过：")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"全部通过（{len(notes)} 项测量）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
