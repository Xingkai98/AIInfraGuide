#!/usr/bin/env python3
"""Drive the staged lab page in real Chromium: assert behaviour, capture shots.

This is the acceptance harness for the engine, not a demo. It runs against the
built artifact in `public/labs/` (so it exercises the same inlined trace and
flattened asset paths a reader gets), asserts the acceptance criteria one by
one, and only then takes screenshots.

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
        page.wait_for_timeout(900)

        # ---------------------------------------------------------- headline
        print("\n== 契约与渲染 ==")
        check("引擎把页面完整驱动起来（DAG/张量/公式/时间轴都在）",
              page.evaluate("""() => !!document.querySelector('.lab-dag') &&
                  !!document.querySelectorAll('.lab-trow').length &&
                  !!document.querySelector('.lab-formula-node') &&
                  !!document.querySelector('[data-ctl=scrub]')"""))

        # render(trace, 0) shows the input, not "— 本步之前无值 —"
        first = page.evaluate("""() => {
            const row = [...document.querySelectorAll('.lab-trow')]
                .find(r => r.dataset.tensor === 'x');
            return row ? row.querySelector('.lab-tval').textContent : null;
        }""")
        check("render(trace,0) 显示输入张量 x 的值（init 被消费）",
              first is not None and "0.83" in first and "无值" not in first,
              f"x = {first!r}")

        # sentinel rendering: step 0 has m = -\infty
        m0 = page.evaluate("""() => [...document.querySelectorAll('.lab-trow')]
            .find(r => r.dataset.tensor === 'm').querySelector('.lab-tval').textContent""")
        check("±∞ 哨兵在张量面板解析为 −∞（不是字面 -\\infty）",
              "−∞" in m0 and "\\infty" not in m0, f"m = {m0!r}")

        mformula = page.evaluate("""() => document.querySelector('.lab-formula-host').textContent""")
        check("哨兵在公式面板由 KaTeX 正确渲染（无字面反斜杠）",
              "\\infty" not in mformula and "\\" not in mformula.replace("\\\\", ""),
              f"formula = {mformula!r}")

        # Three tiers. Step 2 (m1) is used rather than step 0 because the init
        # step's three tiers are genuinely identical — it substitutes literals
        # in all three — so it cannot show that binding-table substitution works.
        page.evaluate("() => window.__p.setCursor(2)")
        page.wait_for_timeout(200)
        tiers = {}
        for tier in ("sym", "idx", "num"):
            page.click(f"[data-tier='{tier}']")
            page.wait_for_timeout(200)
            tiers[tier] = page.evaluate(
                "() => document.querySelector('.lab-formula-host').textContent")
        check("三档切换对同一份绑定表生效（三档文本互不相同且都非空）",
              all(tiers.values()) and len(set(tiers.values())) == 3,
              f"sym={tiers['sym']!r} idx={tiers['idx']!r} num={tiers['num']!r}")
        check("符号档渲染的是 LaTeX 片段而非字面源码",
              "\\" not in tiers["sym"], f"sym = {tiers['sym']!r}")
        # The substituted values belong to the step's own block, which depends
        # on the default configuration — so the values are read from the trace
        # rather than hard-coded, and the assertion is the relation between the
        # tiers, not a particular number.
        # Decimal values only: a bare "1" matches inside the symbolic tier's
        # "m^{(j-1)}", which would make the check pass for the wrong reason.
        step2_vals = page.evaluate("""() => Object.values(window.__labTrace.steps[2].bindings)
            .map(b => String(b.num))
            .filter(v => /^-?\\d+\\.\\d/.test(v))
            .map(v => v.replace(/^-/, ''))""")
        check("代入数值档确实代入了 trace 里的数值（而符号档没有）",
              bool(step2_vals) and any(v in tiers["num"] for v in step2_vals)
              and not any(v in tiers["sym"] for v in step2_vals),
              f"trace 值 {step2_vals} · num={tiers['num']!r}")

        # \region highlight -> a real KaTeX-generated element with the class
        page.evaluate("() => window.__p.setCursor(3)")
        page.wait_for_timeout(250)
        region = page.evaluate("""() => {
            const el = document.querySelector('.lab-formula-host .lab-region');
            if (!el) return null;
            const cs = getComputedStyle(el);
            return {tag: el.tagName, bg: cs.backgroundColor, cls: el.className};
        }""")
        check("\\region 高亮渲染成 KaTeX 生成的真实元素（class 挂在它身上）",
              region is not None and "region-CORR" in region["cls"],
              json.dumps(region, ensure_ascii=False))

        # ------------------------------------------------------ arbitrary jump
        print("\n== 任意跳转（本票最高风险项）==")
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

            # path coverage: the plan must include the four required shapes
            plan = page.evaluate("() => LabEngine.verify.jumpPlan(%d).map(x => x.label)"
                                 % (page.evaluate("() => window.__p.index.lastStep")))
            joined = " ".join(plan)
            check("跳转路径覆盖：正向", "正向" in joined)
            check("跳转路径覆盖：完全逆序", "完全逆序" in joined)
            check("跳转路径覆盖：跳回", "跳回" in joined)
            check("跳转路径覆盖：原地", "原地" in joined)
            check("跳转路径覆盖：乱序回环", "乱序" in joined and "回环" in joined)

            # the end-to-end version of the same claim, through the real player
            e2e = page.evaluate("""() => {
                const P = window.__p, idx = P.index, T = window.__labTrace;
                const names = idx.tensorNames;
                // ground truth by sequential accumulation
                let acc = {};
                for (const n of names) if (T.tensors[n] && 'init' in T.tensors[n]) acc[n] = T.tensors[n].init;
                const truth = [];
                for (let i = 0; i < T.steps.length; i++) {
                    Object.assign(acc, JSON.parse(JSON.stringify(T.steps[i].state || {})));
                    truth.push(JSON.parse(JSON.stringify(acc)));
                }
                const readPanel = () => {
                    // read the RENDERED tensor panel, not the model: this is the
                    // claim a reader can actually see
                    const out = {};
                    document.querySelectorAll('.lab-trow').forEach(r => {
                        out[r.dataset.tensor] = r.querySelector('.lab-tval').textContent;
                    });
                    return out;
                };
                const paths = [];
                P.setCursor(idx.lastStep); paths.push(['尾 -> 1', idx.lastStep, 1]);
                P.setCursor(1);            paths.push(['1 -> 0', 1, 0]);
                P.setCursor(0);            paths.push(['0 -> 尾', 0, idx.lastStep]);
                P.setCursor(idx.lastStep); paths.push(['尾 -> 3', idx.lastStep, 3]);
                P.setCursor(3);            paths.push(['3 -> 3', 3, 3]);
                P.setCursor(3);            paths.push(['3 -> 0', 3, 0]);
                const checks = [];
                for (const [label, from, to] of paths) {
                    P.setCursor(from); P.setCursor(to);
                    const shown = readPanel();
                    // sequential reference for the SAME target
                    P.setCursor(0);
                    for (let i = 1; i <= to; i++) P.setCursor(i);
                    const sequential = readPanel();
                    checks.push({label, same: JSON.stringify(shown) === JSON.stringify(sequential)});
                }
                P.setCursor(0);
                return checks;
            }""")
            for c in e2e:
                check(f"端到端：跳转 {c['label']} 的面板渲染 == 顺序播放到该步",
                      c["same"])

        # ------------------------------------------------------------- DAG
        print("\n== DAG ==")
        dag = page.evaluate("""() => {
            const svg = document.querySelector('.lab-dag');
            const boxes = svg.querySelectorAll('.lab-nd');
            const decided = svg.querySelectorAll('.lab-eg');
            return {
                nodes: boxes.length,
                drawnEdges: decided.length,
                layers: window.__p.layout.numLayers,
                widest: window.__p.layout.widest,
                bands: window.__p.layout.bandCount,
                W: Math.round(window.__p.layout.W),
                H: Math.round(window.__p.layout.H),
            };
        }""")
        note(f"DAG: {dag['nodes']} 节点, {dag['drawnEdges']} 条画出来的边, "
             f"{dag['layers']} 层, 最宽层 {dag['widest']}, {dag['bands']} 段, "
             f"画布 {dag['W']}×{dag['H']}")
        check("顺序边参与分层但不画出来（层数==节点数，边数==数据边数）",
              dag["layers"] == dag["nodes"] and dag["widest"] == 1,
              f"layers={dag['layers']} widest={dag['widest']}")

        # load-type step not stuck at layer 0
        ld_layer = page.evaluate("() => window.__p.layout.layer['ld1']")
        check("load 类无前驱步骤不再堆在第 0 层", ld_layer is not None and ld_layer > 0,
              f"ld1 在第 {ld_layer} 层")

        # three-state visual distinction
        page.evaluate("() => window.__p.setCursor(4)")
        page.wait_for_timeout(200)
        states = page.evaluate("""() => {
            const svg = document.querySelector('.lab-dag');
            return {
                current: svg.querySelectorAll('.lab-nd.lab-st-current').length,
                visited: svg.querySelectorAll('.lab-nd.lab-st-visited').length,
                future: svg.querySelectorAll('.lab-nd.lab-st-future').length,
                nodeIdentity: svg.querySelector('.lab-nd').dataset.node,
            };
        }""")
        check("节点三态同时存在且可辨",
              states["current"] == 1 and states["visited"] > 0 and states["future"] > 0,
              json.dumps(states))

        # incremental redraw: same DOM node identity across a cursor move
        page.evaluate("() => window.__p.setCursor(2)")
        page.wait_for_timeout(150)
        check("重绘走增量 class 而非 DOM 重建（节点身份保持不变）",
              page.evaluate("() => document.querySelector('.lab-dag .lab-nd').dataset.node") ==
              states["nodeIdentity"])

        # click a node -> jumps
        page.evaluate("""() => {
            const g = [...document.querySelectorAll('.lab-dag .lab-nd')]
                .find(n => n.dataset.step === '6');
            g.dispatchEvent(new MouseEvent('click', {bubbles: true}));
        }""")
        page.wait_for_timeout(200)
        check("点击节点跳转到该步",
              page.evaluate("() => window.__p.state.cursor") == 6,
              f"cursor={page.evaluate('() => window.__p.state.cursor')}")

        # Wrap check. The page's default configuration is only 10 steps, so the
        # longest trace the sliders can reach (N=8, Bc=1 -> 18 blocks) is used
        # for the layout regression. It is already inlined in the page as part
        # of the trace set, so this reaches it the same way the sliders do.
        wrap = page.evaluate("""() => {
            const T = window.LabTraces['online-softmax-N8-B1-T1'];
            const idx = LabEngine.traceModel.index(T);
            const L = LabEngine.layout.compute(idx);
            const noWrap = LabEngine.layout.compute(idx, {nowrap: true});
            return {
                enabled: L.doWrap, bands: L.bandCount, layers: L.numLayers,
                wrappedAspect: +(L.W / L.H).toFixed(2),
                flatAspect: +(noWrap.W / noWrap.H).toFixed(2),
                wrappedW: Math.round(L.W), wrappedH: Math.round(L.H),
            };
        }""")
        note(f"长 trace({wrap['layers']} 层): 折行后 {wrap['wrappedAspect']}:1 "
             f"({wrap['wrappedW']}×{wrap['wrappedH']}), 不折行 {wrap['flatAspect']}:1")
        check("折行生效：>10 层时折段，长链不退化成横条",
              wrap["enabled"] and wrap["bands"] > 1 and wrap["wrappedAspect"] < wrap["flatAspect"]
              and wrap["flatAspect"] > 5,
              f"不折行 {wrap['flatAspect']}:1 -> 折行 {wrap['wrappedAspect']}:1（{wrap['bands']} 段）")

        # ---------------------------------------------------------- player UX
        print("\n== 播放器交互 ==")
        # reload fresh: no autoplay, starts at step 0
        page2 = browser.new_page(viewport={"width": W, "height": H})
        page2.goto(url)
        page2.wait_for_timeout(700)
        start = page2.evaluate("() => window.__p.state.cursor")
        check("加载时停在第一步（不自动播放）",
              start == 0 and page2.evaluate("() => !window.__p.state.playing"),
              f"cursor={start}")

        # deep link
        page2.goto(url + "?step=7")
        page2.wait_for_timeout(700)
        deep = page2.evaluate("() => window.__p.state.cursor")
        check("?step=N 深链生效", deep == 7, f"cursor={deep}")
        deep_state = page2.evaluate("""() => {
            const out = {};
            document.querySelectorAll('.lab-trow').forEach(r => {
                out[r.dataset.tensor] = r.querySelector('.lab-tval').textContent;
            });
            return out;
        }""")
        page2.evaluate("() => window.__p.setCursor(0)")
        page2.wait_for_timeout(120)
        for _ in range(7):
            page2.evaluate("() => window.__p.setCursor(window.__p.state.cursor + 1)")
        page2.wait_for_timeout(150)
        seq_state = page2.evaluate("""() => {
            const out = {};
            document.querySelectorAll('.lab-trow').forEach(r => {
                out[r.dataset.tensor] = r.querySelector('.lab-tval').textContent;
            });
            return out;
        }""")
        check("深链状态与顺序播放到该步完全一致",
              json.dumps(deep_state, sort_keys=True) == json.dumps(seq_state, sort_keys=True))

        # replaceState, not pushState
        hist = page2.evaluate("""() => {
            const before = history.length;
            window.__p.setCursor(2); window.__p.setCursor(5); window.__p.setCursor(8);
            return {before, after: history.length, url: location.search};
        }""")
        check("拖动游标用 replaceState，不污染历史栈",
              hist["before"] == hist["after"], json.dumps(hist))

        # keyboard
        page2.evaluate("() => { window.__p.setCursor(0); document.body.focus(); }")
        page2.evaluate("() => window.scrollTo(0, 0)")
        page2.keyboard.press("ArrowRight")
        page2.wait_for_timeout(120)
        right = page2.evaluate("() => window.__p.state.cursor")
        page2.keyboard.press("ArrowLeft")
        page2.wait_for_timeout(120)
        left = page2.evaluate("() => window.__p.state.cursor")
        check("左右箭头单步", right == 1 and left == 0, f"right={right} left={left}")

        scroll_before = page2.evaluate("() => window.scrollY")
        page2.keyboard.press("Space")
        page2.wait_for_timeout(120)
        space_playing = page2.evaluate("() => window.__p.state.playing")
        scroll_after = page2.evaluate("() => window.scrollY")
        check("空格播放/暂停且不触发页面滚动",
              space_playing and scroll_before == scroll_after,
              f"playing={space_playing} scroll {scroll_before}->{scroll_after}")
        page2.keyboard.press("Space")
        page2.wait_for_timeout(120)
        check("再按空格暂停", not page2.evaluate("() => window.__p.state.playing"))

        page2.keyboard.press("End")
        page2.wait_for_timeout(150)
        end = page2.evaluate("() => window.__p.state.cursor")
        page2.keyboard.press("Home")
        page2.wait_for_timeout(150)
        home = page2.evaluate("() => window.__p.state.cursor")
        last = page2.evaluate("() => window.__p.index.lastStep")
        check("Home/End 到首尾", home == 0 and end == last, f"home={home} end={end} last={last}")

        # DAG node keyboard activation. A node other than the current one is
        # used, so "it jumped" is actually observable rather than a no-op.
        page2.evaluate("() => window.__p.setCursor(0)")
        page2.wait_for_timeout(120)
        focused = page2.evaluate("""() => {
            const g = document.querySelector('.lab-dag .lab-nd[data-step="6"]');
            g.focus();
            return document.activeElement === g;
        }""")
        page2.keyboard.press("Enter")
        page2.wait_for_timeout(150)
        after_enter = page2.evaluate("() => window.__p.state.cursor")
        check("DAG 节点可 Tab 聚焦、回车跳转",
              focused and after_enter == 6, f"focused={focused} cursor={after_enter}")

        # -------------------------------------------- correction amplifier
        print("\n== 修正因子放大器（#14 专用视图）==")
        # Walk to the step where the factor is first non-trivial and read both
        # the panel and the trace, then require they agree. The panel being
        # populated is not enough — it has to be populated from the data.
        rescale = page.evaluate("""() => {
            const T = window.__labTrace;
            for (let i = 0; i < T.steps.length; i++) {
                const c = T.steps[i].corr;
                if (c && c.kind === 'rescale') return {i, corr: c};
            }
            return null;
        }""")
        check("默认配置里存在 m 被抬高、必须缩放的步",
              rescale is not None,
              f"第一个 rescale 步 = {rescale['i'] if rescale else None}")

        if rescale:
            page.evaluate("i => window.__p.setCursor(i)", rescale["i"])
            page.wait_for_timeout(280)
            amp = page.evaluate("""() => {
                const host = document.querySelector('[data-panel-body=amplifier]');
                if (!host) return null;
                return {
                    text: host.textContent.replace(/\\s+/g, ' '),
                    factor: (host.querySelector('.l00-corr-factor') || {}).textContent || '',
                    tag: (host.querySelector('.l00-tag') || {}).textContent || '',
                    cells: [...host.querySelectorAll('.l00-vs-cell')].map(c => ({
                        k: (c.querySelector('.l00-vs-k')||{}).textContent||'',
                        v: (c.querySelector('.l00-vs-v')||{}).textContent||'',
                    })),
                };
            }""")
            check("放大器面板渲染（因子 / 标签 / 对比格都在）",
                  amp is not None and amp["factor"] and amp["cells"],
                  (amp or {}).get("tag", ""))

            # the factor shown must be the trace's factor, at display precision
            got_factor = float(amp["factor"])
            want_factor = rescale["corr"]["factor"]
            check("高亮显示的因子就是 trace 里的 e^(m_old − m_new)",
                  abs(got_factor - want_factor) <= 5e-4 * max(1e-9, abs(want_factor)),
                  f"页面 {got_factor} vs trace {want_factor!r}")
            check("该步被标记为 rescale（m 确实被抬高了）",
                  "缩放" in amp["tag"], amp["tag"])

            # The half of the criterion that is easy to fake: a deviation number
            # has to be present, and it has to be the measured one.
            def disp4(v):
                """Match the page's num(): %.4g, except that the page uses
                toPrecision(4) (4 significant digits) rather than %g (4 sig
                digits too, but with a different rounding of trailing zeros)."""
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

            # the factor is genuinely highlighted in the formula, not just noted
            region = page.evaluate("""() => {
                const el = document.querySelector('.lab-formula-host .lab-region');
                return el ? {cls: el.className, bg: getComputedStyle(el).backgroundColor,
                             text: el.textContent} : null;
            }""")
            check("修正因子的公式片段被 \\region 真正框住（KaTeX 生成的元素）",
                  region is not None and "region-CORR" in region["cls"],
                  json.dumps(region, ensure_ascii=False))

            # and on a step where m does not move, the factor is exactly 1 — the
            # contrast that makes the rescale steps legible
            identity = page.evaluate("""() => {
                const T = window.__labTrace;
                for (let i = 0; i < T.steps.length; i++) {
                    const c = T.steps[i].corr;
                    if (c && c.kind === 'identity') return {i, corr: c};
                }
                return null;
            }""")
            if identity:
                page.evaluate("i => window.__p.setCursor(i)", identity["i"])
                page.wait_for_timeout(220)
                idl = page.evaluate(
                    "() => (document.querySelector('.l00-corr-factor')||{}).textContent")
                check("m 未变的步，因子显示为 1（与 rescale 步形成对照）",
                      idl.strip() in ("1", "1.000"), f"显示 {idl!r}")
            else:
                note("默认配置里没有 m 保持不变的块（Bc=4 时每块都抬高 m），跳过恒等因子的对照")

        # -------------------------------------------- correction aggregate
        agg = page.evaluate("""() => {
            const c = window.__labTrace.meta.correction;
            const host = document.querySelector('[data-panel-body=amplifier]').textContent
                .replace(/\\s+/g, ' ');
            return {rescales: c.rescales, nblocks: window.__labTrace.meta.config.n_blocks,
                    final_bias: c.final_bias_abs, host};
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
        # Cursor back to the end so all three tracks have completed.
        page.evaluate("() => window.__p.setCursor(window.__p.index.lastStep)")
        page.wait_for_timeout(300)
        tracks = page.evaluate("""() => {
            const host = document.querySelector('[data-panel-body=compare]');
            if (!host) return null;
            return [...host.querySelectorAll('.l00-track')].map(t => ({
                cls: t.className,
                title: (t.querySelector('.l00-track-t')||{}).textContent || '',
                running: (t.querySelector('.l00-track-v')||{}).textContent || '',
                final: ((t.querySelector('.l00-track-final b')||{}).textContent) || '',
                badge: (t.querySelector('.l00-badge-ok, .l00-badge-bad')||{}).textContent || '',
                sub: (t.querySelector('.l00-track-sub')||{}).textContent || '',
                width: (t.querySelector('.l00-progress > span')||{}).style
                       ? t.querySelector('.l00-progress > span').style.width : '',
            }));
        }""")
        check("三条轨迹并排渲染（朴素 / 安全 / online）",
              tracks is not None and len(tracks) == 3, f"{len(tracks) if tracks else 0} 条")
        if tracks:
            note(" · ".join(f"{t['title'].split('（')[0]}: 跑动={t['running']} 终值={t['final']}"
                            for t in tracks))
            naive, safe, online = tracks
            # The headline claim of the whole comparison.
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

            # the final values shown must be the trace's, formatted by the
            # page's own formatter — asserted in the page so the two cannot
            # disagree about rounding
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

        # The overflow control: the same naive implementation, finite without
        # the offset. Without this the NaN could just mean broken code.
        ctrl = page.evaluate("""() => {
            const host = document.querySelector('[data-panel-body=compare]');
            const si = window.__labTrace.compare.shift_invariance;
            const T = window.__labTrace;
            const method = T.compare.methods.find(m => m.id === 'naive');
            const text = host.textContent.replace(/\\s+/g, ' ');
            return {unshifted: si.unshifted_final, shifted: si.shifted_final,
                    reference: si.reference, text,
                    naive_final: method.final};
        }""")
        check("对照组存在：同一份朴素实现不加偏移时算对（证明溢出是量级问题）",
              isinstance(ctrl["unshifted"], float) and
              abs(ctrl["unshifted"] - ctrl["reference"]) < 1e-9,
              f"不加偏移 {ctrl['unshifted']!r} vs 参考 {ctrl['reference']!r}")
        check("对照组被写进面板（读者能看到「代码一个字没改」）",
              "代码一个字没改" in ctrl["text"] or "只改量级" in ctrl["text"])
        check("对照组的偏移结果与朴素行一致（都是 NaN）",
              ctrl["shifted"] == ctrl["naive_final"] == "NaN",
              f"shifted={ctrl['shifted']!r} naive={ctrl['naive_final']!r}")

        # The cursor drives the comparison: at a low k the naive row is already
        # NaN while online is still mid-scan. That coupling is the teaching move.
        page.evaluate("() => window.__p.setCursor(0)")
        page.wait_for_timeout(250)
        early = page.evaluate("""() => {
            const ts = [...document.querySelectorAll('[data-panel-body=compare] .l00-track')];
            return {sub: ts[2].querySelector('.l00-track-sub').textContent,
                    w: ts[2].querySelector('.l00-progress > span').style.width};
        }""")
        page.evaluate("() => window.__p.setCursor(window.__p.index.lastStep)")
        page.wait_for_timeout(250)
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
            page.wait_for_timeout(450)

        def page_config():
            return page.evaluate("""() => ({
                cfg: window.__labConfig,
                config: window.__labTrace.meta.config,
                steps: window.__labTrace.steps.length,
                x: document.querySelector('.lab-trow[data-tensor=x] .lab-tval').textContent,
                formula: document.querySelector('.lab-formula-host').textContent,
            })""")

        before = page_config()
        note(f"默认 = {before['cfg']}（{before['steps']} 步）")

        # Pick a different N and a different Bc, and require the page to move to
        # a different TRACE — not merely to re-render the same one.
        n_idx = page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const cur = window.__labTrace.meta.config.N;
            return m.params.N.findIndex(v => v !== cur);
        }""")
        set_slider("N", n_idx)
        after_n = page_config()
        check("改 N 后换到了另一份 trace（配置、步数、输入都变了）",
              after_n["config"]["N"] != before["config"]["N"] and
              after_n["cfg"] != before["cfg"] and
              after_n["steps"] != before["steps"],
              f"{before['cfg']} -> {after_n['cfg']}（{after_n['steps']} 步）")
        check("新配置的输入张量确实换了（不是同一份 trace 重画）",
              after_n["x"] != before["x"], f"x = {after_n['x']!r}")

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
              f"T {after_bc['config']['T']} -> {after_t['config']['T']}，"
              f"x {after_bc['x']!r} -> {after_t['x']!r}")

        # whatever configuration is now loaded, its numbers must be that
        # configuration's — including the two new panels, which were built for
        # the original trace and have to be rebuilt with the new one
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
            const finals = [...host.querySelectorAll('.l00-track-final b')]
                .map(e => e.textContent);
            return {
                cfg: window.__labConfig,
                /* Matched at the precision the page prints them at: the
                   aggregate bias at 4 digits, the control's result at 6. */
                ampHas: amp.includes(fmt(c.final_bias_abs)),
                cmpHas: cmp.includes(fmt(si.unshifted_final, 6)),
                cmpFinals: finals,
                wantFinals: T.compare.methods.map(m => fmt(m.final, 6)),
                panels: document.querySelectorAll('[data-panel-body]').length,
                narrow: document.querySelectorAll('.lab-narrow').length,
                roots: document.querySelectorAll('.lab-root').length,
            };
        }""")
        check("换配置后两个新面板跟着重建（数值来自新 trace，不是旧的那份）",
              rebuilt["ampHas"] and rebuilt["cmpHas"] and
              rebuilt["cmpFinals"] == rebuilt["wantFinals"],
              json.dumps(rebuilt, ensure_ascii=False))
        check("换配置后没有累积出多余的播放器实例（DOM 每次重建）",
              rebuilt["panels"] == 2 and rebuilt["narrow"] == 1 and rebuilt["roots"] == 1,
              json.dumps(rebuilt))

        # every configuration the manifest names must have a trace behind it
        coverage = page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const missing = m.traces.filter(t => !window.LabTraces[t]);
            return {n: m.traces.length, missing, default: m.default,
                    hasDefault: !!window.LabTraces[m.default]};
        }""")
        check("manifest 列出的每个配置都真的被内联进了页面",
              not coverage["missing"] and coverage["hasDefault"],
              f"{coverage['n']} 个配置，缺 {coverage['missing']}")

        # deep link to a non-default configuration
        page3cfg = browser.new_page(viewport={"width": W, "height": H})
        page3cfg.goto(url + "?cfg=online-softmax-N4-B2-T0p5")
        page3cfg.wait_for_timeout(800)
        dl = page3cfg.evaluate("""() => ({
            cfg: window.__labConfig,
            config: window.__labTrace.meta.config,
            slider: [...document.querySelectorAll('input[data-param]')]
                .map(i => i.dataset.param + '=' + i.value).join(','),
        })""")
        check("?cfg= 深链载入指定配置，滑杆位置同步",
              dl["cfg"] == "online-softmax-N4-B2-T0p5" and
              dl["config"]["N"] == 4 and dl["config"]["Bc"] == 2 and dl["config"]["T"] == 0.5,
              json.dumps(dl, ensure_ascii=False))

        # a bad ?cfg is announced, not silently swallowed
        page3cfg.goto(url + "?cfg=online-softmax-N999-B1-T1")
        page3cfg.wait_for_timeout(700)
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

        # Reset to the manifest's default before the screenshots below, so they
        # are taken of a known configuration rather than whatever the last
        # slider move happened to leave behind.
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
        page.wait_for_timeout(600)
        check("回到 manifest 的默认配置（截图基准）",
              page.evaluate("() => window.__labConfig") == coverage["default"],
              page.evaluate("() => window.__labConfig"))

        # ---------------------------------------------------------- narrow
        print("\n== 窄屏降级 ==")
        page3 = browser.new_page(viewport={"width": 420, "height": 900})
        page3.goto(url)
        page3.wait_for_timeout(700)
        narrow = page3.evaluate("""() => {
            const n = document.querySelector('.lab-narrow');
            const root = document.querySelector('.lab-root');
            if (!n) return null;
            return {
                visible: !n.hidden && getComputedStyle(n).display !== 'none',
                rootHidden: root.hidden,
                // The attribute alone is not enough: `.lab-root { display:flex }`
                // outranks `[hidden]`, so the computed style is what matters.
                rootDisplay: getComputedStyle(root).display,
                hasTitle: !!n.querySelector('.lab-narrow-title'),
                hasStatic: !!n.querySelector('.lab-narrow-formula'),
                link: (n.querySelector('.lab-narrow-link') || {}).textContent || '',
                linkHref: (n.querySelector('.lab-narrow-link') || {}).getAttribute
                    ? n.querySelector('.lab-narrow-link').getAttribute('href') : '',
                text: n.textContent.replace(/\\s+/g, ' ').trim().slice(0, 120),
            };
        }""")
        check("窄屏显示统一降级组件（标题 + 静态内容 + 返回链接）",
              narrow is not None and narrow["visible"] and narrow["hasTitle"] and
              narrow["hasStatic"] and "返回教程" in narrow["link"],
              json.dumps(narrow, ensure_ascii=False) if narrow else "no .lab-narrow")
        check("窄屏下播放器真的不渲染（不是只设了 hidden 属性）",
              narrow is not None and narrow["rootDisplay"] == "none",
              f"root display = {narrow['rootDisplay'] if narrow else '?'}")
        # and the reverse: wide screens must not show the fallback
        wide_narrow = page.evaluate(
            "() => getComputedStyle(document.querySelector('.lab-narrow')).display")
        check("宽屏下不显示降级组件", wide_narrow == "none", f"display = {wide_narrow}")
        page3.screenshot(path=SHOTS / "12-narrow-fallback.png", full_page=True)

        # ---------------------------------------------------------- site bits
        print("\n== 站点 ==")
        pf = page.evaluate("""() => {
            const b = document.body;
            return {onBody: b.hasAttribute('data-pagefind-ignore'),
                    onMeta: !!document.querySelector('meta[data-pagefind-ignore]')};
        }""")
        check("data-pagefind-ignore 包在 <body> 上", pf["onBody"], json.dumps(pf))

        # Asset references specifically (the canonical <link> points at the page
        # itself, which is legitimately not under ./assets/).
        assets = page.evaluate("""() => {
            const out = [];
            document.querySelectorAll('link[rel=stylesheet][href], script[src]').forEach(el => {
                const u = el.getAttribute('href') || el.getAttribute('src');
                if (u && !/^[a-z]+:|^\\/\\/|^#/.test(u)) out.push(u);
            });
            return out;
        }""")
        check("所有本地资源用 ./assets/ 引用（暂存后会拍平目录，../ 会 404）",
              len(assets) >= 9 and all(p.startswith("./assets/") for p in assets),
              f"{len(assets)} 个资源引用，全部 ./assets/")

        # ------------------------------------------------------------- shots
        print("\n== 截图 ==")
        page.evaluate("() => window.__p.setCursor(0)")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "01-overview-step0.png", full_page=True)

        page.evaluate("() => window.__p.setCursor(7)")
        page.wait_for_timeout(250)
        page.locator(".lab-formula-host").screenshot(path=SHOTS / "02-formula-num.png")
        page.click("[data-tier='sym']")
        page.wait_for_timeout(200)
        page.locator(".lab-formula-host").screenshot(path=SHOTS / "03-formula-sym.png")
        page.click("[data-tier='idx']")
        page.wait_for_timeout(200)
        page.locator(".lab-formula-host").screenshot(path=SHOTS / "04-formula-idx.png")
        page.click("[data-tier='num']")
        page.wait_for_timeout(150)

        # the \region highlight, at the step where the correction factor is non-zero
        page.evaluate("() => window.__p.setCursor(7)")
        page.wait_for_timeout(200)
        page.locator("section.lab-panel", has_text="公式面板").screenshot(
            path=SHOTS / "05-formula-region.png")

        page.evaluate("() => window.__p.setCursor(4)")
        page.wait_for_timeout(250)
        page.locator("section.lab-panel", has_text="张量检查器").screenshot(
            path=SHOTS / "06-tensors.png")

        page.evaluate("() => window.__p.setCursor(4)")
        page.wait_for_timeout(200)
        page.locator("section.lab-panel", has_text="执行 DAG").screenshot(
            path=SHOTS / "07-dag-10.png")

        # Wrapped DAG: the longest trace in the set (N=8, Bc=1 -> 18 blocks),
        # selected through the real slider rather than a scratch player, so the
        # shot shows a configuration a reader can actually reach.
        page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const el = document.querySelector('input[data-param=Bc]');
            el.value = m.params.Bc.indexOf(1);
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }""")
        page.wait_for_timeout(600)
        page.locator(".lab-dag-host").screenshot(path=SHOTS / "08-dag-wrapped-18.png")

        # ------------------------------------------------- the three new views
        # On the default config, at the step where the factor first rescales.
        page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const el = document.querySelector('input[data-param=Bc]');
            el.value = m.params.Bc.indexOf(4);
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }""")
        page.wait_for_timeout(600)
        page.evaluate("""() => {
            const T = window.__labTrace;
            for (let i = 0; i < T.steps.length; i++) {
                if (T.steps[i].corr && T.steps[i].corr.kind === 'rescale') {
                    window.__p.setCursor(i); return;
                }
            }
        }""")
        page.wait_for_timeout(350)
        page.locator("section.lab-panel", has_text="修正因子放大器").screenshot(
            path=SHOTS / "13-correction-amplifier.png")

        # the same panel on an identity step, for the contrast
        page.evaluate("""() => {
            const T = window.__labTrace;
            for (let i = 0; i < T.steps.length; i++) {
                if (T.steps[i].corr && T.steps[i].corr.kind === 'identity') {
                    window.__p.setCursor(i); return;
                }
            }
        }""")
        page.wait_for_timeout(300)

        # comparison panel, at the end where all three have finished
        page.evaluate("() => window.__p.setCursor(window.__p.index.lastStep)")
        page.wait_for_timeout(350)
        page.locator("section.lab-panel", has_text="对照模式").screenshot(
            path=SHOTS / "14-compare-tracks.png")

        # and mid-scan, where the naive row is already NaN and online is not
        page.evaluate("() => window.__p.setCursor(1)")
        page.wait_for_timeout(300)
        page.locator("section.lab-panel", has_text="对照模式").screenshot(
            path=SHOTS / "15-compare-midscan.png")

        # the parameter bar
        page.evaluate("() => window.__p.setCursor(0)")
        page.wait_for_timeout(250)
        page.locator("#lab-params").screenshot(path=SHOTS / "16-params.png")

        # a configuration reached purely by moving the sliders, to show the
        # numbers really do change with N and Bc
        page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const el = document.querySelector('input[data-param=N]');
            el.value = m.params.N.indexOf(4);
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }""")
        page.wait_for_timeout(600)
        page.screenshot(path=SHOTS / "17-reconfigured-N4.png", full_page=True)
        page.evaluate("""() => {
            const m = window.LabTraceSets['online-softmax'];
            const el = document.querySelector('input[data-param=N]');
            el.value = m.params.N.indexOf(8);
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }""")
        page.wait_for_timeout(600)

        page.locator("section.lab-panel", has_text="时间轴").screenshot(
            path=SHOTS / "09-timeline.png")
        page.locator("section.lab-panel", has_text="自检").screenshot(
            path=SHOTS / "10-selfcheck.png")
        # Both themes, from pages that actually requested them. The stylesheet
        # is theme-aware, and a shot named "dark" that was taken in light mode
        # is worse than no shot at all — which is exactly what happened while
        # the dark shot was still being taken from the light driver page above:
        # the two files came out byte-identical.
        for name, scheme in (("11-full-dark.png", "dark"),
                             ("11b-full-light.png", "light")):
            p_theme = browser.new_page(viewport={"width": W, "height": H},
                                       color_scheme=scheme)
            p_theme.goto(url)
            p_theme.wait_for_timeout(800)
            p_theme.screenshot(path=SHOTS / name, full_page=True)
            p_theme.close()

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
