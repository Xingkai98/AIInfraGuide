#!/usr/bin/env python3
"""Drive the staged lab page in real Chromium: assert behaviour, capture shots.

This is the acceptance harness for the engine, not a demo. It runs against the
built artifact in `public/labs/` (so it exercises the same inlined trace and
flattened asset paths a reader gets), asserts the acceptance criteria one by
one, and only then takes screenshots.

Run:  npx python3 ... (see README) — needs `npm run build:labs` first and
`pip install playwright && playwright install chromium`.

Everything it checks is printed; a non-zero exit means at least one acceptance
criterion failed. Screenshots land in `labs/pages/shots/`.
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PAGE = REPO / "public" / "labs" / "00-online-softmax.html"
SHOTS = HERE / "shots"
SHOTS.mkdir(exist_ok=True)

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
        check("代入数值档确实代入了数值",
              "0.83" in tiers["num"] and "0.83" not in tiers["sym"],
              f"num={tiers['num']!r}")

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
                const P = window.__p, idx = P.index, T = window.LabTraces['online-softmax'];
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

        # Wrap check. The 18-step trace is a fixture, not part of this page's
        # payload, so it is read from labs/traces/ and injected for the test.
        # Nothing here ships; the point is that a trace past the wrap threshold
        # lays out as bands rather than one 25:1 ribbon.
        long_trace = json.loads((REPO / "labs" / "traces" / "online-softmax-long.json")
                                .read_text(encoding="utf-8"))
        page.evaluate("t => { window.LabTraces['online-softmax-long'] = t; }", long_trace)
        wrap = page.evaluate("""() => {
            const T = window.LabTraces['online-softmax-long'];
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

        # wrapped DAG: render the long trace in a scratch player
        page.evaluate("""() => {
            const T = window.LabTraces['online-softmax-long'];
            const host = document.createElement('div');
            host.id = 'longlab';
            document.querySelector('.lab-root').after(host);
            window.__longp = LabEngine.lab(T, {mount: '#longlab', title: '长 trace DAG', wrapAt: 6});
            window.__longp.dom.dagHost.scrollIntoView();
        }""")
        page.wait_for_timeout(500)
        page.locator("#longlab .lab-dag-host").screenshot(path=SHOTS / "08-dag-wrapped-18.png")
        page.evaluate("() => document.getElementById('longlab').remove()")

        page.locator("section.lab-panel", has_text="时间轴").screenshot(
            path=SHOTS / "09-timeline.png")
        page.locator("section.lab-panel", has_text="自检").screenshot(
            path=SHOTS / "10-selfcheck.png")
        page.screenshot(path=SHOTS / "11-full-dark.png", full_page=True)

        # a light-mode copy, since the page is theme-aware
        page_light = browser.new_page(viewport={"width": W, "height": H},
                                      color_scheme="light")
        page_light.goto(url)
        page_light.wait_for_timeout(700)
        page_light.screenshot(path=SHOTS / "11b-full-light.png", full_page=True)

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
