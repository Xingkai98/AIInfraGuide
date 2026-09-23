#!/usr/bin/env python3
"""Acceptance harness for the memory-ledger view component (ticket #45).

Runs against the built artifact in `public/labs/view-ledger.html` — i.e. the
same inlined trace and flattened asset paths a reader gets — asserts the ticket's
acceptance criteria one by one, and only then takes screenshots.

It is a companion to `scripts/verify-labs.py`, not a replacement: that one owns
the *engine's* contract (arbitrary jump, lint, player UX, narrow-screen
fallback). This one owns the ledger's. Both are run by hand; neither is wired
into CI, which only runs the Python trace generators.

Run:  npm run build:labs && python3 scripts/verify-ledger.py
Needs `pip install playwright && playwright install chromium`.

Screenshots land in `labs/pages/shots/` (committed, so a reviewer can see the
rendered result without running this).
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "public" / "labs" / "view-ledger.html"
SHOTS = REPO / "labs" / "pages" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1100

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
    trace = json.loads((REPO / "labs" / "traces" / "kv-cache.json").read_text(encoding="utf-8"))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        page.goto(url)
        page.wait_for_timeout(900)

        # ------------------------------------------------------- AC1: mounted
        print("\n== 组件挂载与接口 ==")
        mounted = page.evaluate("""() => {
            const host = document.querySelector('[data-ledger-mount]');
            const r = host && host.querySelector('.lab-ledger:not(.lab-ledger-mx-host)');
            if (!r) return null;
            return {
                bars: r.querySelectorAll('.lab-ledger-bar > span').length,
                rows: r.querySelectorAll('.lab-ledger-table tbody tr').length,
                formulas: r.querySelectorAll('.lab-ledger-fml').length,
                kaTeX: r.querySelectorAll('.lab-ledger-fml .katex').length,
                hasTicks: !!r.querySelector('.lab-ledger-tick'),
                stepId: r.getAttribute('data-step-id'),
            };
        }""")
        check("账本渲染进 lab 页（堆叠条 + 分项表 + 溯源公式都在）",
              mounted is not None and mounted["bars"] > 0 and mounted["rows"] == 4,
              json.dumps(mounted, ensure_ascii=False))
        check("每个分项的显存公式经 KaTeX 渲染（不是字面 LaTeX 源码）",
              mounted is not None and mounted["kaTeX"] == 3 and mounted["formulas"] == 3,
              f'{mounted["kaTeX"] if mounted else "?"} 条公式')
        check("堆叠条带字节刻度轴", mounted is not None and mounted["hasTicks"])

        segs = page.evaluate("""() => {
            const t = window.__labTrace;
            return t.ledger.segments.map(s => ({id: s.id, label: s.label, color: s.color}));
        }""")
        note("分项声明：" + " / ".join(f'{s["label"]}({s["id"]}, {s["color"]})' for s in segs))
        check("分项可从 trace 配置（名称 / 颜色 / 公式），不是写死在组件里",
              len(segs) == 3 and all(s["color"].startswith("#") for s in segs),
              json.dumps(segs, ensure_ascii=False))

        # ------------------------------- AC2: replay drives it from the trace
        print("\n== 回放：账本随上下文增长 ==")
        growth = page.evaluate("""() => {
            const P = window.__p, T = window.__labTrace, out = [];
            for (let i = 0; i < T.steps.length; i++) {
                P.setCursor(i);
                const host = document.querySelector('[data-ledger-mount] .lab-ledger');
                // `.lab-ledger-total` is the derived 合计 row, not a segment;
                // the segment rows are the ones carrying a .lab-ledger-seg cell.
                const cells = [...host.querySelectorAll('.lab-ledger-table tbody tr')]
                    .filter(tr => tr.querySelector('.lab-ledger-seg'))
                    .map(tr => tr.querySelector('.lab-ledger-num').textContent.trim());
                const total = host.querySelector('.lab-ledger-total .lab-ledger-num');
                out.push({i, id: T.steps[i].id, phase: T.steps[i].ledger.config.phase,
                          seq: T.steps[i].ledger.config.seq_len,
                          cells, total: total ? total.textContent.trim() : null,
                          barWidths: [...host.querySelectorAll('.lab-ledger-bar > span')]
                              .map(s => s.style.flexGrow)});
            }
            return out;
        }""")
        note("前 5 步的账本（分项文本）：")
        for row in growth[:5]:
            note(f'  步 {row["i"]:2d} {row["id"]:14s} seq={row["seq"]:2d} → {row["cells"]}')

        # The rendered numbers must equal the trace's published bytes. The cell
        # reads "210,176 B · 205.3 KiB" — the exact count first, then a human
        # scale — so the comparison takes the leading integer and ignores the
        # suffix, which is a display convenience rather than a second fact.
        def leading_int(s):
            return int(s.split()[0].replace(",", ""))

        ok_nums = True
        bad = []
        for row in growth:
            s = trace["steps"][row["i"]]
            want = [s["ledger"]["bytes"][seg["id"]] for seg in trace["ledger"]["segments"]]
            got = [leading_int(c) for c in row["cells"]]
            if got != want:
                ok_nums = False
                bad.append(f'{row["id"]}: 渲染={got} trace={want}')
        check("每一步渲染出的分项字节数 == trace 里发布的字节数", ok_nums,
              "；".join(bad[:2]) if bad else f"{len(growth)} 步全部一致")

        # the 合计 row must be the sum of the rows above it, not an independent
        # number that could silently disagree
        ok_total = True
        for row in growth:
            s = trace["steps"][row["i"]]
            want = sum(s["ledger"]["bytes"][seg["id"]] for seg in trace["ledger"]["segments"])
            got = int(row["total"].split()[0].replace(",", ""))
            if got != want:
                ok_total = False
                bad = [f'{row["id"]}: 合计渲染={got} trace={want}']
                break
        check("合计行 == 各分项之和（推导出来的，不是另写一遍）", ok_total,
              bad[0] if not ok_total else f'{len(growth)} 步全部相等')

        # the KV segment must be strictly non-decreasing across the replay, and
        # strictly increase exactly when new rows are appended
        kv_series = [trace["steps"][r["i"]]["ledger"]["bytes"]["kv_cache"] for r in growth]
        param_series = [trace["steps"][r["i"]]["ledger"]["bytes"]["params"] for r in growth]
        check("KV Cache 分项在回放中单调不减", all(b >= a for a, b in zip(kv_series, kv_series[1:])),
              f"{kv_series[0]:,} → {kv_series[-1]:,} B")
        check("参数分项在回放中恒定（它不随上下文变）", len(set(param_series)) == 1,
              f'{param_series[0]:,} B')

        # widths grow with the bytes: the bar is data, not decoration
        widths = page.evaluate("""() => {
            const P = window.__p, T = window.__labTrace, out = [];
            [0, 4, T.steps.length - 1].forEach(i => {
                P.setCursor(i);
                const host = document.querySelector('[data-ledger-mount] .lab-ledger');
                out.push([...host.querySelectorAll('.lab-ledger-bar > span')]
                    .map(s => Number(s.style.flexGrow)));
            });
            return out;
        }""")
        check("堆叠条的宽度由字节数驱动（读的是真实 flex-grow，不是装饰）",
              widths[0][1] == trace["steps"][0]["ledger"]["bytes"]["kv_cache"] and
              widths[-1][1] > widths[0][1],
              f"首步 {widths[0]} → 末步 {widths[-1]}")

        # ------------------------------- AC3: 2-D matrix layout (for L14)
        print("\n== 二维矩阵布局（供 L14 复用）==")
        mx = page.evaluate("""() => {
            const t = document.querySelector('.lab-ledger-mx');
            if (!t) return null;
            const cols = [...t.querySelectorAll('thead th')].map(th => th.textContent.trim());
            const rows = [...t.querySelectorAll('tbody tr')].map(tr => ({
                label: tr.querySelector('th').textContent.trim(),
                cells: [...tr.querySelectorAll('td')].map(td => td.textContent.trim()),
            }));
            const totals = [...t.querySelectorAll('.lab-ledger-mx-total')].map(x => x.textContent.trim());
            return {cols, rows, totals, bars: t.querySelectorAll('.lab-ledger-mx-bar .lab-ledger-bar').length};
        }""")
        check("矩阵有策略列 × 分项行", mx is not None and len(mx["cols"]) == 5 and len(mx["rows"]) == 4,
              json.dumps(mx["cols"], ensure_ascii=False) if mx else "no matrix")
        check("每列都有一条堆叠条（省显存的过程要看得见）",
              mx is not None and mx["bars"] == 4, f'{mx["bars"] if mx else "?"} 条')
        for r in (mx or {"rows": []})["rows"]:
            note(f'  {r["label"]:8s} {r["cells"]}')
        note(f'  单卡合计 {mx["totals"]}')

        # the matrix's arithmetic: every ZeRO stage must shrink the sharded
        # segments by exactly 1/N, and leave activations alone
        cells = page.evaluate("() => window.__matrixCells")
        N = 4
        z_check = (
            cells["optimizer"]["z1"] * N == cells["optimizer"]["baseline"] and
            cells["grads"]["z2"] * N == cells["grads"]["baseline"] and
            cells["params"]["z3"] * N == cells["params"]["baseline"] and
            cells["params"]["z2"] == cells["params"]["baseline"] and
            cells["grads"]["z1"] == cells["grads"]["baseline"] and
            all(cells["activations"][c] == cells["activations"]["baseline"]
                for c in ("z1", "z2", "z3"))
        )
        check("矩阵数值符合 ZeRO 切分规则（Z1 切优化器状态 / Z2 加切梯度 / Z3 加切参数，激活不变）",
              z_check, json.dumps(cells, ensure_ascii=False))

        # column totals are derived, not supplied
        derived = page.evaluate("""() => {
            const c = window.__matrixCells, m = window.__ledgerMatrix;
            return ['baseline','z1','z2','z3'].map(k =>
                Object.keys(c).reduce((a, r) => a + c[r][k], 0) === m.colTotal(k));
        }""")
        check("矩阵的列合计是推导出来的（不接受外部传入的第二份真相）", all(derived))

        # ------------------------------- AC4: recompute on config change
        print("\n== 改配置 → 重算（纯函数） ==")
        slider = page.evaluate("""() => {
            const set = (key, v) => {
                const inputs = [...document.querySelectorAll('#ctl-row input')];
                // sliders are declared in this order in the page
                const order = ['seq_len','layers','kv_bytes','H'];
                const el = inputs[order.indexOf(key)];
                el.value = String(v);
                el.dispatchEvent(new Event('input', {bubbles: true}));
            };
            const read = () => {
                const host = document.querySelector('#ctl-ledger .lab-ledger');
                return [...host.querySelectorAll('.lab-ledger-table tbody tr')]
                    .map(tr => tr.querySelector('.lab-ledger-num').textContent.trim());
            };
            const out = {};
            set('seq_len', 64); set('layers', 2); set('kv_bytes', 2); set('H', 4);
            out.a = read();
            set('seq_len', 128); out.b = read();       // context doubled
            set('seq_len', 64); set('layers', 4); out.c = read();   // layers doubled
            set('layers', 2); set('kv_bytes', 1); out.d = read();   // fp16 -> fp8
            set('kv_bytes', 2); out.e = read();        // back to the start
            return out;
        }""")

        def num(s):
            return int(s.split()[0].replace(",", ""))

        a = [num(x) for x in slider["a"]]
        b = [num(x) for x in slider["b"]]
        c = [num(x) for x in slider["c"]]
        d = [num(x) for x in slider["d"]]
        e = [num(x) for x in slider["e"]]
        note(f"seq=64  → {slider['a']}")
        note(f"seq=128 → {slider['b']}")
        note(f"L=4    → {slider['c']}")
        note(f"fp8    → {slider['d']}")

        check("上下文长度翻倍 → KV Cache 精确翻倍（params 不动）",
              b[1] == 2 * a[1] and b[0] == a[0], f"{a[1]:,} → {b[1]:,} B")
        check("层数翻倍 → KV Cache 精确翻倍（params 按公式增加，不是翻倍）",
              c[1] == 2 * a[1] and c[0] > a[0], f"KV {a[1]:,} → {c[1]:,} B, 参数 {a[0]:,} → {c[0]:,} B")
        check("KV 精度 fp16 → fp8 → KV Cache 字节数精确减半",
              d[1] * 2 == a[1], f"{a[1]:,} → {d[1]:,} B")
        check("回到原配置 → 完全回到原来的数字（重算是纯函数，没有累积状态）",
              e == a, f"{e} == {a}")

        # the sliders are shown to be a *prediction* for a configuration the
        # trace never ran — which is the whole reason they go through the model
        note("滑杆配置 " + slider["b"][1] + " 是模型预测值；自检面板负责证明"
             "该模型与 trace 用的是同一套公式")

        # ------------------------------- the self-check + its control group
        print("\n== 自检（含对照组） ==")
        verdict = page.evaluate("() => window.__labLedger && window.__labLedger.L05")
        check("账本自检面板存在且跑过", verdict is not None)
        if verdict:
            note(json.dumps(verdict, ensure_ascii=False))
            check(f"逐步对拍：{verdict['steps']} 步 0 不一致",
                  verdict["stepFailures"] == 0, f'{verdict["stepFailures"]} 步不一致')
            check(f"缩放性质：{verdict['properties']} 条全部成立",
                  verdict["propertyFailures"] == 0, f'{verdict["propertyFailures"]} 条不成立')
            check("对照组：故意写错的模型全部被抓到（上面那个 0 才有意义）",
                  not verdict["missedControls"],
                  "漏掉 " + "、".join(verdict["missedControls"]) if verdict["missedControls"]
                  else f'{verdict["controls"]} 种全部抓到')
            check(f"账本契约 lint 干净且 lint 本身是活的"
                  f"（{verdict['lintSabotageTotal']} 种破坏全抓到）",
                  verdict["lintGaps"] == 0 and not verdict["lintSabotageMissed"],
                  f'gaps={verdict["lintGaps"]}, 漏掉={verdict["lintSabotageMissed"]}')

        # the sabotage list must genuinely exercise the ported rules — including
        # the ledger-specific ones this ticket added
        sab = page.evaluate("() => Object.keys(LabEngine.ledger.SABOTAGE_CASES)")
        note(f"JS 侧 lint 对照组共 {len(sab)} 条：{'、'.join(sab)}")
        check("JS 侧 lint 对照组覆盖了本票新增的账本规则（不是只有 L00 的老规则）",
              any("ledger" in s or "分项" in s for s in sab) and
              any("kv_rows_total" in s for s in sab) and len(sab) >= 20,
              f"{len(sab)} 条")

        # The engine's own contract lint — the base rules (reads/writes, graph
        # coverage, binding triples, sentinels) — must also be clean on this
        # trace. The ledger lint above only covers the ledger block; these are
        # two rule sets and both have to hold.
        base_lint = page.evaluate("""() => {
            const r = LabEngine.traceModel.lint(window.__labTrace);
            return {gaps: r.gaps.length, warns: r.warns.length,
                    what: r.gaps.concat(r.warns).map(x => x.what)};
        }""")
        check("引擎基础契约 lint（trace-model.js）在此 trace 上也是干净的",
              base_lint["gaps"] == 0 and base_lint["warns"] == 0,
              json.dumps(base_lint, ensure_ascii=False))

        # The engine's jump check must hold on this trace too: the ledger rides
        # on the player's single paint path, so if jumping were broken here the
        # ledger would be rendered against a state the trace cannot reproduce.
        #
        # (Note what is NOT asserted: `LabEngine.verify.panel`'s lint half. Its
        # control group is a fixed list of mutations written against L00's
        # trace, so two of its ten entries are no-ops on this one. The ledger
        # carries its own contract lint with its own 20-mutation control group,
        # which is what the check above covers.)
        jump = page.evaluate("() => window.__labJump")
        check("引擎的任意跳转纯函数重建在此 trace 上成立（且对照组确实翻车）",
              jump is not None and jump["pureFailures"] == 0 and
              jump["controlFailures"] > 0 and jump["conclusive"] and jump["passed"],
              json.dumps(jump, ensure_ascii=False) if jump else "missing")

        # ------------------------------------------------------------- shots
        print("\n== 截图 ==")
        page.click("[data-tier='num']") if page.query_selector("[data-tier='num']") else None
        page.evaluate("() => window.__p.setCursor(0)")
        page.wait_for_timeout(350)
        page.screenshot(path=SHOTS / "45-01-ledger-overview.png", full_page=True)

        page.locator("section.lab-panel[data-panel='ledger']").screenshot(
            path=SHOTS / "45-02-ledger-step0.png")

        # the last step: the cache at its tallest
        page.evaluate("() => window.__p.setCursor(window.__p.index.lastStep)")
        page.wait_for_timeout(400)
        page.locator("section.lab-panel[data-panel='ledger']").screenshot(
            path=SHOTS / "45-03-ledger-last-step.png")

        page.evaluate("() => window.__p.setCursor(window.__p.index.lastStep - 2)")
        page.wait_for_timeout(300)
        page.locator("section.lab-panel[data-panel='ledger']").screenshot(
            path=SHOTS / "45-04-ledger-decode.png")

        # the slider panel, driven to a context where the cache dominates
        page.evaluate("""() => {
            const inputs = [...document.querySelectorAll('#ctl-row input')];
            const order = ['seq_len','layers','kv_bytes','H'];
            const set = (k, v) => {
                const el = inputs[order.indexOf(k)];
                el.value = String(v);
                el.dispatchEvent(new Event('input', {bubbles: true}));
            };
            set('seq_len', 4096); set('layers', 32); set('kv_bytes', 2); set('H', 8);
        }""")
        page.wait_for_timeout(400)
        page.locator("#ctl-ledger").screenshot(path=SHOTS / "45-05-sliders-long-context.png")

        page.locator("#matrix-demo").screenshot(path=SHOTS / "45-06-matrix-zero.png")
        page.locator("section.lab-panel", has_text="自检 · 账本数值与契约").screenshot(
            path=SHOTS / "45-07-ledger-selfcheck.png")

        page.evaluate("() => window.__p.setCursor(0)")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "45-08-ledger-full-dark.png", full_page=True)

        light = browser.new_page(viewport={"width": W, "height": H}, color_scheme="light")
        light.goto(url)
        light.wait_for_timeout(800)
        light.locator("section.lab-panel[data-panel='ledger']").screenshot(
            path=SHOTS / "45-09-ledger-light.png")

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
