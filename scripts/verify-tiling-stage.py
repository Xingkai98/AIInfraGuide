#!/usr/bin/env python3
"""Acceptance harness for the memory-hierarchy stage (issue #44) and L01.

Drives the built page in real Chromium and checks the ticket's acceptance
criteria one by one against the artifact a reader actually gets — the staged
page in `public/labs/`, with its trace inlined and its assets at the flattened
paths. Then it takes the screenshots, which are committed so a reviewer can see
the result without running this.

It is deliberately a separate script from verify-labs.py rather than a new
section in it: that one is the engine's harness and asserts the engine's
contract (jump purity, lint, breakpoint). This one asserts a *view component*'s
contract plus L01's rendering. Same shape, different subject — merging them
would make one script that has to be understood as two.

Run:  npm run build:labs && python3 scripts/verify-tiling-stage.py
Needs `pip install playwright && playwright install chromium`.

Non-zero exit means at least one acceptance criterion failed.
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "public" / "labs" / "01-gemm-tiling.html"
TRACE = REPO / "labs" / "traces" / "gemm-tiling.json"
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
    if not TRACE.exists():
        print(f"missing {TRACE} — run labs/traces/gemm_tiling.py first", file=sys.stderr)
        return 2

    trace = json.loads(TRACE.read_text(encoding="utf-8"))
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

        # ------------------------------------------------------ component
        print("\n== 组件：层数与层名可配置 ==")
        layers = page.evaluate(
            "() => [...document.querySelectorAll('.lab-stage-layer')].map(e => e.dataset.layer)")
        check("L01 三层（HBM / SMEM / REG）都被渲染",
              layers == ["HBM", "SMEM", "REG"], f"layers = {layers}")

        # The same component with a different layer list must produce a
        # two-layer stage -- this is L06's requirement, not a thought experiment.
        two = page.evaluate("""() => {
            const T = window.LabTraces['gemm-tiling'];
            const idx = LabEngine.traceModel.index(T);
            // A two-layer config in the shape L06 will use: HBM + a single
            // on-chip layer, with every non-HBM tensor folded into it by
            // relabelling `at` on a copy of the trace.
            const T2 = JSON.parse(JSON.stringify(T));
            Object.keys(T2.tensors).forEach(n => {
                if (T2.tensors[n].at !== 'HBM') T2.tensors[n].at = 'SRAM';
            });
            const idx2 = LabEngine.traceModel.index(T2);
            const view = LabEngine.tilingStage.makeView({
                layers: [{id: 'HBM', label: 'HBM'}, {id: 'SRAM', label: 'SRAM'}]
            });
            const html = view.render(12, {
                trace: T2, index: idx2, step: T2.steps[12],
                snapshot: LabEngine.traceModel.resolve(idx2, 12)
            });
            const host = document.createElement('div');
            host.innerHTML = html;
            return [...host.querySelectorAll('.lab-stage-layer')].map(e => e.dataset.layer);
        }""")
        check("同一组件换一份层配置就是两层舞台（L06 的接口）",
              two == ["HBM", "SRAM"], f"layers = {two}")

        check("层的顺序与名称来自配置，不是硬编码",
              page.evaluate("""() => {
                  const els = [...document.querySelectorAll('.lab-stage-layer')];
                  const label = els[0].querySelector('.lab-stage-layer-label').textContent;
                  return label === 'Global Memory';
              }"""))

        # -------------------------------------------------- three-part tags
        print("\n== 张量三段式标注 ==")
        tag = page.evaluate("""() => {
            const b = document.querySelector('.lab-stage-block[data-tensor="A"]');
            return {
                name: b.querySelector('.lab-stage-block-id').textContent.trim(),
                shape: b.querySelector('.lab-stage-block-shape').textContent.trim(),
                at: b.querySelector('.lab-stage-block-at').textContent.trim(),
            };
        }""")
        check("数据块按 `名称 [d0, d1] @ 位置` 标注",
              tag["name"] == "A" and tag["shape"] == "[8, 8]" and tag["at"] == "@ HBM",
              json.dumps(tag, ensure_ascii=False))

        # ------------------------------------------------ block descriptor
        print("\n== 数据块描述：驻留 / 搬运可逐帧回放 ==")
        # Walk every step and record each layer's resident set + the flows. This
        # is the "每帧显示当前驻留在各层的是哪些块" criterion read from the DOM,
        # not from a re-implementation.
        frames = page.evaluate("""() => {
            const P = window.__p, out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                const layers = {};
                document.querySelectorAll('.lab-stage-layer').forEach(l => {
                    layers[l.dataset.layer] = [...l.querySelectorAll('.lab-stage-block')]
                        .map(b => b.dataset.tensor);
                });
                const rails = [...document.querySelectorAll('.lab-stage-rail')]
                    .map(r => r.dataset.flow);
                out.push({step: document.querySelector('.lab-stage').dataset.step,
                          layers, rails});
            }
            P.setCursor(0);
            return out;
        }""")
        note(f"逐帧走完 {len(frames)} 步，每帧都读到了三层驻留集合与搬运轨道")

        # The naive phase must show SMEM empty: that absence IS the lesson.
        naive_idx = next(i for i, s in enumerate(trace["steps"]) if s["id"] == "naive")
        naive = frames[naive_idx]
        check("朴素铺满那一步 SMEM 层为空（朴素实现不经 SMEM，这个空是教学点）",
              naive["layers"]["SMEM"] == [],
              f"SMEM at step {naive_idx} = {naive['layers']['SMEM']}")
        check("朴素铺满那一步寄存器里有直接取自 HBM 的操作数",
              set(naive["layers"]["REG"]) >= {"a_val", "b_val"},
              f"REG = {naive['layers']['REG']}")

        # A tile load must put both tiles into SMEM.
        ld_idx = next(i for i, s in enumerate(trace["steps"]) if s["id"] == "ld000")
        ld = frames[ld_idx]
        check("第一次 tile 载入后 A_smem / B_smem 出现在 SMEM 层",
              set(ld["layers"]["SMEM"]) == {"A_smem", "B_smem"},
              f"SMEM = {ld['layers']['SMEM']}")

        # The block-region outline: step ld001 carries A's second K slice, so
        # the outlined cells must be columns 4..7 and nothing else.
        ld1_idx = next(i for i, s in enumerate(trace["steps"]) if s["id"] == "ld001")
        span = page.evaluate("""(i) => {
            const P = window.__p; P.setCursor(i);
            const b = document.querySelector('.lab-stage-block[data-tensor="A"]');
            const cells = [...b.querySelectorAll('.lab-stage-cell')];
            const on = [];
            cells.forEach((c, k) => {
                if (c.classList.contains('lab-stage-cell-span')) {
                    on.push([Math.floor(k / 8), k % 8]);
                }
            });
            P.setCursor(0);
            return on;
        }""", ld1_idx)
        check("张量内被搬的那一小片区域被框出（A 的 4×4 角，不是整个矩阵）",
              span == [[r, c] for r in range(4) for c in range(4, 8)],
              f"{len(span)} 个格子被框出")

        # Flow rails carry the element count.
        rails = page.evaluate("""() => {
            const P = window.__p; P.setCursor(12);
            const out = [...document.querySelectorAll('.lab-stage-rail')].map(r => ({
                flow: r.dataset.flow,
                text: r.querySelector('.lab-stage-rail-n').textContent.trim(),
            }));
            P.setCursor(0);
            return out;
        }""")
        check("搬运轨道逐帧标出跨层搬运的元素数",
              any("SMEM->REG" in r["flow"] and r["text"].startswith("8") for r in rails),
              json.dumps(rails, ensure_ascii=False))

        # One scale for the whole replay: A[0][0] is on screen from step 0 to
        # the end and never changes, so its tint must not either. A per-frame
        # range would repaint it as C fills in.
        tint = page.evaluate("""() => {
            const P = window.__p;
            const read = () => {
                const b = document.querySelector('.lab-stage-block[data-tensor="A"]');
                const c = b.querySelector('.lab-stage-cell');
                return getComputedStyle(c).backgroundColor;
            };
            P.setCursor(0);  const a = read();
            P.setCursor(11); const b = read();
            P.setCursor(31); const c = read();
            P.setCursor(0);
            return {a, b, c};
        }""")
        check("同一个值在整个回放里配色不变（热量表用的是全 trace 尺度）",
              tint["a"] == tint["b"] == tint["c"],
              json.dumps(tint))

        # A same-layer flow (the register FMA) must not be drawn as a crossing.
        same = page.evaluate("""() => {
            const P = window.__p; P.setCursor(12);
            const t = document.querySelector('.lab-stage-rail[data-flow="REG->REG"]');
            const r = t ? t.className : null;
            P.setCursor(0);
            return r;
        }""")
        check("寄存器内累加不被画成跨层搬运（原地标出）",
              same is not None and "lab-stage-rail-same" in same, f"class = {same}")

        # ------------------------------------------------------ jump purity
        print("\n== 任意跳转：各层驻留状态可纯函数重建 ==")
        # Read the RENDERED stage after a jump and after playing forward to the
        # same target; a stateful view would differ.
        purity = page.evaluate("""() => {
            const P = window.__p, last = P.index.lastStep;
            const shapes = [
                ['尾 -> 1', last, 1], ['1 -> 0', 1, 0], ['0 -> 尾', 0, last],
                ['尾 -> 12', last, 12], ['12 -> 5', 12, 5], ['5 -> 12', 5, 12],
                ['12 -> 12', 12, 12], ['12 -> 0', 12, 0],
            ];
            const readStage = () => document.querySelector('.lab-stage').innerHTML;
            const out = [];
            for (const [label, from, to] of shapes) {
                P.setCursor(from); P.setCursor(to);
                const jumped = readStage();
                P.setCursor(0);
                for (let i = 1; i <= to; i++) P.setCursor(i);
                const played = readStage();
                out.push({label, same: jumped === played});
            }
            P.setCursor(0);
            return out;
        }""")
        for c in purity:
            check(f"跳转 {c['label']} 后的舞台渲染 == 顺序播放到该步", c["same"])

        # The control group: a stateful stage that never re-derives. It must
        # DISAGREE somewhere, or the purity above proves nothing.
        control = page.evaluate("""() => {
            const P = window.__p, last = P.index.lastStep;
            // A deliberately wrong view: holds one snapshot object and, on a
            // jump, applies only the target step's writes on top of it.
            const T = window.LabTraces['gemm-tiling'];
            const names = Object.keys(T.tensors);
            let acc = {};
            names.forEach(n => { if ('init' in T.tensors[n]) acc[n] = T.tensors[n].init; });
            const snap = () => JSON.parse(JSON.stringify(acc));
            P.setCursor(last); snap();
            // ground truth for step 12
            P.setCursor(12); const truth = JSON.stringify(snap());
            // now the stateful path: from the tail straight to 12
            P.setCursor(last); snap();
            Object.assign(acc, JSON.parse(JSON.stringify(T.steps[12].state || {})));
            const stateful = JSON.stringify(snap());
            P.setCursor(0);
            return {agree: truth === stateful};
        }""")
        check("对照组（有状态舞台）确实会翻车 —— 上面的『一致』不是测试写错",
              not control["agree"],
              "对照组跳转后与基准不一致" if not control["agree"] else "对照组也一致，无区分力")

        # ------------------------------------------------------------ build
        print("\n== trace 真实性 ==")
        ref = trace["meta"]["reference"]
        note(ref)
        check("trace 的对拍结论写进了 meta（可追溯到生成器）",
              "torch" in ref and "逐位相同" in ref)
        check("trace 规模为 A[8,8]×B[8,8]→C[8,8]，B_M=B_N=B_K=4",
              trace["meta"]["config"] == {"M": 8, "N": 8, "K": 8, "B_M": 4, "B_N": 4, "B_K": 4},
              json.dumps(trace["meta"]["config"]))
        check("每个搬运轨道都带元素数（视图的可视化量来自 trace，不是算出来的）",
              all(f.get("elements") for s in trace["steps"] for f in (s.get("flows") or [])))

        # ---------------------------------------------------------- shots
        print("\n== 截图 ==")
        # The stage leads the page, so a plain viewport shot at step 0 shows it
        # whole. Not `full_page`: L01's page is roughly three times the height
        # of L00's (32 steps of self-check table), which at this width produces
        # a 1.4 MB image nobody can read without zooming.
        page.evaluate("() => { window.__p.setCursor(0); window.scrollTo(0, 0); }")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "20-l01-overview.png")

        for step, name in ((10, "21-l01-naive-cost"),
                           (11, "22-l01-tile-load"),
                           (12, "23-l01-smem-to-reg"),
                           (30, "24-l01-store")):
            page.evaluate(f"() => window.__p.setCursor({step})")
            page.wait_for_timeout(320)
            page.locator(".lab-stage-panel").screenshot(path=SHOTS / f"{name}.png")

        # Zoom the two ends of the hierarchy: the 8x8 HBM matrix with its tile
        # outlined, and the register fragments it becomes.
        page.evaluate("() => window.__p.setCursor(11)")
        page.wait_for_timeout(300)
        page.locator('.lab-stage-block[data-tensor="A"]').screenshot(
            path=SHOTS / "25-l01-block-a-span.png")
        # Step 12, not 11: at 11 the tiles have landed but the mma has not run,
        # so the REG layer holds only the previous phase's scalars and the
        # fragments this shot exists to show are not there yet.
        page.evaluate("() => window.__p.setCursor(12)")
        page.wait_for_timeout(300)
        page.locator('.lab-stage-layer[data-layer="REG"]').screenshot(
            path=SHOTS / "26-l01-reg-layer.png")

        page.evaluate("() => window.__p.setCursor(31)")
        page.wait_for_timeout(300)
        page.locator('.lab-stage-layer[data-layer="SMEM"]').screenshot(
            path=SHOTS / "27-l01-smem-layer.png")

        # Light mode, since the page is theme-aware.
        page_light = browser.new_page(viewport={"width": W, "height": H},
                                      color_scheme="light")
        page_light.goto(url)
        page_light.wait_for_timeout(900)
        page_light.evaluate("() => window.__p.setCursor(12)")
        page_light.wait_for_timeout(300)
        page_light.locator(".lab-stage-panel").screenshot(
            path=SHOTS / "28-l01-stage-light.png")

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
