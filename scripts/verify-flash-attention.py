#!/usr/bin/env python3
"""Acceptance harness for L06 · FlashAttention V1 and the two-layer stage.

Drives the built page in real Chromium and checks the ticket's acceptance
criteria one by one against the artifact a reader actually gets — the staged page
in `public/labs/`, with its trace set inlined and its assets at the flattened
paths. Then it takes the screenshots, which are committed so a reviewer can see
the result without running this.

It is deliberately a separate script from verify-labs.py (the engine's harness),
verify-tiling-stage.py, verify-ledger.py, verify-gantt.py and verify-ring.py, for
the same reason those are separate from each other: this one asserts a *lab*'s
acceptance criteria plus the two-layer configuration of a shared view component,
and merging them would make one script that has to be understood as seven.

THE ONE JOB THE OTHERS DO NOT HAVE
----------------------------------
This lab's headline claim — *S and P never fall back to HBM* — is the kind of
claim that is easy to assert in prose and easy to get wrong in fact. It is
verified here at THREE independent levels, and each level carries its own control
group:

  1. **Python** (in the generator): two real numpy implementations, both reaching
     memory only through one counting object. FA's access log names no S or P; the
     SAME log object on the standard implementation does.
  2. **The trace** (here, section "S、P 从不落到 HBM"): every frame of the
     rendered replay is walked and the HBM row is read out of the DOM. A
     deliberately broken trace — S moved to HBM — must make the same walk fail.
  3. **The stage's own lint** (here, section "契约 lint 的两份实现"): the
     generator's whole sabotage table is run through both ports in node.

Level 2 is the one a reader can see, and it is why the control matters: a walk
that reports "no S in HBM" on a page that could not have shown S in HBM anyway
proves nothing. The control is a trace where it must appear.

Run:  python3 labs/traces/flash_attention.py          # writes the trace set
      npm run build:labs && python3 scripts/verify-flash-attention.py
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
PAGE = REPO / "public" / "labs" / "04-flash-attention.html"
TRACE_DIR = REPO / "labs" / "traces"
MANIFEST = TRACE_DIR / "flash-attention.manifest.json"
GENERATOR = TRACE_DIR / "flash_attention.py"
STAGE_JS = REPO / "labs" / "assets" / "engine" / "views" / "tiling-stage.js"
SHOTS = REPO / "labs" / "pages" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1100
# The configurations replayed end to end. Not all seven: the walk is O(steps) per
# configuration and three is enough to cover both `d`, both `N` and the smallest
# block size -- the `N8-d16-M128` one is the 118-step shape, which is where a
# layout or aliasing bug would show up first.
WALK_CONFIGS = ["flash-attention-N8-d16-M256", "flash-attention-N8-d8-M128",
                "flash-attention-N16-d16-M256"]

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


# ------------------------------------------------------------- lint parity
def lint_parity():
    """Run the generator's sabotage table through both contract lints.

    The Python lint runs inside the generator and reports as it goes; this asks
    the generator for the payload (`--js-lint`) and pushes the same broken copies
    through `LabEngine.tilingStage.lint` in node.

    The two ports are not meant to be identical rule-for-rule: the Python side
    lints the whole contract (formulas, bindings, graph, `state`, the correction
    block, the exposure/IO summaries), while the JS side lints the fields the
    view and its sibling panels actually render, and leaves `state` / formulas /
    graph to `trace-model.js`, which already carries their JS port. So the
    requirement is not "every rule in both files" — it is:

      * every sabotage is caught by AT LEAST ONE port;
      * every sabotage that is in the component's scope (spans / flows / occ /
        io / meta.sram) is caught by the JS port;
      * every sabotage actually changes the trace, on both sides. A case written
        against a literal that stops biting — a no-op — proves nothing when both
        lints report zero, and it is a different bug from "the lint missed it".
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
load('labs/assets/engine/tensors.js');
load('labs/assets/engine/views/tiling-stage.js');
const NS = g.LabEngine;
// Non-finite floats travel as tags -- `json.dumps` writes bare NaN / Infinity,
// which are legal JavaScript and illegal JSON, so a payload carrying one could
// not be parsed at all. See `_payload_json` in the generator.
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
  const r = NS.tilingStage.lint(tr);
  out.clean[name] = { gaps: r.gaps.length, warns: r.warns.length };
}
for (const [name, s] of Object.entries(payload.sabotages)) {
  if (s.error) { out.sabotage[name] = { error: s.error }; continue; }
  let stage = 0;
  try { stage = NS.tilingStage.lint(s.trace).gaps.length; }
  catch (e) { stage = 'ERR:' + e.message; }
  out.sabotage[name] = { stage: stage, changed: s.changed };
}
process.stdout.write(JSON.stringify(out));
"""
    scratch = Path("/tmp") / "flash-attention-lint-probe.js"
    scratch.write_text(node_src, encoding="utf-8")
    payload_file = Path("/tmp") / "flash-attention-lint-payload.json"
    payload_file.write_text(proc.stdout, encoding="utf-8")
    node = subprocess.run(["node", str(scratch), str(payload_file)],
                          capture_output=True, text=True, cwd=str(REPO))
    if node.returncode != 0:
        check("JS 侧 lint 能在 node 里跑起来", False, node.stderr[-400:])
        return
    res = json.loads(node.stdout)

    clean = {k: v["gaps"] for k, v in res["clean"].items()}
    check("JS 侧 lint 对全部干净 trace 报 0 gap（不是永远报 gap）",
          all(v == 0 for v in clean.values()), json.dumps(clean, ensure_ascii=False))

    total = len(res["sabotage"])
    dead = [k for k, v in res["sabotage"].items() if not v.get("changed")]
    errs = [k for k, v in res["sabotage"].items() if v.get("error")]
    caught = [k for k, v in res["sabotage"].items() if isinstance(v.get("stage"), int)
              and v["stage"] > 0]
    check(f"每一种破坏都真的改变了 trace（没有空操作，共 {total} 种）",
          not dead, "空操作：" + "、".join(sorted(dead)))
    check("每一种破坏都能被施加（不会半路抛异常）",
          not errs, "失败：" + "、".join(sorted(errs)))

    # The component's own scope, by the tag the case list groups under. Listed
    # explicitly rather than inferred so a new case that forgets to land in
    # either list is visible as a count mismatch below.
    component = [k for k in res["sabotage"] if k.startswith(("span ", "flow ", "occ ", "S moved",
                                                            "P removed", "a third layer",
                                                            "step without an occ",
                                                            "step without an io",
                                                            "io.cum"))]
    missed_component = [k for k in component if k not in caught]
    check(f"组件职责内的破坏（spans / flows / occ / io，共 {len(component)} 种）"
          "全部被 JS 侧 lint 抓到",
          not missed_component, "漏掉：" + "、".join(sorted(missed_component)))

    # And the whole table must be caught by at least one of the two ports: a case
    # neither side catches is a rule that exists only in the case's name.
    pyside = subprocess.run([sys.executable, str(GENERATOR)],
                            capture_output=True, text=True, cwd=str(REPO))
    text = pyside.stdout + pyside.stderr
    ok = ("LINT-DEAD" not in text and "SABOTAGE-DEAD" not in text and
          f"对照 {total} 种破坏" in text)
    check(f"Python 侧 lint 独自覆盖了 JS 侧职责之外的规则（对照 {total} 种全部被抓到）",
          ok,
          f"生成器自报 {total} 种全部抓到" if ok else "生成器输出里没有出现「全部被抓到」")
    note(f"两份 lint 各跑 {total} 种破坏：JS 侧覆盖组件字段（{len(component)} 种），"
         f"Python 侧覆盖全契约；两侧都要求「破坏真的改变了 trace」")


# --------------------------------------------------------- the headline claim
def load_config(page, url, cfg_id, timeout=1800):
    """Navigate to a configuration and wait for its player to be up.

    The page reads `?cfg=` on load, so this is the deep link a reader would use,
    not a test-only back door. Reloading rather than driving the slider keeps the
    walk honest: what gets asserted is the page a reader lands on, with the
    config's own trace inlined and its own player bound to it.
    """
    page.goto(f"{url}?cfg={cfg_id}&step=0")
    page.wait_for_timeout(timeout)
    got = page.evaluate("() => window.__labConfig")
    if got != cfg_id:
        raise AssertionError(f"深链 ?cfg={cfg_id} 载入了 {got}")
    return got


def walk_exposure(page, cfg_id, trace):
    """Walk every frame of a rendered replay and read the HBM row out of the DOM.

    This is the claim as a reader can verify it: not "the generator says so" but
    "the picture on screen does not have S or P in the bottom layer at any step".
    Returns the list of frames where a forbidden tensor appeared, plus the count
    of steps walked.

    The caller must have loaded `cfg_id` first (`load_config`): the stage panel
    closes over its own trace, so walking steps from one trace against a stage
    built from another is comparing two different things -- which the component
    catches by throwing, as it should.
    """
    return page.evaluate("""(id) => {
        const T = window.LabTraces[id];
        const P = window.__p;
        const forbidden = T.meta.exposure.forbidden;
        // When each forbidden tensor is first PRODUCED: the first step id that
        // writes it. Until then it does not exist and its absence from SRAM is
        // not evidence of anything.
        const born = {};
        T.steps.forEach((s, i) => {
            (s.writes || []).forEach(w => {
                if (forbidden.indexOf(w) !== -1 && born[w] === undefined) born[w] = i;
            });
        });
        const hits = [];
        let frames = 0;
        for (let i = 0; i <= P.index.lastStep; i++) {
            P.setCursor(i);
            const hbm = [...document.querySelectorAll(
                '.lab-stage-layer[data-layer="HBM"] .lab-stage-block')]
                .map(b => b.dataset.tensor);
            const sram = [...document.querySelectorAll(
                '.lab-stage-layer[data-layer="SRAM"] .lab-stage-block')]
                .map(b => b.dataset.tensor);
            frames++;
            forbidden.forEach(f => {
                if (hbm.indexOf(f) !== -1) hits.push({step: i, tensor: f, where: 'HBM'});
                // And the complement, which is what keeps the first half from
                // being vacuous: once produced, the tensor must be ON SCREEN in
                // SRAM on every later frame. A stage that simply never showed it
                // would pass the "not in HBM" check for the wrong reason.
                if (born[f] !== undefined && i >= born[f] && sram.indexOf(f) === -1) {
                    hits.push({step: i, tensor: f, where: 'neither'});
                }
            });
        }
        P.setCursor(0);
        return {hits: hits, frames: frames, born: born};
    }""", cfg_id)


def main():
    for p, hint in ((PAGE, "run `npm run build:labs` first"),
                    (MANIFEST, "run labs/traces/flash_attention.py first")):
        if not p.exists():
            print(f"missing {p} — {hint}", file=sys.stderr)
            return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    traces = {n: load(n) for n in manifest["traces"]}
    default = manifest["default"]
    url = PAGE.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        page.goto(url)
        page.wait_for_timeout(1800)

        # ============================================ topology of the stage
        print("\n== 双层舞台：HBM / SRAM ==")
        layers = page.evaluate(
            "() => [...document.querySelectorAll('.lab-stage-layer')].map(e => e.dataset.layer)")
        check("舞台只有两层，且顺序是 HBM 在上、SRAM 在下",
              layers == ["HBM", "SRAM"], f"layers = {layers}")
        check("层的名称来自配置，不是硬编码",
              page.evaluate("""() => {
                  const els = [...document.querySelectorAll('.lab-stage-layer')];
                  return els[0].querySelector('.lab-stage-layer-label').textContent
                             .indexOf('Global') !== -1;
              }"""))
        # The component's own interface: it must be driven by the layer list, so
        # the same JS with three layers still produces three (L01's regression).
        two = page.evaluate("""() => {
            const T = window.LabTraces['flash-attention-N8-d16-M256'];
            const idx = LabEngine.traceModel.index(T);
            const view = LabEngine.tilingStage.makeView({
                layers: [{id: 'HBM', label: 'HBM'}, {id: 'SRAM', label: 'SRAM'},
                         {id: 'REG', label: 'REG'}]
            });
            const html = view.render(4, {trace: T, index: idx, step: T.steps[4],
                snapshot: LabEngine.traceModel.resolve(idx, 4)});
            const host = document.createElement('div');
            host.innerHTML = html;
            return [...host.querySelectorAll('.lab-stage-layer')].map(e => e.dataset.layer);
        }""")
        check("同一组件的层数是配置驱动的（三层配置就是三层舞台）",
              two == ["HBM", "SRAM", "REG"], f"layers = {two}")

        # ============================================== the headline claim
        print("\n== S、P 从不落到 HBM（逐帧读 DOM）==")
        exposure = traces[default]["meta"]["exposure"]
        forbidden = exposure["forbidden"]
        note(f"生成器的访问日志：FA {exposure['flash_forbidden_accesses']} 次 · "
             f"标准实现 {exposure['standard_forbidden_accesses']} 次（对照组）")

        for cfg_id in WALK_CONFIGS:
            load_config(page, url, cfg_id)
            r = walk_exposure(page, cfg_id, traces[cfg_id])
            check(f"{cfg_id}：走完 {r['frames']} 帧，HBM 层从未出现 S 或 P"
                  "（且两者确实都在 SRAM 里出现过）",
                  not r["hits"],
                  "干净" if not r["hits"] else json.dumps(r["hits"][:6], ensure_ascii=False))
        # Back to the default for the rest of the checks.
        load_config(page, url, default)

        # ---- the control: the same walk must FAIL on a trace with S in HBM ----
        # Without this, "S never appears in HBM" could be a property of a walk
        # that cannot see anything, or of a stage configured to hide the layer.
        control = page.evaluate("""() => {
            const T = JSON.parse(JSON.stringify(window.LabTraces['flash-attention-N8-d16-M256']));
            // Move S to HBM and give it a resident value on every step, so a
            // walk that is actually reading the layer must find it.
            T.tensors.S.at = 'HBM';
            T.tensors.S.init = T.tensors.S.shape[0] === 0 ? 0
                             : Array(T.tensors.S.shape[0]).fill(
                                 Array(T.tensors.S.shape[1]).fill(0.5));
            const idx = LabEngine.traceModel.index(T);
            const view = LabEngine.tilingStage.makeView({
                layers: [{id: 'HBM', label: 'HBM'}, {id: 'SRAM', label: 'SRAM'}]
            });
            const html = view.render(4, {trace: T, index: idx, step: T.steps[4],
                snapshot: LabEngine.traceModel.resolve(idx, 4)});
            const host = document.createElement('div');
            host.innerHTML = html;
            return [...host.querySelectorAll(
                '.lab-stage-layer[data-layer="HBM"] .lab-stage-block')]
                .map(b => b.dataset.tensor);
        }""")
        check("对照组：把 S 挪到 HBM 后，同一段 DOM 读取确实能看见它 —— "
              "上面的「干净」不是读不到东西",
              "S" in control, f"HBM = {control}")

        # ================================================== SRAM occupancy
        print("\n== SRAM 占用账 ==")
        occ = page.evaluate("""() => {
            const P = window.__p, T = window.__labTrace;
            const out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                const e = document.querySelector('.lab-stage-occ');
                if (!e) { out.push({step: i, occ: null}); continue; }
                out.push({
                    step: i,
                    stepId: T.steps[i].id,
                    el: +e.dataset.occElements,
                    cap: +e.dataset.occCapacity,
                    over: e.className.indexOf('over') !== -1,
                    parts: [...e.querySelectorAll('.lab-stage-occ-seg')].map(s => s.dataset.part),
                    legend: [...e.querySelectorAll('.lab-stage-occ-key')].map(k => k.textContent),
                });
            }
            P.setCursor(0);
            return out;
        }""")
        present = [f for f in occ if f.get("el") is not None]
        sram_live = [f for f in occ if f.get("stepId") != "sum"]
        check("每一步都有 SRAM 占用账（trace 声明了 meta.sram）",
              len(present) == len(sram_live),
              f"{len(present)} / {len(sram_live)} 帧有账"
              + (f"；{len(occ) - len(sram_live)} 帧没有（收尾步 SRAM 已空，是对的）"
                 if len(occ) != len(sram_live) else ""))
        names = present[-1]["parts"] if present else []
        check("占用账列出的是算法真正驻留的块（Q_i / K_j / V_j / O_i / 合并的 S/P / m / l）",
              set(names) >= {"Q_i", "K_j", "V_j", "O_i", "m_i", "l_i"},
              json.dumps(names, ensure_ascii=False))
        # The peak the trace publishes must be the peak the page shows.
        peak = traces[default]["meta"]["sram"]
        shown_peak = max(present, key=lambda f: f["el"]) if present else None
        check("页面上的峰值 == trace 里发布 meta.sram.used",
              shown_peak and shown_peak["el"] == peak["used"],
              f"页面 {shown_peak and shown_peak['el']} / meta {peak['used']}")
        # The over-capacity state must be reachable and legible -- the ticket
        # asks for a prompt when B_r/B_c exceed M.
        over = [f for f in present if f["over"]]
        check("超出 SRAM 容量时占用账进入「超出」状态（教程公式省略的项把它顶出去了）",
              bool(over),
              f"{len(over)} 帧超出，最先在 {over[0]['stepId'] if over else '—'}")
        if over:
            o = over[0]
            check("超出的那一帧标出了超出量，而不是把条裁掉",
                  any("超出" in s for s in o["legend"]),
                  json.dumps(o["legend"][-1:], ensure_ascii=False))

        # ================================================== the IO counter
        print("\n== IO 计数器 ==")
        io = traces[default]["meta"]["io"]
        counter = page.evaluate("""() => {
            const P = window.__p, T = window.__labTrace;
            const out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                const el = document.querySelector('[data-panel="io"]');
                out.push({step: i, cum: T.steps[i].io.cum,
                          text: el ? el.innerText : ''});
            }
            P.setCursor(0);
            return out;
        }""")
        # The displayed total must be the trace's own running sum at every step:
        # a counter that printed `cum` of the wrong step would still be monotone.
        mismatched = []
        for f in counter:
            m = re.search(r"([\d,]+)\s*个元素已访问", f["text"])
            if not m:
                mismatched.append(f["step"])
                continue
            if int(m.group(1).replace(",", "")) != f["cum"]:
                mismatched.append(f["step"])
        check(f"逐帧显示的累计访问量 == trace 里 {len(counter)} 步的 io.cum",
              not mismatched, f"不一致的步：{mismatched[:8]}")
        last = counter[-1]
        m = re.search(r"([\d,]+)\s*个元素已访问", last["text"])
        check("回放结束时计数器 == 实测总量（不是按公式填的）",
              m and int(m.group(1).replace(",", "")) == io["flash"]["elements"],
              f"显示 {m and m.group(1)} / trace {io['flash']['elements']}")

        # The comparison with the standard implementation must be on screen, and
        # it must be the measured pair.
        panel_text = page.evaluate(
            "() => document.querySelector('[data-panel=\"io\"]').innerText")
        check("面板并列给出标准 Attention 的实测 IO 与两者的比值",
              f"{io['standard']['elements']:,}" in panel_text and
              f"{io['flash']['elements']:,}" in panel_text and
              f"{io['ratio']:.3g}"[:4] in panel_text,
              f"标准 {io['standard']['elements']} / FA {io['flash']['elements']}")

        # The set difference: every tensor the control touched, tagged with
        # whether FA touched it.
        std_tensors = set(io["standard"]["tensors"])
        fa_tensors = set(io["flash"]["tensors"])
        check("面板标出了「标准实现访问过、FA 从未访问」的张量",
              bool(std_tensors - fa_tensors) and
              set(forbidden) == std_tensors - fa_tensors,
              f"集合差 = {sorted(std_tensors - fa_tensors)}")

        # The curves.
        curves = traces[default]["meta"]["io"]["curves"]
        n_fa = page.evaluate(
            "() => document.querySelectorAll('[data-panel=\"curves\"] path.fa').length")
        check(f"实测曲线画出了 {len(curves['flash'])} 条 FA 曲线 + 1 条标准曲线",
              n_fa == len(curves["flash"]),
              f"DOM 里 {n_fa} 条 / meta 里 {len(curves['flash'])} 条")
        check("标准曲线与 FA 曲线在同一张图上（同一条横轴）",
              page.evaluate(
                  "() => document.querySelectorAll('[data-panel=\"curves\"] path.std').length") == 1)
        # The law: IO x B_c is constant across block sizes. The table must show
        # the trace's own rows, not a recomputation.
        law = traces[default]["meta"]["io"]["law"]
        table = page.evaluate("""() => {
            const rows = [...document.querySelectorAll('[data-panel="curves"] table tbody tr')];
            return rows.map(r => [...r.querySelectorAll('td')].map(td => td.textContent.trim()));
        }""")
        # The table's own columns: B_c, B_r, N, IO, IO x B_c. Compared field by
        # field rather than against the row's concatenated text, because a
        # substring test on the whole row passes when a number appears in the
        # wrong column.
        def as_int(s_):
            return int(s_.replace(",", ""))
        table_ok = len(table) == len(law["rows"])
        if table_ok:
            for r, row in zip(law["rows"], table):
                # Column 4 is the IO column, 5 the product -- see curvePanel.
                if (as_int(row[0]) != r["Bc"] or as_int(row[1]) != r["Br"] or
                        as_int(row[2]) != r["N"] or as_int(row[3]) != r["elements"] or
                        as_int(row[4]) != r["times_bc"]):
                    table_ok = False
                    break
        check(f"IO × B_c 表里是 {len(law['rows'])} 行实测数据，逐列与 trace 一致"
              "（O(N²d²/M) 的常数说法可检验）",
              table_ok, json.dumps(table, ensure_ascii=False))
        note(f"固定 d={traces[default]['meta']['config']['d']}、N={law['rows'][-1]['N']}："
             + " · ".join(f"B_c={r['Bc']}→{r['times_bc']:.4g}" for r in law["rows"])
             + f"（跨度 {law['spread']:.3f}）")

        # The panel's PROSE must agree with its own chart and its own numbers.
        # Two defects lived here and neither was visible to a numeric assertion:
        # a curve with no point at the current N fell back to its last point and
        # the sentence still said "at N=...", and a footnote quoted one tensor's
        # counts while its subject was two. Both are checked by recomputing the
        # claim from the trace and requiring the rendered text to match.
        cfg_ = traces[default]["meta"]["config"]
        std_at_n = [pt for pt in curves["standard"] if pt["N"] == cfg_["N"]]
        if std_at_n:
            winners = [c for c in curves["flash"]
                       for pt in c["points"] if pt["N"] == cfg_["N"]
                       and pt["elements"] < std_at_n[0]["elements"]]
            absent = [c for c in curves["flash"]
                      if not [pt for pt in c["points"] if pt["N"] == cfg_["N"]]]
            text = page.evaluate(
                "() => document.querySelector('[data-panel=\"curves\"]').innerText")
            # The page phrases the zero case as prose rather than as "0 条"; the
            # assertion mirrors that rather than forcing the number in.
            winners_claim = (f"{len(winners)} 条已经低于红线" if winners
                             else "一条都没有低于红线")
            check(f"曲线面板的结论文案与它自己画的图一致（N={cfg_['N']} 截面上 "
                  f"{len(winners)} 条低于红线、{len(absent)} 条无点）",
                  winners_claim in text and
                  (not absent or f"{len(absent)} 条（B_c =" in text),
                  text[text.find("在这个 N"):text.find("在这个 N") + 90]
                  if "在这个 N" in text else "(没找到结论句)")
        # The P/S footnote's arithmetic: the sum over the forbidden set, not one
        # member of it.
        fb = traces[default]["meta"]["exposure"]["forbidden"]
        want_r = sum(io["standard"]["by_tensor"][t]["read"] for t in fb
                     if t in io["standard"]["by_tensor"])
        want_w = sum(io["standard"]["by_tensor"][t]["write"] for t in fb
                     if t in io["standard"]["by_tensor"])
        io_text = page.evaluate(
            "() => document.querySelector('[data-panel=\"io\"]').innerText")
        check("IO 面板脚注的读/写次数是禁用集合的合计，不是其中一个张量的",
              f"合计 {want_r} 读 + {want_w} 写" in io_text,
              f"应为 {want_r} 读 + {want_w} 写")

        # ======================================== the correction factor (L00)
        print("\n== 修正因子在 m 变化的步上被触发（复用 L00 机制）==")
        corr = traces[default]["meta"]["correction"]
        corr_frames = page.evaluate("""() => {
            const P = window.__p, T = window.__labTrace;
            const out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                out.push({
                    step: i, id: T.steps[i].id,
                    kind: T.steps[i].corr ? T.steps[i].corr.kind : null,
                    shown: document.querySelector('[data-panel="corr"]').innerText,
                });
            }
            P.setCursor(0);
            return out;
        }""")
        triggered = [f for f in corr_frames if f["kind"] == "rescale"]
        check(f"trace 里 {len(triggered)} 步带 rescale 的修正因子",
              len(triggered) > 0)
        # Each rescale step must actually show its factor, not a placeholder.
        first = triggered[0]
        factor = traces[default]["steps"][first["step"]]["corr"]["factor"]
        check("rescale 的那一步在放大器里真的显示出因子值",
              f"{factor:.4g}"[:5] in first["shown"] or
              str(round(factor, 4)) in first["shown"],
              f"步 {first['ste'] if 'ste' in first else first['step']} "
              f"factor={factor:.6g}")
        # And the identity contrast: a block where forgetting costs nothing.
        identity = [f for f in corr_frames if f["kind"] == "identity"]
        check("默认配置里同时有一块 m 未变的 identity（对照组：漏掉它也不出错）",
              len(identity) > 0, f"{len(identity)} 步")
        kinds = {f["kind"] for f in corr_frames if f["kind"]}
        check("三种角色（init / identity / rescale）都出现在回放里",
              kinds == {"init", "identity", "rescale"},
              json.dumps(sorted(kinds), ensure_ascii=False))
        check("整条 trace 的「漏掉修正因子」代价是实测的、非零",
              corr["final_bias_abs"] > 0 and corr["final_bias_rel"] > 0,
              f"偏差 {corr['final_bias_abs']:.4g}（相对 {corr['final_bias_rel'] * 100:.4g}%）")
        # Per-block rows carry how many rows the block-level kind summarises.
        check("逐块的 kind 与逐行的 α 都标了出来（不让整块的 kind 平均掉逐行的差异）",
              all("rows_scaled" in e for e in corr["per_block"]),
              f"共 {corr['rows']} / {corr['rows_total']} 行真的缩放")

        # ==================================================== parameters
        print("\n== 参数：N / d / M 实时重算 ==")
        for cfg_id in manifest["traces"]:
            ok = page.evaluate("""(id) => {
                const T = window.LabTraces[id];
                const c = T.meta.config;
                // The formula the page prints must be the one the trace used.
                return c.Bc === Math.ceil(c.M / (4 * c.d)) &&
                       c.Br === Math.min(c.Bc, c.d) &&
                       c.Tr * c.Br === c.N && c.Tc * c.Bc === c.N;
            }""", cfg_id)
            if not ok:
                check(f"{cfg_id}：trace 里的 B_c / B_r 与公式一致", False)
                break
        else:
            check(f"{len(manifest['traces'])} 个配置的 B_c = ⌈M/4d⌉、B_r = min(B_c, d) 全部成立",
                  True)

        # Driving the sliders must recompute the derived parameters and reload.
        # Try every M position in turn and require that each one either loads a
        # trace whose blocking parameters match the formula, or says why not --
        # never that it silently does nothing.
        slider = page.evaluate("""() => {
            const out = [];
            const s = document.querySelector('[data-param="M"]');
            for (let k = 0; k < s.max; k++) {
                // Re-query each time: `apply` rebuilds the whole bar.
                const el = document.querySelector('[data-param="M"]');
                el.value = String(k);
                el.dispatchEvent(new Event('input', {bubbles: true}));
                const note = document.getElementById('l06-params-note');
                out.push({
                    k: k,
                    cfg: window.__labConfig,
                    note: note ? note.textContent : '',
                    hidden: note ? note.hidden : true,
                    derived: [...document.querySelectorAll('.l06-derived span')]
                               .map(e => e.innerText),
                });
            }
            return out;
        }""")
        page.wait_for_timeout(900)
        bad = []
        for row in slider:
            if not row["hidden"] and row["note"]:
                continue                       # a skipped cell that explains itself
            loaded = traces.get(row["cfg"])
            if loaded is None:
                bad.append(row)
                continue
            c = loaded["meta"]["config"]
            # The derived strip's first four entries are B_c, B_r, T_r, T_c in
            # that order (see `buildParamBar`). Compared position by position
            # rather than by substring: "= 4" would match either of the first
            # two, so a page that swapped them would pass a contains-test.
            d_ = row["derived"]
            want = [f"= {c['Bc']}", f"= {c['Br']}", f"= {c['Tr']}", f"= {c['Tc']}"]
            if len(d_) < 4 or any(w not in d_[i] for i, w in enumerate(want)):
                bad.append(row)
        check(f"M 滑杆的 {len(slider)} 个位置：每个要么载入一份按公式重算的 trace，"
              "要么说明这一格为什么不回放（没有静默失效的位置）",
              not bad,
              "问题位置：" + json.dumps([b["k"] for b in bad]))
        # The skip note must name a reason, not just refuse.
        skipped_rows = [r for r in slider if not r["hidden"] and r["note"]]
        check("被跳过的格子给出的理由是生成脚本自己写的（不是通用文案）",
              all("没有回放" in r["note"] and len(r["note"]) > 30 for r in skipped_rows),
              f"{len(skipped_rows)} 个格子有说明")
        # Back to the default before the remaining checks.
        page.goto(url)
        page.wait_for_timeout(1400)

        # ================================================== jump purity
        print("\n== 任意跳转：纯函数重建 + 对照组 ==")
        jump = page.evaluate("() => window.__labJump")
        check(f"页面对自己报告的 {jump['total']} 次跳转里，纯函数重建 0 次不一致",
              jump["total"] > 0 and jump["pureFailures"] == 0,
              f"total={jump['total']} pureFailures={jump['pureFailures']}")
        check("对照组（有状态重建）确实翻车 —— 上面那个 0 不是测试写错",
              jump["conclusive"] and jump["controlFailures"] > 0,
              f"controlFailures={jump['controlFailures']} conclusive={jump['conclusive']}")

        # And the stage itself must be pure across a jump -- a stateful stage
        # would leave a block from the wrong step on screen with every counter
        # still green.
        stage_purity = page.evaluate("""() => {
            const P = window.__p, last = P.index.lastStep;
            const read = () => document.querySelector('.lab-stage').innerHTML;
            const out = [];
            [[last, 1], [1, 0], [0, last], [last, Math.floor(last / 2)],
             [Math.floor(last / 2), 5], [5, last]].forEach(([from, to]) => {
                P.setCursor(from); P.setCursor(to);
                const jumped = read();
                P.setCursor(0);
                for (let i = 1; i <= to; i++) P.setCursor(i);
                out.push({from, to, same: jumped === read()});
            });
            P.setCursor(0);
            return out;
        }""")
        for c in stage_purity:
            check(f"跳转 {c['from']}→{c['to']} 后的舞台 == 顺序播放到该步",
                  c["same"])

        # =================================================== geometry
        print("\n== 几何：不重叠、不压卡、不被裁 ==")
        # The stage is a picture, and a picture can be wrong in a way no count
        # detects: the previous view ticket found strips overlapping and one
        # clipped off-canvas while every numeric assertion was green. So the
        # geometry is asserted against the rendered boxes, not eyeballed.
        geom = page.evaluate("""() => {
            const P = window.__p, T = window.__labTrace;
            const hits = [];
            const clipped = [];
            const emptyOverlap = [];
            let maxFrame = 0;
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                const stage = document.querySelector('.lab-stage');
                if (!stage) continue;
                const sb = stage.getBoundingClientRect();
                const blocks = [...document.querySelectorAll('.lab-stage-block')].map(b => {
                    const r = b.getBoundingClientRect();
                    return {t: b.dataset.tensor,
                            layer: b.closest('.lab-stage-layer').dataset.layer,
                            l: r.left, top: r.top, r: r.right, b: r.bottom,
                            w: r.width, h: r.height};
                });
                maxFrame = Math.max(maxFrame, blocks.length);
                // Two blocks in the SAME layer must not overlap. (Across layers
                // they cannot, but asserting it per-layer keeps this a layout
                // check rather than a coincidence of the box model.)
                const byLayer = {};
                blocks.forEach(b => { (byLayer[b.layer] = byLayer[b.layer] || []).push(b); });
                Object.keys(byLayer).forEach(L => {
                    const arr = byLayer[L];
                    for (let a = 0; a < arr.length; a++) {
                        for (let c = a + 1; c < arr.length; c++) {
                            const x = arr[a], y = arr[c];
                            if (x.l < y.r && y.l < x.r && x.top < y.b && y.top < x.b) {
                                hits.push({step: i, layer: L, a: x.t, b: y.t});
                            }
                        }
                    }
                });
                // Nothing may be cut off by the page or the panel.
                const host = document.querySelector('.lab-stage-panel') || stage;
                const hb = host.getBoundingClientRect();
                blocks.forEach(b => {
                    if (b.w <= 0 || b.h <= 0) { emptyOverlap.push({step: i, t: b.t}); return; }
                    if (b.r > hb.right + 1 || b.l < hb.left - 1 ||
                        b.b > hb.bottom + 1 || b.top < hb.top - 1) {
                        clipped.push({step: i, t: b.t, layer: b.layer});
                    }
                });
            }
            P.setCursor(0);
            return {hits: hits, clipped: clipped, zeroSized: emptyOverlap,
                    maxFrame: maxFrame};
        }""")
        check(f"逐帧检查所有数据块：同层内两两不重叠（最密的一帧 {geom['maxFrame']} 块）",
              not geom["hits"],
              "干净" if not geom["hits"] else json.dumps(geom["hits"][:6], ensure_ascii=False))
        check("没有任何数据块被舞台面板裁掉（宽度/高度为 0 也算）",
              not geom["clipped"] and not geom["zeroSized"],
              json.dumps((geom["clipped"] + geom["zeroSized"])[:6], ensure_ascii=False))
        check("单元格都被渲染成有尺寸的盒子，不是被压成 0",
              page.evaluate("""() => {
                  const P = window.__p;
                  P.setCursor(0);
                  let zero = 0, total = 0;
                  for (let i = 1; i <= P.index.lastStep; i++) {
                      P.setCursor(i);
                      [...document.querySelectorAll('.lab-stage-block')].forEach(b => {
                          total++;
                          const r = b.getBoundingClientRect();
                          if (r.width < 8 || r.height < 8) zero++;
                      });
                  }
                  P.setCursor(0);
                  return {zero, total};
              }""")["zero"] == 0)

        # ---- a grid must actually CONTAIN its cells ------------------------
        #
        # This is the check that was missing, and its absence is instructive: a
        # grid element's box keeps the width it was given no matter how far its
        # contents spill, so every geometry assertion above passed while each
        # block rendered one visible column and clipped the other fifteen -- and
        # the overflow painted over the neighbouring blocks. `scrollWidth` is the
        # only measurement that sees it: if the content is wider than the box,
        # the tracks are wrong.
        spilling = page.evaluate("""() => {
            const P = window.__p, out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                [...document.querySelectorAll('.lab-stage-grid')].forEach(g => {
                    // 1px of rounding slack per grid, not per cell.
                    if (g.scrollWidth > g.clientWidth + 1) {
                        const b = g.closest('.lab-stage-block');
                        out.push({step: i, tensor: b && b.dataset.tensor,
                                  scroll: g.scrollWidth, client: g.clientWidth});
                    }
                });
            }
            P.setCursor(0);
            return out;
        }""")
        check("每个矩阵网格都装得下自己的格子（没有内容溢出到盒子外）",
              not spilling, json.dumps(spilling[:6], ensure_ascii=False))

        # And the per-cell width must be the intended one, not the block's: a
        # track sized to the whole block is exactly the bug above.
        tracks = page.evaluate("""() => {
            const P = window.__p;
            P.setCursor(4);
            const out = [];
            [...document.querySelectorAll('.lab-stage-grid')].forEach(g => {
                const b = g.closest('.lab-stage-block');
                const cols = getComputedStyle(g).gridTemplateColumns.split(' ').length;
                const w = g.getBoundingClientRect().width / cols;
                out.push({t: b && b.dataset.tensor, cols: cols, cell: Math.round(w)});
            });
            P.setCursor(0);
            return out;
        }""")
        # A cell wider than ~90px means the block's width leaked into the track.
        check("每个格子的宽度是按字符数定的，不是被撑成整块宽度",
              all(t["cell"] <= 90 for t in tracks),
              json.dumps([t for t in tracks if t["cell"] > 90], ensure_ascii=False))

        # The occupancy bar itself must fit and its segments must tile the bar.
        bar = page.evaluate("""() => {
            const P = window.__p;
            let bad = [];
            for (let i = 1; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                const bar = document.querySelector('.lab-stage-occ-bar');
                if (!bar) continue;
                const bb = bar.getBoundingClientRect();
                const segs = [...bar.querySelectorAll('.lab-stage-occ-seg')];
                if (!segs.length) { bad.push({step: i, why: 'no segments'}); continue; }
                const total = segs.reduce((s, e) => s + e.getBoundingClientRect().width, 0);
                // Segments must not exceed the bar they sit in.
                if (total > bb.width + 1) {
                    bad.push({step: i, why: 'overflow', total, bar: bb.width});
                }
                // And each must have a visible width unless it is genuinely tiny.
                segs.forEach(s => {
                    const r = s.getBoundingClientRect();
                    if (r.height <= 0) bad.push({step: i, why: 'zero-height seg',
                                                 part: s.dataset.part});
                });
            }
            P.setCursor(0);
            return bad;
        }""")
        check("占用账的每一段都在条内、且都有可见高度",
              not bar, json.dumps(bar[:6], ensure_ascii=False))

        # The capacity marker's LABEL must be on screen, not just in the DOM.
        # This is the second time a `overflow:hidden` has eaten something here:
        # the label was positioned above the line inside the bar and clipped
        # away, leaving a bare red tick with no number -- present in innerText,
        # absent on screen, and invisible to every assertion that read the DOM.
        cap = page.evaluate("""() => {
            const P = window.__p;
            P.setCursor(4);
            const t = document.querySelector('.lab-stage-occ-captick');
            if (!t) return {found: false};
            // Scroll it into view first: `elementFromPoint` answers for the
            // VIEWPORT, and a point below the fold returns null -- which would
            // read as "clipped" for a label that is perfectly fine.
            t.scrollIntoView({block: 'center'});
            const r = t.getBoundingClientRect();
            const host = t.closest('.lab-stage-occ').getBoundingClientRect();
            const hit = document.elementFromPoint(r.left + r.width / 2,
                                                  r.top + r.height / 2);
            P.setCursor(0);
            return {found: true, w: Math.round(r.width), h: Math.round(r.height),
                    text: t.textContent,
                    inside: r.top >= host.top - 1 && r.bottom <= host.bottom + 1,
                    painted: !!hit && (hit === t || t.contains(hit) || hit.contains(t))};
        }""")
        check("容量刻度线带一个真正可见的 'M = …' 标签（不是被 overflow 裁掉）",
              cap.get("found") and cap["w"] > 10 and cap["h"] > 5 and
              cap["inside"] and cap["painted"],
              json.dumps(cap, ensure_ascii=False))
        check("标签写的是当前配置的 M",
              cap.get("text", "").strip() == f"M = {traces[default]['meta']['sram']['M']}",
              cap.get("text", ""))

        # ---- the layout must not MOVE as the replay runs -------------------
        #
        # A grid track sized by content makes a block's width a function of the
        # cursor: a matrix of zeros is narrow, the same matrix full of numbers is
        # three times as wide, and the layer reflows when the first write lands.
        # Nothing above notices -- every number is still correct -- and a reader
        # watching the stage sees the thing they were tracking jump. Measured
        # here as: a given tensor's box has the same size at every step where it
        # is resident.
        drift = page.evaluate("""() => {
            const P = window.__p, size = {};
            const out = [];
            for (let i = 0; i <= P.index.lastStep; i++) {
                P.setCursor(i);
                [...document.querySelectorAll('.lab-stage-block')].forEach(b => {
                    const r = b.querySelector('.lab-stage-grid');
                    if (!r) return;
                    const rb = r.getBoundingClientRect();
                    const key = b.dataset.tensor;
                    const w = Math.round(rb.width), h = Math.round(rb.height);
                    if (!size[key]) { size[key] = {w, h, step: i}; return; }
                    if (size[key].w !== w || size[key].h !== h) {
                        out.push({tensor: key, step: i, was: size[key], now: {w, h}});
                    }
                });
            }
            P.setCursor(0);
            return out;
        }""")
        check("同一个张量的方块在整个回放里尺寸不变（布局不会随数值长出来而重排）",
              not drift, json.dumps(drift[:6], ensure_ascii=False))

        # ==================================================== trace truth
        print("\n== trace 真实性 ==")
        for cfg_id in WALK_CONFIGS:
            t = traces[cfg_id]
            ex = t["meta"]["exposure"]
            io_ = t["meta"]["io"]
            ok = (ex["flash_forbidden_accesses"] == 0 and
                  ex["standard_forbidden_accesses"] > 0 and
                  io_["flash"]["elements"] == io_["predict"]["flash"] and
                  io_["standard"]["elements"] == io_["predict"]["standard"])
            if not ok:
                check(f"{cfg_id}：实测 IO 与公式预测一致、且 S/P 的 0 有对照组", False)
                break
        else:
            check(f"全部 {len(WALK_CONFIGS)} 个抽查配置：实测 == 预测，S/P 的 0 带对照组", True)
        ref = traces[default]["meta"]["reference"]
        note(ref)
        check("对拍结论写进了 meta.reference（可追溯到生成器）",
              "torch" in ref and "偏差" in ref and "0 次" in ref)
        check("轨迹带真实的逐帧 IO 计数（不是按公式填的）",
              all(traces[c]["steps"][i]["io"]["cum"] ==
                  sum(s["io"]["step"] for s in traces[c]["steps"][:i + 1])
                  for c in WALK_CONFIGS
                  for i in range(len(traces[c]["steps"]))))

        # ======================================================== shots
        print("\n== 截图 ==")
        # Back to the default configuration for the shots, so the filenames match
        # the default the page opens on.
        page.goto(url)
        page.wait_for_timeout(1500)
        page.evaluate("() => { window.__p.setCursor(0); window.scrollTo(0, 0); }")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "40-l06-overview.png")

        # The frame where S first appears in SRAM -- before it, the SRAM row is
        # honestly empty and the reader has nothing to look at yet.
        s_step = page.evaluate("""() => {
            const T = window.__labTrace;
            return T.steps.findIndex(s => s.id === 's00');
        }""")
        page.evaluate(f"() => window.__p.setCursor({s_step})")
        page.wait_for_timeout(320)
        page.locator(".lab-stage-panel").screenshot(path=SHOTS / "41-l06-stage-sram.png")
        page.locator('.lab-stage-layer[data-layer="SRAM"]').screenshot(
            path=SHOTS / "42-l06-sram-layer.png")

        # The write-back is where V1's defect is visible: O/m/l leaving SRAM for
        # HBM, which is the traffic V2 removes.
        wb_step = page.evaluate("""() => window.__labTrace.steps.findIndex(s => s.id === 'wb00')""")
        page.evaluate(f"() => window.__p.setCursor({wb_step})")
        page.wait_for_timeout(320)
        page.locator(".lab-stage-panel").screenshot(path=SHOTS / "43-l06-writeback.png")

        # The HBM row, at a step where S and P are both resident in SRAM -- the
        # shot that makes the absence in this row the point.
        page.locator('.lab-stage-layer[data-layer="HBM"]').screenshot(
            path=SHOTS / "44-l06-hbm-layer.png")

        # The correction strip is shot at the LAST step, not the first. Its
        # per-block fills only render for blocks the cursor has reached, so a
        # shot at step 0 shows four empty tracks -- a legal "not yet" state that
        # proves nothing about whether the fill works. At the end every block is
        # filled, and the identity vs rescale difference is visible as a shape.
        page.evaluate("() => window.__p.setCursor(window.__p.index.lastStep)")
        page.wait_for_timeout(320)
        page.locator(".lab-stage-panel").screenshot(path=SHOTS / "45-l06-final-step.png")
        for pid, name in (("io", "46-l06-io-counter"),
                          ("curves", "47-l06-io-curves"),
                          ("corr", "48-l06-correction")):
            el = page.locator(f'[data-panel="{pid}"]')
            if el.count():
                el.screenshot(path=SHOTS / f"{name}.png")

        # Dark mode, since the page is theme-aware and the DEFAULT is whatever
        # the reader's OS says. Explicitly dark rather than the ambient scheme:
        # the ambient one here is light, which made this shot a byte-identical
        # copy of the light one and proved nothing about theme support.
        page_dark = browser.new_page(viewport={"width": W, "height": H},
                                     color_scheme="dark")
        page_dark.goto(url)
        page_dark.wait_for_timeout(1400)
        page_dark.evaluate(f"() => window.__p.setCursor({s_step})")
        page_dark.wait_for_timeout(320)
        page_dark.locator(".lab-stage-panel").screenshot(
            path=SHOTS / "49-l06-stage-dark.png")
        dark_bg = page_dark.evaluate(
            "() => getComputedStyle(document.body).backgroundColor")
        check("暗色主题下页面确实换了配色（不是亮色的复制品）",
              "246, 247, 249" not in dark_bg and "249, 250, 251" not in dark_bg,
              f"body background = {dark_bg}")

        # And the occupancy bar must show its per-part colours, not one flat
        # block -- the defect this check exists for was invisible to every count.
        page.evaluate(f"() => window.__p.setCursor({s_step})")
        page.wait_for_timeout(220)
        palette = page.evaluate("""() => {
            const segs = [...document.querySelectorAll('.lab-stage-occ-seg')];
            return segs.map(s => getComputedStyle(s).backgroundColor);
        }""")
        check("占用账条的每一段按位置取色（不是同一个颜色平铺）",
              len(set(palette)) > 1, json.dumps(palette, ensure_ascii=False))

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


def trace_steps(traces, cfg_id):
    return len(traces[cfg_id]["steps"])


if __name__ == "__main__":
    sys.exit(main())
