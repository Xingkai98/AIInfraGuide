#!/usr/bin/env python3
"""Drive prototype/index.html with real Chromium, capture screenshots + console.

PROTOTYPE harness for issue #12. Run: python3 prototype/shoot.py
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent.resolve()
URL = (HERE / "index.html").as_uri()
SHOTS = HERE / "shots"
SHOTS.mkdir(exist_ok=True)

W, H = 1600, 1000


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H})
        errors = []
        pg.on("console", lambda m: errors.append(f"[{m.type}] {m.text}")
              if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))

        pg.goto(URL)
        pg.wait_for_timeout(1200)

        # ---------- Q4: DAG layout, arc vs straight, sequence edges, at several sizes
        pg.screenshot(path=SHOTS / "01-first-step.png", full_page=True)
        pg.select_option("#inflate", "0")
        pg.click("#e-arc")
        pg.wait_for_timeout(250)
        pg.locator("#dagbox").screenshot(path=SHOTS / "02a-dag-10-arc.png")
        pg.click("#e-straight")
        pg.wait_for_timeout(250)
        pg.locator("#dagbox").screenshot(path=SHOTS / "02b-dag-10-straight.png")
        pg.click("#e-arc")
        pg.wait_for_timeout(200)
        pg.screenshot(path=SHOTS / "02-dag-arc.png", full_page=True)

        # the seq-edge + wrap comparison, same node count, three settings
        pg.select_option("#inflate", "8")
        pg.select_option("#wrap", "0")
        pg.evaluate("NG.seqEdges = false; state.layout = null; setCursor(0)")
        pg.wait_for_timeout(350)
        pg.locator("#dagbox").screenshot(path=SHOTS / "04a-dag-42-noseq-nosplit.png")
        pg.evaluate("NG.seqEdges = true; state.layout = null; setCursor(0)")
        pg.wait_for_timeout(350)
        pg.locator("#dagbox").screenshot(path=SHOTS / "04b-dag-42-seq-nosplit.png")
        pg.select_option("#wrap", "6")
        pg.wait_for_timeout(350)
        pg.locator("#dagbox").screenshot(path=SHOTS / "04c-dag-42-seq-wrap6.png")

        for val, name in (("0", "03-dag-10-seq-wrap4.png"), ("4", "05-dag-26-seq-wrap4.png"),
                          ("16", "06-dag-74-seq-wrap8.png")):
            pg.select_option("#inflate", val)
            pg.select_option("#wrap", "4" if val in ("0", "4") else "8")
            pg.evaluate("NG.seqEdges = true; state.layout = null; setCursor(0)")
            pg.wait_for_timeout(420)
            n = pg.evaluate("document.querySelectorAll('#dag .nd').length")
            print(f"  inflate={val}: {n} nodes drawn")
            pg.locator("#dagbox").screenshot(path=SHOTS / name)
        pg.select_option("#inflate", "0")
        pg.select_option("#wrap", "0")
        pg.evaluate("NG.seqEdges = false; state.layout = null; setCursor(0)")
        pg.wait_for_timeout(300)

        # ---------- Q3: formula tiers.
        # NOTE: because the player uses history.replaceState (per the D-42
        # decision), the query string is NOT reset between these calls -- so the
        # tier button must be re-clicked after every navigation, or the previous
        # tier (and a previous duplicate of this very screen) leaks through.
        # Step 6 = "m2", the m update; step 7 = "l2", where the correction factor
        # is actually non-zero (block 1's factor is 0 by construction).
        for fn, step, tier in (("07-formula-num", 6, "num"),
                               ("08-formula-idx", 6, "idx"),
                               ("09-formula-sym", 6, "sym")):
            pg.evaluate(f"setCursor({step})")
            pg.click(f"#tier button[data-t='{tier}']")
            pg.wait_for_timeout(280)
            pg.screenshot(path=SHOTS / f"{fn}.png", full_page=True)
        pg.click("#tier button[data-t='num']")
        pg.wait_for_timeout(200)

        # zoomed crops: these are the ones that actually show whether the KaTeX
        # slot substitution is correct
        pg.evaluate("setCursor(7)")          # l2: correction factor = 0.5655 (non-zero!)
        pg.click("#tier button[data-t='num']")
        pg.wait_for_timeout(300)
        pg.locator("#fxbox").screenshot(path=SHOTS / "16-crop-formula-region.png")
        pg.evaluate("setCursor(1)")          # the load step whose VALS slot leaked \left[
        pg.wait_for_timeout(280)
        pg.locator("#fxbox").screenshot(path=SHOTS / "17-crop-formula-vals.png")
        pg.evaluate("setCursor(2)")          # m update: the -inf -> 0.83 step
        pg.wait_for_timeout(250)
        pg.locator("#fxbox").screenshot(path=SHOTS / "18-crop-formula-m.png")
        pg.evaluate("setCursor(4)")          # a1: acc update, tensor panel shows the write
        pg.wait_for_timeout(250)
        pg.locator("#tensors").screenshot(path=SHOTS / "19-crop-tensors.png")

        # ---------- Q1: arbitrary jump, 5 -> 1 -> back
        pg.evaluate("setCursor(5)")   # ld2 (the comm node)
        pg.wait_for_timeout(150)
        pg.screenshot(path=SHOTS / "10-jump-step5.png", full_page=True)
        pg.evaluate("setCursor(1)")
        pg.wait_for_timeout(150)
        pg.screenshot(path=SHOTS / "11-jump-back-to-1.png", full_page=True)
        pg.evaluate("setCursor(9)")
        pg.wait_for_timeout(150)
        pg.screenshot(path=SHOTS / "12-jump-step9-end.png", full_page=True)
        # the "before" toggle: what the tensor panel shows pre-step
        pg.evaluate("setCursor(5)")
        pg.click("#when button[data-w='before']")
        pg.wait_for_timeout(200)
        pg.screenshot(path=SHOTS / "13-when-before.png", full_page=True)
        pg.click("#when button[data-w='after']")
        pg.wait_for_timeout(120)

        # ---------- Q2: lint output
        pg.click("#run")
        pg.wait_for_timeout(400)
        pg.screenshot(path=SHOTS / "14-lint.png", full_page=True)

        # ---------- scroll to bottom: self-check + perf
        pg.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        pg.wait_for_timeout(300)
        pg.screenshot(path=SHOTS / "15-selfcheck-perf.png", full_page=True)

        # ---------- data out
        data = pg.evaluate("""() => ({
          perf: window.__PERF,
          lint: (function(){
            const rows=[];
            document.querySelectorAll('#tests table tr').forEach(tr=>{
              const c=[...tr.children].map(x=>x.textContent);
              if(c.length) rows.push(c);
            });
            return rows;
          })(),
          dagStat: document.getElementById('dagstat').textContent,
          nodeCounts: {},
        })""")
        print("\n--- perf ---")
        print(json.dumps(data["perf"], indent=1))
        print("\n--- dagstat ---", data["dagStat"])

        # timing across node counts
        print("\n--- render timing vs node count ---")
        for val, label in (("0", "10 nodes"), ("2", "18 nodes"), ("4", "26 nodes"),
                           ("8", "42 nodes"), ("16", "74 nodes")):
            pg.select_option("#inflate", val)
            pg.wait_for_timeout(300)
            ms = pg.evaluate("""() => {
                const t=[];
                for(let k=0;k<40;k++){ const s=performance.now(); setCursor(k % 5); t.push(performance.now()-s); }
                t.sort((a,b)=>a-b);
                return {median:t[20], max:t[39]};
            }""")
            print(f"  {label:9s}  median {ms['median']:.2f} ms   max {ms['max']:.2f} ms")

        print("\n--- console errors/warnings ---")
        if errors:
            for e in errors[:25]:
                print("  " + e[:220])
        else:
            print("  (none)")

        b.close()
    print(f"\nscreenshots -> {SHOTS}")


if __name__ == "__main__":
    sys.exit(main())
