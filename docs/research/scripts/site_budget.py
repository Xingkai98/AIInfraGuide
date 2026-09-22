#!/usr/bin/env python3
"""R02 — Turn the raw measurements into the two numbers the decision needs:

  * cold cost   — bytes on the wire for the FIRST lab page a visitor opens
  * warm cost   — bytes for each SUBSEQUENT lab page, once shared assets are cached

Plan A wins only if "warm" is much cheaper than "cold". Plan B has no warm case:
every lab page re-ships the whole engine + KaTeX.

Reads the trace JSONs and the KaTeX dist directory, so every figure traces back
to a measurement rather than an estimate.

Usage:
    python3 docs/research/scripts/site_budget.py --katex-dist /tmp/katex-probe/node_modules/katex/dist \
        --trace-dir /tmp/labs-measure --engine-bytes 46080
"""
from __future__ import annotations

import argparse
import base64
import gzip
import os

FONT_TIERS = {
    "minimal": ["KaTeX_Main-Regular", "KaTeX_Math-Italic", "KaTeX_Size1-Regular",
                "KaTeX_Size2-Regular", "KaTeX_Size3-Regular", "KaTeX_Size4-Regular"],
    "common": ["KaTeX_Main-Regular", "KaTeX_Math-Italic", "KaTeX_Size1-Regular",
               "KaTeX_Size2-Regular", "KaTeX_Size3-Regular", "KaTeX_Size4-Regular",
               "KaTeX_Main-Bold", "KaTeX_Main-Italic", "KaTeX_Main-BoldItalic",
               "KaTeX_Math-BoldItalic", "KaTeX_AMS-Regular", "KaTeX_Caligraphic-Regular"],
}
ALL_FONTS = None  # every woff2 in dist/fonts


def gz(b: bytes) -> int:
    return len(gzip.compress(b, 9))


def read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def b64(n: int) -> int:
    """Encoded length of n raw bytes as base64 (with padding)."""
    return (n + 2) // 3 * 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--katex-dist", required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--engine-bytes", type=int, default=46080)
    ap.add_argument("--labs", type=int, default=21)
    ap.add_argument("--shell-file", default="/tmp/labs-plan-compare/planA/index.html",
                    help="real lab HTML with the trace removed, for the shell size")
    ap.add_argument("--shell-dir", default="/tmp/labs-plan-compare",
                    help="dir measure_page.sh wrote planA/ and planB-*/ into")
    args = ap.parse_args()

    d = args.katex_dist
    js_raw = read(os.path.join(d, "katex.min.js"))
    css_raw = read(os.path.join(d, "katex.min.css"))
    all_woff2 = sorted(f for f in os.listdir(os.path.join(d, "fonts"))
                       if f.endswith(".woff2"))

    def font_bytes(tier):
        names = ALL_FONTS if tier is None else FONT_TIERS[tier]
        if names is None:
            names = [f[:-6] for f in all_woff2]
        return sum(os.path.getsize(os.path.join(d, "fonts", f"{n}.woff2"))
                   for n in names), len(names)

    # Prefer the engine measure_page.sh actually built, so the two scripts
    # report the same page. Fall back to a synthetic one of --engine-bytes.
    engine_path = os.path.join(args.shell_dir, "planA", "engine.js")
    if os.path.exists(engine_path):
        engine_raw = read(engine_path)
    else:
        engine_line = b"  const step%d = { state: %d, nodes: [%d, %d], edges: [] };\n"
        chunks, i = [], 0
        while sum(map(len, chunks)) < args.engine_bytes:
            chunks.append(engine_line % (i, i, i, i + 1))
            i += 1
        engine_raw = b"".join(chunks)[:args.engine_bytes]

    trace_sig6 = read(os.path.join(args.trace_dir, "L06-flash-attention.trace.json"))

    # The lab page's HTML shell — everything that is neither trace nor engine.
    # Default is the measured plan-A page with the trace blob cut out; fall back
    # to a synthetic fragment when measure_page.sh has not been run.
    if args.shell_file and os.path.exists(args.shell_file):
        shell = read(args.shell_file)
        shell = shell.replace(trace_sig6, b"")
    else:
        shell = (b'<header><h1>FlashAttention V1</h1></header>\n<main>'
                 b'<div id="dag"></div><div id="tensors"></div></main>\n') * 20
    SHELL = len(shell)

    print("=" * 74)
    print("PER-PAGE COLD COST, by font tier")
    print("=" * 74)
    hdr = f"{'component':<26} {'raw':>10} {'gzip(A)':>10} {'b64(B)':>10} {'gzip(B)':>10}"
    for tier in ("minimal", "common", "all"):
        fb, fn = font_bytes(None if tier == "all" else tier)
        print(f"\n--- font tier: {tier} ({fn} woff2) ---")
        print(hdr)
        print("-" * 74)

        rows = []

        def row(label, raw: bytes, as_text: bytes, as_b64: bytes):
            """Record one component three ways.

            raw     — the bytes as a standalone file on disk
            as_text — inlined directly into the HTML (JS in <script>, CSS in
                      <style>, JSON in <script type=application/json>)
            as_b64  — base64'd into a data: URI (how binary fonts must go in)
            """
            rows.append((label, len(raw), gz(raw), len(as_text), gz(as_text),
                         len(as_b64), gz(as_b64)))

        tier_names = (ALL_FONTS if tier == "all" else FONT_TIERS[tier])
        if tier_names is None:
            tier_names = [f[:-6] for f in all_woff2]
        font_blob = b"".join(read(os.path.join(d, "fonts", f"{n}.woff2"))
                             for n in tier_names)
        row("engine.js", engine_raw, engine_raw, base64.b64encode(engine_raw))
        row("katex.min.js", js_raw, js_raw, base64.b64encode(js_raw))
        row("katex.min.css", css_raw, css_raw, base64.b64encode(css_raw))
        row("trace.json", trace_sig6, trace_sig6, trace_sig6)
        row("html shell", shell, shell, shell)
        row("fonts (woff2)", font_blob, base64.b64encode(font_blob),
            base64.b64encode(font_blob))

        print(f"{'component':<26} {'raw':>9} {'gzip':>9} | {'txt+gz':>9} | {'b64+gz':>9}")
        for label, r, g, t, tg, b, gb in rows:
            print(f"{label:<26} {r:>9} {g:>9} | {tg:>9} | {gb:>9}")

        a_cold = sum(g for _, _, g, _, _, _, _ in rows)
        # B-smart: text where gzip still works (JS/CSS/JSON/HTML), base64 only
        # for the fonts, which are already compressed and so lose nothing.
        b_smart = sum(tg for _, _, _, _, tg, _, _ in rows)
        b_naive = sum(gb for _, _, _, _, _, _, gb in rows)
        print("-" * 74)
        print(f"{'PLAN A cold (gzip)':<26} {'':>9} {a_cold:>9}   "
              f"({a_cold / 1024:.0f} KB, {len(rows) + fn - 1} requests)")
        print(f"{'PLAN B-smart single file':<26} {'':>9} {'':>9} | {b_smart:>9} | "
              f"({b_smart / 1024:.0f} KB, 1 request)")
        print(f"{'PLAN B-naive (all b64)':<26} {'':>9} {'':>9} | {'':>9} | {b_naive:>9} "
              f"({b_naive / 1024:.0f} KB, 1 request)")
        b_cold = b_smart

        # Warm: shell + trace are all that change; the rest is cached and
        # revalidated with a 304 (~600 bytes of request/response headers).
        a_warm = gz(shell) + gz(trace_sig6) + 600
        print(f"{'PLAN A warm (2nd+ lab)':<26} {'':>10} {a_warm:>10}   "
              f"({a_warm / 1024:.1f} KB)")
        print(f"{'PLAN B 2nd+ lab':<26} {'':>10} {'':>10} {b_cold:>10}   "
              f"({b_cold / 1024:.0f} KB — no caching possible)")

        n = args.labs
        print(f"\n  over {n} lab pages:")
        print(f"    plan A cold+warm : {a_cold + (n - 1) * a_warm:>9} bytes "
              f"({(a_cold + (n - 1) * a_warm) / 1024:.0f} KB)")
        print(f"    plan B (all cold): {n * b_cold:>9} bytes "
              f"({n * b_cold / 1024:.0f} KB, {n * b_cold / (a_cold + (n - 1) * a_warm):.1f}x)")

    print()
    print("=" * 74)
    print("base64 gzip tax, measured")
    print("=" * 74)
    for name, blob in (("katex.min.js", js_raw), ("katex.min.css", css_raw)):
        b = base64.b64encode(blob)
        print(f"{name:<16} file+gzip {gz(blob):>8}   b64+gzip {gz(b):>8}   "
              f"cost {gz(b) - gz(blob):>7} bytes ({gz(b) / gz(blob):.2f}x)")
    fblob = b"".join(read(os.path.join(d, "fonts", f)) for f in all_woff2)
    print(f"{'all woff2':<16} file+gzip {gz(fblob):>8}   b64+gzip {gz(base64.b64encode(fblob)):>8}   "
          f"cost {gz(base64.b64encode(fblob)) - gz(fblob):>7} bytes "
          f"({gz(base64.b64encode(fblob)) / gz(fblob):.2f}x)")


if __name__ == "__main__":
    main()
