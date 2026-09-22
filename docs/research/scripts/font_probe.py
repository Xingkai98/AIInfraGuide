#!/usr/bin/env python3
"""R02 — Which KaTeX font files does a browser ACTUALLY download?

The CSS-rule analysis over-counts: katex.min.css declares 20 @font-face rules,
but a browser only fetches a font file when a rendered glyph needs that family.
So instead of reasoning about the CSS, load a real lab page in Chromium and
count the requests.

This is the number that decides the font tier — if a page of real lab formulas
pulls 3 fonts rather than 12, the "which subset do we vendor" question mostly
answers itself, and plan B's weight drops accordingly.

Usage:
    python3 docs/research/scripts/font_probe.py /tmp/labs-plan-compare/planA/index.html
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter


async def probe(page_path: str) -> None:
    from playwright.async_api import async_playwright

    served = os.path.dirname(os.path.abspath(page_path))
    url = "file://" + os.path.abspath(page_path)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        ctx = await browser.new_context()
        page = await ctx.new_page()

        requests: list[tuple[str, int]] = []
        page.on("response", lambda r: requests.append((r.url, r.headers.get("content-length", "?"))))

        await page.goto(url, wait_until="networkidle")
        # Rendered math may arrive after load (the probe page fetches its
        # formula list); wait for it, then force a layout so the font loads
        # that KaTeX's CSS triggers actually fire before we read the log.
        await page.wait_for_function(
            "() => document.querySelectorAll('.katex').length > 0", timeout=15000)
        await page.evaluate("() => document.fonts.ready")
        await page.evaluate("() => document.body.offsetHeight")
        await page.wait_for_timeout(1500)
        rendered = await page.evaluate("() => document.querySelectorAll('.katex').length")
        print(f"formulas rendered        : {rendered}")

        fonts = [u for u, _ in requests if u.endswith(".woff2")]
        print(f"page: {page_path}")
        print(f"total network responses: {len(requests)}")
        print(f"woff2 font files fetched : {len(fonts)}")
        total = 0
        for f in sorted(fonts):
            size = os.path.getsize(os.path.join(served, "fonts", os.path.basename(f)))
            total += size
            print(f"   {os.path.basename(f):<34} {size:>8} bytes")
        print(f"   {'TOTAL':<34} {total:>8} bytes ({total / 1024:.1f} KB)")
        by_kind = Counter(os.path.splitext(u)[1] or "(none)" for u, _ in requests)
        print("by type:", dict(by_kind))

        await browser.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else \
        "/tmp/labs-plan-compare/planA/index.html"
    asyncio.run(probe(target))
