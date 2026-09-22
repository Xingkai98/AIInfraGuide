#!/usr/bin/env bash
# R02 — Wire size of plan A (shared assets) vs plan B (single file).
#
# Runs three variants so the comparison isolates *why* the sizes differ:
#
#   A        one HTML + shared engine.js / lab.css / katex.min.{js,css} / fonts
#   B-naive  single file, EVERYTHING base64'd (what "just inline it" means
#            literally, and what the design doc's 1.5 MB estimate assumed)
#   B-smart  single file, done the way a real inliner does it: JS and CSS as
#            text in <script>/<style>, base64 reserved for the woff2 fonts
#
# The gap between B-naive and B-smart is the whole argument: base64 wrecks
# gzip's view of already-compressible *text*, but costs almost nothing on
# already-compressed *binaries*.
#
# Everything is measured raw and gzip'd, because GitHub Pages serves gzip and
# only the gzip number is what a reader actually downloads.
set -euo pipefail

VERSION="${KATEX_VERSION:-0.16.45}"
PROBE="${TMPDIR:-/tmp}/katex-probe"
DIST="$PROBE/node_modules/katex/dist"
WORK="${TMPDIR:-/tmp}/labs-plan-compare"
ENGINE_BYTES="${ENGINE_BYTES:-46080}"   # 45 KB, the middle of the doc's 30-60 KB
LABEL="${LABEL:-L06-flash-attention}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -d "$DIST" ] || { echo "run measure_katex.sh first (no $DIST)" >&2; exit 1; }
rm -rf "$WORK"; mkdir -p "$WORK"

gz()  { gzip -9 -c "$1" | wc -c | tr -d ' '; }
raw() { stat -c%s "$1"; }
kb()  { awk -v b="$1" 'BEGIN { printf "%7.1f", b/1024 }'; }

python3 "$HERE/gen_trace.py" --outdir "$WORK" --write >/dev/null
TRACE="$WORK/$LABEL.trace.json"

# ---- synthesise an engine of a realistic size and mix ------------------------
python3 - "$WORK/engine.js" "$ENGINE_BYTES" <<'PY'
import sys, random
out, n = sys.argv[1], int(sys.argv[2])
random.seed(7)
words = ["const","let","function","return","this","step","state","nodes","edges",
         "tensor","shape","values","bindings","formula","render","svg","path",
         "class","if","else","for","of","in","new","Map","Set","Array","Object"]
lines, size = [], 0
while size < n:
    p = random.random()
    if p < 0.45:
        ln = f"  {random.choice(words)} {random.choice(words)}{random.randint(0,999)} = {random.choice(words)}.{random.choice(words)}({random.randint(0,99)});"
    elif p < 0.7:
        ln = f"  // {random.choice(words)} {random.choice(words)} {random.randint(0,999)}"
    elif p < 0.9:
        ln = f"  const {random.choice(words)}{random.randint(0,99)} = {{ {random.choice(words)}: {random.randint(0,999)}, {random.choice(words)}: \"{random.choice(words)}\" }};"
    else:
        ln = ""
    lines.append(ln); size += len(ln) + 1
open(out, "w").write("\n".join(lines))
PY

MINIMAL_FONTS="KaTeX_Main-Regular KaTeX_Math-Italic KaTeX_Size1-Regular KaTeX_Size2-Regular KaTeX_Size3-Regular KaTeX_Size4-Regular"
COMMON_FONTS="$MINIMAL_FONTS KaTeX_Main-Bold KaTeX_Main-Italic KaTeX_Main-BoldItalic KaTeX_Math-BoldItalic KaTeX_AMS-Regular KaTeX_Caligraphic-Regular"

BODY_HTML='<header><h1>FlashAttention V1</h1><p>外循环 i 遍历 Q 块，内循环 j 遍历 K/V 块，每轮载入 K_j, V_j 后算 S_ij = Q_i K_j^T / sqrt(d)，行最大值、P_ij = exp(S_ij - m_new)，再用修正因子更新 l_i 与 O_i。S 和 P 从不落回 HBM，这是 FlashAttention 全部 IO 优势的来源。</p></header>
<main><div id="dag"></div><div id="tensors"></div><div id="formula"></div><div id="player"></div></main>'

# ---------------------------------------------------------------------------
# Plan A — shared assets
# ---------------------------------------------------------------------------
mkdir -p "$WORK/planA/fonts"
cp "$WORK/engine.js" "$WORK/planA/engine.js"
cp "$DIST/katex.min.js" "$DIST/katex.min.css" "$WORK/planA/"
for f in $COMMON_FONTS; do cp "$DIST/fonts/$f.woff2" "$WORK/planA/fonts/"; done
cp "$TRACE" "$WORK/planA/"

python3 - "$WORK/planA/index.html" "$TRACE" "$BODY_HTML" <<'PY'
import sys
out, trace, body = sys.argv[1:4]
blob = open(trace, encoding="utf-8").read()
open(out, "w", encoding="utf-8").write(f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L06 FlashAttention V1 · 交互实验室</title>
<link rel="stylesheet" href="katex.min.css"><link rel="stylesheet" href="lab.css">
</head><body>
{body}
<script id="trace" type="application/json">{blob}</script>
<script src="katex.min.js"></script><script src="engine.js"></script>
</body></html>
""")
PY

# ---------------------------------------------------------------------------
# Plan B — single file, two ways of inlining
# ---------------------------------------------------------------------------
mkdir -p "$WORK/planB-naive" "$WORK/planB-smart"

# naive: base64 everything, including the JS
python3 - "$WORK/planB-naive/index.html" "$WORK/engine.js" "$DIST" "$TRACE" "$BODY_HTML" $COMMON_FONTS <<'PY'
import base64, sys
out, engine, dist, trace, body = sys.argv[1:6]
fonts = sys.argv[6:]
blob = open(trace, encoding="utf-8").read()
css = open(f"{dist}/katex.min.css", encoding="utf-8").read()
for name in fonts:
    data = base64.b64encode(open(f"{dist}/fonts/{name}.woff2", "rb").read()).decode()
    css = css.replace(f"fonts/{name}.woff2", f"data:font/woff2;base64,{data}")
def b64file(p): return base64.b64encode(open(p, "rb").read()).decode()
open(out, "w", encoding="utf-8").write(f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L06 FlashAttention V1 · 交互实验室</title><style>{css}</style></head><body>
{body}
<script id="trace" type="application/json">{blob}</script>
<script>{b64file(f"{dist}/katex.min.js")}</script>
<script>{b64file(engine)}</script>
</body></html>
""")
PY

# smart: fonts base64'd (binaries), JS and CSS as text (where gzip still works)
python3 - "$WORK/planB-smart/index.html" "$WORK/engine.js" "$DIST" "$TRACE" "$BODY_HTML" $COMMON_FONTS <<'PY'
import base64, sys
out, engine, dist, trace, body = sys.argv[1:6]
fonts = sys.argv[6:]
blob = open(trace, encoding="utf-8").read()
css = open(f"{dist}/katex.min.css", encoding="utf-8").read()
for name in fonts:
    data = base64.b64encode(open(f"{dist}/fonts/{name}.woff2", "rb").read()).decode()
    css = css.replace(f"fonts/{name}.woff2", f"data:font/woff2;base64,{data}")
js = open(f"{dist}/katex.min.js", encoding="utf-8").read()
eng = open(engine, encoding="utf-8").read()
open(out, "w", encoding="utf-8").write(f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L06 FlashAttention V1 · 交互实验室</title><style>{css}</style></head><body>
{body}
<script id="trace" type="application/json">{blob}</script>
<script>{js}</script>
<script>{eng}</script>
</body></html>
""")
PY

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
report() {
  local label="$1" path="$2"
  local r g
  r=$(raw "$path"); g=$(gz "$path")
  printf '%-34s %10s %10s\n' "$label" "$r" "$g"
}

echo "========================================================================"
echo "PLAN A — shared assets (first visit downloads everything)"
echo "========================================================================"
printf '%-34s %10s %10s\n' "file" "raw" "gzip"
printf '%-34s %10s %10s\n' "----" "---" "----"
a_files=0; a_raw=0; a_gz=0
for f in index.html katex.min.js katex.min.css engine.js "$LABEL.trace.json"; do
  r=$(raw "$WORK/planA/$f"); g=$(gz "$WORK/planA/$f")
  printf '%-34s %10s %10s\n' "$f" "$r" "$g"
  a_files=$((a_files+1)); a_raw=$((a_raw+r)); a_gz=$((a_gz+g))
done
f_raw=0; f_n=0
for p in "$WORK/planA/fonts"/*.woff2; do f_raw=$((f_raw+$(raw "$p"))); f_n=$((f_n+1)); done
printf '%-34s %10s %10s   (%d files; gzip ~= raw)\n' "fonts/*.woff2" "$f_raw" "$f_raw" "$f_n"
a_raw=$((a_raw+f_raw)); a_gz=$((a_gz+f_raw))
echo "------------------------------------------------------------------------"
printf '%-34s %10s %10s\n' "PLAN A COLD" "$a_raw" "$a_gz"
printf '%-34s %10s %10s\n' "  (KB)" "$(kb $a_raw)K" "$(kb $a_gz)K"
printf '%-34s %s requests\n\n' "  HTTP requests" "$((a_files+f_n))"

echo "========================================================================"
echo "PLAN B — single self-contained HTML"
echo "========================================================================"
printf '%-34s %10s %10s\n' "file" "raw" "gzip"
b_raw=0; b_gz=0
for v in naive smart; do
  p="$WORK/planB-$v/index.html"
  r=$(raw "$p"); g=$(gz "$p")
  printf '%-34s %10s %10s\n' "B-$v (1 file, 1 request)" "$r" "$g"
done
b_raw=$(raw "$WORK/planB-smart/index.html"); b_gz=$(gz "$WORK/planB-smart/index.html")
n_raw=$(raw "$WORK/planB-naive/index.html"); n_gz=$(gz "$WORK/planB-naive/index.html")
echo "------------------------------------------------------------------------"
printf '%-34s %10s %10s\n' "PLAN B-SMART (recommended)" "$b_raw" "$b_gz"
printf '%-34s %10s %10s\n' "  (KB)" "$(kb $b_raw)K" "$(kb $b_gz)K"
printf '%-34s %s request\n\n' "  HTTP requests" "1"

echo "========================================================================"
echo "HEADLINE"
echo "========================================================================"
awk -v ac="$a_gz" -v an="$a_raw" -v bs="$b_gz" -v bn="$b_raw" -v bn2="$n_gz" 'BEGIN {
  printf "  plan A cold      : %7.0f KB gzip (%7.0f KB raw)\n", ac/1024, an/1024;
  printf "  plan B smart     : %7.0f KB gzip (%7.0f KB raw)   %+.1f%% vs A cold\n", bs/1024, bn/1024, 100*(bs-ac)/ac;
  printf "  plan B naive     : %7.0f KB gzip                     %+.1f%% vs A cold\n", bn2/1024, 100*(bn2-ac)/ac;
  printf "\n  -> correct inlining makes single-file cost about the same as plan A FIRST page\n";
  printf "     naive base64-everything inflates the page by %.0f%%\n", 100*(bn2-ac)/ac;
}'
echo
echo "artifacts under $WORK"
