#!/usr/bin/env bash
# R02 — Measure KaTeX raw / gzip / font overhead.
#
# Usage:
#   docs/research/scripts/measure_katex.sh [katex_version]
#
# Installs KaTeX into a throwaway dir under $TMPDIR (never touching the repo's
# package.json / package-lock.json) and prints a table of raw and gzip sizes.
# Default version is pinned to whatever the repo's lockfile already resolves
# transitively, so the numbers match what a vendored copy would ship.
set -euo pipefail

VERSION="${1:-0.16.45}"
PROBE="${TMPDIR:-/tmp}/katex-probe"
DIST="$PROBE/node_modules/katex/dist"

# gzip a file (or stdin) at max compression and print the byte count.
gz() { gzip -9 -c "$1" | wc -c | tr -d ' '; }
raw() { stat -c%s "$1"; }
kb() { awk -v b="$1" 'BEGIN { printf "%.1f", b/1024 }'; }

mkdir -p "$PROBE"
cd "$PROBE"
[ -f package.json ] || npm init -y >/dev/null
npm install --silent "katex@${VERSION}" >/dev/null

echo "KaTeX version: $(node -p "require('$PROBE/node_modules/katex/package.json').version")"
echo

echo "=== Core assets ==="
printf '%-22s %10s %10s\n' file raw gzip
for f in katex.min.js katex.min.css; do
  printf '%-22s %10s %10s\n' "$f" "$(raw "$DIST/$f")" "$(gz "$DIST/$f")"
done
echo

echo "=== Fonts referenced by katex.min.css ==="
# The minified CSS references 20 distinct woff2 files; count them from the CSS
# itself so the number is derived, not hard-coded.
grep -o 'fonts/[A-Za-z0-9_.-]*\.woff2' "$DIST/katex.min.css" | sort -u > /tmp/katex-fonts-used.txt
NUSED=$(wc -l < /tmp/katex-fonts-used.txt)
ALL_COUNT=$(ls "$DIST"/fonts/*.woff2 | wc -l)
echo "distinct woff2 referenced by CSS: $NUSED  (files on disk: $ALL_COUNT)"

total_all=0
while read -r p; do total_all=$((total_all + $(raw "$DIST/$p"))); done < /tmp/katex-fonts-used.txt
echo "all-woff2 raw total:       $total_all bytes  ($(kb "$total_all") KB)"
printf 'all-woff2 gzip total:      %s bytes  (%s KB)  [gzip cannot shrink pre-compressed woff2]\n' \
  "$(tar cf - -C "$DIST" $(cat /tmp/katex-fonts-used.txt) | gzip -9 -c | wc -c | tr -d ' ')" "0.0"

# Subsets. "minimal" = what a plain formula (letters, digits, +−=, ∑, √, parens)
# actually touches; "common" adds bold/italic/caligraphic/AMS for richer content.
MINIMAL="KaTeX_Main-Regular KaTeX_Math-Italic KaTeX_Size1-Regular KaTeX_Size2-Regular KaTeX_Size3-Regular KaTeX_Size4-Regular"
COMMON="KaTeX_Main-Regular KaTeX_Main-Bold KaTeX_Main-Italic KaTeX_Main-BoldItalic KaTeX_Math-Italic KaTeX_Math-BoldItalic KaTeX_Size1-Regular KaTeX_Size2-Regular KaTeX_Size3-Regular KaTeX_Size4-Regular KaTeX_AMS-Regular KaTeX_Caligraphic-Regular"

for set in MINIMAL COMMON; do
  eval "members=\$$set"
  tot=0
  for n in $members; do tot=$((tot + $(raw "$DIST/fonts/$n.woff2"))); done
  printf '%-8s subset: %2d fonts, %8d bytes (%s KB)\n' "$set" \
    "$(echo $members | wc -w)" "$tot" "$(kb $tot)"
done
echo

echo "=== base64 inflation (inlining into a single HTML) ==="
echo "base64 is 4/3 of the input, plus the data: URI wrapper."
for f in katex.min.js katex.min.css; do
  r=$(raw "$DIST/$f"); g=$(gz "$DIST/$f")
  printf '%-16s raw %8d -> base64 %8d  (%.2fx)\n' "$f" "$r" "$(( (r+2)/3*4 ))" \
    "$(awk -v a=$(( (r+2)/3*4 )) -v b=$r 'BEGIN{printf "%.3f", a/b}')"
  printf '%-16s gz  %8d -> base64 %8d  (%.2fx)\n' "$f" "$g" "$(( (g+2)/3*4 ))" \
    "$(awk -v a=$(( (g+2)/3*4 )) -v b=$g 'BEGIN{printf "%.3f", a/b}')"
done
