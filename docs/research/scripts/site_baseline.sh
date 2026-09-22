#!/usr/bin/env bash
# R02 — Existing-site baseline: dist/ size, HTML page count and wire size, and
# what the `pagefind --site dist` build step costs.
#
# Assumes `npm ci` has already run. Runs the three build steps separately so
# each one's wall time is attributable.
#
# Usage:
#   npm ci && docs/research/scripts/site_baseline.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

ms() { echo $(( ($(date +%s%N) - $1) / 1000000 )); }

echo "=== node $(node -v) / npm $(npm -v) ==="

t=$(date +%s%N); npx astro check  >/tmp/r02-check.log  2>&1; echo "astro check : $(ms $t) ms"
t=$(date +%s%N); npx astro build  >/tmp/r02-build.log  2>&1; echo "astro build : $(ms $t) ms"
t=$(date +%s%N); npx pagefind --site dist >/tmp/r02-pf.log 2>&1; echo "pagefind    : $(ms $t) ms"

echo
echo "=== pagefind report ==="
grep -E 'Indexed [0-9]+ pages|Indexed [0-9]+ words|Finished' /tmp/r02-pf.log || true

echo
echo "=== dist/ ==="
echo "total bytes      : $(du -sb dist | cut -f1)"
echo "HTML page count  : $(find dist -name '*.html' | wc -l)"
echo "pagefind index   : $(du -sb dist/pagefind | cut -f1) bytes in $(find dist/pagefind -type f | wc -l) files"

echo
echo "=== all HTML: raw vs gzip ==="
python3 - <<'PY'
import glob, gzip
raw = gz = n = 0
biggest = (0, "")
for p in glob.glob('dist/**/*.html', recursive=True):
    b = open(p, 'rb').read()
    g = len(gzip.compress(b, 9))
    raw += len(b); gz += g; n += 1
    if g > biggest[0]:
        biggest = (g, p)
print(f"{n} pages: {raw/1e6:.1f} MB raw -> {gz/1e6:.1f} MB gzip")
print(f"average : {raw/n/1024:.0f} KB raw -> {gz/n/1024:.0f} KB gzip")
print(f"largest on the wire: {biggest[1]} at {biggest[0]/1024:.0f} KB gzip")
PY

echo
echo "=== ten largest HTML pages (raw / gzip) ==="
find dist -name '*.html' -printf '%s\t%p\n' | sort -rn | head -10 | \
while IFS=$'\t' read -r sz p; do
  g=$(gzip -9 -c "$p" | wc -c)
  printf '%9s %9s  %s\n' "$sz" "$g" "$(echo "$p" | sed 's|dist/||')"
done
