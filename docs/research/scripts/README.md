# R02 measurement scripts

Reproduce every number in [`../size-budget.md`](../size-budget.md).

```bash
# 1. KaTeX overhead (installs into a throwaway dir; never touches this repo's
#    package.json / package-lock.json)
docs/research/scripts/measure_katex.sh                       # default 0.16.45

# 2. Simulated traces -> bytes
docs/research/scripts/gen_trace.py --outdir /tmp/labs-measure --write

# 3. Plan A vs plan B page weight
docs/research/scripts/measure_page.sh                        # writes /tmp/labs-plan-compare

# 4. Cold/warm budget tables for 1..21 lab pages
docs/research/scripts/site_budget.py \
  --katex-dist /tmp/katex-probe/node_modules/katex/dist \
  --trace-dir /tmp/labs-measure

# 5. Which font files a browser actually fetches (needs playwright + chromium)
docs/research/scripts/font_probe.py /tmp/fontprobe/wide.html

# 6. Existing site baseline (needs `npm ci` first)
npm ci && docs/research/scripts/site_baseline.sh
```

All scripts are read-only with respect to the repo. The only side effects are
in `$TMPDIR`: `/tmp/katex-probe` (the KaTeX install), `/tmp/labs-measure` (the
generated traces and report JSON), and `/tmp/labs-plan-compare` (the assembled
plan A / plan B pages).

## A note on `gen_trace.py`

The traces are **shape-faithful simulations**, not recordings of a real
PyTorch run. The recurrences are genuine — the online-softmax trace is asserted
against `numpy` to 1e-9, and the FlashAttention trace is asserted to be exactly
softmax — but the inputs come from a seeded RNG rather than a model. That is
sufficient for R02, whose subject is wire size, not numerical fidelity. When
the real `labs/traces/*.py` generators exist, re-run the measurement against
their output; the schema is meant to match.
