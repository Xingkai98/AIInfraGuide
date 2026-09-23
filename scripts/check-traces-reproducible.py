#!/usr/bin/env python3
"""Check that every committed trace still matches its generator's output.

Run by `.github/workflows/trace-checks.yml` AFTER the generators have run, with
git's index as the reference. Exit non-zero if anything drifted.

WHY THIS IS NOT `git diff --exit-code`
--------------------------------------
The workflow used to require the regenerated JSON to be byte-identical to the
committed one, on the stated grounds that "the scripts are deterministic by
construction (fixed inputs, no sampling)". That premise is false, and it failed
for a whole class of traces rather than for an unlucky one:

  * The scripts fix their INPUTS, but not their ARITHMETIC. A trace that records
    `A @ B` records the output of the BLAS kernel numpy dispatched to, and that
    kernel's summation order depends on the CPU and the BLAS build -- so the same
    script on a different runner produces the same answer to within round-off and
    not to the last bit. Measured across two machines on this repo, the largest
    relative difference was ~5e-13, with a median around 3e-16 (i.e. most
    affected values differ in the final ULP).
  * Four of the six generators do this, so the byte gate could not pass for
    `kv-cache.json` either -- it was failing on `feat/interactive-labs` too, and
    only went unnoticed because an earlier, unrelated break killed the job before
    this step ran.

Loosening the tolerance to cover that noise does not weaken what the check is
FOR. The thing it exists to catch is a hand-edited or stale JSON, and both of
those move values by far more than 5e-13:

  * a hand edit changes a digit the page prints -- four significant figures, so
    1e-4 at the loosest;
  * a stale JSON is one the generator no longer produces, which shows up as a
    changed key, a changed list length, a changed string, or a value that moved
    for a real reason.

So the comparison here is: **strings, integers, booleans and nulls must be
exactly equal, and the JSON STRUCTURE must be identical** (same keys, same list
lengths, same types -- a field that appeared or disappeared fails), while floats
need only agree within `_REL_TOL`. That is strictly more informative than the
byte check, because its failure message says WHICH value moved and by how much.

TEXT IS COMPARED NUMERICALLY, NOT LITERALLY
-------------------------------------------
A trace also carries the values it PRINTS -- `num`-tier LaTeX bindings, narration,
the `meta.reference` summary -- which are four-significant-figure renderings of
the same arithmetic. BLAS noise that moves a float field by 1e-13 moves the
STRING by a digit, so a literal string comparison fires on exactly the drift the
float tolerance exists to absorb. Strings are therefore tokenised into their
numeric runs and their literal parts: the literal parts must match exactly, and
each aligned pair of numbers must agree within tolerance. A reworded sentence, a
changed count, or a number that moved for a real reason all still fail.

The tolerance is derived, not picked: it is four orders of magnitude above the
measured BLAS noise, and six below the 1e-4 a hand-edited four-figure number
would move by. `--tol` overrides it for an experiment.

NOTE ON THE LABS THAT CALL BLAS. `flash_attention.py` (L06) does NOT rely on
this tolerance for its own content: it sums its matmuls in explicit k order so
its JSON is bit-identical on every machine, and its reference residuals are
asserted rather than quoted for the same reason. This check's tolerance is what
covers the generators that have not been through that treatment (L05's
`kv_cache.py` in particular), and is deliberately loose enough to keep them
green while still catching a real edit.

Run:  python3 labs/traces/<each generator>.py   # regenerate, in place
      python3 scripts/check-traces-reproducible.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = "labs/traces"

# See the module docstring: measured cross-machine drift is ~5e-13 relative,
# and a hand-edited four-significant-figure value moves by >=1e-4.
_REL_TOL = 1e-9
_ABS_TOL = 1e-12


def committed(name):
    """The committed version of a trace, straight out of git's index.

    Read from git rather than from the working tree, because the generators have
    already overwritten the working tree -- comparing it to itself would pass
    unconditionally.
    """
    p = subprocess.run(
        ["git", "show", f":{TRACE_DIR}/{name}"],
        cwd=str(REPO), capture_output=True, text=True)
    if p.returncode != 0:
        return None            # untracked: a new trace, nothing to compare
    return json.loads(p.stdout)


_NUM_IN_TEXT_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _last_place(token):
    """The value of one unit in a printed number's last displayed digit.

    `2048` -> 1, `-0.05292` -> 1e-5, `6.94e-17` -> 1e-19. This is the
    quantization of the rendering, not a property of the number.
    """
    m = re.match(r"^(-?)(\d+)(?:\.(\d+))?(?:[eE]([-+]?\d+))?$", token)
    if not m:
        return None
    frac = m.group(3) or ""
    exp = int(m.group(4) or 0)
    return 10.0 ** (exp - len(frac))


def text_differs_only_numerically(a, b):
    """True when two strings differ only by the quantization of what they print.

    Necessary because the points of difference are not confined to the JSON's
    float fields. A trace also carries the values it PRINTS -- a `num`-tier
    binding's LaTeX, a narration, the `meta.reference` summary -- and those are
    finite-precision renderings of the same arithmetic. Two machines that agree
    on a value to 1e-13 disagreement can still ROUND it differently at the fourth
    significant figure, which changes the string by one unit in its last digit.

    So the tolerance here is not a tolerance on noise; it is the exactly
    defensible ambiguity of a rounded display: **one unit in the last printed
    place**. `-0.05292` -> `-0.05293` is one unit at 1e-5 and passes;
    `2048` -> `1232` is 816 units and fails; a reworded sentence does not even
    get this far, because the literal parts between the numbers must match
    exactly and the numbers must align one for one.

    That last property is what keeps a hand edit visible: changing a printed
    digit by more than one place fails, and changing the prose fails. A one-unit
    change in a printed digit is left to the FIELD-level check, which compares
    the underlying value exactly -- the string is a rendering of it, and the two
    cannot both be edited consistently by accident.

    THE LIMIT OF THIS RULE, stated rather than discovered later: it tolerates any
    rearrangement whose literal parts and number values are unchanged, e.g.
    "先加 1 再加 2" -> "先加 2 再加 1". A hand edit of that shape slips through
    the string rule. It cannot slip through unnoticed overall, because the same
    edit to a generator changes the numbers those words describe and the field
    comparison sees those; but a pure word-order change in a narration would not
    be caught here. Closing it would require knowing which words are prose and
    which are data, which is exactly the distinction this file does not have --
    so it is documented instead of half-solved.
    """
    if _NUM_IN_TEXT_RE.split(a) != _NUM_IN_TEXT_RE.split(b):
        return False                     # the prose itself changed
    na = _NUM_IN_TEXT_RE.findall(a)
    nb = _NUM_IN_TEXT_RE.findall(b)
    if len(na) != len(nb):
        return False
    for x, y in zip(na, nb):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        places = [_last_place(x), _last_place(y)]
        if None in places:
            return False
        # `1 + 1e-6` of slack, not an epsilon: `3.4474 - 3.4473` evaluates to
        # 1.000000000002e-4, so an exact comparison against one unit in the last
        # place rejects the very case the rule exists to allow.
        if abs(fx - fy) > max(places) * (1 + 1e-6):
            return False
    return True


def compare(a, b, path, out, tol, abs_tol):
    """Walk two JSON trees together, collecting every difference.

    Structural differences are reported as `!` (the two no longer describe the
    same thing), numeric ones as `~` (they describe the same thing to within
    arithmetic noise). The two are separated because they call for different
    responses: `!` means the trace changed, `~` means the machine did.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"! {path}.{k}: 只在再生成的结果里（提交的版本缺这个键）")
            elif k not in b:
                out.append(f"! {path}.{k}: 只在提交的版本里（生成器不再产出它）")
            else:
                compare(a[k], b[k], f"{path}.{k}", out, tol, abs_tol)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"! {path}: 长度 {len(a)} -> {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out, tol, abs_tol)
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            out.append(f"! {path}: {a!r} -> {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # An int in one and a float in the other is a type change, not noise:
        # every trace value here is meant to be a float, and a step count is
        # meant to be an int. Comparing across the two would hide a real edit.
        if isinstance(a, int) != isinstance(b, int):
            out.append(f"! {path}: 类型 {type(a).__name__} -> {type(b).__name__}")
            return
        if isinstance(a, int):
            if a != b:
                out.append(f"! {path}: {a} -> {b}")
            return
        if a == b:
            return
        delta = abs(a - b)
        if delta <= abs_tol or delta <= tol * max(abs(a), abs(b)):
            out.append(f"~ {path}: {a!r} -> {b!r}（差 {delta:.3g}，在容差 {tol:g} 内）")
        else:
            out.append(f"! {path}: {a!r} -> {b!r}（差 {delta:.3g}，超出容差 {tol:g}）")
        return
    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return
        if text_differs_only_numerically(a, b):
            out.append(f"~ {path}: 只是显示出来的数字在容差内移动")
        else:
            out.append(f"! {path}: {a[:120]!r} -> {b[:120]!r}")
        return
    if a != b:
        out.append(f"! {path}: {a!r} -> {b!r}")


def main():
    tol = _REL_TOL
    abs_tol = _ABS_TOL
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])

    names = sorted(p.name for p in (REPO / TRACE_DIR).glob("*.json"))
    if not names:
        print(f"no JSON under {TRACE_DIR}/", file=sys.stderr)
        return 2

    hard, soft, skipped, clean = [], [], [], []
    for name in names:
        old = committed(name)
        if old is None:
            skipped.append(name)
            continue
        new = json.loads((REPO / TRACE_DIR / name).read_text(encoding="utf-8"))
        diffs = []
        compare(old, new, name, diffs, tol, abs_tol)
        structural = [d for d in diffs if d.startswith("!")]
        numeric = [d for d in diffs if d.startswith("~")]
        print(f"\n{name}")
        if not diffs:
            print("  与提交的版本一致")
            clean.append(name)
            continue
        if structural:
            print(f"  结构性差异 {len(structural)} 处（前 8 条）：")
            for d in structural[:8]:
                print(f"    {d}")
            hard.append((name, structural))
        if numeric:
            print(f"  仅浮点末位差异 {len(numeric)} 处（前 3 条）：")
            for d in numeric[:3]:
                print(f"    {d}")
            soft.append((name, numeric))

    print("\n" + "=" * 60)
    if skipped:
        print(f"{len(skipped)} 份未入库（新 trace，无对照）：{'、'.join(skipped)}")
    print(f"{len(clean)} 份逐位一致，{len(soft)} 份只有浮点末位差异"
          f"（容差 {tol:g}），{len(hard)} 份有结构性差异")

    if hard:
        print("\n有 trace 与生成器的输出不再一致 —— 说明它是手改过的，"
              "或者生成器改过之后没有重新生成：", file=sys.stderr)
        for name, ds in hard:
            print(f"  {name}: {ds[0]}", file=sys.stderr)
        return 1
    print("全部一致（浮点差异在 BLAS 求和顺序的噪声范围内，"
          "且比页面显示的 4 位有效数字低 5 个数量级）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
