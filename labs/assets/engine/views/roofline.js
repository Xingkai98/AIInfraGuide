/* L00 engine · view: Roofline plot (Roofline 图).
 *
 * The picture the L01 parameter sliders move: arithmetic intensity on the x
 * axis, the performance ceiling the Roofline permits on the y, two roofs, and
 * one dot per implementation. Built for L01's GEMM tiling replay, designed so
 * any lab with a `meta.roofline` block can drive it — nothing in this file
 * knows what GEMM is, or that the tile has three parameters.
 *
 * WHAT IS DERIVED AND WHAT IS DECLARED
 *
 * Everything is declared, and that is the point. The points, the roofs and the
 * hardware model all arrive in `trace.meta.roofline`; this file computes a
 * layout and nothing else. It does not multiply a bandwidth by an intensity to
 * find a ceiling, because the ceiling it draws is the one the trace published —
 * a second implementation of that product would be a second opinion, and the
 * page would have no way to say which one it drew. See labs/traces/gemm_tiling.py
 * for where the numbers come from and the Python lint that re-derives them.
 *
 * THE AXES ARE LOG-LOG, AND THAT IS NOT DECORATION
 *
 * The quantity the lab is about spans two orders of magnitude: the naive
 * kernel's 0.25 FLOP/Byte against the tutorial's 128×128 tile at 32, and the
 * ceilings from 0.22 to 15.7 TFLOP/s. On linear axes every toy configuration
 * lands in the same pixel as the naive one and the whole parameter story is
 * invisible — which is the shape of bug this project has already shipped once
 * (see the layout search in ring.js). So both axes are logarithmic, and the
 * geometry is asserted off the rendered DOM rather than eyeballed.
 *
 * WHAT IT SHOWS
 *
 *   - the bandwidth roof (`bandwidth × AI`) and the compute roof (`peak_flops`),
 *     meeting at the ridge point;
 *   - a dashed ridge marker with its value, which is the number the whole
 *     discussion is relative to;
 *   - one dot per point, filled by role, each labelled with its arithmetic
 *     intensity so a reader can watch the `tiled` dot slide right as they move
 *     the sliders — and the `naive` dot stay exactly where it is, which is the
 *     control that makes the movement readable;
 *   - `bound` per point: which roof is binding there. Stated rather than left
 *     to the reader to infer from which line the dot touches.
 *
 * CONFIG
 *
 *   id / label:  the panel's identity, as for every view.
 *   labelPlacement(point, geom) -> 'above' | 'below'   optional override.
 *
 * render(cursor, ctx) returns an HTML string. It ignores the cursor: the
 * Roofline is a property of the configuration, not of the replay step. It is
 * still a panel so the page's reader sees it beside the stage, and so the
 * engine's paint loop is the only thing that has to know when to redraw.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;

  /* ============================================================== log scale */

  var LOG10 = Math.LN10;

  function log10(x) { return Math.log(x) / LOG10; }

  /**
   * The pixel mapping for one axis: a log domain, and `t(x)` into [0, 1].
   *
   * `lo`/`hi` are PADDED from the data by the caller, so a dot never sits on
   * the frame — a point drawn exactly on the axis is indistinguishable from an
   * axis tick and reads as part of the furniture.
   */
  function axis(lo, hi) {
    var a = log10(lo), b = log10(hi), span = (b - a) || 1;
    return {
      lo: lo, hi: hi, a: a, b: b, span: span,
      t: function (x) { return (log10(x) - a) / span; },
      at: function (f) { return Math.pow(10, a + f * span); }
    };
  }

  /**
   * Log-axis ticks at 1, 2 and 5 times each decade inside the domain.
   *
   * The 2 and 5 matter here: the tile sizes are powers of two, so the tiled
   * point's arithmetic intensities are 0.5, 0.667, 0.8, 1, 1.333, 2 — and a
   * decade-only axis would leave every one of them unlabelled and unreadable.
   */
  function ticks(ax) {
    var out = [];
    var k0 = Math.floor(ax.a) - 1, k1 = Math.ceil(ax.b) + 1;
    for (var k = k0; k <= k1; k++) {
      [1, 2, 5].forEach(function (m) {
        var v = m * Math.pow(10, k);
        if (v >= ax.lo && v <= ax.hi) out.push(v);
      });
    }
    return out;
  }

  /**
   * A tick's label, short enough to not collide with its neighbour.
   *
   * In the trace's own units a value like 2.25e11 is nine characters of noise;
   * `0.2` is the same information. The caller passes the divisor (1e12 turns
   * FLOP/s into TFLOP/s) and the suffix carries the unit.
   */
  function tickLabel(v, divisor) {
    var s = v / divisor;
    if (s >= 1) return String(Math.round(s * 100) / 100);
    if (s >= 0.1) return String(Math.round(s * 100) / 100);
    return s.toPrecision(1);
  }

  function g(v) {
    if (!isFinite(v)) return String(v);
    var s = Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)
      ? v.toExponential(2) : v.toPrecision(4);
    return s.replace(/(\.\d*?)0+e/, '$1e').replace(/\.e/, 'e')
            .replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  }

  /* ================================================================= render */

  function makeView(config) {
    var cfg = config || {};
    if (!cfg.trace) {
      throw new Error('roofline: makeView needs { trace } — the points are the ' +
                      'trace\'s, this view does not compute them');
    }
    var trace = cfg.trace;
    var roof = trace.meta && trace.meta.roofline;
    if (!roof || !roof.points) {
      throw new Error('roofline: this trace has no meta.roofline block');
    }

    var W = cfg.width || 760;
    var H = cfg.height || 360;
    /* Margins, and why these numbers rather than round ones: the left and
     * bottom hold tick labels plus the axis titles, the top holds the topmost
     * point's label when that label is placed above its dot. The right is small
     * because the rightmost point's label goes to its LEFT — see `placeLabel`. */
    var PAD = { l: 72, r: 26, t: 30, b: 66 };
    var plotW = W - PAD.l - PAD.r;
    var plotH = H - PAD.t - PAD.b;

    /* The domain is derived from the trace's own points plus the ridge, padded
     * by a third of a decade on each side. A fixed domain would be a second,
     * silent statement about the data — and it would clip a configuration whose
     * intensity happened to fall outside whatever range looked right for the
     * default tile. */
    /* FLOP/s -> whatever the trace draws its y axis in. The divisor is the
     * trace's, not this file's: the label says TFLOP/s and the ticks must mean
     * the same thing, and a constant here would be a second, silent statement
     * about the axis that the generator could not contradict. */
    var Y_DIV = roof.y_divisor;
    var xs = roof.points.map(function (p) { return p.ai; }).concat([roof.ridge]);
    var ys = roof.points.map(function (p) { return p.ceiling; }).concat([roof.peak_flops]);
    var xax = axis(Math.min.apply(null, xs) / Math.pow(10, 0.35),
                   Math.max.apply(null, xs) * Math.pow(10, 0.35));
    var yax = axis(Math.min.apply(null, ys) / Math.pow(10, 0.35),
                   Math.max.apply(null, ys) * Math.pow(10, 0.35));

    function px(ai) { return PAD.l + xax.t(ai) * plotW; }
    function py(fl) { return PAD.t + (1 - yax.t(fl)) * plotH; }

    /**
     * Which side of its dot a label goes on, and the sign of the offset.
     *
     * A pure function of where the dot landed, so it cannot drift from the
     * layout. Two real collisions drive it, and the first screenshot of this
     * chart had both:
     *
     *   - a dot near the BOTTOM of the plot has no room beneath it for text
     *     without leaving the frame. The naive dot is always the lowest thing
     *     on the chart, so this is not hypothetical;
     *   - a dot sitting ON the bandwidth roof has no room to its upper-left,
     *     because the roof runs through exactly there. The tiled dot is always
     *     on that roof by construction — its ceiling IS `bandwidth × ai` — so a
     *     label placed above-left of it lands on the dashed line every time.
     *
     * So the label goes BELOW a dot that is on the bandwidth roof or near the
     * floor, and above otherwise. `bound` is the trace's own statement of which
     * roof is binding, which is why this reads it rather than comparing
     * coordinates: whether a dot sits on the diagonal is a property of the
     * point, not of the pixel it landed on.
     */
    function placeLabel(p, y) {
      if (y > PAD.t + plotH * 0.80) return 'below';
      if (p && p.bound === 'bandwidth') return 'below';
      return 'above';
    }

    function render() {
      var s = [];
      s.push('<div class="lab-roof" data-peak="' + esc(String(roof.peak_flops)) +
             '" data-bandwidth="' + esc(String(roof.bandwidth)) +
             '" data-ridge="' + esc(String(roof.ridge)) + '">');

      s.push('<svg class="lab-roof-svg" viewBox="0 0 ' + W + ' ' + H +
             '" data-w="' + W + '" data-h="' + H + '" role="img"' +
             ' aria-label="Roofline：算术强度与可达算力上界">');

      /* The plot frame, emitted first so everything else paints over it. Its
       * own element (rather than the svg's box) is what the harness measures
       * against: "inside the plot" and "inside the svg" are different claims,
       * and a dot can satisfy the second while sitting on an axis label. */
      s.push('<rect class="lab-roof-plot" x="' + PAD.l + '" y="' + PAD.t +
             '" width="' + plotW + '" height="' + plotH + '"></rect>');

      // ---- gridlines and tick labels, both axes
      ticks(xax).forEach(function (v) {
        var x = px(v);
        s.push('<line class="lab-roof-grid" x1="' + x.toFixed(2) + '" y1="' + PAD.t +
               '" x2="' + x.toFixed(2) + '" y2="' + (PAD.t + plotH) + '"></line>');
        s.push('<text class="lab-roof-tick lab-roof-tick-x" x="' + x.toFixed(2) +
               '" y="' + (PAD.t + plotH + 16) + '">' + esc(tickLabel(v, 1)) +
               '</text>');
      });
      ticks(yax).forEach(function (v) {
        var y = py(v);
        s.push('<line class="lab-roof-grid" x1="' + PAD.l + '" y1="' + y.toFixed(2) +
               '" x2="' + (PAD.l + plotW) + '" y2="' + y.toFixed(2) + '"></line>');
        s.push('<text class="lab-roof-tick lab-roof-tick-y" x="' + (PAD.l - 8) +
               '" y="' + (y + 3.5).toFixed(2) + '">' + esc(tickLabel(v, Y_DIV)) +
               '</text>');
      });

      // ---- the two roofs, and the ridge where they meet
      //
      // Drawn as polyline segments rather than two infinite lines clipped to
      // the frame: the bandwidth roof is only the roof to the LEFT of the ridge
      // and the compute roof only to the right, and drawing them across the
      // whole plot would show a ceiling that does not exist on the far side.
      var ridgeX = px(roof.ridge);
      var ridgeY = py(roof.peak_flops);
      s.push('<polyline class="lab-roof-roof lab-roof-roof-bw" points="' +
             PAD.l + ',' + py(yax.lo).toFixed(2) + ' ' + ridgeX.toFixed(2) + ',' +
             ridgeY.toFixed(2) + '"></polyline>');
      s.push('<polyline class="lab-roof-roof lab-roof-roof-compute" points="' +
             ridgeX.toFixed(2) + ',' + ridgeY.toFixed(2) + ' ' +
             (PAD.l + plotW) + ',' + ridgeY.toFixed(2) + '"></polyline>');
      s.push('<line class="lab-roof-ridge" x1="' + ridgeX.toFixed(2) + '" y1="' +
             (PAD.t + plotH) + '" x2="' + ridgeX.toFixed(2) + '" y2="' +
             ridgeY.toFixed(2) + '"></line>');
      s.push('<text class="lab-roof-ridge-n" x="' + (ridgeX + 5).toFixed(2) +
             '" y="' + (PAD.t + plotH - 5) + '">ridge ' + esc(g(roof.ridge)) +
             '</text>');

      // ---- the points
      //
      // Emitted in the trace's own order (naive, tiled, doc — the Python lint
      // requires exactly that order) so the DOM order is the reading order and
      // a screenshot and the data agree.
      roof.points.forEach(function (p) {
        var x = px(p.ai), y = py(p.ceiling);
        var where = (cfg.labelPlacement && cfg.labelPlacement(p, { x: x, y: y }))
          || placeLabel(p, y);
        var ly = where === 'below' ? y + 15 : y - 11;
        /* The rightmost dot's label goes left of it. It is the only one that
         * can run out of frame on the right, and at the doc tile's 32 FLOP/Byte
         * it sits close enough to the edge that a right-anchored label would be
         * clipped — a label half outside the viewBox is silently cut, which
         * reads as a rendering glitch rather than as a layout bug. */
        var rightmost = cfg.rightmostId || 'doc';
        var anchor = p.id === rightmost ? 'end' : 'start';
        var lx = p.id === rightmost ? x - 9 : x + 9;
        s.push('<g class="lab-roof-pt lab-roof-pt-' + esc(p.id) +
               '" data-id="' + esc(p.id) + '" data-ai="' + esc(String(p.ai)) +
               '" data-ceiling="' + esc(String(p.ceiling)) +
               '" data-bound="' + esc(p.bound) + '">' +
               '<circle class="lab-roof-dot" cx="' + x.toFixed(2) + '" cy="' +
               y.toFixed(2) + '" r="5"><title>' + esc(p.label + '：AI ' + g(p.ai) +
               ' FLOP/B，上界 ' + g(p.ceiling / Y_DIV) + ' TFLOP/s（' +
               p.bound + '-bound）') + '</title></circle>' +
               '<text class="lab-roof-pt-l" x="' + lx.toFixed(2) + '" y="' +
               ly.toFixed(2) + '" text-anchor="' + anchor + '">' +
               esc(p.label + ' · AI ' + g(p.ai)) + '</text>' +
               '</g>');
      });

      // ---- axis titles
      s.push('<text class="lab-roof-ax" x="' + (PAD.l + plotW / 2) + '" y="' +
             (PAD.t + plotH + 34) + '">' + esc(roof.x_label) + '</text>');
      s.push('<text class="lab-roof-ax lab-roof-ax-y" transform="translate(16,' +
             (PAD.t + plotH / 2) + ') rotate(-90)">' +
             esc(roof.y_label) + '</text>');
      /* The peak value is stated here, beside the axis title, rather than in a
       * corner caption. The corner version collided with the rightmost x tick
       * label — both occupy that strip — and the collision was invisible to
       * every count-based assertion. The axis title has the whole bottom margin
       * to itself, and "the roof tops out at N" is an axis fact anyway. */
      s.push('<text class="lab-roof-unit" x="' + (PAD.l + plotW / 2) + '" y="' +
             (PAD.t + plotH + 48) + '">' +
             esc('峰值 ' + g(roof.peak_flops / Y_DIV) + ' ' + roof.y_unit +
                 ' / 带宽 ' + g(roof.bandwidth / 1e9) + ' GB/s') +
             '</text>');

      s.push('</svg>');

      /* The note under the chart, verbatim from the trace: it says the dots are
       * a ceiling and not a measurement, and that sentence belongs to the
       * generator that computed them, not to this file. */
      s.push('<p class="lab-roof-note">' + esc(roof.note) + '</p>');
      s.push('</div>');
      return s.join('');
    }

    return { render: render, roof: roof };
  }

  /**
   * The player's `panels: [{id, label, render(step, ctx)}]` entry.
   *
   * `render` ignores the step: unlike the memory-hierarchy stage, this picture
   * is a property of the configuration and not of the cursor. It is still a
   * panel rather than page furniture because it belongs beside the stage in the
   * engine's grid, and because the engine's paint loop is the only thing that
   * should have to know when a panel is stale.
   */
  function panel(config, context) {
    var view = makeView({ trace: context.trace });
    return {
      id: config.id || 'roofline',
      label: config.label || 'Roofline：为什么分块有用',
      render: function () { return view.render(); }
    };
  }

  /* ================================================================= lint
   *
   * The JS port of the `meta.roofline` contract. Python
   * (labs/traces/gemm_tiling.py) is the authority and runs before a trace is
   * written; this exists so a lab author editing a trace in the browser gets the
   * same verdict without a Python round trip, exactly as trace-model.js's lint
   * mirrors online_softmax.py's.
   *
   * THE TWO PORTS MUST AGREE. The acceptance harness runs one sabotage table
   * through both and requires the same verdict from each; a rule that exists on
   * one side only is a rule tested on neither.
   *
   * WHAT THIS DOES NOT DO, and why it is not a gap: it does not re-check the
   * formulas, bindings, graph or `state` — those are trace-model.js's contract
   * and that file already carries their JS port. Nor does it verify that the
   * points describe a REAL kernel, which is not knowable in a browser: a
   * bandwidth and a peak are the hardware model the page chose, and nothing here
   * can tell a plausible pair from an invented one. What it checks is that the
   * chart it is about to draw is internally coherent — every number on it
   * re-derived from every other.
   */
  var ROOF_IDS = ['naive', 'tiled', 'doc'];

  function close(a, b, rel) {
    if (typeof a !== 'number' || typeof b !== 'number') return false;
    if (a === b) return true;
    var d = Math.abs(a - b);
    return d <= 1e-12 || d <= (rel === undefined ? 1e-12 : rel) * Math.max(Math.abs(a), Math.abs(b));
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var meta = (trace && trace.meta) || {};
    var roof = meta.roofline;

    if (!roof || typeof roof !== 'object') {
      gaps.push({ what: 'meta.roofline 缺失', why: '参数滑杆的 Roofline 图没有数据。' });
      return { gaps: gaps, warns: warns, infos: infos };
    }

    var REQUIRED = ['peak_flops', 'bandwidth', 'ridge', 'points',
                    'x_label', 'y_label', 'y_divisor', 'y_unit', 'note'];
    REQUIRED.forEach(function (k) {
      if (!(k in roof)) {
        gaps.push({ what: 'meta.roofline 缺 ' + k, why: '图上的轴或标题会显示 undefined。' });
      }
    });

    var peak = roof.peak_flops, bw = roof.bandwidth;
    /* `haveModel` gates every check below that divides by, multiplies by or
     * compares against the ridge. A trace MISSING the ridge has already been
     * reported once, in the loop above; walking on regardless would turn that
     * report into a NaN comparison that silently passes — or a crash — which
     * reads as a broken lint rather than as a broken trace. */
    var haveModel = typeof peak === 'number' && isFinite(peak) &&
                    typeof bw === 'number' && isFinite(bw) && bw > 0 &&
                    typeof roof.ridge === 'number' && isFinite(roof.ridge);

    if (haveModel && !close(roof.ridge, peak / bw)) {
      gaps.push({ what: 'meta.roofline.ridge = ' + JSON.stringify(roof.ridge) +
                  ' 应为峰值算力 / 带宽 = ' + (peak / bw),
                  why: '平衡点与硬件模型脱节 —— 图上的竖直参考线会在错误的位置。' });
    }

    /* The divisor and the unit are the two halves of one statement about what
     * the y axis means: naming a unit in the label and dividing the ticks by
     * something else would make the chart unreadable by exactly the factor
     * between them. */
    var yd = roof.y_divisor;
    if (typeof yd !== 'number' || !isFinite(yd) || yd <= 0) {
      gaps.push({ what: 'meta.roofline.y_divisor = ' + JSON.stringify(yd) +
                  ' 应为正数', why: '纵轴刻度按它换算。' });
    } else {
      var unit = roof.y_unit;
      if (typeof unit !== 'string' || !unit) {
        gaps.push({ what: 'meta.roofline.y_unit = ' + JSON.stringify(unit) +
                    ' 应为非空字符串', why: '轴标题要写出单位。' });
      } else if (String(roof.y_label || '').indexOf(unit) === -1) {
        gaps.push({ what: 'meta.roofline.y_label 里没有写明单位 "' + unit + '"',
                    why: '刻度按 ' + yd + ' 换算，读者无从知道单位是什么。' });
      }
    }

    var pts = roof.points;
    var ids = Array.isArray(pts) ? pts.map(function (p) { return p && p.id; }) : [];
    if (ids.join(',') !== ROOF_IDS.join(',')) {
      gaps.push({ what: 'meta.roofline.points 的 id 是 [' + ids.join(', ') +
                  ']，应该是 [' + ROOF_IDS.join(', ') + ']',
                  why: '点的顺序也参与渲染 —— 图例与 DOM 顺序按它排。' });
      return { gaps: gaps, warns: warns, infos: infos };
    }
    if (!haveModel) return { gaps: gaps, warns: warns, infos: infos };

    var cfg = meta.config || {};
    var flops = 2 * cfg.M * cfg.N * cfg.K;
    var byId = {};
    pts.forEach(function (p) { byId[p.id] = p; });

    pts.forEach(function (p) {
      var pid = p.id;
      var bs = p.bytes, ai = p.ai, pf = p.flops;

      if (typeof bs !== 'number' || !isFinite(bs) || bs <= 0 || bs !== Math.floor(bs)) {
        gaps.push({ what: 'meta.roofline.points["' + pid + '"].bytes = ' +
                    JSON.stringify(bs) + ' 应为正整数',
                    why: '算术强度是按它算出来的。' });
        return;
      }
      if (p.elements * 4 !== bs) {
        gaps.push({ what: 'meta.roofline.points["' + pid + '"] 的 elements 与 bytes 不符',
                    why: '同一个点的两个说法互相矛盾。' });
      }
      if (typeof pf !== 'number' || !isFinite(pf)) {
        gaps.push({ what: 'meta.roofline.points["' + pid + '"].flops = ' +
                    JSON.stringify(pf) + ' 不是数',
                    why: '算术强度要由它和 bytes 一起重新推一遍。' });
        return;
      }
      /* The point's intensity must be its OWN flops over its OWN bytes. This is
       * the rule that keeps "读取 1024 → 256 个元素" and "算术强度 0.25 → 2.0"
       * from becoming two unrelated claims about the same kernel. */
      if (!close(ai, pf / bs)) {
        gaps.push({ what: 'meta.roofline.points["' + pid + '"].ai = ' +
                    JSON.stringify(ai) + ' 与它自己的 flops / bytes = ' + (pf / bs) +
                    ' 不一致',
                    why: '图上的点和它标称的访存量是两套说法。' });
        return;
      }
      var want = ai < roof.ridge ? 'bandwidth' : 'compute';
      if (p.bound !== want) {
        gaps.push({ what: 'meta.roofline.points["' + pid + '"].bound = ' +
                    JSON.stringify(p.bound) + '，而在 AI = ' + ai + ' 上起作用的是 "' +
                    want + '" 的屋顶',
                    why: '页面上「受限于哪条屋顶」的结论会与图矛盾。' });
      }
      var wantCeiling = Math.min(peak, bw * ai);
      if (!close(p.ceiling, wantCeiling)) {
        gaps.push({ what: 'meta.roofline.points["' + pid + '"].ceiling = ' +
                    JSON.stringify(p.ceiling) + ' 应为 min(峰值, 带宽×AI) = ' + wantCeiling,
                    why: '点的纵坐标不是 Roofline 给出的那个上界。' });
      }
    });

    /* The naive and tiled points describe THIS configuration, so their element
     * counts are the traffic counts and their FLOPs are 2MNK. The doc point is
     * the tutorial's 128-cubed example and is deliberately not tied to this
     * trace's sizes — tying it would be the wrong rule, not a stricter one. */
    var tr = meta.traffic || {};
    [['tiled', 'tiled_reads'], ['naive', 'naive_reads']].forEach(function (pair) {
      var p = byId[pair[0]];
      if (p.elements !== tr[pair[1]]) {
        gaps.push({ what: 'meta.roofline 的 ' + pair[0] + ' 点用了 ' +
                    JSON.stringify(p.elements) + ' 个元素，而 meta.traffic.' +
                    pair[1] + ' = ' + JSON.stringify(tr[pair[1]]),
                    why: '图上的点和访存计数器是两个互相矛盾的说法。' });
      }
      if (p.flops !== flops) {
        gaps.push({ what: 'meta.roofline 的 ' + pair[0] + ' 点用了 ' +
                    JSON.stringify(p.flops) + ' FLOPs，而本配置是 2MNK = ' + flops,
                    why: '同一个问题的两个点的运算量应该相同。' });
      }
    });
    if (typeof tr.tiled_ai === 'number' && !close(byId.tiled.ai, tr.tiled_ai)) {
      gaps.push({ what: 'meta.roofline 的 tiled 点 AI = ' + JSON.stringify(byId.tiled.ai) +
                  ' 与 meta.traffic.tiled_ai = ' + JSON.stringify(tr.tiled_ai) + ' 不一致',
                  why: '页面上的两行数字来自同一个量，不能各说各话。' });
    }
    if (typeof tr.tiled_ai === 'number' && typeof tr.tiled_ai_closed === 'number' &&
        !close(tr.tiled_ai, tr.tiled_ai_closed)) {
      gaps.push({ what: 'meta.traffic 的两种算法不一致：计数式 ' +
                  JSON.stringify(tr.tiled_ai) + ' vs 闭式 ' +
                  JSON.stringify(tr.tiled_ai_closed),
                  why: '§4.3 的闭式与真实计数必须落在同一个数上。' });
    }
    /* Where the reading starts. The naive kernel has no tile at all, so its
     * point must be the same one on every configuration: 2MNK FLOPs over 2MNK
     * fp32 reads, i.e. 0.25. That stillness is the control the tiled point's
     * movement is read against — a version of this lab in which both points
     * moved would show movement without showing WHAT caused it. */
    var naiveAi = flops / (2 * cfg.M * cfg.N * cfg.K * 4);
    if (!close(byId.naive.ai, naiveAi)) {
      gaps.push({ what: 'meta.roofline 的 naive 点 AI = ' + JSON.stringify(byId.naive.ai) +
                  ' 应恒为 2MNK/8MNK = ' + naiveAi + '，与分块参数无关',
                  why: '朴素实现没有 tile，它的点不该随滑杆移动 —— 那是对照组。' });
    }

    infos.push({ what: pts.length + ' 个点 · ridge ' + g(roof.ridge) +
                 ' · ' + roof.points.map(function (p) { return p.id; }).join(' / '),
                 why: '' });
    return { gaps: gaps, warns: warns, infos: infos };
  }

  NS.roofline = {
    makeView: makeView,
    panel: panel,
    lint: lint,
    log10: log10
  };
})(typeof window !== 'undefined' ? window : globalThis);
