/* Engine · view component: the with/without KV cache attention comparison.
 *
 * The lab's central teaching point, as a picture: the SAME attention operation,
 * computed two ways, with the two operand shapes drawn side by side.
 *
 *     without a cache         with a cache
 *     Q [t, d]                Q [n_q, d]
 *     K [t, d]                K [t, d]
 *     S [H, t, t]             S [H, n_q, t]
 *
 * The two columns share one scale, so the difference is the width of a
 * rectangle rather than a number in a table: at decode the cached column's
 * score matrix collapses to a single row while the uncached column's stays
 * square and keeps growing with the context.
 *
 * WHAT THIS FILE DELIBERATELY DOES NOT KNOW
 *
 * It does not know what a KV cache is, or that `n_q` is the prompt length at
 * prefill and 1 at decode. It draws whatever two modes the trace publishes on
 * `step.attn_cost` — shapes, element counts, FLOPs — and computes a layout from
 * them. Every number it labels comes from that block; the only arithmetic here
 * is turning element counts into pixel areas, and `check()` tests that too.
 *
 * WHY THE BARS ARE AREA-SCALED AND THE SHAPES ARE NOT
 *
 * The operand rectangles are drawn to scale because that is the whole point —
 * a `[t,d]` and a `[1,d]` operands really do differ by a factor of `t`. The
 * score matrices are drawn as GRIDS of at most `maxCells` cells, because a
 * `[4, 256, 256]` matrix is 262,144 cells and no reader can count them; the
 * grid's dimensions are labelled with the true shape and its area carries the
 * ratio, which is the honest way to draw something too big to draw.
 *
 * CONFIG
 *
 *   id / label:  the panel's identity, as for every view.
 *   maxCells:    the cap on drawn grid cells per mode (default 1024).
 *
 * render(step, ctx) returns an HTML string for the attention step and a short
 * "no comparison here" note otherwise — the comparison only exists on a step
 * that published one, and saying so is better than drawing an empty frame.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;

  function g(v) {
    if (!isFinite(v)) return String(v);
    var s = Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)
      ? v.toExponential(2) : v.toPrecision(4);
    return s.replace(/(\.\d*?)0+e/, '$1e').replace(/\.e/, 'e')
            .replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  }

  function shapeText(shape) {
    return '[' + shape.join(', ') + ']';
  }

  function elems(shape) {
    return shape.reduce(function (a, b) { return a * b; }, 1);
  }

  /* ================================================================= render */

  function makeView(config) {
    var cfg = config || {};
    if (!cfg.trace) {
      throw new Error('attn-shape: makeView needs { trace } — the two modes are ' +
                      'the trace\'s, this view does not compute them');
    }
    var maxCells = cfg.maxCells || 1024;
    /* The narrow column's floor, as a fraction of the panel's width. 15% of a
     * ~1500px panel is 225px, which is about what the facts list needs to stay
     * on one line per row; below that the labels wrap and the column reads as
     * broken rather than as small. */
    var MIN_COL = cfg.minColShare === undefined ? 0.15 : cfg.minColShare;
    /* The tallest a score matrix may be drawn, in pixels. It is published to
     * the stylesheet as `--as-gh` (see `column`) rather than written twice:
     * the component needs it to derive the grid's max width from its aspect
     * ratio, and the CSS needs it as the max height, and those two numbers
     * disagreeing is exactly how a "square" matrix starts drawing 6.8:1. */
    var gridH = cfg.gridHeight || 190;
    /* The stylesheet's `min-height` for a score grid, in pixels. Published to
     * the CSS as `--as-gh-min` and used here to decide whether a rectangle gets
     * compressed, so the component's test and the browser's floor are the same
     * number rather than two that have to be kept in step by hand. */
    var MIN_GRID_H = cfg.minGridHeight || 18;

    /* The scale the operand bars are drawn against. `undefined` until `render`
     * computes it from both modes, which is deliberately the widest QUERY count
     * on the chart rather than each column's own: bars normalised per column
     * would all look the same width and the collapse would vanish. */
    var operandScale = 1;

    /**
     * One mode's column: the operand shapes, and the score matrix drawn to a
     * cell cap.
     */
    function column(mode, title, cls, gridW) {
      var score = mode.score_shape;
      var trueRows = Math.max(1, score[1]);     // queries
      var trueCols = Math.max(1, score[2]);     // context
      // Rows and columns of the DRAWN grid, scaled down together so the
      // rectangle keeps the true aspect ratio as far as whole cells allow: a
      // square [t,t] must not be drawn wider than it is tall, or the picture
      // would stop showing the thing it is about. `cap` is the side of the
      // largest square that fits the cell budget, so the drawn matrix never
      // exceeds it on either axis.
      var cap = Math.max(1, Math.floor(Math.sqrt(maxCells)));
      var scale = Math.min(1, cap / Math.max(trueRows, trueCols));
      var rows = Math.max(1, Math.round(trueRows * scale));
      var cols = Math.max(1, Math.round(trueCols * scale));
      var capped = rows !== trueRows || cols !== trueCols;
      /* Whether the drawn RECTANGLE had to be compressed, which is a different
       * question from whether the cell grid was capped.
       *
       * The rectangle takes its column's width W and its own aspect ratio's
       * height W/ar — UNLESS that lands under the stylesheet's `min-height`, at
       * which point the height floors and the drawn ratio is no longer the true
       * one. So the test is exactly that: `W < minHeight × ar`.
       *
       * It bites only for the strips. A [t, t] square has ar = 1, so it clears
       * the floor at any width; a [1, t] strip's true ratio is t:1 — 125:1 at
       * N=128 — and honouring it at 18px tall would need 2250px of width, which
       * no column has. The drawn rectangle then reads ~12:1, and that is a
       * distortion, so it is LABELLED rather than hidden: the caption prints
       * the true ratio, and the harness asserts the label appears exactly when
       * the drawn and true ratios disagree. The alternative — dropping the
       * height floor so the strip is honestly 2px tall — is not a picture. */
      var trueAr = trueCols / trueRows;
      var compressed = gridW < MIN_GRID_H * trueAr;
      var s = [];
      s.push('<div class="lab-as-col ' + cls + '" data-mode="' + esc(mode.id) + '"' +
             ' data-queries="' + esc(String(mode.queries)) + '"' +
             ' data-score-elems="' + esc(String(mode.score_elems)) + '">');
      s.push('<div class="lab-as-col-t">' + esc(title) + '</div>');

      // ---- the operands, drawn to scale against the shared key axis
      //
      // Q's bar is as wide as its query count, K's as wide as the context:
      // both are drawn against the SAME scale (`scale` below is the widest
      // query count anywhere on the chart, i.e. the uncached mode's `t`), so
      // the two columns' K bars line up and the cached Q collapses against
      // them. A bar per mode against its own maximum would have hidden exactly
      // the difference the picture is for.
      s.push('<div class="lab-as-ops">');
      [['Q', mode.q_shape], ['K', mode.k_shape]].forEach(function (pair) {
        var rows = pair[1][0];
        /* `pct` is EXACT — no floor. A floor baked in here would make the
         * reported width disagree with the drawn proportion at small ratios,
         * which is the one thing this picture must not do; the readable minimum
         * is applied by the stylesheet's `min-width` instead, where it affects
         * only how thin a bar can draw and not what the data says. */
        var pct = 100 * rows / operandScale;
        s.push('<div class="lab-as-op" data-op="' + pair[0] + '"' +
               ' data-rows="' + rows + '" data-cols="' + pair[1][1] + '"' +
               ' data-elems="' + elems(pair[1]) + '"' +
               ' data-width-pct="' + g(pct) + '"' +
               ' style="width:' + pct.toFixed(3) + '%">' +
               '<span class="lab-as-op-l">' + esc(pair[0]) + '</span>' +
               '<span class="lab-as-op-s">' + esc(shapeText(pair[1])) + '</span>' +
               '</div>');
      });
      s.push('</div>');

      // ---- the score matrix, drawn to the cell cap
      s.push('<div class="lab-as-grid-wrap">' +
             '<div class="lab-as-grid" data-rows="' + rows + '" data-cols="' + cols +
             '" data-true-rows="' + trueRows + '" data-true-cols="' + trueCols +
             '" data-capped="' + (capped ? '1' : '0') + '"' +
             /* Three things are load-bearing in this one style attribute.
              *
              * `minmax(0, 1fr)` and not `1fr`: a `1fr` track's minimum is
              * `auto`, i.e. the cell's min-content width — and a cell with
              * `aspect-ratio: 1` inside a row whose height is stretched by this
              * container's `min-height` reports a min-content width from that
              * ratio. The tracks then refuse to shrink and the grid paints 32
              * columns of 16px inside a 211px box. `minmax(0, 1fr)` removes the
              * floor, so the columns divide the box they are in.
              *
              * `max-width` from the height cap: `aspect-ratio` alone is not
              * enough, because a block element's width is its container's and
              * `max-height` then clamps the height without touching the width —
              * the "square" [4,125,125] matrix drew 1287x190, a 6.8:1 lie about
              * the shape it is there to show. Deriving the width from the cap
              * and the true ratio makes the rectangle answer to its own shape.
              * `--as-gh` is the same cap the stylesheet uses, so the two cannot
              * drift: the component sets the variable, the CSS consumes it. */
             ' style="--as-gh:' + gridH + 'px;--as-gh-min:' + MIN_GRID_H +
             'px;grid-template-columns:repeat(' + cols +
             ',minmax(0,1fr));aspect-ratio:' + trueCols + '/' + trueRows +
             ';max-width:min(100%,calc(var(--as-gh) * ' + trueCols + ' / ' + trueRows +
             '))"' +
             ' data-drawn-ar="' + (trueCols / trueRows).toFixed(3) + '"' +
             ' data-compressed="' + (compressed ? '1' : '0') + '">');
      var n = rows * cols;
      for (var i = 0; i < n; i++) {
        // The causal mask, made visible: in the uncached square the cells above
        // the diagonal are the ones the mask zeroes. Drawn only when the grid
        // is the true shape (an uncapped draw), because on a capped grid the
        // cells no longer map one-to-one onto positions and tinting them would
        // be a claim about specific entries that the layout cannot support.
        var masked = false;
        if (!capped && score[1] === score[2]) {
          var r = Math.floor(i / cols), col = i % cols;
          masked = col > r;
        }
        s.push('<span class="lab-as-cell' + (masked ? ' lab-as-cell-masked' : '') +
               '"></span>');
      }
      s.push('</div>');

      s.push('<div class="lab-as-grid-s">S ' + esc(shapeText(score)) +
             ' <b>' + esc(g(mode.score_elems)) + '</b> 个元素' +
             (capped ? '（单元格按比例缩绘）' : '') +
             (compressed ? '<br>矩形按列宽压缩，真实长宽比 ' +
                           (trueCols / trueRows).toFixed(1) + ':1' : '') +
             '</div>');
      s.push('</div>');

      s.push('<dl class="lab-as-facts">' +
        '<dt>查询数 Q</dt><dd data-fact="queries">' + esc(g(mode.queries)) + '</dd>' +
        '<dt>分数矩阵</dt><dd data-fact="score_elems">' + esc(g(mode.score_elems)) + '</dd>' +
        '<dt>注意力 FLOPs</dt><dd data-fact="attn_flops">' + esc(g(mode.attn_flops)) + '</dd>' +
        '<dt>投影 FLOPs</dt><dd data-fact="proj_flops">' + esc(g(mode.proj_flops)) + '</dd>' +
        '<dt>KV 读取</dt><dd data-fact="kv_read_elems">' + esc(g(mode.kv_read_elems)) + '</dd>' +
        '</dl>');
      s.push('</div>');
      return s.join('');
    }

    /**
     * The comparison to show at a cursor.
     *
     * The comparison only exists on attention steps, but the panel is on
     * screen for every step — and leaving it blank four frames out of five
     * would hide the lab's central picture behind a click. So a non-attention
     * cursor shows the most recent attention step's comparison, and before the
     * first one exists (cursor 0, the K/V append) the first one ahead.
     *
     * This is still a pure function of (trace, cursor) — the same two inputs
     * `resolve` takes — so the player's "jump anywhere and get the same frame"
     * guarantee is untouched. What it is NOT is a claim that the step you are
     * looking at is this one: `from` names the step the numbers came from, and
     * the caller prints it.
     */
    function nearest(step, trace) {
      if (step && step.attn_cost) return { step: step, from: step, exact: true };
      var steps = (trace && trace.steps) || [];
      var at = steps.indexOf(step);
      if (at < 0) return null;
      var found = null;
      for (var i = at; i >= 0; i--) {
        if (steps[i].attn_cost) { found = steps[i]; break; }
      }
      var exact = false;
      if (!found) {
        for (var j = at; j < steps.length; j++) {
          if (steps[j].attn_cost) { found = steps[j]; break; }
        }
      }
      return found ? { step: found, from: found, exact: exact } : null;
    }

    function render(step, ctx) {
      var picked = nearest(step, cfg.trace);
      var c = picked && picked.from.attn_cost;
      if (!c) {
        return '<div class="lab-as"><p class="lab-as-none">' +
          '这份 trace 里没有任何注意力步 —— 有/无 KV Cache 的对比没有数据。' +
          '</p></div>';
      }
      var nc = c.no_cache, wc = c.with_cache;
      nc.id = 'no_cache';
      wc.id = 'with_cache';
      // The uncached mode always has the most queries (it queries every
      // position), so it sets the scale; `Math.max` rather than a pick so a
      // future trace where that stops holding still draws both bars in frame.
      operandScale = Math.max(nc.queries, wc.queries, 1);

      var ratio = nc.score_elems / Math.max(wc.score_elems, 1);
      // The two columns are given widths in PROPORTION to their score-matrix
      // element counts, floored so the narrow one stays readable.
      var total = nc.score_elems + wc.score_elems;

      var s = [];
      s.push('<div class="lab-as" data-ratio="' + esc(g(ratio)) + '"' +
             ' data-from-step="' + esc(picked.from.id) + '"' +
             ' data-exact="' + (picked.exact ? '1' : '0') + '">');
      s.push('<div class="lab-as-hd">' +
        '<span class="lab-as-hd-t">同一次注意力，两种算法</span>' +
        '<span class="lab-as-hd-s">' + esc(picked.from.title) + ' · 上下文 t = ' +
        esc(g(c.t)) + ' · 本步查询 n_q = ' + esc(g(c.n_q)) +
        (picked.exact ? '' : '（光标停在第 ' + esc(String(ctx ? ctx.cursor : '?')) +
                             ' 步，下面显示最近一次注意力计算：' + esc(picked.from.id) + '）') +
        '</span></div>');
      s.push('<div class="lab-as-cols">');
      /* The two columns are sized by their score matrices' element counts,
       * FLOORED at `MIN_COL` so the narrow column stays readable. The floor is
       * a percentage rather than a pixel width on purpose: a pixel floor would
       * grow with the viewport while the wide column's share did not, so at
       * large ratios the narrow column would silently creep wider and the
       * comparison would stop being proportional.
       *
       * `data-share` carries the un-floored ratio the harness reads, so the
       * picture's proportionality can be asserted separately from its
       * readability. */
      var share = total > 0 ? wc.score_elems / total : 0.5;
      var floored = Math.max(MIN_COL, share);
      /* GRID, not flex, and the reason is a real bug the first version had.
       *
       * `flex: <huge> 1 0` with a `min-width` on the sibling looks like it
       * should work: grow shares the free space, the min bids up the narrow
       * column. It does not. A flex item with `flex-basis: 0` has nothing to
       * shrink FROM, so when the floor pushes the total past the container the
       * browser cannot take the space back out of the wide item and the row
       * overflows — 2879px of columns inside a 1556px panel at N=128.
       *
       * Grid's `minmax(min, <fr>)` is the shape this actually is: the track
       * grows in proportion to the element counts, and the `min` is a real
       * floor the fr distribution accounts for rather than one it collides
       * with. The min is a percentage (of the grid container) rather than a
       * pixel width so it cannot creep wider as the viewport grows while the
       * other track's share stands still. */
      s.push('<div class="lab-as-cols" data-nc-share="' + esc(g(1 - share)) +
             '" data-wc-share="' + esc(g(share)) + '"' +
             ' data-wc-floored="' + (floored > share ? '1' : '0') + '"' +
             ' style="grid-template-columns:minmax(0,' + nc.score_elems + 'fr) ' +
             'minmax(' + (MIN_COL * 100).toFixed(2) + '%,' + wc.score_elems + 'fr)">');
      /* Each column's usable width, so the score-matrix rectangle can be
       * clamped to it rather than overflowing. The panel's own width is not
       * known until it is in the DOM, so this is a conservative estimate from
       * the pane width the page lays the panel out in — an estimate is fine
       * for a CLAMP, because being a few pixels off only changes whether the
       * compression label appears at the margin, and the harness asserts the
       * label against the drawn geometry rather than against this number. */
      var paneW = cfg.paneWidth || 1500;
      var gap = 12, chrome = 46;   // the grid's gap, borders and padding
      var ncW = (paneW - gap) * (1 - floored) - chrome;
      var wcW = (paneW - gap) * floored - chrome;

      s.push('<div class="lab-as-slot">' +
             column(nc, '无 cache：整段前缀重算一遍', 'lab-as-nc', ncW) + '</div>');
      s.push('<div class="lab-as-slot">' +
             column(wc, '有 cache：只算本步的新查询', 'lab-as-wc', wcW) + '</div>');
      s.push('</div>');

      /* The headline, and it carries the comparison's own guard: at prefill the
       * two are equal, and the sentence has to say so rather than let a reader
       * conclude the picture is broken. */
      var identical = nc.score_elems === wc.score_elems && nc.queries === wc.queries;
      s.push('<p class="lab-as-sum">' + (identical
        ? '本步两种模式工作量<b>完全相同</b>（' + esc(g(nc.score_elems)) +
          ' 个分数矩阵元素）：Prefill 时缓存还是空的，没有历史可以复用，' +
          '所以这一刻两条路重合 —— 差别要到第一个 Decode 步才出现。'
        : '分数矩阵 <b>' + esc(g(nc.score_elems)) + ' → ' + esc(g(wc.score_elems)) +
          '</b> 个元素，少做 <b>' + esc(g(ratio)) + '×</b>；查询数 ' +
          esc(g(nc.queries)) + ' → ' + esc(g(wc.queries)) + '。' +
          '但真正把「不缓存」拖垮的是<b>投影</b>：' + esc(g(nc.proj_flops)) + ' vs ' +
          esc(g(wc.proj_flops)) + ' FLOPs —— 不缓存就得把整段前缀的 Q/K/V 重新算一遍。')
        + '</p>');
      s.push('</div>');
      return s.join('');
    }

    return { render: render };
  }

  /**
   * The player's `panels: [{id, label, render(step, ctx)}]` entry.
   */
  function panel(config, context) {
    var view = makeView({ trace: context.trace,
                          maxCells: config && config.maxCells });
    return {
      id: (config && config.id) || 'attn-shape',
      label: (config && config.label) || '有 / 无 KV Cache：同一次注意力的两种 shape',
      render: function (step, ctx) { return view.render(step, ctx); }
    };
  }

  /* ================================================================= lint
   *
   * The JS port of the `step.attn_cost` contract that this view renders.
   * `labs/traces/kv_cache.py` is the authority and runs before a trace is
   * written; this exists so a lab author editing a trace in the browser gets
   * the same verdict without a Python round trip.
   *
   * THE TWO PORTS MUST AGREE. The acceptance harness runs one sabotage table
   * through both and requires the same verdict from each; a rule that exists on
   * one side only is a rule tested on neither.
   *
   * It lives beside the view rather than in `trace-model.js` for the same
   * reason the ledger's does: `trace-model.js` was frozen once the engine
   * landed, and this block is this file's subject.
   */
  var COST_SHAPE_KEYS = ['q_shape', 'k_shape', 'score_shape'];
  var COST_INT_KEYS = ['queries', 'score_elems', 'kv_read_elems', 'kv_write_elems',
                       'attn_flops', 'proj_flops'];
  // The fields that describe the COMPUTATION rather than the cache traffic --
  // what prefill's two modes must agree on. See `lint` below.
  var WORK_KEYS = ['queries', 'q_shape', 'k_shape', 'score_shape', 'score_elems',
                   'attn_flops', 'proj_flops'];

  function isNonNegInt(v) {
    return typeof v === 'number' && isFinite(v) && v >= 0 && Math.floor(v) === v;
  }

  function isShape(v) {
    return Array.isArray(v) && v.length > 0 && v.every(function (x) {
      return typeof x === 'number' && isFinite(x) && x > 0 && Math.floor(x) === x;
    });
  }

  function prod(shape) {
    return shape.reduce(function (a, b) { return a * b; }, 1);
  }

  function sameWork(a, b) {
    return WORK_KEYS.every(function (k) { return JSON.stringify(a[k]) === JSON.stringify(b[k]); });
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var steps = (trace && trace.steps) || [];
    var withCost = steps.filter(function (s) { return s && s.attn_cost !== undefined; });

    if (!withCost.length) {
      if (((trace || {}).meta || {}).attn_cost) {
        gaps.push({ what: 'trace 没有步骤带 attn_cost，但 meta.attn_cost 存在',
                    why: '汇总没有逐帧依据。' });
      }
      return { gaps: gaps, warns: warns, infos: infos, present: false };
    }

    withCost.forEach(function (s) {
      var sid = s.id, c = s.attn_cost;
      ['t', 'n_q', 'H', 'd_head', 'layers', 'no_cache', 'with_cache'].forEach(function (k) {
        if (!(k in c)) {
          gaps.push({ what: '步骤 "' + sid + '" 的 attn_cost 缺 "' + k + '"', why: '画不出来。' });
        }
      });
      ['no_cache', 'with_cache'].forEach(function (mode) {
        var m = c[mode];
        if (typeof m !== 'object' || m === null) {
          gaps.push({ what: '步骤 "' + sid + '" 的 attn_cost["' + mode + '"] 不是对象',
                      why: '两栏之一缺失，对比无从谈起。' });
          return;
        }
        COST_SHAPE_KEYS.forEach(function (k) {
          if (!isShape(m[k])) {
            gaps.push({ what: '步骤 "' + sid + '" 的 attn_cost["' + mode + '"].' + k +
                              '=' + JSON.stringify(m[k]),
                        why: '不是正整数的 shape。' });
          }
        });
        COST_INT_KEYS.forEach(function (k) {
          if (!isNonNegInt(m[k])) {
            gaps.push({ what: '步骤 "' + sid + '" 的 attn_cost["' + mode + '"].' + k +
                              '=' + JSON.stringify(m[k]),
                        why: '必须是非负整数。' });
          }
        });
      });
      if (gaps.length) return;

      var nc = c.no_cache, wc = c.with_cache;
      /* The score matrix's element count IS its shape's product. This is the
       * rule that stops "the shape got wider" and "the work got bigger" from
       * being two independent claims on the same picture. */
      [['no_cache', nc], ['with_cache', wc]].forEach(function (pair) {
        var m = pair[1];
        if (m.score_shape[0] !== c.H || m.score_shape[1] !== m.queries ||
            m.score_shape[2] !== c.t) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ' + pair[0] + '.score_shape=' +
                            JSON.stringify(m.score_shape) + ' 应为 [H,queries,t]=[' +
                            c.H + ',' + m.queries + ',' + c.t + ']',
                      why: '分数矩阵的三维就是头 × 查询 × 上下文。' });
        } else if (m.score_elems !== prod(m.score_shape)) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ' + pair[0] + '.score_elems=' +
                            m.score_elems + ' 与 shape 的乘积 ' + prod(m.score_shape) + ' 不符',
                      why: '同一个矩阵的两个说法互相矛盾。' });
        }
        var wantAttn = 4 * c.H * m.queries * c.t * c.d_head;
        if (m.attn_flops !== wantAttn) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ' + pair[0] + '.attn_flops=' +
                            m.attn_flops + ' 应为 4·H·q·t·d_h=' + wantAttn,
                      why: '注意力 FLOPs 与它自己的 shape 脱节。' });
        }
        if (m.q_shape.length !== 2 || m.q_shape[0] !== m.queries) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ' + pair[0] + '：q_shape=' +
                            JSON.stringify(m.q_shape) + ' 与 queries=' + m.queries + ' 不一致',
                      why: 'Q 的行数就是查询数。' });
        }
        /* K's rows are the context, in BOTH modes: the cached path reads the
         * whole cache, and the uncached path recomputes the whole prefix — so
         * neither may claim a shorter key axis than the context it attended
         * over. This is the rule that keeps a page from drawing a wide score
         * matrix over a narrow K. */
        if (m.k_shape.length !== 2 || m.k_shape[0] !== c.t) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ' + pair[0] + '：k_shape=' +
                            JSON.stringify(m.k_shape) + ' 的行数应为上下文 t=' + c.t,
                      why: 'K 的行数就是上下文长度，两种模式都一样。' });
        }
        var wantProj = 6 * m.queries * (m.q_shape[1] || 0) * (m.q_shape[1] || 0) * c.layers;
        if (m.proj_flops !== wantProj) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ' + pair[0] + '.proj_flops=' +
                            m.proj_flops + ' 应为 6·q·d²·L=' + wantProj,
                      why: '不缓存的代价主要落在重新投影上，这个数不能随便写。' });
        }
        // The cache traffic, re-derived from the ledger's own KV formula.
        var wantRead = pair[0] === 'with_cache' ? 2 * c.layers * c.H * c.d_head * c.t : 0;
        var wantWrite = pair[0] === 'with_cache' ? 2 * c.layers * c.H * c.d_head * c.n_q : 0;
        if (m.kv_read_elems !== wantRead || m.kv_write_elems !== wantWrite) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ' + pair[0] + ' KV 访存 (' +
                            m.kv_read_elems + ', ' + m.kv_write_elems + ') 应为 (' +
                            wantRead + ', ' + wantWrite + ')',
                      why: '不缓存就没有缓存可读可写，它的代价在投影 FLOPs 上。' });
        }
      });

      if (nc.queries !== c.t) {
        gaps.push({ what: '步骤 "' + sid + '"：无缓存时 queries=' + nc.queries +
                          ' 应等于上下文 t=' + c.t,
                    why: '不缓存就要把整段前缀的每个位置都当查询重算一遍。' });
      }
      if (wc.queries !== c.n_q) {
        gaps.push({ what: '步骤 "' + sid + '"：有缓存时 queries=' + wc.queries +
                          ' 应等于本步新查询数 n_q=' + c.n_q,
                    why: '有缓存时只有本步的新 token 需要查询。' });
      }
    });

    /* The prefill control, and the divergence that makes it meaningful. */
    var prefills = withCost.filter(function (s) { return s.id.indexOf('prefill') === 0; });
    prefills.forEach(function (s) {
      if (!sameWork(s.attn_cost.no_cache, s.attn_cost.with_cache)) {
        gaps.push({ what: '步骤 "' + s.id + '"：Prefill 时两种模式必须做同样的工作',
                    why: '缓存是空的，没有历史可以复用 —— 这是 decode 分歧的对照。' });
      }
    });
    if (!prefills.length) {
      gaps.push({ what: '没有任何 prefill 步骤带 attn_cost',
                  why: '两种模式的重合点无法校验。' });
    } else {
      var diverged = withCost.filter(function (s) {
        return !sameWork(s.attn_cost.no_cache, s.attn_cost.with_cache);
      });
      if (!diverged.length) {
        gaps.push({ what: '所有步骤两种模式都相同',
                    why: 'decode 阶段没有分歧，这个对比演示不了任何东西。' });
      }
    }

    var meta = ((trace || {}).meta || {}).attn_cost;
    if (meta && typeof meta.totals === 'object' && meta.totals) {
      ['no_cache', 'with_cache'].forEach(function (mode) {
        var tot = meta.totals[mode];
        if (typeof tot !== 'object' || tot === null) {
          gaps.push({ what: 'meta.attn_cost.totals 缺 "' + mode + '"', why: '合计不全。' });
          return;
        }
        ['score_elems', 'attn_flops', 'proj_flops', 'kv_read_elems',
         'kv_write_elems'].forEach(function (k) {
          var want = withCost.reduce(function (a, s) { return a + s.attn_cost[mode][k]; }, 0);
          if (tot[k] !== want) {
            gaps.push({ what: 'meta.attn_cost.totals["' + mode + '"].' + k + '=' + tot[k] +
                              ' 与逐步之和 ' + want + ' 不符',
                        why: '页面上的合计必须等于它下面那些步。' });
          }
        });
      });
    } else {
      gaps.push({ what: '步骤带了 attn_cost，但 meta.attn_cost.totals 缺失',
                  why: '页面上的合计会无处可查。' });
    }

    infos.push({ what: '对比覆盖 ' + withCost.length + ' 个注意力步，' +
                        prefills.length + ' 个 prefill 步两种模式重合',
                 why: '' });
    return { gaps: gaps, warns: warns, infos: infos, present: true };
  }

  /* Each entry breaks one rule above, on a real trace. The names parallel the
   * Python `SABOTAGE_CASES` in `labs/traces/kv_cache.py`, so the two tables can
   * be matched up entry for entry. */
  var SABOTAGE_CASES = {
    'attn meta.attn_cost 被删但步骤还留着': function (t) { delete t.meta.attn_cost; },
    'attn 某个 decode 步没有 attn_cost': function (t) { delete t.steps[5].attn_cost; },
    'attn score_elems 与 shape 乘积不符': function (t) {
      t.steps[1].attn_cost.with_cache.score_elems += 1;
    },
    'attn 有缓存时声称查询数等于上下文': function (t) {
      t.steps[5].attn_cost.with_cache.queries = t.steps[5].attn_cost.t;
    },
    'attn 无缓存时没有重算整段前缀': function (t) {
      t.steps[5].attn_cost.no_cache.queries = 1;
    },
    'attn prefill 两种模式出现分歧': function (t) {
      t.steps[1].attn_cost.with_cache.score_elems = 7;
    },
    'attn 没有任何一步出现分歧': function (t) {
      [5, 9, 13, 17].forEach(function (i) {
        t.steps[i].attn_cost.with_cache =
          JSON.parse(JSON.stringify(t.steps[i].attn_cost.no_cache));
      });
    },
    'attn score_shape 不是 [H, queries, t]': function (t) {
      t.steps[1].attn_cost.with_cache.score_shape = [4, 1, 4];
    },
    'attn attn_flops 不是 4Hqtd_h': function (t) {
      t.steps[5].attn_cost.with_cache.attn_flops = 1;
    },
    'attn K 行数与上下文不符': function (t) {
      t.steps[9].attn_cost.with_cache.k_shape = [3, 64];
    },
    'attn 有缓存却把 KV 读取记成 0': function (t) {
      t.steps[5].attn_cost.with_cache.kv_read_elems = 0;
    },
    'attn 无缓存时投影 FLOPs 不随 t 增长': function (t) {
      t.steps[5].attn_cost.no_cache.proj_flops = 6 * t.steps[5].attn_cost.t;
    },
    'attn 某个字段不是整数': function (t) {
      t.steps[5].attn_cost.with_cache.queries = '1';
    },
    'attn 合计与逐步之和不符': function (t) {
      t.meta.attn_cost.totals.with_cache.score_elems = 1;
    },
    'attn 合计缺一栏': function (t) { delete t.meta.attn_cost.totals.no_cache; }
  };

  /**
   * Prove the lint is not a function that always returns zero. Returns the list
   * of mutations it failed to catch.
   */
  function sabotageChecks(trace) {
    var missed = [];
    Object.keys(SABOTAGE_CASES).forEach(function (name) {
      var copy = JSON.parse(JSON.stringify(trace));
      try {
        SABOTAGE_CASES[name](copy);
      } catch (err) {
        missed.push(name + '（破坏本身失败：' + err.message + '）');
        return;
      }
      var r = lint(copy);
      if (!r.gaps.length && !r.warns.length) missed.push(name);
    });
    return missed;
  }

  NS.attnShape = {
    makeView: makeView,
    panel: panel,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES,
    WORK_KEYS: WORK_KEYS
  };
})(typeof window !== 'undefined' ? window : globalThis);
