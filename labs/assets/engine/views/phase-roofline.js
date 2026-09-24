/* Engine · view component: the Prefill/Decode Roofline — and L05's trace-level
 * contract ports.
 *
 * The chart L05's fifth acceptance criterion is about: Prefill and Decode
 * placed on the tutorial's Roofline (1.1 §4), so the reader can see WHY they
 * land where they do rather than being told.
 *
 * IT RENDERS THROUGH THE SHARED ROOFLINE VIEW, AND THAT IS THE POINT
 *
 * The drawing — log-log axes, the two roofs meeting at the ridge, dots with
 * labels placed so they do not collide — is `views/roofline.js`'s, unchanged.
 * This file does not re-implement a pixel of it. What it adds is a SHIM: it
 * translates `trace.meta.phase_roofline` into the shape `roofline.makeView`
 * expects, calls it, and returns the string.
 *
 * That is deliberately a second key rather than a second chart. `roofline.js`'s
 * lint is not field-gated: it hard-requires L01's three roles (`naive`,
 * `tiled`, `doc`), `2MNK` FLOPs and a `meta.traffic` block, and would report
 * every L05 trace as broken. The repo's rule for a shared component is that its
 * rules are gated on the field the trace declares — that is how
 * `tiling-stage.js` serves four labs without turning three of them red — and
 * the engine is frozen for this ticket, so L05 declares a different key and
 * carries its own rules (below). The RENDERER, which is the part worth sharing,
 * is shared in full.
 *
 * WHY THE SHIM IS NOT TRIVIAL
 *
 * `roofline.makeView` derives its axis domains from the points it is handed.
 * L05's three points span four orders of magnitude of arithmetic intensity
 * (0.97 to 493 FLOP/Byte), so the chart is the same log-log picture L01 draws,
 * for the same reason: on a linear axis the prefill and decode dots would land
 * in the same pixel and the whole comparison would be invisible.
 *
 * WHAT IT ADDS AROUND THE CHART
 *
 * The chart alone shows three dots. The lesson needs two more things, and they
 * are what this file renders beside it:
 *
 *   - a per-phase table of the FLOP and byte breakdown, so "Prefill is
 *     compute-bound, Decode is memory-bound" is a claim a reader can check
 *     against the terms that produced it rather than a caption to believe;
 *   - the sentence that says WHY, which is a statement about the weight traffic
 *     being amortised over the prompt — and which the trace publishes in
 *     `traffic_note` rather than this file inventing.
 *
 * TWO TRACE-LEVEL BLOCKS, ONE PORT FILE
 *
 * Besides `meta.phase_roofline`, this file ports `meta.kv_crossover` — the
 * context length at which the cache outweighs the weights. That block is
 * *ledger* subject matter, so by the repo's rule its rules belong beside the
 * ledger's port in `views/ledger.js`; they are here instead because
 * `labs/assets/engine/` is frozen for this ticket and `ledger.js` is one of the
 * files it freezes. This is the same compromise `labs/pages/view-ledger.html`
 * documents for `verify.js`'s L00-specific sabotage list ("reported to the
 * tracker as a follow-up rather than fixed here"), and it is recorded in
 * `labs/traces/README.md` for whoever picks it up. Both blocks are trace-level
 * summaries this lab introduced, which is why they can share a port file
 * without either one's rules leaking into the other's subject.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;

  function g(v) {
    if (!isFinite(v)) return String(v);
    if (v === 0) return '0';
    var s = Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3)
      ? v.toExponential(2) : v.toPrecision(4);
    return s.replace(/(\.\d*?)0+e/, '$1e').replace(/\.e/, 'e')
            .replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  }

  /**
   * `meta.phase_roofline` -> the shape `roofline.makeView` renders.
   *
   * The only real translation is the point ids: the shared view derives label
   * placement from `bound` and from `rightmostId`, and it reads `id` to decide
   * which dot's label goes left. Everything else passes through.
   */
  function asRoofline(phase) {
    return {
      peak_flops: phase.peak_flops,
      bandwidth: phase.bandwidth,
      ridge: phase.ridge,
      x_label: phase.x_label,
      y_label: phase.y_label,
      y_divisor: phase.y_divisor,
      y_unit: phase.y_unit,
      note: phase.note,
      points: phase.points.map(function (p) {
        return {
          id: p.id, label: p.label, ai: p.ai,
          /* The shared view derives a point's `ai` from its FLOPs and bytes
           * when the lint checks it; the renderer only needs the three fields
           * below. Carried through anyway so a reader of this shim can see the
           * whole point object arrives intact. */
          elements: p.bytes / 4, bytes: p.bytes, flops: p.flops,
          ceiling: p.ceiling, bound: p.bound
        };
      })
    };
  }

  /* ================================================================= render */

  function makeView(config) {
    var cfg = config || {};
    if (!cfg.trace) {
      throw new Error('phase-roofline: makeView needs { trace } — the points are ' +
                      'the trace\'s, this view does not compute them');
    }
    var trace = cfg.trace;
    var phase = trace.meta && trace.meta.phase_roofline;
    if (!phase || !phase.points) {
      throw new Error('phase-roofline: this trace has no meta.phase_roofline block');
    }
    if (!NS.roofline) {
      throw new Error('phase-roofline: views/roofline.js must be loaded first — ' +
                      'this view renders through it rather than drawing its own chart');
    }

    /* The chart itself, drawn by the shared view. `rightmostId` is 'doc',
     * which with four orders of magnitude between the points is always the
     * rightmost dot: its label would otherwise run off the right edge. */
    var chart = NS.roofline.makeView({
      trace: { meta: { roofline: asRoofline(phase) } },
      width: cfg.width || 720,
      height: cfg.height || 380,
      rightmostId: 'doc'
    });

    /**
     * Re-anchor the labels that would leave the frame.
     *
     * WHY THIS IS HERE AND NOT IN roofline.js. That view can left-anchor exactly
     * ONE point (`cfg.rightmostId`), which was enough for L01: its three points
     * are spread over two decades of intensity, so only the doc dot is ever near
     * the right edge. L05's are not. A big model with a short prompt puts the
     * prefill dot at 360 FLOP/Byte against the reference's 493 — 1.4× apart on a
     * log axis, i.e. adjacent — and the prefill label, which runs to the RIGHT
     * of its dot, then leaves the viewBox and is silently clipped. That is a
     * real bug the acceptance harness caught at `kv-cache-N256-L4-int8`, and the
     * engine is frozen for this ticket, so the fix lives in the wrapper.
     *
     * HOW. The chart's own geometry is already in the markup it returns: each
     * point group carries its dot's `cx`, and the plot's width is a known
     * constant of the shared view. So a point whose dot sits in the right
     * quarter gets its label flipped to `text-anchor: end` at `cx - 9`, which is
     * exactly the treatment `rightmostId` gives the doc dot. No layout is
     * re-derived — the numbers come from the chart that was drawn.
     */
    function reanchorCrowdedLabels(html) {
      var PLOT = { l: 72, r: 26 };            // roofline.js's PAD, left/right
      var plotW = (cfg.width || 720) - PLOT.l - PLOT.r;
      /* The threshold is a quarter of the plot. A label placed to the right of
       * a dot whose centre is past 75% has at most 25% of the plot to run in,
       * and the AI labels are longer than that at any readable font size. */
      var cut = PLOT.l + plotW * 0.75;
      var flipped = [];
      var out = html.replace(
        /<g class="lab-roof-pt lab-roof-pt-([A-Za-z0-9_]+)"([\s\S]*?)<\/g>/g,
        function (whole, id, body) {
          var m = /<circle[^>]*\scx="([-\d.]+)"/.exec(body);
          if (!m || Number(m[1]) < cut) return whole;
          var flippedBody = body.replace(
            /(<text class="lab-roof-pt-l" x=")[-\d.]+(")/, '$1' + (Number(m[1]) - 9) + '$2');
          if (flippedBody === body) return whole;
          flippedBody = flippedBody.replace(
            /(<text class="lab-roof-pt-l"[^>]*text-anchor=")start(")/, '$1end$2');
          flipped.push(id);
          return '<g class="lab-roof-pt lab-roof-pt-' + id + '"' + flippedBody + '</g>';
        });
      return { html: out, flipped: flipped };
    }

    var byId = {};
    phase.points.forEach(function (p) { byId[p.id] = p; });
    var pre = byId.prefill, dec = byId.decode, doc = byId.doc;

    /**
     * The per-phase breakdown table.
     *
     * It exists because "Prefill 是 compute-bound、Decode 是 memory-bound" is
     * otherwise an assertion the reader has to take on faith. The terms that
     * produce the intensity are all published, so they are shown: the same
     * weight bytes in both rows, a KV read that is ~1/t of prefill's, and the
     * FLOP counts that differ by a factor of the prompt length. The conclusion
     * follows from the two columns.
     */
    function breakdown() {
      function row(p, name, note) {
        var pa = p.parts, ai = p.ai;
        return '<tr data-phase="' + esc(p.id) + '">' +
          '<th class="lab-pr-row"><b>' + esc(name) + '</b>' +
            '<span class="lab-pr-note">' + esc(note) + '</span></th>' +
          '<td>' + esc(g(p.flops)) + '</td>' +
          '<td>' + esc(g(pa.weights_bytes)) + '</td>' +
          '<td>' + esc(g(pa.kv_read_bytes + pa.kv_write_bytes)) + '</td>' +
          '<td><b>' + esc(g(ai)) + '</b></td>' +
          '<td class="lab-pr-b-' + esc(p.bound) + '">' +
            esc(p.bound === 'compute' ? '算力' : '带宽') + '</td>' +
          '</tr>';
      }
      return '<table class="lab-pr-table"><thead><tr>' +
          '<th>阶段</th><th>FLOPs</th><th>权重字节</th><th>KV 字节</th>' +
          '<th>算术强度</th><th>受限于</th>' +
        '</tr></thead><tbody>' +
        row(pre, 'Prefill', '整段 prompt 一次前向') +
        row(dec, 'Decode', '1 个 token 一次前向') +
        row(doc, '参考模型', '7B / 512 prompt') +
        '</tbody></table>';
    }

    function render() {
      var s = [];
      s.push('<div class="lab-pr" data-ratio="' +
             esc(g(pre.ai / Math.max(dec.ai, 1e-30))) + '">');
      s.push('<div class="lab-pr-hd">' +
        '<span class="lab-pr-hd-t">Prefill 与 Decode 落在同一条 Roofline 上</span>' +
        '<span class="lab-pr-hd-s">' + esc(phase.model) + ' · 平衡点 ridge = ' +
        esc(g(phase.ridge)) + ' FLOP/Byte</span></div>');
      var drawn = reanchorCrowdedLabels(chart.render());
      s.push('<div class="lab-pr-chart" data-flipped="' +
             esc(drawn.flipped.join(' ')) + '">' + drawn.html + '</div>');
      s.push(breakdown());
      s.push('<p class="lab-pr-why">' + esc(phase.traffic_note) + '</p>');
      s.push('<p class="lab-pr-why">' +
        '两次前向读的是<b>同一份权重</b>（' + esc(g(pre.parts.weights_bytes)) +
        ' 字节），但 Prefill 用它换了 ' + esc(g(pre.config.prompt)) + ' 个 token 的计算，' +
        'Decode 只换了 <b>1 个</b>：算术强度因此相差 ' +
        '<b>' + esc(g(pre.ai / Math.max(dec.ai, 1e-30))) + '×</b>。' +
        '这就是「Prefill 是 compute-bound、Decode 是 memory-bound」的全部算术 —— ' +
        '不是两种不同的运算，是同一个权重搬运被摊到了多少个 token 上。' +
        (pre.bound === 'bandwidth'
          ? ' 注意本配置的两个点都还落在<b>带宽屋顶</b>上：' +
            '这个玩具模型只有 ' + esc(g(pre.parts.weights_bytes)) +
            ' 字节权重，要像参考模型那样够到算力屋顶，' +
            'prompt 还得长得多，或者模型还得大得多 —— 图上把参考点画出来正是为了说这件事。'
          : '') +
        '</p>');
      s.push('</div>');
      return s.join('');
    }

    return { render: render, chart: chart, phase: phase };
  }

  /**
   * The player's `panels: [{id, label, render(step, ctx)}]` entry.
   *
   * Ignores the cursor, like the shared Roofline: the phases' placement is a
   * property of the configuration, not of the replay step.
   */
  function panel(config, context) {
    var view = makeView({ trace: context.trace });
    return {
      id: (config && config.id) || 'phase-roofline',
      label: (config && config.label) || 'Roofline：Prefill 与 Decode 的位置',
      render: function () { return view.render(); }
    };
  }

  /* ================================================================= lint
   *
   * The JS port of the `meta.phase_roofline` contract. Python
   * (`labs/traces/kv_cache.py`) is the authority and runs before a trace is
   * written; this exists so a lab author editing a trace in the browser gets
   * the same verdict without a Python round trip.
   *
   * THE TWO PORTS MUST AGREE. The acceptance harness runs one sabotage table
   * through both and requires the same verdict from each.
   *
   * WHAT IT DOES NOT DO: it does not verify that the hardware model is real.
   * A bandwidth and a peak are the pair the trace chose, and nothing in a
   * browser can tell a plausible pair from an invented one. What it checks is
   * that the chart it is about to draw is internally coherent — every number on
   * it re-derived from every other — and that the reference point is one that
   * does not move.
   */
  var PHASE_IDS = ['prefill', 'decode', 'doc'];

  function close(a, b, rel) {
    if (typeof a !== 'number' || typeof b !== 'number') return false;
    if (a === b) return true;
    var d = Math.abs(a - b);
    return d <= 1e-12 || d <= (rel === undefined ? 1e-12 : rel) * Math.max(Math.abs(a), Math.abs(b));
  }

  /**
   * The `meta.kv_crossover` contract. See this file's header for why it is
   * ported here rather than beside the ledger's own port.
   *
   * One rule with teeth, in two halves: the crossover must equal
   * `params_bytes / per_token_bytes`, AND both of those must equal the figures
   * the ledger block already publishes. Without the second half the page could
   * print a crossover computed from a different model than the bar drawn beside
   * it — which is exactly the failure this whole module exists to make
   * impossible.
   */
  function lintCrossover(trace) {
    var gaps = [];
    var cross = ((trace || {}).meta || {}).kv_crossover;
    if (typeof cross !== 'object' || cross === null) {
      gaps.push({ what: 'meta.kv_crossover 缺失',
                  why: '页面上「KV 何时反超参数」无处可查。' });
      return gaps;
    }
    var bad = ['params_bytes', 'per_token_bytes', 'context'].filter(function (k) {
      return typeof cross[k] !== 'number' || !isFinite(cross[k]) || cross[k] <= 0;
    });
    if (bad.length) {
      gaps.push({ what: 'meta.kv_crossover 的 ' + bad.join(' / ') + ' 不是正数',
                  why: '反超点是两个正数的商。' });
      return gaps;
    }
    if (!close(cross.context, cross.params_bytes / cross.per_token_bytes)) {
      gaps.push({ what: 'meta.kv_crossover.context = ' + JSON.stringify(cross.context) +
                        ' 应为 params_bytes / per_token_bytes = ' +
                        (cross.params_bytes / cross.per_token_bytes),
                  why: '反超点与它自己的两个输入矛盾。' });
    }
    var steps = ((trace || {}).steps || []).filter(function (s) { return s && s.ledger; });
    if (steps.length) {
      var led = steps[steps.length - 1].ledger;
      if (led.bytes && led.bytes.params !== cross.params_bytes) {
        gaps.push({ what: 'meta.kv_crossover.params_bytes = ' + cross.params_bytes +
                          ' 与末步账本的参数分项 ' + led.bytes.params + ' 不符',
                    why: '两处说的是同一个模型的参数。' });
      }
    }
    var cfg = ((trace || {}).meta || {}).config || {};
    var wantPerToken = 2 * (cfg.L || 0) * (cfg.H || 0) * (cfg.d_head || 0) *
                       (cfg.kv_bytes || 0);
    if (cross.per_token_bytes !== wantPerToken) {
      gaps.push({ what: 'meta.kv_crossover.per_token_bytes = ' +
                        cross.per_token_bytes + ' 应为 2·L·H_kv·d_h·b = ' + wantPerToken,
                  why: '每 token 的 KV 就是那条闭式，不能是另一个数。' });
    }
    return gaps;
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var meta = (trace && trace.meta) || {};
    var roof = meta.phase_roofline;

    /* The crossover's rules ride along with this port, but they are a SEPARATE
     * block: a trace may publish `kv_crossover` without a Roofline or the other
     * way round, so each is checked and reported on its own. */
    var crossGaps = lintCrossover(trace);
    if (!roof || typeof roof !== 'object') {
      gaps.push({ what: 'meta.phase_roofline 缺失',
                  why: 'Prefill/Decode 的 Roofline 没有数据。' });
      return { gaps: gaps.concat(crossGaps), warns: warns, infos: infos };
    }

    if (!roof || typeof roof !== 'object') {
      gaps.push({ what: 'meta.phase_roofline 缺失',
                  why: 'Prefill/Decode 的 Roofline 没有数据。' });
      return { gaps: gaps, warns: warns, infos: infos };
    }

    var REQUIRED = ['peak_flops', 'bandwidth', 'ridge', 'points', 'x_label', 'y_label',
                    'y_divisor', 'y_unit', 'note', 'traffic_note', 'model'];
    REQUIRED.forEach(function (k) {
      if (!(k in roof)) {
        gaps.push({ what: 'meta.phase_roofline 缺 ' + k,
                    why: '轴或说明会显示 undefined。' });
      }
    });

    var peak = roof.peak_flops, bw = roof.bandwidth;
    /* `haveModel` gates every check below that divides by or compares against
     * the ridge. A trace MISSING the ridge has already been reported once, in
     * the loop above; walking on regardless would turn that report into a NaN
     * comparison that silently passes — which reads as a broken lint rather
     * than as a broken trace. */
    var haveModel = typeof peak === 'number' && isFinite(peak) && peak > 0 &&
                    typeof bw === 'number' && isFinite(bw) && bw > 0 &&
                    typeof roof.ridge === 'number' && isFinite(roof.ridge);
    if (!haveModel) {
      return { gaps: gaps.concat(crossGaps), warns: warns, infos: infos };
    }

    if (!close(roof.ridge, peak / bw)) {
      gaps.push({ what: 'meta.phase_roofline.ridge = ' + JSON.stringify(roof.ridge) +
                        ' 应为峰值算力 / 带宽 = ' + (peak / bw),
                  why: '平衡点与硬件模型脱节 —— 图上的竖直参考线会在错误的位置。' });
    }
    var yd = roof.y_divisor;
    if (typeof yd !== 'number' || !isFinite(yd) || yd <= 0) {
      gaps.push({ what: 'meta.phase_roofline.y_divisor = ' + JSON.stringify(yd) + ' 应为正数',
                  why: '纵轴刻度按它换算。' });
    } else if (typeof roof.y_unit !== 'string' || !roof.y_unit) {
      gaps.push({ what: 'meta.phase_roofline.y_unit = ' + JSON.stringify(roof.y_unit) +
                        ' 应为非空字符串', why: '轴标题要写出单位。' });
    } else if (String(roof.y_label || '').indexOf(roof.y_unit) === -1) {
      gaps.push({ what: 'meta.phase_roofline.y_label 里没有写明单位 "' + roof.y_unit + '"',
                  why: '刻度按 ' + yd + ' 换算，读者无从知道单位是什么。' });
    }

    var pts = roof.points;
    var ids = Array.isArray(pts) ? pts.map(function (p) { return p && p.id; }) : [];
    if (ids.join(',') !== PHASE_IDS.join(',')) {
      gaps.push({ what: 'meta.phase_roofline.points 的 id 是 [' + ids.join(', ') +
                        ']，应该是 [' + PHASE_IDS.join(', ') + ']',
                  why: '点的顺序也参与渲染 —— 图例与 DOM 顺序按它排。' });
      return { gaps: gaps.concat(crossGaps), warns: warns, infos: infos };
    }

    pts.forEach(function (p) {
      var pid = p.id;
      var bs = p.bytes, ai = p.ai, pf = p.flops, parts = p.parts;
      if (typeof bs !== 'number' || !isFinite(bs) || bs <= 0 || bs !== Math.floor(bs)) {
        gaps.push({ what: 'meta.phase_roofline.points["' + pid + '"].bytes = ' +
                          JSON.stringify(bs) + ' 应为正整数',
                    why: '算术强度是按它算出来的。' });
        return;
      }
      var WANT_PARTS = ['proj_flops', 'ffn_flops', 'attn_flops', 'head_flops',
                        'weights_bytes', 'kv_read_bytes', 'kv_write_bytes'];
      if (typeof parts !== 'object' || parts === null ||
          WANT_PARTS.some(function (k) { return !(k in parts); })) {
        gaps.push({ what: 'meta.phase_roofline.points["' + pid + '"].parts 不完整',
                    why: '明细表的每一列都要有来源。' });
        return;
      }
      /* The total is its own parts, added up. One summary number cannot hide a
       * dropped term when the terms are published beside it and re-added. */
      var flopsFromParts = parts.proj_flops + parts.ffn_flops + parts.attn_flops +
                           parts.head_flops;
      var bytesFromParts = parts.weights_bytes + parts.kv_read_bytes + parts.kv_write_bytes;
      if (pf !== flopsFromParts) {
        gaps.push({ what: 'meta.phase_roofline.points["' + pid + '"].flops = ' +
                          JSON.stringify(pf) + ' 与各分项之和 ' + flopsFromParts + ' 不符',
                    why: '页面上的合计与它自己的明细是两套说法。' });
      }
      if (bs !== bytesFromParts) {
        gaps.push({ what: 'meta.phase_roofline.points["' + pid + '"].bytes = ' +
                          JSON.stringify(bs) + ' 与各分项之和 ' + bytesFromParts + ' 不符',
                    why: '访存总量与它自己的明细是两套说法。' });
      }
      /* The point's intensity must be its OWN flops over its OWN bytes. */
      if (!close(ai, pf / bs)) {
        gaps.push({ what: 'meta.phase_roofline.points["' + pid + '"].ai = ' +
                          JSON.stringify(ai) + ' 与它自己的 flops / bytes = ' + (pf / bs) +
                          ' 不一致',
                    why: '图上的点和它标称的访存量是两套说法。' });
        return;
      }
      var wantBound = ai < roof.ridge ? 'bandwidth' : 'compute';
      if (p.bound !== wantBound) {
        gaps.push({ what: 'meta.phase_roofline.points["' + pid + '"].bound = ' +
                          JSON.stringify(p.bound) + '，而在 AI = ' + ai +
                          ' 上起作用的是 "' + wantBound + '" 的屋顶',
                    why: '页面上「受限于哪条屋顶」的结论会与图矛盾。' });
      }
      var wantCeiling = Math.min(peak, bw * ai);
      if (!close(p.ceiling, wantCeiling)) {
        gaps.push({ what: 'meta.phase_roofline.points["' + pid + '"].ceiling = ' +
                          JSON.stringify(p.ceiling) + ' 应为 min(峰值, 带宽×AI) = ' +
                          wantCeiling,
                    why: '点的纵坐标不是 Roofline 给出的那个上界。' });
      }
      if (typeof p.config !== 'object' || p.config === null) {
        gaps.push({ what: 'meta.phase_roofline.points["' + pid + '"].config 缺失',
                    why: '点的模型参数没有声明，无法判断它是不是本配置。' });
      }
    });

    /* The live points describe THIS configuration; the reference point is a
     * different model and must be the SAME dot on every configuration, because
     * that stillness is what the two live points move against. */
    var byId = {};
    pts.forEach(function (p) { byId[p.id] = p; });
    var cfg = meta.config || {};
    var preCfg = (byId.prefill.config || {});
    if (preCfg.n_q !== cfg.prompt_len || preCfg.t !== cfg.prompt_len ||
        preCfg.prompt !== cfg.prompt_len) {
      gaps.push({ what: 'meta.phase_roofline 的 prefill 点用了 n_q=' + preCfg.n_q +
                        ', t=' + preCfg.t + ', prompt=' + preCfg.prompt +
                        '，而本配置的 prompt 是 ' + cfg.prompt_len,
                  why: 'Prefill 点必须描述这次回放真的跑过的 prompt。' });
    }
    var decCfg = (byId.decode.config || {});
    if (decCfg.n_q !== 1 || decCfg.t !== cfg.N) {
      gaps.push({ what: 'meta.phase_roofline 的 decode 点用了 n_q=' + decCfg.n_q +
                        ', t=' + decCfg.t + '，而本配置的 decode 是 1 个 token 对 ' +
                        cfg.N + ' 个位置',
                  why: 'Decode 点必须描述这次回放真的跑过的那一步。' });
    }
    ['prefill', 'decode'].forEach(function (pid) {
      var c = byId[pid].config || {};
      if (c.layers !== cfg.L || c.dtype !== cfg.dtype) {
        gaps.push({ what: 'meta.phase_roofline 的 ' + pid + ' 点与本配置的层数/精度不一致：' +
                          JSON.stringify(c) + ' vs L=' + cfg.L + ', dtype=' + cfg.dtype,
                    why: '同一个配置的两个说法互相矛盾。' });
      }
    });
    var docCfg = (byId.doc.config || {});
    if (docCfg.layers !== 32 || docCfg.t !== 512 || docCfg.prompt !== 512 ||
        docCfg.dtype !== 'fp16') {
      gaps.push({ what: 'meta.phase_roofline 的 doc 参考点随配置变了：' +
                        JSON.stringify(docCfg) + ' 应为 {layers: 32, t: 512, ' +
                        'prompt: 512, dtype: "fp16"}',
                  why: '参考点必须每个配置都一样 —— 那是对照组。' });
    }

    /* The lab's headline claim, as a relation between the two live points. */
    if (byId.decode.ai > byId.prefill.ai + 1e-12) {
      gaps.push({ what: 'meta.phase_roofline：decode 的 AI=' + byId.decode.ai +
                        ' 高于 prefill 的 ' + byId.prefill.ai,
                  why: '权重复用恰恰让 prefill 的算术强度更高，反过来就不对了。' });
    }
    if (typeof roof.traffic_note !== 'string' || !roof.traffic_note) {
      gaps.push({ what: 'meta.phase_roofline.traffic_note 缺失',
                  why: '访存口径没有声明，图上的数字无从解释。' });
    }

    infos.push({ what: '三个阶段点 · ridge ' + g(roof.ridge) + ' · prefill/decode AI 比 ' +
                        g(byId.prefill.ai / Math.max(byId.decode.ai, 1e-30)),
                 why: '' });
    return { gaps: gaps.concat(crossGaps), warns: warns, infos: infos };
  }

  /* Each entry breaks one rule above, on a real trace. The names parallel the
   * Python `SABOTAGE_CASES` in `labs/traces/kv_cache.py`. */
  var SABOTAGE_CASES = {
    'meta.phase_roofline 被删掉': function (t) { delete t.meta.phase_roofline; },
    'ridge 不等于 峰值/带宽': function (t) { t.meta.phase_roofline.ridge = 5.0; },
    '硬件模型与生成器不一致': function (t) { t.meta.phase_roofline.peak_flops = 1.0; },
    '点顺序被打乱': function (t) {
      t.meta.phase_roofline.points = t.meta.phase_roofline.points.slice().reverse();
    },
    'ai 与自己点的 flops/bytes 不符': function (t) {
      t.meta.phase_roofline.points[0].ai = 1.0;
    },
    'ceiling 不是 min(峰值, 带宽×ai)': function (t) {
      t.meta.phase_roofline.points[1].ceiling = 1.0;
    },
    '点归属到错误的屋顶': function (t) {
      t.meta.phase_roofline.points[1].bound = 'compute';
    },
    '合计与分项之和不符': function (t) {
      t.meta.phase_roofline.points[0].parts.ffn_flops = 1;
    },
    '缺少一个访存分项': function (t) {
      delete t.meta.phase_roofline.points[0].parts.kv_read_bytes;
    },
    'y 轴除数缺失': function (t) { delete t.meta.phase_roofline.y_divisor; },
    'y 轴单位没有写进标题': function (t) {
      t.meta.phase_roofline.y_label = '可达算力上界';
    },
    '访存口径说明缺失': function (t) { delete t.meta.phase_roofline.traffic_note; },
    'decode 点被绑到了参考模型': function (t) {
      t.meta.phase_roofline.points[1].config.t = 512;
    },
    'prefill 点没有用本配置的 prompt': function (t) {
      t.meta.phase_roofline.points[0].config.n_q = 1;
    },
    'prefill 点没有用本配置的层数': function (t) {
      t.meta.phase_roofline.points[0].config.layers = 64;
    },
    '参考点随配置移动': function (t) {
      t.meta.phase_roofline.points[2].config.layers = t.meta.config.L;
      t.meta.phase_roofline.points[2].config.t = t.meta.config.N;
    },
    '参考点跟着精度滑杆走': function (t) {
      t.meta.phase_roofline.points[2].config.dtype = 'fp8';
    },
    'decode 的强度高于 prefill': function (t) {
      t.meta.phase_roofline.points[1].ai = t.meta.phase_roofline.points[0].ai * 2;
    },
    // -- meta.kv_crossover, ported by this file (see its header)
    'crossover 块被删掉': function (t) { delete t.meta.kv_crossover; },
    'crossover 不等于 params / per_token': function (t) {
      t.meta.kv_crossover.context = 1.0;
    },
    'crossover 用了别的参数个数': function (t) {
      t.meta.kv_crossover.params_bytes += 2;
    },
    'crossover 的每 token 成本不是 2LHd_hb': function (t) {
      t.meta.kv_crossover.per_token_bytes = 8;
    },
    '点的 config 缺字段': function (t) {
      delete t.meta.phase_roofline.points[0].config.prompt;
    }
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

  NS.phaseRoofline = {
    makeView: makeView,
    panel: panel,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES,
    asRoofline: asRoofline,
    /* The number formatter, exported so the acceptance harness can render a
     * trace value into the string this view WOULD print and compare it against
     * the string it DID print. Re-implementing it in Python is what the first
     * version of that harness did, and it drifted on the first value with a
     * trailing zero — the whole point of comparing against the DOM is that the
     * number is the thing under test, not the formatting. */
    format: g
  };
})(typeof window !== 'undefined' ? window : globalThis);
