/* Engine · view component: Pre-Norm vs Post-Norm, as a measurement.
 *
 * L04's fourth teaching point — "Pre-Norm 与 Post-Norm 的对照能看出数值尺度上的
 * 差异" — and the material for it is the one piece of ready-made contrast in the
 * whole corpus: `3.6-LayerNorm与残差连接深入理解.md` §4.1 has two mermaid graphs
 * whose node names are IDENTICAL and whose only difference is where the
 * LayerNorm sits.
 *
 * The graphs are drawn from the trace's `meta.norm_contrast` flow lists, which
 * are the mermaid pair converted by hand (the reasoning is in
 * `labs/traces/README.md`; R01's finding was that parsing mermaid pays off for
 * topology and not for a two-line contrast, and this is a two-line contrast).
 * Converting them by hand is only defensible if the data is still MEASURED, so
 * the picture is not the argument — the two curves beside it are:
 *
 *   `residual_rms`   the RMS of the residual stream after every block, to depth
 *                    32, under both placements. Post-Norm holds at 1.0000 — not
 *                    because it is well behaved but because that is what a
 *                    LayerNorm output IS. That flat line is the control which
 *                    makes Pre-Norm's climb readable as a climb.
 *   `grad_norm`      ‖∂L/∂(residual entering block k)‖ for a fixed probe
 *                    direction, i.e. the number the tutorial's §4.2 formulas
 *                    are about. Pre-Norm's comes back amplified by tens;
 *                    Post-Norm's by hundreds. The generator re-runs the whole
 *                    sweep on two probe seeds the trace does not publish and
 *                    requires the ordering to survive, so this is a property of
 *                    the architecture rather than of one draw.
 *
 * WHY BOTH CURVES
 * A reader who only saw the gradient curve would read it as "Post-Norm is
 * broken". The forward curve is the other half of the same fact, and it is the
 * one the tutorial's §4.3 cites from Xiong et al.: Post-Norm's clean 1.0 is
 * bought by holding the residual stream down, and it is exactly the x that
 * Pre-Norm lets grow. Showing one without the other is the misreading this view
 * exists to prevent.
 *
 * THE NUMBERS ARE NOT COMPUTED HERE
 * Both series arrive in the trace, measured in torch autograd. This file draws
 * them, scales them, and refuses to draw a trace whose series do not carry the
 * contrast the page claims — which is what `lint()` is for.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;

  function num(v, digits) {
    if (typeof v !== 'number') return String(v);
    var s = v.toPrecision(digits || 4);
    return s.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  }

  /* The flow, drawn as boxes and arrows from a list of node names.
   *
   * A list, not a graph: `3.6 §4.1`'s two mermaid diagrams are both linear
   * chains, and the difference between them is the index of one element. The
   * `x` node is drawn as the branch point, because that is what makes the
   * residual path visible — it appears once as the input and once as the thing
   * that rejoins at `Add`. */
  function drawFlow(flow, cls, label, formula) {
    var html = '<div class="lab-nc-flow ' + cls + '">' +
      '<span class="lab-nc-flow-t">' + esc(label) + '</span>' +
      '<span class="lab-nc-fml">' + (NS.formula ? NS.formula.renderInline(formula) : esc(formula)) +
      '</span>' +
      '<div class="lab-nc-chain">';
    flow.forEach(function (node, i) {
      var kind = node === 'LN' ? 'ln' : node === 'Add' ? 'add'
        : node === 'x' ? 'in' : node === 'SubLayer' ? 'sub' : 'node';
      html += '<span class="lab-nc-node lab-nc-' + kind + '"' +
        ' data-node="' + esc(node) + '" data-index="' + i + '">' + esc(node) + '</span>';
      if (i + 1 < flow.length) html += '<span class="lab-nc-arrow">→</span>';
    });
    html += '</div></div>';
    return html;
  }

  /* One series as an inline SVG polyline. The y axis is shared between the two
   * placements of a panel — deliberately, because the comparison IS the
   * difference in height, and two independently scaled charts would show two
   * similar-looking curves. */
  function drawSeries(series, mode, cls, yMax, w, h) {
    var pad = { l: 6, r: 6, t: 6, b: 6 };
    var n = series.length;
    var innerW = w - pad.l - pad.r;
    var innerH = h - pad.t - pad.b;
    var pts = series.map(function (v, i) {
      var x = pad.l + (n === 1 ? innerW / 2 : innerW * i / (n - 1));
      var y = pad.t + innerH - innerH * (v / yMax);
      return [x, y];
    });
    var d = pts.map(function (p, i) {
      return (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1);
    }).join(' ');
    var html = '<svg class="lab-nc-svg ' + cls + '" viewBox="0 0 ' + w + ' ' + h + '"' +
      ' preserveAspectRatio="none" data-mode="' + esc(mode) + '"' +
      ' data-max="' + esc(String(yMax)) + '"' +
      ' data-min="' + esc(String(Math.min.apply(null, series))) + '"' +
      ' data-last="' + esc(String(series[n - 1])) + '"' +
      ' data-first="' + esc(String(series[0])) + '"' +
      ' width="' + w + '" height="' + h + '">' +
      '<path class="lab-nc-line" d="' + d + '" />' +
      pts.map(function (p, i) {
        return '<circle class="lab-nc-dot" cx="' + p[0].toFixed(1) + '" cy="' +
          p[1].toFixed(1) + '" r="2.6" data-depth="' + series.length + '"' +
          ' data-i="' + i + '"><title>深度 ' + (i + 1) + '：' + num(series[i], 4) +
          '</title></circle>';
      }).join('') +
      '</svg>';
    return html;
  }

  function makeView(config) {
    var cfg = config || {};
    var W = cfg.chartWidth || 300;
    var H = cfg.chartHeight || 120;

    /* `step` is unused and MUST STAY IN THE SIGNATURE: `player.js` calls every
     * panel's `render(step, ctx)` positionally, so dropping the first parameter
     * would silently make `ctx` the step and the trace lookup would find
     * nothing. This panel is trace-level by design — the Pre/Post contrast does
     * not change as the cursor moves — which is what `lint()`'s gate says too. */
    function render(step, ctx) {
      var trace = (ctx && ctx.trace) || (cfg.trace) || null;
      var nc = trace && trace.meta && trace.meta.norm_contrast;
      if (!nc) {
        return '<div class="lab-nc-none">本页的 trace 没有带 Pre/Post-Norm 对照数据。</div>';
      }
      var modes = nc.modes || [];
      var depths = nc.depths || [];
      var pre = modes[0], post = modes[1];

      /* The shared y scale for each chart, over BOTH placements — see
       * `drawSeries` above for why they may not be scaled separately. */
      function maxOf(key) {
        var vals = [];
        modes.forEach(function (m) {
          (m[key] || []).forEach(function (v) { vals.push(v); });
        });
        return Math.max.apply(null, vals.concat([1e-12])) * 1.08;
      }
      var rmsMax = maxOf('residual_rms');
      var gradMax = maxOf('grad_amp_log');
      /* The gradient chart's floor is 0 (log10 of "no amplification"), not the
       * series minimum: a curve that never rises above 1× should sit ON the
       * baseline, and scaling from the data's own minimum would lift it into
       * the middle of the frame and make a flat line look like a result. */
      gradMax = Math.max(gradMax, 0.1);

      var html = '<div class="lab-nc" data-depths="' + depths.length + '">';

      html += '<div class="lab-nc-hd">' +
        '<span class="lab-nc-hd-t">同一组节点名，只有 LayerNorm 的位置不同</span>' +
        '<span class="lab-nc-hd-s">' + esc(nc.difference && nc.difference.note || '') +
        '</span></div>';

      html += '<div class="lab-nc-flows">' +
        drawFlow(pre.flow, 'lab-nc-pre', pre.label, pre.formula) +
        drawFlow(post.flow, 'lab-nc-post', post.label, post.formula) +
        '</div>';

      /* --- the two measurements. */
      html += '<div class="lab-nc-charts">';

      html += '<div class="lab-nc-chart" data-chart="rms">' +
        '<span class="lab-nc-chart-t">残差流的 RMS，随深度（1 → ' +
        depths[depths.length - 1] + ' 层）</span>' +
        '<div class="lab-nc-series">' +
        modes.map(function (m) {
          return '<div class="lab-nc-row" data-mode="' + esc(m.id) + '"' +
            ' data-first="' + esc(String(m.residual_rms[0])) + '"' +
            ' data-last="' + esc(String(m.residual_rms[m.residual_rms.length - 1])) + '">' +
            '<span class="lab-nc-row-t">' + esc(m.label) + '</span>' +
            drawSeries(m.residual_rms, m.id, 'lab-nc-s-' + m.id, rmsMax, W, H) +
            '<span class="lab-nc-row-v">' + esc(num(m.residual_rms[0], 4)) + ' → <b>' +
              esc(num(m.residual_rms[m.residual_rms.length - 1], 4)) + '</b></span>' +
            '</div>';
        }).join('') +
        '</div>' +
        '<p class="lab-nc-note">Post-Norm 那条恒等于 1.0000 —— 这不是它「稳定」，' +
        '而是 LayerNorm 的输出本来就该是单位方差。这是个对照组：' +
        'Pre-Norm 那条一路涨上去，正是它的残差流<b>没有</b>被归一化钳住。' +
        '两边共用一根纵轴，所以高度差是真的差。</p>' +
        '</div>';

      html += '<div class="lab-nc-chart" data-chart="grad">' +
        '<span class="lab-nc-chart-t">梯度回传到每个 block 边界时的放大倍数（log₁₀，' +
        '深度 ' + depths[depths.length - 1] + ' 层）</span>' +
        '<div class="lab-nc-series">' +
        modes.map(function (m) {
          return '<div class="lab-nc-row" data-mode="' + esc(m.id) + '"' +
            ' data-in-over-out="' + esc(String(m.grad_in_over_out)) + '">' +
            '<span class="lab-nc-row-t">' + esc(m.label) + '</span>' +
            drawSeries(m.grad_amp_log, m.id, 'lab-nc-g-' + m.id, gradMax, W, H) +
            '<span class="lab-nc-row-v">放大 <b>' +
              esc(num(m.grad_in_over_out, 4)) + '×</b></span>' +
            '</div>';
        }).join('') +
        '</div>' +
        '<p class="lab-nc-note">这是 ' +
        (NS.formula ? NS.formula.renderInline((nc.probe || {}).loss || '') : '') +
        ' 对每个 block 入口的梯度范数，' + esc((nc.probe || {}).direction || '') +
        '；纵轴是最外层边界与该处之比取 log₁₀，所以线上每高 1 就是 10 倍。' +
        'Pre-Norm 的残差路径上没有任何归一化，梯度可以直接穿回去；' +
        'Post-Norm 每经过一个 block 都要被 LayerNorm 的雅可比再调一次，' +
        '越靠浅层放大得越厉害。这就是 ' +
        '<code>3.6 §4.2</code> 那两个乘积公式的差别，量出来的 —— ' +
        '不是「看起来更陡」，是同一批权重、同一个探针方向下差了几十倍。</p>' +
        '</div>';

      html += '</div>';   /* /charts */

      html += '<p class="lab-nc-sum">' +
        '<b>' + esc(pre.label) + ' ' + esc(num(pre.grad_in_over_out, 3)) +
        '× vs ' + esc(post.label) + ' ' + esc(num(post.grad_in_over_out, 3)) +
        '×。</b>两张图说的是同一件事的两面：Post-Norm 用每次归一化换来了干净的 ' +
        '1.0 残差流，代价是梯度回传多穿了几层雅可比；Pre-Norm 让残差流自由增长，' +
        '换来一条未经调制的回传路径。层数越多，这个差别越大 —— ' +
        '这就是 96 层的 GPT-3 不用 Post-Norm 的原因。' +
        '</p>';

      html += '</div>';
      return html;
    }

    return { render: render };
  }

  function panel(config, context) {
    var view = makeView({ trace: context && context.trace });
    return {
      id: (config && config.id) || 'norm-contrast',
      label: (config && config.label) || 'Pre-Norm vs Post-Norm：残差尺度与梯度回传',
      render: function (step, ctx) {
        return view.render(step, { trace: (ctx && ctx.trace) || (context && context.trace) });
      }
    };
  }

  /* ================================================================= lint
   *
   * The JS port of the `meta.norm_contrast` contract. `labs/traces/decoder_block.py`
   * is the authority; this is the port beside the view, and THE TWO MUST AGREE.
   *
   * NOTE THE GATING DIRECTION. This block is trace-level rather than per-step,
   * so its gate is the block itself: a trace with no `meta.norm_contrast` and no
   * LayerNorm steps lints clean. A trace that HAS LayerNorm steps and no
   * contrast block is a gap — that is the arrangement `roofline.js` should have
   * had (see the debt recorded in `labs/traces/README.md`) and the one this view
   * copies from `tiling-stage.js`.
   */
  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var steps = (trace && trace.steps) || [];
    var nc = ((trace || {}).meta || {}).norm_contrast;
    var withLn = steps.filter(function (s) { return s && s.ln !== undefined; });

    if (nc === undefined) {
      if (withLn.length) {
        gaps.push({ what: 'trace 有 LayerNorm 步骤，但 meta.norm_contrast 缺失',
                    why: 'Pre/Post 的对照无处可查 —— 这正是本 lab 的第四个验收点。' });
      }
      return { gaps: gaps, warns: warns, infos: infos, present: false };
    }

    var depths = nc.depths;
    if (!Array.isArray(depths) || depths.length < 2) {
      gaps.push({ what: 'meta.norm_contrast.depths 缺失或不足两个点',
                  why: '一个点画不出趋势，「随深度增长」就不成立。' });
      return { gaps: gaps, warns: warns, infos: infos, present: true };
    }
    var modes = nc.modes;
    if (!Array.isArray(modes) || modes.length !== 2) {
      gaps.push({ what: 'meta.norm_contrast.modes 不是两条',
                  why: '对照要有两侧。' });
      return { gaps: gaps, warns: warns, infos: infos, present: true };
    }
    if (modes[0].id !== 'pre' || modes[1].id !== 'post') {
      gaps.push({ what: 'meta.norm_contrast.modes 的顺序是 [' +
                        modes.map(function (m) { return m.id; }).join(', ') +
                        ']，应为 [pre, post]',
                  why: '顺序也参与渲染。' });
    }

    var flows = [];
    modes.forEach(function (m) {
      var flow = m.flow;
      if (!Array.isArray(flow) || !flow.length) {
        gaps.push({ what: 'meta.norm_contrast["' + m.id + '"] 缺 flow',
                    why: '数据流画不出来。' });
        return;
      }
      flows.push(flow);
      if (flow.indexOf('LN') === -1) {
        gaps.push({ what: 'meta.norm_contrast["' + m.id + '"].flow 里没有 LN',
                    why: '这个 block 讲的就是它的位置。' });
      }
      ['residual_rms', 'grad_norm'].forEach(function (key) {
        var series = m[key];
        if (!Array.isArray(series) || series.length !== depths.length) {
          gaps.push({ what: 'meta.norm_contrast["' + m.id + '"].' + key + ' 有 ' +
                            (Array.isArray(series) ? series.length : '?') +
                            ' 项，应为 ' + depths.length + '（每个深度一项）',
                      why: '横轴对不齐，曲线画不全。' });
        }
      });
      if (Array.isArray(m.grad_norm)) {
        for (var i = 0; i < m.grad_norm.length; i++) {
          if (!Array.isArray(m.grad_norm[i]) || m.grad_norm[i].length !== depths[i]) {
            gaps.push({ what: 'meta.norm_contrast["' + m.id + '"].grad_norm[' + i +
                              ']（深度 ' + depths[i] + '）的边界数不对',
                        why: '每个 block 边界一个梯度范数。' });
            break;
          }
        }
      }
      if (!(typeof m.grad_in_over_out === 'number' && m.grad_in_over_out > 0)) {
        gaps.push({ what: 'meta.norm_contrast["' + m.id + '"].grad_in_over_out = ' +
                          JSON.stringify(m.grad_in_over_out),
                    why: '结论行上会显示一个不成比例的数。' });
      }
      /* The charted series, which is the ratio in log10 — checked against the
       * ratio it claims to be, because the two are published side by side and
       * a page that printed one while drawing the other would be two
       * statements about one quantity. */
      var amp = m.grad_amp_log;
      if (!Array.isArray(amp) || amp.length !== depths.length) {
        gaps.push({ what: 'meta.norm_contrast["' + m.id + '"].grad_amp_log 有 ' +
                          (Array.isArray(amp) ? amp.length : '?') + ' 项，应为 ' +
                          depths.length,
                    why: '梯度那张图的纵轴画不出来。' });
      } else if (amp.some(function (v) { return typeof v !== 'number' || !isFinite(v); })) {
        gaps.push({ what: 'meta.norm_contrast["' + m.id + '"].grad_amp_log 含非数',
                    why: '裸 NaN / Infinity 是非法 JSON。' });
      } else if (typeof m.grad_in_over_out === 'number' && m.grad_in_over_out > 0) {
        var want = Math.log(m.grad_in_over_out) / Math.LN10;
        if (Math.abs(amp[amp.length - 1] - want) > 1e-6 * Math.max(1, Math.abs(want))) {
          gaps.push({ what: 'meta.norm_contrast["' + m.id + '"].grad_amp_log 最深处 ' +
                            amp[amp.length - 1] + '，而 grad_in_over_out = ' +
                            m.grad_in_over_out + ' 对应 ' + want,
                      why: '画出来的曲线和结论行的数字不是同一件事。' });
        }
        if (amp[0] < -1e-6) {
          gaps.push({ what: 'meta.norm_contrast["' + m.id + '"].grad_amp_log[0] = ' +
                            amp[0] + ' 是负的',
                      why: '最外层边界相对自己的放大倍数必然是 1（log 为 0）。' });
        }
      }
    });

    if (flows.length === 2) {
      /* The `#40` claim, enforced: same node set, norm at a different index,
       * and nothing else different. Position-by-position comparison is wrong
       * here — moving one element shifts everything between its old and new
       * index — so the test is "strip the norm from both and the remainders
       * match". */
      var a = flows[0], b = flows[1];
      if (a.slice().sort().join(',') !== b.slice().sort().join(',')) {
        gaps.push({ what: '两条数据流的节点集合不同：' + a.join(' → ') + ' vs ' +
                          b.join(' → '),
                    why: '「只有 LN 位置不同」描述的是同一个 block。' });
      } else if (a.indexOf('LN') === b.indexOf('LN')) {
        gaps.push({ what: '两条数据流里 LN 都在第 ' + a.indexOf('LN') + ' 位',
                    why: 'Pre/Post-Norm 的差别就是它的位置。' });
      } else {
        var ra = a.filter(function (x) { return x !== 'LN'; });
        var rb = b.filter(function (x) { return x !== 'LN'; });
        if (ra.join(',') !== rb.join(',')) {
          gaps.push({ what: '除 LN 之外两条数据流也不一样：' + ra.join(' → ') + ' vs ' +
                            rb.join(' → '),
                      why: '节点名应该完全相同。' });
        }
        var at = (nc.difference || {}).at_index;
        if (at !== b.indexOf('LN')) {
          gaps.push({ what: 'meta.norm_contrast.difference.at_index = ' + at +
                            '，Post-Norm 的 LN 实际在第 ' + b.indexOf('LN') + ' 位',
                    why: '图上标注的位置会指错。' });
        }
        if ((nc.difference || {}).only !== 'LN') {
          gaps.push({ what: 'meta.norm_contrast.difference.only = ' +
                            JSON.stringify((nc.difference || {}).only),
                      why: '这一处差别就是 LN。' });
        }
      }
    }

    if (modes.length === 2) {
      var pre = modes[0], post = modes[1];
      /* The ordering is the lesson. Two identical series would render two
       * overlapping curves and teach nothing. */
      if (!(post.grad_in_over_out > pre.grad_in_over_out)) {
        gaps.push({ what: 'Post-Norm 的梯度放大 ' + post.grad_in_over_out +
                          ' 没有大于 Pre-Norm 的 ' + pre.grad_in_over_out,
                    why: '这正是这张图要展示的差别，反过来就不成立了。' });
      }
      if (Array.isArray(pre.residual_rms) && pre.residual_rms.length > 1) {
        for (var i = 1; i < pre.residual_rms.length; i++) {
          if (!(pre.residual_rms[i] > pre.residual_rms[i - 1])) {
            gaps.push({ what: 'Pre-Norm 的残差流 RMS 不是随深度单调上升（第 ' + i +
                              ' 项 ' + pre.residual_rms[i - 1] + ' → ' +
                              pre.residual_rms[i] + '）',
                        why: '「随深度增长」是这条曲线的全部内容。' });
            break;
          }
        }
      }
      if (Array.isArray(post.residual_rms) && post.residual_rms.length) {
        var flat = post.residual_rms.every(function (v) {
          return Math.abs(v - post.residual_rms[0]) < 1e-3;
        });
        if (!flat) {
          gaps.push({ what: 'Post-Norm 的残差流 RMS 没有保持恒定：' +
                            post.residual_rms.slice(0, 4).join(', ') + '…',
                    why: 'LN 的输出本来就是单位方差 —— 这是那张图的对照组。' });
        }
      }
    }

    var probe = nc.probe || {};
    if (!probe.loss || !probe.direction) {
      gaps.push({ what: 'meta.norm_contrast.probe 缺 loss / direction',
                  why: '梯度这个数是量出来的，得说清量的是什么，否则读者只能当它是画出来的。' });
    }
    if ((nc.order || {}).grad !== 'post>pre') {
      gaps.push({ what: 'meta.norm_contrast.order.grad = ' +
                        JSON.stringify((nc.order || {}).grad),
                  why: '页面上的结论行与它不一致。' });
    }
    infos.push({ what: 'Pre/Post 对照：深度 1…' + depths[depths.length - 1] +
                        '，两条数据流仅 LN 位置不同', why: '' });
    return { gaps: gaps, warns: warns, infos: infos, present: true };
  }

  var SABOTAGE_CASES = {
    'contrast 块被整个删掉': function (t) { delete t.meta.norm_contrast; },
    'contrast 只发布了一条': function (t) { t.meta.norm_contrast.modes.pop(); },
    'contrast 两条的顺序反了': function (t) { t.meta.norm_contrast.modes.reverse(); },
    'contrast 两条数据流完全相同': function (t) {
      t.meta.norm_contrast.modes[1].flow = t.meta.norm_contrast.modes[0].flow.slice();
    },
    'contrast 两条数据流差在多处': function (t) {
      t.meta.norm_contrast.modes[1].flow = ['x', 'Add', 'SubLayer', 'LN'];
    },
    'contrast 唯一那处差别不是 LN': function (t) {
      t.meta.norm_contrast.modes[1].flow = ['x', 'LN', 'Add', 'SubLayer'];
    },
    'contrast at_index 与数据对不上': function (t) {
      t.meta.norm_contrast.difference.at_index = 0;
    },
    'contrast only 不是 LN': function (t) {
      t.meta.norm_contrast.difference.only = 'Add';
    },
    'contrast 一条流少了 LN 节点': function (t) {
      var f = t.meta.norm_contrast.modes[0].flow;
      f.splice(f.indexOf('LN'), 1);
    },
    'contrast 曲线少一个深度': function (t) {
      t.meta.norm_contrast.modes[0].residual_rms.pop();
    },
    'contrast 梯度少一个边界': function (t) {
      t.meta.norm_contrast.modes[0].grad_norm[0].pop();
    },
    'contrast Post 的梯度放大不再更大': function (t) {
      t.meta.norm_contrast.modes[1].grad_in_over_out = 1;
    },
    'contrast Pre 的残差流不再增长': function (t) {
      var n = t.meta.norm_contrast.depths.length;
      t.meta.norm_contrast.modes[0].residual_rms = new Array(n).fill(1);
    },
    'contrast Post 的残差流不再恒定': function (t) {
      t.meta.norm_contrast.modes[1].residual_rms[0] = 9;
    },
    'contrast probe 说明被删': function (t) {
      delete t.meta.norm_contrast.probe.loss;
    },
    'contrast probe 方向被删': function (t) {
      delete t.meta.norm_contrast.probe.direction;
    },
    'contrast 声明的顺序反了': function (t) {
      t.meta.norm_contrast.order = { grad: 'pre>post' };
    },
    'contrast 深度列表不足两点': function (t) {
      t.meta.norm_contrast.depths = [1];
    }
  };

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
      if (JSON.stringify(copy) === JSON.stringify(trace)) {
        missed.push(name + '（空操作：trace 没变）');
        return;
      }
      var r = lint(copy);
      if (!r.gaps.length && !r.warns.length) missed.push(name);
    });
    return missed;
  }

  NS.normContrast = {
    makeView: makeView,
    panel: panel,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES
  };
})(typeof window !== 'undefined' ? window : globalThis);
