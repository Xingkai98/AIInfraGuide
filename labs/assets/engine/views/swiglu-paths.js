/* Engine · view component: the three SwiGLU paths and the elementwise gate.
 *
 * L04's third teaching point: "SwiGLU 的 gate/up/down 三条路径与逐元素乘在回放
 * 中可见，维度全程标注", for
 *
 *     SwiGLU(x) = W_down( SiLU(W_gate x) ⊙ W_up x )
 *
 * HOW "维度全程标注" IS SATISFIED
 * Not by printing a shape next to each box, which a reader would have to trust.
 * Every hop in the diagram is drawn with the shape the trace published for it,
 * and each hop is a multiplication whose operands are on screen: `[N, d] ×
 * [d, d_ff] → [N, d_ff]` for gate and up, `⊙` between two `[N, d_ff]`, then
 * `[N, d_ff] × [d_ff, d] → [N, d]`. The reader can check the inner dimensions
 * cancel at every step, which is what "the dimensions are labelled" is for.
 *
 * THE THREE PATHS, AND WHICH IS WHICH
 * `gate` goes through SiLU, `up` does not, and they must be the SAME shape
 * because `⊙` multiplies them position by position. `down` is what brings the
 * result back to `d_model` — the step that makes the FFN's output addable to the
 * residual stream again. The view highlights whichever path the current step is
 * on, and shows all three shapes at every step, so a reader who jumps straight
 * to the multiply can still see where its operands came from.
 *
 * THE ⊙ IS SHOWN, NOT ASSERTED
 * The multiply step carries one cell of the real matrices — its row and column,
 * both operands, and the product — published by the generator from the arrays
 * it actually computed. The view draws them as an equation with a checkmark on
 * it, and `lint()` re-derives `a × b = product` from those three numbers. A
 * picture of an elementwise multiply that did not show an element being
 * multiplied would be an icon, not a measurement.
 *
 * WHAT THIS FILE DELIBERATELY DOES NOT KNOW
 * It does not know what SiLU is, what d_ff should be, or how SwiGLU is defined.
 * It draws the shapes the trace names and the cell the trace published. The
 * formulas in the picture are labels from the trace's `swiglu.paths`, so the
 * component carries no LaTeX of its own.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;

  function shapeText(shape) {
    return '[' + (shape || []).join(', ') + ']';
  }

  function num(v, digits) {
    if (typeof v !== 'number') return String(v);
    var s = v.toPrecision(digits || 4);
    return s.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  }

  /* The `(step, ctx)` render signature is the engine's contract (see the note in
   * `views/norm-axis.js`); `makeView` takes no config of its own, so it does not
   * pretend to. */
  function makeView() {
    /* One hop: operands on the left and right, result on the right of the `=`.
     * `on` is whether this hop belongs to the step the cursor is on, which is
     * what the highlight is for. */
    function hop(id, label, lhs, op, rhs, out, on, note) {
      return '<div class="lab-sw-hop' + (on ? ' lab-sw-on' : '') +
        '" data-hop="' + esc(id) + '" data-on="' + (on ? '1' : '0') + '">' +
        '<span class="lab-sw-hop-t">' + esc(label) + '</span>' +
        '<span class="lab-sw-expr">' +
          '<span class="lab-sw-t lab-sw-lhs">' + esc(shapeText(lhs)) + '</span>' +
          '<span class="lab-sw-op">' + esc(op) + '</span>' +
          '<span class="lab-sw-t lab-sw-rhs">' + esc(shapeText(rhs)) + '</span>' +
          '<span class="lab-sw-op">=</span>' +
          '<span class="lab-sw-t lab-sw-out">' + esc(shapeText(out)) + '</span>' +
        '</span>' +
        (note ? '<span class="lab-sw-note">' + esc(note) + '</span>' : '') +
        '</div>';
    }

    function render(step, ctx) {
      var sw = step && step.swiglu;
      if (!sw) {
        return '<div class="lab-sw-none">这一步不在 SwiGLU 里 —— 回放里第 7…11 步是它，' +
               '每一步的维度都在这里。</div>';
      }

      var h = sw.shapes;
      var d = sw.d_model;
      var dff = sw.d_ff;
      var on = function (p) { return sw.path === p; };

      var html = '<div class="lab-sw" data-step-id="' + esc(step.id) + '"' +
        ' data-path="' + esc(sw.path) + '"' +
        ' data-d-model="' + d + '" data-d-ff="' + dff + '">';

      html += '<div class="lab-sw-hd">' +
        '<span class="lab-sw-hd-t">SwiGLU：三条路径，维度全程标注</span>' +
        '<span class="lab-sw-hd-s">x ' + esc(shapeText(h.x)) +
        ' · W_gate/W_up ' + esc(shapeText(h.W_gate)) +
        ' · W_down ' + esc(shapeText(h.W_down)) + '</span></div>';

      /* --- the three paths, as three columns of hops. */
      html += '<div class="lab-sw-paths">';

      html += '<div class="lab-sw-path' + (on('gate') ? ' lab-sw-on' : '') +
        '" data-path="gate">' +
        '<span class="lab-sw-path-t">① gate 路径</span>' +
        hop('gate', '升维', h.x, '×', h.W_gate, h.gate, on('gate'),
            'W_gate 的形状是 ' + shapeText(h.W_gate) + '，所以 [N,' + d + '] 乘出 [N,' +
            dff + '] —— 内维 ' + d + ' 对消') +
        hop('silu', 'SiLU（逐元素，形状不变）', h.gate, '→',
            h.gate, h.silu, on('gate'), '激活不改形状：' + shapeText(h.silu)) +
        '</div>';

      html += '<div class="lab-sw-path' + (on('up') ? ' lab-sw-on' : '') +
        '" data-path="up">' +
        '<span class="lab-sw-path-t">② up 路径</span>' +
        hop('up', '升维（不过激活）', h.x, '×', h.W_up, h.up, on('up'),
            'W_up 与 W_gate 同形，所以 up 与 gate 同形 —— 这是 ⊙ 的前提') +
        '<div class="lab-sw-hop lab-sw-hop-gap"><span class="lab-sw-hop-t">' +
          '（这一路没有激活）</span></div>' +
        '</div>';

      html += '<div class="lab-sw-path' + (on('down') ? ' lab-sw-on' : '') +
        '" data-path="down">' +
        '<span class="lab-sw-path-t">③ down 路径</span>' +
        hop('down', '降维回 d_model', h.gated, '×', h.W_down, h.out, on('down'),
            'W_down 的形状是 ' + shapeText(h.W_down) + '，把 ' + dff + ' 降回 ' + d +
            ' —— 只有这样它才能和残差流相加') +
        '</div>';

      html += '</div>';

      /* --- the gate itself: the one step where two tensors meet. */
      var m = sw.multiply || {};
      html += '<div class="lab-sw-mul' + (on('gate') ? '' : ' lab-sw-mul-dim') +
        '" data-path="gate" data-matches="' +
        (m.product_matches ? '1' : '0') + '"' +
        ' data-row="' + m.row + '" data-col="' + m.col + '">' +
        '<div class="lab-sw-mul-h">' +
          '<span class="lab-sw-mul-t">⊙ 逐元素相乘（门控）</span>' +
          '<span class="lab-sw-mul-s">两侧都是 ' + esc(shapeText(h.silu)) +
          '，共 ' + (sw.elements.gated || 0).toLocaleString('en-US') + ' 个位置</span>' +
        '</div>' +
        '<div class="lab-sw-mul-eq">' +
          '<span class="lab-sw-mul-side lab-sw-mul-a">SiLU(G)<sub>' +
            m.row + ',' + m.col + '</sub> = ' + esc(num(m.a, 6)) + '</span>' +
          '<span class="lab-sw-mul-op">×</span>' +
          '<span class="lab-sw-mul-side lab-sw-mul-b">U<sub>' +
            m.row + ',' + m.col + '</sub> = ' + esc(num(m.b, 6)) + '</span>' +
          '<span class="lab-sw-mul-op">=</span>' +
          '<span class="lab-sw-mul-side lab-sw-mul-p lab-sg-ok">S<sub>' +
            m.row + ',' + m.col + '</sub> = ' + esc(num(m.product, 6)) +
            (m.product_matches ? ' ✓' : ' ✗') + '</span>' +
        '</div>' +
        '<p class="lab-sw-mul-note">这是矩阵里真实的一个位置，' +
          '不是示意：其余 ' + ((sw.elements.gated || 0) - 1).toLocaleString('en-US') +
          ' 个位置都是同一个操作。SiLU 只作用在 gate 这一路上 —— ' +
          'up 那一路保持线性，两者相乘就构成了门控。</p>' +
        '</div>';

      /* --- the shape table, so a reader can check every hop at once. */
      html += '<table class="lab-sw-table"><thead><tr>' +
        '<th>张量</th><th>形状</th><th>元素数</th></tr></thead><tbody>';
      [['x（子层输入）', 'x', null],
       ['W_gate', 'W_gate', null],
       ['W_up', 'W_up', null],
       ['gate（升维后）', 'gate', 'gate'],
       ['SiLU(gate)', 'silu', 'silu'],
       ['up（升维后）', 'up', 'up'],
       ['S = SiLU(gate) ⊙ up', 'gated', 'gated'],
       ['W_down', 'W_down', null],
       ['F（降维回 d_model）', 'out', null]].forEach(function (row) {
        var shape = h[row[1]];
        html += '<tr class="' + (row[2] && row[2] === sw.path ? 'lab-sw-cur' : '') + '">' +
          '<td>' + esc(row[0]) + '</td>' +
          '<td class="lab-sw-shape">' + esc(shapeText(shape)) + '</td>' +
          '<td class="lab-sw-elems">' +
            esc(row[2] ? (sw.elements[row[2]] || 0).toLocaleString('en-US') : '—') +
          '</td></tr>';
      });
      html += '</tbody></table>';

      html += '</div>';
      return html;
    }

    return { render: render };
  }

  function panel(config, context) {
    var view = makeView();
    return {
      id: (config && config.id) || 'swiglu-paths',
      label: (config && config.label) || 'SwiGLU：三条路径与逐元素门控，维度全程标注',
      render: function (step, ctx) { return view.render(step, ctx); }
    };
  }

  /* ================================================================= lint
   *
   * The JS port of the `step.swiglu` contract. `labs/traces/decoder_block.py` is
   * the authority; this is the port beside the view, and THE TWO MUST AGREE —
   * each carries its own sabotage case, and the acceptance harness runs one
   * table through both.
   *
   * FIELD-GATED: every rule fires only on a step that carries `swiglu`, with
   * the block-level requirement gated one level down (`meta.decoder`).
   */
  var PATHS = ['gate', 'up', 'down'];

  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  function sameShape(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var steps = (trace && trace.steps) || [];
    var withSw = steps.filter(function (s) { return s && s.swiglu !== undefined; });
    var withRes = steps.filter(function (s) { return s && s.residual !== undefined; });
    var withLn = steps.filter(function (s) { return s && s.ln !== undefined; });
    var dec = ((trace || {}).meta || {}).decoder;

    if (!withSw.length && !withRes.length && !withLn.length) {
      if (dec !== undefined) {
        gaps.push({ what: 'meta.decoder 存在，但没有任何步骤带 step.swiglu',
                    why: '声明与数据不符。' });
      }
      return { gaps: gaps, warns: warns, infos: infos, present: false };
    }
    if (dec === undefined) {
      gaps.push({ what: 'trace 有步骤带本 lab 的步骤字段，但 meta.decoder 缺失',
                  why: '逐字段的规则无法门控。' });
      return { gaps: gaps, warns: warns, infos: infos, present: true };
    }
    if (withSw.length) {
      var listed = dec.steps_with_swiglu;
      if (!Array.isArray(listed) || listed.slice().sort().join(',') !==
          withSw.map(function (s) { return s.id; }).sort().join(',')) {
        gaps.push({ what: 'meta.decoder.steps_with_swiglu 与真正带 swiglu 的步骤不一致',
                    why: '声明与数据各说各话。' });
      }
    }

    withSw.forEach(function (s) {
      var sid = s.id, sw = s.swiglu, h = sw.shapes || {};
      var n = (h.x || [])[0], d = sw.d_model, dff = sw.d_ff;
      if (!isNum(n) || !isNum(d) || !isNum(dff)) {
        gaps.push({ what: '步骤 "' + sid + '" 的 swiglu 缺 d_model / d_ff / x.shape',
                    why: '尺寸标注就无从谈起。' });
        return;
      }
      /* The three weights. `nn.Linear` stores (out, in), and the tutorial's own
       * module does too — so W_gate and W_up are (d_ff, d) and W_down is
       * (d, d_ff). A picture that drew W_down the same way round as the other
       * two would show a hop whose inner dimensions do not cancel. */
      [['W_gate', [dff, d]], ['W_up', [dff, d]], ['W_down', [d, dff]]]
        .forEach(function (pair) {
          if (!sameShape(h[pair[0]], pair[1])) {
            gaps.push({ what: '步骤 "' + sid + '" 的 shapes["' + pair[0] + '"] = ' +
                              shapeText(h[pair[0]]) + '，应为 ' + shapeText(pair[1]),
                        why: '升维与降维的方向标反，读者按图算内维就对不上了。' });
          }
        });
      /* The intermediates, all at d_ff. */
      ['gate', 'up', 'silu', 'gated'].forEach(function (k) {
        if (!sameShape(h[k], [n, dff])) {
          gaps.push({ what: '步骤 "' + sid + '" 的 shapes["' + k + '"] = ' +
                            shapeText(h[k]) + '，应为 ' + shapeText([n, dff]),
                      why: '这一路是升维后的中间量。' });
        }
      });
      if (!sameShape(h.out, [n, d])) {
        gaps.push({ what: '步骤 "' + sid + '" 的 shapes["out"] = ' + shapeText(h.out) +
                          '，应为 ' + shapeText([n, d]),
                    why: 'down 路径必须降回 d_model —— 否则残差加就不成立。' });
      }
      /* ⊙: the two operands must be the same shape, and that shape must be the
       * one the multiply's own element count was taken over. */
      if (!sameShape(h.silu, h.up)) {
        gaps.push({ what: '步骤 "' + sid + '" 逐元素相乘的两侧 shape 不同（' +
                          shapeText(h.silu) + ' vs ' + shapeText(h.up) + '）',
                    why: '⊙ 要求两侧完全相同 —— 这也是 gate 与 up 必须同维的原因。' });
      }
      var nelem = (sw.elements || {}).gated;
      if (nelem !== n * dff) {
        gaps.push({ what: '步骤 "' + sid + '" 的 elements.gated = ' + nelem +
                          '，应为 N×d_ff = ' + (n * dff),
                    why: '逐元素相乘覆盖的位置数就是矩阵的元素数。' });
      }
      if (PATHS.indexOf(sw.path) === -1) {
        gaps.push({ what: '步骤 "' + sid + '" 的 swiglu.path = ' +
                          JSON.stringify(sw.path) + ' 不属于 ' + PATHS.join(' / '),
                    why: '这一格决定哪条路径高亮，不在词表里就画不出来。' });
      }
      /* The measured cell. `product == a × b` is the rule that keeps the ⊙ from
       * being an icon: the three numbers have to be a real multiplication. */
      var m = sw.multiply || {};
      if (!isNum(m.row) || !isNum(m.col)) {
        gaps.push({ what: '步骤 "' + sid + '" 的 swiglu.multiply 缺行列下标',
                    why: '没有位置就无法核对。' });
      } else if (!(m.row >= 0 && m.row < n && m.col >= 0 && m.col < dff)) {
        gaps.push({ what: '步骤 "' + sid + '" 的 swiglu.multiply 位置 (' + m.row + ',' +
                          m.col + ') 越出 ' + shapeText([n, dff]),
                    why: '取到了一个矩阵里不存在的位置。' });
      }
      ['a', 'b', 'product'].forEach(function (k) {
        if (!isNum(m[k])) {
          gaps.push({ what: '步骤 "' + sid + '" 的 swiglu.multiply.' + k + ' 不是数',
                      why: '公式面板上会出现 undefined。' });
        }
      });
      if (m.product_matches !== true) {
        gaps.push({ what: '步骤 "' + sid + '" 声称 S = a×b，但 product_matches 不是 true',
                    why: '那这个逐元素乘就不是跑出来的。' });
      }
      if (isNum(m.a) && isNum(m.b) && isNum(m.product) &&
          m.product !== m.a * m.b) {
        gaps.push({ what: '步骤 "' + sid + '" 的 multiply：a×b = ' + (m.a * m.b) +
                          ' 但 product = ' + m.product,
                    why: '同一格的两个字段互相矛盾。' });
      }
      if (m.rows_checked !== n * dff) {
        gaps.push({ what: '步骤 "' + sid + '" 的 multiply.rows_checked = ' +
                          m.rows_checked + '，应为 ' + (n * dff),
                    why: '声称核对的元素数必须等于矩阵的元素数。' });
      }
    });

    if (withSw.length) {
      var seen = {};
      withSw.forEach(function (s) { seen[s.swiglu.path] = true; });
      PATHS.forEach(function (p) {
        if (!seen[p]) {
          gaps.push({ what: '回放里没有任何一步走 "' + p + '" 路径',
                      why: '三条路径没有走全，读者看不到完整的一遍。' });
        }
      });
      infos.push({ what: 'SwiGLU 覆盖 ' + withSw.length + ' 步，三条路径都走到',
                   why: '' });
    }
    return { gaps: gaps, warns: warns, infos: infos, present: true };
  }

  var SABOTAGE_CASES = {
    'swiglu 块被删掉但 meta 还列着': function (t) { delete t.steps[7].swiglu; },
    'swiglu up 路径的形状错了': function (t) {
      t.steps[8].swiglu.shapes.up = [t.steps[8].swiglu.shapes.x[0],
                                     t.steps[8].swiglu.d_model];
    },
    'swiglu W_down 被转置': function (t) {
      t.steps[11].swiglu.shapes.W_down = [t.steps[11].swiglu.d_ff,
                                          t.steps[11].swiglu.d_model];
    },
    'swiglu down 没有降回 d_model': function (t) {
      t.steps[11].swiglu.shapes.out = [t.steps[11].swiglu.shapes.x[0],
                                       t.steps[11].swiglu.d_ff];
    },
    'swiglu ⊙ 的两侧形状不同': function (t) {
      t.steps[10].swiglu.shapes.up = [t.steps[10].swiglu.shapes.x[0],
                                      t.steps[10].swiglu.d_model];
    },
    'swiglu gate 路径的形状错了': function (t) {
      t.steps[7].swiglu.shapes.gate = [t.steps[7].swiglu.shapes.x[0],
                                       t.steps[7].swiglu.d_model];
    },
    'swiglu 元素数不对': function (t) { t.steps[10].swiglu.elements.gated = 8; },
    'swiglu 取的位置在矩阵之外': function (t) { t.steps[10].swiglu.multiply.row = 99; },
    'swiglu 声称的乘积不是算出来的': function (t) {
      t.steps[10].swiglu.multiply.product = 12345;
    },
    'swiglu product_matches 被清掉': function (t) {
      t.steps[10].swiglu.multiply.product_matches = false;
    },
    'swiglu 核对的元素数不覆盖矩阵': function (t) {
      t.steps[10].swiglu.multiply.rows_checked = 1;
    },
    'swiglu 路径 id 不认识': function (t) { t.steps[7].swiglu.path = 'sideways'; },
    'swiglu 某条路径没有被走到': function (t) {
      t.steps.forEach(function (s) { if (s.swiglu) s.swiglu.path = 'gate'; });
    },
    'swiglu meta.decoder 丢了清单': function (t) { delete t.meta.decoder.steps_with_swiglu; }
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

  NS.swigluPaths = {
    makeView: makeView,
    panel: panel,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES,
    PATHS: PATHS
  };
})(typeof window !== 'undefined' ? window : globalThis);
