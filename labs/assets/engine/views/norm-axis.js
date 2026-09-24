/* Engine · view component: which axis LayerNorm reduces over.
 *
 * L04's second teaching point, and the ticket calls it out as the one most
 * easily told badly: "LayerNorm 的均值/方差是在 d_model 维上、对每个 token
 * 独立计算——在视图上可辨". A sentence saying so is not 可辨. This view makes it
 * decidable by drawing the array twice.
 *
 * THE MECHANISM, WHICH IS THE WHOLE DESIGN
 * The trace publishes, for the real `[N, d_model]` array, three statistic pairs
 * that were computed in the generator:
 *
 *   `row`             the NORMALIZED array's per-token statistics (mean and
 *                     std down each row) — 0 and 1
 *   `column`          the same array's per-feature statistics (down each
 *                     column) — NOT 0 and 1
 *   `counterfactual`  the same formula with the reduction one axis over, and
 *                     its two statistics — which come out the other way round
 *
 * Those three together are what makes the axis legible rather than asserted.
 * Rows at 0/1 is consistent with normalizing nothing at all; it is the columns
 * NOT also being 0/1 that says the normalization happened along the rows, and
 * it is the counterfactual flipping that says the asymmetry is about the axis
 * and not about this particular array. All three are drawn, and `lint()`
 * refuses a trace where any leg of that argument is missing.
 *
 * The N per-token μ and σ are drawn as two strips of N cells, because "one mean
 * per token" is a claim about the SHAPE of that vector — a global mean would be
 * one cell, and a reader can count N cells.
 *
 * WHAT THIS FILE DELIBERATELY DOES NOT KNOW
 * It does not know the LayerNorm formula, ε, or what d_model is. It draws the
 * statistics it is handed and the axis the trace names. The counterfactual is
 * NOT computed here — it arrives in the trace, computed by the same generator
 * that computed the primary one, so the two are guaranteed to be the same
 * formula evaluated the same way.
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

  /* A statistic pair, with the verdict spelled out. `want` is what the array
   * SHOULD read on this axis if the normalization ran along the other one; the
   * component does not decide that, the caller passes it, so the rule lives in
   * one place (the trace) and the drawing only reports the comparison. */
  function statRow(label, stat, wantZeroOne) {
    var centred = Math.abs(stat.mean_max_abs) < 1e-6;
    var unit = Math.abs(stat.std_min - 1) < 1e-3 && Math.abs(stat.std_max - 1) < 1e-3;
    var isZeroOne = centred && unit;
    var cls = isZeroOne ? 'lab-na-yes' : 'lab-na-no';
    return '<tr class="' + cls + '" data-stat="' + esc(label) + '"' +
      ' data-zero-one="' + (isZeroOne ? '1' : '0') + '">' +
      '<th>' + esc(label) + '</th>' +
      '<td>|均值| 最大 <b>' + esc(num(stat.mean_max_abs, 3)) + '</b></td>' +
      '<td>标准差 <b>' + esc(num(stat.std_min, 4)) + '</b> … <b>' +
        esc(num(stat.std_max, 4)) + '</b></td>' +
      '<td class="lab-na-verdict">' +
        (isZeroOne
          ? (wantZeroOne ? '0 / 1 ✓' : '0 / 1 —— 但它不该是')
          : (wantZeroOne ? '不是 0/1 —— 但它该是' : '不是 0 / 1')) +
      '</td></tr>';
  }

  function strip(label, values, cls) {
    return '<div class="lab-na-strip ' + cls + '">' +
      '<span class="lab-na-strip-t">' + esc(label) + '</span>' +
      '<span class="lab-na-cells" data-count="' + values.length + '">' +
      values.map(function (v) {
        return '<span class="lab-na-cell">' + esc(num(v, 3)) + '</span>';
      }).join('') + '</span></div>';
  }

  /* The `(step, ctx)` render signature is the engine's contract — `player.js`
   * calls every panel's `render(step, ctx)` — so the two names are fixed even
   * where a panel ignores one of them, exactly as in `views/attn-shape.js` and
   * `views/gantt.js`. What is NOT fixed is `makeView`'s own parameter list,
   * which is why the unused `config` that used to sit here is gone. */
  function makeView() {
    function render(step, ctx) {
      var ln = step && step.ln;
      if (!ln) {
        return '<div class="lab-na-none">这一步不做 LayerNorm —— 回放里有两次' +
               '（第 1 步与第 6 步），每一次的统计量都在这里。</div>';
      }

      var html = '<div class="lab-na" data-step-id="' + esc(step.id) + '"' +
        ' data-axis="' + esc(ln.axis) + '"' +
        ' data-tokens="' + ln.tokens + '"' +
        ' data-d-model="' + ln.d_model + '"' +
        ' data-row-zero-one="' +
          ((Math.abs(ln.row.mean_max_abs) < 1e-6 &&
            Math.abs(ln.row.std_min - 1) < 1e-3 &&
            Math.abs(ln.row.std_max - 1) < 1e-3) ? '1' : '0') + '"' +
        ' data-col-zero-one="' +
          ((Math.abs(ln.column.mean_max_abs) < 1e-6 &&
            Math.abs(ln.column.std_min - 1) < 1e-3 &&
            Math.abs(ln.column.std_max - 1) < 1e-3) ? '1' : '0') + '">';

      var shape = '[' + ln.tokens + ', ' + ln.d_model + ']';

      html += '<div class="lab-na-hd">' +
        '<span class="lab-na-hd-t">μ 和 σ 各 ' + ln.tokens + ' 个 —— ' +
        '每个 token 一对，在它自己的 ' + ln.d_model + ' 个特征上求</span>' +
        '<span class="lab-na-hd-s">输入 ' + esc(shape) + '，归约轴 = ' +
        esc(ln.axis) + '（最后一维）。ε = ' + esc(String(ln.eps)) + '</span></div>';

      html += '<div class="lab-na-mu">' +
        strip('μ（逐 token）', ln.mu, 'lab-na-mu-s') +
        strip('σ（逐 token）', ln.sigma, 'lab-na-sg-s') +
        '</div>';

      html += '<p class="lab-na-note">' + ln.mu.length + ' 个 μ 互不相同（极差 <b>' +
        esc(num(ln.mu_spread, 3)) + '</b>），' + ln.sigma.length + ' 个 σ 也是（极差 <b>' +
        esc(num(ln.sigma_spread, 3)) + '</b>）。如果归一化是在整个 ' + esc(shape) +
        ' 上求一个全局 μ、一个全局 σ，这两条带子会各只有一格 —— ' +
        '这里有几格就是有几个。</p>';

      /* --- the decidability argument. */
      html += '<div class="lab-na-hd lab-na-hd-2">' +
        '<span class="lab-na-hd-t">归一化之后，统计量落在哪个轴上</span>' +
        '<span class="lab-na-hd-s">行 = 某个 token 的 ' + ln.d_model +
        ' 个特征；列 = 某个特征在 ' + ln.tokens + ' 个 token 上的取值</span></div>';

      html += '<table class="lab-na-table"><thead><tr>' +
        '<th>沿哪个方向统计</th><th>|均值| 的最大值</th><th>标准差范围</th>' +
        '<th>是不是 0 / 1</th></tr></thead><tbody>' +
        statRow('行（每个 token 内，' + ln.d_model + ' 个特征）', ln.row, true) +
        statRow('列（每个特征上，' + ln.tokens + ' 个 token）', ln.column, false) +
        '</tbody></table>';

      html += '<p class="lab-na-note">' +
        '如果归一化是<b>沿 token 轴</b>做的（即对每个特征、跨所有 token 求 μ、σ），' +
        '那么该落在 0/1 的就是<b>列</b>、不是行 —— 两个轴都会经过一次同样的公式，' +
        '差别只在归约方向。下面对照组把这条反事实也算出来了。</p>';

      html += '<table class="lab-na-table lab-na-table-cf"><thead><tr>' +
        '<th>反事实：沿 token 轴归约</th><th>|均值| 的最大值</th>' +
        '<th>标准差范围</th><th>是不是 0 / 1</th></tr></thead><tbody>' +
        statRow('行（每个 token 内）', ln.counterfactual_row, false) +
        statRow('列（每个特征上）', ln.counterfactual_column, true) +
        '</tbody></table>';

      var flipped =
        flips(ln.row) !== flips(ln.counterfactual_row) &&
        flips(ln.column) !== flips(ln.counterfactual_column);
      html += '<p class="lab-na-sum" data-flipped="' + (flipped ? '1' : '0') + '">' +
        (flipped
          ? '<b>两个轴正好对调。</b>所以「行落在 0/1」这件事本身不是证据 —— ' +
            '换个轴做，它就落到列上去了。真要判断这份 trace 归一化在哪个方向，' +
            '靠的是把两个方向都算一遍、看哪一个对得上。'
          : '<b>两组统计没有对调 —— 这条 trace 无法区分两个轴。</b>') +
        '</p>';

      html += '</div>';
      return html;
    }

    function flips(stat) {
      return Math.abs(stat.mean_max_abs) < 1e-6 &&
             Math.abs(stat.std_min - 1) < 1e-3 &&
             Math.abs(stat.std_max - 1) < 1e-3;
    }

    return { render: render };
  }

  function panel(config, context) {
    var view = makeView();
    return {
      id: (config && config.id) || 'norm-axis',
      label: (config && config.label) || 'LayerNorm：μ、σ 在哪一维上，逐 token 还是全局',
      render: function (step, ctx) { return view.render(step, ctx); }
    };
  }

  /* ================================================================= lint
   *
   * The JS port of the `step.ln` contract. `labs/traces/decoder_block.py` is the
   * authority; this is the port beside the view, so a lab author editing a trace
   * in the browser gets the same verdict. THE TWO PORTS MUST AGREE, and each
   * carries its own sabotage case — a rule that exists on one side only is a
   * rule tested on neither.
   *
   * FIELD-GATED: every rule fires only on a step that carries `ln`, and the
   * block-level requirement is gated one level down (`meta.decoder`).
   */
  var AXIS_TOL = 1e-6;

  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  function zeroOne(stat) {
    return !!stat && Math.abs(stat.mean_max_abs) < AXIS_TOL &&
      Math.abs(stat.std_min - 1) < 1e-3 && Math.abs(stat.std_max - 1) < 1e-3;
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var steps = (trace && trace.steps) || [];
    var withLn = steps.filter(function (s) { return s && s.ln !== undefined; });
    var withRes = steps.filter(function (s) { return s && s.residual !== undefined; });
    var dec = ((trace || {}).meta || {}).decoder;

    if (!withLn.length && !withRes.length) {
      if (dec !== undefined) {
        gaps.push({ what: 'meta.decoder 存在，但没有任何步骤带 step.ln',
                    why: '声明与数据不符。' });
      }
      return { gaps: gaps, warns: warns, infos: infos, present: false };
    }
    if (dec === undefined) {
      gaps.push({ what: 'trace 有步骤带本 lab 的步骤字段，但 meta.decoder 缺失',
                  why: '逐字段的规则无法门控。' });
      return { gaps: gaps, warns: warns, infos: infos, present: true };
    }
    if (withLn.length) {
      var listed = dec.steps_with_ln;
      if (!Array.isArray(listed) || listed.slice().sort().join(',') !==
          withLn.map(function (s) { return s.id; }).sort().join(',')) {
        gaps.push({ what: 'meta.decoder.steps_with_ln 与真正带 ln 的步骤不一致',
                    why: '声明与数据各说各话。' });
      }
    }

    withLn.forEach(function (s) {
      var sid = s.id, ln = s.ln;
      if (ln.axis !== 'd_model') {
        gaps.push({ what: '步骤 "' + sid + '" 的 ln.axis = ' + JSON.stringify(ln.axis) +
                          '，本 lab 的归一化轴是 d_model',
                    why: '轴上标错，整张图就反了。' });
      }
      ['mu', 'sigma'].forEach(function (k) {
        if (!Array.isArray(ln[k]) || ln[k].length !== ln.tokens) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ln.' + k + ' 有 ' +
                            (Array.isArray(ln[k]) ? ln[k].length : '?') +
                            ' 个值，应该是每个 token 一个（' + ln.tokens + ' 个）',
                      why: '「逐 token」这件事就是靠这个数量来数出来的。' });
        }
      });
      if (!isNum(ln.tokens) || ln.tokens < 2) {
        gaps.push({ what: '步骤 "' + sid + '" 的 ln.tokens = ' + ln.tokens,
                    why: '只有一个 token 时「逐 token」和「全局」无法区分。' });
      }
      ['mu_spread', 'sigma_spread'].forEach(function (k) {
        if (!isNum(ln[k]) || ln[k] <= 0) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ln.' + k + ' = ' + ln[k],
                      why: '所有 token 的统计量完全相同的话，逐 token 这件事就看不出来。' });
        }
      });

      var legs = ['row', 'column', 'counterfactual_row', 'counterfactual_column'];
      var missing = legs.filter(function (k) { return !ln[k] || !isNum(ln[k].mean_max_abs); });
      if (missing.length) {
        gaps.push({ what: '步骤 "' + sid + '" 的 ln 缺 ' + missing.join('、') + ' 统计',
                    why: '「沿哪个轴」没有可判据 —— 这正是这张图存在的理由。' });
        return;
      }
      /* The primary reading: rows 0/1, columns NOT. Both halves are required.
       * "The rows came out 0/1" alone is consistent with a picture that
       * normalized nothing at all, and this is the rule that refuses it. */
      if (!zeroOne(ln.row)) {
        gaps.push({ what: '步骤 "' + sid + '" 归一化后每一行应落在 0/1，实测 |均值| 最大 ' +
                          ln.row.mean_max_abs + '、标准差 ' + ln.row.std_min + '…' +
                          ln.row.std_max,
                    why: '沿 d_model 归一化的定义就是这个。' });
      }
      if (zeroOne(ln.column)) {
        gaps.push({ what: '步骤 "' + sid + '" 的列统计也落在了 0/1',
                    why: '那这条 trace 无法区分「沿 d_model」与「沿 token」—— ' +
                         '两个轴都对得上，图就没有信息。' });
      }
      /* The counterfactual must show the flip, or the asymmetry above is not
       * evidence about the axis: a trace whose rows read 0/1 could equally be
       * one that normalized nothing and whose columns happened to look right. */
      if (zeroOne(ln.counterfactual_row)) {
        gaps.push({ what: '步骤 "' + sid + '" 沿 token 轴归一化的对照组，行统计也落在 0/1',
                    why: '这个对照组没有翻转，上面那个不对称就不构成证据。' });
      }
      if (!zeroOne(ln.counterfactual_column)) {
        gaps.push({ what: '步骤 "' + sid + '" 沿 token 轴归一化的对照组，列统计没有落在 0/1',
                    why: '对照组本身写错了。' });
      }
    });

    if (withLn.length) {
      infos.push({ what: 'LayerNorm ' + withLn.length + ' 处，各带行/列/反事实三组统计',
                   why: '' });
    }
    return { gaps: gaps, warns: warns, infos: infos, present: true };
  }

  var SABOTAGE_CASES = {
    'normaxis ln 块被删掉但 meta 还列着': function (t) { delete t.steps[1].ln; },
    'normaxis 轴被写成 token 轴': function (t) { t.steps[1].ln.axis = 'tokens'; },
    'normaxis μ 只有一个而不是每 token 一个': function (t) {
      t.steps[1].ln.mu = t.steps[1].ln.mu.slice(0, 1);
    },
    'normaxis σ 只有一个': function (t) { t.steps[1].ln.sigma = t.steps[1].ln.sigma.slice(0, 1); },
    'normaxis μ 全相同': function (t) { t.steps[1].ln.mu_spread = 0; },
    'normaxis σ 全相同': function (t) { t.steps[1].ln.sigma_spread = 0; },
    'normaxis 行统计没有中心化': function (t) { t.steps[1].ln.row.mean_max_abs = 0.4; },
    'normaxis 行的标准差不是 1': function (t) { t.steps[1].ln.row.std_max = 3; },
    'normaxis 列也落在 0/1，轴就分不出来了': function (t) {
      var r = t.steps[1].ln;
      r.column.mean_max_abs = 0.0; r.column.std_min = 1.0; r.column.std_max = 1.0;
    },
    'normaxis 对照组不再翻转': function (t) {
      t.steps[1].ln.counterfactual_column.std_min = 4.0;
      t.steps[1].ln.counterfactual_column.std_max = 5.0;
    },
    'normaxis 对照组的行也落在 0/1': function (t) {
      var r = t.steps[1].ln.counterfactual_row;
      r.mean_max_abs = 0.0; r.std_min = 1.0; r.std_max = 1.0;
    },
    'normaxis 第二处 LayerNorm 没查': function (t) { t.steps[6].ln.row.std_min = 5; },
    'normaxis 只有一个 token': function (t) { t.steps[1].ln.tokens = 1; },
    'normaxis meta.decoder 丢了 ln 清单': function (t) { delete t.meta.decoder.steps_with_ln; }
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

  NS.normAxis = {
    makeView: makeView,
    panel: panel,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES
  };
})(typeof window !== 'undefined' ? window : globalThis);
