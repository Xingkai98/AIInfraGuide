/* Engine · view component: the residual add and its shape requirement.
 *
 * L04's first teaching point, and the ticket is explicit that the confirmation
 * is the easy half: "残差相加的两侧 shape 相同时在图上被确认，**并演示若
 * shape 不匹配会怎样**". So this view draws two things, and the second one is
 * the reason it exists:
 *
 *   the add     the two operands as aligned [N, d_model] blocks, with their
 *               shapes printed on them and the result below — the confirmation
 *   the shapes  every candidate right-hand side the trace measured, each with
 *               the OUTCOME THAT ACTUALLY HAPPENED when it was added: the one
 *               that is the residual, the ones that broadcast silently, and
 *               the ones that raised
 *
 * WHY THE BROADCAST ROWS ARE THE POINT
 * "The two shapes are equal" is a one-line fact, and a picture that only showed
 * it would leave a reader believing the shape check is a formality they can
 * skip. It is not: `[d]` added to `[N, d]` is a per-feature bias, it is legal
 * arithmetic, it produces the right shape, and nothing raises — but the block
 * it builds is not a residual block. The three silent cases and the three that
 * raise are drawn side by side so the reader can see that the failure mode is
 * not "it crashes" but "it does not crash, and it is wrong".
 *
 * The exception text on the error rows is numpy's own, published by the
 * generator from the run that produced it — not a description of what a shape
 * error would say.
 *
 * WHAT THIS FILE DELIBERATELY DOES NOT KNOW
 * It does not know what numpy's broadcasting rules are, which shape pairs are
 * compatible, or what `d_model` means. It draws whatever outcomes the trace
 * published, and its only arithmetic is turning two shapes into a row of cells.
 * `lint()` states the rules it may therefore carry — all of them about the
 * trace's own internal consistency, none of them a re-derivation of numpy.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;

  function shapeText(shape) {
    return '[' + (shape || []).join(', ') + ']';
  }

  function elems(shape) {
    return (shape || []).reduce(function (a, b) { return a * b; }, 1);
  }

  /* A cell grid for one operand, drawn to at most `maxCells` cells. A [128, 64]
   * operand is 8192 cells and no reader can count them; the grid's dimensions
   * are labelled with the true shape and the drawn size is capped, which is the
   * honest way to draw something too big to draw. The cap is deliberately the
   * same for both operands, so the two blocks' DRAWN sizes carry the ratio of
   * their element counts. */
  function drawGrid(shape, cls, maxCells) {
    var rows = shape[0] || 1;
    var cols = shape.length > 1 ? shape[1] : 1;
    var cap = Math.max(1, Math.floor(Math.sqrt(maxCells)));
    var scale = Math.min(1, cap / Math.max(rows, cols));
    var dr = Math.max(1, Math.round(rows * scale));
    var dc = Math.max(1, Math.round(cols * scale));
    var capped = dr !== rows || dc !== cols;
    var html = '<div class="lab-sg-block ' + cls + '"' +
      ' data-rows="' + rows + '" data-cols="' + cols + '"' +
      ' data-drawn-rows="' + dr + '" data-drawn-cols="' + dc + '"' +
      ' data-capped="' + (capped ? '1' : '0') + '"' +
      ' style="grid-template-columns:repeat(' + dc + ',1fr)">';
    for (var i = 0; i < dr * dc; i++) html += '<span class="lab-sg-cell"></span>';
    html += '</div>';
    return html;
  }

  /* ================================================================= render */

  function makeView(config) {
    var cfg = config || {};
    var maxCells = cfg.maxCells || 256;

    function operand(res, side, label, cls) {
      var shape = side === 'lhs' ? res.lhs_shape : res.rhs_shape;
      return '<div class="lab-sg-operand">' +
        '<span class="lab-sg-op-t">' + esc(label) + '</span>' +
        drawGrid(shape, cls, maxCells) +
        '<span class="lab-sg-op-s">' + esc(shapeText(shape)) + '</span>' +
        '</div>';
    }

    function render(step, ctx) {
      var res = step && step.residual;
      if (!res) {
        return '<div class="lab-sg-none">这一步不是残差相加 —— 走完整条回放可以看到两处' +
               '残差（第 5 步与第 12 步），每一步的形状都在这里。</div>';
      }

      var cases = res.cases || [];
      var same = cases.filter(function (c) { return c.outcome === 'same'; });
      var silent = cases.filter(function (c) { return c.outcome === 'broadcast'; });
      var raised = cases.filter(function (c) { return c.outcome === 'error'; });

      var html = '<div class="lab-sg" data-step-id="' + esc(step.id) + '"' +
        ' data-lhs="' + esc(shapeText(res.lhs_shape)) + '"' +
        ' data-rhs="' + esc(shapeText(res.rhs_shape)) + '"' +
        ' data-equal="' + (res.equal ? '1' : '0') + '"' +
        ' data-silent="' + silent.length + '"' +
        ' data-raised="' + raised.length + '">';

      /* --- the confirmation: the two sides, their shapes, the result. */
      html += '<div class="lab-sg-hd">' +
        '<span class="lab-sg-hd-t">两侧 shape 相同 —— 逐元素相加</span>' +
        '<span class="lab-sg-hd-s">' + esc(shapeText(res.lhs_shape)) +
        ' 与 ' + esc(shapeText(res.rhs_shape)) +
        ' 都是 ' + elems(res.lhs_shape).toLocaleString('en-US') + ' 个元素</span></div>';

      html += '<div class="lab-sg-add">' +
        operand(res, 'lhs', 'x（残差流）', 'lab-sg-lhs') +
        '<span class="lab-sg-op">+</span>' +
        operand(res, 'rhs', 'F(x)（子层输出）', 'lab-sg-rhs') +
        '<span class="lab-sg-op">=</span>' +
        '<div class="lab-sg-operand">' +
          '<span class="lab-sg-op-t">y（结果）</span>' +
          drawGrid(res.result_shape, 'lab-sg-out', maxCells) +
          '<span class="lab-sg-op-s">' + esc(shapeText(res.result_shape)) + '</span>' +
        '</div></div>';

      html += '<p class="lab-sg-note">两侧逐元素相加，结果形状不变 —— ' +
        '用双循环重算一遍的最大相对偏差是 <b>' +
        esc(String(res.element_loop_max_rel_dev)) + '</b>' +
        '，所以「逐元素」是量出来的，不是这句话本身。</p>';

      /* --- the counterexamples. This is the half the ticket calls out. */
      html += '<div class="lab-sg-hd lab-sg-hd-2">' +
        '<span class="lab-sg-hd-t">如果右侧 shape 不是它 —— 真的跑一遍</span>' +
        '<span class="lab-sg-hd-s">' + cases.length + ' 个候选右操作数：' +
        same.length + ' 个是残差加，' + silent.length + ' 个<b>静默广播</b>，' +
        raised.length + ' 个抛异常。numpy 与 torch 两侧的结果一致。</span></div>';

      html += '<div class="lab-sg-cases">';
      cases.forEach(function (c) {
        var cls = c.outcome;
        html += '<div class="lab-sg-case lab-sg-' + esc(cls) + '"' +
          ' data-case="' + esc(c.id) + '" data-outcome="' + esc(cls) + '">' +
          '<div class="lab-sg-case-h">' +
            '<span class="lab-sg-case-s">' + esc(shapeText(c.rhs_shape)) + '</span>' +
            '<span class="lab-sg-case-b lab-sg-b-' + esc(cls) + '">' +
              esc(cls === 'same' ? '残差加'
                : cls === 'broadcast' ? '静默广播，不报错'
                : '抛异常') + '</span>' +
          '</div>' +
          '<div class="lab-sg-case-l">' + esc(c.label) + '</div>' +
          '<div class="lab-sg-case-r">' +
            (cls === 'error'
              /* numpy's own words. Verbatim, because a paraphrase of an
               * exception is a claim about numpy rather than a reading of it. */
              ? '<code class="lab-sg-msg">' + esc(c.message) + '</code>'
              : '结果 ' + esc(shapeText(c.result_shape)) +
                (cls === 'broadcast'
                  ? '<span class="lab-sg-warn">形状对，但它不是残差连接</span>'
                  : '<span class="lab-sg-ok">两侧完全相同</span>')) +
          '</div>' +
          '<div class="lab-sg-case-x">torch 侧：' +
            esc(c.torch === 'same' ? '同样相加'
              : c.torch === 'broadcast' ? '同样静默广播'
              : c.torch === 'error' ? '同样抛异常' : String(c.torch)) + '</div>' +
        '</div>';
      });
      html += '</div>';

      html += '<p class="lab-sg-sum">' +
        '<b>' + silent.length + ' 个候选不会报错，但它们是错的。</b>' +
        '形状检查不是一个形式：<code>[d]</code> 加 <code>[N, d]</code> 是合法的逐特征' +
        '偏置，结果形状也对，可它构造出的不是残差块 —— 这条路径不会在训练崩溃时报错，' +
        '只会让模型学不到该学的东西。要让它失败，得两侧的第一个不匹配维度真的对不上。' +
        '</p>';

      html += '</div>';
      return html;
    }

    return { render: render };
  }

  function panel(config, context) {
    var view = makeView({ maxCells: config && config.maxCells });
    return {
      id: (config && config.id) || 'shape-guard',
      label: (config && config.label) || '残差相加：两侧 shape 与它不匹配时会怎样',
      render: function (step, ctx) { return view.render(step, ctx); }
    };
  }

  /* ================================================================= lint
   *
   * The JS port of the `step.residual` contract this view renders.
   * `labs/traces/decoder_block.py` is the authority and runs before a trace is
   * written; this exists so a lab author editing a trace in the browser gets
   * the same verdict without a Python round trip.
   *
   * THE TWO PORTS MUST AGREE. The acceptance harness runs one sabotage table
   * through both and requires the same verdict from each; a rule that exists on
   * one side only is a rule tested on neither.
   *
   * FIELD-GATED, on purpose. Every rule below fires only on a step that carries
   * a `residual` block, and the block-level requirement is gated one level DOWN
   * (`meta.decoder`), so a trace with no residual semantics at all lints clean
   * and the sabotage that deletes the decoder map is still caught. That is the
   * arrangement `tiling-stage.js` uses to serve four labs without turning three
   * of them red — see `labs/traces/README.md`.
   */
  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  function isShape(v) {
    return Array.isArray(v) && v.length > 0 && v.every(function (x) {
      return typeof x === 'number' && isFinite(x) && x > 0 && Math.floor(x) === x;
    });
  }

  function sameShape(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var steps = (trace && trace.steps) || [];
    var withRes = steps.filter(function (s) { return s && s.residual !== undefined; });
    var withLn = steps.filter(function (s) { return s && s.ln !== undefined; });
    var dec = ((trace || {}).meta || {}).decoder;

    /* The gate. A trace with any L04 step field must declare `meta.decoder`;
     * a trace with none of them lints clean and says so. */
    if (!withRes.length && !withLn.length) {
      if (dec !== undefined) {
        gaps.push({ what: 'meta.decoder 存在，但没有任何步骤带 step.residual / step.ln',
                    why: '声明与数据不符。' });
      }
      return { gaps: gaps, warns: warns, infos: infos, present: false };
    }
    if (dec === undefined) {
      gaps.push({ what: 'trace 有步骤带本 lab 的步骤字段，但 meta.decoder 缺失',
                  why: '逐字段的规则无法门控 —— 误伤别的 lab 的 trace 正是这样发生的。' });
      return { gaps: gaps, warns: warns, infos: infos, present: true };
    }
    if (withRes.length) {
      var listed = dec.steps_with_residual;
      if (!Array.isArray(listed) || listed.slice().sort().join(',') !==
          withRes.map(function (s) { return s.id; }).sort().join(',')) {
        gaps.push({ what: 'meta.decoder.steps_with_residual 与真正带 residual 的步骤不一致',
                    why: '声明与数据各说各话。' });
      }
    }

    withRes.forEach(function (s) {
      var sid = s.id, r = s.residual;
      if (!isShape(r.lhs_shape) || !isShape(r.rhs_shape)) {
        gaps.push({ what: '步骤 "' + sid + '" 的残差两侧 shape 不是正整数数组',
                    why: '画不出格子。' });
        return;
      }
      if (!sameShape(r.lhs_shape, r.rhs_shape)) {
        gaps.push({ what: '步骤 "' + sid + '" 的残差两侧 shape 不同：' +
                          shapeText(r.lhs_shape) + ' vs ' + shapeText(r.rhs_shape),
                    why: '残差连接要求两侧完全相同 —— 这正是本视图的论点。' });
      }
      if (!sameShape(r.result_shape, r.lhs_shape)) {
        gaps.push({ what: '步骤 "' + sid + '" 的残差结果 shape ' +
                          shapeText(r.result_shape) + ' 与操作数 ' +
                          shapeText(r.lhs_shape) + ' 不同',
                    why: 'd_model 一路不变，结果必须与输入同形。' });
      }
      if (r.equal !== true) {
        gaps.push({ what: '步骤 "' + sid + '" 的 residual.equal 不是 true',
                    why: '图上那一行确认会显示成否定。' });
      }
      if (!isNum(r.element_loop_max_rel_dev) || r.element_loop_max_rel_dev > 1e-12) {
        gaps.push({ what: '步骤 "' + sid + '" 的逐元素相加与双循环重算不一致（' +
                          r.element_loop_max_rel_dev + '）',
                    why: '「逐元素」这个说法就没有依据了。' });
      }

      var cases = r.cases;
      if (!Array.isArray(cases) || cases.length < 4) {
        gaps.push({ what: '步骤 "' + sid + '" 的 residual.cases 缺失或太短',
                    why: '没有对照，「形状相同」这句话就没有说服力。' });
        return;
      }
      var ids = {}, outcomes = { same: 0, broadcast: 0, error: 0 };
      cases.forEach(function (c) {
        if (ids[c.id]) {
          gaps.push({ what: '步骤 "' + sid + '" 的 residual.cases 有重复 id "' + c.id + '"',
                      why: '两个候选会争抢同一格。' });
        }
        ids[c.id] = true;
        if (!isShape(c.rhs_shape)) {
          gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 的 rhs_shape 非法',
                      why: '画不出格子。' });
        }
        if (c.outcome === 'same') {
          outcomes.same++;
          if (!sameShape(c.rhs_shape, r.lhs_shape)) {
            gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 自称 same，' +
                              '但它的 shape 与操作数不同',
                        why: '同一个词在这一格和别的格含义不同。' });
          }
          if (!sameShape(c.result_shape, r.lhs_shape)) {
            gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 自称 same，' +
                              '结果 shape 却是 ' + shapeText(c.result_shape),
                        why: '形状相同就是形状相同。' });
          }
        } else if (c.outcome === 'broadcast') {
          outcomes.broadcast++;
          /* THE RULE WITH TEETH. A broadcast is "the operand shapes differ and
           * the result took the left one's shape". Checking the result shape
           * alone would call a legitimate per-feature bias a plain add, which
           * is exactly the confusion this picture exists to prevent. */
          if (sameShape(c.rhs_shape, r.lhs_shape)) {
            gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 自称广播，' +
                              '但两侧 shape 完全相同',
                        why: '两侧相同就没有广播可言。' });
          }
          if (!sameShape(c.result_shape, r.lhs_shape)) {
            gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 自称广播，' +
                              '结果 shape ' + shapeText(c.result_shape) +
                              ' 却不是操作数的形状',
                        why: '广播的结果取的是被广播的那一侧的形状。' });
          }
        } else if (c.outcome === 'error') {
          outcomes.error++;
          if (typeof c.message !== 'string' || !c.message.trim()) {
            gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 报错却没有异常文本',
                        why: '那就不是跑出来的，是编的。' });
          }
          if (c.result_shape !== null && c.result_shape !== undefined) {
            gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 报错却有结果 shape',
                        why: '自相矛盾。' });
          }
        } else {
          gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 的 outcome = ' +
                            JSON.stringify(c.outcome),
                      why: '只允许 same / broadcast / error 三种。' });
        }
        if (['same', 'broadcast', 'error'].indexOf(c.torch) === -1) {
          gaps.push({ what: '步骤 "' + sid + '" 候选 "' + c.id + '" 没有 torch 侧结果',
                      why: '「会广播」是一个库的结论，除非另一个库也问了。' });
        }
      });
      ['same', 'broadcast', 'error'].forEach(function (k) {
        if (!outcomes[k]) {
          gaps.push({ what: '步骤 "' + sid + '" 的候选里没有 "' + k + '" 的结果',
                      why: '题目要求演示「shape 不匹配会怎样」，缺了它这一步就只剩下' +
                           '「它们相等」。' });
        }
      });
      if (outcomes.same !== 1) {
        gaps.push({ what: '步骤 "' + sid + '" 有 ' + outcomes.same +
                          ' 个候选被标为 same，真正是残差加的只能有一个',
                    why: '否则「哪一个是残差」就不确定了。' });
      }
      /* The two lists the page prints a count from must agree with the case
       * list they summarise — otherwise the summary line and the rows below it
       * are two independent claims about one fact. */
      var silent = cases.filter(function (c) { return c.outcome === 'broadcast'; })
                        .map(function (c) { return c.id; });
      var raised = cases.filter(function (c) { return c.outcome === 'error'; })
                        .map(function (c) { return c.id; });
      if (JSON.stringify(r.silent) !== JSON.stringify(silent)) {
        gaps.push({ what: '步骤 "' + sid + '" 的 residual.silent 与 cases 里的广播项不符',
                    why: '汇总行和明细行会互相矛盾。' });
      }
      if (JSON.stringify(r.raised) !== JSON.stringify(raised)) {
        gaps.push({ what: '步骤 "' + sid + '" 的 residual.raised 与 cases 里的报错项不符',
                    why: '同上。' });
      }
      if (!silent.length) {
        gaps.push({ what: '步骤 "' + sid + '" 没有静默广播的候选',
                    why: '「不匹配就会崩」这句话本身是错的，得让读者看见。' });
      }
      if (!raised.length) {
        gaps.push({ what: '步骤 "' + sid + '" 没有报错的候选',
                    why: '缺了它，读者只看到一半。' });
      }
    });

    if (withRes.length) {
      infos.push({ what: '残差相加 ' + withRes.length + ' 处，各带一组实测的候选形状',
                   why: '' });
    }
    return { gaps: gaps, warns: warns, infos: infos, present: true };
  }

  /* Each entry breaks one rule above, on a real trace. The names parallel the
   * Python `SABOTAGE_CASES` in `labs/traces/decoder_block.py`, so the two tables
   * can be matched up entry for entry. */
  var SABOTAGE_CASES = {
    'shapeguard residual 块被删掉但 meta 还列着': function (t) {
      delete t.steps[5].residual;
    },
    'shapeguard 两侧 shape 不同': function (t) {
      var r = t.steps[5].residual;
      r.rhs_shape = r.lhs_shape.slice(0, -1).concat([Math.max(1, r.lhs_shape[1] - 1)]);
    },
    'shapeguard 结果 shape 与操作数不同': function (t) {
      var r = t.steps[5].residual;
      r.result_shape = r.lhs_shape.slice(0, -1).concat([Math.max(1, r.lhs_shape[1] - 1)]);
    },
    'shapeguard equal 标记被清掉': function (t) { t.steps[5].residual.equal = false; },
    'shapeguard 逐元素对拍被改坏': function (t) {
      t.steps[5].residual.element_loop_max_rel_dev = 0.5;
    },
    'shapeguard 所有候选都被标成 same': function (t) {
      t.steps[5].residual.cases.forEach(function (c) {
        c.outcome = 'same';
        c.result_shape = t.steps[5].residual.lhs_shape;
      });
    },
    'shapeguard 广播那一条被拿掉': function (t) {
      var r = t.steps[5].residual;
      r.cases.forEach(function (c) {
        if (c.outcome === 'broadcast') {
          c.outcome = 'same'; c.result_shape = r.lhs_shape;
        }
      });
      r.silent = [];
    },
    'shapeguard 报错那一条被拿掉': function (t) {
      var r = t.steps[5].residual;
      r.cases.forEach(function (c) {
        if (c.outcome === 'error') {
          c.outcome = 'same'; c.result_shape = r.lhs_shape;
        }
      });
      r.raised = [];
    },
    'shapeguard 报错那一条没有异常文本': function (t) {
      t.steps[5].residual.cases.forEach(function (c) {
        if (c.outcome === 'error') delete c.message;
      });
    },
    'shapeguard 广播那一条被写成两面同形': function (t) {
      var r = t.steps[5].residual;
      r.cases.forEach(function (c) {
        if (c.outcome === 'broadcast') c.rhs_shape = r.lhs_shape.slice();
      });
    },
    'shapeguard 广播那一条的结果取了别的形状': function (t) {
      var r = t.steps[5].residual;
      r.cases.forEach(function (c) {
        if (c.outcome === 'broadcast') {
          c.result_shape = c.rhs_shape.slice();
        }
      });
    },
    'shapeguard silent 列表被清空': function (t) { t.steps[5].residual.silent = []; },
    'shapeguard raised 列表被清空': function (t) { t.steps[5].residual.raised = []; },
    'shapeguard 某条候选缺 torch 结果': function (t) {
      delete t.steps[5].residual.cases[0].torch;
    },
    'shapeguard 某条候选的 outcome 不认识': function (t) {
      t.steps[5].residual.cases[0].outcome = 'maybe';
    },
    'shapeguard 候选 id 重复': function (t) {
      t.steps[5].residual.cases[1].id = t.steps[5].residual.cases[0].id;
    },
    'shapeguard 两处残差只查了一处': function (t) {
      var r = t.steps[12].residual;
      r.result_shape = r.lhs_shape.slice(0, -1).concat([Math.max(1, r.lhs_shape[1] - 1)]);
    },
    'shapeguard meta.decoder 丢了 residual 清单': function (t) {
      delete t.meta.decoder.steps_with_residual;
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

  NS.shapeGuard = {
    makeView: makeView,
    panel: panel,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES,
    shapeText: shapeText,
    elems: elems
  };
})(typeof window !== 'undefined' ? window : globalThis);
