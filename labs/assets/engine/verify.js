/* L00 engine · self-check.
 *
 * The engine's central claim is that render(trace, i) is a pure function of
 * (trace, i), so jumping anywhere is exactly as correct as playing forward
 * there. This file is how that claim is checked, and — more importantly — how
 * the check is kept honest.
 *
 * A GREEN RUN ON ITS OWN PROVES NOTHING. A test that is mis-wired, or that
 * compares a value against itself, passes forever. So every run also executes a
 * CONTROL GROUP: a deliberately WRONG player that keeps one mutable state object
 * and, on a jump, only applies the target step's writes on top of whatever it
 * was already showing. It never re-reads earlier steps, so jumping backwards
 * leaves stale values behind. That is precisely the failure the pure design
 * exists to prevent.
 *
 * The result is only reported as a pass when the pure function agrees everywhere
 * AND the control group disagrees somewhere. If both pass, the jump set has no
 * discriminating power and the run is reported as inconclusive rather than
 * green.
 *
 * The same shape is used for the trace lint: the clean trace is checked, and a
 * set of deliberately broken copies must be caught, or the lint is reported as
 * dead.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var TM = NS.traceModel;

  function deepEqual(a, b) {
    if (a === b) return true;
    if (typeof a !== typeof b) return false;
    if (a === null || b === null) return false;
    if (typeof a !== 'object') return false;
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    if (Array.isArray(a)) {
      if (a.length !== b.length) return false;
      for (var i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false;
      return true;
    }
    var ka = Object.keys(a).sort();
    var kb = Object.keys(b).sort();
    if (ka.length !== kb.length) return false;
    for (var j = 0; j < ka.length; j++) {
      if (ka[j] !== kb[j]) return false;
      if (!deepEqual(a[ka[j]], b[kb[j]])) return false;
    }
    return true;
  }

  function clone(v) {
    return v === undefined ? undefined : JSON.parse(JSON.stringify(v));
  }

  /* ---------------------------------------------------------------- jump set
   *
   * Deliberately not "0..N in order". The paths that break a naive player are
   * the ones that move backwards or skip: a full reverse walk, jumps to the
   * head and tail, and a jumbled loop. All of them are exercised.
   */
  function jumpPlan(lastStep) {
    var plan = [];
    var i;
    for (i = 0; i <= lastStep; i++) plan.push({ label: '正向 → ' + i, target: i });
    for (i = lastStep; i >= 0; i--) plan.push({ label: '完全逆序 → ' + i, target: i });
    plan.push({ label: '跳到尾 → ' + lastStep, target: lastStep });
    plan.push({ label: '跳回 → 0', target: 0 });
    if (lastStep >= 6) {
      plan.push({ label: '5 → 1（跳回）', target: 1 });
      plan.push({ label: '1 → 5（回环）', target: 5 });
      plan.push({ label: '5 → 3（乱序）', target: 3 });
      plan.push({ label: '3 → 3（原地）', target: 3 });
      plan.push({ label: '3 → 5（前进）', target: 5 });
      plan.push({ label: '5 → 2（回环）', target: 2 });
      plan.push({ label: '2 → 5（回环）', target: 5 });
      plan.push({ label: '5 → 0（回到头）', target: 0 });
    }
    return plan;
  }

  /**
   * Score the pure reconstruction against a forward-accumulated ground truth.
   *
   * The ground truth is built by mutating one object step by step — i.e. it is
   * the thing a correct sequential player would be showing. If the pure
   * function and this agree at every jump target, jumping is indistinguishable
   * from having played there.
   */
  function runJumpCheck(trace, idx) {
    var names = idx.tensorNames;

    /* ground truth: walk forward, mutating */
    var acc = {};
    names.forEach(function (n) {
      if (trace.tensors[n] && Object.prototype.hasOwnProperty.call(trace.tensors[n], 'init')) {
        acc[n] = clone(trace.tensors[n].init);
      }
    });
    var truth = [];
    for (var i = 0; i < trace.steps.length; i++) {
      var st = trace.steps[i].state || {};
      Object.keys(st).forEach(function (k) { acc[k] = clone(st[k]); });
      truth.push(clone(acc));
    }

    function expected(name, i) {
      if (truth[i] && Object.prototype.hasOwnProperty.call(truth[i], name)) return truth[i][name];
      var spec = trace.tensors[name];
      if (spec && Object.prototype.hasOwnProperty.call(spec, 'init')) return spec.init;
      return undefined;
    }

    /* The control group. One mutable object, writes applied on top of whatever
     * is already there, no re-derivation. Seeded from init so it starts fair. */
    var controlState = {};
    names.forEach(function (n) {
      if (trace.tensors[n] && Object.prototype.hasOwnProperty.call(trace.tensors[n], 'init')) {
        controlState[n] = clone(trace.tensors[n].init);
      }
    });

    var plan = jumpPlan(idx.lastStep);
    var rows = [];
    var pureFailures = 0;
    var controlFailures = 0;

    plan.forEach(function (jump) {
      var resolved = TM.resolve(idx, jump.target);
      var pureDetail = [];
      names.forEach(function (name) {
        var got = resolved[name] ? resolved[name].value : undefined;
        if (!deepEqual(got, expected(name, jump.target))) {
          pureDetail.push(name + '：重建=' + JSON.stringify(got) +
                          '，应=' + JSON.stringify(expected(name, jump.target)));
        }
      });

      var stepState = trace.steps[jump.target].state || {};
      Object.keys(stepState).forEach(function (k) { controlState[k] = clone(stepState[k]); });
      var controlDetail = [];
      names.forEach(function (name) {
        if (!deepEqual(controlState[name], expected(name, jump.target))) {
          if (controlDetail.length < 3) {
            controlDetail.push(name + '：显示=' + JSON.stringify(controlState[name]) +
                               '，应=' + JSON.stringify(expected(name, jump.target)));
          }
        }
      });

      if (pureDetail.length) pureFailures++;
      if (controlDetail.length) controlFailures++;

      rows.push({
        label: jump.label,
        target: jump.target,
        pureOk: pureDetail.length === 0,
        controlOk: controlDetail.length === 0,
        pureDetail: pureDetail.slice(0, 3).join('；'),
        controlDetail: controlDetail.join('；')
      });
    });

    return {
      total: plan.length,
      pureFailures: pureFailures,
      controlFailures: controlFailures,
      /* Only meaningful when the control group actually failed somewhere. */
      conclusive: controlFailures > 0,
      passed: pureFailures === 0 && controlFailures > 0,
      rows: rows
    };
  }

  /* ------------------------------------------------------------- lint check
   *
   * Each mutation breaks one rule. The lint must catch every one, or the clean
   * verdict it gives the real trace means nothing.
   */
  var SABOTAGES = {
    '删掉被读张量的声明与 init': function (t) { delete t.tensors.x; },
    'state 写入未声明的张量': function (t) { t.steps[2].state.ghost = 0; },
    '绑定缺一档': function (t) { delete t.steps[2].bindings.MOLD.idx; },
    '绑定降成二元组': function (t) { t.steps[2].bindings.MNEW = { num: '0.83' }; },
    '公式引用不存在的 slot': function (t) { t.steps[2].formula.num += '\\slot{GHOST}'; },
    '公式引用不存在的 region': function (t) { t.steps[3].formula.sym += '\\region{GHOST}{x}'; },
    '步骤没有对应图节点': function (t) { t.graph.nodes.splice(0, 1); },
    '缺一个档位的公式': function (t) { delete t.steps[0].formula.idx; },
    'state 用裸 -Infinity': function (t) { t.steps[0].state.m = '-Infinity'; },
    'state 用未知哨兵串': function (t) { t.steps[0].state.m = '-inf'; }
  };

  function runLintCheck(trace) {
    var clean = TM.lint(trace);
    var caught = [];
    var missed = [];
    Object.keys(SABOTAGES).forEach(function (name) {
      var copy = clone(trace);
      try {
        SABOTAGES[name](copy);
      } catch (err) {
        missed.push(name + '（破坏本身失败：' + err.message + '）');
        return;
      }
      var result = TM.lint(copy);
      if (result.gaps.length > 0) caught.push(name);
      else missed.push(name);
    });
    return {
      cleanGaps: clean.gaps.length,
      cleanWarns: clean.warns.length,
      caught: caught,
      missed: missed,
      /* A clean trace plus a live lint is the only combination that means
       * anything. */
      passed: clean.gaps.length === 0 && missed.length === 0
    };
  }

  /* ------------------------------------------------------------------ report
   *
   * Renders the result as a panel. The page mounts it; the acceptance run reads
   * `window.__labVerify` for the same numbers rather than scraping the DOM.
   */
  function panel(host, trace, idx) {
    var jumps = runJumpCheck(trace, idx);
    var lint = runLintCheck(trace);
    var clean = TM.lint(trace);

    var html = '';

    html += '<div class="lab-verdict ' + (jumps.passed ? 'lab-ok' : 'lab-bad') + '">';
    html += '<b>任意跳转 · 纯函数重建：</b>' + jumps.total + ' 次跳转（正向 / 完全逆序 / 跳回 / 乱序回环），' +
            (jumps.pureFailures === 0
              ? '<b>0 次不一致</b>，与顺序累积基准逐位相同。'
              : '<b>' + jumps.pureFailures + ' 次不一致</b>。');
    html += '<br><b>对照组（故意做错的有状态播放器）：</b>同一批跳转里 <b>' +
            jumps.controlFailures + ' 次不一致</b>。';
    if (!jumps.conclusive) {
      html += '<br><b>⚠️ 对照组也全过了 —— 这组跳转没有区分力，上面的 0 不能当作证据。</b>';
    } else if (jumps.pureFailures === 0) {
      html += '对照组确实翻车，说明上面那个 0 不是测试写错。';
    }
    html += '</div>';

    html += '<table class="lab-vtable"><thead><tr>' +
            '<th>跳转</th><th>纯函数 render(trace,i)</th><th>对照组</th><th>说明</th>' +
            '</tr></thead><tbody>';
    var shown = 0;
    jumps.rows.forEach(function (r) {
      var interesting = !r.pureOk || !r.controlOk || r.target <= 1 ||
                        r.label.indexOf('跳回') !== -1 || r.label.indexOf('乱序') !== -1 ||
                        r.label.indexOf('原地') !== -1 || r.label.indexOf('回环') !== -1;
      if (!interesting || shown >= 18) return;
      shown++;
      html += '<tr>' +
        '<td>' + NS.formula.escapeText(r.label) + '</td>' +
        '<td class="' + (r.pureOk ? 'lab-pass' : 'lab-fail') + '">' +
          (r.pureOk ? '一致' : '不一致') + '</td>' +
        '<td class="' + (r.controlOk ? 'lab-pass' : 'lab-fail') + '">' +
          (r.controlOk ? '一致' : '不一致') + '</td>' +
        '<td class="lab-sub">' + NS.formula.escapeText(r.pureOk ? r.controlDetail : r.pureDetail) + '</td>' +
        '</tr>';
    });
    html += '</tbody></table>';

    html += '<div class="lab-verdict ' + (lint.passed ? 'lab-ok' : 'lab-bad') + '">' +
            '<b>契约 lint：</b>干净的 trace 报 <b>' + lint.cleanGaps + '</b> 个 gap / ' +
            lint.cleanWarns + ' 个 warn；对照组 ' + lint.caught.length + ' 种破坏全部被抓到' +
            (lint.missed.length ? '，<b>漏掉 ' + lint.missed.length + ' 种：' +
              NS.formula.escapeText(lint.missed.join('、')) + '</b>' : '') +
            (lint.passed ? '。lint 是活的，所以上面的 0 有意义。' : '。') +
            '</div>';

    html += '<details class="lab-vdetails"><summary>lint 明细</summary><table class="lab-vtable">' +
            '<thead><tr><th>级别</th><th>发现</th><th>含义</th></tr></thead><tbody>';
    clean.gaps.concat(clean.warns, clean.infos).forEach(function (x) {
      html += '<tr><td class="' + (x.sev === 'gap' ? 'lab-fail' : 'lab-sub') + '">' +
              (x.sev || 'info') + '</td><td>' + NS.formula.escapeText(x.what) + '</td>' +
              '<td class="lab-sub">' + NS.formula.escapeText(x.why) + '</td></tr>';
    });
    html += '</tbody></table></details>';

    host.innerHTML = html;

    var result = {
      jumps: { total: jumps.total, pureFailures: jumps.pureFailures,
               controlFailures: jumps.controlFailures, passed: jumps.passed,
               conclusive: jumps.conclusive },
      lint: { gaps: lint.cleanGaps, warns: lint.cleanWarns,
              caught: lint.caught.length, missed: lint.missed, passed: lint.passed }
    };
    global.__labVerify = global.__labVerify || {};
    global.__labVerify[trace.meta && trace.meta.lab || 'lab'] = result;
    return result;
  }

  NS.verify = {
    runJumpCheck: runJumpCheck,
    runLintCheck: runLintCheck,
    panel: panel,
    jumpPlan: jumpPlan,
    deepEqual: deepEqual
  };
})(typeof window !== 'undefined' ? window : globalThis);
