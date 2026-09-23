/* L00 engine · trace model — pure, no DOM.
 *
 * This file answers the one question the whole architecture rests on:
 *
 *     given (trace, cursor), what is every tensor's value?
 *
 * and answers it as a PURE FUNCTION. There is no accumulator, no undo log, no
 * "previous step". Jumping from step N back to step 1 recomputes everything
 * from the trace, so it cannot disagree with having played forward to step 1.
 *
 * Two things make that possible, and both are contract requirements on the
 * trace rather than cleverness here:
 *
 *   - `tensors[].init` gives an initial value to anything a step reads but no
 *     step writes (the input vector, typically). Without it, render(trace, 0)
 *     is wrong from the first step.
 *   - `step.state` records every tensor value that CHANGED during that step,
 *     in full. Full values, never deltas — a delta would need the inverse
 *     operation this design exists to not need.
 *
 * A note on scale: resolve() is O(steps) per call. At the 10–40 steps a lab
 * ships that is a few hundred operations, far below noticing. It is also the
 * version that is obviously correct, which matters more here than a faster one.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});

  /* JSON cannot represent ±∞ or NaN, so the trace contract spells them as
   * strings. These three are the whole vocabulary; `lint` rejects anything
   * else that looks like a sentinel. */
  var SENTINEL = {
    NEG_INF: '-\\infty',
    POS_INF: '+\\infty',
    NAN: 'NaN'
  };

  /* How a sentinel reads to a human, in the tensor panel. The formula panel
   * never needs this: there the sentinel already IS LaTeX and goes to KaTeX. */
  var SENTINEL_TEXT = {
    '-\\infty': '−∞',
    '+\\infty': '+∞',
    'NaN': 'NaN'
  };

  function isSentinel(v) {
    return typeof v === 'string' && Object.prototype.hasOwnProperty.call(SENTINEL_TEXT, v);
  }

  /* ============================================================ index
   *
   * Precomputed, trace-shaped facts the rest of the engine reads. Built once.
   * Nothing here depends on the cursor.
   */
  function index(trace) {
    if (!trace || !Array.isArray(trace.steps)) {
      throw new Error('trace-model: trace.steps must be an array');
    }
    if (!trace.graph || !Array.isArray(trace.graph.nodes)) {
      throw new Error('trace-model: trace.graph.nodes must be an array');
    }

    var stepOf = {};
    trace.steps.forEach(function (s, i) {
      if (Object.prototype.hasOwnProperty.call(stepOf, s.id)) {
        throw new Error('trace-model: duplicate step id "' + s.id + '"');
      }
      stepOf[s.id] = i;
    });

    /* Graph nodes are joined to steps by id. The trace deliberately does not
     * carry a redundant `step` index on each node: a second copy of the same
     * fact can only ever drift out of sync with the first. */
    var nodes = trace.graph.nodes.map(function (n) {
      var step = Object.prototype.hasOwnProperty.call(stepOf, n.id) ? stepOf[n.id] : -1;
      return {
        id: n.id,
        step: step,
        kind: n.kind || 'op',
        op: n.op || '',
        title: n.title || n.id,
        phase: n.phase || '',
        label: n.label || n.id
      };
    });

    var placedIds = {};
    nodes.forEach(function (n) { placedIds[n.id] = true; });

    var edges = (trace.graph.edges || []).filter(function (e) {
      return placedIds[e.from] && placedIds[e.to];
    }).map(function (e) {
      return { from: e.from, to: e.to, tensor: e.tensor || '' };
    });

    var tensorNames = Object.keys(trace.tensors || {});

    return {
      trace: trace,
      stepOf: stepOf,
      nodes: nodes,
      edges: edges,
      tensorNames: tensorNames,
      lastStep: trace.steps.length - 1
    };
  }

  /* ============================================================ resolve
   *
   * The pure reconstruction. For each tensor: the value written by the latest
   * step at or before `upto`, falling back to `tensors[].init`.
   *
   * `from` is the writing step index (-1 for init, null for never-set). It is
   * what lets the tensor panel say where a value came from without a second
   * scan — the prototype had to rescan to answer that.
   */
  function resolve(idx, upto) {
    var trace = idx.trace;
    var out = {};
    var i, name;

    for (i = 0; i < idx.tensorNames.length; i++) {
      name = idx.tensorNames[i];
      out[name] = { value: undefined, from: null, source: 'none' };
    }
    for (i = 0; i < idx.tensorNames.length; i++) {
      name = idx.tensorNames[i];
      var spec = trace.tensors[name];
      if (spec && Object.prototype.hasOwnProperty.call(spec, 'init')) {
        out[name] = { value: spec.init, from: -1, source: 'init' };
      }
    }

    var last = Math.min(upto, idx.lastStep);
    for (i = 0; i <= last; i++) {
      var state = trace.steps[i].state;
      if (!state) continue;
      for (var key in state) {
        if (!Object.prototype.hasOwnProperty.call(state, key)) continue;
        if (!out[key]) out[key] = { value: undefined, from: null, source: 'none', stray: true };
        out[key].value = state[key];
        out[key].from = i;
        out[key].source = 'write';
      }
    }
    return out;
  }

  /* Which tensors a step changes. Used for the panel's read/write badges. */
  function toNumber(v) {
    if (typeof v === 'number') return v;
    if (v === SENTINEL.NEG_INF) return -Infinity;
    if (v === SENTINEL.POS_INF) return Infinity;
    if (v === SENTINEL.NAN) return NaN;
    return NaN;
  }

  /* ============================================================ lint
   *
   * The author-side contract check. Python (labs/traces/online_softmax.py) is
   * the source of truth — it runs before a trace is written and in CI. This is
   * a port so a lab author editing a trace in the browser gets the same verdict
   * without a Python round trip. The two must agree; when a rule changes, it
   * changes in both.
   */
  var SLOT_RE = /\\slot\{([A-Za-z0-9_]+)\}/g;
  var REGION_RE = /\\region\{([A-Za-z0-9_]+)\}/g;
  var LATEX_IN_TEXT_RE = /\\[a-zA-Z]+\{/;
  var BAD_LITERAL_RE = /^(NaN|Infinity|-Infinity|undefined|null)$/;

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var declared = {};
    Object.keys(trace.tensors || {}).forEach(function (t) { declared[t] = true; });

    var stepIds = {};
    trace.steps.forEach(function (s) { stepIds[s.id] = true; });

    var written = {}, read = {};
    trace.steps.forEach(function (s) {
      (s.writes || []).forEach(function (t) { written[t] = true; });
      (s.reads || []).forEach(function (t) { read[t] = true; });
    });

    Object.keys(read).sort().forEach(function (t) {
      if (!declared[t]) {
        gaps.push({
          what: '张量 "' + t + '" 被读（reads[]），但 tensors[] 里根本没有声明',
          why: '引擎不知道它的 shape / 位置，也没有初始值可以渲染。'
        });
      } else if (!written[t] && !Object.prototype.hasOwnProperty.call(trace.tensors[t], 'init')) {
        gaps.push({
          what: '张量 "' + t + '" 被读但从未被写，且 tensors[] 里没有 init',
          why: 'render(trace, 0) 无法得知它的值 —— 纯函数重建从第一步起就是错的。'
        });
      }
    });
    Object.keys(written).sort().forEach(function (t) {
      if (!declared[t]) {
        gaps.push({
          what: '步骤写了张量 "' + t + '"，但 tensors[] 里没有声明',
          why: '引擎不知道它的 shape / 位置，无法渲染。'
        });
      }
    });

    var nodeIds = {};
    (trace.graph.nodes || []).forEach(function (n) { nodeIds[n.id] = true; });
    trace.steps.forEach(function (s) {
      if (!nodeIds[s.id]) {
        gaps.push({
          what: '步骤 "' + s.id + '" 在 graph.nodes 里没有对应节点',
          why: 'DAG 与时间轴不同步 —— 跳到该步时图上没有可高亮的节点。'
        });
      }
    });
    (trace.graph.nodes || []).forEach(function (n) {
      if (!stepIds[n.id]) {
        warns.push({
          what: '图节点 "' + n.id + '" 没有对应步骤',
          why: '点击它无处可跳。'
        });
      }
    });

    trace.steps.forEach(function (s) {
      var sid = s.id;
      var formula = s.formula;
      if (!formula) {
        gaps.push({ what: '步骤 "' + sid + '" 没有 formula', why: '公式面板空着。' });
        return;
      }
      ['sym', 'idx', 'num'].forEach(function (tier) {
        if (typeof formula[tier] !== 'string') {
          gaps.push({ what: '步骤 "' + sid + '" 缺 formula.' + tier, why: '三档显示会缺一档。' });
          return;
        }
        var m;
        SLOT_RE.lastIndex = 0;
        while ((m = SLOT_RE.exec(formula[tier]))) {
          var key = m[1];
          var bdg = (s.bindings || {})[key];
          if (!bdg) {
            gaps.push({
              what: '步骤 "' + sid + '" 的 formula.' + tier + ' 引用了 \\slot{' + key + '}，但 bindings 里没有',
              why: '公式上会留下未替换的占位符。'
            });
          } else if (bdg[tier] === undefined || bdg[tier] === null) {
            gaps.push({
              what: '步骤 "' + sid + '" 的绑定 "' + key + '" 缺 "' + tier + '" 形态',
              why: '三档中的某一档没有值可填。'
            });
          } else if (BAD_LITERAL_RE.test(String(bdg[tier]))) {
            gaps.push({
              what: '步骤 "' + sid + '" 绑定 "' + key + '.' + tier + '" = ' + bdg[tier],
              why: 'JSON 不能表示 ±∞ / NaN。用哨兵字符串 "-\\infty"。'
            });
          }
        }
        REGION_RE.lastIndex = 0;
        while ((m = REGION_RE.exec(formula[tier]))) {
          if (!(s.regions || {})[m[1]]) {
            gaps.push({
              what: '步骤 "' + sid + '" 的 formula.' + tier + ' 引用了 \\region{' + m[1] + '}，但 step.regions 里没有',
              why: '被框住的那一段没有文字说明。'
            });
          }
        }
      });

      var used = {};
      ['sym', 'idx', 'num'].forEach(function (tier) {
        var m;
        SLOT_RE.lastIndex = 0;
        while ((m = SLOT_RE.exec(formula[tier] || ''))) used[m[1]] = true;
      });
      Object.keys(s.bindings || {}).forEach(function (k) {
        if (!used[k]) {
          warns.push({
            what: '步骤 "' + sid + '" 的绑定 "' + k + '" 没被任何档位引用',
            why: '冗余数据。'
          });
        }
      });

      Object.keys(s.bindings || {}).forEach(function (k) {
        var bdg = s.bindings[k];
        var keys = Object.keys(bdg).sort().join(',');
        if (keys !== 'idx,num,sym') {
          gaps.push({
            what: '步骤 "' + sid + '" 的绑定 "' + k + '" 字段是 {' + keys + '} 而不是 {sym, idx, num}',
            why: '契约要求显式三元组，不允许靠 key 猜。'
          });
        }
      });

      if (LATEX_IN_TEXT_RE.test(s.narration || '')) {
        warns.push({
          what: '步骤 "' + sid + '" 的 narration 里含 LaTeX 命令',
          why: 'narration 是纯文本，反斜杠会原样显示。'
        });
      }

      Object.keys(s.state || {}).forEach(function (k) {
        if (!declared[k]) {
          gaps.push({
            what: '步骤 "' + sid + '" 的 state 写了未声明张量 "' + k + '"',
            why: '无法渲染。'
          });
        }
        if ((s.writes || []).indexOf(k) === -1) {
          warns.push({
            what: '步骤 "' + sid + '" 的 state 里有 "' + k + '"，但它不在 writes[] 里',
            why: '读写声明与 state 不一致；引擎信 state，writes[] 只用于高亮。'
          });
        }
        var items = Array.isArray(s.state[k]) ? s.state[k] : [s.state[k]];
        items.forEach(function (item) {
          if (typeof item === 'string' && !isSentinel(item)) {
            gaps.push({
              what: '步骤 "' + sid + '" 的 state["' + k + '"] 含未知哨兵字符串 "' + item + '"',
              why: 'state 字符串只允许 -\\infty / +\\infty / NaN。'
            });
          }
        });
      });

      Object.keys(s.regions || {}).forEach(function (key) {
        var referenced = ['sym', 'idx', 'num'].some(function (t) {
          return (formula[t] || '').indexOf('\\region{' + key + '}') !== -1;
        });
        if (!referenced) {
          warns.push({
            what: '步骤 "' + sid + '" 声明了 region "' + key + '" 但公式里没有引用',
            why: '冗余数据。'
          });
        }
      });
    });

    infos.push({
      what: trace.steps.length + ' 步 / ' + (trace.graph.nodes || []).length + ' 节点 / ' +
            (trace.graph.edges || []).length + ' 数据边',
      why: ''
    });
    infos.push({
      what: 'state 写过的张量：' + Object.keys(written).sort().join(', '),
      why: '纯函数重建需要覆盖的就是这些。'
    });

    return { gaps: gaps, warns: warns, infos: infos };
  }

  NS.traceModel = {
    SENTINEL: SENTINEL,
    SENTINEL_TEXT: SENTINEL_TEXT,
    isSentinel: isSentinel,
    toNumber: toNumber,
    index: index,
    resolve: resolve,
    lint: lint
  };
})(typeof window !== 'undefined' ? window : globalThis);
