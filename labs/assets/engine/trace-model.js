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

  /* The fields L00's correction-factor amplifier and comparison panel read.
   * Named here rather than inline so a missing one is reported as a named gap
   * instead of surfacing as an undefined in the page. */
  var CORR_KEYS = ['kind', 'factor', 'm_old', 'm_new', 'o_correct', 'o_uncorrected',
                   'bias_abs', 'bias_rel'];
  var CORR_KINDS = ['init', 'identity', 'rescale'];
  var METHOD_IDS = ['naive', 'safe', 'online'];

  /* A trace value: a finite number, null (not produced yet), or one of the
   * three sentinels. A non-finite number is NOT acceptable even though bare
   * `NaN` / `Infinity` are legal JavaScript — they are illegal JSON, and the
   * sentinel vocabulary exists precisely to replace them. */
  function valueOk(v) {
    if (v === null || v === undefined) return true;
    if (typeof v === 'string') return isSentinel(v);
    return typeof v === 'number' && isFinite(v);
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var declared = {};
    Object.keys(trace.tensors || {}).forEach(function (t) { declared[t] = true; });

    var stepIds = {};
    trace.steps.forEach(function (s) { stepIds[s.id] = true; });

    var cfgN = ((trace.meta || {}).config || {}).N;

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

      // The correction factor. It may only ride on a step whose formula shows
      // it, and its numbers must be real: the amplifier's entire claim is that
      // the deviation is measured rather than described.
      var corr = s.corr;
      if (corr) {
        var corrKeys = Object.keys(corr).sort().join(',');
        if (corrKeys !== CORR_KEYS.slice().sort().join(',')) {
          gaps.push({
            what: '步骤 "' + sid + '" 的 corr 字段是 {' + corrKeys + '} 而不是 {' +
                  CORR_KEYS.join(', ') + '}',
            why: '放大器按固定字段读取，缺一个就会显示 undefined。'
          });
        }
        if (CORR_KINDS.indexOf(corr.kind) === -1) {
          gaps.push({
            what: '步骤 "' + sid + '" 的 corr.kind = ' + JSON.stringify(corr.kind) +
                  ' 不属于 ' + CORR_KINDS.join(' / '),
            why: '这个字段决定高亮的颜色与解释文案。'
          });
        }
        if (!(s.regions || {}).CORR) {
          gaps.push({
            what: '步骤 "' + sid + '" 带了 corr，却没有 \\region{CORR} 的说明',
            why: '被框住的那一段没有文字解释。'
          });
        }
        CORR_KEYS.forEach(function (k) {
          if (k === 'kind') return;
          if (!valueOk(corr[k])) {
            gaps.push({
              what: '步骤 "' + sid + '" 的 corr.' + k + ' = ' + JSON.stringify(corr[k]) +
                    ' 不是数、null 或哨兵字符串',
              why: '裸 NaN / Infinity 是非法 JSON —— 用哨兵字符串。'
            });
          }
        });
        if (corr.kind === 'identity' && corr.factor !== 1) {
          gaps.push({
            what: '步骤 "' + sid + '" 的 corr.kind 是 identity，但因子是 ' +
                  JSON.stringify(corr.factor) + ' 而不是 1',
            why: 'm 没变时 e^(m_old − m_new) 必然是 1 —— 两个字段在互相矛盾。'
          });
        }
        if (corr.kind === 'rescale' && !(typeof corr.factor === 'number' && corr.factor < 1)) {
          gaps.push({
            what: '步骤 "' + sid + '" 的 corr.kind 是 rescale，因子 ' +
                  JSON.stringify(corr.factor) + ' 却不小于 1',
            why: 'm 抬高时 e^(m_old − m_new) 必然小于 1。'
          });
        }
      }
    });

    // ---- progress axis: k is a real HBM read count and can only grow.
    var n = cfgN === undefined ? null : cfgN;
    var lastK = 0;
    trace.steps.forEach(function (s) {
      if (typeof s.k !== 'number') {
        gaps.push({
          what: '步骤 "' + s.id + '" 没有 k（截至本步已从 HBM 读取的元素数）',
          why: '对照面板无法把它对齐到共同的时间轴。'
        });
        return;
      }
      if (n !== null && (s.k < 0 || s.k > n || s.k !== Math.floor(s.k))) {
        gaps.push({
          what: '步骤 "' + s.id + '" 的 k = ' + s.k + ' 不是 0..' + n + ' 的整数',
          why: 'k 是已读元素数，落在输入长度之外没有意义。'
        });
      } else if (s.k < lastK) {
        gaps.push({
          what: '步骤 "' + s.id + '" 的 k = ' + s.k + ' 比上一步的 ' + lastK + ' 还小',
          why: 'k 是已读元素数，只能单调不减。'
        });
      }
      lastK = Math.max(lastK, s.k);
    });
    if (n !== null && trace.steps.length &&
        trace.steps[trace.steps.length - 1].k !== n) {
      gaps.push({
        what: '最后一步的 k = ' + trace.steps[trace.steps.length - 1].k +
              ' 而不是 N = ' + n,
        why: '整条输入必须都被读过。'
      });
    }

    // ---- the trace-level correction summary
    var metaCorr = (trace.meta || {}).correction;
    if (!metaCorr || typeof metaCorr !== 'object') {
      gaps.push({
        what: 'meta.correction 缺失',
        why: '放大器面板报不出整条 trace 的最终偏差。'
      });
    } else {
      ['final_o_correct', 'final_o_uncorrected', 'final_bias_abs', 'final_bias_rel',
       'per_block', 'rescales'].forEach(function (key) {
        if (!(key in metaCorr)) {
          gaps.push({
            what: 'meta.correction 缺 ' + key,
            why: '放大器的结论行会显示 undefined。'
          });
        }
      });
      var perBlock = metaCorr.per_block;
      var nBlocks = ((trace.meta || {}).config || {}).n_blocks;
      if (!Array.isArray(perBlock)) {
        gaps.push({ what: 'meta.correction.per_block 不是数组', why: '每块一格的条带画不出来。' });
      } else {
        if (nBlocks !== undefined && perBlock.length !== nBlocks) {
          gaps.push({
            what: 'meta.correction.per_block 有 ' + perBlock.length + ' 项，应该是 n_blocks = ' +
                  nBlocks + ' 项',
            why: '条带的格数与块数对不上。'
          });
        }
        perBlock.forEach(function (e, i) {
          if (CORR_KINDS.indexOf(e && e.kind) === -1) {
            gaps.push({
              what: 'meta.correction.per_block[' + i + '].kind = ' +
                    JSON.stringify(e && e.kind) + ' 非法',
              why: '条带按 kind 上色。'
            });
          }
        });
        var stepped = perBlock.filter(function (e) { return e && e.kind === 'rescale'; }).length;
        if (metaCorr.rescales !== stepped) {
          gaps.push({
            what: 'meta.correction.rescales = ' + JSON.stringify(metaCorr.rescales) +
                  ' 与 per_block 里 rescale 的项数 ' + stepped + ' 不一致',
            why: '结论行的「几块真的做了缩放」会与条带矛盾。'
          });
        }
      }
    }

    // ---- the comparison block: three real runs on one shared axis
    var cmp = trace.compare;
    if (!cmp || typeof cmp !== 'object') {
      gaps.push({ what: 'compare 缺失', why: '对照模式没有数据。' });
    } else {
      var methods = cmp.methods || [];
      var ids = methods.map(function (m) { return m.id; });
      if (ids.join(',') !== METHOD_IDS.join(',')) {
        gaps.push({
          what: 'compare.methods 的 id 是 [' + ids.join(', ') + ']，应该是 [' +
                METHOD_IDS.join(', ') + ']',
          why: '顺序也参与渲染 —— 三行要按固定顺序并排。'
        });
      }
      methods.forEach(function (m) {
        var frames = m.frames || [];
        if (!frames.length) {
          gaps.push({ what: 'compare.methods["' + m.id + '"] 没有 frames', why: '这一行画不出轨迹。' });
          return;
        }
        if (frames[0].k !== 0 || frames[frames.length - 1].k !== m.reads) {
          gaps.push({
            what: 'compare.methods["' + m.id + '"] 的 frames 覆盖 k = ' + frames[0].k + '..' +
                  frames[frames.length - 1].k + '，应该从 0 到它自己的 reads = ' + m.reads,
            why: '横轴对不齐，三条轨迹没法在同一 k 上比较。'
          });
        }
        if (m.reads !== m.passes * n) {
          gaps.push({
            what: 'compare.methods["' + m.id + '"] 的 reads = ' + m.reads +
                  ' 与 passes × N = ' + m.passes + ' × ' + n + ' 不一致',
            why: '「读了几遍」是这张对照表的全部论点。'
          });
        }
        if (!valueOk(m.final)) {
          gaps.push({
            what: 'compare.methods["' + m.id + '"].final = ' + JSON.stringify(m.final) + ' 非法',
            why: '裸 NaN / Infinity 是非法 JSON。'
          });
        }
        var ks = frames.map(function (f) { return f.k; });
        for (var i = 1; i < ks.length; i++) {
          if (ks[i] <= ks[i - 1]) {
            gaps.push({
              what: 'compare.methods["' + m.id + '"] 的 frames 在 k = ' + ks[i] + ' 处没有严格递增',
              why: '同一个 k 有两帧时取哪一帧是不确定的。'
            });
            break;
          }
        }
        // Every value a frame carries is printed at some point in the replay,
        // so each one has to satisfy the same value contract as `state`.
        frames.forEach(function (f) {
          Object.keys(f).forEach(function (key) {
            if (key === 'k') return;
            if (!valueOk(f[key])) {
              gaps.push({
                what: 'compare.methods["' + m.id + '"] 的 frame k = ' + f.k +
                      ' 字段 ' + key + ' = ' + JSON.stringify(f[key]) + ' 非法',
                why: '裸 NaN / Infinity 是非法 JSON —— 用哨兵字符串。'
              });
            }
          });
        });
      });
      var byId = {};
      methods.forEach(function (m) { byId[m.id] = m; });

      /* One axis is shared across all three rows, so it has to span the widest
       * method. Understating it clips the other rows' bars, with no other
       * symptom — so it is a gap, not a warning. */
      if (methods.length) {
        var widest = Math.max.apply(null, methods.map(function (m) { return m.reads || 0; }));
        if (cmp.axis_max !== widest) {
          gaps.push({
            what: 'compare.axis_max = ' + JSON.stringify(cmp.axis_max) +
                  '，应该是三条轨迹里最大的 reads = ' + widest,
            why: '横轴共用，取小了会把其它轨迹的条裁掉。'
          });
        }
      }

      if ((byId.naive || {}).final === SENTINEL.POS_INF) {
        warns.push({
          what: 'compare 的朴素版最终值是 +∞ 而不是 NaN',
          why: '检查是否真的复现了「指数和先溢出、再 +∞/+∞」这条路径。'
        });
      }

      // The shift-invariance control. This is what makes the naive row's NaN a
      // lesson about logit magnitude rather than evidence of broken code.
      var si = cmp.shift_invariance;
      if (!si || typeof si !== 'object') {
        gaps.push({
          what: 'compare.shift_invariance 缺失',
          why: '对照面板说不清朴素版是量级问题还是代码本来就写错了。'
        });
      } else {
        ['offset', 'unshifted_final', 'unshifted_sum_exp', 'shifted_final',
         'shifted_sum_exp', 'reference'].forEach(function (key) {
          if (!(key in si)) {
            gaps.push({
              what: 'compare.shift_invariance 缺 ' + key,
              why: '对照组的那句说明会显示 undefined。'
            });
          }
        });
        for (var sk in si) {
          if (sk === 'offset' || sk === 'note') continue;
          if (!valueOk(si[sk])) {
            gaps.push({
              what: 'compare.shift_invariance.' + sk + ' = ' + JSON.stringify(si[sk]) + ' 非法',
              why: '裸 NaN / Infinity 是非法 JSON。'
            });
          }
        }
        if (typeof si.shifted_final === 'number' && isFinite(si.shifted_final)) {
          gaps.push({
            what: 'compare.shift_invariance.shifted_final = ' + si.shifted_final + ' 是有限值',
            why: '加上偏移的朴素路径本该溢出，否则这个对照没有说服力。'
          });
        }
        var onlineFinal = (byId.online || {}).final;
        ['unshifted_final', 'reference'].forEach(function (key) {
          if (typeof si[key] === 'number' && typeof onlineFinal === 'number' &&
              Math.abs(si[key] - onlineFinal) > 1e-9 * Math.max(1, Math.abs(onlineFinal))) {
            gaps.push({
              what: 'compare.shift_invariance.' + key + ' = ' + si[key] +
                    ' 与 online 版的终值 ' + onlineFinal + ' 不一致',
              why: '对照组的「本该算对」不成立 —— 页面会自相矛盾。'
            });
          }
        });
      }
    }

    infos.push({
      what: trace.steps.length + ' 步 / ' + (trace.graph.nodes || []).length + ' 节点 / ' +
            (trace.graph.edges || []).length + ' 数据边' +
            (metaCorr && typeof metaCorr.rescales === 'number'
              ? ' / ' + metaCorr.rescales + ' 次修正' : ''),
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
