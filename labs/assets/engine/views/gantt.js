/* L00 engine · view: gantt (时间 × 资源 的调度甘特).
 *
 * One row per resource, one coloured block per unit of time that resource spent
 * doing something. Built for L10's scheduling replay, designed so L16's 1F1B
 * (a row per pipeline stage, micro-batches as blocks) and L17's DDP overlap (a
 * compute track above a communication track) are the same component with a
 * different row list.
 *
 * WHAT IS DERIVED AND WHAT IS DECLARED
 *
 * The bars are DECLARED, on `step.bars`, and they have to be: nothing in the
 * trace's `state` says what a resource was doing during a step, and a scheduler
 * that computed three tokens and one that computed none can leave the same
 * queue lengths behind. See labs/traces/continuous_batching.py for the ledger
 * that the declarations are asserted against.
 *
 * Everything the drawing needs is then DERIVED from those declarations by a
 * fold over steps 0..cursor. That is what makes arbitrary jumps work: there is
 * no accumulated DOM state and no "advance" path, only `render(trace, i)`.
 * Jumping to step 40 rebuilds the whole chart from the trace, so it cannot
 * disagree with having played forward to 40.
 *
 * WHY IT FOLDS INSTEAD OF DRAWING ONE STEP
 *
 * A gantt's whole subject is history: "how long was this row busy" is not a
 * property of the current step. So the view runs `steps[0..i]` through a fold
 * that turns consecutive same-kind bars into spans. The fold is a pure function
 * of (trace, cursor) — the reason it is not a mutable accumulator is exactly the
 * reason the engine has no undo log.
 *
 * Adding one step is O(bars in that step), because the fold is incremental
 * between paints, so playback does not re-scan the run for every frame. The
 * cache is checked against the cursor range it covers and re-seeded from 0 the
 * moment the cursor is not an extension of it — a backwards jump is not a
 * special case, it is just a fold that starts over.
 *
 * THE SHARED TIME AXIS
 *
 * `colWidth` is the same for every gantt on the page, and the axis length comes
 * from the caller (or from the trace itself for a lone chart). Two charts drawn
 * with the same axis are read against each other, which is the entire point of
 * the comparison mode: a reader comparing Static against Continuous at the same
 * column is comparing the same tick.
 *
 * CONFIG
 *
 *   rows:    [{id, label, sub?, track?}]   optional; defaults to trace.gantt.rows
 *   tracks:  [{id, label}]                 optional; groups rows into sub-lanes,
 *                                          so L17 can put compute above comm in
 *                                          one chart instead of two
 *   kinds:   {<kind>: {label, className?}} presentation for the trace's kinds;
 *                                          a kind with no entry still renders,
 *                                          in the default style
 *   colWidth: number                       px per tick (default 16)
 *   rowHeight: number                      px per row (default 15)
 *   axis:     {max, label}                 shared axis; `max` is the tick count
 *   maxTicks: number                       fallback axis length for a lone chart
 *   title:    string
 *   caption:  string
 *   compact:  boolean                      drop per-row sub-labels (comparison)
 *
 * render(cursor, ctx) returns an HTML string, called by the player on every
 * paint with the same (trace, i) it just resolved, so the chart can never show
 * a tick the rest of the page is not on.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;
  var escAttr = NS.formula.escapeAttr;

  /* ================================================================ the fold
   *
   * Turn `steps[0..cursor]` into spans per row. A span is a maximal run of
   * consecutive steps in which one row had the same kind:
   *
   *     {row, kind, from, to, label, tip}
   *
   * `from` is inclusive and `to` exclusive, both in step indices, which is what
   * the geometry needs and what makes "grew this frame" a comparison rather
   * than a second way of counting. `grewAt` marks the span the cursor is
   * currently extending — that is the bar the reader watches grow.
   */
  function emptyState(trace) {
    var order = [], spans = {};
    (trace.gantt.rows || []).forEach(function (r) {
      order.push(r.id);
      spans[r.id] = [];
    });
    return { order: order, spans: spans, last: -1 };
  }

  /**
   * Apply one step to a fold state — the ONLY place a span is ever created or
   * extended.
   *
   * Both the from-scratch fold and the incremental cache call this, so "the
   * cached chart equals the recomputed one" is true by construction rather than
   * by two implementations being kept in step by hand. That matters because the
   * whole reason the cache is allowed to exist is that it cannot change what is
   * drawn; the acceptance harness checks it anyway, at every cursor.
   *
   * `grewAt` is the step at which this span last grew — i.e. the running edge.
   * It is set on every extension, so after a full fold only the spans that the
   * final step extended carry it, and a `grewAt === state.last` test selects
   * exactly the bars the reader is watching move.
   */
  function applyStep(state, trace, i) {
    var step = trace.steps[i];
    var bars = step.bars || [];
    var seen = {};
    bars.forEach(function (bar) {
      if (!state.spans[bar.row]) { state.spans[bar.row] = []; state.order.push(bar.row); }
      seen[bar.row] = bar;
      var list = state.spans[bar.row];
      var cur = list[list.length - 1];
      if (cur && cur.to === i && cur.kind === bar.kind &&
          cur.label === (bar.label || '')) {
        cur.to = i + 1;
        cur.grewAt = i;
      } else {
        list.push({ row: bar.row, kind: bar.kind, from: i, to: i + 1,
                    label: bar.label || '', tip: [], grewAt: i });
      }
    });
    /* A row with no bar this step ends its span. That is the honest reading:
     * "this resource was doing nothing", not "it kept doing the last thing" — a
     * request that left the waiting queue and is now decoding is not still
     * waiting, and a chart that drew it as such would be lying about the one
     * transition the replay exists to show. The span is simply left behind; a
     * later bar for the same row starts a new one, because `cur.to === i` fails
     * on the gap. */
    if (step.narration) {
      Object.keys(seen).forEach(function (row) {
        var list = state.spans[row];
        var cur = list[list.length - 1];
        if (cur && cur.tip.length < 2) cur.tip.push(step.narration);
      });
    }
    state.last = i;
    return state;
  }

  /** The definition: fold steps 0..cursor from nothing. */
  function fold(trace, cursor) {
    var state = emptyState(trace);
    var last = Math.min(cursor, trace.steps.length - 1);
    for (var i = 0; i <= last; i++) applyStep(state, trace, i);
    return state;
  }

  /* ============================================================== geometry */

  /**
   * Pixel geometry for one row's spans, in a chart `cols` ticks wide.
   *
   * Deliberately a pure function of (span, cols, colWidth) with no DOM
   * measurement: every chart on the page agrees on where tick 17 is because they
   * all compute it the same way, not because they were laid out the same way.
   */
  function box(span, colWidth, gap) {
    var x = span.from * colWidth;
    var w = (span.to - span.from) * colWidth - gap;
    if (w < 3) w = 3;          // a one-tick bar still has to be visible
    return { x: x, w: w };
  }

  /* ================================================================ render */

  function kindsOf(trace) {
    return (trace.gantt && trace.gantt.kinds) || [];
  }

  function kindLabel(cfg, trace, kind) {
    var custom = (cfg.kinds || {})[kind];
    if (custom && custom.label) return custom.label;
    var fromTrace = (trace.gantt && trace.gantt.kindsLabel || {})[kind];
    return fromTrace || kind;
  }

  function kindClass(cfg, kind) {
    var custom = (cfg.kinds || {})[kind];
    if (custom && custom.className) return custom.className;
    return '';
  }

  function makeView(config) {
    var cfg = config || {};
    if (!cfg.trace) throw new Error('gantt: config.trace is required');
    var trace = cfg.trace;
    var gantt = trace.gantt;
    if (!gantt || !gantt.rows || !gantt.rows.length) {
      throw new Error('gantt: trace.gantt.rows is required (see the trace contract)');
    }
    var rows = cfg.rows || gantt.rows;
    var tracks = cfg.tracks || gantt.tracks || null;
    var colWidth = cfg.colWidth || 16;
    var rowHeight = cfg.rowHeight || 15;

    /* The fold is memoised against the cursor range it covers, and re-seeded
     * from scratch whenever the cursor is not an extension of it. Playing
     * forward is the cheap path (O(bars in the new step)); every other move —
     * backwards, a jump, a scrub — recomputes from step 0, which is the pure
     * path and therefore the one that defines correctness. Because both paths
     * go through `applyStep`, the memo cannot change what is drawn; it only
     * changes how much work it took to find out. */
    var cache = null;

    function folded(cursor) {
      var target = Math.min(cursor, trace.steps.length - 1);
      if (cache && target >= cache.last && target - cache.last <= 1) {
        for (var i = cache.last + 1; i <= target; i++) applyStep(cache, trace, i);
        return cache;
      }
      cache = fold(trace, cursor);
      return cache;
    }

    function render(cursor, ctx) {
      var state = folded(cursor);
      var cols = (cfg.axis && cfg.axis.max) || cfg.maxTicks || (state.last + 1);
      var width = Math.max(cols * colWidth, 1);
      var html = '';

      html += '<div class="lab-gantt" data-cursor="' + escAttr(String(cursor)) + '"' +
              ' data-axis-max="' + escAttr(String(cols)) + '">';

      if (cfg.title) {
        html += '<div class="lab-gantt-hd"><span class="lab-gantt-title">' +
                esc(cfg.title) + '</span>' +
                (cfg.caption ? '<span class="lab-gantt-caption">' + esc(cfg.caption) +
                 '</span>' : '') + '</div>';
      }

      /* The tick ruler. Every gridline is one tick, and the cursor's own column
       * is marked — that marker is what makes the shared axis readable when two
       * charts are side by side. */
      html += '<div class="lab-gantt-axis" style="width:' + width + 'px">';
      html += '<span class="lab-gantt-cursor" style="left:' +
              ((state.last + 1) * colWidth) + 'px"></span>';
      var stride = cols > 40 ? 5 : (cols > 20 ? 2 : 1);
      for (var c = 0; c <= cols; c += stride) {
        html += '<span class="lab-gantt-tick" style="left:' + (c * colWidth) + 'px">' +
                c + '</span>';
      }
      html += '</div>';

      var drawnTracks = {};
      rows.forEach(function (row) {
        // A track heading is emitted once, before the first of its rows. L10
        // declares no tracks (one chart = one track), so this is inert there and
        // is what L17 will use for its compute-above-communication split.
        if (tracks && row.track && !drawnTracks[row.track]) {
          drawnTracks[row.track] = true;
          var tk = null;
          tracks.forEach(function (t) { if (t.id === row.track) tk = t; });
          html += '<div class="lab-gantt-track" data-track="' + escAttr(row.track) + '">' +
                  esc((tk && tk.label) || row.track) + '</div>';
        }

        html += '<div class="lab-gantt-row" data-row="' + escAttr(row.id) + '"' +
                ' style="height:' + rowHeight + 'px">';
        html += '<div class="lab-gantt-rowlabel" title="' +
                escAttr(row.sub || row.label || row.id) + '">' +
                '<span class="lab-gantt-rowid">' + esc(row.label || row.id) + '</span>';
        if (row.sub && !cfg.compact) {
          html += '<span class="lab-gantt-rowspec">' + esc(row.sub) + '</span>';
        }
        html += '</div>';

        var lane = state.spans[row.id] || [];
        html += '<div class="lab-gantt-lane" style="width:' + width + 'px">';
        lane.forEach(function (span) {
          var g = box(span, colWidth, 1);
          var cls = 'lab-gantt-bar lab-gantt-' + safeKind(span.kind) +
                    (kindClass(cfg, span.kind) ? ' ' + escAttr(kindClass(cfg, span.kind)) : '') +
                    (span.grewAt === state.last ? ' lab-gantt-growing' : '');
          var tip = span.label ? span.label : kindLabel(cfg, trace, span.kind);
          var title = 't' + span.from + '–' + span.to + ' · ' +
                      kindLabel(cfg, trace, span.kind) +
                      (span.label ? ' · ' + span.label : '') +
                      (span.tip.length ? ' — ' + span.tip[0] : '');
          html += '<span class="' + cls + '" style="left:' + g.x + 'px;width:' + g.w +
                  'px" data-kind="' + escAttr(span.kind) + '"' +
                  ' data-from="' + span.from + '" data-to="' + span.to + '"' +
                  ' title="' + escAttr(title) + '">' +
                  (rowHeight >= 14 && g.w >= 14 ? '<i>' + esc(tip) + '</i>' : '') +
                  '</span>';
        });
        html += '</div>';
        html += '</div>';
      });

      if (cfg.legend !== false) html += legend(cfg, gantt, rows, state);
      html += '</div>';
      return html;
    }

    return {
      render: render,
      rows: rows,
      tracks: tracks,
      fold: fold,
      /* Exposed so the acceptance harness can compare the incremental fold
       * against the from-scratch one at every cursor — the claim that they agree
       * is the whole reason the cache is allowed to exist. */
      folded: folded,
      reset: function () { cache = null; }
    };
  }

  function safeKind(kind) {
    return String(kind).replace(/[^A-Za-z0-9_-]/g, '_');
  }

  /* Which kinds the chart can actually show, in the trace's own order, so the
   * legend explains the picture rather than a vocabulary. */
  function legend(cfg, gantt, rows, state) {
    var used = {};
    Object.keys(state.spans).forEach(function (row) {
      state.spans[row].forEach(function (s) { used[s.kind] = true; });
    });
    var kinds = kindsOf(gantt).filter(function (k) { return used[k]; });
    if (!kinds.length) return '';
    return '<div class="lab-gantt-legend">' + kinds.map(function (k) {
      return '<span class="lab-gantt-key"><i class="lab-gantt-swatch lab-gantt-' +
             safeKind(k) + '"></i>' + esc(kindLabel(cfg, gantt, k)) + '</span>';
    }).join('') + '</div>';
  }

  /**
   * The player's `panels: [{id, label, render(step, ctx)}]` entry.
   *
   * The player hands panels `(step, {cursor, snapshot, before})`, which is not
   * enough to draw a gantt — a gantt needs the whole run up to the cursor. So
   * this closes over the trace once, the same way the memory-hierarchy stage
   * does, and reads the trace itself rather than re-deriving it from the step.
   */
  function panel(config, context) {
    if (!context || !context.trace) throw new Error('gantt: panel needs {trace}');
    var view = makeView({
      trace: context.trace,
      rows: config.rows,
      tracks: config.tracks,
      kinds: config.kinds,
      colWidth: config.colWidth,
      rowHeight: config.rowHeight,
      axis: config.axis,
      maxTicks: config.maxTicks,
      title: config.title,
      caption: config.caption,
      compact: config.compact,
      legend: config.legend
    });
    return {
      id: config.id || 'gantt',
      label: config.label || '调度甘特',
      render: function (step, ctx) { return view.render(ctx.cursor, ctx); },
      view: view
    };
  }

  /**
   * A side-by-side comparison of N gantts on one shared axis.
   *
   * Not two independent charts with matching numbers: the axis length is
   * computed once, from the widest arm, and handed to every chart, so the
   * columns line up by construction. That is the property the L10 comparison
   * rests on — "Static finished at tick 47, Continuous at 27" is only readable
   * as a claim about the same timeline if both charts are drawn against it.
   *
   * `render(cursor)` paints every arm at the same cursor. The arms may have
   * different lengths (they do: that is the finding), and an arm shorter than
   * the cursor simply holds its last frame.
   */
  function comparison(config, context) {
    var arms = config.arms || [];
    if (arms.length < 2) throw new Error('gantt: a comparison needs at least two arms');

    /* One axis for all of them, taken from the longest arm. Understating it
     * would clip the other arms' bars, which is the one failure that would make
     * the comparison silently wrong rather than visibly broken — so it is
     * computed from the data, never configured.
     *
     * The same number is then handed to every arm's view as its axis. Passing
     * each arm its own length instead would let the charts drift apart: both
     * would be internally consistent and the reader would be comparing tick 30
     * of one against tick 30 of the other at two different x positions, which is
     * exactly the mistake the shared axis exists to prevent. */
    var axisMax = arms.reduce(function (m, a) {
      return Math.max(m, (a.trace.steps || []).length);
    }, 0);

    var views = arms.map(function (arm) {
      return {
        label: arm.label,
        note: arm.note || '',
        metrics: arm.metrics || null,
        maxTicks: (arm.trace.steps || []).length,
        view: makeView({
          trace: arm.trace,
          rows: config.rows,
          tracks: config.tracks,
          kinds: config.kinds,
          colWidth: config.colWidth || 13,
          rowHeight: config.rowHeight || 13,
          compact: true,
          legend: false,
          axis: { max: axisMax }
        })
      };
    });

    function render(cursor, ctx) {
      var html = '<div class="lab-gantt-compare" data-axis-max="' +
                 escAttr(String(axisMax)) + '">';
      views.forEach(function (v) {
        var at = Math.min(cursor, v.maxTicks - 1);
        html += '<div class="lab-gantt-arm' +
                (v.metrics && v.metrics.winner ? ' lab-gantt-arm-win' : '') + '">';
        html += '<div class="lab-gantt-armhd">';
        html += '<span class="lab-gantt-armlabel">' + esc(v.label) + '</span>';
        if (v.note) html += '<span class="lab-gantt-armnote">' + esc(v.note) + '</span>';
        html += '</div>';
        if (v.metrics) html += metricStrip(v.metrics);
        html += '<div class="lab-gantt-armbody lab-scroll">' +
                v.view.render(at, ctx) + '</div>';
        if (at < cursor) {
          // Shorter arm: say so rather than letting the chart look truncated.
          html += '<div class="lab-gantt-done">已收干（第 ' + v.maxTicks +
                  ' 步结束）—— 右侧留白即它比另一条早结束的时间</div>';
        }
        html += '</div>';
      });
      html += '</div>';
      return html;
    }

    return {
      id: config.id || 'gantt-compare',
      label: config.label || '对照',
      render: function (step, ctx) { return render(ctx.cursor, ctx); },
      views: views,
      axisMax: axisMax
    };
  }

  function metricStrip(metrics) {
    var out = [];
    (metrics.items || []).forEach(function (it) {
      out.push('<span class="lab-gantt-metric"><b>' + esc(String(it.value)) + '</b> ' +
               esc(it.label) + '</span>');
    });
    return '<div class="lab-gantt-metrics">' + out.join('') + '</div>';
  }

  /* =================================================================== lint
   *
   * The author-side half of this view's contract lives in the trace generator
   * (labs/traces/continuous_batching.py). This is the engine-side half, and the
   * rules must be the same list in both places: a rule that exists on one side
   * only is a rule tested on neither, which is the failure mode this pair
   * exists to prevent. Both sides carry a sabotage control group for the same
   * reason.
   *
   * The rules live here rather than in trace-model.js because they are THIS
   * view's contract — rows, tracks, kinds, bars — and no other view or lab reads
   * them. trace-model.js is the engine's general lint (declared tensors,
   * graph/step agreement, three-tier bindings) and belongs to every lab alike.
   */
  var ROW_RE = /^[A-Za-z0-9_-]+$/;
  var BAR_KEYS = ['row', 'kind'];
  var BAR_OPTIONAL = ['label'];

  function own(obj, k) {
    return Object.prototype.hasOwnProperty.call(obj, k);
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var gantt = trace && trace.gantt;
    var declared = {};
    Object.keys((trace && trace.tensors) || {}).forEach(function (t) { declared[t] = true; });

    if (!gantt || typeof gantt !== 'object') {
      gaps.push({ what: 'gantt 缺失 —— 甘特视图没有行与 kind 的定义，一帧都画不出来',
                  why: '' });
      return { gaps: gaps, warns: warns, infos: infos };
    }

    var rows = gantt.rows;
    if (!Array.isArray(rows)) {
      gaps.push({ what: 'gantt.rows 不是数组', why: '' });
      rows = [];
    } else if (!rows.length) {
      gaps.push({ what: 'gantt.rows 是空的 —— 甘特图一行都画不出来', why: '' });
    }

    var rowIds = [], rowTracks = {};
    var trackIds = [];
    var tracks = gantt.tracks;
    if (tracks !== undefined && tracks !== null) {
      if (!Array.isArray(tracks) || !tracks.length) {
        gaps.push({ what: 'gantt.tracks 存在但不是非空数组 —— 给不出任何轨名', why: '' });
        tracks = [];
      }
      tracks.forEach(function (tk, i) {
        if (!tk || typeof tk !== 'object' || !tk.id) {
          gaps.push({ what: 'gantt.tracks[' + i + '] 没有 id', why: '' });
          return;
        }
        if (trackIds.indexOf(tk.id) !== -1) {
          gaps.push({ what: 'gantt.tracks 里 id "' + tk.id + '" 重复', why: '' });
        }
        trackIds.push(tk.id);
      });
    } else {
      tracks = null;
    }

    (rows || []).forEach(function (row, i) {
      if (!row || typeof row !== 'object' || !row.id) {
        gaps.push({ what: 'gantt.rows[' + i + '] 没有 id', why: '' });
        return;
      }
      if (!ROW_RE.test(row.id)) {
        gaps.push({ what: 'gantt.rows[' + i + '].id = ' + JSON.stringify(row.id) +
                    ' 含非法字符（只允许字母数字下划线连字符）', why: '' });
      }
      if (rowIds.indexOf(row.id) !== -1) {
        gaps.push({ what: 'gantt.rows 里 id "' + row.id + '" 重复 —— 两行同名，' +
                    '条目分不清该画在哪一行', why: '' });
      }
      rowIds.push(row.id);
      if (!row.label) {
        warns.push({ what: 'gantt.rows[' + i + '] ("' + row.id + '") 没有 label，' +
                     '只能显示 id', why: '' });
      }
      if (own(row, 'track')) {
        if (!tracks) {
          gaps.push({ what: 'gantt.rows[' + i + '] 指定了 track = ' +
                      JSON.stringify(row.track) + '，但 gantt.tracks 根本没声明 —— ' +
                      '这一行没有轨可归', why: '' });
        } else if (trackIds.indexOf(row.track) === -1) {
          gaps.push({ what: 'gantt.rows[' + i + '] 的 track = ' +
                      JSON.stringify(row.track) + ' 不在 gantt.tracks 里', why: '' });
        } else {
          if (!rowTracks[row.track]) rowTracks[row.track] = [];
          rowTracks[row.track].push(row.id);
        }
      }
    });

    var kinds = gantt.kinds;
    if (!Array.isArray(kinds) || !kinds.length) {
      gaps.push({ what: 'gantt.kinds 不是非空数组 —— 条目的 kind 没有词表可查', why: '' });
      kinds = [];
    }
    kinds.forEach(function (k) {
      if (typeof k !== 'string' || !k) {
        gaps.push({ what: 'gantt.kinds 里有非字符串项 ' + JSON.stringify(k), why: '' });
      }
    });
    var seenKind = {};
    kinds.forEach(function (k) {
      if (seenKind[k]) gaps.push({ what: 'gantt.kinds 里有重复项 "' + k + '"', why: '' });
      seenKind[k] = true;
    });

    var usedRows = {}, usedKinds = {}, barCounts = {};
    (trace.steps || []).forEach(function (s) {
      var sid = s.id;
      var bars = s.bars;
      if (bars === undefined || bars === null) {
        gaps.push({ what: '步骤 "' + sid + '" 没有 bars —— 甘特图在这一帧没有内容可画',
                    why: '' });
        return;
      }
      if (!Array.isArray(bars)) {
        gaps.push({ what: '步骤 "' + sid + '" 的 bars 不是数组', why: '' });
        return;
      }
      var perRow = [];
      bars.forEach(function (bar, j) {
        if (!bar || typeof bar !== 'object') {
          gaps.push({ what: '步骤 "' + sid + '" 的 bars[' + j + '] 不是对象', why: '' });
          return;
        }
        // Unknown keys are rejected rather than ignored: a bar carrying a field
        // this component does not read is a field its author believed was doing
        // something.
        var extra = Object.keys(bar).filter(function (k) {
          return BAR_KEYS.indexOf(k) === -1 && BAR_OPTIONAL.indexOf(k) === -1;
        });
        if (extra.length) {
          gaps.push({ what: '步骤 "' + sid + '" 的 bars[' + j + '] 多了字段 {' +
                      extra.sort().join(', ') + '} —— 契约只认 {row, kind, label?}',
                      why: '' });
        }
        BAR_KEYS.forEach(function (req) {
          if (!own(bar, req)) {
            gaps.push({ what: '步骤 "' + sid + '" 的 bars[' + j + '] 缺 "' + req + '"',
                        why: '' });
          }
        });
        if (rowIds.indexOf(bar.row) === -1) {
          gaps.push({ what: '步骤 "' + sid + '" 的 bars[' + j + '].row = ' +
                      JSON.stringify(bar.row) + ' 不在 gantt.rows 里 —— ' +
                      '这一条画不到任何一行上', why: '' });
        } else {
          usedRows[bar.row] = true;
          perRow.push(bar.row);
          barCounts[bar.row] = (barCounts[bar.row] || 0) + 1;
        }
        if (kinds.indexOf(bar.kind) === -1) {
          gaps.push({ what: '步骤 "' + sid + '" 的 bars[' + j + '].kind = ' +
                      JSON.stringify(bar.kind) + ' 不在 gantt.kinds 词表里 —— ' +
                      '条目会没有任何样式', why: '' });
        } else {
          usedKinds[bar.kind] = true;
        }
        if (own(bar, 'label') && typeof bar.label !== 'string') {
          gaps.push({ what: '步骤 "' + sid + '" 的 bars[' + j + '].label = ' +
                      JSON.stringify(bar.label) + ' 不是字符串', why: '' });
        }
      });
      var dupes = perRow.filter(function (r, i) { return perRow.indexOf(r) !== i; });
      if (dupes.length) {
        gaps.push({ what: '步骤 "' + sid + '" 里 ' +
                    dupes.filter(function (v, i, a) { return a.indexOf(v) === i; })
                      .sort().join('、') +
                    ' 被声明了不止一个 kind —— 同一 tick 一个资源只能有一种活动',
                    why: '' });
      }
    });

    rowIds.forEach(function (rid) {
      if (!usedRows[rid]) {
        warns.push({ what: 'gantt.rows 里的 "' + rid + '" 从头到尾没有任何条目 —— ' +
                     '这一行永远是空的，是不是忘了给它写 bars', why: '' });
      }
    });
    Object.keys(rowTracks).forEach(function (tid) {
      if (rowTracks[tid].length < 2) {
        warns.push({ what: '轨 "' + tid + '" 只含一行 —— 单行成轨，和不用轨看不出区别',
                     why: '' });
      }
    });
    if (rowIds.length && !Object.keys(usedRows).length) {
      gaps.push({ what: '没有任何 bars 指向 gantt.rows 里的行 —— 甘特图会是空的', why: '' });
    }

    infos.push({ what: (trace.steps || []).length + ' 步 / ' + rowIds.length + ' 行' +
                 (trackIds.length ? ' / ' + trackIds.length + ' 轨' : '') + ' / ' +
                 Object.keys(usedKinds).length + ' 种活动（' +
                 Object.keys(usedKinds).sort().join(', ') + '）', why: '' });
    infos.push({ what: '活动分布：' + Object.keys(barCounts).sort().map(function (k) {
      return k + '=' + barCounts[k];
    }).join('、'), why: '' });

    return { gaps: gaps, warns: warns, infos: infos };
  }

  NS.gantt = {
    makeView: makeView,
    panel: panel,
    comparison: comparison,
    fold: fold,
    lint: lint,
    kindsOf: kindsOf
  };
})(typeof window !== 'undefined' ? window : globalThis);
