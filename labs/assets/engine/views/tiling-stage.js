/* L00 engine · view: memory-hierarchy stage (内存层级舞台).
 *
 * One picture of where every data block lives right now, and which boundary it
 * just crossed. Built for L01's GEMM replay (HBM -> SMEM -> REG), designed so
 * L06's FlashAttention stage is the same component with a different layer list
 * -- HBM then SRAM, two layers instead of three, S and P never appearing in the
 * HBM row at all. Nothing in this file knows what GEMM is.
 *
 * WHAT IS DERIVED AND WHAT IS DECLARED
 *
 * Residency is DERIVED: for each tensor, resolve(trace, i) already says its
 * value at the cursor, and `tensors[].at` says which layer it belongs to. So
 * "what is in each layer at step i" is a pure function of the trace and needs
 * no per-step data -- jumping to step 19 rebuilds the stage from scratch and
 * cannot disagree with having played there.
 *
 * Traffic is DECLARED, on `step.flows`. It cannot be derived, and the trace
 * generator's docstring records why: an accumulator update and a 64-element
 * block store both look like "a write" but move nothing, and a fragment load
 * and an accumulator update both look like "a write to REG" but only one
 * crosses a boundary. See labs/traces/gemm_tiling.py.
 *
 * WHAT IT SHOWS FOR A LAYER
 *
 *   - every tensor currently resident there, as a grid of cells heat-mapped
 *     from a shared scale across all layers, so a magnitude comparison between
 *     blocks is meaningful rather than a comparison of two independent scales;
 *   - the values of the tensor the current step wrote, because those are the
 *     numbers the replay is about;
 *   - a "no tensor is resident here" line for an empty layer, which is how the
 *     naive phase shows that SMEM is never touched -- an absence the reader is
 *     meant to notice, not a gap in the layout;
 *   - a per-layer busy/unused badge for the phase the cursor is in.
 *
 * CONFIG
 *
 *   layers:   [{ id, label, sub? }]   required; ids are the tensors' `at` values
 *   order:    [tensorName, ...]       optional; which tensors to place, and in
 *                                     what order. Defaults to the trace's own
 *                                     declaration order, which is how a trace
 *                                     controls its own layout.
 *   blockLabel(name, spec) -> string  optional; overrides a block's caption
 *   emptyText: string                 optional; the line an empty layer shows
 *
 * L06's config is the same shape with two layers:
 *
 *   layers: [{ id: 'HBM', label: 'HBM' }, { id: 'SRAM', label: 'SRAM' }]
 *   order:  ['Q', 'K', 'V', 'O', 'S', 'P']   // sortable to taste
 *
 * render(cursor, ctx) returns an HTML string. It is called by the player on
 * every paint with the same (trace, i) it just resolved, so the stage can never
 * show a step the rest of the page is not on.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;

  /* ============================================================ derivations */

  /**
   * For each layer, the tensors resident there at this cursor, in `order`.
   *
   * A tensor with no value is omitted rather than shown as a placeholder: the
   * trace's own contract (`init` or an earlier write) is what makes a value
   * exist, and a block that has never been touched genuinely is not in the
   * layer yet. The exception is a scalar starting at a real 0 -- `acc` before
   * the first FMA -- which has a value and is shown as 0.
   */
  function residency(trace, snapshot, layers, order) {
    var byLayer = {};
    layers.forEach(function (l) { byLayer[l.id] = []; });
    order.forEach(function (name) {
      var spec = trace.tensors[name] || {};
      var rec = snapshot[name];
      if (!rec || rec.value === undefined) return;
      var at = spec.at;
      if (!byLayer[at]) return;      // a tensor in a layer the config omits
      byLayer[at].push({ name: name, spec: spec, rec: rec });
    });
    return byLayer;
  }

  /**
   * Which tensors the current step moved, as a set of names.
   *
   * Used to ring the blocks a step touched. A flow lists layers, not tensors,
   * so this falls back to the step's own read/write lists -- which is the right
   * granularity for "what is lit up", as opposed to "how much moved", and the
   * two are allowed to differ (the mma step writes three tensors and crosses
   * one boundary).
   */
  function movedNames(step) {
    var set = {};
    (step.writes || []).forEach(function (t) { set[t] = 'w'; });
    (step.reads || []).forEach(function (t) { if (!set[t]) set[t] = 'r'; });
    return set;
  }

  /* ============================================================ value cells */

  /* A shared numeric range across every resident tensor, so a cell's tint means
   * the same thing in the HBM row as in the REG row. Per-tensor scaling would
   * make a 0.02 look as hot as a 2.0 and quietly lie about magnitudes.
   *
   * `resident` is one frame's blocks. The panel factory calls valueRange over
   * EVERY step instead and hands the result down, because a per-frame range has
   * the same defect one level up: A[0][0] is on screen for the whole replay, and
   * a range that grows as C fills in would repaint it from 0.42 alpha to 0.31
   * without its value changing. Same argument as the rail scale, same fix. */
  function valueRange(resident) {
    var min = Infinity, max = -Infinity;
    resident.forEach(function (group) {
      group.forEach(function (block) {
        flattenNums(block.rec.value).forEach(function (n) {
          if (n < min) min = n;
          if (n > max) max = n;
        });
      });
    });
    if (!isFinite(min) || !isFinite(max)) return { min: 0, max: 1, span: 1 };
    var span = max - min;
    if (span === 0) span = 1;
    return { min: min, max: max, span: span };
  }

  function flattenNums(value, out) {
    out = out || [];
    if (Array.isArray(value)) {
      value.forEach(function (v) { flattenNums(v, out); });
    } else {
      var n = NS.tensors.formatCell(value);
      var f = Number(n);
      if (isFinite(f)) out.push(f);
      else if (n === '−∞') out.push(NaN);
    }
    return out;
  }

  function shapeOf(spec) {
    return spec.shape || [];
  }

  /**
   * The block's cells. Rank 2 lays out as rows; rank 1 as a single row; a
   * scalar has no grid and is shown as its value in the caption instead.
   */
  function cells(block, range, opts) {
    var shape = shapeOf(block.spec);
    if (shape.length === 0) return null;

    var cells = NS.tensors.flatten(block.rec.value);
    var perRow = shape.length > 1 ? shape[shape.length - 1] : cells.length;
    var span = opts.spans && opts.spans[block.name];
    var html = '';
    for (var i = 0; i < cells.length; i++) {
      var row = Math.floor(i / perRow), col = i % perRow;
      var inside = span && row >= span[0][0] && row < span[0][1] &&
                   col >= span[1][0] && col < span[1][1];
      html += cell(cells[i], range, inside);
    }
    return html;
  }

  function cell(v, range, inside) {
    var f = numeric(v);
    var t = isNaN(f) ? 0 : (f - range.min) / range.span;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    var alpha = t * 0.82;
    // The whole attribute is in the string, quotes included: emitting the bare
    // declaration makes the browser read `background:rgba(...)` as an attribute
    // NAME and drop it, so every cell renders untinted with no error anywhere.
    var style = t > 0.002
      ? ' style="background:rgba(56,132,255,' + alpha.toFixed(3) + ')"' : '';
    /* White text only once the tint is actually opaque enough to carry it.
     * The tensor inspector flips at 0.62, which is fine for its handful of
     * large cells; here the shared cross-layer range puts most of A and B near
     * t=0.6 (only C reaches the extremes), so the same threshold paints half
     * of an 8x8 in white-on-mid-blue. The flip is where alpha clears ~0.7. */
    var dark = alpha > 0.7;
    // A cell with no value yet reads as a hole in the block rather than as a
    // zero: SMEM tiles before the first load, C before a block is written.
    var empty = v === undefined || v === null;
    return '<span class="lab-stage-cell' + (dark ? ' lab-stage-cell-dark' : '') +
           (empty ? ' lab-stage-cell-empty' : '') +
           (inside ? ' lab-stage-cell-span' : '') + '"' + style + '>' +
           esc(empty ? '' : NS.tensors.formatCell(v)) + '</span>';
  }

  function numeric(v) {
    if (typeof v === 'number') return v;
    if (NS.traceModel.isSentinel(v)) return NS.traceModel.toNumber(v);
    return NaN;
  }

  /* ================================================================ layout */

  function label(block, cfg) {
    // `blockLabel`, not `label`: `label` is the panel's own caption string in
    // the player's panel config, and colliding the two made this view crash on
    // the first paint.
    if (cfg.blockLabel) return cfg.blockLabel(block.name, block.spec);
    var spec = block.spec;
    if (shapeOf(spec).length === 0) {
      return block.name + ' = ' + NS.tensors.formatCell(block.rec.value);
    }
    return block.name;
  }

  /* A block written by a step in another phase is still resident — registers
   * keep their value — but it is not part of what the reader is currently
   * watching. L01 makes this concrete: the naive pass's scalars stay in REG
   * while the tiled pass runs, and showing them at full strength would suggest
   * one kernel holds both. Marked rather than hidden, because hiding it would
   * be the bigger lie. Inputs (`init`) belong to every phase and are never
   * marked. */
  function carried(block, step, trace) {
    if (block.rec.source !== 'write') return false;
    var src = trace.steps[block.rec.from];
    return !!src && !!src.phase && src.phase !== step.phase;
  }

  function renderBlock(block, cfg, range, opts) {
    var spec = block.spec;
    var shape = shapeOf(spec);
    var tags = movedNames(opts.step)[block.name];
    var cls = 'lab-stage-block' +
      (tags === 'w' ? ' lab-stage-wrote' : tags === 'r' ? ' lab-stage-read' : '');
    if (opts.spanHit[block.name]) cls += ' lab-stage-span';
    if (carried(block, opts.step, opts.trace)) cls += ' lab-stage-carried';

    var html = '<div class="' + cls + '" data-tensor="' + esc(block.name) + '">';
    html += '<div class="lab-stage-block-hd">';
    // The three-part annotation, the same convention the tensor inspector uses:
    // name [d0, d1, …] @ layer.
    html += '<span class="lab-stage-block-id">' + esc(label(block, cfg)) + '</span>';
    html += '<span class="lab-stage-block-shape">' + esc(NS.tensors.shapeLabel(shape)) +
            '</span>';
    html += '<span class="lab-stage-block-at">@ ' + esc(spec.at || '?') + '</span>';
    if (block.rec.source === 'init') html += '<span class="lab-stage-block-src">初值</span>';
    else if (block.rec.source === 'write') {
      html += '<span class="lab-stage-block-src">' +
              (carried(block, opts.step, opts.trace)
                ? '上一阶段 · 第 ' + block.rec.from + ' 步'
                : '第 ' + block.rec.from + ' 步') + '</span>';
    }
    html += '</div>';

    var grid = cells(block, range, opts);
    if (grid) {
      // Rank 1 is a single row, rank 2 lays out by the trailing dimension (the
      // row length), which is the same expression in both cases.
      var cols = shape[shape.length - 1];
      html += '<div class="lab-stage-grid" style="grid-template-columns:repeat(' +
              cols + ',minmax(0,1fr))">' + grid + '</div>';
    }
    html += '</div>';
    return html;
  }

  /* ============================================================ flow rails */

  /**
   * The lane column between two layers, with one rail per declared flow and a
   * static arrow when the step is an ordinary compute step that crosses
   * nothing. A flow with elements too small to render at the current scale
   * still gets its number printed -- the quantity is the point, not the width.
   */
  function rails(flows, maxElements) {
    if (!flows || !flows.length) return '';
    return flows.map(function (f) {
      var same = f.from === f.to;
      var frac = maxElements > 0 ? Math.min(1, f.elements / maxElements) : 0;
      var pct = (18 + frac * 82).toFixed(1);
      var bytes = f.elements * 4;
      var title = f.note ? ' title="' + esc(f.note) + '"' : '';
      return '<div class="lab-stage-rail' + (same ? ' lab-stage-rail-same' : '') +
             '" data-flow="' + esc(f.from + '->' + f.to) + '"' + title + '>' +
        '<span class="lab-stage-rail-bar" style="width:' + pct + '%"></span>' +
        '<span class="lab-stage-rail-n">' + esc(String(f.elements)) + ' 个 · ' +
        esc(String(bytes)) + ' B</span>' +
        (same ? '<span class="lab-stage-rail-tag">原地</span>' : '<span class="lab-stage-arrow">↓</span>') +
        '</div>';
    }).join('');
  }

  /* ================================================================ render */

  function makeView(config) {
    var cfg = config || {};
    var layers = cfg.layers || [];
    if (!layers.length) throw new Error('tiling-stage: config.layers is required');
    layers.forEach(function (l) {
      if (!l.id) throw new Error('tiling-stage: every layer needs an id');
    });

    function render(cursor, ctx) {
      var trace = ctx.trace;
      var idx = ctx.index;
      var step = ctx.step;
      var snapshot = ctx.snapshot;

      var order = cfg.order || idx.tensorNames;
      var byLayer = residency(trace, snapshot, layers, order);

      var all = [];
      layers.forEach(function (l) { all.push(byLayer[l.id]); });
      // Trace-wide when the caller supplied one (see the panel factory); the
      // per-frame range is a fallback for a bare makeView() call, and is the
      // wrong choice for a replay for exactly the reason below.
      var range = ctx.range || valueRange(all);

      // A step's flows are laid on the boundary they name. Flows that start and
      // end in the same layer (the in-register FMA) belong to that layer rather
      // than to a boundary, and are drawn inside its band.
      var flows = step.flows || [];
      var spans = step.spans || {};
      // Computed by the caller (the panel factory owns the trace); a bare
      // makeView() falls back to this step, which is right for a one-shot
      // render and wrong for a replay, hence the explicit pass-through.
      var maxFlow = ctx.maxFlow || flows.reduce(function (m, f) {
        return Math.max(m, f.elements);
      }, 0);
      var spanHit = {};
      Object.keys(spans).forEach(function (k) { spanHit[k] = true; });

      var html = '<div class="lab-stage" data-step="' + esc(step.id) + '">';

      // The phases of the replay, as a header strip. It answers "where am I in
      // the algorithm" without the reader having to read the DAG.
      var phases = phaseList(trace);
      html += '<div class="lab-stage-phases">';
      phases.forEach(function (p) {
        html += '<span class="lab-stage-phase' +
                (p.name === step.phase ? ' lab-stage-phase-on' : '') + '">' +
                esc(p.name) + '<b>' + p.count + '</b></span>';
      });
      html += '</div>';

      for (var li = 0; li < layers.length; li++) {
        var l = layers[li];
        var resident = byLayer[l.id];
        var used = resident.length > 0;

        html += '<div class="lab-stage-layer' + (used ? '' : ' lab-stage-layer-empty') +
                '" data-layer="' + esc(l.id) + '">';
        html += '<div class="lab-stage-layer-hd">';
        html += '<span class="lab-stage-layer-id">' + esc(l.id) + '</span>';
        html += '<span class="lab-stage-layer-label">' + esc(l.label || '') + '</span>';
        if (l.sub) html += '<span class="lab-stage-layer-sub">' + esc(l.sub) + '</span>';
        html += '<span class="lab-stage-layer-count">' +
                (used ? resident.length + ' 块驻留' : '本步无数据') + '</span>';
        html += '</div>';

        html += '<div class="lab-stage-slots">';
        if (!used) {
          // Not a layout hole -- the statement that this layer is not involved
          // in this step. In the naive phase it is how a reader sees that SMEM
          // is skipped entirely.
          html += '<div class="lab-stage-none">' +
                  esc(cfg.emptyText || '这一层在本步没有任何数据块驻留') + '</div>';
        } else {
          resident.forEach(function (block) {
            html += renderBlock(block, cfg, range, {
              step: step, trace: trace, spans: spans, spanHit: spanHit
            });
          });
        }
        html += '</div>';

        // Same-layer flows (the register-resident FMA) render inside the band.
        var same = flows.filter(function (f) { return f.from === l.id && f.to === l.id; });
        if (same.length) html += '<div class="lab-stage-inband">' + rails(same, maxFlow) + '</div>';

        html += '</div>';

        // The boundary to the next layer down, if the step crossed it.
        if (li < layers.length - 1) {
          var next = layers[li + 1].id;
          var crossing = flows.filter(function (f) {
            return (f.from === l.id && f.to === next) || (f.from === next && f.to === l.id);
          });
          html += '<div class="lab-stage-lane' + (crossing.length ? ' lab-stage-lane-on' : '') + '">';
          if (crossing.length) {
            html += rails(crossing, maxFlow);
          } else {
            html += '<span class="lab-stage-lane-idle">' + esc(l.id + ' → ' + next) + '</span>';
          }
          html += '</div>';
        }
      }

      html += '</div>';
      return html;
    }

    return { render: render, layers: layers };
  }

  function phaseList(trace) {
    var order = [];
    var counts = {};
    trace.steps.forEach(function (s) {
      var p = s.phase || '';
      if (!counts[p]) { counts[p] = 0; order.push(p); }
      counts[p] += 1;
    });
    return order.map(function (name) { return { name: name, count: counts[name] }; });
  }

  /**
   * The player's `panels: [{id, label, render(step, ctx)}]` entry for this view.
   *
   * The player hands panels (step, {cursor, snapshot, before}), which is the
   * same information in a flatter shape; the stage wants the whole context
   * (it reads the trace for the phase list and the layer order), so this closes
   * over it once instead of re-passing it on every frame.
   */
  function panel(config, context) {
    var view = makeView(config);
    // The index is built by the player, but the panel has to exist before
    // LabEngine.lab() is called (it goes into that call's config). Building a
    // second one from the same trace is a few hundred operations on a 30-step
    // trace and produces the identical object graph, so the page does not have
    // to thread the player's instance through; if it has one, it can pass it.
    var index = context.index || NS.traceModel.index(context.trace);

    /* Two trace-wide scales, computed once by walking the steps rather than by
     * resolving each cursor (O(state entries), not O(steps x steps)).
     *
     * The rail scale: the largest single flow in the whole trace. Scaling per
     * step would draw the 8-element tile load and the 1024-element naive pass
     * at the same width, so a reader comparing two steps would see a ratio that
     * is not there.
     *
     * The heat scale: the numeric range across every value any step puts in a
     * tensor this view will show. Per-frame, A[0][0] would repaint as C fills in
     * even though A never changes.
     *
     * Both are the same principle: a visual encoding must mean one thing for
     * the whole replay, or comparisons between frames are meaningless. */
    var maxFlow = 0;
    context.trace.steps.forEach(function (s) {
      (s.flows || []).forEach(function (f) {
        if (f.elements > maxFlow) maxFlow = f.elements;
      });
    });

    // Only the tensors this config actually places: a trace may carry tensors
    // in layers the stage does not show (L06 keeps Q in HBM and O in SRAM while
    // an intermediate never leaves registers), and letting those set the scale
    // would dim everything visible for the sake of something invisible.
    var shown = {};
    var order = config.order || index.tensorNames;
    var shownLayers = {};
    config.layers.forEach(function (l) { shownLayers[l.id] = true; });
    order.forEach(function (n) {
      var spec = context.trace.tensors[n];
      if (spec && shownLayers[spec.at]) shown[n] = true;
    });

    var min = Infinity, max = -Infinity;
    function scan(v) {
      if (Array.isArray(v)) { v.forEach(scan); return; }
      var f = numeric(v);
      if (isNaN(f)) return;
      if (f < min) min = f;
      if (f > max) max = f;
    }
    order.forEach(function (n) {
      if (!shown[n]) return;
      if (Object.prototype.hasOwnProperty.call(context.trace.tensors[n], 'init')) {
        scan(context.trace.tensors[n].init);
      }
    });
    context.trace.steps.forEach(function (s) {
      Object.keys(s.state || {}).forEach(function (n) {
        if (shown[n]) scan(s.state[n]);
      });
    });
    var range = isFinite(min) && isFinite(max)
      ? { min: min, max: max, span: (max - min) || 1 }
      : null;

    return {
      id: config.id || 'tiling-stage',
      label: config.label || '内存层级舞台',
      render: function (step, ctx) {
        return view.render(ctx.cursor, {
          trace: context.trace,
          index: index,
          step: step,
          snapshot: ctx.snapshot,
          maxFlow: maxFlow,
          range: range
        });
      }
    };
  }

  NS.tilingStage = {
    makeView: makeView,
    panel: panel,
    residency: residency,
    layerIds: function (trace) {
      var out = [];
      Object.keys(trace.tensors || {}).forEach(function (t) {
        var at = trace.tensors[t].at;
        if (at && out.indexOf(at) === -1) out.push(at);
      });
      return out;
    }
  };
})(typeof window !== 'undefined' ? window : globalThis);
