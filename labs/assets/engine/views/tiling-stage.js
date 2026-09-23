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
      /* A span carries one [lo, hi) range per axis, and `inside` means every one
       * of them contains this cell's coordinate on that axis.
       *
       * The decomposition reads the axes the span DECLARES rather than assuming
       * two. That is a bug fix, not a generality: L01 only ever outlined rank-2
       * tiles, so `span[1]` was always there -- until L06 started spanning `m_i`,
       * a `B_r` vector whose span has exactly one entry, and indexing `span[1]`
       * threw. The last axis is the fastest-varying one, which is `% perRow`;
       * the others fall out of successive division. For a rank-2 tensor this is
       * the same row/column split it always was. */
      var inside = false;
      if (span) {
        inside = true;
        var rest = i;
        for (var ax = 0; ax < span.length && inside; ax++) {
          var stride = 1;
          for (var k = ax + 1; k < shape.length; k++) stride *= shape[k];
          var coord = Math.floor(rest / stride) % (shape[ax] || 1);
          rest = rest % stride;
          inside = coord >= span[ax][0] && coord < span[ax][1];
        }
      }
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

    /* The block is sized, not just its grid: `widths[name]` is the wider of the
     * grid and the caption, so neither the values arriving nor the step index
     * growing can move the box. */
    var boxW = opts.widths && opts.widths[block.name];
    var boxStyle = boxW ? ' style="width:' + boxW.box + 'px"' : '';
    var html = '<div class="' + cls + '" data-tensor="' + esc(block.name) + '"' +
               boxStyle + '>';
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
      /* The TRACK width -- per column, not per block.
       *
       * A fluid `minmax(0, 1fr)` track is sized by its CONTENT, so an uninitialised
       * matrix of zeros is a narrow box and the same matrix after a write is a wide
       * one -- measured on this lab, an 8-column block went from 18px to 57px per
       * cell as values arrived. The block then reflows, the layer changes height,
       * and the whole stage jumps mid-playback: the reader loses the object they
       * were tracking, and any two frames are no longer comparable. Fixing the
       * width per tensor (from the widest value the trace ever puts in it) makes
       * the geometry a function of the TRACE rather than of the cursor -- the same
       * principle the shared heat scale and rail scale already follow.
       *
       * `cell` is the track width and `box` is the whole block; they are two
       * different numbers and passing the block's width here made every block
       * `cols` times too wide -- one visible column per tensor, the rest overflowed
       * and clipped, and neighbouring blocks painted over each other. The
       * acceptance harness's geometry checks all passed while that was true,
       * because a grid's own box keeps the width it was given however far its
       * contents spill; see the `scrollWidth` assertion there. */
      var w = opts.widths && opts.widths[block.name];
      var tracks = w
        ? 'repeat(' + cols + ',' + w.cell + 'px)'
        : 'repeat(' + cols + ',minmax(0,1fr))';
      html += '<div class="lab-stage-grid" style="grid-template-columns:' +
              tracks + '">' + grid + '</div>';
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

  /* ============================================================ occupancy
   *
   * The budget bar: a layer's on-chip capacity against what the current step
   * actually holds, broken into the parts the account names.
   *
   * WHERE EACH NUMBER COMES FROM, because the split matters. The PARTS come from
   * `step.occ` -- the trace measured them off the block shapes the algorithm
   * holds, so the view is formatting a measurement rather than re-deriving one.
   * The CAPACITY comes from the config, via `layer.budget(trace)`, because it is
   * a property of the hardware model the page chose to replay (L06's `M`) and
   * not of any step. The component knows neither the key name nor the formula:
   * it is told a number and shown some parts.
   *
   * An account that exceeds capacity is NOT clipped or clamped: the bar is
   * scaled to the larger of the two, and a capacity marker is drawn where `M`
   * falls so the overflow is legible as a distance past a line rather than as a
   * bar that happens to be full. Silently clamping would turn the one
   * configuration a reader most needs to see -- the one that does not fit --
   * into a bar that looks fine.
   */
  function budget(layer, trace, step, cfg) {
    var occ = step.occ;
    if (!occ || !occ.parts || !occ.parts.length) return null;
    var spec = layer.budget(trace) || {};
    var capacity = spec.capacity;
    if (!capacity || capacity <= 0) return null;
    var over = occ.elements > capacity;
    var scale = Math.max(capacity, occ.elements);

    var html = '<div class="lab-stage-occ' + (over ? ' lab-stage-occ-over' : '') +
               '" data-occ-elements="' + esc(String(occ.elements)) +
               '" data-occ-capacity="' + esc(String(capacity)) + '">';
    html += '<div class="lab-stage-occ-hd">';
    html += '<span class="lab-stage-occ-t">' + esc(spec.label || '占用账') + '</span>';
    html += '<span class="lab-stage-occ-n">' + occ.elements + ' / ' + capacity +
            ' 个元素</span>';
    html += '</div>';

    // The bar itself: one segment per part, plus a ghost segment for the slack
    // so a nearly-empty budget reads as nearly empty rather than as full.
    //
    // `--occ-i` is what gives each segment its palette slot, and it has to be
    // set HERE as well as on the legend dot. Without it every segment falls back
    // to slot 0, the bar renders as one flat colour, and the legend advertises a
    // seven-colour decomposition the bar does not have -- a mismatch that is
    // invisible to every count-based assertion and obvious in a screenshot.
    html += '<div class="lab-stage-occ-bar">';
    occ.parts.forEach(function (p, i) {
      var pct = 100 * p.elements / scale;
      var title = p.name + ' ' + (p.label || '') + ' · ' + p.elements + ' 个元素';
      html += '<span class="lab-stage-occ-seg" data-part="' + esc(p.name) + '"' +
              ' style="--occ-i:' + i + ';width:' + pct.toFixed(3) + '%"' +
              ' title="' + esc(title) + '"></span>';
    });
    if (!over) {
      html += '<span class="lab-stage-occ-slack" style="width:' +
              (100 * (capacity - occ.elements) / scale).toFixed(3) + '%"></span>';
    }
    /* Where the capacity line falls, when the account has crossed it.
     *
     * This has to be drawn INSIDE the bar, and the red border on the bar cannot
     * carry the message on its own: an over-capacity account has no slack
     * segment, so the parts tile the whole width and the border sits underneath
     * them -- an inset shadow or a background tint is simply painted over, and
     * the only surviving signal was the red caption. The marker is a child of
     * the bar, so nothing covers it.
     *
     * The LABEL is not, though. The bar is `overflow: hidden` (it has to be, or
     * the rounded corners leak), so a label positioned above the line inside the
     * bar is clipped away -- and a bare red tick with no number is a mark the
     * reader has to guess at. It goes in the bar's own row above, positioned by
     * the same percentage, outside the clipping box. */
    if (over) {
      var capPct = (100 * capacity / scale).toFixed(3);
      html += '<span class="lab-stage-occ-cap" style="left:' + capPct + '%"></span>';
      html += '</div>';
      html += '<div class="lab-stage-occ-scale"><span class="lab-stage-occ-captick"' +
              ' style="left:' + capPct + '%">M = ' + capacity + '</span></div>';
    } else {
      html += '</div>';
    }

    // A legend, because a stack of anonymous segments is a decoration. Each
    // entry carries its own size so the arithmetic is checkable by eye.
    html += '<div class="lab-stage-occ-legend">';
    occ.parts.forEach(function (p, i) {
      html += '<span class="lab-stage-occ-key" data-part="' + esc(p.name) + '">' +
              '<i class="lab-stage-occ-dot" style="--occ-i:' + i + '"></i>' +
              esc(p.name) + '<b>' + esc(shapeLabel(p.shape)) + '</b>' +
              '<em>' + p.elements + '</em></span>';
    });
    /* The slack row, and what it means when there is none.
     *
     * Two different things get said here and conflating them left the legend
     * short a line: a positive slack is the room left over (a gap in the bar,
     * with no colour), while a negative one is the amount BY WHICH the parts
     * together exceed the capacity -- which is not a segment either, it is the
     * sum of the segments past the marker. So the over case names the marker it
     * is measured against. */
    var slack = capacity - occ.elements;
    html += '<span class="lab-stage-occ-key lab-stage-occ-key-slack">' +
            (slack >= 0
              ? '余量 <em>' + slack + '</em>'
              : '超出 M = ' + capacity + ' 共 <em>' + (-slack) + '</em> 个元素') +
            '</span>';
    html += '</div>';

    if (spec.note) html += '<div class="lab-stage-occ-note">' + esc(spec.note(trace, step)) + '</div>';
    html += '</div>';
    return html;
  }

  function shapeLabel(shape) {
    return '[' + (shape || []).join(', ') + ']';
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
              step: step, trace: trace, spans: spans, spanHit: spanHit,
              widths: ctx.widths
            });
          });
        }
        html += '</div>';

        // The occupancy account, when the trace carries one and the layer asks
        // for it. See `budget()` below for why the capacity comes from the
        // config and the parts come from the step.
        if (l.budget) {
          var b = budget(l, trace, step, cfg);
          if (b) html += b;
        }

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

    /* One width per tensor, resolved over the WHOLE trace rather than the frame.
     *
     * A grid track sized by content makes the block's width a function of the
     * cursor: a matrix of zeros is narrow, the same matrix full of `-0.09786` is
     * three times as wide, and the layer reflows the moment the first write
     * lands. Measured on this lab, an 8-column block went from 18px per cell to
     * 57px, and the HBM row went from one line to three -- an 825x165 hole where
     * the reader was looking. Every count stayed correct, so nothing noticed.
     *
     * Both halves have to be fixed, and the second is the one that is easy to
     * miss:
     *
     *   - the CELL, from the widest string the tensor ever holds;
     *   - the BLOCK, which is wider than its grid when the caption is wider than
     *     the matrix. A `B_r`-long vector has a 4-cell grid and a caption reading
     *     `m_i [4] @ SRAM  第 16 步`, and that caption's own width grows with the
     *     step index -- so a fixed grid still reflows at step 9.
     *
     * This is the same principle the heat scale and the rail scale already
     * follow: a visual encoding must mean one thing for the whole replay, and
     * size is an encoding. */
    var CELL_PX = 6.2;     // per character at the cell font (10.5px monospace)
    var CELL_PAD = 7;
    var cellChars = {};
    function measure(name, v) {
      if (Array.isArray(v)) { v.forEach(function (x) { measure(name, x); }); return; }
      var s = NS.tensors.formatCell(v);
      var w = cellChars[name] === undefined ? 0 : cellChars[name];
      if (s.length > w) cellChars[name] = s.length;
    }
    var placed = {};
    order.forEach(function (name) { placed[name] = true; });
    context.trace.steps.forEach(function (s) {
      Object.keys(s.state || {}).forEach(function (name) {
        if (placed[name]) measure(name, s.state[name]);
      });
    });
    order.forEach(function (name) {
      var spec = context.trace.tensors[name];
      if (spec && Object.prototype.hasOwnProperty.call(spec, 'init')) measure(name, spec.init);
    });

    /* The widest caption any step can put on a block: `name [shape] @ layer`,
     * plus the longest provenance badge. Counted in half-widths so the CJK in
     * `第 N 步` / `初值` / `上一阶段` is not under-measured at a monospace
     * per-character rate. */
    function halfWidths(s) {
      var n = 0;
      for (var i = 0; i < s.length; i++) {
        n += s.charCodeAt(i) > 0x2e7f ? 2 : 1;
      }
      return n;
    }
    var srcHalf = 0;
    context.trace.steps.forEach(function (s, i) {
      srcHalf = Math.max(srcHalf, halfWidths('第 ' + i + ' 步'));
    });
    srcHalf = Math.max(srcHalf, halfWidths('上一阶段 · 第 ' +
                                           context.trace.steps.length + ' 步'),
                       halfWidths('初值'));

    var widths = {};
    order.forEach(function (name) {
      var spec = context.trace.tensors[name] || {};
      var shape = spec.shape || [];
      if (!shape.length) return;          // a scalar has no grid
      var cols = shape[shape.length - 1];
      /* One column's width, from the widest string the tensor ever holds.
       *
       * `CELL_PAD` has to cover the cell's own horizontal padding (2*3px), its
       * 1px border on each side, and a pixel of rounding -- a cell whose content
       * box is exactly the text width still overflows its track by the padding,
       * and `scrollWidth` sees every pixel of it. */
      var cell = Math.max(20, Math.round((cellChars[name] || 1) * CELL_PX) + 10);
      var headHalf = halfWidths(name + ' ' + NS.tensors.shapeLabel(shape) + ' @ ' +
                                (spec.at || '?')) + srcHalf;
      /* The block's own horizontal padding and border (6px 7px 7px + 1px border,
       * box-sizing:border-box), added to whichever of the two contents is wider.
       * The grid has no padding of its own -- the padding belongs to the block --
       * so the grid's track sum is the CONTENT width and the block's inline width
       * has to be content + padding, or the content overflows the box by exactly
       * the padding and `scrollWidth` says so. */
      var PAD = 7 * 2 + 1 * 2;
      var gridNeed = cols * cell + (cols - 1);      // tracks + the 1px column gaps
      var head = Math.round(headHalf * 4.6) + 34;   // caption + its gaps
      widths[name] = {
        cell: cell,
        box: Math.max(gridNeed, head) + PAD,
      };
    });

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
          range: range,
          widths: widths
        });
      }
    };
  }

  /* ================================================================ lint
   *
   * The JS port of the contract for everything this view and its sibling panels
   * render: `step.spans`, `step.flows`, `step.occ`, `step.io`, and the trace-level
   * `meta.sram` / `meta.io` / `meta.exposure` / `meta.correction` blocks. Python
   * (`labs/traces/flash_attention.py`) is the authority and runs before a trace is
   * written; this exists so a lab author editing a trace in the browser gets the
   * same verdict without a Python round trip, exactly as `trace-model.js`'s lint
   * mirrors `online_softmax.py`'s.
   *
   * THE TWO PORTS MUST AGREE. The acceptance harness runs one sabotage table
   * through both and requires the same verdict from each; a rule that exists on
   * one side only is a rule tested on neither.
   *
   * WHAT THIS DOES NOT DO, and why it is not a gap: it does not re-check `state`,
   * formulas, bindings or the graph -- those are `trace-model.js`'s contract and
   * that file already carries their JS port. Duplicating them here would give a lab
   * author two places to fix and two places to let drift.
   *
   * Nor does it re-derive the numbers in `meta.exposure`. Those are the count from
   * a real memory-access log, and there is no access log in a browser -- a JS
   * check on them could only compare the published literals against each other.
   * What it checks instead is that the claim is internally coherent (the control
   * group actually saw what the claim says was never touched), and Python checks
   * the values, because Python is where the run happens.
   */
  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var steps = (trace && trace.steps) || [];
    var tensors = (trace && trace.tensors) || {};
    var declared = Object.keys(tensors);
    var declaredSet = {};
    declared.forEach(function (t) { declaredSet[t] = true; });

    var layers = [];
    declared.forEach(function (t) {
      var at = tensors[t].at;
      if (at && layers.indexOf(at) === -1) layers.push(at);
    });

    var cfg = ((trace || {}).meta || {}).config || {};
    var sramNames = declared.filter(function (t) { return tensors[t].at === 'SRAM'; });

    /* Which optional blocks does this trace carry?
     *
     * The gates matter because this component is SHARED: L01 drives the same
     * stage with three layers, no occupancy account, no IO counter and no
     * correction factors, and requiring any of them would turn a perfectly good
     * trace red.
     *
     * Each switch is declared by the trace itself, and by a field one level down
     * from what it gates -- `steps[].occ`, not `meta.sram` -- so the rule cannot
     * become circular: gating "meta.sram is required" on "meta.sram exists" would
     * mean the sabotage that deletes it lints clean. This is the same shape as
     * `trace-model.js` deriving `wantsAxis` from `step.k` rather than from the
     * `compare` block whose own presence check depends on it. */
    var wantsOcc = steps.some(function (s) { return s.occ; });
    var wantsIo = steps.some(function (s) { return s.io; });
    var hasCorr = steps.some(function (s) { return s.corr; });
    /* The exposure block is about S and P specifically, so its gate is the trace
     * declaring those tensors -- not the block's own presence, which would make
     * the "it is missing" rule unable to fire. */
    var hasForbidden = declared.indexOf('S') !== -1 && declared.indexOf('P') !== -1;

    /* Which tensors hold a value after step `upto` -- the rule trace-model's
     * resolve() uses, re-derived so the occupancy account is checked against the
     * trace rather than against itself. */
    var liveAt = [];
    (function () {
      var live = {};
      declared.forEach(function (t) {
        if (Object.prototype.hasOwnProperty.call(tensors[t], 'init')) live[t] = true;
      });
      steps.forEach(function (s, i) {
        var state = s.state || {};
        Object.keys(state).forEach(function (k) { live[k] = true; });
        liveAt[i] = live;
        live = {};
        Object.keys(liveAt[i]).forEach(function (k) { live[k] = true; });
      });
    })();

    /* The shape a part must declare.
     *
     * For a plain part the authority is the tensor's OWN declaration, which the
     * view already has -- no lab knowledge needed, and the rule is exactly "the
     * account's shape agrees with the tensor it names".
     *
     * The merged buffer has no single declaration to defer to (it is one
     * allocation covering two tensors), so its shape comes from the config: the
     * `B_r x B_c` the blocking parameters imply. A component that instead
     * guessed "the merged buffer is whatever shape the two names suggest" would
     * need to know which tensors are merged, which is the lab's fact, not the
     * view's. */
    function partShape(name, covers) {
      if (covers) return [cfg.Br, cfg.Bc];
      var spec = tensors[name] || {};
      return spec.shape || null;
    }

    steps.forEach(function (s, idx) {
      var sid = s.id;

      // ---- spans: a sub-region of a declared tensor, inside its shape -----
      Object.keys(s.spans || {}).forEach(function (key) {
        if (!declaredSet[key]) {
          gaps.push({ what: '步骤 "' + sid + '" 的 spans 指向未声明张量 "' + key + '"',
                      why: '引擎不知道它的 shape，框不出这块区域。' });
          return;
        }
        var shape = tensors[key].shape || [];
        var axes = s.spans[key];
        if (axes.length !== shape.length) {
          gaps.push({ what: '步骤 "' + sid + '" 的 spans["' + key + '"] 给了 ' +
                      axes.length + ' 个轴，但张量是 ' + shape.length + ' 维',
                      why: '轴数对不上，框选会错位。' });
          return;
        }
        axes.forEach(function (rng, ax) {
          var lo = rng[0], hi = rng[1];
          if (!(lo >= 0 && lo < hi && hi <= shape[ax])) {
            gaps.push({ what: '步骤 "' + sid + '" 的 spans["' + key + '"] 第 ' + ax +
                        ' 轴 [' + lo + ', ' + hi + ') 超出 shape ' + shape[ax],
                        why: '框选的区域落在张量之外。' });
          }
        });
      });

      // ---- flows: both ends must be layers this trace declares tensors in -
      (s.flows || []).forEach(function (f) {
        ['from', 'to'].forEach(function (end) {
          if (layers.indexOf(f[end]) === -1) {
            gaps.push({ what: '步骤 "' + sid + '" 的 flows 有一端的层 "' + f[end] +
                        '" 不在 tensors[].at 用到的层里',
                        why: '轨道会画到一个不存在的层上。' });
          }
        });
        if (typeof f.elements !== 'number' || f.elements <= 0 ||
            f.elements !== Math.floor(f.elements)) {
          gaps.push({ what: '步骤 "' + sid + '" 的 flows 搬运量为 ' + JSON.stringify(f.elements) +
                      '，应为正整数',
                      why: '轨道的长度按它算，0 或负数画不出。' });
        }
      });

      // ---- occ: the account must name exactly what is resident ------------
      var occ = s.occ;
      if (!occ) {
        if (wantsOcc) {
          gaps.push({ what: '步骤 "' + sid + '" 没有 occ', why: 'SRAM 占用账缺这一步 —— ' +
                      'trace 声明了 meta.sram，每一步就该有一份账。' });
        }
        return;
      }
      if (Object.keys(occ).sort().join(',') !== 'elements,parts') {
        gaps.push({ what: '步骤 "' + sid + '" 的 occ 字段是 {' +
                    Object.keys(occ).sort().join(', ') + '}，而不是 {elements, parts}',
                    why: '占用账按固定字段渲染。' });
        return;
      }
      var live = (liveAt[idx] || {});
      var covered = {};
      var total = 0;
      (occ.parts || []).forEach(function (p) {
        var name = p.name;
        var isPlain = declaredSet[name] && tensors[name].at === 'SRAM';
        if (isPlain) {
          covered[name] = true;
          if (p.covers) {
            gaps.push({ what: '步骤 "' + sid + '" 的 occ["' + name + '"] 带 covers',
                        why: '一个张量自己就是一个分配，没有可合并的对象。' });
          }
        } else if (!p.covers) {
          gaps.push({ what: '步骤 "' + sid + '" 的 occ 里有一项 "' + name +
                      '"，既不是 SRAM 张量，也没有声明它 covers 了哪些张量',
                      why: '账目里出现了引擎画不出来的东西。' });
          return;
        } else {
          // The merged buffer: `covers` names the tensors on the allocation, and
          // they must be exactly the ones resident now -- so an account cannot
          // claim the buffer holds both S and P before P exists, and the view
          // needs to know nothing about which lab merges what.
          p.covers.forEach(function (t) {
            if (sramNames.indexOf(t) === -1) {
              gaps.push({ what: '步骤 "' + sid + '" 的 occ["' + name + '"].covers = [' +
                          p.covers.join(', ') + '] 里有非 SRAM 张量 "' + t + '"',
                          why: '合并缓冲只可能覆盖同一层里的张量。' });
              return;
            }
            if (live[t]) covered[t] = true;
          });
          var want = (p.covers || []).filter(function (t) { return live[t]; }).sort();
          var got = (p.covers || []).slice().sort();
          if (want.join(',') !== got.join(',')) {
            gaps.push({ what: '步骤 "' + sid + '" 的 occ["' + name + '"].covers = [' +
                        got.join(', ') + ']，而此刻驻留 SRAM 的是 [' + want.join(', ') + ']',
                        why: '合并缓冲声明覆盖的块与纯函数重建出来的驻留集合对不上。' });
          }
        }
        var shape = partShape(name, isPlain ? null : p.covers);
        if (shape && p.shape && p.shape.join(',') !== shape.join(',')) {
          gaps.push({ what: '步骤 "' + sid + '" 的 occ["' + name + '"].shape = [' +
                      (p.shape || []).join(', ') + '] 与配置推出的 [' + shape.join(', ') +
                      '] 不一致',
                      why: '占用账的形状与分块参数脱节。' });
        }
        if (typeof p.elements !== 'number' || p.elements <= 0) {
          gaps.push({ what: '步骤 "' + sid + '" 的 occ["' + name + '"].elements = ' +
                      JSON.stringify(p.elements) + ' 应为正整数',
                      why: '占用账的量按元素数显示。' });
        }
        total += p.elements || 0;
      });
      var coveredList = Object.keys(covered).sort();
      var liveSram = sramNames.filter(function (t) { return live[t]; }).sort();
      if (coveredList.join(',') !== liveSram.join(',')) {
        gaps.push({ what: '步骤 "' + sid + '" 的 occ 覆盖了 [' + coveredList.join(', ') +
                    ']，而此刻驻留 SRAM 的是 [' + liveSram.join(', ') + ']',
                    why: '占用账与纯函数重建出来的驻留集合对不上 —— 图上的账是错的。' });
      }
      if (occ.elements !== total) {
        gaps.push({ what: '步骤 "' + sid + '" 的 occ.elements = ' + JSON.stringify(occ.elements) +
                    ' 与各项之和不符（' + total + '）',
                    why: '总量与分解自相矛盾。' });
      }
    });

    /* ---- the IO counter: a running total that can only grow ---------------- */
    var lastIo = 0;
    steps.forEach(function (s) {
      var io = s.io;
      if (!io) {
        if (wantsIo) {
          gaps.push({ what: '步骤 "' + s.id + '" 没有 io', why: 'IO 计数器缺这一步 —— ' +
                      'trace 声明了逐帧计数，就每一步都该有一格。' });
        }
        return;
      }
      if (Object.keys(io).sort().join(',') !== 'cum,step') {
        gaps.push({ what: '步骤 "' + s.id + '" 的 io 字段是 {' +
                    Object.keys(io).sort().join(', ') + '}，而不是 {step, cum}',
                    why: 'IO 计数器按这两个字段累加。' });
        return;
      }
      if (typeof io.step !== 'number' || io.step < 0 || io.step !== Math.floor(io.step)) {
        gaps.push({ what: '步骤 "' + s.id + '" 的 io.step = ' + JSON.stringify(io.step) +
                    ' 应为非负整数',
                    why: '本步的 HBM 访问量按元素数计。' });
      }
      if (io.cum < lastIo) {
        gaps.push({ what: '步骤 "' + s.id + '" 的 io.cum = ' + io.cum + ' 比上一步的 ' +
                    lastIo + ' 小',
                    why: '累计访问量只能单调不减 —— 它数的是已经发生过的搬运。' });
      } else if (io.cum !== lastIo + io.step) {
        gaps.push({ what: '步骤 "' + s.id + '" 的 io.cum = ' + io.cum + ' 与上一步的 ' +
                    lastIo + ' 加 io.step ' + io.step + ' 对不上',
                    why: '逐帧数字与累计值自相矛盾。' });
      }
      lastIo = io.cum;
    });

    /* ---- meta.sram: the account's peak must be one of the per-step accounts - */
    var sram = (trace.meta || {}).sram;
    if (!sram) {
      if (wantsOcc) {
        gaps.push({ what: 'meta.sram 缺失', why: '占用账的峰值与分解没有随 trace 发布。' });
      }
    } else {
      var peak = null;
      steps.forEach(function (s) {
        if (s.occ && (!peak || s.occ.elements > peak.elements)) {
          peak = { elements: s.occ.elements, id: s.id };
        }
      });
      if (sram.used !== (peak ? peak.elements : 0)) {
        gaps.push({ what: 'meta.sram.used = ' + JSON.stringify(sram.used) +
                    ' 与逐帧峰值 ' + (peak ? peak.elements : 0) + ' 不一致',
                    why: '结论里报的峰值与图上的账对不上。' });
      }
      if (sram.peak_step !== (peak ? peak.id : null)) {
        gaps.push({ what: 'meta.sram.peak_step = ' + JSON.stringify(sram.peak_step) +
                    '，而最占 SRAM 的一步是 ' + JSON.stringify(peak ? peak.id : null),
                    why: '页面按这个名字跳到峰值那一步。' });
      }
      if (typeof sram.dominant !== 'number' || typeof cfg.Bc !== 'number' ||
          typeof cfg.d !== 'number' || sram.dominant !== 4 * cfg.Bc * cfg.d) {
        gaps.push({ what: 'meta.sram.dominant = ' + JSON.stringify(sram.dominant) +
                    ' 应为 4·B_c·d = ' + (4 * (cfg.Bc || 0) * (cfg.d || 0)),
                    why: '教程公式的主导项与分块参数脱节。' });
      }
      if (sram.M !== cfg.M) {
        gaps.push({ what: 'meta.sram.M = ' + JSON.stringify(sram.M) +
                    ' 与 config.M = ' + JSON.stringify(cfg.M) + ' 不一致',
                    why: '占用账的容量与配置脱节 —— 滑杆会显示一个不认识自己的账本。' });
      }
      if (!sram.parts || !sram.parts.length) {
        gaps.push({ what: 'meta.sram.parts 为空', why: '占用账画不出分解。' });
      }
    }

    /* ---- config: the blocking arithmetic, which every shape here depends on --
     *
     * Gated on the trace declaring `Bc` / `Br`, because these are the FLASH
     * ATTENTION blocking parameters: L01's GEMM trace has `B_M` / `B_N` / `B_K`
     * and L05's KV cache has neither, and neither is wrong. A trace that has the
     * parameters must have them consistent; a trace that does not is a different
     * lab, not a broken one. */
    var hasBlocking = cfg.Bc !== undefined && cfg.Br !== undefined;
    if (hasBlocking) {
      if (cfg.Br !== Math.min(cfg.Bc, cfg.d)) {
        gaps.push({ what: 'config.Br = ' + JSON.stringify(cfg.Br) + ' 应为 min(B_c, d) = ' +
                    Math.min(cfg.Bc, cfg.d),
                    why: 'B_r 的定义是教程写死的。' });
      }
      if (cfg.Tr * cfg.Br !== cfg.N || cfg.Tc * cfg.Bc !== cfg.N) {
        gaps.push({ what: 'T_r B_r = ' + (cfg.Tr * cfg.Br) + '、T_c B_c = ' + (cfg.Tc * cfg.Bc) +
                    '，应都等于 N = ' + cfg.N,
                    why: '块数与块大小乘不回序列长度。' });
      }
      if (cfg.n_blocks !== cfg.Tr * cfg.Tc) {
        gaps.push({ what: 'config.n_blocks = ' + JSON.stringify(cfg.n_blocks) +
                    ' 应为 T_r × T_c = ' + (cfg.Tr * cfg.Tc),
                    why: '修正因子的条带格数按它渲染。' });
      }
      if (cfg.n_steps !== steps.length) {
        gaps.push({ what: 'config.n_steps = ' + JSON.stringify(cfg.n_steps) +
                    ' 与实际步数 ' + steps.length + ' 不符',
                    why: '页面上的步数标注会与实际不符。' });
      }
    }

    /* ---- meta.correction: the amplifier's summary -------------------------- */
    var mc = (trace.meta || {}).correction;
    if (!mc) {
      if (hasCorr) {
        gaps.push({ what: 'meta.correction 缺失', why: '修正因子放大器报不出结论。' });
      }
    } else {
      if (mc.per_block && mc.per_block.length !== cfg.n_blocks) {
        gaps.push({ what: 'meta.correction.per_block 有 ' + mc.per_block.length +
                    ' 项，应该是 n_blocks = ' + cfg.n_blocks + ' 项',
                    why: '条带的格数与块数对不上。' });
      }
      var stepped = (mc.per_block || []).filter(function (e) {
        return e.kind === 'rescale';
      }).length;
      if (mc.per_block && mc.rescales !== stepped) {
        gaps.push({ what: 'meta.correction.rescales = ' + JSON.stringify(mc.rescales) +
                    ' 与 per_block 里 rescale 的项数 ' + stepped + ' 不一致',
                    why: '结论行的「几块真的做了缩放」会与条带矛盾。' });
      }
      if (!mc.final_bias_abs || !mc.final_bias_rel) {
        gaps.push({ what: 'meta.correction 的最终偏差是 ' + JSON.stringify(mc.final_bias_abs) +
                    ' / ' + JSON.stringify(mc.final_bias_rel),
                    why: '「漏掉修正因子」本该有可测的代价，否则 L00 的放大器无从谈起。' });
      }
    }

    /* ---- meta.exposure: the claim, and the control that makes it mean something
     *
     * Coherence only -- see the note at the top of this function for why the
     * numbers themselves are Python's to check. */
    var ex = (trace.meta || {}).exposure;
    if (!ex) {
      if (hasForbidden) {
        gaps.push({ what: 'meta.exposure 缺失',
                    why: '「S、P 不落回 HBM」的实测证据没有随 trace 发布。' });
      }
    } else {
      ['forbidden', 'flash_hbm_tensors', 'standard_hbm_tensors',
       'flash_forbidden_accesses', 'standard_forbidden_accesses'].forEach(function (k) {
        if (!(k in ex)) {
          gaps.push({ what: 'meta.exposure 缺 ' + k, why: '页面上的结论行会显示 undefined。' });
        }
      });
      if (ex.flash_forbidden_accesses !== 0) {
        gaps.push({ what: 'meta.exposure.flash_forbidden_accesses = ' +
                    JSON.stringify(ex.flash_forbidden_accesses) + '，应为 0',
                    why: '这是整个 lab 的核心主张。' });
      }
      if (!ex.standard_forbidden_accesses) {
        gaps.push({ what: 'meta.exposure.standard_forbidden_accesses 为 0',
                    why: '对照组失效 —— FA 那边的 0 就没有区分力。' });
      }
      (ex.forbidden || []).forEach(function (name) {
        if ((ex.flash_hbm_tensors || []).indexOf(name) !== -1) {
          gaps.push({ what: 'S/P 之一 "' + name + '" 出现在 flash_hbm_tensors 里',
                      why: '禁用集合与实测集合自相矛盾。' });
        }
        if ((ex.standard_hbm_tensors || []).indexOf(name) === -1) {
          gaps.push({ what: 'meta.exposure.standard_hbm_tensors 里没有 "' + name + '"',
                      why: '对照组本该在它自己的日志里记录这个名字。' });
        }
      });
    }

    /* ---- meta.io: the measurement and the prediction must describe each other */
    var mi = (trace.meta || {}).io;
    if (!mi) {
      if (wantsIo) {
        gaps.push({ what: 'meta.io 缺失', why: 'IO 计数器没有数据。' });
      }
    } else {
      var pred = mi.predict || {};
      ['standard', 'flash'].forEach(function (side) {
        var measured = (mi[side] || {}).elements;
        if (measured !== pred[side]) {
          gaps.push({ what: 'meta.io.' + side + '.elements = ' + JSON.stringify(measured) +
                      ' 与预测 ' + JSON.stringify(pred[side]) + ' 不一致',
                      why: '实测与公式脱节 —— 页面上那两行会是两个互相矛盾的说法。' });
        }
        if (!measured) {
          gaps.push({ what: 'meta.io.' + side + '.elements 为 0', why: 'IO 计数器的读数是空的。' });
        }
      });
      var curves = mi.curves || {};
      ['ns', 'standard', 'flash'].forEach(function (k) {
        if (!curves[k] || !curves[k].length) {
          gaps.push({ what: 'meta.io.curves.' + k + ' 为空', why: '对比图画不出来。' });
        }
      });
      var rows = (mi.law || {}).rows || [];
      if (rows.length < 2) {
        gaps.push({ what: 'meta.io.law.rows 只有 ' + rows.length + ' 行',
                    why: 'IO × B_c 近似恒定的说法至少要两个不同块大小才谈得上检验。' });
      } else {
        var times = rows.map(function (r) { return r.times_bc; });
        var bad = times.some(function (v) {
          return typeof v !== 'number' || !isFinite(v) || v <= 0;
        });
        if (bad) {
          gaps.push({ what: 'meta.io.law.rows 的 times_bc 非正：' + JSON.stringify(times),
                      why: 'IO × B_c 是正数。' });
        } else if (Math.max.apply(null, times) / Math.min.apply(null, times) - 1 > 0.35) {
          gaps.push({ what: 'meta.io.law.rows 的 IO×B_c 跨度 ' +
                      (Math.max.apply(null, times) / Math.min.apply(null, times) - 1).toFixed(3) +
                      ' 过大', why: 'O(N^2 d^2 / M) 的常数说法不成立。' });
        }
      }
    }

    infos.push({ what: steps.length + ' 步 / ' + layers.length + ' 层', why: '' });
    return { gaps: gaps, warns: warns, infos: infos };
  }

  /* The merged-buffer convention lives in the trace, not here: a part that
   * accounts for tensors other than its own name lists them in `covers`. This
   * component never learns which lab merges which two buffers, and no lab has to
   * spell a name in a way the component will recognise. */

  NS.tilingStage = {
    makeView: makeView,
    panel: panel,
    residency: residency,
    lint: lint,
    shapeLabel: shapeLabel,
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
