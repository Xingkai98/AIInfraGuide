/* L00 engine · tensor inspector.
 *
 * Every tensor is shown the same three-part way, which is a repo-wide
 * convention rather than a local choice:
 *
 *     name [d0, d1, …] @ 位置
 *
 * Values render as a grid of cells tinted by a heatmap, so a tensor's shape is
 * legible at a glance and its magnitude distribution is visible without
 * reading every number. Cells that changed on this step are outlined, and a
 * tensor written by the current step is badged — the panel is answering "what
 * did this step actually do", not just "what are the numbers".
 *
 * The heatmap is an alpha overlay rather than an interpolated colour pair, so
 * it reads correctly on both the light and dark theme without a second palette:
 * low values are nearly transparent, high values are a strong blue, and the
 * text colour flips once the cell gets dark enough.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = function (s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c];
    });
  };

  function shapeLabel(shape) {
    if (!shape || !shape.length) return '标量';
    return '[' + shape.join(', ') + ']';
  }

  /* Flatten nested arrays in row-major order. A trace may give either a nested
   * value or a flat one against the declared shape; both land in the same
   * order, so the caller only needs `shape` to lay them back out. */
  function flatten(value) {
    var out = [];
    (function walk(v) {
      if (Array.isArray(v)) { for (var i = 0; i < v.length; i++) walk(v[i]); }
      else out.push(v);
    })(value);
    return out;
  }

  function formatNumber(x) {
    if (typeof x !== 'number') return String(x);
    if (!isFinite(x)) return x > 0 ? '∞' : '−∞';
    if (Number.isInteger(x)) return String(x);
    var s = x.toPrecision(4);
    return s.indexOf('e') === -1 ? s.replace(/0+$/, '').replace(/\.$/, '') : s;
  }

  function formatCell(v) {
    if (NS.traceModel.isSentinel(v)) return NS.traceModel.SENTINEL_TEXT[v];
    if (v === undefined) return '—';
    if (v === null) return '∅';
    return formatNumber(v);
  }

  function numeric(v) {
    if (typeof v === 'number') return v;
    if (NS.traceModel.isSentinel(v)) return NS.traceModel.toNumber(v);
    return NaN;
  }

  /* Group the flat cell list into blocks -> rows -> cells for the declared
   * shape. Shapes of rank 1 and 2 are the common case; higher ranks nest. */
  function nest(cells, shape) {
    if (!shape || shape.length === 0) return { rows: [cells], dims: [] };
    if (shape.length === 1) return { rows: [cells], dims: [shape[0]] };
    var rows = [];
    var perRow = shape[shape.length - 1];
    for (var i = 0; i < cells.length; i += perRow) rows.push(cells.slice(i, i + perRow));
    return { rows: rows, dims: shape };
  }

  function renderValue(spec, value, prevValue, changed) {
    if (value === undefined) {
      return '<div class="lab-tv-empty">— 本步之前无值 —</div>';
    }
    var shape = spec.shape || [];
    var cells = flatten(value);
    var prevCells = prevValue === undefined ? null : flatten(prevValue);

    var nums = cells.map(numeric).filter(function (x) { return !isNaN(x); });
    var min = nums.length ? Math.min.apply(null, nums) : 0;
    var max = nums.length ? Math.max.apply(null, nums) : 1;
    var span = max - min || 1;

    var laid = nest(cells, shape);
    var html = shape.length > 1 ? '<div class="lab-tv-blocks">' : '<div class="lab-tv-row">';

    laid.rows.forEach(function (row, r) {
      if (shape.length > 1) html += '<div class="lab-tv-row">';
      row.forEach(function (cell, c) {
        var idx = r * row.length + c;
        var n = numeric(cell);
        var t = isNaN(n) ? 0 : (n - min) / span;
        var style = t > 0.001
          ? ' style="background:rgba(56,132,255,' + (t * 0.82).toFixed(3) + ')"'
          : '';
        var dark = t > 0.62;
        var cellChanged = changed && prevCells &&
          (prevCells.length !== cells.length || prevCells[idx] !== cell);
        html += '<span class="lab-tcell' + (dark ? ' lab-tcell-dark' : '') +
                (cellChanged ? ' lab-tcell-chg' : '') + '"' + style + '>' +
                esc(formatCell(cell)) + '</span>';
      });
      if (shape.length > 1) html += '</div>';
    });
    html += '</div>';
    return html;
  }

  /**
   * @param {object} idx      trace index
   * @param {object} snapshot resolve(idx, cursor) — values AFTER the step
   * @param {object} previous resolve(idx, cursor-1) — values BEFORE the step
   * @param {object} step     the current step
   * @param {object} opts     { when: 'after' | 'before' }
   */
  function render(idx, snapshot, previous, step, opts) {
    var when = (opts && opts.when) || 'after';
    var shown = when === 'after' ? snapshot : previous;
    var ref = when === 'after' ? previous : null;
    var wrote = {};
    (step.writes || []).forEach(function (t) { wrote[t] = true; });
    var read = {};
    (step.reads || []).forEach(function (t) { read[t] = true; });

    var html = '';
    idx.tensorNames.forEach(function (name) {
      var spec = idx.trace.tensors[name] || {};
      var rec = shown[name] || { value: undefined, from: null };
      var cls = wrote[name] ? 'lab-t-wrote' : (read[name] ? 'lab-t-read' : '');
      var provenance = rec.source === 'init' ? '初值'
        : rec.source === 'write' ? '第 ' + rec.from + ' 步写入'
        : '';

      html += '<div class="lab-trow ' + cls + '" data-tensor="' + esc(name) + '">';
      html += '<div class="lab-tname">';
      html += '<span class="lab-tname-id">' + esc(name) + '</span>';
      html += '<span class="lab-tshape">' + esc(shapeLabel(spec.shape)) + '</span>';
      html += '<span class="lab-tat">@ ' + esc(spec.at || '?') + '</span>';
      html += '<span class="lab-tbadges">';
      if (wrote[name]) html += '<span class="lab-badge lab-badge-w">写</span>';
      if (read[name]) html += '<span class="lab-badge lab-badge-r">读</span>';
      html += '</span>';
      if (provenance) html += '<span class="lab-tfrom">' + esc(provenance) + '</span>';
      html += '</div>';
      html += '<div class="lab-tval">' +
              renderValue(spec, rec.value, ref ? (ref[name] || {}).value : undefined,
                          wrote[name] && when === 'after') +
              '</div>';
      html += '</div>';
    });
    return html;
  }

  NS.tensors = {
    render: render,
    formatCell: formatCell,
    formatNumber: formatNumber,
    flatten: flatten,
    shapeLabel: shapeLabel
  };
})(typeof window !== 'undefined' ? window : globalThis);
