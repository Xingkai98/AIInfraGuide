/* L00 engine · DAG layout — pure, no DOM.
 *
 * Layer assignment is Kahn's algorithm taking the LONGEST path to each node, so
 * a node sits one column right of its furthest-reaching predecessor.
 *
 * The one non-obvious rule here is the SYNTHETIC SEQUENCE EDGE, and it is not
 * an optimisation — without it the layout is wrong for a whole class of traces.
 *
 * The trace's real edges are data dependencies. A step like "load block into
 * SRAM" reads `x` and writes only a staging buffer, so nothing in the graph
 * depends on it and nothing it depends on has a producer: Kahn leaves it at
 * layer 0. One such step per block means layer 0 grows linearly with the block
 * count while every other column stays 1 wide — the canvas degenerates into a
 * strip (measured on the prototype: 10 nodes -> widest layer 3; 74 nodes ->
 * widest layer 19).
 *
 * Adding `step i -> step i+1` for consecutive trace order fixes it: those edges
 * participate in layering only and are NEVER DRAWN. With them every layer of a
 * linear algorithm holds exactly one node, and the layout is clean.
 *
 * Past ~10 layers a single row is a 25:1 ribbon, so layers wrap into bands of
 * `wrapAt` columns — the second half of the same fix.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});

  /* Node box geometry, in SVG user units. The layout is expressed in these and
   * the SVG scales the whole thing with preserveAspectRatio, so the numbers
   * only have to be internally consistent. */
  var NG = {
    w: 78,
    h: 27,
    xgap: 104,
    ygap: 34,
    pad: 18,
    /* Extra horizontal room reserved on the right when wrapping, so the
     * band-to-band connector has a gutter to run in. Without it the connector
     * is routed outside the viewBox and silently clipped. */
    gutter: 104,
    /* Extra vertical space between bands, for the same reason. */
    bandGap: 34,
    /* The canvas aspect ratio the column/band split aims for. The DAG panel is
     * roughly this shape, so matching it means the SVG fills the panel instead
     * of being letterboxed into a thin strip. */
    targetAspect: 3.2,
    /* Default columns per band when a caller asks for a fixed wrap. */
    wrapAt: 6
  };

  /**
   * Choose how many columns a band gets.
   *
   * Picking a fixed number does not work: the right count depends on the layer
   * count. A chain of 10 layers in one row is ~1750x63 — 28:1, which is worse
   * than the 8:1 ribbon wrapping exists to prevent, and it renders each node
   * about 70px wide. The same 10 layers split 4/4/2 come out ~660x185 (3.6:1)
   * with 180px nodes.
   *
   * So the split is chosen to land near the panel's own aspect ratio: for each
   * candidate column count compute the resulting canvas shape and keep the
   * closest. That reproduces P01's hand-picked values (it used 4 columns for a
   * 10-node graph and 6 for 42) without hard-coding a table.
   */
  function chooseColumns(numLayers) {
    var best = { cols: numLayers, aspect: 0, bands: 1 };
    var bestErr = Infinity;
    for (var c = 1; c <= numLayers; c++) {
      var bands = Math.ceil(numLayers / c);
      var cols = Math.ceil(numLayers / bands);
      var w = NG.pad * 2 + cols * NG.w + Math.max(0, cols - 1) * NG.xgap;
      var h = NG.pad * 2 + bands * NG.h + Math.max(0, bands - 1) * NG.bandGap;
      var aspect = w / h;
      var err = Math.abs(aspect - NG.targetAspect);
      if (err < bestErr - 1e-9) {
        bestErr = err;
        best = { cols: cols, aspect: aspect, bands: bands };
      }
    }
    return best;
  }

  function layout(idx, opts) {
    opts = opts || {};
    var wrapAt = opts.wrapAt || 0;

    var nodes = idx.nodes;
    var byId = {};
    nodes.forEach(function (n) { byId[n.id] = n; });

    var preds = {}, succs = {};
    nodes.forEach(function (n) { preds[n.id] = []; succs[n.id] = []; });

    var all = idx.edges.slice();
    /* Synthetic sequence edges, in trace order. `seq: true` marks them as
     * layout-only so the renderer skips them. */
    for (var i = 0; i + 1 < nodes.length; i++) {
      all.push({ from: nodes[i].id, to: nodes[i + 1].id, seq: true, tensor: '' });
    }

    all.forEach(function (e) {
      if (!byId[e.from] || !byId[e.to]) return;
      succs[e.from].push(e.to);
      preds[e.to].push(e.from);
    });

    var indeg = {};
    nodes.forEach(function (n) { indeg[n.id] = preds[n.id].length; });
    var layer = {};
    nodes.forEach(function (n) { layer[n.id] = 0; });

    var queue = nodes
      .filter(function (n) { return indeg[n.id] === 0; })
      .map(function (n) { return n.id; });
    var enqueued = {};
    queue.forEach(function (id) { enqueued[id] = true; });
    while (queue.length) {
      var id = queue.shift();
      succs[id].forEach(function (s) {
        if (layer[id] + 1 > layer[s]) layer[s] = layer[id] + 1;
        if (--indeg[s] === 0 && !enqueued[s]) { enqueued[s] = true; queue.push(s); }
      });
    }

    var cols = {};
    nodes.forEach(function (n) {
      var L = layer[n.id];
      (cols[L] = cols[L] || []).push(n.id);
    });

    var layerIds = Object.keys(cols).map(Number);
    var numLayers = nodes.length ? Math.max.apply(null, layerIds) + 1 : 0;
    var widest = 1;
    layerIds.forEach(function (L) { widest = Math.max(widest, cols[L].length); });

    /* A chain of N layers in one row is N*182 x 63 — a ribbon, and the wider the
     * chain the worse it gets. Wrapping is what turns it back into something
     * with a sane aspect ratio.
     *
     * Three modes, in priority order:
     *   nowrap   — force a single row (the layout tests compare against this,
     *              and it is what a caller gets if they insist)
     *   wrapAt>0 — force that many columns per band
     *   default  — choose the split from the layer count so the canvas lands
     *              near targetAspect (see chooseColumns)
     *
     * The default wraps even at ten layers. P01's report reads as if a short
     * chain should stay in one row, but its own recommended screenshot for the
     * 10-node graph is the four-column one: at ten layers a single row is
     * ~28:1 and each node renders about 70px wide, which is not what
     * "readable" looks like. */
    var split;
    if (opts.nowrap) {
      split = { cols: numLayers, bands: 1 };
    } else if (wrapAt > 0) {
      split = { cols: wrapAt, bands: Math.ceil(numLayers / wrapAt) };
    } else {
      split = chooseColumns(numLayers);
    }
    var doWrap = split.bands > 1;
    var colsPerBand = split.cols;
    var bandCount = split.bands;

    var bandH = widest * NG.h + (widest - 1) * NG.ygap;
    var pos = {};

    layerIds.forEach(function (L) {
      var ids = cols[L];
      var band = Math.floor(L / colsPerBand);
      var col = L % colsPerBand;
      var span = ids.length * NG.h + (ids.length - 1) * NG.ygap;
      var top = NG.pad + band * (bandH + NG.bandGap) + (bandH - span) / 2;
      ids.forEach(function (nid, k) {
        pos[nid] = {
          x: NG.pad + col * (NG.w + NG.xgap),
          y: top + k * (NG.h + NG.ygap),
          layer: L,
          band: band
        };
      });
    });

    var W = NG.pad * 2 + colsPerBand * NG.w + Math.max(0, colsPerBand - 1) * NG.xgap +
            (bandCount > 1 ? NG.gutter : 0);
    var H = NG.pad * 2 + (bandCount ? bandCount * bandH + (bandCount - 1) * NG.bandGap : 0);

    return {
      pos: pos,
      layer: layer,
      cols: cols,
      W: Math.max(W, 1),
      H: Math.max(H, 1),
      colsPerBand: colsPerBand,
      bandCount: bandCount,
      numLayers: numLayers,
      widest: widest,
      doWrap: doWrap,
      ng: NG
    };
  }

  NS.layout = { compute: layout, NG: NG };
})(typeof window !== 'undefined' ? window : globalThis);
