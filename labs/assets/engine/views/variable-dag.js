/* L00 engine · view: variable DAG — nodes are VARIABLES, edges are COMPUTATIONS.
 *
 * This is a different picture of the same trace than `engine/dag.js` draws. That
 * one lays out the STEPS (plus synthetic sequence edges) so a reader can see the
 * order of operations; this one lays out the TENSORS and connects them with the
 * computations that move data between them, so a reader can see the algorithm.
 * Both read the same `tensors` + `steps[].reads/writes/state`; neither needs a
 * field the trace does not already carry.
 *
 * THE SHAPE OF THE VIEW
 *   [ graph | formula sidebar ]        one screen, no page scroll
 *   [        control bar      ]        pinned to the bottom of the view
 *
 * "One screen" is a per-STEP budget, not a per-lab one: whatever step the cursor
 * is on — its graph, its formula, its values — fits without scrolling. That is
 * the whole reason the four-panel instrument layout is not reused here; four
 * stacked panels cannot make that promise.
 *
 * WHAT THIS FILE DELIBERATELY DOES NOT DO
 *   - It does not render the formula. `engine/formula.js` owns the three tiers
 *     and the \slot / \region preprocessor, and it is reused through
 *     buildSkeleton()/patch() rather than reimplemented. The three traps in that
 *     file's header (KaTeX eats injected HTML; a \region body can contain
 *     \slot; a slot's value is LaTeX, not text) are traps this file has no
 *     opinion about.
 *   - It does not reconstruct state. `engine/trace-model.js` owns render(trace,i)
 *     as a pure function, and the value grid reads TM.resolve().
 *   - It does not know what the variables are called or what the formulas say.
 *     The `slot -> tensor` map comes from the trace (`step.binding_vars`), which
 *     the generator writes because it is the only side that knows both that LaTeX
 *     writes the tensor `l` as `\ell` and that `x` must not match inside `x_blk`.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var TM = NS.traceModel;

  /* Geometry is fixed in viewBox units and scaled to fit by
     preserveAspectRatio, so a narrow panel shows the same drawing smaller
     rather than a reflowed one.
     These are the DEFAULTS; a lab whose graph is deeper or wider than L00's
     six-node one passes its own through `opts.geo`. The numbers are a budget,
     not a style: a node's height times the band count plus the gaps is the
     drawing's height, and the drawing's height against the panel's is what
     decides whether the labels end up readable at 13px or at 4px. */
  var GEO = {
    /* L00's numbers, and they stay L00's: a lab that passes its own `geo` is
       not entitled to move the defaults for everyone else, and the L00 page
       does not pass one. L04 does (its graph is twice as deep), which is what
       the option is for. */
    NW: 116, NH: 46, GAPX: 112, GAPY: 74, PAD: 26,
    LOOP_W: 34, LOOP_GAP: 18,
    /* A layered drawing has edges that skip over a layer. Drawn as one bezier
       they pass straight through the boxes of the layer they skipped -- see
       `railPath` for what is done instead and why. RAIL_PAD is the width of
       the outermost rail's margin, so it has to fit the operation label that
       sits on the rail, not just the wire. */
    RAIL_GAP: 20, RAIL_PAD: 36,
    /* The smallest scale at which a node label is still a label. Below it the
       drawing is rendered at this scale and the panel pans, rather than being
       shrunk into a thumbnail. */
    MIN_SCALE: 0.8
  };

  var SENTINEL_TEXT = (TM && TM.SENTINEL_TEXT) || {};

  function esc(s) {
    return String(s === undefined || s === null ? '' : s)
      .replace(/[&<>"]/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
      });
  }

  function cls(id) {
    return 'vd-n-' + String(id).replace(/[^A-Za-z0-9_]/g, '_');
  }

  /* `el.className = …` throws on an SVG element: the property exists but has
     only a getter (it is an SVGAnimatedString). Setting the attribute is the
     one form that works for both HTML and SVG. */
  function setClass(el, value) {
    if (el) el.setAttribute('class', value);
  }

  /* ============================================================ model
   *
   * The graph, derived from the trace. Pure: same trace in, same model out, no
   * DOM. Keeping it separate is what lets the acceptance harness assert the
   * edges against the trace's own reads/writes rather than against pixels.
   */

  function build(trace, opts) {
    var idx = TM.index(trace);
    /* How an edge is chosen, and why it is a choice.
     *
     * `carried` (the default) draws one edge per step: from the value the
     * update is *about* to the variable it updates. L00 is the trace that
     * motivated it -- `reads[0]` there is `x_blk` on every op step, so the
     * naive rule draws a star out of the staging buffer and never shows that m
     * gates l and acc. One edge per step is exactly right when a step carries
     * one value forward, and it is what L00's acceptance harness asserts.
     *
     * `operands` draws one edge per (read, write) pair the step actually
     * performs. A step with several operands -- a QKV projection, a residual
     * add, an elementwise multiply -- writes several variables from several
     * reads, and the carried rule silently drops all but one of them: L04's
     * `x.scores` reads (q, k) and the carried rule keeps only q; `x.res1` reads
     * (x, attn_out) and it keeps only attn_out, so the residual connection
     * disappears from a drawing whose whole subject is the residual
     * connection. Neither rule is a special case of the other: pick by whether
     * the trace's steps are single-carried-value updates or multi-operand
     * computations, which is a property of the ALGORITHM the trace replays,
     * not of the lab. */
    var mode = (opts && opts.edgeMode) === 'operands' ? 'operands' : 'carried';
    var idxTM = idx;

    /* The nodes are the TENSORS, not `graph.nodes` -- that array is the step
       graph (init, ld1, m1, …) and it is what the engine's step DAG draws. The
       whole point of this view is that the nodes are the variables, so the node
       list comes from `tensors` and the steps become the edges. */
    var declared = Object.keys(trace.tensors || {});
    var nodes = declared.slice();

    var steps = trace.steps.map(function (s, i) {
      var reads = (s.reads || []).filter(function (t) { return declared.indexOf(t) !== -1; });
      var writes = (s.writes || []).filter(function (t) { return declared.indexOf(t) !== -1; });
      return {
        i: i,
        id: s.id,
        title: s.title || s.id,
        op: s.op || '',
        kind: s.kind || 'op',
        reads: reads,
        writes: writes,
        formula: s.formula || {},
        bindings: s.bindings || {},
        bindingVars: s.binding_vars || null,
        state: s.state || {},
        regions: s.regions || {},
        corr: s.corr || null,
        narration: s.narration || '',
        k: s.k
      };
    });

    /* --- the drawn edges.
     *
     * One edge per step, from the step's carried input to the variable it
     * updates. Choosing the carried read rather than reads[0] is the whole
     * difference between a picture of the algorithm and a picture of a fan-out:
     * `reads[0]` is `x_blk` on every op step of every L00 config, so the naive
     * rule draws a star out of the staging buffer and never shows that m gates
     * l and acc. The carried read is the one the update is *about*.
     *
     * Edges are keyed by their (from, to) pair and drawn ONCE, carrying every
     * step that uses them. Two steps that move data along the same dependency
     * (block 1's `m` update and block 2's) are the same line on screen, and two
     * pixel-identical <path>s cannot be painted independently -- the later one
     * simply covers the earlier, so the progress trail would be whichever was
     * written last. Sharing the element and deriving its state from its whole
     * step set is the only version of this that stays correct.
     */
    /* Layers first: the edge derivation needs to know which variable sits
       upstream of which, so the ordering and the drawing are decided by the
       same computation rather than by two that can disagree. */
    var layers = computeLayers(nodes, steps);
    var depth = layers.depth;

    var edgeMap = {}, edges = [];
    var selfLoops = {}, loops = [];

    function addEdge(from, to, step, isCarried) {
      var key = from + '>' + to;
      if (!edgeMap[key]) {
        edgeMap[key] = { from: from, to: to, steps: [], ops: [], carried: false };
        edges.push(edgeMap[key]);
      }
      edgeMap[key].steps.push(step.i);
      if (edgeMap[key].ops.indexOf(step.op) === -1) edgeMap[key].ops.push(step.op);
      /* WHICH EDGE OF A STEP WEARS THE STEP'S NAME.
       *
       * `operands` mode draws every dependency a step performs, and a step like
       * L04's `x.attn` performs six of them -- all with the same op. Labelling
       * all six prints "ATTENTION" six times across one band, which is not six
       * facts, it is one fact said six times, and the labels collide with each
       * other and with the wires. So the LABEL follows the rule the whole graph
       * follows in `carried` mode: the step names the edge it carries its
       * result along, the one from the value the update is *about*. The other
       * edges are still drawn -- they are dependencies and dropping them is
       * what this mode exists to stop -- they just do not repeat the name. */
      if (isCarried !== false) edgeMap[key].carried = true;
    }

    function addLoop(id, step) {
      if (!selfLoops[id]) {
        selfLoops[id] = { id: id, steps: [], ops: [] };
        loops.push(selfLoops[id]);
      }
      selfLoops[id].steps.push(step.i);
      if (selfLoops[id].ops.indexOf(step.op) === -1) selfLoops[id].ops.push(step.op);
    }

    steps.forEach(function (s) {
      if (!s.writes.length) return;

      /* A step that reads what it writes is carrying a value across iterations,
         not moving data between two variables. That is a self-loop, drawn as
         one, once per variable, however many steps carry it. The recurrence
         `m <- max(m, .)` IS the algorithm's shape, so it has to be visible
         rather than discarded as a degenerate edge. */
      var selfReading = s.writes.filter(function (w) { return s.reads.indexOf(w) !== -1; });
      selfReading.forEach(function (w) { addLoop(w, s); });

      /* The variable this step produces: the write that sits furthest along,
         which is the step's output rather than one of its inputs. */
      var target = s.writes[0];
      s.writes.forEach(function (w) {
        if ((depth[w] || 0) > (depth[target] || 0)) target = w;
      });

      /* Among the reads strictly BEFORE the target, take the one furthest
         along: that is the value being carried forward, and it is what the
         formula's \slot names are about (m from the block, l from m, acc from
         m and l, the output from both). Taking reads[0] instead draws a star
         out of whichever buffer happens to be listed first -- `x_blk` on every
         op step of every L00 config -- and never shows that m gates the rest. */
      var source = null;
      s.reads.forEach(function (r) {
        if (r === target) return;
        if ((depth[r] || 0) >= (depth[target] || 0)) return;
        if (!source || (depth[r] || 0) > (depth[source] || 0)) source = r;
      });
      var carriedKey = source ? source + '>' + target : null;

      if (mode === 'operands') {
        /* Every operand of the step reaches every variable it produces. Both
           ends are the trace's own reads/writes, so an edge drawn here is a
           dependency the replay actually performed -- nothing is inferred from
           proximity or from the order the arrays happen to be listed in. */
        s.reads.forEach(function (r) {
          if (selfReading.indexOf(r) !== -1) return;   /* drawn as a self-loop */
          s.writes.forEach(function (w) {
            if (r === w) return;
            /* Every layer-consuming pair is upstream by construction, so this
               guard should never fire; it is here so that a trace whose
               layering disagrees with its reads/writes reports a back-edge
               instead of drawing an arrow that points backwards. */
            if ((depth[r] || 0) >= (depth[w] || 0)) return;
            addEdge(r, w, s, r + '>' + w === carriedKey);
          });
        });
        return;
      }

      if (!source) return;
      addEdge(source, target, s, true);
    });

    return {
      nodes: nodes,
      tensors: trace.tensors || {},
      steps: steps,
      edges: edges,
      edgeMode: mode,
      showLocation: !opts || opts.nodeLocation !== false,
      loops: loops,
      layers: layers,
      /* Steps that draw nothing at all: no reads and no write we can attribute
         (the initialiser). Reported so the header can say so instead of leaving
         a reader to wonder why step 1 has no edge. */
      edgeLess: steps.filter(function (s) {
        return !s.reads.length && !s.writes.length;
      }).map(function (s) { return s.i; }),
      initSteps: steps.filter(function (s) {
        return !s.reads.length && s.writes.length;
      }).map(function (s) { return s.i; }),
      idx: idx
    };
  }

  /* ============================================================ layout */

  function computeLayers(nodes, steps) {
    var layer = {};
    nodes.forEach(function (n) { layer[n] = 0; });

    /* Every read -> write pair constrains the layering, so a dependency that is
       not the drawn edge (x_blk feeding l directly, say) still pushes l below
       x_blk. Self-loops are skipped: `layer[m] = max(layer[m], ...) + 1` never
       converges. */
    var pairs = [];
    steps.forEach(function (s) {
      s.reads.forEach(function (r) {
        s.writes.forEach(function (w) {
          if (r !== w) pairs.push([r, w]);
        });
      });
    });

    for (var pass = 0; pass < nodes.length + 2; pass++) {
      pairs.forEach(function (p) {
        if (layer[p[0]] === undefined || layer[p[1]] === undefined) return;
        layer[p[1]] = Math.max(layer[p[1]], layer[p[0]] + 1);
      });
    }

    /* `base + 1` accumulates gaps -- the layers that come out are routinely not
       contiguous (0, 1, 17, 18, 19 was measured on a longer trace). Anything
       that sizes the canvas from the layer NUMBER gets a canvas thousands of
       units wide with the drawing crushed into one corner, so the layers are
       renumbered to the set that actually occurs before anything is measured. */
    var used = [];
    nodes.forEach(function (n) {
      if (used.indexOf(layer[n]) === -1) used.push(layer[n]);
    });
    used.sort(function (a, b) { return a - b; });
    var rank = {};
    used.forEach(function (L, i) { rank[L] = i; });

    /* Two lookups, and they are not interchangeable: `rank` maps a layer NUMBER
       to its position in the compacted sequence, `depth` maps a NODE to that
       position. Indexing the first by a node name yields undefined for every
       node, which silently degrades "the furthest-progressed input" to
       "whatever reads[0] happens to be" -- and reads[0] is x_blk on every op
       step, so the graph comes out as a star out of the staging buffer. */
    var bands = [];
    var depth = {};
    nodes.forEach(function (n) {
      var b = rank[layer[n]];
      depth[n] = b;
      (bands[b] = bands[b] || []).push(n);
    });
    return { bands: bands, nBands: used.length, raw: layer, rank: rank, depth: depth };
  }

  function computePositions(model, targetAspect, vertical, geo) {
    var G = geo || GEO;
    var NW = G.NW, NH = G.NH, GAPX = G.GAPX, GAPY = G.GAPY, PAD = G.PAD;
    /* Reuse the layering the edge derivation used, so the drawing cannot be
       laid out on a different ordering than the one the edges were chosen
       against. */
    var L = model.layers || computeLayers(model.nodes, model.steps);
    var widest = 1;
    L.bands.forEach(function (b) { widest = Math.max(widest, b.length); });

    /* Which way round? The canvas is letterboxed into the panel, so the wrong
       choice squashes the drawing into a band across the middle. The judgement
       has to be the DISTANCE to the panel's aspect ratio, not a comparison of
       the two candidate ratios: for a square-ish panel the vertical arrangement
       (0.7:1) is the closer fit, but 0.7 < 5.0, so "pick the larger" selects the
       horizontal one and crushes the graph. Log distance, because the ratios are
       symmetric -- 2:1 is as far from 1:1 as 1:2 is. */
    var wH = L.nBands * NW + (L.nBands - 1) * GAPX;
    var hH = widest * NH + (widest - 1) * GAPY;
    var wV = widest * NW + (widest - 1) * GAPX;
    var hV = L.nBands * NH + (L.nBands - 1) * GAPY;

    if (vertical === undefined || vertical === null) {
      var target = targetAspect > 0 ? targetAspect : 1;
      vertical = Math.abs(Math.log((wV / hV) / target)) <=
                 Math.abs(Math.log((wH / hH) / target));
    }

    var pos = {};
    L.bands.forEach(function (band, b) {
      band.forEach(function (id, i) {
        pos[id] = vertical
          ? { x: PAD + i * (NW + GAPX), y: PAD + b * (NH + GAPY), band: b }
          : { x: PAD + b * (NW + GAPX), y: PAD + i * (NH + GAPY), band: b };
      });
    });

    var maxX = 0, maxY = 0;
    Object.keys(pos).forEach(function (id) {
      maxX = Math.max(maxX, pos[id].x);
      maxY = Math.max(maxY, pos[id].y);
    });

    /* A self-loop bulges out to the right of its node, so the canvas has to
       reserve that width or the loop is cropped at the edge. */
    var right = model.loops.length ? G.LOOP_GAP + G.LOOP_W + PAD : PAD;

    /* An edge that skips a layer is routed on a rail outside the row of boxes
       (`railPath`); the canvas has to reserve that space or the rail is cropped
       at the edge. One side per orientation -- below a horizontal drawing,
       right of a vertical one -- so the reserve is a single number and no
       coordinate has to be shifted. */
    var rails = railPlan(model, pos);
    var nRails = rails.nRails;
    var extra = nRails ? G.RAIL_PAD + nRails * G.RAIL_GAP : 0;

    return {
      pos: pos,
      vertical: vertical,
      bands: L.bands,
      nBands: L.nBands,
      rails: rails,
      railCount: nRails,
      geo: G,
      w: maxX + NW + right + (vertical ? extra : 0),
      h: maxY + NH + PAD + (vertical ? 0 : extra)
    };
  }

  /* ---------------------------------------------------------------- rails
   *
   * Which edges skip a layer.
   *
   * A layered drawing puts every edge between adjacent layers inside the gap,
   * where there is nothing to collide with. An edge that spans two or more
   * bands -- L04's residual additions, which carry `x` from band 0 to band 5 --
   * drawn as a single bezier passes straight through every box in the bands it
   * skips. That is not a hypothesis: it is what the first version drew, and the
   * harness's "no edge through an unrelated box" predicate is what found it.
   *
   * The fix that needs no hand-placed geometry: route the edge out to a rail
   * outside the drawing, along it past the skipped bands, and back in at the
   * target. The rail's offset is a function of how many bands the edge skips,
   * so edges that skip further run further out and the ones that skip least
   * stay closest to the boxes they pass. The number of rails is therefore
   * bounded by the band count, not by the number of edges, and the canvas can
   * reserve exactly the space they need.
   */
  function railPlan(model, pos) {
    var levels = [], skipping = [];
    (model.edges || []).forEach(function (e) {
      var a = pos[e.from], b = pos[e.to];
      if (!a || !b) return;
      var d = b.band - a.band;
      e.bandDelta = d;
      if (Math.abs(d) <= 1) { e.rail = null; return; }
      /* Level 0 is the innermost rail: an edge that skips one band travels
         closest to the boxes it passed, and one that skips five sits further
         out. That ordering is what keeps the rails from coinciding. */
      e.rail = {level: Math.abs(d) - 2};
      skipping.push(e);
      if (levels.indexOf(e.rail.level) === -1) levels.push(e.rail.level);
    });
    levels.sort(function (x, y) { return x - y; });
    /* The rails are ranked by POSITION IN THE SORTED LIST, not by their level
       number. Levels are `|bandDelta| - 2`, so a graph whose edges all skip one
       or two bands produces levels 0 and 1 but one that only ever skips one and
       six produces 0 and 4 -- and `railCount - 1 - level` then goes negative and
       the rail is drawn off the canvas. Exactly the bug the pilot recorded for
       layer numbers; it recurred here on the first graph deeper than L00's. */
    skipping.forEach(function (e) { e.rail.rank = levels.indexOf(e.rail.level); });
    return { levels: levels, nRails: levels.length, edges: skipping };
  }

  /* ============================================================ geometry
   *
   * One place where a path between two nodes is turned into a curve, so the
   * drawing code and the harness's geometry predicates agree about where an
   * edge actually is.
   */

  /* Adjacent bands: the straight/offset bezier inside the gap, unchanged. A
     skipped band goes to `railPath` instead -- see `railPlan` for why. */
  function edgePath(L, from, to) {
    var e = null, edges = (L.rails && L.rails.edges) || [];
    for (var k = 0; k < edges.length; k++) {
      if (edges[k].from === from && edges[k].to === to) { e = edges[k]; break; }
    }
    if (e && e.rail) return railPath(L, e);
    return directPath(L, from, to);
  }

  function directPath(L, from, to) {
    var a = L.pos[from], b = L.pos[to];
    if (!a || !b) return null;
    var NW = L.geo.NW, NH = L.geo.NH;
    var x1, y1, x2, y2, d, lx, ly;

    if (L.vertical) {
      x1 = a.x + NW / 2; y1 = a.y + NH;
      x2 = b.x + NW / 2; y2 = b.y;
      if (Math.abs(x1 - x2) < 3) {
        d = 'M' + x1 + ' ' + y1 + ' L' + x2 + ' ' + y2;
      } else {
        var my = (y1 + y2) / 2;
        d = 'M' + x1 + ' ' + y1 + ' C' + x1 + ' ' + my + ' ' + x2 + ' ' + my + ' ' + x2 + ' ' + y2;
      }
      lx = (x1 + x2) / 2 + 9; ly = (y1 + y2) / 2;
    } else {
      x1 = a.x + NW; y1 = a.y + NH / 2;
      x2 = b.x; y2 = b.y + NH / 2;
      if (Math.abs(y1 - y2) < 3) {
        d = 'M' + x1 + ' ' + y1 + ' L' + x2 + ' ' + y2;
      } else {
        var mx = (x1 + x2) / 2;
        d = 'M' + x1 + ' ' + y1 + ' C' + mx + ' ' + y1 + ' ' + mx + ' ' + y2 + ' ' + x2 + ' ' + y2;
      }
      lx = (x1 + x2) / 2; ly = Math.min(y1, y2) - 7;
    }
    return { d: d, lx: lx, ly: ly, x1: x1, y1: y1, x2: x2, y2: y2, rail: false };
  }

  /* A self-loop leaves the node's right edge and comes back to it. It is drawn
     as a bezier outside the box so it never crosses the node's own label. */
  function loopPath(L, id) {
    var p = L.pos[id];
    if (!p) return null;
    var NW = L.geo.NW, NH = L.geo.NH, G = L.geo;
    var x0 = p.x + NW, yTop = p.y + NH * 0.26, yBot = p.y + NH * 0.74;
    var reach = x0 + G.LOOP_GAP + G.LOOP_W;
    var d = 'M' + x0 + ' ' + yTop +
            ' C' + reach + ' ' + yTop + ' ' + reach + ' ' + yBot + ' ' + x0 + ' ' + yBot;
    return { d: d, lx: reach - 2, ly: (yTop + yBot) / 2 };
  }

  /* An edge that skips one or more bands: out of the source, along a rail
     outside the drawing, and back in at the target.
   *
   * WHERE EVERY SEGMENT RUNS, and why. The obvious route -- leave the source's
   * side, run straight to the rail, run along it, come back -- passes through
   * the source's own band-mates: in L04's `q, k, v` band, leaving `q` sideways
   * crosses `k` and `v`. So each segment is placed in the empty channel between
   * two bands instead:
   *
   *   vertical    down out of the source into the gap below its band, across to
   *               the rail through that gap, along the rail (which is clear of
   *               everything by construction), back through the gap above the
   *               target's band, and in through the target's top edge.
   *   horizontal  out of the source into the gap on its downstream side, down
   *               through that gap to the rail below the drawing, across, and up
   *               through the gap on the target's upstream side.
   *
   * Both routes only ever travel inside a gap or along the rail, so no segment
   * can cross a box -- which is the property the harness measures.
   *
   * The rail's offset grows with the number of bands skipped, so longer detours
   * run further out and no two rails coincide.
   */
  var RAIL_R = 9;     /* corner chamfer, in viewBox units */

  function railPath(L, e) {
    var a = L.pos[e.from], b = L.pos[e.to];
    if (!a || !b) return null;
    var G = L.geo;
    var NW = G.NW, NH = G.NH, GX = G.GAPX, GY = G.GAPY;
    var rank = L.railCount - 1 - e.rail.rank;
    var d, lx, ly;

    /* A polyline with chamfered corners.
     *
     * Each interior vertex is cut back along the INCOMING ray and forward along
     * the OUTGOING one, and a quadratic through the vertex joins the two. The
     * first version rounded toward the NEXT point instead of away from the
     * previous one, which put the line's endpoint near the far end of the
     * segment and made the path double back through it -- the drawn `d` swept
     * across the whole canvas and the harness's "no edge through an unrelated
     * box" predicate is what caught it. The cut-back length is capped by half of
     * EACH adjacent segment so two short segments cannot eat into each other. */
    function poly(pts) {
      var out = 'M' + pts[0].x + ' ' + pts[0].y;
      for (var i = 1; i < pts.length; i++) {
        var p = pts[i];
        if (i === pts.length - 1) { out += ' L' + p.x + ' ' + p.y; break; }
        var nx = pts[i + 1].x - p.x, ny = pts[i + 1].y - p.y;
        var nlen = Math.sqrt(nx * nx + ny * ny);
        var px = p.x - pts[i - 1].x, py = p.y - pts[i - 1].y;
        var plen = Math.sqrt(px * px + py * py);
        if (nlen < 1e-6 || plen < 1e-6) continue;
        var r = Math.min(RAIL_R, nlen / 2, plen / 2);
        out += ' L' + (p.x - px / plen * r) + ' ' + (p.y - py / plen * r) +
               ' Q' + p.x + ' ' + p.y + ' ' +
               (p.x + nx / nlen * r) + ' ' + (p.y + ny / nlen * r);
      }
      return out;
    }

    if (L.vertical) {
      var rail = L.w - G.RAIL_PAD - rank * G.RAIL_GAP;
      var x0 = a.x + NW / 2, yA = a.y + NH + GY / 2;
      var x1 = b.x + NW / 2, yB = b.y - GY / 2;
      d = poly([
        { x: x0, y: a.y + NH }, { x: x0, y: yA }, { x: rail, y: yA },
        { x: rail, y: yB }, { x: x1, y: yB }, { x: x1, y: b.y }
      ]);
      lx = rail; ly = (yA + yB) / 2;
    } else {
      var railY = L.h - G.RAIL_PAD - rank * G.RAIL_GAP;
      var ya = a.y + NH / 2, xa = a.x + NW + GX / 2;
      var yb = b.y + NH / 2, xb = b.x - GX / 2;
      d = poly([
        { x: a.x + NW, y: ya }, { x: xa, y: ya }, { x: xa, y: railY },
        { x: xb, y: railY }, { x: xb, y: yb }, { x: b.x, y: yb }
      ]);
      lx = (xa + xb) / 2; ly = railY - 6;
    }
    return { d: d, lx: lx, ly: ly, x1: 0, y1: 0, x2: 0, y2: 0, rail: true };
  }

  /* The rectangle a node occupies, in the same coordinate space the SVG draws
     in. Exported so the harness can assert against the geometry instead of
     re-deriving it. */
  function nodeRect(L, id) {
    var p = L.pos[id];
    if (!p) return null;
    var G = (L && L.geo) || GEO;
    return { x: p.x, y: p.y, w: G.NW, h: G.NH };
  }

  /* ============================================================ lint
   *
   * The JS half of the `binding_vars` contract. The authoritative copy lives in
   * the trace generator (labs/traces/online_softmax.py), which runs it before
   * the file is written; this port exists so a page that edits a trace in the
   * browser gets the same verdict without a round trip. Both sides carry their
   * own sabotage cases -- a rule only one side has is a rule only one side
   * tests.
   */
  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var declared = Object.keys(trace.tensors || {});

    (trace.steps || []).forEach(function (s) {
      var sid = s.id;
      var bvars = s.binding_vars;
      if (!bvars || typeof bvars !== 'object' || Array.isArray(bvars)) {
        gaps.push('步骤 "' + sid + '" 没有 binding_vars —— 变量图无法把公式里的变量与图节点对应起来');
        return;
      }
      var binds = s.bindings || {};
      Object.keys(binds).forEach(function (k) {
        if (!Object.prototype.hasOwnProperty.call(bvars, k)) {
          gaps.push('步骤 "' + sid + '" 的绑定 "' + k + '" 在 binding_vars 里没有条目 —— 点这个变量不会有任何高亮');
        }
      });
      /* The step's own dataflow. An entry naming a tensor the step neither
         reads nor writes is not a harmless extra: it is a link drawn between
         two things this step has nothing to do with, which is the same class
         of error as an edge drawn on proximity. The Python port carries the
         same rule and its own sabotage case. */
      var scope = (s.reads || []).concat(s.writes || []);
      Object.keys(bvars).forEach(function (k) {
        if (!Object.prototype.hasOwnProperty.call(binds, k)) {
          gaps.push('步骤 "' + sid + '" 的 binding_vars 有 "' + k + '"，但 bindings 里没有这个 slot');
          return;
        }
        if (!Array.isArray(bvars[k])) {
          gaps.push('步骤 "' + sid + '" 的 binding_vars["' + k + '"] 不是张量名列表');
          return;
        }
        bvars[k].forEach(function (name) {
          if (declared.indexOf(name) === -1) {
            gaps.push('步骤 "' + sid + '" 的 binding_vars["' + k + '"] 指向未声明的张量 "' + name + '" —— 图上没有这个节点可高亮');
          } else if (scope.indexOf(name) === -1) {
            gaps.push('步骤 "' + sid + '" 的 binding_vars["' + k + '"] 指向 "' + name +
              '"，但这一步既不读也不写它 —— 高亮会连到一个与本步无关的节点');
          }
        });
      });
    });

    var unread = [];
    (trace.steps || []).forEach(function (s) {
      if (!s.binding_vars) unread.push(s.id);
    });
    if (!unread.length) {
      infos.push('binding_vars 覆盖 ' + (trace.steps || []).length + ' 步');
    }
    return { gaps: gaps, warns: warns, infos: infos };
  }

  /* This view's own control group, in the same shape the other views use: each
   * entry breaks one rule on the REAL trace, and a lint that reports zero gaps
   * for a broken copy is a lint that is not checking.
   *
   * WHAT IS NOT HERE, AND WHY. Two of the ways `binding_vars` can go wrong are
   * structurally valid and semantically false -- an entry emptied, or rewired
   * to a different tensor the step happens to touch -- and this lint cannot
   * judge them, because "which tensor SHOULD the ℓ slot name" is a fact about
   * L04's formulas that a generic view has no business knowing. Those rules
   * live in the generator (`SLOT_VARS_REQUIRED` in decoder_block.py) and their
   * control cases live beside it. The harness requires every case to be caught
   * by at least one port, so moving them there is a decision about which file
   * owns the rule, not a way of dropping the test.
   *
   * The mutations reach for steps and slots by their ROLES, so they keep biting
   * if the replay's shape changes: `stepWith` finds the first step carrying the
   * slot rather than indexing a literal, and the slot is named by the step's
   * own formula rather than by a string written twice. */
  function stepWith(trace, slot, linked) {
    var out = -1;
    (trace.steps || []).forEach(function (s, i) {
      if (out !== -1 || !s.binding_vars) return;
      var names = s.binding_vars[slot];
      if (!names) return;
      if (linked ? names.length > 0 : names.length === 0) out = i;
    });
    if (out === -1) throw new Error('no step with a ' + (linked ? 'linked' : 'empty') +
      ' "' + slot + '" binding');
    return out;
  }

  var SABOTAGE_CASES = {
    'binding_vars 整块删除': function (t) { delete t.steps[1].binding_vars; },
    'binding_vars 少一个活 slot 的条目': function (t) {
      /* The slot is read out of the step's OWN formula, so the case cannot
         drift from the trace it is written against. */
      var i = stepWith(t, 'MU', true);
      var m = /\\slot\{([A-Za-z0-9_]+)\}/.exec(t.steps[i].formula.num ||
                                               t.steps[i].formula.idx ||
                                               t.steps[i].formula.sym);
      if (!m) throw new Error('no \\slot in the target step');
      delete t.steps[i].binding_vars[m[1]];
    },
    'binding_vars 多一个不存在的 slot': function (t) {
      t.steps[1].binding_vars.GHOST = ['x'];
    },
    'binding_vars 的条目不是列表': function (t) { t.steps[1].binding_vars.MU = 'x'; },
    'binding_vars 指向未声明的张量': function (t) {
      t.steps[stepWith(t, 'MU', true)].binding_vars.MU = ['ghost'];
    },
    'binding_vars 指向本步不读也不写的张量': function (t) {
      var i = stepWith(t, 'MU', true);
      var own = t.steps[i].binding_vars.MU[0];
      var other = Object.keys(t.tensors).filter(function (n) {
        return n !== own && t.steps[i].reads.indexOf(n) === -1 &&
               t.steps[i].writes.indexOf(n) === -1;
      })[0];
      t.steps[i].binding_vars.MU = [other];
    }
  };

  function sabotageChecks(trace) {
    var before = JSON.stringify(trace);
    var caught = [], missed = [], skipped = [], noop = [];
    Object.keys(SABOTAGE_CASES).forEach(function (name) {
      var copy = JSON.parse(before);
      try {
        SABOTAGE_CASES[name](copy);
      } catch (err) {
        /* Most of these mutate the trace, so they always apply; the failure
         * mode this bucket catches is a mutation whose preconditions are not
         * in this trace at all. Same treatment as `verify.js`. */
        skipped.push(name);
        return;
      }
      if (JSON.stringify(copy) === before) { noop.push(name); return; }
      try {
        if (lint(copy).gaps.length > 0) caught.push(name);
        else missed.push(name);
      } catch (err) {
        missed.push(name);
      }
    });
    return { caught: caught, missed: missed, skipped: skipped, noop: noop,
             total: Object.keys(SABOTAGE_CASES).length };
  }

  /* ============================================================ value grid
   *
   * A tensor's value at the cursor, laid out by its rank: a 2-D array is a grid
   * of rows, a 1-D array is a single row, a scalar is one cell. The numbers come
   * from TM.resolve(), the engine's pure reconstruction -- not from a running
   * total this view maintains, which is a second chance to disagree with the
   * trace.
   */
  function formatValue(v, digits) {
    if (v === undefined) return '—';
    if (TM && TM.isSentinel && TM.isSentinel(v)) return SENTINEL_TEXT[v];
    if (typeof v === 'number') {
      if (!isFinite(v)) return v > 0 ? '+∞' : '−∞';
      if (Number.isInteger(v)) return String(v);
      return v.toPrecision(digits || 4).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
    }
    return String(v);
  }

  function shapeText(shape) {
    if (!shape || !shape.length) return '标量';
    return '[' + shape.join('×') + ']';
  }

  /* How much of a tensor is printed, and why there is a limit at all.
   *
   * A rank-3 tensor of L04's size is 256 numbers and a `[8, 176]` activation is
   * 1408; printed in full, one step's values would be taller than the screen
   * the step is supposed to fit on, and the reader would be scrolling a wall of
   * digits to find the one they wanted. So each drawn block is capped and the
   * cap is SAID: `⋯` marks an elided row, column or slice, and the header keeps
   * the tensor's true shape. That is the same bargain `views/shape-guard.js`
   * strikes for a `[128, 64]` operand -- draw what fits, label what is true --
   * and the numbers that are drawn are the trace's own.
   *
   * The defaults are what a laptop sidebar can hold; a lab with more room
   * passes its own through `opts.maxRows` / `maxCols` / `maxSlices`. */
  var VALUE_LIMITS = { maxRows: 8, maxCols: 8, maxSlices: 2 };

  function elided(n, cls) {
    return n > 0 ? '<span class="vd-cell vd-cell-more' + (cls ? ' ' + cls : '') +
      '">⋯ ' + n + '</span>' : '';
  }

  /* One rank-2 block, capped on both axes. */
  function gridBody(rows, lim) {
    var nR = rows.length, nC = rows[0] ? rows[0].length : 0;
    var showR = Math.min(nR, lim.maxRows), showC = Math.min(nC, lim.maxCols);
    var out = '';
    for (var r = 0; r < showR; r++) {
      out += '<div class="vd-vrow">';
      for (var c = 0; c < showC; c++) {
        out += '<span class="vd-cell">' + esc(formatValue(rows[r][c])) + '</span>';
      }
      if (showC < nC) out += elided(nC - showC);
      out += '</div>';
    }
    if (showR < nR) {
      out += '<div class="vd-vrow">' + elided(nR - showR, 'vd-cell-rowmore') + '</div>';
    }
    return out;
  }

  function sliceHeader(i, n) {
    return '<div class="vd-vslice">第 ' + (i + 1) + ' / ' + n + ' 张</div>';
  }

  function valueGrid(name, model, snapshot, opts) {
    var lim = {};
    Object.keys(VALUE_LIMITS).forEach(function (k) { lim[k] = VALUE_LIMITS[k]; });
    Object.keys((opts && opts.limits) || {}).forEach(function (k) {
      lim[k] = opts.limits[k];
    });

    var spec = model.tensors[name] || {};
    var entry = snapshot[name];
    if (!entry || entry.value === undefined) return '';
    var val = entry.value;
    var shape = spec.shape || [];
    var body;

    if (!Array.isArray(val)) {
      body = '<span class="vd-cell vd-cell-1">' + esc(formatValue(val)) + '</span>';
    } else if (shape.length <= 1 || !Array.isArray(val[0])) {
      var show = Math.min(val.length, lim.maxRows * lim.maxCols);
      body = '<div class="vd-vrow">' + val.slice(0, show).map(function (v) {
        return '<span class="vd-cell">' + esc(formatValue(v)) + '</span>';
      }).join('') + elided(val.length - show) + '</div>';
    } else if (shape.length === 2 || !Array.isArray(val[0][0])) {
      body = gridBody(val, lim);
    } else {
      /* Rank 3 and up: the leading index is a STACK of grids, not another row
         of one. Drawn as one grid per slice, because `[H, N, N]` is H score
         matrices and a reader who sees them run together horizontally would
         read the whole thing as one wide matrix -- which is exactly the
         misreading the shape is there to prevent. */
      var slices = val.slice(0, lim.maxSlices).map(function (sl, i) {
        return '<div class="vd-vslice-b">' + sliceHeader(i, val.length) + gridBody(sl, lim) + '</div>';
      }).join('');
      body = '<div class="vd-vslices">' + slices +
        (val.length > lim.maxSlices
          ? '<div class="vd-vslice-b vd-vslice-more">' +
            elided(val.length - lim.maxSlices, 'vd-cell-rowmore') +
            '<div class="vd-vslice">其余 ' + (val.length - lim.maxSlices) + ' 张同形</div></div>'
          : '') +
        '</div>';
    }

    var where = entry.from === -1 ? '初值'
              : (entry.from === null ? '未产生' : '第 ' + (entry.from + 1) + ' 步写入');
    return '<div class="vd-vg"><div class="vd-vg-hd"><span class="vd-vg-n">' + esc(name) +
      '</span><span class="vd-vg-s">' + esc(shapeText(shape)) + '</span>' +
      '<span class="vd-vg-at">' + esc(where) + '</span></div>' + body + '</div>';
  }

  /* ============================================================ SVG scene */

  /* How a tensor's rank is drawn on its node.
   *
   * Rank is the one property of a variable that a reader of a dataflow graph
   * needs before any number appears: `[8,64]` and `[4,8,8]` are different
   * kinds of thing to multiply, to add, or to draw, and a graph that prints
   * only the name leaves that to the sidebar. The glyph is a filled rectangle
   * whose proportions are the shape's own leading two dimensions, clamped:
   *
   *   rank 0  one small square        a scalar
   *   rank 1  a wide flat bar         a row vector
   *   rank ≥2 two nested rectangles   a matrix (the inner one is the rank-3+
   *                                   "one of many" layer, so a batch of
   *                                   matrices does not look like a matrix)
   *
   * It is a glyph and not a data plot -- the drawn size carries the ARGUMENT
   * (how many dimensions, and which of them is the outer one), never the
   * values, and the exact shape is printed beside it in text. Generic to any
   * trace: it reads `tensors[].shape` and nothing else.
   */
  function shapeGlyph(shape, x, y) {
    var s = shape || [];
    var rank = s.length;
    if (rank === 0) {
      return '<rect class="vd-shape" x="' + x + '" y="' + y + '" width="7" height="7" rx="1.5"/>';
    }
    if (rank === 1) {
      return '<rect class="vd-shape" x="' + x + '" y="' + (y + 1.5) + '" width="16" height="5" rx="1.5"/>';
    }
    var w = 20, h = 14;
    var out = '<rect class="vd-shape" x="' + x + '" y="' + y + '" width="' + w +
              '" height="' + h + '" rx="2"/>';
    if (rank >= 3) {
      out += '<rect class="vd-shape-inner" x="' + (x + 3) + '" y="' + (y + 3) +
             '" width="' + (w - 6) + '" height="' + (h - 6) + '" rx="1.5"/>';
    }
    return out;
  }

  function scene(L, model) {
    var out = '<defs>' +
      '<marker id="vd-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">' +
        '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker>' +
      '<marker id="vd-ar-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">' +
        '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker>' +
      '</defs>';

    model.edges.forEach(function (e) {
      var g = edgePath(L, e.from, e.to);
      if (!g) return;
      e.geom = g;
      out += '<path class="vd-e vd-e-off' + (e.rail ? ' vd-e-rail' : '') +
        '" data-vd-edge="' + esc(e.from + '>' + e.to) +
        '" data-vd-rail="' + (e.rail ? '1' : '0') +
        '" d="' + g.d + '" marker-end="url(#vd-ar)"/>';
      /* Only the carried edge wears the step's name -- see `addEdge`. The rule
         is deliberately the same one the carried mode uses for its edges, so a
         graph read in either mode says the same thing about what the algorithm
         is doing; the operands mode just shows more of the wiring. */
      if (e.carried && e.ops.length === 1) {
        out += '<text class="vd-el vd-el-off" data-vd-el="' + esc(e.from + '>' + e.to) +
          '" x="' + g.lx + '" y="' + g.ly + '" text-anchor="middle">' +
          esc(e.ops[0].toUpperCase()) + '</text>';
      }
    });

    model.loops.forEach(function (lp) {
      var g = loopPath(L, lp.id);
      if (!g) return;
      lp.geom = g;
      out += '<path class="vd-e vd-e-loop vd-e-off" data-vd-loop="' + esc(lp.id) +
        '" d="' + g.d + '" marker-end="url(#vd-ar)"/>';
      out += '<text class="vd-el vd-el-loop vd-el-off" data-vd-loop-l="' + esc(lp.id) +
        '" x="' + g.lx + '" y="' + g.ly + '" text-anchor="middle">↺</text>';
    });

    model.nodes.forEach(function (id) {
      var p = L.pos[id];
      if (!p) return;
      var spec = model.tensors[id] || {};
      var NW = L.geo.NW, NH = L.geo.NH;
      out += '<g class="vd-n ' + cls(id) + '" data-vd-node="' + esc(id) + '" tabindex="0" role="button" ' +
        'aria-label="变量 ' + esc(id) + ' 形状 ' + esc(shapeText(spec.shape)) + '">' +
        '<rect class="vd-box vd-box-off" x="' + p.x + '" y="' + p.y + '" width="' + NW +
          '" height="' + NH + '" rx="10"/>' +
        '<g class="vd-nshape vd-nshape-off">' + shapeGlyph(spec.shape, p.x + 11, p.y + 17) + '</g>' +
        '<text class="vd-nl vd-nl-off" x="' + (p.x + 37) + '" y="' + (p.y + 20) + '">' + esc(id) + '</text>' +
        '<text class="vd-ns vd-ns-off" x="' + (p.x + 11) + '" y="' + (p.y + 35) + '">' +
          esc(shapeText(spec.shape)) +
          (model.showLocation && spec.at ? ' · ' + esc(spec.at) : '') + '</text>' +
        '</g>';
    });

    return out;
  }

  /* ============================================================ mount */

  var DEFAULTS = {
    title: '',
    lead: '',
    /* 默认符号档：读者先看到算法长什么样，再自己切到数值看具体算例。
       数值档把 Σᵢ exp(...) 展开成四个具体项，算法骨架反而看不出来。 */
    tier: 'sym',
    param: 'step',
    digits: 4,
    /* Node boxes are a fixed size and the drawing is scaled to fit, so a lab
       whose graph is deeper than L00's has to say so or its labels come out at
       4px. `geo` is merged over the defaults, field by field. */
    geo: null,
    edgeMode: 'carried',
    /* The second line of a node box is the tensor's shape. The first line may
       also carry `tensors[].at` -- L00's values there are short places ("HBM",
       "SRAM"), which is what a node has room for. L04's are sentences ("多头注
       意力的 Q"), and a sentence in a 112px box is not a label. */
    nodeLocation: true,
    valueLimits: null,
    onStep: null
  };

  function mount(host, trace, opts) {
    var cfg = {};
    Object.keys(DEFAULTS).forEach(function (k) { cfg[k] = DEFAULTS[k]; });
    Object.keys(opts || {}).forEach(function (k) { cfg[k] = opts[k]; });
    var geo = {};
    Object.keys(GEO).forEach(function (k) { geo[k] = GEO[k]; });
    Object.keys(cfg.geo || {}).forEach(function (k) { geo[k] = cfg.geo[k]; });
    cfg.geo = geo;

    if (!host) throw new Error('variableDag: no mount element');
    if (!NS.formula) throw new Error('engine/formula.js must be loaded first');

    var model = build(trace, {edgeMode: cfg.edgeMode, nodeLocation: cfg.nodeLocation});
    var idx = model.idx;
    var lastStep = idx.lastStep;

    var state = {
      cursor: 0,
      tier: cfg.tier,
      pinned: {},
      playing: false,
      timer: null,
      layout: null,
      skeleton: null,
      skeletonKey: '',
      valueCache: {}
    };

    host.innerHTML =
      '<div class="vd-root">' +
        '<header class="vd-hd">' +
          '<div class="vd-hd-t">' +
            '<h2 class="vd-title">' + esc(cfg.title || (trace.meta && trace.meta.title) || '变量图') + '</h2>' +
            '<p class="vd-legend">节点 = 变量 · 边 = 一次计算 · 点变量或公式里的变量，三处一起亮</p>' +
          '</div>' +
          '<span class="vd-pos" data-vd="pos"></span>' +
        '</header>' +
        '<main class="vd-main">' +
          '<div class="vd-dagwrap"><svg class="vd-dag" role="img" aria-label="变量依赖图"></svg></div>' +
          '<aside class="vd-side">' +
            '<div class="vd-op"><span class="vd-op-k" data-vd="op"></span>' +
              '<span class="vd-op-id" data-vd="opid"></span>' +
              '<span class="vd-seg" role="group" aria-label="公式档位" data-vd="tiers"></span></div>' +
            '<div class="vd-stitle" data-vd="stitle"></div>' +
            '<div class="vd-fml" data-vd="fml"></div>' +
            '<div class="vd-io">' +
              '<div class="vd-io-g"><b>读</b><span data-vd="reads"></span></div>' +
              '<div class="vd-io-g"><b>写</b><span data-vd="writes"></span></div>' +
            '</div>' +
            '<div class="vd-vals" data-vd="vals"></div>' +
            '<p class="vd-narr" data-vd="narr"></p>' +
          '</aside>' +
        '</main>' +
        '<footer class="vd-ctl">' +
          '<button type="button" data-vd="prev" aria-label="上一步">← 上一步</button>' +
          '<button type="button" data-vd="play" aria-label="播放">▶ 播放</button>' +
          '<button type="button" data-vd="next" aria-label="下一步">下一步 →</button>' +
          '<input type="range" data-vd="scrub" min="0" max="' + lastStep + '" step="1" value="0" aria-label="时间轴游标">' +
          '<span class="vd-count" data-vd="count"></span>' +
        '</footer>' +
      '</div>';

    var root = host.querySelector('.vd-root');
    var q = function (name) { return root.querySelector('[data-vd="' + name + '"]'); };
    var svg = root.querySelector('.vd-dag');
    var dagwrap = root.querySelector('.vd-dagwrap');

    function targetAspect() {
      var r = dagwrap.getBoundingClientRect();
      if (!r.width || !r.height) return 1;
      return r.width / r.height;
    }

    var sceneKey = '';
    function ensureScene() {
      /* The drawing is rebuilt only when the geometry changes (a different
         trace, or a resize that flips the orientation). Stepping the cursor
         never rebuilds it -- it toggles classes -- so a node's identity is
         stable across a cursor move and nothing on screen flickers. */
      var aspect = targetAspect();
      var L = computePositions(model, aspect, null, cfg.geo);
      var key = Math.round(L.w) + 'x' + Math.round(L.h) + ':' + L.nBands +
                ':' + model.nodes.length;
      if (key === sceneKey) { state.layout = L; return L; }
      sceneKey = key;
      state.layout = L;
      svg.setAttribute('viewBox', '0 0 ' + L.w + ' ' + L.h);
      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      svg.innerHTML = scene(L, model);
      svg.dataset.vdW = String(L.w);
      svg.dataset.vdH = String(L.h);
      svg.dataset.vdOrientation = L.vertical ? 'vertical' : 'horizontal';
      bindScene();
      /* A layered drawing scaled down far enough stops being a drawing: at
         ~0.35 the 13px node labels come out under 5px and the reader has a
         texture, not a graph. Measured, not guessed -- L04's twelve-band graph
         letterboxed into its panel lands there with the defaults. Below
         `MIN_SCALE` the drawing is NOT shrunk further: it is rendered at the
         minimum and the panel pans, which costs a scroll and keeps the labels
         readable. Above it nothing changes, which is why L00 never sees this
         path: its six-band graph never comes close. */
      fitScale(L);
      return L;
    }

    function fitScale(L) {
      var r = dagwrap.getBoundingClientRect();
      if (!r.width || !r.height) return;
      var s = Math.min(r.width / L.w, r.height / L.h);
      if (s >= cfg.geo.MIN_SCALE) {
        svg.style.width = '';
        svg.style.height = '';
        dagwrap.dataset.vdFit = 'contain';
        return;
      }
      var w = L.w * cfg.geo.MIN_SCALE, h = L.h * cfg.geo.MIN_SCALE;
      svg.style.width = w + 'px';
      svg.style.height = h + 'px';
      dagwrap.dataset.vdFit = 'pan';
      /* The cursor's own node is what a reader is looking at, so a panned
         drawing is scrolled to it rather than to the middle. */
      var hot = stepSet(state.cursor);
      var first = model.nodes.filter(function (n) { return hot[n]; })[0];
      var box = first && L.pos[first];
      if (box) {
        var cx = (box.x + cfg.geo.NW / 2) * cfg.geo.MIN_SCALE;
        var cy = (box.y + cfg.geo.NH / 2) * cfg.geo.MIN_SCALE;
        dagwrap.scrollTop = Math.max(0, Math.min(h - r.height, cy - r.height / 2));
        dagwrap.scrollLeft = Math.max(0, Math.min(w - r.width, cx - r.width / 2));
      }
    }

    function stepSet(i) {
      var s = model.steps[i];
      var set = {};
      s.reads.concat(s.writes).forEach(function (n) { set[n] = true; });
      return set;
    }

    function pastSet(upto) {
      var set = {};
      for (var k = 0; k < upto; k++) {
        model.steps[k].reads.concat(model.steps[k].writes).forEach(function (n) { set[n] = true; });
      }
      return set;
    }

    /* Three node states, and the reason there are three.
     *
     * A two-state drawing ("this step touches it / it does not") puts every
     * untouched node at the same faint weight, so a reader who has walked five
     * steps sees a picture that looks exactly like the one at step zero and
     * concludes that nothing has happened yet. The third state -- visited -- is
     * what makes progress visible, and it is painted the same way the timeline's
     * chips are, so "where am I" reads the same in both places. */
    function paintNodes(i) {
      var hot = stepSet(i);
      var past = pastSet(i);
      Array.prototype.forEach.call(svg.querySelectorAll('[data-vd-node]'), function (g) {
        var id = g.dataset.vdNode;
        var st = hot[id] ? 'on' : (past[id] ? 'past' : 'off');
        g.dataset.vdState = st;
        /* SVG elements have no writable `className` -- it is a read-only
           SVGAnimatedString -- so the class is set as an attribute. */
        setClass(g.querySelector('.vd-box'), 'vd-box vd-box-' + st);
        setClass(g.querySelector('.vd-nl'), 'vd-nl vd-nl-' + st);
        setClass(g.querySelector('.vd-ns'), 'vd-ns vd-ns-' + st);
        setClass(g.querySelector('.vd-nshape'), 'vd-nshape vd-nshape-' + st);
        g.classList.toggle('vd-n-pin', !!state.pinned[id]);
        g.setAttribute('aria-pressed', hot[id] ? 'true' : 'false');
      });
    }

    function paintEdges(i) {
      Array.prototype.forEach.call(svg.querySelectorAll('[data-vd-edge]'), function (p) {
        var pair = p.dataset.vdEdge.split('>');
        var e = null;
        for (var k = 0; k < model.edges.length; k++) {
          if (model.edges[k].from === pair[0] && model.edges[k].to === pair[1]) { e = model.edges[k]; break; }
        }
        var st = 'off';
        if (e) {
          if (e.steps.indexOf(i) !== -1) st = 'on';
          else if (e.steps.some(function (s) { return s < i; })) st = 'done';
        }
        setClass(p, 'vd-e vd-e-' + st);
        p.setAttribute('marker-end', st === 'on' ? 'url(#vd-ar-on)' : 'url(#vd-ar)');
        var lbl = svg.querySelector('[data-vd-el="' + pair[0] + '>' + pair[1] + '"]');
        if (lbl) setClass(lbl, 'vd-el vd-el-' + (st === 'off' ? 'off' : (st === 'on' ? 'on' : 'past')));
      });

      Array.prototype.forEach.call(svg.querySelectorAll('[data-vd-loop]'), function (p) {
        var id = p.dataset.vdLoop;
        var lp = null;
        for (var k = 0; k < model.loops.length; k++) {
          if (model.loops[k].id === id) { lp = model.loops[k]; break; }
        }
        var st = 'off';
        if (lp) {
          if (lp.steps.indexOf(i) !== -1) st = 'on';
          else if (lp.steps.some(function (s) { return s < i; })) st = 'done';
        }
        setClass(p, 'vd-e vd-e-loop vd-e-' + st);
        p.setAttribute('marker-end', st === 'on' ? 'url(#vd-ar-on)' : 'url(#vd-ar)');
        var lbl = svg.querySelector('[data-vd-loop-l="' + id + '"]');
        if (lbl) setClass(lbl, 'vd-el vd-el-loop vd-el-' + (st === 'off' ? 'off' : (st === 'on' ? 'on' : 'past')));
      });
    }

    /* --- the formula sidebar, rendered by the engine */
    function paintFormula(step) {
      var box = q('fml');
      var key = step.id + '|' + state.tier;
      if (state.skeletonKey !== key) {
        state.skeleton = NS.formula.buildSkeleton(step, state.tier, 'vd' + step.i + '-', state.valueCache);
        box.innerHTML = '';
        box.appendChild(state.skeleton.node);
        state.skeletonKey = key;
      }
      NS.formula.patch(state.skeleton, step, state.tier, state.valueCache);

      /* A formula wider than its box scrolls (see the stylesheet for why not
         wrap), and a scroll container whose scrollbar is invisible until you
         touch it reads as a clipped formula rather than a scrollable one. The
         fade is the affordance; it is applied only when there IS something to
         scroll to, so it never appears on a formula that already fits. */
      var node = box.querySelector('.lab-formula-node');
      box.classList.toggle('vd-fml-more',
        !!node && node.scrollWidth - node.clientWidth > 1);
    }

    function chips(list, model) {
      if (!list.length) return '<span class="vd-chip vd-chip-none">—</span>';
      return list.map(function (n) {
        var spec = model.tensors[n] || {};
        return '<button type="button" class="vd-chip" data-vd-var="' + esc(n) + '">' +
          esc(n) + ' <b>' + esc(shapeText(spec.shape)) + '</b></button>';
      }).join('');
    }

    function paintSide(i, step, snapshot) {
      q('op').textContent = (step.op || step.kind || '').toUpperCase();
      q('opid').textContent = step.id;
      q('stitle').textContent = step.title;
      q('reads').innerHTML = chips(step.reads, model);
      q('writes').innerHTML = chips(step.writes, model);

      /* The step's own outputs, plus any input still showing its `init`. That
         second set is not decoration: `x` is read by every load step and never
         written by any, so without it the input's value would never appear
         anywhere in the lab -- and "this page shows what the algorithm is
         actually holding" is the promise the whole exercise rests on. */
      var gridNames = step.writes.slice();
      step.reads.forEach(function (n) {
        if (gridNames.indexOf(n) !== -1) return;
        var entry = snapshot[n];
        if (entry && entry.source === 'init') gridNames.push(n);
      });
      q('vals').innerHTML = gridNames.map(function (n) {
        return valueGrid(n, model, snapshot, {limits: cfg.valueLimits});
      }).join('');
      q('narr').textContent = step.narration;
      paintFormula(step);
      paintPins();
    }

    /* --- bidirectional highlight
     *
     * Clicking a variable anywhere -- a graph node, a read/write chip, or the
     * slot that variable fills in the formula -- pins it, and every other place
     * the same variable appears lights up too. The mapping from a formula slot
     * to a variable is `step.binding_vars`, written by the trace generator; it
     * is not guessed here, because the guesses are all wrong: `x` occurs inside
     * `x_blk`, and the tensor `l` is written `\ell` in the formula, which has no
     * identifier `l` in it anywhere.
     */
    function slotsOf(step) {
      var out = [];
      Array.prototype.forEach.call(q('fml').querySelectorAll('[id]'), function (el) {
        var m = /-slot-([A-Za-z0-9_]+)$/.exec(el.id);
        if (m) out.push({ name: m[1], el: el });
      });
      return out;
    }

    function paintPins() {
      var step = model.steps[state.cursor];
      var bvars = step.bindingVars || {};
      slotsOf(step).forEach(function (s) {
        var vars = bvars[s.name] || [];
        var lit = vars.some(function (v) { return !!state.pinned[v]; });
        s.el.classList.toggle('vd-slot-on', lit);
        s.el.dataset.vdVars = vars.join(' ');
      });
      Array.prototype.forEach.call(root.querySelectorAll('[data-vd-var]'), function (el) {
        el.classList.toggle('vd-chip-on', !!state.pinned[el.dataset.vdVar]);
      });
    }

    function togglePin(name) {
      if (state.pinned[name]) delete state.pinned[name];
      else state.pinned[name] = true;
      paintNodes(state.cursor);
      paintPins();
    }

    function bindScene() {
      Array.prototype.forEach.call(svg.querySelectorAll('[data-vd-node]'), function (g) {
        g.addEventListener('click', function (ev) { ev.stopPropagation(); togglePin(g.dataset.vdNode); });
        g.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') {
            ev.preventDefault();
            togglePin(g.dataset.vdNode);
          }
        });
      });
      /* Clicking a formula slot pins every variable that slot mentions: a slot
         like \sum_i \exp(x^{(j)}_i - m^{(j)}) names two of them, and lighting
         only one would be a lie about what the formula reads. */
      slotsOf(model.steps[state.cursor]).forEach(function (s) {
        s.el.classList.add('vd-slot');
      });
      q('fml').addEventListener('click', function (ev) {
        var el = ev.target.closest ? ev.target.closest('[id*="-slot-"]') : null;
        if (!el) return;
        ev.stopPropagation();
        var m = /-slot-([A-Za-z0-9_]+)$/.exec(el.id);
        var vars = (model.steps[state.cursor].bindingVars || {})[m[1]] || [];
        var anyOn = vars.some(function (v) { return !!state.pinned[v]; });
        vars.forEach(function (v) {
          if (anyOn) delete state.pinned[v];
          else state.pinned[v] = true;
        });
        paintNodes(state.cursor);
        paintPins();
      });
    }

    root.addEventListener('click', function (ev) {
      var chip = ev.target.closest ? ev.target.closest('[data-vd-var]') : null;
      if (chip) { togglePin(chip.dataset.vdVar); ev.stopPropagation(); return; }
    });

    /* --- one render path, shared by the buttons, the scrubber, the keyboard
     * and a deep link. There is no "+1 fast path" and no undo path for -1: a
     * second path is a second chance to disagree with the first. */
    function paint() {
      var i = state.cursor;
      var step = model.steps[i];
      var snapshot = TM.resolve(idx, i);

      ensureScene();
      paintNodes(i);
      paintEdges(i);
      state.skeletonKey = '';           /* the sidebar is rebuilt per step */
      paintSide(i, step, snapshot);

      q('pos').textContent = (i + 1) + ' / ' + (lastStep + 1);
      q('count').textContent = step.id + ' · ' + (step.op || step.kind || '');
      q('prev').disabled = i === 0;
      q('next').disabled = i === lastStep;
      q('scrub').value = String(i);
      root.querySelectorAll('[data-vd-tier]').forEach(function (b) {
        b.classList.toggle('vd-on', b.dataset.vdTier === state.tier);
        b.setAttribute('aria-pressed', b.dataset.vdTier === state.tier ? 'true' : 'false');
      });

      try {
        var url = new URL(global.location.href);
        url.searchParams.set(cfg.param, String(i));
        global.history.replaceState(null, '', url);
      } catch (err) { /* file:// or a blocked history API: the view still works */ }

      /* The callback context matches the one `engine/player.js` hands an extra
         panel's `render(step, ctx)`: same field names, same meanings. A panel
         written for the player therefore runs here unchanged. */
      if (cfg.onStep) {
        cfg.onStep(i, step, {
          cursor: i,
          snapshot: snapshot,
          before: i > 0 ? TM.resolve(idx, i - 1) : null,
          model: model,
          /* The panels L04's page mounts read `ctx.trace` (the engine player
             hands them one); the view has it and handing it on costs nothing.
             Without it a panel written for the player silently renders an
             empty block here -- which is a failure mode with no error in it. */
          trace: trace
        });
      }
    }

    function setCursor(i) {
      var next = Math.max(0, Math.min(lastStep, i));
      if (next === state.cursor && state.skeleton) return;
      state.cursor = next;
      paint();
    }

    function setTier(tier) {
      if (NS.formula.TIERS.indexOf(tier) === -1 || tier === state.tier) return;
      state.tier = tier;
      state.skeletonKey = '';
      paintFormula(model.steps[state.cursor]);
      paint();
    }

    function stop() {
      if (state.timer) global.clearInterval(state.timer);
      state.timer = null;
      state.playing = false;
      q('play').textContent = '▶ 播放';
      q('play').classList.remove('vd-on');
      q('play').setAttribute('aria-label', '播放');
    }

    function play() {
      if (state.cursor >= lastStep) setCursor(0);
      state.playing = true;
      q('play').textContent = '❚❚ 暂停';
      q('play').classList.add('vd-on');
      q('play').setAttribute('aria-label', '暂停');
      state.timer = global.setInterval(function () {
        if (state.cursor >= lastStep) { stop(); return; }
        setCursor(state.cursor + 1);
      }, 900);
    }

    function toggle() { if (state.playing) stop(); else play(); }

    function onKey(ev) {
      var t = ev.target;
      var typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
                         t.tagName === 'SELECT' || t.isContentEditable);
      if (typing && ev.key !== 'Escape') return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      switch (ev.key) {
        case ' ':
        case 'Spacebar':
          ev.preventDefault();       /* or the page scrolls on every toggle */
          toggle();
          break;
        case 'ArrowRight': ev.preventDefault(); stop(); setCursor(state.cursor + 1); break;
        case 'ArrowLeft': ev.preventDefault(); stop(); setCursor(state.cursor - 1); break;
        case 'Home': ev.preventDefault(); stop(); setCursor(0); break;
        case 'End': ev.preventDefault(); stop(); setCursor(lastStep); break;
        default: break;
      }
    }

    function onResize() { ensureScene(); }
    var resizeTimer = null;
    function onResizeDebounced() {
      if (resizeTimer) global.clearTimeout(resizeTimer);
      resizeTimer = global.setTimeout(onResize, 120);
    }

    /* --- controls */
    NS.formula.TIERS.forEach(function (t) {
      var b = document.createElement('button');
      b.type = 'button';
      b.dataset.vdTier = t;
      b.textContent = { sym: '符号', idx: '索引', num: '数值' }[t];
      b.setAttribute('aria-pressed', 'false');
      b.className = 'vd-tier' + (t === state.tier ? ' vd-on' : '');
      q('tiers').appendChild(b);
    });
    q('tiers').addEventListener('click', function (ev) {
      var b = ev.target.closest ? ev.target.closest('[data-vd-tier]') : null;
      if (b) setTier(b.dataset.vdTier);
    });
    q('prev').addEventListener('click', function () { stop(); setCursor(state.cursor - 1); });
    q('next').addEventListener('click', function () { stop(); setCursor(state.cursor + 1); });
    q('play').addEventListener('click', toggle);
    q('scrub').addEventListener('input', function (ev) { stop(); setCursor(Number(ev.target.value)); });
    document.addEventListener('keydown', onKey);
    global.addEventListener('resize', onResizeDebounced);

    state.cursor = Math.max(0, Math.min(lastStep, readStepFromUrl(cfg.param, lastStep)));
    paint();

    var api = {
      index: idx,
      model: model,
      state: state,
      root: root,
      svg: svg,
      layout: function () { return state.layout; },
      nodeRect: function (id) { return nodeRect(state.layout, id); },
      setCursor: setCursor,
      setTier: setTier,
      play: play,
      stop: stop,
      toggle: toggle,
      next: function () { stop(); setCursor(state.cursor + 1); },
      prev: function () { stop(); setCursor(state.cursor - 1); },
      paint: paint,
      togglePin: togglePin,
      lint: function () { return lint(trace); },
      destroy: function () {
        stop();
        document.removeEventListener('keydown', onKey);
        global.removeEventListener('resize', onResizeDebounced);
      }
    };
    host.variableDag = api;
    return api;
  }

  function readStepFromUrl(param, lastStep) {
    var raw = null;
    try { raw = new URLSearchParams(global.location.search).get(param); } catch (e) { return 0; }
    if (raw === null || raw === '') return 0;
    var n = parseInt(raw, 10);
    if (isNaN(n)) return 0;
    return Math.max(0, Math.min(lastStep, n));
  }

  NS.variableDag = {
    mount: mount,
    build: build,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES,
    computeLayers: computeLayers,
    computePositions: computePositions,
    railPlan: railPlan,
    edgePath: edgePath,
    directPath: directPath,
    railPath: railPath,
    loopPath: loopPath,
    nodeRect: nodeRect,
    shapeGlyph: shapeGlyph,
    shapeText: shapeText,
    formatValue: formatValue,
    VALUE_LIMITS: VALUE_LIMITS,
    GEO: GEO
  };
})(typeof window !== 'undefined' ? window : globalThis);
