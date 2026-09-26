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
     rather than a reflowed one. */
  var NW = 116, NH = 46, GAPX = 112, GAPY = 74, PAD = 26;
  var LOOP_W = 34, LOOP_GAP = 18;

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

  function build(trace) {
    var idx = TM.index(trace);

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

    steps.forEach(function (s) {
      if (!s.writes.length) return;

      /* The variable this step produces: the write that sits furthest along,
         which is the step's output rather than one of its inputs. */
      var target = s.writes[0];
      s.writes.forEach(function (w) {
        if ((depth[w] || 0) > (depth[target] || 0)) target = w;
      });

      /* A step that reads what it writes is carrying a value across iterations,
         not moving data between two variables. That is a self-loop, drawn as
         one, once per variable, however many steps carry it. The recurrence
         `m <- max(m, .)` IS the algorithm's shape, so it has to be visible
         rather than discarded as a degenerate edge. */
      if (s.reads.indexOf(target) !== -1) {
        if (!selfLoops[target]) {
          selfLoops[target] = { id: target, steps: [], ops: [] };
          loops.push(selfLoops[target]);
        }
        selfLoops[target].steps.push(s.i);
        if (selfLoops[target].ops.indexOf(s.op) === -1) selfLoops[target].ops.push(s.op);
      }

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
      if (!source) return;

      var key = source + '' + target;
      if (!edgeMap[key]) {
        edgeMap[key] = { from: source, to: target, steps: [], ops: [] };
        edges.push(edgeMap[key]);
      }
      edgeMap[key].steps.push(s.i);
      if (edgeMap[key].ops.indexOf(s.op) === -1) edgeMap[key].ops.push(s.op);
    });

    return {
      nodes: nodes,
      tensors: trace.tensors || {},
      steps: steps,
      edges: edges,
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

  function computePositions(model, targetAspect, vertical) {
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
    var right = model.loops.length ? LOOP_GAP + LOOP_W + PAD : PAD;
    return {
      pos: pos,
      vertical: vertical,
      bands: L.bands,
      nBands: L.nBands,
      w: maxX + NW + right,
      h: maxY + NH + PAD
    };
  }

  /* ============================================================ geometry
   *
   * One place where a path between two nodes is turned into a curve, so the
   * drawing code and the harness's geometry predicates agree about where an
   * edge actually is.
   */

  function edgePath(L, from, to) {
    var a = L.pos[from], b = L.pos[to];
    if (!a || !b) return null;
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
    return { d: d, lx: lx, ly: ly, x1: x1, y1: y1, x2: x2, y2: y2 };
  }

  /* A self-loop leaves the node's right edge and comes back to it. It is drawn
     as a bezier outside the box so it never crosses the node's own label. */
  function loopPath(L, id) {
    var p = L.pos[id];
    if (!p) return null;
    var x0 = p.x + NW, yTop = p.y + NH * 0.26, yBot = p.y + NH * 0.74;
    var reach = x0 + LOOP_GAP + LOOP_W;
    var d = 'M' + x0 + ' ' + yTop +
            ' C' + reach + ' ' + yTop + ' ' + reach + ' ' + yBot + ' ' + x0 + ' ' + yBot;
    return { d: d, lx: reach - 2, ly: (yTop + yBot) / 2 };
  }

  /* The rectangle a node occupies, in the same coordinate space the SVG draws
     in. Exported so the harness can assert against the geometry instead of
     re-deriving it. */
  function nodeRect(L, id) {
    var p = L.pos[id];
    return p ? { x: p.x, y: p.y, w: NW, h: NH } : null;
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

  function valueGrid(name, model, snapshot) {
    var spec = model.tensors[name] || {};
    var entry = snapshot[name];
    if (!entry || entry.value === undefined) return '';
    var val = entry.value;
    var shape = spec.shape || [];
    var body;

    if (!Array.isArray(val)) {
      body = '<span class="vd-cell vd-cell-1">' + esc(formatValue(val)) + '</span>';
    } else if (shape.length <= 1 || !Array.isArray(val[0])) {
      body = '<div class="vd-vrow">' + val.map(function (v) {
        return '<span class="vd-cell">' + esc(formatValue(v)) + '</span>';
      }).join('') + '</div>';
    } else {
      body = val.map(function (row) {
        return '<div class="vd-vrow">' + row.map(function (v) {
          return '<span class="vd-cell">' + esc(formatValue(v)) + '</span>';
        }).join('') + '</div>';
      }).join('');
    }

    var where = entry.from === -1 ? '初值'
              : (entry.from === null ? '未产生' : '第 ' + (entry.from + 1) + ' 步写入');
    return '<div class="vd-vg"><div class="vd-vg-hd"><span class="vd-vg-n">' + esc(name) +
      '</span><span class="vd-vg-s">' + esc(shapeText(shape)) + '</span>' +
      '<span class="vd-vg-at">' + esc(where) + '</span></div>' + body + '</div>';
  }

  /* ============================================================ SVG scene */

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
      out += '<path class="vd-e vd-e-off" data-vd-edge="' + esc(e.from + '>' + e.to) +
        '" d="' + g.d + '" marker-end="url(#vd-ar)"/>';
      if (e.ops.length === 1) {
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
      out += '<g class="vd-n ' + cls(id) + '" data-vd-node="' + esc(id) + '" tabindex="0" role="button" ' +
        'aria-label="变量 ' + esc(id) + '">' +
        '<rect class="vd-box vd-box-off" x="' + p.x + '" y="' + p.y + '" width="' + NW +
          '" height="' + NH + '" rx="10"/>' +
        '<text class="vd-nl vd-nl-off" x="' + (p.x + 11) + '" y="' + (p.y + 20) + '">' + esc(id) + '</text>' +
        '<text class="vd-ns vd-ns-off" x="' + (p.x + 11) + '" y="' + (p.y + 35) + '">' +
          esc(shapeText(spec.shape)) + (spec.at ? ' · ' + esc(spec.at) : '') + '</text>' +
        '</g>';
    });

    return out;
  }

  /* ============================================================ mount */

  var DEFAULTS = {
    title: '',
    lead: '',
    tier: 'num',
    param: 'step',
    digits: 4,
    onStep: null
  };

  function mount(host, trace, opts) {
    var cfg = {};
    Object.keys(DEFAULTS).forEach(function (k) { cfg[k] = DEFAULTS[k]; });
    Object.keys(opts || {}).forEach(function (k) { cfg[k] = opts[k]; });

    if (!host) throw new Error('variableDag: no mount element');
    if (!NS.formula) throw new Error('variableDag: engine/formula.js must be loaded first');

    var model = build(trace);
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
      var L = computePositions(model, aspect);
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
      return L;
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
        return valueGrid(n, model, snapshot);
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
          model: model
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
    computeLayers: computeLayers,
    computePositions: computePositions,
    edgePath: edgePath,
    loopPath: loopPath,
    nodeRect: nodeRect,
    formatValue: formatValue,
    GEO: { NW: NW, NH: NH, GAPX: GAPX, GAPY: GAPY, PAD: PAD, LOOP_W: LOOP_W, LOOP_GAP: LOOP_GAP }
  };
})(typeof window !== 'undefined' ? window : globalThis);
