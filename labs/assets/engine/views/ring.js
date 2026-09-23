/* L00 engine · view: ring topology (环形拓扑 + 缓冲区状态 + 通信量累计).
 *
 * Four things kept in sync on one picture, because for this algorithm they are
 * one picture and separating them is what makes ring AllReduce hard to read:
 *
 *   1. the ring itself — N ranks on a circle, each with one outbound link, and
 *      the links that are carrying data THIS step lit up;
 *   2. each rank's buffer — one cell per chunk, showing how many ranks'
 *      contributions each chunk currently holds (the quantity that goes from 1
 *      to N, and the reason ReduceScatter ends with "one finished chunk each");
 *   3. the step timeline — which of the 2(N-1) steps the cursor is on, and
 *      which phase it belongs to;
 *   4. the traffic accumulator — bytes sent per rank, accumulated over
 *      steps[0..cursor], against the naive hub's curve on the same axes.
 *
 * WHAT IS DERIVED AND WHAT IS DECLARED
 *
 * `step.transfers` is DECLARED — what crossed which link, carrying which chunk,
 * in which mode. It cannot be derived: the same shape of transfer means `+=` in
 * ReduceScatter and `=` in AllGather, and a component that guessed from the
 * phase would be re-deriving the schedule it is supposed to be displaying. See
 * labs/traces/ring_allreduce.py.
 *
 * Everything drawn is then DERIVED from those declarations by a fold over steps
 * 0..cursor, exactly as the gantt view does and for exactly the same reason:
 * there is no "advance" path and no accumulated DOM state, so jumping to step 11
 * rebuilds the picture from the trace and cannot disagree with having played
 * there. The per-chunk counts, the per-rank cumulative bytes, and the list of
 * lit links are all folds; none of them is a running total held between frames.
 *
 * The alternative — a stateful accumulator that applies each step as the cursor
 * advances — is what the control group in scripts/verify-ring.py implements, and
 * it is wrong on a backwards jump. The fold is what makes that impossible.
 *
 * WHY THE BUFFER CELLS ARE COUNTS AND NOT NUMBERS
 *
 * A chunk's actual values are in the trace (`buf_<r>`) and the tensor inspector
 * already shows them. What the reader needs here is the state of the REDUCTION:
 * a cell reading "3/N" says how close that chunk is to done, which is the thing
 * that changes across the 2(N-1) steps, and a cell holding four floats says
 * nothing at a glance about whether three ranks or one have contributed. The
 * full values are one panel away; the count is the picture.
 *
 * CONFIG
 *
 *   size:      number   px across the topology square (default 260)
 *   showTree:  boolean  draw the tree control topology beside the ring
 *                       (default true; L16/L17 reuse this view without it)
 *
 * render(cursor, ctx) returns an HTML string, called by the player on every
 * paint with the same (trace, i) it just resolved, so this can never show a
 * step the rest of the page is not on.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = NS.formula.escapeText;
  var escAttr = NS.formula.escapeAttr;

  var MODE_ADD = 'add';
  var MODE_COPY = 'copy';

  /* Bytes per element. Read from the trace's own config rather than assumed:
   * the trace states it (`meta.config.item_bytes`), the arithmetic on the page
   * is that number times an element count, and hard-coding 4 here would make
   * the one place the two could disagree the one place nobody looks. */
  function itemBytes(trace) {
    var cfg = (trace && trace.meta && trace.meta.config) || {};
    return cfg.item_bytes || 4;
  }

  /* =============================================================== the folds
   *
   * Three folds over the same declarative transfer list. They are separate
   * functions rather than one pass because they answer unrelated questions and
   * the acceptance harness checks each against the generator's own account of
   * it — a combined pass would make "the counts are right" and "the bytes are
   * right" one assertion, and they fail for different reasons.
   */

  /**
   * Per-rank, per-chunk counts: how many ranks' data is in that chunk now.
   *
   * Seeded at 1 everywhere, because a rank holds its own contribution to every
   * chunk before any communication happens. `add` sums the source's count in;
   * `copy` REPLACES it. Those two rules are the whole difference between
   * ReduceScatter and AllGather, and they are read off the transfer's declared
   * mode rather than inferred from the phase.
   */
  function countsAfter(trace, cursor, ranks, chunks) {
    // `ranks` is the id array, not the id->index map: the map is a plain
    // object and has no `length`, and an earlier draft read `.length` off it,
    // built a zero-row grid, and then indexed into it. The array is the shape
    // this needs and the map is derived from it below.
    var n = ranks.length;
    var rankIndex = {};
    ranks.forEach(function (rid, i) { rankIndex[rid] = i; });
    var counts = [];
    var r, c;
    for (r = 0; r < n; r++) {
      counts.push([]);
      for (c = 0; c < chunks; c++) counts[r].push(1);
    }
    var last = Math.min(cursor, trace.steps.length - 1);
    for (var i = 0; i <= last; i++) {
      var xfers = trace.steps[i].transfers || [];
      for (var j = 0; j < xfers.length; j++) {
        var x = xfers[j];
        var from = rankIndex[x.from], to = rankIndex[x.to];
        if (from === undefined || to === undefined) continue;
        if (x.chunk < 0 || x.chunk >= chunks) continue;
        counts[to][x.chunk] = (x.mode === MODE_COPY)
          ? counts[from][x.chunk]
          : counts[to][x.chunk] + counts[from][x.chunk];
      }
    }
    return counts;
  }

  /**
   * Per-rank bytes sent, accumulated over steps 0..cursor.
   *
   * One entry per step (the running totals after that step), which is what the
   * accumulator draws. Folding rather than accumulating is what makes a
   * backwards jump correct; the harness compares this against the generator's
   * `curves.ring.per_rank_cumulative` at every cursor.
   */
  function bytesSeries(trace) {
    var n = (trace.ring.ranks || []).length;
    var rankIndex = rankIndexOf(trace);
    var running = [];
    var r;
    for (r = 0; r < n; r++) running.push(0);
    var series = [];
    for (var i = 0; i < trace.steps.length; i++) {
      var xfers = trace.steps[i].transfers || [];
      for (var j = 0; j < xfers.length; j++) {
        var x = xfers[j];
        var from = rankIndex[x.from];
        if (from === undefined) continue;
        running[from] += (x.elements || 0) * itemBytes(trace);
      }
      series.push(running.slice());
    }
    return series;
  }

  /** Which links are carrying data during step `i`, as "from>to" keys. */
  function activeLinks(step) {
    var set = {};
    (step.transfers || []).forEach(function (x) { set[x.from + '>' + x.to] = x; });
    return set;
  }

  /* ============================================================ comparisons */

  /**
   * A control schedule's cumulative curve, from the trace's own numbers.
   *
   * `busiest_cumulative[i]` is the load on that schedule's busiest link after
   * its step `i`, which the generator folded out of the same transfer list.
   * Nothing here is synthesized — an earlier draft padded the hub's curve to
   * the ring's step count with a straight line, which is a drawing of nothing
   * (the hub has exactly 2(N-1) steps too, so the padding was dead code that
   * only would have hidden a real mismatch had there been one).
   *
   * A schedule shorter than the cursor holds its last value, which is the
   * honest reading of "it finished at that step" — the same rule the gantt
   * comparison uses for a finished arm.
   */
  function controlCurve(entry) {
    var c = entry && entry.curves;
    if (!c || !c.busiest_cumulative) return null;
    return {
      label: entry.label || '',
      note: entry.note || '',
      steps: c.steps,
      series: c.busiest_cumulative,
      total: c.link_load_total,
      links: c.links,
      aggregate: c.total
    };
  }

  function curveAt(curve, at) {
    if (!curve.series.length) return 0;
    return curve.series[Math.min(at, curve.series.length - 1)];
  }

  /* ================================================================= render */

  function rankIndexOf(trace) {
    var out = {};
    (trace.ring.ranks || []).forEach(function (r, i) { out[r.id] = i; });
    return out;
  }

  function rankIdsOf(trace) {
    return (trace.ring.ranks || []).map(function (r) { return r.id; });
  }

  /* The gap between the outermost strip and the canvas edge, so nothing is
   * ever flush against the viewBox. */
  var PAD = 6;

  /**
   * The topology's geometry, chosen so the picture satisfies an INVARIANT
   * rather than to a hand-tuned constant.
   *
   * The invariant: no rank's buffer strip may intersect any other rank's node
   * ring, and no two strips may intersect each other. That is the property the
   * acceptance harness measures off the rendered DOM, and the first version of
   * this view violated it in two ways that no count could ever catch — at N=8
   * the strips grazed their neighbours' rings, and the leftmost one ran past
   * the viewBox and was cut off — while every number on the page stayed
   * correct. A picture can be wrong in a way only a picture shows.
   *
   * So `R` is SEARCHED, not chosen: start at a minimum that looks like a ring
   * and grow until the invariant holds. That way changing the cell size, the
   * chunk count or the rank count cannot silently reintroduce the overlap —
   * the search absorbs it, and the harness's geometric check fails loudly if it
   * ever cannot.
   *
   * Why growing R helps: adjacent ranks sit `2R·sin(pi/n)` apart, so a wider
   * ring is a wider gap for the strips to fit between. The strips themselves
   * shrink at N=8 (`tight`), because eight cells at the four-rank size are
   * wider than the gap between two ranks can ever be.
   */
  function computeLayout(n, chunks) {
    var tight = n >= 8;
    var cellW = tight ? 11 : 15;
    var cellH = tight ? 12 : 13;
    var gap = tight ? 1 : 2;
    var nodeR = tight ? 13 : 15;
    var ringR = nodeR + 3;
    var stripOff = nodeR + 9;
    var stripLen = chunks * (cellW + gap) - gap;
    var minR = n === 2 ? 74 : n === 4 ? 92 : 100;
    var R = minR;
    // Bounded so a miscomputed cell size cannot spin forever; the harness's
    // geometric check is what catches a layout that never becomes clear.
    for (var guard = 0; guard < 200 && !clear(R); guard++) R += 2;

    function stripsAt(radius) {
      var out = [];
      for (var i = 0; i < n; i++) {
        var ang = -Math.PI / 2 + (2 * Math.PI * i) / n;
        var x = radius * Math.cos(ang), y = radius * Math.sin(ang);
        var upper = y <= 0;
        var top = upper ? y - stripOff - cellH : y + stripOff;
        out.push({ i: i, x: x, y: y, upper: upper,
                   x0: x - stripLen / 2, x1: x + stripLen / 2,
                   y0: top, y1: top + cellH });
      }
      return out;
    }

    /** Rect (x0,y0)-(x1,y1) vs circle (cx,cy,rad). */
    function hitsCircle(r, cx2, cy2, rad) {
      var nx = Math.max(r.x0, Math.min(cx2, r.x1));
      var ny = Math.max(r.y0, Math.min(cy2, r.y1));
      var dx = nx - cx2, dy = ny - cy2;
      return dx * dx + dy * dy < rad * rad;
    }

    function rectsHit(a, b) {
      return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
    }

    function clear(radius) {
      var st = stripsAt(radius);
      for (var i = 0; i < st.length; i++) {
        for (var j = 0; j < st.length; j++) {
          if (i === j) continue;
          if (hitsCircle(st[i], st[j].x, st[j].y, ringR)) return false;
          if (j > i && rectsHit(st[i], st[j])) return false;
        }
      }
      return true;
    }

    var halfW = R + stripLen / 2 + PAD;
    // The action caption sits above an upper node's strip (and below a lower
    // one's), so the tall ranks need room for the strip PLUS that line.
    var halfH = R + stripOff + cellH + PAD;
    return {
      R: R, cellW: cellW, cellH: cellH, gap: gap, nodeR: nodeR,
      stripOff: stripOff, stripLen: stripLen,
      W: Math.round(2 * halfW), H: Math.round(2 * halfH),
      cx: Math.round(2 * halfW) / 2, cy: Math.round(2 * halfH) / 2
    };
  }

  /**
   * Point on the circle for rank `i` of `n`.
   *
   * Takes the centre and radius explicitly rather than deriving them from a
   * square: the canvas is sized from its own content (see `computeLayout`), so
   * the caller is the only thing that knows where the circle actually is.
   */
  function ringPoint(i, n, circle) {
    // Start at the top and go clockwise, which is how the tutorials draw a
    // ring and the direction the data travels.
    var angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
    return {
      x: circle.cx + circle.r * Math.cos(angle),
      y: circle.cy + circle.r * Math.sin(angle)
    };
  }

  /**
   * The topology: N nodes on a circle, one directed arc per link, the links
   * carrying data this step drawn live and arrowed, the rest faint.
   *
   * Drawn as SVG rather than with positioned divs because the arcs have to
   * curve: a straight chord between antipodal ranks on an 8-rank ring passes
   * through the middle where the labels are, and at N=2 it would sit exactly
   * under them.
   */
  /**
   * The topology: N nodes on a circle, one directed arc per link, the links
   * carrying data this step drawn live and arrowed, the rest faint. Each node
   * carries its buffer as a horizontal strip of `chunks` cells, and a progress
   * ring showing how many of those chunks are fully reduced.
   *
   * TWO LAYOUT DECISIONS THAT LOOK LIKE DETAILS AND ARE NOT
   *
   * The buffer strips are always HORIZONTAL, even though they hang off a
   * circle, and they sit on the radial side of their node (above the node for
   * nodes in the top half, below for the bottom half). Both parts matter:
   *
   *   - Rotating a strip to follow the radius makes its digits rotate too, and
   *     a ring with eight ranks then has cells readable at eight angles. The
   *     table below the ring shows the same data upright; a picture whose
   *     numbers have to be read sideways is a picture nobody reads.
   *   - Placing the strip OUTWARD (never back across the circle) is what keeps
   *     it off the arcs and off its own node. The first version extended every
   *     strip to the right regardless of which side of the ring the node was
   *     on, so the left-hand ranks drew their buffers straight through
   *     themselves — visible immediately in a screenshot and invisible in every
   *     assertion, which is why the screenshots are part of the acceptance
   *     procedure rather than a nicety.
   *
   * The canvas is sized from the CONTENT (the strip is `chunks` cells wide and
   * grows with N) rather than fixed, because at N=8 the strips are three times
   * the width they are at N=2 and a fixed viewBox clipped them.
   *
   * Drawn as SVG rather than positioned divs because the arcs have to curve: a
   * straight chord between antipodal ranks on an 8-rank ring passes through the
   * middle where the labels are, and at N=2 it would sit exactly under them.
   */
  function topology(trace, step, cfg) {
    var n = trace.ring.ranks.length;
    var chunks = trace.ring.chunks;
    var counts = cfg.counts;
    var live = activeLinks(step);

    var L = computeLayout(n, chunks);
    var W = L.W, H = L.H, cx = L.cx, cy = L.cy, R = L.R;
    var cellW = L.cellW, cellH = L.cellH, gap = L.gap;
    var stripLen = L.stripLen, nodeR = L.nodeR, stripOff = L.stripOff;

    var out = [];
    out.push('<svg class="lab-ring-svg" viewBox="0 0 ' + W + ' ' + H +
             '" width="' + W + '" height="' + H +
             '" data-n="' + n + '" data-chunks="' + chunks + '" role="img">');

    // The arcs first, so the nodes paint over their endpoints.
    for (var i = 0; i < n; i++) {
      var a = ringPoint(i, n, { cx: cx, cy: cy, r: R });
      var b = ringPoint((i + 1) % n, n, { cx: cx, cy: cy, r: R });
      // Pull both ends in so the arc stops at the node's edge rather than under
      // its label.
      var dx = b.x - a.x, dy = b.y - a.y;
      var len = Math.sqrt(dx * dx + dy * dy) || 1;
      var ux = dx / len, uy = dy / len;
      var x1 = a.x + ux * (nodeR + 4), y1 = a.y + uy * (nodeR + 4);
      var x2 = b.x - ux * (nodeR + 12), y2 = b.y - uy * (nodeR + 12);
      // Bow the arc away from the centre. The normal's sign is chosen so the
      // push is OUTWARD — without that, a chord on one side of the circle bows
      // inward and crosses the label column in the middle.
      var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      var nx = -(y2 - y1), ny = (x2 - x1);
      var nl = Math.sqrt(nx * nx + ny * ny) || 1;
      nx /= nl; ny /= nl;
      if ((cx - mx) * nx + (cy - my) * ny > 0) { nx = -nx; ny = -ny; }
      var ctrlX = mx + nx * 16, ctrlY = my + ny * 16;
      var isLive = !!live[trace.ring.ranks[i].id + '>' +
                          trace.ring.ranks[(i + 1) % n].id];
      out.push('<path class="lab-ring-arc' + (isLive ? ' lab-ring-arc-live' : '') +
               '" d="M' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' Q' +
               ctrlX.toFixed(1) + ' ' + ctrlY.toFixed(1) + ' ' +
               x2.toFixed(1) + ' ' + y2.toFixed(1) + '"' +
               ' data-link="' + escAttr(trace.ring.ranks[i].id + '>' +
                                       trace.ring.ranks[(i + 1) % n].id) + '"' +
               ' marker-end="url(#lab-ring-arrow)"></path>');
    }

    // Markers, defined inside this svg so two rings on one page cannot collide
    // by id.
    out.push('<defs><marker id="lab-ring-arrow" viewBox="0 0 8 8" refX="7" refY="4"' +
             ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">' +
             '<path d="M0 0 L8 4 L0 8 z" class="lab-ring-arrowhead"></path>' +
             '</marker></defs>');

    var circumference = 2 * Math.PI * nodeR;
    for (var k = 0; k < n; k++) {
      var p = ringPoint(k, n, { cx: cx, cy: cy, r: R });
      var rid = trace.ring.ranks[k].id;
      var sending = live[rid + '>' + trace.ring.ranks[(k + 1) % n].id];
      var receiving = null;
      for (var key in live) {
        if (key.split('>')[1] === rid) { receiving = live[key]; break; }
      }

      // How many of this rank's chunks are fully reduced. The progress ring is
      // the one-node summary of the story: it reads 0/N through ReduceScatter
      // apart from the single chunk this rank owns, then fills up during
      // AllGather.
      var done = 0;
      for (var cc = 0; cc < chunks; cc++) if (counts[k][cc] === n) done++;
      var dash = (done / n) * circumference;

      out.push('<g class="lab-ring-node' +
               (sending ? ' lab-ring-node-send' : '') +
               (receiving ? ' lab-ring-node-recv' : '') +
               '" data-rank="' + escAttr(rid) + '" data-done="' + done + '">');
      out.push('<circle class="lab-ring-node-prog" cx="' + p.x.toFixed(1) +
               '" cy="' + p.y.toFixed(1) + '" r="' + (nodeR + 3) + '"' +
               ' stroke-dasharray="' + dash.toFixed(2) + ' ' +
               (circumference - dash).toFixed(2) + '"' +
               ' transform="rotate(-90 ' + p.x.toFixed(1) + ' ' +
               p.y.toFixed(1) + ')"></circle>');
      out.push('<circle class="lab-ring-node-disc" cx="' + p.x.toFixed(1) +
               '" cy="' + p.y.toFixed(1) + '" r="' + nodeR + '"></circle>');
      out.push('<text class="lab-ring-node-id" x="' + p.x.toFixed(1) + '" y="' +
               (p.y + 3.5).toFixed(1) + '">' + esc(rid) + '</text>');
      out.push('</g>');

      // The buffer strip, on the radial side of the node.
      var upper = p.y <= cy;
      var stripY = upper ? p.y - stripOff - cellH : p.y + stripOff;
      // No clamping: `computeLayout` sized the canvas so the outermost strip
      // fits with `PAD` to spare. An earlier version clamped instead, which
      // silently slid the left-hand strips off-centre — and at N=8 pushed one
      // of them past the viewBox edge entirely, where it was simply cut off.
      var bx = Math.round(p.x - stripLen / 2);

      out.push('<g class="lab-ring-buf" data-rank="' + escAttr(rid) + '">');
      out.push('<rect class="lab-ring-buf-bg" x="' + (bx - 3) + '" y="' +
               (stripY - 3) + '" width="' + (stripLen + 6) + '" height="' +
               (cellH + 6) + '" rx="4"></rect>');
      for (var c2 = 0; c2 < chunks; c2++) {
        var have = counts[k][c2];
        var full = have === n;
        var cls = 'lab-ring-cell' + (full ? ' lab-ring-cell-full'
          : have > 1 ? ' lab-ring-cell-part' : '');
        var title = 'rank ' + rid + ' · chunk ' + c2 + ' · ' + have + '/' + n +
                    (full ? '（本块已归约完成）' : '');
        out.push('<g class="' + cls + '" data-rank="' + escAttr(rid) +
                 '" data-chunk="' + c2 + '" data-count="' + have + '">' +
                 '<rect x="' + (bx + c2 * (cellW + gap)) + '" y="' + stripY +
                 '" width="' + cellW + '" height="' + cellH +
                 '" rx="2.5"><title>' + esc(title) + '</title></rect>' +
                 '<text x="' + (bx + c2 * (cellW + gap) + cellW / 2).toFixed(1) +
                 '" y="' + (stripY + cellH - 3.5).toFixed(1) + '">' + have +
                 '</text></g>');
      }
      out.push('</g>');

      /* The per-rank caption ("收↔发 · 累加") deliberately does NOT live in
       * the SVG. It was here first, and it cost more than it was worth: an
       * SVG text box's footprint depends on the font the browser actually
       * resolves, so laying out around it meant the invariant the search
       * enforces had to predict text metrics — and it got them slightly wrong
       * twice, landing captions on the strips two ranks over. The table below
       * already prints the same sentence per rank, in a column, in a font that
       * cannot collide with anything. One statement of a fact, in the place
       * that can state it exactly. */
    }

    out.push('</svg>');
    return out.join('');
  }

  /* ---------------------------------------------------------- the tree view */

  /**
   * The control topology, drawn as a tree. Only the edges are structural — the
   * per-step traffic is not drawn here because the point of showing the tree is
   * the SHAPE (log-depth, one root link) next to the ring's (linear depth,
   * every link equal). Drawing its schedule too would invite reading this as a
   * second replay, which it is not.
   */
  function treeView(trace, cfg) {
    var tree = trace.ring.compare && trace.ring.compare.tree;
    if (!tree) return '';
    var n = trace.ring.ranks.length;
    var edges = tree.edges || [];
    var depths = tree.depths || [];
    var maxDepth = depths.reduce(function (m, d) { return Math.max(m, d); }, 0);
    var width = cfg.size || 260;
    var levelH = 34;
    var height = (maxDepth + 1) * levelH + 16;
    var byDepth = [];
    for (var d = 0; d <= maxDepth; d++) byDepth.push([]);
    depths.forEach(function (dep, i) { byDepth[dep].push(i); });

    function pos(i) {
      var dep = depths[i];
      var row = byDepth[dep];
      var slot = row.indexOf(i);
      var x = width * (slot + 1) / (row.length + 1);
      var y = 14 + dep * levelH;
      return { x: x, y: y };
    }

    var out = ['<svg class="lab-ring-tree" viewBox="0 0 ' + width + ' ' + height +
               '" width="' + width + '" height="' + height + '" data-n="' + n + '">'];
    edges.forEach(function (e) {
      var from = trace.ring.ranks.findIndex(function (r) { return r.id === e.from; });
      var to = trace.ring.ranks.findIndex(function (r) { return r.id === e.to; });
      if (from < 0 || to < 0) return;
      var a = pos(from), b = pos(to);
      out.push('<line class="lab-ring-tedge" x1="' + a.x.toFixed(1) + '" y1="' +
               (a.y + 9).toFixed(1) + '" x2="' + b.x.toFixed(1) + '" y2="' +
               (b.y - 9).toFixed(1) + '"></line>');
    });
    for (var i = 0; i < n; i++) {
      var p = pos(i);
      out.push('<circle class="lab-ring-tnode" cx="' + p.x.toFixed(1) + '" cy="' +
               p.y.toFixed(1) + '" r="9"></circle>' +
               '<text class="lab-ring-tlabel" x="' + p.x.toFixed(1) + '" y="' +
               (p.y + 3.5).toFixed(1) + '">' + esc(trace.ring.ranks[i].id) +
               '</text>');
    }
    out.push('</svg>');
    return out.join('');
  }

  /* ------------------------------------------------------- the accumulator */

  /**
   * The traffic accumulator: cumulative bytes per link, Ring against the two
   * control topologies, on ONE shared y-scale.
   *
   * The y-max is the run's own largest curve, never a per-curve auto-scale. If
   * each curve were scaled to its own maximum they would all reach the top of
   * the chart and the factor of N — the entire content of this panel — would be
   * invisible. Same argument as the gantt's shared time axis, and the same
   * failure if it were got wrong: two charts that each look right and are not
   * comparable.
   *
   * The ring's curve is the AVERAGE across ranks, which for this schedule is
   * also each rank's value (every rank sends exactly 2(N-1)·S/N). The generator
   * asserts that flatness, so the average is not hiding a spread.
   */
  function accumulator(trace, series, cursor) {
    var n = trace.ring.ranks.length;
    var ib = itemBytes(trace);
    var chunkBytes = trace.ring.chunk_elements * ib;
    var steps = trace.steps.length;
    var at = Math.min(cursor, steps - 1);
    var ringNow = series[at] || series[series.length - 1] || [];
    var ringCurve = series.map(function (row) {
      var s = 0;
      for (var i = 0; i < row.length; i++) s += row[i];
      return row.length ? s / row.length : 0;
    });

    var controls = [];
    ['naive', 'tree'].forEach(function (key) {
      var c = controlCurve(trace.ring.compare && trace.ring.compare[key]);
      if (c) controls.push(c);
    });

    var seriesMax = 0;
    ringCurve.forEach(function (v) { if (v > seriesMax) seriesMax = v; });
    controls.forEach(function (c) {
      c.series.forEach(function (v) { if (v > seriesMax) seriesMax = v; });
    });
    if (seriesMax <= 0) seriesMax = chunkBytes;

    var W = 460, H = 108, padL = 8, padB = 16, padT = 8, padR = 8;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var spanX = Math.max(1, steps - 1);

    function line(curve, cls) {
      var pts = [];
      for (var i = 0; i < steps; i++) {
        var v = curveAt(curve, i);
        var x = padL + plotW * (steps <= 1 ? 0 : i / spanX);
        var y = padT + plotH * (1 - Math.min(1, v / seriesMax));
        pts.push(x.toFixed(1) + ',' + y.toFixed(1));
      }
      return '<polyline class="' + cls + '" points="' + pts.join(' ') + '"></polyline>';
    }

    var html = '<div class="lab-ring-acc" data-max="' + seriesMax +
               '" data-at="' + at + '">';
    html += '<div class="lab-ring-acc-hd"><span>通信量累计</span>' +
            '<span class="lab-ring-acc-sub">单链路累计字节 · 竖线 = 当前步</span></div>';

    html += '<svg class="lab-ring-acc-svg" viewBox="0 0 ' + W + ' ' + H +
            '" width="100%" height="' + H + '">';
    // The cursor rule first, so every curve paints over it.
    var cx = padL + plotW * (steps <= 1 ? 0 : at / spanX);
    html += '<line class="lab-ring-acc-cursor" x1="' + cx.toFixed(1) + '" y1="' +
            padT + '" x2="' + cx.toFixed(1) + '" y2="' + (padT + plotH) +
            '"></line>';
    html += '<line class="lab-ring-acc-axis" x1="' + padL + '" y1="' +
            (padT + plotH) + '" x2="' + (padL + plotW) + '" y2="' +
            (padT + plotH) + '"></line>';
    controls.forEach(function (c, i) {
      html += line(c, 'lab-ring-acc-line ' +
        (i === 0 ? 'lab-ring-acc-line-naive' : 'lab-ring-acc-line-tree'));
    });
    html += '<polyline class="lab-ring-acc-line lab-ring-acc-line-ring" points="' +
            ringCurve.map(function (v, i) {
              var x = padL + plotW * (steps <= 1 ? 0 : i / spanX);
              var y = padT + plotH * (1 - Math.min(1, v / seriesMax));
              return x.toFixed(1) + ',' + y.toFixed(1);
            }).join(' ') + '"></polyline>';
    html += '<text class="lab-ring-acc-ylab" x="' + padL + '" y="' +
            (padT + 8) + '">' + esc(String(seriesMax)) + ' B</text>';
    html += '</svg>';

    html += '<div class="lab-ring-acc-legend">';
    html += '<span class="lab-ring-acc-key"><i class="lab-ring-acc-sw lab-ring-acc-line-ring">' +
            '</i>Ring · 每卡 ' + esc(String(Math.round(ringCurve[at] || 0))) + ' B</span>';
    controls.forEach(function (c, i) {
      html += '<span class="lab-ring-acc-key"><i class="lab-ring-acc-sw ' +
              (i === 0 ? 'lab-ring-acc-line-naive' : 'lab-ring-acc-line-tree') +
              '"></i>' + esc(c.label) + ' · 瓶颈链路 ' +
              esc(String(curveAt(c, at))) + ' B</span>';
    });
    html += '</div>';

    // The quantitative claim, next to the curves that show it. The ratio is
    // read off the totals rather than recomputed from N, so it is a statement
    // about this trace and a wrong trace would show a wrong number.
    var ringTotal = ringCurve[steps - 1] || 0;
    var naiveTotal = controls.length ? controls[0].total : 0;
    var ratio = ringTotal ? naiveTotal / ringTotal : n;
    html += '<div class="lab-ring-acc-note"><b>' + esc(ratio.toFixed(2)) +
            '×</b> —— 朴素方案的瓶颈链路总共搬了环形的这么多倍' +
            '（理论值 N = ' + n + '）。三条调度搬的<b>总</b>字节数是同一个数，' +
            '区别只在于摊到几条链路上：环形的 ' + n + ' 条各担一份，' +
            '中心化的一条全担。</div>';
    html += '</div>';
    return html;
  }

  /* ================================================================== view */

  function makeView(config) {
    var cfg = config || {};
    if (!cfg.trace) throw new Error('ring: config.trace is required');
    var trace = cfg.trace;
    if (!trace.ring || !trace.ring.ranks || !trace.ring.ranks.length) {
      throw new Error('ring: trace.ring.ranks is required (see the trace contract)');
    }
    var ranks = rankIdsOf(trace);
    var n = ranks.length;
    var chunks = trace.ring.chunks;
    var series = bytesSeries(trace);

    function render(cursor, ctx) {
      var at = Math.max(0, Math.min(cursor, trace.steps.length - 1));
      var step = trace.steps[at];
      var counts = countsAfter(trace, at, ranks, chunks);
      var html = '<div class="lab-ring" data-cursor="' + at + '" data-n="' + n +
                 '" data-chunks="' + chunks + '">';

      html += timeline(trace, at);
      html += '<div class="lab-ring-main">';
      html += topology(trace, step, {
        size: cfg.size || 260, counts: counts
      });
      if (cfg.showTree !== false) {
        var tv = treeView(trace, { size: cfg.size || 260 });
        if (tv) {
          html += '<div class="lab-ring-treewrap"><div class="lab-ring-treehd">' +
                  '对照拓扑 · 二叉树</div>' + tv +
                  '<div class="lab-ring-treenote">' +
                  esc((trace.ring.compare.tree.note || '')) + '</div></div>';
        }
      }
      html += '</div>';

      html += table(trace, counts, step);
      html += accumulator(trace, series, at);
      html += '</div>';
      return html;
    }

    return {
      render: render,
      counts: function (cursor) { return countsAfter(trace, cursor, ranks, chunks); },
      series: function () { return series; },
      /* Exposed so the acceptance harness can compare the fold the component
       * actually uses against a from-scratch recomputation, rather than
       * trusting that they are the same code. */
      fold: function (cursor) { return countsAfter(trace, cursor, ranks, chunks); }
    };
  }

  /**
   * The step timeline: 2(N-1) ticks, grouped by phase, with the cursor marked.
   *
   * A tick is one COMMUNICATION step, and the tick's colour says which phase
   * it belongs to. This is the answer to "where am I in the algorithm" that the
   * reader needs before any of the rest is legible — ReduceScatter and
   * AllGather have the same length and the same picture, and only this strip
   * says which one the cursor is in.
   *
   * A tick's `data-step` is the TRACE step index it stands for, which is not
   * its position in the strip: the trace opens with a prepare frame (step 0)
   * that moves nothing and therefore has no tick. So tick `k` of the first
   * phase is trace step `k+1`, and a reader (or the harness) can compare
   * `data-step` to the cursor directly instead of keeping a hand-counted
   * offset. At cursor 0 nothing is lit, which is the honest reading — no link
   * has carried anything yet.
   */
  function timeline(trace, at) {
    var n = trace.ring.ranks.length;
    var phases = trace.ring.phases || [];
    // Where the first tick's trace index sits. Derived from the trace rather
    // than assumed, so a trace without a prepare frame does not silently shift
    // the whole strip by one.
    var first = (trace.steps[0] && (trace.steps[0].transfers || []).length === 0)
      ? 1 : 0;
    var html = '<div class="lab-ring-tl">';
    var k0 = 0;
    phases.forEach(function (ph) {
      var count = n - 1;
      html += '<div class="lab-ring-tl-phase" data-phase="' + escAttr(ph.id) + '">';
      html += '<div class="lab-ring-tl-hd">' + esc(ph.label) +
              '<span class="lab-ring-tl-n">' + count + ' 步</span></div>';
      html += '<div class="lab-ring-tl-ticks">';
      for (var k = 0; k < count; k++, k0++) {
        var stepIdx = first + k0;
        var on = stepIdx === at;
        var past = stepIdx < at;
        html += '<span class="lab-ring-tl-tick' + (on ? ' lab-ring-tl-on' : '') +
                (past ? ' lab-ring-tl-past' : '') + '" data-step="' + stepIdx +
                '" title="第 ' + stepIdx + ' 步 · ' + esc(ph.label) + ' 第 ' +
                (k + 1) + ' / ' + count + '"><i>' + (k + 1) + '</i></span>';
      }
      html += '</div></div>';
    });
    html += '</div>';
    return html;
  }

  /**
   * The per-rank buffer table: one row per rank, one cell per chunk, the count
   * in the cell. The topology's strips are the same data in radial form; both
   * are drawn because the table is where a reader checks a specific number and
   * the ring is where they see the pattern.
   */
  function table(trace, counts, step) {
    var n = trace.ring.ranks.length;
    var chunks = trace.ring.chunks;
    var chunkBytes = trace.ring.chunk_elements * (trace.meta.item_bytes || 4);
    var live = activeLinks(step);
    var html = '<table class="lab-ring-table"><thead><tr>' +
               '<th class="lab-ring-th-rank">rank</th>';
    for (var c = 0; c < chunks; c++) {
      html += '<th>chunk ' + c + '<span class="lab-ring-th-b">' +
              chunkBytes + 'B</span></th>';
    }
    html += '<th class="lab-ring-th-act">本步</th></tr></thead><tbody>';
    for (var r = 0; r < n; r++) {
      var rid = trace.ring.ranks[r].id;
      var sending = live[rid + '>' + trace.ring.ranks[(r + 1) % n].id];
      var receiving = null;
      for (var key in live) {
        if (key.split('>')[1] === rid) { receiving = live[key]; break; }
      }
      html += '<tr data-rank="' + escAttr(rid) + '"><td class="lab-ring-td-rank">' +
              esc(rid) + '</td>';
      for (var c2 = 0; c2 < chunks; c2++) {
        var have = counts[r][c2];
        var full = have === n;
        html += '<td class="lab-ring-td-cell' + (full ? ' lab-ring-full' : '') +
                (have > 1 ? ' lab-ring-part' : '') + '" data-chunk="' + c2 +
                '" data-count="' + have + '">' + have +
                '<span class="lab-ring-td-n">/' + n + '</span></td>';
      }
      var act = [];
      if (sending) act.push('发 chunk ' + sending.chunk);
      if (receiving) act.push('收 chunk ' + receiving.chunk +
                              (receiving.mode === MODE_COPY ? '（覆盖）' : '（累加）'));
      html += '<td class="lab-ring-td-act">' +
              (act.length ? esc(act.join(' · ')) : '—') + '</td></tr>';
    }
    html += '</tbody></table>';
    return html;
  }

  /**
   * The player's `panels: [{id, label, render(step, ctx)}]` entry.
   *
   * The player hands panels `(step, {cursor, snapshot, before})`, which is not
   * enough to draw a topology that unfolds over the whole run — the counts at
   * the cursor are a fold over steps 0..cursor. So this closes over the trace
   * once, the same way the gantt and the memory-hierarchy stage do.
   */
  function panel(config, context) {
    if (!context || !context.trace) throw new Error('ring: panel needs {trace}');
    var view = makeView({
      trace: context.trace,
      size: config.size,
      showTree: config.showTree
    });
    return {
      id: config.id || 'ring',
      label: config.label || '环形拓扑',
      render: function (step, ctx) { return view.render(ctx.cursor, ctx); },
      view: view
    };
  }

  /* =================================================================== lint
   *
   * The author-side half of this view's contract lives in the trace generator
   * (labs/traces/ring_allreduce.py). This is the engine-side half, and the
   * rules must be the same list in both places: a rule that exists on one side
   * only is a rule tested on neither, which is the failure mode this pair
   * exists to prevent. Both sides carry a sabotage control group for the same
   * reason.
   *
   * The rules live here rather than in trace-model.js because they are THIS
   * view's contract — ranks, chunks, transfers, the control schedules — and no
   * other view or lab reads them. trace-model.js is the engine's general lint
   * (declared tensors, graph/step agreement, three-tier bindings) and belongs
   * to every lab alike. In particular: NONE of these rules may be written as a
   * global requirement, because a trace of any other lab has no `ring` block
   * and would report a false gap — which is the exact failure five earlier
   * traces hit.
   */
  var RANK_RE = /^[A-Za-z0-9_-]+$/;
  var XFER_KEYS = ['from', 'to', 'chunk', 'elements'];
  var XFER_OPTIONAL = ['mode'];
  var MODES = [MODE_ADD, MODE_COPY];

  function own(obj, k) {
    return Object.prototype.hasOwnProperty.call(obj, k);
  }

  function isPosInt(v) {
    return typeof v === 'number' && isFinite(v) && v >= 1 && Math.floor(v) === v;
  }

  function isNonNegInt(v) {
    return typeof v === 'number' && isFinite(v) && v >= 0 && Math.floor(v) === v;
  }

  /**
   * Fold the chunk counts out of a transfer list — the same two rules the view
   * uses, on the same vocabulary. Returns null when the list is malformed,
   * because folding nonsense would produce a verdict about the fold rather than
   * about the trace; the malformation itself is already a gap by then.
   */
  function foldCounts(rankIds, chunks, steps) {
    var at = {};
    rankIds.forEach(function (rid, i) { at[rid] = i; });
    var counts = [];
    for (var r = 0; r < rankIds.length; r++) {
      counts.push([]);
      for (var c = 0; c < chunks; c++) counts[r].push(1);
    }
    for (var i = 0; i < steps.length; i++) {
      var xfers = steps[i].transfers;
      if (!Array.isArray(xfers)) return null;
      for (var j = 0; j < xfers.length; j++) {
        var x = xfers[j];
        var from = at[x.from], to = at[x.to];
        if (from === undefined || to === undefined) return null;
        if (!isNonNegInt(x.chunk) || x.chunk >= chunks) return null;
        if (MODES.indexOf(x.mode || MODE_ADD) === -1) return null;
        counts[to][x.chunk] = ((x.mode || MODE_ADD) === MODE_COPY)
          ? counts[from][x.chunk]
          : counts[to][x.chunk] + counts[from][x.chunk];
      }
    }
    return counts;
  }

  function lintTransfers(label, steps, rankIds, chunks, gaps) {
    if (!Array.isArray(steps) || !steps.length) {
      gaps.push({ what: label + ' 不是非空数组 —— 这条调度没有任何一步可放', why: '' });
      return;
    }
    steps.forEach(function (st, si) {
      var sid = label + '[' + si + ']';
      if (!st || typeof st !== 'object') {
        gaps.push({ what: sid + ' 不是对象', why: '' });
        return;
      }
      if (st.phase !== 'reduce' && st.phase !== 'gather') {
        gaps.push({ what: sid + '.phase = ' + JSON.stringify(st.phase) +
                    " 不是 'reduce' / 'gather' —— 视图按它给时间轴上色", why: '' });
      }
      if (!isNonNegInt(st.k)) {
        gaps.push({ what: sid + '.k = ' + JSON.stringify(st.k) + ' 不是非负整数', why: '' });
      }
      if (!Array.isArray(st.transfers)) {
        gaps.push({ what: sid + '.transfers 不是数组', why: '' });
        return;
      }
      var seen = {};
      st.transfers.forEach(function (x, xi) {
        var xid = sid + '.transfers[' + xi + ']';
        if (!x || typeof x !== 'object') {
          gaps.push({ what: xid + ' 不是对象', why: '' });
          return;
        }
        var extra = Object.keys(x).filter(function (k) {
          return XFER_KEYS.indexOf(k) === -1 && XFER_OPTIONAL.indexOf(k) === -1;
        });
        if (extra.length) {
          gaps.push({ what: xid + ' 多了字段 {' + extra.sort().join(', ') +
                      '} —— 契约只认 {from, to, chunk, elements, mode?}', why: '' });
        }
        var missing = XFER_KEYS.filter(function (k) { return !own(x, k); });
        if (missing.length) {
          gaps.push({ what: xid + ' 缺 {' + missing.sort().join(', ') + '}', why: '' });
          return;
        }
        if (rankIds.indexOf(x.from) === -1) {
          gaps.push({ what: xid + '.from = ' + JSON.stringify(x.from) +
                      ' 不是本 trace 声明的 rank', why: '' });
        }
        if (rankIds.indexOf(x.to) === -1) {
          gaps.push({ what: xid + '.to = ' + JSON.stringify(x.to) +
                      ' 不是本 trace 声明的 rank', why: '' });
        }
        if (x.from === x.to) {
          gaps.push({ what: xid + ' 的 from 与 to 都是 ' + JSON.stringify(x.from) +
                      ' —— 自己发给自己不是一次跨链路传输', why: '' });
        }
        if (!isNonNegInt(x.chunk) || x.chunk >= chunks) {
          gaps.push({ what: xid + '.chunk = ' + JSON.stringify(x.chunk) +
                      ' 不在 0..' + (chunks - 1) + ' 里', why: '' });
        }
        if (!isPosInt(x.elements)) {
          gaps.push({ what: xid + '.elements = ' + JSON.stringify(x.elements) +
                      ' 不是正整数', why: '' });
        }
        var mode = own(x, 'mode') ? x.mode : MODE_ADD;
        if (MODES.indexOf(mode) === -1) {
          gaps.push({ what: xid + '.mode = ' + JSON.stringify(mode) + ' 不是 ' +
                      MODES.join(' / ') +
                      ' —— add（累加）与 copy（覆盖）是两种不同的算术', why: '' });
        }
        var key = x.from + '>' + x.to + '#' + x.chunk;
        if (seen[key]) {
          gaps.push({ what: xid + ' 与同一 tick 的另一条传输重复（' + key +
                      '）—— 同一 tick 同一条链路上一个 chunk 只传一次', why: '' });
        }
        seen[key] = true;
      });
    });

    var counts = foldCounts(rankIds, chunks, steps);
    if (counts === null) return;
    for (var r = 0; r < rankIds.length; r++) {
      for (var c = 0; c < chunks; c++) {
        if (counts[r][c] !== rankIds.length) {
          gaps.push({
            what: label + ' 结束时 rank "' + rankIds[r] + '" 的 chunk ' + c +
                  ' 只含 ' + counts[r][c] + ' 份数据（应为 ' + rankIds.length +
                  '）—— 这条调度没有完成一次完整的 AllReduce',
            why: ''
          });
          return;
        }
      }
    }
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var declared = {};
    Object.keys((trace && trace.tensors) || {}).forEach(function (t) { declared[t] = true; });
    var ring = trace && trace.ring;

    if (!ring || typeof ring !== 'object') {
      gaps.push({ what: 'ring 缺失 —— 环形视图没有卡数、chunk 大小与归约对象的定义，' +
                        '一帧都画不出来', why: '' });
      return { gaps: gaps, warns: warns, infos: infos };
    }

    var n = ring.n;
    if (!isPosInt(n) || n < 2) {
      gaps.push({ what: 'ring.n = ' + JSON.stringify(n) + ' 不是 ≥ 2 的整数', why: '' });
      return { gaps: gaps, warns: warns, infos: infos };
    }

    var rankIds = [];
    if (!Array.isArray(ring.ranks)) {
      gaps.push({ what: 'ring.ranks 不是数组', why: '' });
    } else {
      ring.ranks.forEach(function (rk, i) {
        var rid = rk && rk.id;
        if (typeof rid !== 'string' || !rid) {
          gaps.push({ what: 'ring.ranks[' + i + '] 没有 id', why: '' });
          return;
        }
        if (!RANK_RE.test(rid)) {
          gaps.push({ what: 'ring.ranks[' + i + '].id = ' + JSON.stringify(rid) +
                      ' 含非法字符（只允许字母数字下划线连字符）', why: '' });
        }
        if (rankIds.indexOf(rid) !== -1) {
          gaps.push({ what: 'ring.ranks 里 id "' + rid + '" 重复 —— ' +
                      '两条传输会指向同一张卡', why: '' });
        }
        rankIds.push(rid);
        if (!rk.label) {
          warns.push({ what: 'ring.ranks[' + i + '] ("' + rid +
                       '") 没有 label，只能显示 id', why: '' });
        }
      });
      if (ring.ranks.length !== n) {
        gaps.push({ what: 'ring.ranks 有 ' + ring.ranks.length + ' 项，但 ring.n = ' +
                    n + ' —— 拓扑上画的节点数与调度里的卡数不是一回事', why: '' });
      }
    }

    var ce = ring.chunk_elements;
    if (!isPosInt(ce)) {
      gaps.push({ what: 'ring.chunk_elements = ' + JSON.stringify(ce) + ' 不是正整数',
                  why: '' });
      return { gaps: gaps, warns: warns, infos: infos };
    }

    var op = ring.op;
    if (!op || typeof op !== 'object') {
      gaps.push({ what: 'ring.op 缺失 —— 视图说不出这一次归约作用在哪个张量上', why: '' });
    } else {
      ['id', 'label', 'tensor', 'elements', 'bytes'].forEach(function (k) {
        if (!own(op, k)) gaps.push({ what: 'ring.op 缺 "' + k + '"', why: '' });
      });
      if (!declared[op.tensor]) {
        gaps.push({ what: 'ring.op.tensor = ' + JSON.stringify(op.tensor) +
                    ' 不在 tensors 里 —— 「一次归约作用于一个梯度张量」这句话就落不了地',
                    why: '' });
      }
      if (!isPosInt(op.elements)) {
        gaps.push({ what: 'ring.op.elements = ' + JSON.stringify(op.elements) +
                    ' 不是正整数', why: '' });
      }
      if (!isPosInt(op.bytes)) {
        gaps.push({ what: 'ring.op.bytes = ' + JSON.stringify(op.bytes) +
                    ' 不是正整数', why: '' });
      } else if (isPosInt(op.elements) && op.bytes !== op.elements * 4) {
        gaps.push({ what: 'ring.op.bytes = ' + op.bytes + ' 与 elements × 4 = ' +
                    (op.elements * 4) + ' 不一致', why: '' });
      }
    }

    var algorithm = ring.algorithm;
    if (algorithm !== 'ring' && algorithm !== 'tree') {
      gaps.push({ what: 'ring.algorithm = ' + JSON.stringify(algorithm) +
                  " 不是 'ring' / 'tree'", why: '' });
    }

    // The chunk count is derived from S and the chunk size; `chunks` is a
    // convenience copy and is checked against them rather than trusted.
    var total = isPosInt(op && op.elements) ? op.elements : null;
    var chunks = total !== null && total % ce === 0 ? total / ce : null;
    if (chunks === null) {
      gaps.push({ what: 'ring.op.elements = ' + JSON.stringify(total) +
                  ' 除不尽 chunk_elements = ' + ce, why: '' });
    } else if (own(ring, 'chunks') && ring.chunks !== chunks) {
      gaps.push({ what: 'ring.chunks = ' + JSON.stringify(ring.chunks) +
                  ' 与 elements / chunk_elements = ' + chunks + ' 不一致', why: '' });
    }
    if (algorithm === 'ring' && chunks !== null && chunks !== n) {
      gaps.push({ what: 'ring.algorithm 是 ring，但 chunk 数 ' + chunks +
                  ' 与卡数 ' + n + ' 不等 —— 环形要求 S 正好切成 N 块', why: '' });
    }
    var useChunks = chunks === null ? Math.max(1, n) : chunks;

    // ---- the trace's own steps
    var own_steps = [];
    (trace.steps || []).forEach(function (s) {
      var xfers = s.transfers;
      if (xfers === undefined || xfers === null) {
        gaps.push({ what: '步骤 "' + s.id + '" 没有 transfers —— ' +
                    '环形视图在这一帧不知道有没有数据过链', why: '' });
        return;
      }
      if (!Array.isArray(xfers)) {
        gaps.push({ what: '步骤 "' + s.id + '" 的 transfers 不是数组', why: '' });
        return;
      }
      (s.writes || []).forEach(function (w) {
        if (!declared[w]) {
          gaps.push({ what: '步骤 "' + s.id + '" 写了未声明的张量 "' + w + '"', why: '' });
        }
      });
      own_steps.push({ phase: s.phase_id, k: isNonNegInt(s.k) ? s.k : -1,
                       transfers: xfers });
    });
    if (own_steps.length) {
      lintTransfers('trace.steps', own_steps, rankIds, useChunks, gaps);
      if (algorithm === 'ring' && rankIds.length) {
        lintRingRotation('trace.steps', own_steps, rankIds, ce, gaps);
      }
    }

    // ---- the control schedules
    var compare = ring.compare;
    if (compare !== null && compare !== undefined) {
      if (typeof compare !== 'object') {
        gaps.push({ what: 'ring.compare 不是对象', why: '' });
      } else {
        ['naive', 'tree'].forEach(function (key) {
          if (!own(compare, key)) return;
          var entry = compare[key];
          var label = 'ring.compare.' + key;
          if (!entry || typeof entry !== 'object') {
            gaps.push({ what: label + ' 不是对象', why: '' });
            return;
          }
          if (!entry.label) {
            gaps.push({ what: label + ' 缺 label —— 对照曲线的图例会是一片空白', why: '' });
          }
          if (key === 'tree') {
            if (!Array.isArray(entry.edges) || !entry.edges.length) {
              gaps.push({ what: label + '.edges 不是非空数组 —— 树拓扑画不出来', why: '' });
            } else {
              var listed = {};
              entry.edges.forEach(function (e) {
                if (!e || typeof e !== 'object') {
                  gaps.push({ what: label + '.edges 里有非对象项', why: '' });
                  return;
                }
                if (rankIds.indexOf(e.from) === -1) {
                  gaps.push({ what: label + '.edges 的 from = ' + JSON.stringify(e.from) +
                              ' 不是声明的 rank', why: '' });
                }
                if (rankIds.indexOf(e.to) === -1) {
                  gaps.push({ what: label + '.edges 的 to = ' + JSON.stringify(e.to) +
                              ' 不是声明的 rank', why: '' });
                }
                listed[e.from] = true;
              });
              var missing = rankIds.slice(1).filter(function (r) { return !listed[r]; });
              if (missing.length) {
                gaps.push({ what: label + '.edges 里 ' + missing.join('、') +
                            ' 没有父节点 —— 这棵树接不到根上', why: '' });
              }
            }
          }
          var entryChunks = (isPosInt(entry.chunk_elements) && total !== null &&
                             total % entry.chunk_elements === 0)
            ? total / entry.chunk_elements : useChunks;
          lintTransfers(label + '.steps', entry.steps, rankIds, entryChunks, gaps);
        });
      }
    }

    infos.push({ what: 'N=' + n + ' · ' + (trace.steps || []).length + ' 步 · ' +
                 ce + ' 元素/chunk · ' + useChunks + ' 块', why: '' });
    return { gaps: gaps, warns: warns, infos: infos };
  }

  /**
   * The rules that make a schedule a RING, and not just a valid collective.
   *
   * Gated on `algorithm === 'ring'` because they are false of the tree by
   * construction (a tree's leaves send and its interior receives; that is the
   * point of it). The downstream of rank `i` comes from the DECLARED rank order,
   * not from parsing the rank's name, so a trace that named its ranks anything
   * else is still checked.
   */
  function lintRingRotation(label, steps, rankIds, chunkElements, gaps) {
    var n = rankIds.length;
    var successor = {};
    for (var i = 0; i < n; i++) successor[rankIds[i]] = rankIds[(i + 1) % n];
    steps.forEach(function (st, si) {
      var xfers = st.transfers;
      if (!Array.isArray(xfers) || !xfers.length) return;
      if (xfers.length !== n) {
        gaps.push({ what: label + '[' + si + '] 有 ' + xfers.length + ' 条传输，' +
                    '环形是 ' + n + ' 张卡一步各发一条', why: '' });
      }
      var senders = xfers.map(function (x) { return x && x.from; });
      var uniq = senders.filter(function (v, i) { return senders.indexOf(v) === i; });
      if (uniq.length !== senders.length) {
        gaps.push({ what: label + '[' + si + '] 里有 rank 一步发了不止一块 —— ' +
                    '环形每步每卡只发一块（这也是「每卡每步 S/N」的前提）', why: '' });
      }
      xfers.forEach(function (x) {
        if (!x || typeof x !== 'object') return;
        if (own(successor, x.from) && x.to !== successor[x.from]) {
          gaps.push({ what: label + '[' + si + ']：rank ' + x.from + ' 发给了 ' +
                      x.to + '，环上的下游应当是 ' + successor[x.from], why: '' });
        }
        if (x.elements !== chunkElements) {
          gaps.push({ what: label + '[' + si + ']：一条传输搬了 ' + x.elements +
                      ' 个元素，但环形每步搬一个 chunk = ' + chunkElements +
                      ' 个 —— «每步传 S/N» 这个数字就是带宽最优性的全部依据',
                      why: '' });
        }
      });
    });
  }

  /* Each entry breaks one rule above, on the real trace. The names deliberately
   * parallel the Python SABOTAGE_CASES in labs/traces/ring_allreduce.py, so a
   * reader can match them up; a rule with no entry here is a rule this port
   * does not actually test. */
  var SABOTAGE_CASES = {
    'ring 整块缺失': function (t) { delete t.ring; },
    'ring.n 不是整数': function (t) { t.ring.n = '4'; },
    'ring.n 小于 2': function (t) { t.ring.n = 1; },
    'ring.ranks 与 n 不符': function (t) { t.ring.ranks.pop(); },
    'rank id 重复': function (t) {
      t.ring.ranks.push(JSON.parse(JSON.stringify(t.ring.ranks[0])));
    },
    'rank id 含非法字符': function (t) { t.ring.ranks[0].id = 'rank 0'; },
    'rank 没有 id': function (t) { delete t.ring.ranks[0].id; },
    'chunk_elements 为 0': function (t) { t.ring.chunk_elements = 0; },
    'chunk 数不等于卡数': function (t) { t.ring.chunk_elements = 8; },
    'ring.op 缺失': function (t) { delete t.ring.op; },
    'ring.op.tensor 未声明': function (t) { t.ring.op.tensor = 'grad2'; },
    'ring.op.bytes 与 elements 不符': function (t) { t.ring.op.bytes = 1; },
    'ring.algorithm 非法': function (t) { t.ring.algorithm = 'star'; },
    '步骤没有 transfers': function (t) { delete t.steps[_firstXferStep(t)].transfers; },
    '一条传输发给不存在的 rank': function (t) {
      _firstXfer(t).to = 'r99';
    },
    '一条传输发给自己': function (t) {
      var x = _firstXfer(t); x.to = x.from;
    },
    'chunk 下标越界': function (t) { _firstXfer(t).chunk = 99; },
    'elements 是 0': function (t) { _firstXfer(t).elements = 0; },
    'mode 不在词表里': function (t) { _firstXfer(t).mode = 'merge'; },
    '传输多了个字段': function (t) { _firstXfer(t).color = 'red'; },
    '同一 tick 重复传同一个 chunk': function (t) {
      var st = t.steps[_firstXferStep(t)];
      st.transfers.push(JSON.parse(JSON.stringify(st.transfers[0])));
    },
    '发错了下游（不是环上的下一个）': function (t) {
      _firstXfer(t).to = t.ring.ranks[0].id;
    },
    '每步搬的不是一个 chunk': function (t) {
      _firstXfer(t).elements = t.ring.chunk_elements + 1;
    },
    '某一步少了一条传输': function (t) {
      t.steps[_firstXferStep(t)].transfers.pop();
    },
    '对照调度缺 label': function (t) { delete t.ring.compare.naive.label; },
    '树的边指向不存在的 rank': function (t) { t.ring.compare.tree.edges[0].to = 'r99'; },
    '树有一个节点接不到根上': function (t) {
      t.ring.compare.tree.edges.shift();
    },
    '标签不是非负整数': function (t) { t.ring.compare.naive.steps[0].k = -1; }
  };

  function _firstXferStep(t) {
    for (var i = 0; i < t.steps.length; i++) {
      if (t.steps[i].transfers && t.steps[i].transfers.length) return i;
    }
    return 0;
  }

  function _firstXfer(t) {
    return t.steps[_firstXferStep(t)].transfers[0];
  }

  /**
   * Prove the lint is not a function that always returns zero. Returns the list
   * of mutations it failed to catch, and separately the ones that did not
   * change the trace at all.
   *
   * The second list exists because a mutation written against a literal is a
   * no-op on some traces and a real break on others, and both report "0 gaps" —
   * see the same split in labs/traces/ring_allreduce.py.
   */
  function sabotageChecks(trace) {
    var missed = [], dead = [];
    var before = JSON.stringify(trace);
    Object.keys(SABOTAGE_CASES).forEach(function (name) {
      var copy = JSON.parse(JSON.stringify(trace));
      try {
        SABOTAGE_CASES[name](copy);
      } catch (err) {
        missed.push(name + '（破坏本身失败：' + err.message + '）');
        return;
      }
      if (JSON.stringify(copy) === before) {
        dead.push(name + '（破坏后 trace 没变，是空操作）');
        return;
      }
      var r = lint(copy);
      if (!r.gaps.length && !r.warns.length) missed.push(name);
    });
    return { missed: missed, dead: dead };
  }

  /* ============================================================ self-check
   *
   * `check(trace)` runs the three things that decide whether this component's
   * picture can be trusted, and reports them separately because they fail for
   * different reasons:
   *
   *   1. PURITY — the fold at cursor `c` equals having played forward to `c`.
   *      Every jump shape the player can produce, against a forward walk.
   *
   *   2. CONTROL — a deliberately wrong, stateful fold must DISAGREE on a
   *      backwards jump, or (1) proves nothing. This is the check that makes
   *      the purity claim evidence rather than a tautology, and it is the
   *      reason the component is written as a fold in the first place.
   *
   *      It has to live here rather than being delegated to the engine's
   *      generic `verify.runJumpCheck`, and that is worth stating: the
   *      engine's control group operates on TENSOR STATE, and it has no
   *      discriminating power on this trace. Every step of a ring rewrites
   *      every rank's whole buffer, so a stateful player that applies only the
   *      target step's state happens to land on the right answer — the ring's
   *      own completeness is what defeats it. The view's rendering does not
   *      depend on tensor state alone: it depends on the FOLD of the transfer
   *      list, and a stateful fold of that is wrong on a backwards jump. So
   *      the control belongs at the level the claim is made.
   *
   *   3. LINT — the contract holds and the sabotage table is live.
   */
  function check(trace) {
    var ranks = rankIdsOf(trace);
    var chunks = trace.ring.chunks;
    var last = trace.steps.length - 1;
    var nRef = ranks.length;

    // ---- 1. purity: fold(c) == play forward to c, for every jump shape.
    var plan = [];
    for (var i = 0; i <= last; i++) plan.push({ label: '正向 → ' + i, target: i });
    for (var j = last; j >= 0; j--) plan.push({ label: '完全逆序 → ' + j, target: j });
    if (last >= 3) {
      [[last, 1], [1, 0], [0, last], [last, 2], [2, 0], [2, last],
       [last, last], [1, 2], [2, 1]].forEach(function (pair) {
        plan.push({ label: pair[0] + ' → ' + pair[1], target: pair[1] });
      });
    }

    /* The reference: walk forward from step 0, one step at a time, holding an
     * accumulator — the way a reader watching the replay without ever
     * scrubbing would see it. This is deliberately NOT `countsAfter`, which
     * folds the same list in a different way (it starts from a seeded grid and
     * replays every step up to the cursor). Two implementations, one answer. */
    var forwardMemo = [];
    (function buildForward() {
      var acc = [];
      for (var fr2 = 0; fr2 < nRef; fr2++) {
        acc.push([]);
        for (var fc2 = 0; fc2 < chunks; fc2++) acc[fr2].push(1);
      }
      var ridx = {};
      ranks.forEach(function (rid, i) { ridx[rid] = i; });
      for (var s = 0; s < trace.steps.length; s++) {
        (trace.steps[s].transfers || []).forEach(function (x) {
          var from = ridx[x.from], to = ridx[x.to];
          if (from === undefined || to === undefined) return;
          if (x.chunk < 0 || x.chunk >= chunks) return;
          acc[to][x.chunk] = (x.mode === MODE_COPY)
            ? acc[from][x.chunk]
            : acc[to][x.chunk] + acc[from][x.chunk];
        });
        forwardMemo.push(JSON.parse(JSON.stringify(acc)));
      }
    })();

    var pureRows = [];
    var pureFailures = 0;
    plan.forEach(function (jump) {
      var viaFold = countsAfter(trace, jump.target, ranks, chunks);
      var viaPlay = forwardMemo[jump.target];
      var same = JSON.stringify(viaFold) === JSON.stringify(viaPlay);
      if (!same) pureFailures++;
      pureRows.push({ label: jump.label, ok: same });
    });

    // ---- 2. control: an accumulator that never re-derives.
    //
    // It starts at cursor 0 and, for each jump, applies ONLY the target step's
    // transfers on top of what it already holds. Playing forward that is
    // correct; jumping backwards it is not, because it keeps the later steps'
    // counts — which is the bug this component's fold exists to make
    // impossible.
    var stateful = [];
    for (var r = 0; r < nRef; r++) {
      stateful.push([]);
      for (var c = 0; c < chunks; c++) stateful[r].push(1);
    }
    var at = 0;

    function applyStepCounts(state, step) {
      var ridx = {};
      ranks.forEach(function (rid, i) { ridx[rid] = i; });
      (step.transfers || []).forEach(function (x) {
        var from = ridx[x.from], to = ridx[x.to];
        if (from === undefined || to === undefined) return;
        if (x.chunk < 0 || x.chunk >= chunks) return;
        state[to][x.chunk] = (x.mode === MODE_COPY)
          ? state[from][x.chunk]
          : state[to][x.chunk] + state[from][x.chunk];
      });
    }

    var controlRows = [];
    plan.forEach(function (jump) {
      // advance-only: apply the steps from wherever we are to the target,
      // silently doing nothing when the target is behind us.
      for (var s = at + 1; s <= jump.target; s++) {
        applyStepCounts(stateful, trace.steps[s]);
      }
      at = jump.target;
      var truth = forwardMemo[jump.target];
      var agree = JSON.stringify(stateful) === JSON.stringify(truth);
      controlRows.push({ label: jump.label, ok: agree, target: jump.target });
    });
    var controlFailures = controlRows.filter(function (x) { return !x.ok; }).length;

    var lintResult = lint(trace);
    var sab = sabotageChecks(trace);

    return {
      jumps: {
        total: plan.length,
        pureFailures: pureFailures,
        controlFailures: controlFailures,
        rows: pureRows,
        /* Conclusive only if the control group actually caught something. */
        conclusive: controlFailures > 0,
        passed: pureFailures === 0 && controlFailures > 0
      },
      control: controlRows,
      lint: lintResult,
      sabotage: sab,
      passed: pureFailures === 0 && controlFailures > 0 &&
              lintResult.gaps.length === 0 && !sab.missed.length && !sab.dead.length
    };
  }

  /**
   * Render `check()`'s verdict into `host`, in the same shape the ledger's
   * panel uses: an honest sentence per check, and a warning when a control
   * group had nothing to catch.
   */
  function panelSelfCheck(host, trace) {
    var r = check(trace);
    var html = '';
    html += '<div class="lab-verdict ' + (r.jumps.passed ? 'lab-ok' : 'lab-bad') + '">' +
      '<b>任意跳转 · 纯函数折叠：</b>' + r.jumps.total + ' 次跳转（正向 / 完全逆序 / ' +
      '跳回 / 乱序），' +
      (r.jumps.pureFailures === 0
        ? '<b>0 次不一致</b>，与顺序播放到该步逐位相同。'
        : '<b>' + r.jumps.pureFailures + ' 次不一致</b>。') +
      '<br><b>对照组（有状态、只往后累积的折叠）：</b>同一批跳转里 <b>' +
      r.jumps.controlFailures + ' 次不一致</b>。' +
      (r.jumps.controlFailures === 0
        ? '<b>⚠️ 对照组全过了 —— 这组跳转没有区分力，上面的 0 不能当作证据。</b>'
        : '对照组确实在倒退跳转上翻车，说明上面那个 0 不是测试写错。') +
      '</div>';
    html += '<div class="lab-verdict ' +
      (r.lint.gaps.length === 0 && !r.sabotage.missed.length &&
       !r.sabotage.dead.length ? 'lab-ok' : 'lab-bad') + '">' +
      '<b>环形契约 lint：</b>干净的 trace 报 <b>' + r.lint.gaps.length + '</b> 个 gap / ' +
      r.lint.warns.length + ' 个 warn；对照组 ' +
      Object.keys(SABOTAGE_CASES).length + ' 种破坏' +
      (r.sabotage.dead.length
        ? '，<b>' + r.sabotage.dead.length + ' 种在本配置下是空操作：' +
          esc(r.sabotage.dead.join('、')) + '</b>'
        : '逐个确认改变了 trace') +
      (r.sabotage.missed.length
        ? '，<b>漏掉 ' + r.sabotage.missed.length + ' 种：' +
          esc(r.sabotage.missed.join('、')) + '</b>'
        : '且全部被抓到') + '。' +
      (r.lint.gaps.length === 0 && !r.sabotage.missed.length &&
       !r.sabotage.dead.length
        ? 'lint 是活的，所以上面的 0 有意义。' : '') +
      '</div>';
    html += '<div class="lab-verdict lab-ok"><b>引擎张量层（LabEngine.verify）：</b>' +
      '本 trace 的每一步都<b>整体重写</b>每张卡的缓冲区 —— 这是环形的真实状态，' +
      '每步每卡确实都收到数据。这一性质让引擎那套基于「张量状态稀疏更新」的通用对照组' +
      '在这份 trace 上<b>没有区分力</b>（它会与真值一致，而不是「没测」）。' +
      '所以本组件的对照组放在更上层：折叠的是<b>传输列表</b>而不是张量状态，' +
      '那里有状态版本确实会翻车。</div>';
    host.innerHTML = html;
    global.__labRingCheck = global.__labRingCheck || {};
    global.__labRingCheck[(trace.meta && trace.meta.lab) || 'lab'] = {
      jumps: r.jumps.total,
      pureFailures: r.jumps.pureFailures,
      controlFailures: r.jumps.controlFailures,
      conclusive: r.jumps.conclusive,
      lintGaps: r.lint.gaps.length,
      sabotageMissed: r.sabotage.missed,
      sabotageDead: r.sabotage.dead,
      passed: r.passed
    };
    return r;
  }

  NS.ring = {
    makeView: makeView,
    panel: panel,
    topology: topology,
    countsAfter: countsAfter,
    bytesSeries: bytesSeries,
    foldCounts: foldCounts,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES,
    check: check,
    panelSelfCheck: panelSelfCheck
  };
})(typeof window !== 'undefined' ? window : globalThis);
