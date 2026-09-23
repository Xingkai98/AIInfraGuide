/* L00 engine · DAG canvas.
 *
 * The DOM is built ONCE per trace and never rebuilt. Moving the cursor only
 * toggles classes, for two reasons that both showed up in the prototype:
 *
 *   - `innerHTML` on the whole SVG measured 0.47 ms/frame — more than a full
 *     KaTeX re-render (0.35 ms), i.e. the most expensive thing on the page.
 *     Toggling a class on a few dozen elements costs approximately nothing.
 *   - Rebuilding throws away the Tab focus. A reader who tabs to a node and
 *     presses Enter to jump would lose their place in the tab order on the very
 *     next render. Class toggling keeps focus exactly where it was.
 *
 * Node state is a pure function of (node's step, cursor): before -> visited,
 * equal -> current, after -> future. Nothing is accumulated.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var SVGNS = 'http://www.w3.org/2000/svg';

  function el(name, attrs) {
    var node = document.createElementNS(SVGNS, name);
    for (var k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) node.setAttribute(k, attrs[k]);
    }
    return node;
  }

  function edgePath(a, b, L) {
    var ng = L.ng;
    var x1 = a.x + ng.w;
    var y1 = a.y + ng.h / 2;
    var x2 = b.x;
    var y2 = b.y + ng.h / 2;

    if (a.band !== b.band) {
      /* Wrapped edge: leave the right edge of the band, run down the gutter
       * outside the node columns, and re-enter from the left. Routing it
       * through the middle would cross every box in the band.
       *
       * The gutter is measured from the canvas edge, not from the last column,
       * so the connector stays inside the viewBox — the layout reserves
       * `gutter` px of width for exactly this and nothing else. */
      var gutterX = L.W - ng.gutter / 2;
      var laneY = b.y - ng.h - ng.bandGap / 2;
      return 'M' + x1 + ',' + y1 +
             ' L' + gutterX + ',' + y1 +
             ' L' + gutterX + ',' + laneY +
             ' L' + (b.x - ng.xgap / 2) + ',' + laneY +
             ' L' + (b.x - ng.xgap / 2) + ',' + y2 +
             ' L' + x2 + ',' + y2;
    }

    if (b.layer > a.layer + 1) {
      /* Long edge inside a band: bow it above the row so it reads as "skips
       * over these" rather than cutting through them. */
      var mx = (x1 + x2) / 2;
      return 'M' + x1 + ',' + y1 + ' Q' + mx + ',' + (y1 - 22) + ' ' + x2 + ',' + y2;
    }

    var cx = (x1 + x2) / 2;
    return 'M' + x1 + ',' + y1 + ' C' + cx + ',' + y1 + ' ' + cx + ',' + y2 + ' ' + x2 + ',' + y2;
  }

  function stateClass(step, cursor) {
    if (step < 0) return 'lab-st-future';
    if (step === cursor) return 'lab-st-current';
    return step < cursor ? 'lab-st-visited' : 'lab-st-future';
  }

  /**
   * @param {Element} host    container to append the <svg> to
   * @param {object} idx      trace index
   * @param {object} L        layout result
   * @param {function} onJump called with a step index when a node is activated
   */
  function create(host, idx, L, onJump) {
    var ng = L.ng;
    while (host.firstChild) host.removeChild(host.firstChild);

    var svg = el('svg', {
      class: 'lab-dag',
      viewBox: '0 0 ' + L.W + ' ' + L.H,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'group',
      'aria-label': '执行 DAG，' + idx.nodes.length + ' 个节点'
    });
    /* A well-proportioned graph would otherwise be magnified until its nodes
     * are cartoonishly large — a 660x185 layout stretched across a 1560px panel
     * scales 2.4x. Capping the displayed width keeps a node near a readable
     * ~130px and centres the rest as margin. */
    svg.style.maxWidth = Math.round((130 / ng.w) * L.W) + 'px';

    var defs = el('defs');
    [['lab-arw', '#586379'], ['lab-arw-c', '#4ea1ff']].forEach(function (spec) {
      var marker = el('marker', {
        id: spec[0], viewBox: '0 0 8 6', refX: '7.5', refY: '3',
        markerWidth: '7', markerHeight: '6', orient: 'auto-start-reverse'
      });
      marker.appendChild(el('path', { d: 'M0,0 L8,3 L0,6 z', fill: spec[1] }));
      defs.appendChild(marker);
    });
    svg.appendChild(defs);

    var edgeLayer = el('g', { class: 'lab-dag-edges' });
    var nodeLayer = el('g', { class: 'lab-dag-nodes' });
    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);

    /* ---- edges. Only real data edges are drawn; the layout's synthetic
     * sequence edges exist purely to assign layers. */
    var edgeViews = idx.edges.map(function (e) {
      var a = L.pos[e.from];
      var b = L.pos[e.to];
      if (!a || !b) return null;
      var path = el('path', { class: 'lab-eg', d: edgePath(a, b, L), 'marker-end': 'url(#lab-arw)' });
      edgeLayer.appendChild(path);
      var label = null;
      if (e.tensor) {
        label = el('text', {
          class: 'lab-eg-label',
          x: (a.x + ng.w + b.x) / 2,
          y: (a.y + b.y) / 2 + ng.h / 2 - 4
        });
        label.textContent = e.tensor;
        edgeLayer.appendChild(label);
      }
      return { path: path, label: label, from: idx.stepOf[e.from], to: idx.stepOf[e.to] };
    }).filter(Boolean);

    /* ---- nodes */
    var nodeViews = idx.nodes.map(function (n) {
      var p = L.pos[n.id];
      if (!p) return null;
      var clickable = n.step >= 0;
      var g = el('g', {
        class: 'lab-nd lab-kind-' + n.kind + ' ' + stateClass(n.step, 0),
        'data-step': String(n.step),
        'data-node': n.id
      });
      if (clickable) {
        /* Tab-reachable and Enter-activatable, per the player interaction
         * decision. The role/aria pair is what makes the SVG node announce
         * itself as a control rather than as decoration. */
        g.setAttribute('tabindex', '0');
        g.setAttribute('role', 'button');
        g.setAttribute('aria-label', '第 ' + n.step + ' 步：' + n.title);
      } else {
        g.setAttribute('aria-hidden', 'true');
      }

      g.appendChild(el('rect', { class: 'lab-halo', x: p.x - 2.5, y: p.y - 2.5,
                                 width: ng.w + 5, height: ng.h + 5, rx: 7 }));
      g.appendChild(el('rect', { class: 'lab-box', x: p.x, y: p.y,
                                 width: ng.w, height: ng.h, rx: 5 }));
      g.appendChild(el('circle', { class: 'lab-kinddot', cx: p.x + 8, cy: p.y + 8, r: 2.4 }));

      var idText = el('text', { class: 'lab-nd-id', x: p.x + ng.w / 2 + 3, y: p.y + 12 });
      idText.textContent = n.id;
      g.appendChild(idText);

      var opText = el('text', { class: 'lab-nd-op', x: p.x + ng.w / 2 + 3, y: p.y + 21 });
      opText.textContent = n.op;
      g.appendChild(opText);

      if (clickable) {
        var activate = function (ev) {
          if (ev) ev.preventDefault();
          onJump(n.step);
        };
        g.addEventListener('click', activate);
        g.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') activate(ev);
        });
      }
      nodeLayer.appendChild(g);
      return { g: g, step: n.step };
    }).filter(Boolean);

    var lastCursor = -1;
    function update(cursor) {
      if (cursor === lastCursor) return;
      lastCursor = cursor;
      nodeViews.forEach(function (v) {
        var want = stateClass(v.step, cursor);
        var cls = v.g.classList;
        if (!cls.contains(want)) {
          cls.remove('lab-st-current', 'lab-st-visited', 'lab-st-future');
          cls.add(want);
        }
      });
      edgeViews.forEach(function (v) {
        var isCurrent = v.to === cursor;
        var isVisited = v.to >= 0 && v.to < cursor;
        var path = v.path.classList;
        path.toggle('lab-st-current', isCurrent);
        path.toggle('lab-st-visited', !isCurrent && isVisited);
        /* The arrowhead colour is a separate marker, so it has to be swapped
         * alongside the class rather than expressed in CSS. */
        v.path.setAttribute('marker-end', isCurrent ? 'url(#lab-arw-c)' : 'url(#lab-arw)');
        if (v.label) {
          v.label.classList.toggle('lab-st-current', isCurrent);
          v.label.classList.toggle('lab-st-visited', !isCurrent && isVisited);
        }
      });
    }

    host.appendChild(svg);
    var ar = (L.W / L.H).toFixed(2);

    return {
      svg: svg,
      update: update,
      stats: {
        nodes: idx.nodes.length,
        edges: idx.edges.length,
        layers: L.numLayers,
        bands: L.bandCount,
        widest: L.widest,
        W: L.W,
        H: L.H,
        aspect: ar,
        wrapped: L.doWrap
      }
    };
  }

  NS.dag = { create: create, edgePath: edgePath, stateClass: stateClass };
})(typeof window !== 'undefined' ? window : globalThis);
