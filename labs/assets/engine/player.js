/* L00 engine · player — the public entry point.
 *
 * One lab page = one `LabEngine.lab(trace, config)` call. The page holds a
 * trace and a little chrome; every algorithm-shaped decision lives in the
 * engine, which is the property that lets the remaining labs be data rather
 * than code.
 *
 * The interaction rules implemented here are decisions, not defaults:
 *
 *   - Opens on the FIRST step. No autoplay: a reader arriving from a tutorial
 *     needs to see what the thing is before it starts moving.
 *   - `?step=N` deep links, so an entry card can point at a teaching moment.
 *   - Dragging the cursor updates the URL with `history.replaceState`, never
 *     `pushState` — scrubbing is not navigation, and a back button that walks
 *     back through 30 scrub positions is hostile.
 *   - Space toggles play (with preventDefault, or the page scrolls instead),
 *     arrows step, Home/End jump to the ends.
 *
 * Every cursor move is the SAME code path regardless of where it came from:
 * resolve the state from scratch, then paint. There is no "fast path for +1"
 * and no "undo" path for -1, because a second path is a second chance to
 * disagree with the first.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var TM = NS.traceModel;
  var esc = NS.formula.escapeText;

  var DEFAULTS = {
    mount: '#lab',
    title: '',
    lead: '',
    tier: 'num',
    /* Playback cadence, in ms per step. */
    speed: 900,
    /* Query parameter carrying the step index. */
    param: 'step',
    /* Element to append the narrow-screen fallback to. Defaults to the mount. */
    narrowMount: null,
    tutorial: null,
    /* Optional extra panels rendered after the formula, by id. Each entry is
     * { id, label, render(step, ctx) -> html }. */
    panels: []
  };

  function applyDefaults(config) {
    var out = {};
    Object.keys(DEFAULTS).forEach(function (k) { out[k] = DEFAULTS[k]; });
    Object.keys(config || {}).forEach(function (k) { out[k] = config[k]; });
    return out;
  }

  function parseStepFromUrl(param, lastStep) {
    var raw = null;
    try {
      raw = new URLSearchParams(global.location.search).get(param);
    } catch (err) {
      return 0;
    }
    if (raw === null || raw === '') return 0;
    var n = parseInt(raw, 10);
    if (isNaN(n)) return 0;
    return Math.max(0, Math.min(lastStep, n));
  }

  function formatStepLabel(i, total) {
    return '第 ' + i + ' / ' + total + ' 步';
  }

  /* ============================================================ the player */
  function lab(trace, config) {
    var cfg = applyDefaults(config);
    var idx = TM.index(trace);

    var layoutOpts = {};
    if (cfg.wrapAt) layoutOpts.wrapAt = cfg.wrapAt;
    if (cfg.wrapThreshold !== undefined) layoutOpts.wrapThreshold = cfg.wrapThreshold;
    var L = NS.layout.compute(idx, layoutOpts);

    var mount = typeof cfg.mount === 'string'
      ? document.querySelector(cfg.mount)
      : cfg.mount;
    if (!mount) throw new Error('player: mount element not found: ' + cfg.mount);

    /* Read ?step=N BEFORE the first paint. paint() writes the cursor back into
     * the URL with replaceState, so painting first would overwrite the deep
     * link with ?step=0 and the reader would land on the first step — which is
     * exactly the bug this ordering exists to prevent. */
    var startStep = parseStepFromUrl(cfg.param, idx.lastStep);

    var state = {
      cursor: startStep,
      tier: cfg.tier,
      when: 'after',
      playing: false,
      timer: null,
      skeleton: null,
      skeletonKey: '',
      valueCache: {}
    };

    var dom = buildChrome(mount, cfg, trace, idx, L);

    /* ---- one render path, used by every cursor change and tier change */
    function paint() {
      var i = state.cursor;
      var step = trace.steps[i];
      var after = TM.resolve(idx, i);
      var before = i > 0 ? TM.resolve(idx, i - 1) : emptySnapshot(idx);

      dom.dag.update(i);
      dom.tensorBox.innerHTML = NS.tensors.render(idx, after, before, step, { when: state.when });

      var key = step.id + '|' + state.tier;
      if (state.skeletonKey !== key) {
        state.skeleton = NS.formula.buildSkeleton(step, state.tier, 's' + i + '-', state.valueCache);
        dom.formulaBox.innerHTML = '';
        dom.formulaBox.appendChild(state.skeleton.node);
        state.skeletonKey = key;
      }
      NS.formula.patch(state.skeleton, step, state.tier, state.valueCache);

      dom.narration.innerHTML = '<span class="lab-narration-h">' + esc(step.title) + '</span>' +
        (step.narration ? ' — ' + esc(step.narration) : '');

      renderRegions(dom.regionBox, step);

      dom.scrub.value = String(i);
      dom.pos.textContent = formatStepLabel(i, idx.lastStep);
      dom.stepChips.innerHTML = renderChips(trace, i);
      bindChips(dom.stepChips);
      dom.title.textContent = cfg.title || (trace.meta && trace.meta.title) || 'Lab';

      (cfg.panels || []).forEach(function (panel) {
        var host = dom.panelHosts[panel.id];
        if (host) host.innerHTML = panel.render(step, { cursor: i, snapshot: after, before: before });
      });

      try {
        var url = new URL(global.location.href);
        url.searchParams.set(cfg.param, String(i));
        global.history.replaceState(null, '', url);
      } catch (err) { /* file:// or a blocked history API: the player still works */ }
    }

    function setCursor(i) {
      var next = Math.max(0, Math.min(idx.lastStep, i));
      if (next === state.cursor && state.skeleton) return;
      state.cursor = next;
      paint();
    }

    function setTier(tier) {
      if (NS.formula.TIERS.indexOf(tier) === -1 || tier === state.tier) return;
      state.tier = tier;
      dom.tierButtons.forEach(function (b) {
        b.classList.toggle('lab-on', b.dataset.tier === tier);
        b.setAttribute('aria-pressed', b.dataset.tier === tier ? 'true' : 'false');
      });
      paint();
    }

    function setWhen(when) {
      if (when === state.when) return;
      state.when = when;
      dom.whenButtons.forEach(function (b) {
        b.classList.toggle('lab-on', b.dataset.when === when);
        b.setAttribute('aria-pressed', b.dataset.when === when ? 'true' : 'false');
      });
      paint();
    }

    /* ---- playback */
    function stop() {
      if (state.timer) global.clearInterval(state.timer);
      state.timer = null;
      state.playing = false;
      dom.playBtn.textContent = '▶ 播放';
      dom.playBtn.classList.remove('lab-on');
      dom.playBtn.setAttribute('aria-label', '播放');
    }

    function play() {
      if (state.cursor >= idx.lastStep) setCursor(0);
      state.playing = true;
      dom.playBtn.textContent = '❚❚ 暂停';
      dom.playBtn.classList.add('lab-on');
      dom.playBtn.setAttribute('aria-label', '暂停');
      state.timer = global.setInterval(function () {
        if (state.cursor >= idx.lastStep) { stop(); return; }
        setCursor(state.cursor + 1);
      }, cfg.speed);
    }

    function toggle() { if (state.playing) stop(); else play(); }

    /* ---- keyboard. Registered on the document so the page does not have to be
     * focused, but suppressed while the reader is typing or driving the
     * scrubber. */
    function onKey(ev) {
      var t = ev.target;
      var typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
                         t.tagName === 'SELECT' || t.isContentEditable);
      if (typing && ev.key !== 'Escape') return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      switch (ev.key) {
        case ' ':
        case 'Spacebar':
          /* Without this the page scrolls on every play/pause. */
          ev.preventDefault();
          toggle();
          break;
        case 'ArrowRight': ev.preventDefault(); stop(); setCursor(state.cursor + 1); break;
        case 'ArrowLeft': ev.preventDefault(); stop(); setCursor(state.cursor - 1); break;
        case 'Home': ev.preventDefault(); stop(); setCursor(0); break;
        case 'End': ev.preventDefault(); stop(); setCursor(idx.lastStep); break;
        default: break;
      }
    }
    document.addEventListener('keydown', onKey);

    /* ---- narrow-screen fallback: the player is replaced, not reflowed */
    var narrowHost = cfg.narrowMount
      ? (typeof cfg.narrowMount === 'string' ? document.querySelector(cfg.narrowMount) : cfg.narrowMount)
      : mount;
    var narrowEl = NS.narrow.build(idx, cfg, { initialSnapshot: TM.resolve(idx, 0) });
    narrowHost.parentNode.insertBefore(narrowEl, narrowHost.nextSibling);

    function applyBreakpoint() {
      var narrow = global.innerWidth < NS.narrow.BREAKPOINT;
      narrowEl.hidden = !narrow;
      mount.hidden = narrow;
      if (narrow) stop();
    }
    global.addEventListener('resize', applyBreakpoint);
    applyBreakpoint();

    /* ---- controls */
    dom.playBtn.addEventListener('click', toggle);
    dom.prevBtn.addEventListener('click', function () { stop(); setCursor(state.cursor - 1); });
    dom.nextBtn.addEventListener('click', function () { stop(); setCursor(state.cursor + 1); });
    dom.homeBtn.addEventListener('click', function () { stop(); setCursor(0); });
    dom.endBtn.addEventListener('click', function () { stop(); setCursor(idx.lastStep); });
    dom.scrub.addEventListener('input', function (e) { stop(); setCursor(Number(e.target.value)); });
    dom.tierButtons.forEach(function (b) { b.addEventListener('click', function () { setTier(b.dataset.tier); }); });
    dom.whenButtons.forEach(function (b) { b.addEventListener('click', function () { setWhen(b.dataset.when); }); });

    function bindChips(host) {
      Array.prototype.forEach.call(host.querySelectorAll('[data-step]'), function (chip) {
        chip.addEventListener('click', function () { stop(); setCursor(Number(chip.dataset.step)); });
      });
    }

    paint();

    var api = {
      index: idx,
      dom: dom,
      layout: L,
      state: state,
      setCursor: setCursor,
      setTier: setTier,
      setWhen: setWhen,
      play: play,
      stop: stop,
      toggle: toggle,
      goToUrlStep: function () { setCursor(parseStepFromUrl(cfg.param, idx.lastStep)); },
      paint: paint,
      stats: dom.stats,
      destroy: function () {
        stop();
        document.removeEventListener('keydown', onKey);
        global.removeEventListener('resize', applyBreakpoint);
      }
    };

    dom.apiRef.current = api;
    return api;
  }

  /* ============================================================ chrome */

  function buildChrome(mount, cfg, trace, idx, L) {
    var stats = {
      steps: trace.steps.length,
      nodes: idx.nodes.length,
      edges: idx.edges.length,
      layers: L.numLayers,
      widest: L.widest,
      bands: L.bandCount,
      canvas: Math.round(L.W) + '×' + Math.round(L.H)
    };

    /* Optional extra panels are declared by id in the config and rendered by
     * the caller's own `render`; the engine only owns their box. */
    var extraPanels = (cfg.panels || []).map(function (p) {
      return '<section class="lab-panel lab-panel-extra" data-panel="' + esc(p.id) + '">' +
             '<div class="lab-panel-hd"><span class="lab-panel-t">' + esc(p.label) + '</span></div>' +
             '<div class="lab-panel-bd" data-panel-body="' + esc(p.id) + '"></div></section>';
    }).join('');

    mount.classList.add('lab-root');
    mount.innerHTML =
      '<header class="lab-header">' +
        '<div class="lab-header-main">' +
          '<h1 class="lab-title" id="lab-title"></h1>' +
          (cfg.lead ? '<p class="lab-lead">' + esc(cfg.lead) + '</p>' : '') +
        '</div>' +
        '<div class="lab-header-meta">' +
          '<span class="lab-kbd-hint">键盘 <kbd>←</kbd><kbd>→</kbd> <kbd>空格</kbd> ' +
          '<kbd>Home</kbd>/<kbd>End</kbd></span>' +
        '</div>' +
      '</header>' +

      '<section class="lab-panel lab-panel-dag">' +
        '<div class="lab-panel-hd">' +
          '<span class="lab-panel-t">执行 DAG</span>' +
          '<span class="lab-panel-sub" data-stat="dag"></span>' +
        '</div>' +
        '<div class="lab-dag-host"></div>' +
      '</section>' +

      '<div class="lab-cols">' +
        '<section class="lab-panel">' +
          '<div class="lab-panel-hd">' +
            '<span class="lab-panel-t">张量检查器</span>' +
            '<span class="lab-ctl lab-seg" role="group" aria-label="张量视图时点"></span>' +
          '</div>' +
          '<div class="lab-tensor-host lab-scroll"></div>' +
        '</section>' +

        '<section class="lab-panel">' +
          '<div class="lab-panel-hd">' +
            '<span class="lab-panel-t">公式面板</span>' +
            '<span class="lab-ctl lab-seg" role="group" aria-label="公式档位"></span>' +
          '</div>' +
          '<div class="lab-formula-host lab-scroll"></div>' +
          '<div class="lab-region-host"></div>' +
          '<div class="lab-narration"></div>' +
        '</section>' +
      '</div>' +

      extraPanels +

      '<section class="lab-panel lab-panel-timeline">' +
        '<div class="lab-panel-hd"><span class="lab-panel-t">时间轴</span>' +
          '<span class="lab-panel-sub">任意跳转 = 纯函数 render(trace, i)，无反向操作</span>' +
        '</div>' +
        '<div class="lab-timeline">' +
          '<div class="lab-tl-row">' +
            '<button data-ctl="home" title="Home" aria-label="回到第一步">⏮</button>' +
            '<button data-ctl="prev" title="←" aria-label="上一步">◀</button>' +
            '<button data-ctl="play" title="空格" aria-label="播放">▶ 播放</button>' +
            '<button data-ctl="next" title="→" aria-label="下一步">▶</button>' +
            '<button data-ctl="end" title="End" aria-label="跳到末尾">⏭</button>' +
            '<input type="range" data-ctl="scrub" min="0" max="' + idx.lastStep + '" value="0" step="1" aria-label="时间轴游标">' +
            '<span class="lab-pos" data-stat="pos"></span>' +
          '</div>' +
          '<div class="lab-chips" data-stat="chips"></div>' +
        '</div>' +
      '</section>';

    var segTiers = mount.querySelector('.lab-seg[aria-label="公式档位"]');
    NS.formula.TIERS.forEach(function (t) {
      var label = { sym: '符号式', idx: '代入索引', num: '代入数值' }[t];
      var btn = document.createElement('button');
      btn.dataset.tier = t;
      btn.textContent = label;
      btn.setAttribute('aria-pressed', 'false');
      btn.classList.toggle('lab-on', t === cfg.tier);
      segTiers.appendChild(btn);
    });

    var segWhen = mount.querySelector('.lab-seg[aria-label="张量视图时点"]');
    [['after', '本步结束后'], ['before', '本步开始前']].forEach(function (pair) {
      var btn = document.createElement('button');
      btn.dataset.when = pair[0];
      btn.textContent = pair[1];
      btn.setAttribute('aria-pressed', pair[0] === 'after' ? 'true' : 'false');
      btn.classList.toggle('lab-on', pair[0] === 'after');
      segWhen.appendChild(btn);
    });

    var dagHost = mount.querySelector('.lab-dag-host');
    /* The node handler is bound before the api object exists, so it reaches it
     * through this holder rather than capturing an undefined. Clicking a node
     * then goes through the same setCursor the keyboard and the scrubber use. */
    var apiRef = { current: null };
    var dagView = NS.dag.create(dagHost, idx, L, function (step) {
      if (!apiRef.current) return;
      apiRef.current.stop();
      apiRef.current.setCursor(step);
    });
    mount.querySelector('[data-stat="dag"]').textContent =
      stats.nodes + ' 节点 / ' + stats.edges + ' 边 · ' +
      stats.layers + ' 层' + (stats.bands > 1 ? ' × ' + stats.bands + ' 段' : '') +
      ' · 画布 ' + stats.canvas;
    dagView.update(0);

    var panelHosts = {};
    (cfg.panels || []).forEach(function (p) {
      panelHosts[p.id] = mount.querySelector('[data-panel-body="' + p.id + '"]');
    });

    return {
      dag: dagView,
      dagHost: dagHost,
      tensorBox: mount.querySelector('.lab-tensor-host'),
      formulaBox: mount.querySelector('.lab-formula-host'),
      regionBox: mount.querySelector('.lab-region-host'),
      narration: mount.querySelector('.lab-narration'),
      stepChips: mount.querySelector('[data-stat="chips"]'),
      scrub: mount.querySelector('[data-ctl="scrub"]'),
      pos: mount.querySelector('[data-stat="pos"]'),
      playBtn: mount.querySelector('[data-ctl="play"]'),
      prevBtn: mount.querySelector('[data-ctl="prev"]'),
      nextBtn: mount.querySelector('[data-ctl="next"]'),
      homeBtn: mount.querySelector('[data-ctl="home"]'),
      endBtn: mount.querySelector('[data-ctl="end"]'),
      title: mount.querySelector('#lab-title'),
      tierButtons: Array.prototype.slice.call(segTiers.querySelectorAll('button')),
      whenButtons: Array.prototype.slice.call(segWhen.querySelectorAll('button')),
      panelHosts: panelHosts,
      stats: stats,
      apiRef: apiRef
    };
  }

  function renderRegions(host, step) {
    var regions = step.regions;
    if (!regions || !Object.keys(regions).length) {
      host.classList.remove('lab-show');
      host.innerHTML = '';
      return;
    }
    host.classList.add('lab-show');
    host.innerHTML = Object.keys(regions).map(function (k) {
      return '<div class="lab-region-note"><b>' + esc(k) + '</b> · ' + esc(regions[k]) + '</div>';
    }).join('');
  }

  function renderChips(trace, cursor) {
    return trace.steps.map(function (s, k) {
      var cls = 'lab-chip' + (k === cursor ? ' lab-on' : (k < cursor ? ' lab-done' : '')) +
                (s.kind === 'comm' ? ' lab-chip-comm' : '');
      return '<button class="' + cls + '" data-step="' + k + '" title="' + esc(s.title) + '">' +
             k + '.' + esc(s.id) + '</button>';
    }).join('');
  }

  function emptySnapshot(idx) {
    var out = {};
    idx.tensorNames.forEach(function (n) { out[n] = { value: undefined, from: null, source: 'none' }; });
    return out;
  }

  NS.lab = lab;
  NS._internals = { parseStepFromUrl: parseStepFromUrl, emptySnapshot: emptySnapshot };
})(typeof window !== 'undefined' ? window : globalThis);
