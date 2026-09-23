/* Engine · view component: memory ledger.
 *
 * The stacked bar that answers "where did the memory go", in two layouts:
 *
 *   bar()     one-dimensional — segments stacked into one bar, for a single
 *             configuration. This is L05's layout: as the context grows, the
 *             KV Cache block visibly widens while the weights stay put.
 *   matrix()  two-dimensional — segments on one axis, strategies on the other,
 *             one cell per (segment, strategy). This is L14's layout:
 *             baseline/Z1/Z2/Z3 across, 参数/梯度/优化器状态/激活 down, every
 *             cell showing its own byte count.
 *
 * WHAT THIS FILE DELIBERATELY DOES NOT KNOW
 * It does not know the memory formulas. A cost model is *lab* content — L05
 * prices a KV cache, L14 prices ZeRO shards, and a shared component that named
 * one of those keys would be the same mistake `narrow.js` warns about when it
 * refuses to know a lab's config keys. So the accounting arrives as a
 * **model object** supplied by the page:
 *
 *     { id, segments, predict(cfg) -> {segId: bytes}, properties: [...] }
 *
 * `predict` must be a pure function of the config. That is what makes a slider
 * a recomputation rather than an animation, and it is what `check()` tests.
 *
 * THE THREE THINGS THE PAGE ACTUALLY CALLS
 *   ledger.bar(host, opts)          mount a stacked bar; opts.model drives it
 *   ledger.matrix(host, opts)       mount a strategy × segment grid
 *   ledger.panel(host, trace, model) run the self-check and render the verdict
 *
 * THE SELF-CHECK, AND WHY IT CARRIES A CONTROL GROUP
 * A renderer that agrees with its own data proves nothing. `check()` therefore
 * runs three independent things and refuses to report a pass unless all three
 * hold:
 *
 *   1. For every step, the model's prediction for that step's config must equal
 *      the bytes the trace published. The published bytes were computed in
 *      Python from real array sizes (`arr.size`), never from the closed form —
 *      so agreement is a genuine cross-check, not a tautology.
 *   2. The model's declared scaling properties must hold (context doubled →
 *      KV Cache exactly doubled, etc.). These catch a formula that is merely
 *      self-consistent while being wrong, which check 1 cannot see.
 *   3. A deliberately broken copy of the model must FAIL checks 1 or 2. If it
 *      passes, the harness has no discriminating power and the run is reported
 *      as inconclusive rather than green.
 *
 * This is the same shape as `verify.js`'s jump check and lint check, for the
 * same reason.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = function (s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  };

  /* ============================================================ formatting */

  var KIB = 1024;
  var UNITS = [
    { at: KIB * KIB * KIB, suffix: 'GiB' },
    { at: KIB * KIB, suffix: 'MiB' },
    { at: KIB, suffix: 'KiB' }
  ];

  /**
   * Bytes as an exact count plus a human scale, e.g. "210,176 B · 205.2 KiB".
   * The exact number is never dropped: the lab's whole claim is that the figure
   * came out of the formula, and a rounded display would make two different
   * configurations look identical.
   */
  function formatBytes(n, unit) {
    var u = unit || 'B';
    var exact = Number(n).toLocaleString('en-US') + ' ' + u;
    if (n < KIB) return exact;
    for (var i = 0; i < UNITS.length; i++) {
      if (n >= UNITS[i].at) {
        return exact + ' · ' + (n / UNITS[i].at).toFixed(1) + ' ' + UNITS[i].suffix;
      }
    }
    return exact;
  }

  function formatPercent(x) {
    if (x <= 0) return '0%';
    if (x < 0.001) return '<0.1%';
    return (x * 100).toFixed(x < 0.1 ? 1 : 0) + '%';
  }

  /* ============================================================== geometry
   *
   * The one piece of real arithmetic in the renderer, and it is pure: bytes in,
   * laid-out bars out. Keeping it separate from the DOM is what lets `check()`
   * test the thing that is easy to get wrong (a segment dropped, a total that
   * disagrees with its parts) without a browser.
   */
  function stack(segments, bytes, unit) {
    var bars = segments.map(function (seg) {
      var v = bytes && Object.prototype.hasOwnProperty.call(bytes, seg.id) ? bytes[seg.id] : 0;
      return {
        id: seg.id,
        label: seg.label,
        color: seg.color,
        formula: seg.formula,
        bytes: v,
        /* Kept as a fraction of the TOTAL, not of the largest segment: the
         * whole point of a stacked bar is that the widths add up. */
        frac: 0,
        text: formatBytes(v, unit)
      };
    });
    var total = bars.reduce(function (a, b) { return a + b.bytes; }, 0);
    bars.forEach(function (b) { b.frac = total > 0 ? b.bytes / total : 0; });
    return { bars: bars, total: total, unit: unit || 'B' };
  }

  /* The inverse direction: a segment's byte count read back off the geometry.
   * `check()` uses it to prove the bar's widths and the model's prediction are
   * the same number, rather than trusting that they were rendered from it. */
  function bytesFromWidths(stacked, width) {
    var totalWidth = Math.round(width);
    var used = 0;
    var out = {};
    stacked.bars.forEach(function (b, i) {
      var w = i === stacked.bars.length - 1
        ? totalWidth - used                       // last block absorbs the rounding
        : Math.round(b.frac * totalWidth);
      used += w;
      out[b.id] = w;
    });
    return { widths: out, used: used, totalWidth: totalWidth };
  }

  /* ============================================================ 1-D stacked bar
   *
   * opts = {
   *   segments: [{id,label,color,formula}],
   *   unit: 'B',
   *   model: null | {predict(cfg)},     // when set, update() can take a config
   *   baseConfig: {},                   // merged into the config passed to predict
   *   caption: function(bytes, cfg) -> html
   * }
   *
   * Returns { update(bytesOrCfg), bytes(), destroy() }. `update` accepts either
   * a raw {segId: bytes} map (the replay path: the trace already has the
   * numbers) or a config object, in which case the model recomputes.
   */
  function bar(host, opts) {
    var cfg = opts || {};
    var segments = cfg.segments || [];
    var unit = cfg.unit || 'B';
    var state = { bytes: null, config: null, hover: null, last: null };

    var root = document.createElement('div');
    root.className = 'lab-ledger';
    root.innerHTML =
      '<div class="lab-ledger-bar" role="img"></div>' +
      '<div class="lab-ledger-ticks"></div>' +
      '<table class="lab-ledger-table"><thead><tr>' +
        '<th class="lab-ledger-th-seg">分项</th>' +
        '<th class="lab-ledger-th-num">字节数</th>' +
        '<th class="lab-ledger-th-pct">占比</th>' +
      '</tr></thead><tbody></tbody></table>' +
      '<div class="lab-ledger-formula"></div>';
    host.appendChild(root);

    var barEl = root.querySelector('.lab-ledger-bar');
    var ticksEl = root.querySelector('.lab-ledger-ticks');
    var bodyEl = root.querySelector('tbody');
    var formulaEl = root.querySelector('.lab-ledger-formula');

    /* The tick row is the axis. It is labelled in real byte values, derived
     * from the stack it is drawn under — an unlabelled stacked bar invites the
     * reader to guess at proportions, which is the failure mode this component
     * exists to avoid. */
    function paintTicks(stacked) {
      var marks = [0, 0.25, 0.5, 0.75, 1];
      ticksEl.innerHTML = marks.map(function (t) {
        return '<span class="lab-ledger-tick" style="left:' + (t * 100) + '%">' +
               esc(shortBytes(stacked.total * t, unit)) + '</span>';
      }).join('');
    }

    function paint() {
      var stacked = stack(segments, state.bytes, unit);
      state.last = stacked;

      barEl.innerHTML = stacked.bars.map(function (b) {
        /* A zero-width block is not drawn at all: an empty coloured sliver
         * reads as "a little bit of this", which is the opposite of zero. */
        if (b.bytes <= 0) return '';
        return '<span class="lab-ledger-blk' +
               (state.hover === b.id ? ' lab-ledger-blk-on' : '') +
               '" data-seg="' + esc(b.id) + '" style="flex-grow:' + b.bytes +
               ';background:' + esc(b.color) + '" title="' +
               esc(b.label + '：' + b.text) + '"></span>';
      }).join('');
      barEl.setAttribute('aria-label',
        '显存账本：' + stacked.bars.map(function (b) {
          return b.label + ' ' + formatPercent(b.frac);
        }).join('，') + '，合计 ' + formatBytes(stacked.total, unit));
      barEl.classList.toggle('lab-ledger-bar-empty', stacked.total === 0);
      if (stacked.total === 0) {
        barEl.innerHTML = '<span class="lab-ledger-empty">本步没有常驻占用</span>';
      }

      paintTicks(stacked);

      bodyEl.innerHTML = stacked.bars.map(function (b) {
        return '<tr class="lab-ledger-row' +
               (state.hover === b.id ? ' lab-ledger-row-on' : '') +
               '" data-seg="' + esc(b.id) + '">' +
          '<td class="lab-ledger-seg"><span class="lab-ledger-sw" style="background:' +
            esc(b.color) + '"></span>' + esc(b.label) + '</td>' +
          '<td class="lab-ledger-num">' + esc(b.text) + '</td>' +
          '<td class="lab-ledger-pct">' + esc(formatPercent(b.frac)) + '</td>' +
        '</tr>';
      }).join('') +
      '<tr class="lab-ledger-total"><td>合计</td>' +
        '<td class="lab-ledger-num">' + esc(formatBytes(stacked.total, unit)) + '</td>' +
        '<td class="lab-ledger-pct">100%</td></tr>';

      formulaEl.innerHTML = stacked.bars.map(function (b) {
        return '<div class="lab-ledger-fml' +
               (state.hover === b.id ? ' lab-ledger-fml-on' : '') + '">' +
          '<b style="color:' + esc(b.color) + '">' + esc(b.label) + '</b> ' +
          (NS.formula && b.formula ? NS.formula.renderInline(b.formula) : esc(b.formula || '')) +
          '</div>';
      }).join('');

      bindRows();
    }

    function bindRows() {
      Array.prototype.forEach.call(root.querySelectorAll('[data-seg]'), function (el) {
        el.addEventListener('mouseenter', function () {
          state.hover = el.dataset.seg;
          paint();
        });
        el.addEventListener('mouseleave', function () {
          state.hover = null;
          paint();
        });
      });
    }

    function update(input) {
      if (input && cfg.model && looksLikeConfig(input)) {
        var merged = {};
        Object.keys(cfg.baseConfig || {}).forEach(function (k) { merged[k] = cfg.baseConfig[k]; });
        Object.keys(input).forEach(function (k) { merged[k] = input[k]; });
        state.config = merged;
        state.bytes = cfg.model.predict(merged);
      } else {
        state.config = null;
        state.bytes = input || {};
      }
      paint();
      if (cfg.caption) {
        var cap = host.querySelector('.lab-ledger-caption');
        if (!cap) {
          cap = document.createElement('div');
          cap.className = 'lab-ledger-caption';
          host.appendChild(cap);
        }
        cap.innerHTML = cfg.caption(state.bytes, state.config);
      }
      return state.last;
    }

    /* A config object is a flat map of numbers/strings; a bytes map has only
     * the declared segment ids as keys. Distinguishing them this way means the
     * caller can hand `update()` either without a flag. */
    function looksLikeConfig(v) {
      var keys = Object.keys(v);
      if (!keys.length) return false;
      var segIds = segments.map(function (s) { return s.id; });
      return keys.some(function (k) { return segIds.indexOf(k) === -1; });
    }

    paint();
    return {
      update: update,
      bytes: function () { return state.bytes; },
      stacked: function () { return state.last; },
      root: root,
      destroy: function () { if (root.parentNode) root.parentNode.removeChild(root); }
    };
  }

  function shortBytes(n, unit) {
    var u = unit || 'B';
    if (n >= KIB * KIB) return (n / (KIB * KIB)).toFixed(1) + 'M';
    if (n >= KIB) return (n / KIB).toFixed(0) + 'K';
    return Math.round(n) + u;
  }

  /* =========================================================== 2-D matrix
   *
   * opts = {
   *   rows: [{id,label,color,formula}],   // segments — same shape as bar()
   *   cols: [{id,label,note}],            // strategies
   *   unit: 'B',
   *   totalLabel: '单卡合计'
   * }
   *
   * Returns { update(cells), destroy() }. `cells` is
   * `{rowId: {colId: bytes}}`. The column totals are derived here rather than
   * accepted as input — a total that arrives from outside is a second source of
   * truth for a number the reader can already see.
   *
   * Each column also gets a stacked bar, because the matrix's whole point in
   * L14 is watching one bar shrink from baseline to Z3 while another grows.
   */
  function matrix(host, opts) {
    var cfg = opts || {};
    var rows = cfg.rows || [];
    var cols = cfg.cols || [];
    var unit = cfg.unit || 'B';
    var cells = {};

    var root = document.createElement('div');
    root.className = 'lab-ledger lab-ledger-mx-host';
    root.innerHTML = '<table class="lab-ledger-mx"><thead></thead>' +
                     '<tbody></tbody><tfoot></tfoot></table>';
    host.appendChild(root);

    function colTotal(colId) {
      return rows.reduce(function (a, r) {
        var v = (cells[r.id] || {})[colId];
        return a + (typeof v === 'number' ? v : 0);
      }, 0);
    }

    function paint() {
      var totals = cols.map(function (c) { return colTotal(c.id); });
      var max = Math.max.apply(null, totals.concat([1]));

      root.querySelector('thead').innerHTML =
        '<tr><th class="lab-ledger-mx-corner">分项 \\ 策略</th>' +
        cols.map(function (c) {
          return '<th><span class="lab-ledger-mx-col">' + esc(c.label) + '</span>' +
                 (c.note ? '<span class="lab-ledger-mx-note">' + esc(c.note) + '</span>' : '') +
                 '</th>';
        }).join('') + '</tr>';

      root.querySelector('tbody').innerHTML = rows.map(function (r) {
        return '<tr><th class="lab-ledger-mx-row">' +
            '<span class="lab-ledger-sw" style="background:' + esc(r.color) + '"></span>' +
            esc(r.label) + '</th>' +
          cols.map(function (c) {
            var v = (cells[r.id] || {})[c.id];
            var t = max > 0 && typeof v === 'number' ? v / max : 0;
            /* Intensity carries the magnitude and the number carries the
             * value. Neither alone is enough: a bare tint is unreadable at
             * several candidates, and a bare number grid hides the shape that
             * is the reason to use a matrix at all. */
            var style = t > 0.002
              ? ' style="background:color-mix(in srgb, ' + esc(r.color) + ' ' +
                (8 + t * 62).toFixed(1) + '%, transparent)"'
              : '';
            return '<td class="lab-ledger-mx-cell"' + style + '>' +
                   (typeof v === 'number' ? esc(shortBytes(v, unit)) : '—') +
                   '<span class="lab-ledger-mx-exact">' +
                   (typeof v === 'number' ? esc(Number(v).toLocaleString('en-US')) : '') +
                   '</span></td>';
          }).join('') + '</tr>';
      }).join('');

      root.querySelector('tfoot').innerHTML =
        '<tr class="lab-ledger-mx-bars"><th class="lab-ledger-mx-row">' +
          esc(cfg.totalLabel || '单卡合计') + '</th>' +
        cols.map(function (c, i) {
          var stacked = stack(rows, cellsForColumn(c.id), unit);
          return '<td class="lab-ledger-mx-bar">' +
            '<div class="lab-ledger-bar lab-ledger-bar-sm">' +
            stacked.bars.map(function (b) {
              return b.bytes > 0
                ? '<span style="flex-grow:' + b.bytes + ';background:' + esc(b.color) + '"></span>'
                : '';
            }).join('') + '</div>' +
            '<span class="lab-ledger-mx-total">' + esc(formatBytes(totals[i], unit)) + '</span>' +
            '</td>';
        }).join('') + '</tr>';
    }

    function cellsForColumn(colId) {
      var out = {};
      rows.forEach(function (r) {
        var v = (cells[r.id] || {})[colId];
        if (typeof v === 'number') out[r.id] = v;
      });
      return out;
    }

    function update(next) {
      cells = next || {};
      paint();
      return { totals: cols.map(function (c) { return colTotal(c.id); }) };
    }

    paint();
    return { update: update, root: root, colTotal: colTotal,
             destroy: function () { if (root.parentNode) root.parentNode.removeChild(root); } };
  }

  /* ================================================================ lint port
   *
   * The author-side half of the ledger contract lives in the trace generator
   * (`labs/traces/kv_cache.py` → `lint_ledger`). This is the port, so a lab
   * author editing a trace in the browser gets the same verdict without a
   * Python round trip. **The two must agree; when a rule changes it changes in
   * both.** Each rule below is exercised by the matching entry in
   * `SABOTAGE_CASES`, and the Python side by its own `SABOTAGE_CASES` — a rule
   * that only exists on one side is a rule that is not tested on that side.
   *
   * It lives here rather than in `trace-model.js` because the ledger block is
   * this file's subject; `trace-model.js` was frozen when the engine landed.
   */
  var COLOR_RE = /^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$/;

  function isNonNegInt(v) {
    return typeof v === 'number' && isFinite(v) && v >= 0 && Math.floor(v) === v;
  }

  function lint(trace) {
    var gaps = [], warns = [], infos = [];
    var steps = (trace && trace.steps) || [];
    var withLedger = steps.filter(function (s) { return s && s.ledger !== undefined; });
    var block = trace && trace.ledger;

    if (!block) {
      if (withLedger.length) {
        gaps.push({
          what: 'trace 没有 ledger 块，但 ' + withLedger.length + ' 个步骤带了 step.ledger',
          why: '分项没有声明，组件只能显示一堆没有名字的数字。'
        });
      }
      return { gaps: gaps, warns: warns, infos: infos, present: false };
    }

    if (typeof block.unit !== 'string' || !block.unit) {
      gaps.push({
        what: 'ledger.unit 缺失',
        why: '分项数值的单位没有声明，组件无法标注。'
      });
    }

    var segments = block.segments;
    if (!Array.isArray(segments) || !segments.length) {
      gaps.push({ what: 'ledger.segments 缺失或为空', why: '至少要有一个分项。' });
      return { gaps: gaps, warns: warns, infos: infos, present: true };
    }

    var ids = [];
    segments.forEach(function (seg) {
      var sid = seg && seg.id;
      if (typeof sid !== 'string' || !sid) {
        gaps.push({ what: 'ledger.segments 里有分项缺少 id：' + JSON.stringify(seg),
                    why: '没有 id 就无法与 step.ledger.bytes 对应。' });
        return;
      }
      ids.push(sid);
      if (typeof seg.label !== 'string' || !seg.label) {
        gaps.push({ what: 'ledger 分项 "' + sid + '" 缺 label', why: '图例上会是一片空白。' });
      }
      if (!COLOR_RE.test(String(seg.color || ''))) {
        gaps.push({
          what: 'ledger 分项 "' + sid + '" 的 color=' + JSON.stringify(seg.color) +
                ' 不是 #rgb / #rrggbb',
          why: 'CSS 会静默丢弃它，色块变成透明 —— 没有报错，只是画不出来。'
        });
      }
      if (typeof seg.formula !== 'string' || !seg.formula) {
        gaps.push({
          what: 'ledger 分项 "' + sid + '" 缺 formula',
          why: '数值就无法追溯到显存公式，而这正是本组件存在的理由。'
        });
      }
    });

    var seen = {};
    ids.forEach(function (id) {
      if (seen[id]) {
        gaps.push({ what: 'ledger.segments 有重复 id："' + id + '"',
                    why: '两个分项会争抢同一个字节数。' });
      }
      seen[id] = true;
    });

    var declared = {};
    ids.forEach(function (i) { declared[i] = true; });

    steps.forEach(function (s) {
      var sid = s.id;
      var led = s.ledger;
      if (led === undefined) {
        gaps.push({
          what: '步骤 "' + sid + '" 没有 ledger',
          why: '「每个分项都要有值」是账本能画成堆叠条的前提。'
        });
        return;
      }
      if (typeof led !== 'object' || led === null) {
        gaps.push({ what: '步骤 "' + sid + '" 的 ledger 不是对象', why: '无法解析。' });
        return;
      }
      var bytes = led.bytes;
      if (typeof bytes !== 'object' || bytes === null) {
        gaps.push({ what: '步骤 "' + sid + '" 的 ledger 缺 bytes', why: '没有数值可画。' });
        return;
      }
      Object.keys(declared).forEach(function (k) {
        if (!Object.prototype.hasOwnProperty.call(bytes, k)) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ledger 少了分项 "' + k + '"',
                      why: '堆叠条会缺一块，但总数却仍在，读者无从发现。' });
        }
      });
      Object.keys(bytes).forEach(function (k) {
        if (!declared[k]) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ledger 有未声明的分项 "' + k + '"',
                      why: '它不会出现在图例里，却会进合计。' });
        }
        if (!isNonNegInt(bytes[k])) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ledger.bytes["' + k + '"]=' +
                            JSON.stringify(bytes[k]),
                      why: '字节数必须是非负整数。' });
        }
      });

      var c = led.config;
      if (typeof c !== 'object' || c === null) {
        gaps.push({
          what: '步骤 "' + sid + '" 的 ledger 缺 config',
          why: '没有它就无法独立重算这一步的字节数，对拍会退化成自证。'
        });
        return;
      }
      ['phase', 'n_q', 'layers', 'seq_len', 'kv_rows_total'].forEach(function (k) {
        if (!Object.prototype.hasOwnProperty.call(c, k)) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ledger.config 缺 "' + k + '"',
                      why: '重算这一步需要它。' });
        }
      });
      ['n_q', 'layers', 'seq_len', 'kv_rows_total'].forEach(function (k) {
        if (!isNonNegInt(c[k])) {
          gaps.push({ what: '步骤 "' + sid + '" 的 ledger.config["' + k + '"]=' +
                            JSON.stringify(c[k]),
                      why: '必须是非负整数。' });
        }
      });
      /* The cache holds one row per layer per position, so the two counts in
       * `config` cannot disagree. Without this, `kv_rows_total` could be half
       * what it should be and every downstream check would still pass — they
       * all take it as given. */
      if (isNonNegInt(c.layers) && c.layers >= 1 &&
          isNonNegInt(c.seq_len) && isNonNegInt(c.kv_rows_total)) {
        if (c.kv_rows_total !== c.layers * c.seq_len) {
          gaps.push({
            what: '步骤 "' + sid + '" 的 ledger.config 自相矛盾：kv_rows_total=' +
                  c.kv_rows_total + ' 但 layers×seq_len=' + (c.layers * c.seq_len),
            why: '每层每个位置恰好一行，这两个数必须相等。'
          });
        }
      }
    });

    infos.push({
      what: '账本分项：' + ids.join(' / ') + '（单位 ' + (block.unit || '?') + '）',
      why: ''
    });
    if (steps.length) {
      var last = steps[steps.length - 1].ledger;
      if (last && last.bytes) {
        infos.push({
          what: '末步合计 ' + formatBytes(Object.keys(last.bytes).reduce(function (a, k) {
            return a + last.bytes[k];
          }, 0), block.unit),
          why: ''
        });
      }
    }
    return { gaps: gaps, warns: warns, infos: infos, present: true };
  }

  /* Each entry breaks one rule above, on the real trace. The names deliberately
   * parallel the Python `SABOTAGE_CASES` in `labs/traces/kv_cache.py`, so a
   * reader can match them up; a rule with no entry here is a rule this port
   * does not actually test. */
  var SABOTAGE_CASES = {
    'ledger 块被删掉但步骤还留着': function (t) { delete t.ledger; },
    'ledger.unit 缺失': function (t) { delete t.ledger.unit; },
    '分项 id 重复': function (t) {
      t.ledger.segments.push(JSON.parse(JSON.stringify(t.ledger.segments[0])));
    },
    '分项颜色不是 CSS 颜色': function (t) { t.ledger.segments[0].color = 'cornflowerblue'; },
    '分项缺 formula': function (t) { delete t.ledger.segments[0].formula; },
    '分项缺 label': function (t) { delete t.ledger.segments[0].label; },
    '分项缺 id': function (t) { delete t.ledger.segments[0].id; },
    'segments 为空': function (t) { t.ledger.segments = []; },
    '某一步没有 ledger': function (t) { delete t.steps[2].ledger; },
    '某一步的 ledger 少一个分项': function (t) { delete t.steps[2].ledger.bytes.kv_cache; },
    '某一步的 ledger 多一个分项': function (t) { t.steps[2].ledger.bytes.ghost = 1; },
    '字节数为负': function (t) { t.steps[2].ledger.bytes.params = -1; },
    '字节数是小数': function (t) { t.steps[2].ledger.bytes.params = 1.5; },
    '字节数是字符串': function (t) { t.steps[2].ledger.bytes.params = '210176'; },
    'ledger.config 缺失': function (t) { delete t.steps[2].ledger.config; },
    'ledger.config 缺一个键': function (t) { delete t.steps[2].ledger.config.kv_rows_total; },
    'ledger.config 缺 layers': function (t) { delete t.steps[2].ledger.config.layers; },
    'ledger.config 的 seq_len 不是整数': function (t) { t.steps[2].ledger.config.seq_len = '5'; },
    'kv_rows_total 与 layers×seq_len 不符': function (t) {
      t.steps[2].ledger.config.kv_rows_total = Math.floor(
        t.steps[2].ledger.config.kv_rows_total / 2);
    },
    'ledger.bytes 不是对象': function (t) { t.steps[2].ledger.bytes = [1, 2, 3]; }
  };

  /**
   * Prove the lint is not a function that always returns zero. Returns the list
   * of mutations it failed to catch.
   */
  function sabotageChecks(trace) {
    var missed = [];
    Object.keys(SABOTAGE_CASES).forEach(function (name) {
      var copy = JSON.parse(JSON.stringify(trace));
      try {
        SABOTAGE_CASES[name](copy);
      } catch (err) {
        missed.push(name + '（破坏本身失败：' + err.message + '）');
        return;
      }
      var r = lint(copy);
      if (!r.gaps.length && !r.warns.length) missed.push(name);
    });
    return missed;
  }

  /* ============================================================ model check
   *
   * A model is `{ id, segments, predict(cfg), properties }`:
   *
   *   predict(cfg) -> {segId: bytes}     pure; cfg is a flat map
   *   properties   -> [{name, run(predict) -> {ok, detail}}]
   *
   * `check()` runs the model against a trace and reports all three verdicts
   * (published bytes, scaling properties, control group). It never returns
   * `passed: true` unless the control group failed, because a harness whose
   * broken copy also passes is a harness that is not measuring anything.
   */
  function check(trace, model) {
    var rows = [];
    var failed = 0;

    /* --- 1. every published step, predicted from its own config -------------
     * The trace's bytes came out of Python's `arr.size` over real arrays; the
     * model derives them from a shape formula. Agreement is a cross-check.
     */
    var steps = (trace && trace.steps) || [];
    steps.forEach(function (s) {
      if (!s.ledger) return;
      var cfg = stepConfig(trace, s);
      var got, want = s.ledger.bytes, mismatch;
      try {
        got = model.predict(cfg);
        mismatch = Object.keys(want).filter(function (k) { return got[k] !== want[k]; });
      } catch (err) {
        /* A model that throws on a config the trace actually ran is a failure
         * of the model, and has to be reported as one. Letting it propagate
         * would take the whole panel down and turn a red verdict into no
         * verdict at all. */
        mismatch = ['抛出 ' + err.message];
      }
      if (mismatch.length) failed++;
      rows.push({
        label: s.id,
        ok: mismatch.length === 0,
        detail: mismatch.map(function (k) {
          return typeof k === 'string' && k.indexOf('抛出') === 0
            ? k
            : k + ' 预测=' + got[k] + ' 实算=' + want[k];
        }).join('；')
      });
    });

    /* --- 2. the declared scaling properties -------------------------------- */
    var propRows = (model.properties || []).map(function (p) {
      var r;
      try {
        r = p.run(model.predict);
      } catch (err) {
        r = { ok: false, detail: '抛出 ' + err.message };
      }
      if (!r.ok) failed++;
      return { label: p.name, ok: !!r.ok, detail: r.detail || '' };
    });

    /* --- 3. the control group --------------------------------------------- */
    /* Each broken model is wrong in a way that is easy to write by accident:
     * a missing factor of 2 (K and V), a missing layer count. Both are caught
     * by the per-step check even though both satisfy every scaling property —
     * which is exactly why the two checks are not redundant. */
    var CONTROLS = {
      'KV Cache 漏掉 K/V 两份中的一份（少乘 2）': function (cfg) {
        var b = model.predict(cfg);
        return shallow(b, { kv_cache: Math.round(b.kv_cache / 2) });
      },
      'KV Cache 漏掉层数因子（只算一层）': function (cfg) {
        var b = model.predict(cfg);
        return shallow(b, { kv_cache: Math.round(b.kv_cache / Math.max(cfg.layers, 1)) });
      },
      'KV Cache 把层数算了两次（L 与 N 都含 L）': function (cfg) {
        var b = model.predict(cfg);
        return shallow(b, { kv_cache: b.kv_cache * Math.max(cfg.layers, 1) });
      },
      '参数按 fp16 计价错成 fp32（多乘 2）': function (cfg) {
        var b = model.predict(cfg);
        return shallow(b, { params: b.params * 2 });
      }
    };
    var controlRows = Object.keys(CONTROLS).map(function (name) {
      var broken = { predict: CONTROLS[name] };
      var caughtBySteps = 0;
      steps.forEach(function (s) {
        if (!s.ledger) return;
        var got;
        try {
          got = broken.predict(stepConfig(trace, s));
        } catch (err) {
          caughtBySteps++;   // throwing on a config the trace ran is caught too
          return;
        }
        var mismatch = Object.keys(s.ledger.bytes).some(function (k) {
          return got[k] !== s.ledger.bytes[k];
        });
        if (mismatch) caughtBySteps++;
      });
      var caughtByProps = (model.properties || []).filter(function (p) {
        var r;
        try { r = p.run(broken.predict); } catch (err) { return true; }
        return !r.ok;
      }).length;
      return {
        label: name,
        caught: caughtBySteps > 0 || caughtByProps > 0,
        detail: '逐步对拍抓到 ' + caughtBySteps + '/' + steps.length + ' 步，' +
                '缩放性质抓到 ' + caughtByProps + '/' + (model.properties || []).length + ' 条'
      };
    });
    var missedControls = controlRows.filter(function (r) { return !r.caught; });

    var lintResult = lint(trace);
    var missedSabotage = sabotageChecks(trace);

    return {
      steps: rows,
      stepFailures: failed,
      properties: propRows,
      propertyFailures: propRows.filter(function (r) { return !r.ok; }).length,
      controls: controlRows,
      missedControls: missedControls,
      lint: lintResult,
      lintSabotageMissed: missedSabotage,
      sabotageTotal: Object.keys(SABOTAGE_CASES).length,
      /* Conclusive *and* green, or not green at all. */
      passed: failed === 0 && missedControls.length === 0 &&
              lintResult.gaps.length === 0 && missedSabotage.length === 0
    };
  }

  /**
   * The config a step ran under: the trace's global config with the step's own
   * ledger config laid over it. The step wins, because a step's config is what
   * the bytes were actually counted with.
   */
  function stepConfig(trace, s) {
    var cfg = {};
    var meta = (trace.meta && trace.meta.config) || {};
    Object.keys(meta).forEach(function (k) { cfg[k] = meta[k]; });
    Object.keys(s.ledger.config || {}).forEach(function (k) { cfg[k] = s.ledger.config[k]; });
    return cfg;
  }

  function shallow(base, over) {
    var out = {};
    Object.keys(base).forEach(function (k) { out[k] = base[k]; });
    Object.keys(over).forEach(function (k) { out[k] = over[k]; });
    return out;
  }

  /* ============================================================== the panel
   *
   * The page mounts this once; the acceptance harness reads `window.__labLedger`
   * for the same numbers rather than scraping the DOM.
   */
  function panel(host, trace, model) {
    var r = check(trace, model);
    var html = '';

    html += '<div class="lab-verdict ' + (r.passed ? 'lab-ok' : 'lab-bad') + '">';
    html += '<b>账本数值 · 逐步对拍：</b>' + r.steps.length + ' 步，' +
            (r.stepFailures === 0
              ? '<b>0 步不一致</b>（每一格都由 shape 公式从该步 config 重算，' +
                '与 trace 里的实算字节数逐位相同）。'
              : '<b>' + r.stepFailures + ' 步不一致</b>。');
    html += '<br><b>缩放性质：</b>' +
            (r.propertyFailures === 0
              ? (model.properties || []).length + ' 条全部成立（这些是公式的性质，' +
                '不是某一次求值的结果 —— 自洽但错误的公式过不了）。'
              : '<b>' + r.propertyFailures + ' 条不成立</b>。');
    html += '<br><b>对照组（故意写错的模型）：</b> ' + r.controls.length + ' 种，' +
            (r.missedControls.length === 0
              ? '全部被抓到 —— 所以上面的 0 不是对拍写错。'
              : '<b>漏掉 ' + r.missedControls.length + ' 种，这组对拍没有区分力。</b>');
    html += '</div>';

    html += '<table class="lab-vtable"><thead><tr>' +
            '<th>逐步对拍</th><th>结果</th><th>说明</th></tr></thead><tbody>';
    r.steps.slice(0, 8).forEach(function (x) {
      html += '<tr><td>' + esc(x.label) + '</td>' +
        '<td class="' + (x.ok ? 'lab-pass' : 'lab-fail') + '">' +
          (x.ok ? '一致' : '不一致') + '</td>' +
        '<td class="lab-sub">' + esc(x.detail) + '</td></tr>';
    });
    if (r.steps.length > 8) {
      html += '<tr><td class="lab-sub" colspan="3">…其余 ' + (r.steps.length - 8) +
              ' 步全部一致（每一步四个子阶段都算）</td></tr>';
    }
    html += '</tbody></table>';

    html += '<details class="lab-vdetails" open><summary>缩放性质与对照组明细</summary>' +
            '<table class="lab-vtable"><thead><tr><th>检查</th><th>结果</th><th>说明</th></tr></thead><tbody>';
    r.properties.forEach(function (p) {
      html += '<tr><td>' + esc(p.label) + '</td>' +
        '<td class="' + (p.ok ? 'lab-pass' : 'lab-fail') + '">' + (p.ok ? '成立' : '不成立') +
        '</td><td class="lab-sub">' + esc(p.detail) + '</td></tr>';
    });
    r.controls.forEach(function (c) {
      html += '<tr><td>' + esc(c.label) + '</td>' +
        '<td class="' + (c.caught ? 'lab-pass' : 'lab-fail') + '">' +
        (c.caught ? '被抓到' : '漏掉') + '</td>' +
        '<td class="lab-sub">' + esc(c.detail) + '</td></tr>';
    });
    html += '</tbody></table></details>';

    html += '<div class="lab-verdict ' +
            (r.lint.gaps.length === 0 && !r.lintSabotageMissed.length ? 'lab-ok' : 'lab-bad') + '">' +
            '<b>账本契约 lint：</b>干净的 trace 报 <b>' + r.lint.gaps.length + '</b> 个 gap；' +
            '对照组 ' + r.sabotageTotal + ' 种破坏' +
            (r.lintSabotageMissed.length
              ? '，<b>漏掉 ' + r.lintSabotageMissed.length + ' 种：' +
                esc(r.lintSabotageMissed.join('、')) + '</b>'
              : '全部被抓到') +
            (r.lintSabotageMissed.length ? '。' : '。lint 是活的，所以上面的 0 有意义。') +
            '</div>';

    html += '<details class="lab-vdetails"><summary>lint 明细（前 12 条）</summary>' +
            '<table class="lab-vtable"><tbody>';
    r.lint.gaps.concat(r.lint.warns, r.lint.infos).slice(0, 12).forEach(function (x) {
      html += '<tr><td>' + esc(x.what) + '</td><td class="lab-sub">' + esc(x.why) + '</td></tr>';
    });
    html += '</tbody></table></details>';

    host.innerHTML = html;

    var result = {
      passed: r.passed,
      steps: r.steps.length,
      stepFailures: r.stepFailures,
      properties: (model.properties || []).length,
      propertyFailures: r.propertyFailures,
      controls: r.controls.length,
      missedControls: r.missedControls.map(function (c) { return c.label; }),
      lintGaps: r.lint.gaps.length,
      lintSabotageTotal: r.sabotageTotal,
      lintSabotageMissed: r.lintSabotageMissed
    };
    global.__labLedger = global.__labLedger || {};
    global.__labLedger[(trace.meta && trace.meta.lab) || 'lab'] = result;
    return result;
  }

  NS.ledger = {
    bar: bar,
    matrix: matrix,
    stack: stack,
    bytesFromWidths: bytesFromWidths,
    formatBytes: formatBytes,
    formatPercent: formatPercent,
    lint: lint,
    sabotageChecks: sabotageChecks,
    SABOTAGE_CASES: SABOTAGE_CASES,
    check: check,
    panel: panel
  };
})(typeof window !== 'undefined' ? window : globalThis);
