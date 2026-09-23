/* L00 engine · narrow-screen fallback.
 *
 * Below 1024px the player is replaced wholesale, not reflowed. The reason is
 * that a lab's value comes from four things being visible at once — the DAG,
 * the tensor inspector, the formula panel and the timeline. Squeezing those
 * into a phone width produces something that is technically laid out and
 * pedagogically useless, so the honest move is to say so and offer the reader
 * the thing that still works.
 *
 * Every lab gets this from the engine rather than implementing its own, so the
 * breakpoint and the shape of the fallback are identical everywhere. The page
 * supplies only the tutorial link, which is the one part the engine cannot
 * know.
 */
(function (global) {
  'use strict';

  var NS = (global.LabEngine = global.LabEngine || {});
  var esc = function (s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  };

  var BREAKPOINT = 1024;

  /**
   * Build the fallback element.
   *
   * The static content is the trace's own first step: the same formula and the
   * same initial tensor values a reader would see at the start of the replay.
   * That keeps the promise that every number on a lab page comes from the trace
   * — a hand-drawn "overview" graphic would not.
   */
  function build(idx, config, context) {
    var trace = idx.trace;
    var meta = trace.meta || {};
    var cfg = meta.config || {};
    var root = document.createElement('div');
    root.className = 'lab-narrow';
    root.hidden = true;

    var title = config.title || meta.title || '交互实验室';
    var lead = config.lead || '';

    var html = '';
    html += '<h1 class="lab-narrow-title">' + esc(title) + '</h1>';
    if (lead) html += '<p class="lab-narrow-lead">' + esc(lead) + '</p>';

    var facts = [];
    if (cfg.N !== undefined) facts.push('N = ' + cfg.N);
    if (cfg.Bc !== undefined) facts.push('Bc = ' + cfg.Bc);
    if (cfg.n_blocks !== undefined) facts.push('块数 = ' + cfg.n_blocks);
    facts.push('共 ' + trace.steps.length + ' 步');
    if (facts.length) {
      html += '<div class="lab-narrow-facts">' +
        facts.map(function (f) { return '<span class="lab-chip-static">' + esc(f) + '</span>'; }).join('') +
        '</div>';
    }

    /* Build the step-0 formula through the same skeleton + patch path the
     * player uses, so the fallback shows real substituted values rather than
     * the notation. \slot is the trace's own macro, not a KaTeX primitive —
     * rendering the raw string would print the macro names. */
    var first = trace.steps[0];
    var scaffold = NS.formula.buildSkeleton(first, 'num', 'narrow-', {});
    NS.formula.patch(scaffold, first, 'num', {});

    html += '<div class="lab-narrow-card">';
    html += '<div class="lab-narrow-card-hd">第 0 步 · ' +
            esc(first.title || first.id) + '</div>';
    html += '<div class="lab-narrow-formula">' + scaffold.node.innerHTML + '</div>';
    if (trace.steps[0].narration) {
      html += '<p class="lab-narrow-narration">' + esc(trace.steps[0].narration) + '</p>';
    }
    html += '</div>';

    var start = context && context.initialSnapshot;
    if (start) {
      html += '<div class="lab-narrow-card">';
      html += '<div class="lab-narrow-card-hd">初始张量</div>';
      html += '<div class="lab-narrow-tensors">';
      idx.tensorNames.forEach(function (name) {
        var spec = trace.tensors[name] || {};
        var rec = start[name];
        if (!rec || rec.value === undefined) return;
        var value = Array.isArray(rec.value)
          ? '[' + rec.value.map(NS.tensors.formatCell).join(', ') + ']'
          : NS.tensors.formatCell(rec.value);
        html += '<div class="lab-narrow-tensor">';
        html += '<code>' + esc(name) + ' ' + esc(NS.tensors.shapeLabel(spec.shape)) +
                ' @ ' + esc(spec.at || '?') + '</code>';
        html += '<span>' + esc(value) + '</span>';
        html += '</div>';
      });
      html += '</div></div>';
    }

    html += '<p class="lab-narrow-note">这个实验室的完整回放需要更宽的屏幕 —— ' +
            '依赖图、张量网格、公式面板与时间轴要同时可见才有教学效果。' +
            '当前宽度下只展示第一步的静态内容。</p>';

    var tutorial = config.tutorial || {};
    html += '<div class="lab-narrow-actions">';
    if (tutorial.href) {
      html += '<a class="lab-narrow-link" href="' + esc(tutorial.href) + '">← ' +
              esc(tutorial.label || '返回教程正文') + '</a>';
    }
    html += '</div>';

    root.innerHTML = html;
    return root;
  }

  NS.narrow = { BREAKPOINT: BREAKPOINT, build: build };
})(typeof window !== 'undefined' ? window : globalThis);
