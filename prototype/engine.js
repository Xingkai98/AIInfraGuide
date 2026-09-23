/* PROTOTYPE engine for issue #12 — throwaway.
 *
 * One core idea is being tested: render(trace, i) is a PURE FUNCTION of
 * (trace, cursor). No accumulation, no undo log, no "previous step" needed.
 * Everything you see below is recomputed from scratch on every jump.
 *
 * The other three things being tested (contract lint, KaTeX tiers, DAG layout)
 * are bolted on around that core.
 */
'use strict';

const T = window.PROTO_TRACE;
const $ = (id) => document.getElementById(id);
const SVGNS = 'http://www.w3.org/2000/svg';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

/* ============================================================ STATE (derived)
 *
 * resolveState() is the whole answer to "does arbitrary jumping work?".
 * For each tensor: scan every step and keep the LAST write at or before the
 * cursor. That is the "provenance fold". It is O(steps) per render and needs
 * no mutation, so jumping backwards is free.
 */
function writesOf(trace, upto) {
  // tensor -> {stepIdx, value}; last writer wins
  const out = {};
  for (let i = 0; i <= upto; i++) {
    const st = trace.steps[i];
    for (const [k, v] of Object.entries(st.state || {})) {
      out[k] = {from: i, value: v};
    }
  }
  return out;
}

function resolveState(trace, upto) {
  const w = writesOf(trace, upto);
  const out = {};
  for (const [name, spec] of Object.entries(trace.tensors)) {
    if (spec.init !== undefined) out[name] = {value: spec.init, from: -1, source: 'init'};
    else if (w[name]) out[name] = {value: w[name].value, from: w[name].from, source: 'write'};
    else out[name] = {value: undefined, from: null, source: 'none'};
  }
  // a value written by a step always beats the static init
  for (const [name, rec] of Object.entries(w)) {
    out[name] = {value: rec.value, from: rec.from, source: 'write'};
    if (!trace.tensors[name]) out[name].stray = true;
  }
  return out;
}

/* ==================================================== 1. DAG (self-layout)
 *
 * Hand-rolled layered layout. Layer = longest-path depth (Kahn topological
 * sort, layer[n] = max(layer[preds]) + 1). Column = index within layer.
 * Node x = layer * XGAP, y = spread within the layer's column.
 */
const NG = {w: 78, h: 27, xgap: 104, ygap: 34, pad: 18, wrapAt: 0, seqEdges: false};

function layout(trace) {
  const nodes = trace.graph.nodes;
  const byId = {};
  nodes.forEach((n) => { byId[n.id] = n; });
  const preds = {}, succs = {};
  nodes.forEach((n) => { preds[n.id] = []; succs[n.id] = []; });
  const allEdges = trace.graph.edges.slice();
  if (NG.seqEdges) {
    // Synthetic sequence edges (step i -> step i+1 in trace order). The data
    // edges alone leave a step with no producing predecessor -- e.g. a `load`
    // whose only input is a tensor nothing writes -- stuck at layer 0, which
    // inflates the widest layer and the canvas height. See the report.
    for (let i = 0; i + 1 < nodes.length; i++) {
      allEdges.push({from: nodes[i].id, to: nodes[i + 1].id, tensor: '', seq: true});
    }
  }
  for (const e of allEdges) {
    if (!byId[e.from] || !byId[e.to]) continue;
    succs[e.from].push(e.to);
    preds[e.to].push(e.from);
  }
  const indeg = {};
  nodes.forEach((n) => { indeg[n.id] = preds[n.id].length; });
  const layer = {};
  nodes.forEach((n) => { layer[n.id] = 0; });
  const q = nodes.filter((n) => indeg[n.id] === 0).map((n) => n.id);
  const seen = new Set(q);
  while (q.length) {
    const id = q.shift();
    for (const s of succs[id]) {
      layer[s] = Math.max(layer[s], layer[id] + 1);
      if (--indeg[s] === 0 && !seen.has(s)) { seen.add(s); q.push(s); }
    }
  }
  // cycles would leave nodes unvisited; they just keep layer 0 and get flagged
  const cols = {};
  const pos = {};
  for (const n of nodes) {
    const L = layer[n.id];
    (cols[L] = cols[L] || []).push(n.id);
  }
  const maxCol = Math.max(0, ...Object.values(cols).map((c) => c.length));
  const maxLayer = Math.max(...Object.values(layer));
  // A single-pass layered layout is one-dimensional: width grows with depth,
  // height grows with the widest layer. For a chain like online-softmax that
  // gives ~8:1 (measured below), which is unreadable in a normal viewport.
  // `wrapAt` breaks the column axis into rows of N layers -- the cheap fix.
  const wrap = NG.wrapAt || (maxLayer + 1);
  const rowsPerBand = Math.ceil((maxLayer + 1) / wrap);
  for (const [L, ids] of Object.entries(cols)) {
    const total = ids.length;
    const span = (total - 1) * (NG.h + NG.ygap);
    const band = Math.floor(Number(L) / wrap);
    const colInBand = Number(L) % wrap;
    // vertical offset accumulated from earlier bands
    const bandTop = NG.pad + band * (Math.max(0, (maxCol - 1) * (NG.h + NG.ygap)) + NG.h + 2 * NG.ygap);
    const top = bandTop + (Math.max(0, (maxCol - 1) * (NG.h + NG.ygap)) - span) / 2;
    ids.forEach((id, k) => {
      pos[id] = {x: NG.pad + colInBand * (NG.w + NG.xgap),
                 y: top + k * (NG.h + NG.ygap), layer: Number(L), band};
    });
  }
  const bandW = Math.min(wrap, maxLayer + 1);
  const W = NG.pad * 2 + bandW * NG.w + Math.max(0, bandW - 1) * NG.xgap;
  const H = NG.pad * 2 + rowsPerBand * (Math.max(maxCol, 1) * NG.h
            + Math.max(0, maxCol - 1) * NG.ygap)
            + Math.max(0, rowsPerBand - 1) * 2 * NG.ygap;
  return {pos, layer, W, H, cols, bandW, rowsPerBand};
}

let EDGE_MODE = 'arc';

function drawDag(trace, cursor, opts) {
  const L = opts.layout;
  const svg = $('dag');
  svg.setAttribute('viewBox', `0 0 ${L.W} ${L.H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  const cur = trace.graph.nodes[cursor] ? trace.graph.nodes[cursor].id : null;
  const curStep = cursor;

  let h = `<defs><marker id="arw" viewBox="0 0 8 6" refX="7.5" refY="3"
    markerWidth="7" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L8,3 L0,6 z" fill="#586379"/></marker>
    <marker id="arwc" viewBox="0 0 8 6" refX="7.5" refY="3"
    markerWidth="7" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L8,3 L0,6 z" fill="#4ea1ff"/></marker></defs>`;

  // ---- edges first so nodes paint over them
  for (const e of trace.graph.edges) {
    const a = L.pos[e.from], c = L.pos[e.to];
    if (!a || !c) continue;
    const fromStep = byIdStep(trace, e.from), toStep = byIdStep(trace, e.to);
    const cls = toStep === curStep ? 'current' : (toStep < curStep ? 'visited' : '');
    const x1 = a.x + NG.w, y1 = a.y + NG.h / 2;
    const x2 = c.x, y2 = c.y + NG.h / 2;
    let d;
    if (c.band !== a.band) {
      // wrap edge: leave the right edge of the band, re-enter the left of the
      // next one, routed outside the node columns so it never crosses a box
      const ex = a.x + NG.w, ey = a.y + NG.h / 2;
      const sx = c.x, sy = c.y + NG.h / 2;
      const right = NG.pad + L.bandW * (NG.w + NG.xgap) - NG.xgap / 2;
      d = `M${ex},${ey} L${right},${ey} L${right},${sy - NG.h - NG.ygap / 2} `
        + `L${sx - NG.xgap / 2},${sy - NG.h - NG.ygap / 2} L${sx - NG.xgap / 2},${sy} L${sx},${sy}`;
    } else if (EDGE_MODE === 'arc' && c.layer > a.layer + 1) {
      // long edge: bow it above the row so it doesn't cut through nodes
      const mx = (x1 + x2) / 2, my = y1 - 22;
      d = `M${x1},${y1} Q${mx},${my} ${x2},${y2}`;
    } else {
      const mx = (x1 + x2) / 2;
      d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
    }
    h += `<path class="eg ${cls}" d="${d}" ${cls === 'current'
      ? 'marker-end="url(#arwc)"' : ''}></path>`;
    h += `<text class="eglbl ${cls}" x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 3}">${esc(e.tensor || '')}</text>`;
  }

  // ---- nodes
  for (const n of trace.graph.nodes) {
    const p = L.pos[n.id];
    if (!p) continue;
    const st = n.step;
    const cls = st === curStep ? 'current' : (st < curStep ? 'visited' : 'future');
    h += `<g class="nd kind-${esc(n.kind || 'op')} ${cls}" data-step="${st}" tabindex="0"
        role="button" aria-label="${esc(n.title)}">`;
    h += `<rect class="halo" x="${p.x - 2.5}" y="${p.y - 2.5}" width="${NG.w + 5}" height="${NG.h + 5}"></rect>`;
    h += `<rect x="${p.x}" y="${p.y}" width="${NG.w}" height="${NG.h}"></rect>`;
    h += `<circle class="kinddot" cx="${p.x + 7}" cy="${p.y + 7}" r="2.4"></circle>`;
    h += `<text x="${p.x + NG.w / 2 + 3}" y="${p.y + 12}">${esc(n.id)}</text>`;
    h += `<text class="sub" x="${p.x + NG.w / 2 + 3}" y="${p.y + 21}">${esc(n.op)}</text>`;
    h += `</g>`;
  }
  svg.innerHTML = h;

  svg.querySelectorAll('.nd').forEach((g) => {
    const go = () => setCursor(Number(g.dataset.step));
    g.addEventListener('click', go);
    g.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); go(); }
    });
  });
  const ar = (L.W / L.H).toFixed(2);
  $('dagstat').textContent =
    `${trace.graph.nodes.length} 节点 / ${trace.graph.edges.length} 边 · 分层 ${Object.keys(L.cols).length} 列`
    + (L.rowsPerBand > 1 ? ` × ${L.rowsPerBand} 段` : '')
    + ` · 画布 ${Math.round(L.W)}×${Math.round(L.H)} (${ar}:1)`
    + ` · 屏上节点宽 ${(78 * Math.min(1, 1560 / L.W)).toFixed(0)}px`;
}

function byIdStep(trace, id) {
  const n = trace.graph.nodes.find((x) => x.id === id);
  return n ? n.step : -1;
}

/* ============================================ 2. contract lint (does it suffice?)
 *
 * Walks the trace and reports what the *engine actually needed* but the trace
 * did not provide. This is the deliverable the ticket asks for: "which fields
 * are missing". Rules are checked against the draft contract, not invented.
 */
function lint(trace, derived) {
  const out = [];
  const ids = new Set(trace.steps.map((s) => s.id));
  const declared = new Set(Object.keys(trace.tensors));

  // init: any tensor read anywhere but never written and never given an init
  const written = new Set(), read = new Set();
  for (const s of trace.steps) {
    (s.writes || []).forEach((t) => written.add(t));
    (s.reads || []).forEach((t) => read.add(t));
  }
  for (const t of read) if (!written.has(t) && trace.tensors[t] && trace.tensors[t].init === undefined) {
    out.push({sev: 'gap', what: `张量 "${t}" 被读但从未被写，且 tensors[] 里没有初始值`,
              why: 'render(trace, i) 无法从 trace 得知它在第 0 步的值。要么给 tensors[].init，要么加一个写它的步骤。'});
  }
  for (const t of written) if (!declared.has(t)) {
    out.push({sev: 'gap', what: `步骤写了张量 "${t}"，但 tensors[] 里没有声明`,
              why: '引擎无法知道它的 shape / 位置 / 该如何渲染。'});
  }

  // graph / steps consistency
  const nodeIds = new Set(trace.graph.nodes.map((n) => n.id));
  for (const s of trace.steps) if (!nodeIds.has(s.id)) {
    out.push({sev: 'gap', what: `步骤 "${s.id}" 在 graph.nodes 里没有对应节点`,
              why: 'DAG 与时间轴会不同步——跳到该步时 DAG 上没有可高亮的节点。'});
  }
  for (const n of trace.graph.nodes) if (!ids.has(n.id)) {
    out.push({sev: 'warn', what: `图节点 "${n.id}" 没有对应步骤`, why: '点击它无处可跳。'});
  }

  // formula tiers: every slot in the string must have a binding with all 3 forms
  const slotRe = /\\slot\{([A-Za-z0-9_]+)\}/g;
  const regionRe = /\\region\{([A-Za-z0-9_]+)\}/g;
  const BAD_NUM = /^(NaN|Infinity|-Infinity|undefined|null)$/;
  trace.steps.forEach((s, i) => {
    if (!s.formula) { out.push({sev: 'gap', what: `步骤 "${s.id}" 没有 formula`, why: '公式面板空着。'}); return; }
    for (const tier of ['sym', 'idx', 'num']) {
      if (typeof s.formula[tier] !== 'string') {
        out.push({sev: 'gap', what: `步骤 "${s.id}" 缺 formula.${tier}`, why: '三档显示会缺一档。'});
        continue;
      }
      let m;
      slotRe.lastIndex = 0;
      while ((m = slotRe.exec(s.formula[tier]))) {
        const key = m[1];
        const bdg = (s.bindings || {})[key];
        if (!bdg) out.push({sev: 'gap', what: `步骤 "${s.id}" 的 formula.${tier} 引用了 \slot{${key}}，但 bindings 里没有`,
                            why: '公式上会出现未替换的黄色占位符。'});
        else {
          if (bdg[tier] === undefined) out.push({sev: 'gap', what: `步骤 "${s.id}" 的绑定 "${key}" 缺 "${tier}" 形态`,
                                                 why: '三档中的某一档没有值可填。'});
          else if (BAD_NUM.test(String(bdg[tier]))) out.push({sev: 'gap', what: `步骤 "${s.id}" 绑定 "${key}".${tier} = ${bdg[tier]}`,
                                                 why: 'JSON 不能表示 ±∞ / NaN。用字符串 "-\\infty" 或哨兵值。'});
        }
      }
      regionRe.lastIndex = 0;
      while ((m = regionRe.exec(s.formula[tier]))) {
        const key = m[1];
        if (!s.regions || !s.regions[key]) out.push({sev: 'gap', what: `步骤 "${s.id}" 的 formula.${tier} 引用了 \region{${key}}，但 step.regions 里没有`,
                            why: '被框住的那一段没有文字说明。'});
      }
    }
    // bindings that nothing references
    const used = new Set();
    for (const tier of ['sym', 'idx', 'num']) {
      let m; slotRe.lastIndex = 0;
      while ((m = slotRe.exec(s.formula[tier]))) used.add(m[1]);
    }
    for (const k of Object.keys(s.bindings || {})) if (!used.has(k)) {
      out.push({sev: 'warn', what: `步骤 "${s.id}" 的绑定 "${k}" 没被任何档位引用`, why: '冗余数据。'});
    }
    // 3-tuple shape check: the D06 decision says {sym, idx, num}
    for (const [k, v] of Object.entries(s.bindings || {})) {
      const keys = Object.keys(v).sort().join(',');
      if (keys !== 'idx,num,sym') out.push({sev: 'gap', what: `步骤 "${s.id}" 的绑定 "${k}" 字段是 {${keys}} 而不是 {sym, idx, num}`,
                                             why: 'D06 明确要求显式三元组，不允许靠 key 猜。'});
    }
    // narration must not contain LaTeX (it renders as literal backslashes)
    if (/\\[a-zA-Z]+\{/.test(s.narration || '')) out.push({sev: 'warn', what: `步骤 "${s.id}" 的 narration 里含 LaTeX 命令`,
                                                          why: 'narration 是纯文本，反斜杠会原样显示。'});
    // state keys not declared
    for (const k of Object.keys(s.state || {})) if (!declared.has(k)) {
      out.push({sev: 'gap', what: `步骤 "${s.id}" 的 state 写了未声明张量 "${k}"`, why: '无法渲染。'});
    }
  });

  // stray writes: step declares state for a tensor that is in tensors[] but
  // never in that step's writes[] -- engine can still use it, but it's a smell
  trace.steps.forEach((s) => {
    for (const k of Object.keys(s.state || {})) {
      if (s.writes && !s.writes.includes(k)) out.push({sev: 'warn',
        what: `步骤 "${s.id}" 的 state 里有 "${k}"，但它不在 writes[] 里`,
        why: '读写声明与 state 不一致；引擎目前信 state，writes[] 只用于高亮。'});
    }
  });

  // does anything actually read `graph.edges[].tensor`? report if it's dead weight
  const edgeTensors = new Set(trace.graph.edges.map((e) => e.tensor));
  out.push({sev: 'info', what: `graph.edges[].tensor 共 ${edgeTensors.size} 种取值，引擎只用来画边标签`,
            why: '能否省掉取决于边标签对教学是否必要。'});

  const w = derived ? derived.writes : writesOf(trace, trace.steps.length - 1);
  out.push({sev: 'info', what: `全程共 ${Object.keys(w).length} 个张量被 state 写过：${Object.keys(w).join(', ')}`,
            why: '纯函数重建需要覆盖的就是这些。'});

  return out;
}

/* ================================================ 3. KaTeX three tiers
 *
 * Two render paths, measured against each other:
 *   full  = katex.renderToString on the whole formula (what a naive player does)
 *   patch = skeleton rendered ONCE per (step,tier); cursor moves only rewrite
 *           the textContent of .slot-* spans.
 * The slot values are HTML-escaped plain text, injected via a placeholder that
 * KaTeX renders as an empty span we can find again by index.
 */
const SLOT_PH = '\\slot';

// Build-time macros. \slot{X} becomes \htmlId{slot-X}{\phantom{...}} so that
// KaTeX itself stamps the class+id onto the span it renders for that group --
// no string surgery on the output HTML. \region{K}{...} becomes a \htmlClass
// span wrapping the same way. Both macros are expanded by KaTeX during the
// build render only; the patch render never contains them.
/* PREPROCESSOR.
 *
 * KaTeX's macro API takes a macro NAME only -- it cannot be told "this macro
 * takes two brace groups and here is a function of both". Passing a JS callback
 * for `\slot` works because a macro can consume its own argument, but a
 * callback for `\region{K}{body}` was lossy: KaTeX hands the argument back as a
 * token list it has already mutated (verified: `\slot{MOLD}` arrives as the
 * single token "DLOM"), which mangles multi-token bodies.
 *
 * Expanding to \htmlId / \htmlClass ourselves is both simpler and more honest:
 * plain string work we fully control, using two KaTeX primitives that are
 * documented to emit a real element with our class/id on it. Needs
 * `trust: true` -- \htmlId and \htmlClass are trust-gated.
 */
function readGroup(s, i) {          // s[i] === '{'; returns [body, indexAfterClose]
  let depth = 0;
  for (let k = i; k < s.length; k++) {
    if (s[k] === '{') depth++;
    else if (s[k] === '}') { depth--; if (depth === 0) return [s.slice(i + 1, k), k + 1]; }
  }
  return [s.slice(i + 1), s.length];   // unbalanced: take the rest
}

function expand(raw, pad) {
  let out = '', i = 0;
  while (i < raw.length) {
    if (raw.startsWith('\\slot{', i)) {
      const [name, j] = readGroup(raw, i + 5);
      // \phantom pads to the width of the value this slot will show, so later
      // swaps do not reflow the formula around it.
      out += `\\htmlId{slot-${name.trim()}}{\\phantom{${pad[name.trim()] || '\\;'}}}`;
      i = j;
    } else if (raw.startsWith('\\region{', i)) {
      const [name, j] = readGroup(raw, i + 7);
      const [body, k] = readGroup(raw, j);
      // recurse: a region body commonly contains \slot{...} (e.g. the correction
      // factor is a region over \exp(\slot{MOLD} - \slot{MNEW}))
      out += `\\htmlClass{region-${name.trim()}}{${expand(body, pad)}}`;
      i = k;
    } else { out += raw[i]; i++; }
  }
  return out;
}

// The naive player: render the whole formula string with KaTeX on every frame.
function renderFull(step, tier) {
  return katex.renderToString(step.formula[tier],
    {displayMode: false, throwOnError: false, output: 'html'});
}

/* The split player.
 *
 * A slot's VALUE can itself be LaTeX -- e.g. `\max_i x^{(2)}_i`, or
 * `-\infty`. So "just set textContent" is WRONG: it prints raw TeX. Measured
 * above, that bug is visible on screen as `\left[1.4, 0.35\right]`.
 *
 * The fix that keeps the skeleton idea: render each slot's value with KaTeX
 * SEPARATELY (small, cached per (step,tier,slot)), and put it inside the span
 * that the main formula's \htmlId stamped out. The main formula is rendered
 * once per (step,tier) with \phantom padding sized to the widest value, so
 * swapping the value in cannot reflow the surrounding layout.
 */
const SLOT_CACHE = new Map();     // `${stepId}|${tier}|${slot}` -> html string
function renderSlotValue(step, tier, name) {
  const key = `${step.id}|${tier}|${name}`;
  if (SLOT_CACHE.has(key)) return SLOT_CACHE.get(key);
  const raw = (step.bindings[name] || {})[tier];
  const tex = raw === undefined ? '?' : String(raw);
  let html;
  try {
    html = katex.renderToString(tex, {throwOnError: true, output: 'html'});
  } catch (e) {
    // The trace gave us something KaTeX cannot parse. This is a CONTRACT
    // question, not an engine bug: a binding's num/idx form must be either a
    // plain number or valid TeX, and the trace offers no way to say which.
    html = `<span class="texerr" title="${esc(String(e.message).slice(0, 120))}">${esc(tex)}</span>`;
  }
  SLOT_CACHE.set(key, html);
  return html;
}

function buildSkeleton(step, tier) {
  // The skeleton is laid out with the step's OWN num-tier values in place --
  // i.e. the widest form the step will show. Values for other tiers are usually
  // shorter (sym < idx < num in character count), so padding against num is
  // enough in practice, and it costs exactly one extra render.
  const pad = {};
  for (const [name, b] of Object.entries(step.bindings || {})) {
    const v = b.num !== undefined ? b.num : b[tier];
    pad[name] = String(v === undefined ? '?' : v);
  }
  const html = katex.renderToString(expand(step.formula[tier], pad), {
    displayMode: false, throwOnError: true, output: 'html',
    trust: true, strict: false,
  });
  const div = document.createElement('div');
  div.innerHTML = html;
  const spans = {};
  div.querySelectorAll('[id^="slot-"]').forEach((el) => { spans[el.id.slice(5)] = el; });
  // \htmlClass already put a region-* class on the span KaTeX made; just add the
  // visual box. No Range surgery needed -- that was the whole reason for it.
  div.querySelectorAll('[class*="region-"]').forEach((el) => el.classList.add('regionboxed'));
  const out = document.createElement('span');
  while (div.firstChild) out.appendChild(div.firstChild);
  return {node: out, spans, pad};
}

// Patch the slot spans in place: each gets its own small KaTeX render.
function patchSlots(skel, step, tier) {
  let changed = 0;
  for (const [k, sp] of Object.entries(skel.spans)) {
    const html = renderSlotValue(step, tier, k);
    if (sp.__html !== html) { sp.innerHTML = html; sp.__html = html; changed++; }
  }
  return changed;
}

/* ==================================================== rendering pipeline */

let state = {
  cursor: 0, tier: 'num', when: 'after', playing: false, timer: null,
  inflation: 0, layout: null, skel: null, skelKey: '', trace: null,
};

// Build a synthetic inflation of the trace so we can answer "at what node count
// does this get ugly?". Purely for the layout experiment.
function inflate(trace, extraBlocks) {
  if (!extraBlocks) return trace;
  const t = JSON.parse(JSON.stringify(trace));
  // drop the tail (last 3 real block steps + fin) and append N more blocks
  const real = t.steps.filter((s) => /^[mla](1|2)$/.test(s.id) || s.id === 'init' || /^ld(1|2)$/.test(s.id));
  const fin = t.steps.find((s) => s.id === 'fin');
  const tpl = real.filter((s) => /^(ld2|m2|l2|a2)$/.test(s.id));
  t.steps = real.slice();
  t.graph.nodes = t.graph.nodes.filter((n) => real.some((s) => s.id === n.id));
  for (let b = 3; b <= 2 + extraBlocks; b++) {
    for (const base of tpl) {
      const s = JSON.parse(JSON.stringify(base));
      s.id = base.id.replace('2', String(b));
      s.title = s.title.replace('第 2 块', `第 ${b} 块`);
      s.state = JSON.parse(JSON.stringify(s.state));
      s.narration = `（合成节点，仅为图布局实验；数值沿用第 2 块）`;
      t.steps.push(s);
      t.graph.nodes.push({id: s.id, step: t.steps.length - 1, title: s.title,
                          kind: s.kind, op: s.op, phase: s.phase});
    }
  }
  t.steps.push(JSON.parse(JSON.stringify(fin)));
  t.graph.nodes.push({id: 'fin', step: t.steps.length - 1, title: fin.title,
                      kind: fin.kind, op: fin.op, phase: fin.phase});
  // rewire node.step indices
  t.graph.nodes.forEach((n) => { n.step = t.steps.findIndex((s) => s.id === n.id); });
  // rebuild a chain of block to block edges
  const es = [];
  es.push({from: 'init', to: 'm1', tensor: 'm'}, {from: 'init', to: 'l1', tensor: 'l'},
          {from: 'init', to: 'a1', tensor: 'acc'});
  const last = 2 + extraBlocks;
  for (let b = 1; b <= last; b++) {
    es.push({from: `ld${b}`, to: `m${b}`, tensor: 'x_blk'},
            {from: `m${b}`, to: `l${b}`, tensor: 'm'},
            {from: `m${b}`, to: `a${b}`, tensor: 'm'},
            {from: `l${b}`, to: `a${b}`, tensor: 'l'});
    if (b < last) es.push({from: `m${b}`, to: `m${b+1}`, tensor: 'm'},
                          {from: `l${b}`, to: `l${b+1}`, tensor: 'l'},
                          {from: `a${b}`, to: `a${b+1}`, tensor: 'acc'});
    else es.push({from: `a${b}`, to: 'fin', tensor: 'acc'},
                 {from: `l${b}`, to: 'fin', tensor: 'l'});
  }
  t.graph.edges = es;
  return t;
}

function currentTrace() {
  return state.inflation ? inflate(T, state.inflation) : T;
}

let RENDER_MS = [];
function setCursor(i) {
  const t0 = performance.now();
  const tr = currentTrace();
  i = Math.max(0, Math.min(tr.steps.length - 1, i));
  state.cursor = i;
  const step = tr.steps[i];

  if (!state.layout || state.layoutFor !== tr) {
    state.layout = layout(tr);
    state.layoutFor = tr;
    state.skel = null; state.skelKey = '';
  }
  drawDag(tr, i, {layout: state.layout});

  // --- tensors: pure fold, "after" = cursor, "before" = cursor-1
  const at = state.when === 'after' ? i : i - 1;
  const before = at < 0 ? {} : resolveState(tr, at);
  const after = resolveState(tr, i);
  const shown = state.when === 'after' ? after : before;
  renderTensors(tr, shown, i, step, state.when === 'after' ? after : before);

  // --- formula: rebuild skeleton only when (step, tier) changes
  const key = `${tr.steps.length}:${i}:${state.tier}`;
  if (state.skelKey !== key) {
    state.skel = buildSkeleton(step, state.tier);
    const fx = $('fx'); fx.innerHTML = ''; fx.appendChild(state.skel.node);
    state.skelKey = key;
  }
  patchSlots(state.skel, step, state.tier);

  // --- region box
  const rb = $('regionbox');
  if (step.regions) {
    const keys = Object.keys(step.regions);
    rb.classList.add('show');
    rb.innerHTML = keys.map((k) => `<b>${esc(k)}</b> · ${esc(step.regions[k])}`).join('<br>');
    document.querySelectorAll('.regionboxed').forEach((el) =>
      el.classList.toggle('hl', true));
  } else {
    rb.classList.remove('show'); rb.innerHTML = '';
  }

  // --- narration, chips, scrubber, url
  $('narration').innerHTML =
    `<span class="h">${esc(step.title)}</span> — ${esc(step.narration)}`;
  $('scrub').value = i;
  $('pos').textContent = `第 ${i} / ${tr.steps.length - 1} 步`;
  renderChips(tr, i);
  const url = `?step=${i}`;
  $('url').textContent = url;
  try { history.replaceState(null, '', url); } catch (e) {}
  RENDER_MS.push(performance.now() - t0);
}

function renderTensors(tr, shown, i, step, ref) {
  const box = $('tensors');
  const wrote = new Set(step.writes || []);
  const read = new Set(step.reads || []);
  let h = '';
  for (const [name, spec] of Object.entries(tr.tensors)) {
    const rec = shown[name] || {value: undefined};
    const cls = wrote.has(name) ? 'wrote' : (read.has(name) ? 'read' : '');
    const shape = spec.shape && spec.shape.length ? `[${spec.shape.join(',')}]` : '标量';
    h += `<div class="trow ${cls}">`;
    h += `<div class="tname">${esc(name)} <span class="shp">${esc(shape)}</span>`
       + `<br><span class="at">${esc(spec.at || '?')}</span>`
       + (wrote.has(name) ? `<span class="badge w">写</span>` : '')
       + (read.has(name) ? `<span class="badge">读</span>` : '') + `</div>`;
    h += `<div class="tval">${fmtValue(rec.value, spec, ref[name])}</div>`;
    h += `</div>`;
  }
  box.innerHTML = h;
}

function fmtValue(v, spec, prevRec) {
  if (v === undefined) return `<span class="inf">— 本步之前无值 —</span>`;
  if (v === null) return `<span class="inf">-∞</span>`;
  const isInf = (x) => x === null;
  if (Array.isArray(v)) {
    return v.map((x, k) => {
      const changed = prevRec && Array.isArray(prevRec.value)
        && (prevRec.value.length !== v.length || prevRec.value[k] !== x);
      return `<span class="tcell ${changed ? 'chg' : ''}">${isInf(x) ? '-∞' : fmtNum(x)}</span>`;
    }).join('');
  }
  return `<span class="tcell ${prevRec && prevRec.value !== v ? 'chg' : ''}">`
       + `${isInf(v) ? '-∞' : fmtNum(v)}</span>`;
}

function fmtNum(x) {
  if (typeof x !== 'number') return esc(String(x));
  if (Number.isInteger(x)) return String(x);
  return x.toPrecision(4).replace(/0+$/, '').replace(/\.$/, '');
}

function renderChips(tr, cur) {
  $('chips').innerHTML = tr.steps.map((s, k) =>
    `<span class="chip ${k === cur ? 'current' : (k < cur ? 'done' : '')} ${s.kind === 'comm' ? 'comm' : ''}"
       data-step="${k}" title="${esc(s.title)}">${k}.${esc(s.id)}</span>`).join('');
  $('chips').querySelectorAll('.chip').forEach((c) =>
    c.addEventListener('click', () => setCursor(Number(c.dataset.step))));
}

/* ============================================================ self-check */

let TRACE_SABOTAGE = null;

function runTests() {
  // The real trace already carries the fields the draft contract lacked (that is
  // the finding). To prove the lint actually detects the absence, we lint a
  // deliberately-degraded copy side by side.
  const broken = JSON.parse(JSON.stringify(T));
  delete broken.tensors.x.init;                        // read-but-never-written, no init
  broken.steps[3].bindings.BLKMAX = {sym: 'x', idx: 'x', num: NaN};  // NaN leaks in
  broken.steps[5].formula.num = broken.steps[5].formula.num.replace(/\\slot\{LNEW\}/, ''); // dropped slot
  broken.steps[2].bindings = Object.fromEntries(                 // 2-tuple instead of 3
    Object.entries(broken.steps[2].bindings).map(([k, v]) => [k, {num: v.num}]));

  const L = lint(T);
  const LB = lint(broken);
  const gaps = L.filter((x) => x.sev === 'gap');
  const warns = L.filter((x) => x.sev === 'warn');
  const infos = L.filter((x) => x.sev === 'info');
  const bgaps = LB.filter((x) => x.sev === 'gap');
  const btotal = LB.filter((x) => x.sev !== 'info').length;

  let h = '';
  h += `<div class="verdict ${bgaps.length ? 'ok' : 'bad'}">`
    + `<b>lint 有效性对照：</b>干净的 trace 报 <b>${gaps.length}</b> 个 gap；`
    + `故意破坏的副本（删掉 <code>tensors.x.init</code>、塞入 <code>NaN</code>、`
    + `删掉一个 <code>\\slot</code> 引用、把三元组降成二元组）报 <b>${btotal}</b> 个问题。`
    + (bgaps.length ? ` lint 确实抓得住缺口，所以下面那个 ${gaps.length} 是有意义的。`
                    : ` ⚠️ lint 没抓到破坏——lint 本身有问题，下面的 0 不可信。`)
    + `</div>`;

  h += `<div style="margin-bottom:6px" class="sub">A · 契约 lint（对着 D06 草案逐条查）</div>`;
  h += `<table><tr><th>级别</th><th>发现</th><th>含义</th></tr>`;
  for (const x of gaps.concat(warns, infos)) {
    const c = x.sev === 'gap' ? 'fail' : (x.sev === 'warn' ? 'k' : 'pass');
    h += `<tr><td class="${c}">${x.sev}</td><td>${esc(x.what)}</td><td class="sub">${esc(x.why)}</td></tr>`;
  }
  h += `</table>`;
  if (bgaps.length) {
    h += `<div class="sub" style="margin:9px 0 4px">↳ 同一个 lint 跑在<b>故意破坏的副本</b>上（证明它不是永远报 0）：</div>`;
    h += `<table><tr><th>级别</th><th>发现</th></tr>`;
    for (const x of LB.filter((y) => y.sev !== 'info')) {
      h += `<tr><td class="${x.sev === 'gap' ? 'fail' : 'k'}">${x.sev}</td><td>${esc(x.what)}</td></tr>`;
    }
    h += `</table>`;
  }

  // ---- B: arbitrary jump. Rebuild from scratch at every cursor and compare to a
  // forward-accumulated ground truth computed by mutating state step by step.
  h += `<div style="margin:12px 0 6px" class="sub">B · 任意跳转正确性：纯函数重建 vs 顺序累积（基准）</div>`;
  const fwd = [];              // ground truth: mutate as we walk forward
  let acc = {};
  for (const [n, sp] of Object.entries(T.tensors)) if (sp.init !== undefined) acc[n] = sp.init;
  acc = JSON.parse(JSON.stringify(acc));
  for (let i = 0; i < T.steps.length; i++) {
    Object.assign(acc, T.steps[i].state || {});
    fwd.push(JSON.parse(JSON.stringify(acc)));
  }
  const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const seq = [];                                 // 1) forward
  for (let i = 0; i < T.steps.length; i++) seq.push(['顺序 →', i]);
  for (let i = T.steps.length - 1; i >= 0; i--) seq.push(['完全逆序 →', i]);  // 2) all the way back
  seq.push(['5 → 1（跳回）', 1], ['1 → 0', 0], ['0 → 9（跳到尾）', 9],
           ['9 → 3（乱序）', 3], ['3 → 3（原地）', 3], ['3 → 5', 5],
           ['5 → 2', 2], ['2 → 5（回环）', 5]);   // 3) jumbled round-trips

  const ground = (n, i) => (n in fwd[i] ? fwd[i][n]
      : (T.tensors[n].init !== undefined ? T.tensors[n].init : undefined));

  const NAMES = Object.keys(T.tensors);

  // The control group: a naive "stateful player" that holds ONE mutable state
  // object and, on any jump, just applies that step's writes on top of whatever
  // it was already showing. It never re-reads earlier steps, so jumping backwards
  // leaves stale values behind. This is the exact failure the pure-function
  // design exists to prevent -- if the control can't fail, the 0 above is not
  // evidence of anything.
  const sabState = {};
  for (const [n, s] of Object.entries(T.tensors)) if (s.init !== undefined) sabState[n] = s.init;

  function checkPure(target) {
    const st = resolveState(T, target);
    let ok = true, detail = '';
    for (const n of NAMES) {
      const got = st[n] ? st[n].value : undefined;
      const exp = ground(n, target);
      if (!eq(got, exp)) { ok = false; detail += `${n}: 重建=${JSON.stringify(got)} 应=${JSON.stringify(exp)}; `; }
    }
    return {ok, detail};
  }
  function checkSabotage(target) {
    Object.assign(sabState, T.steps[target].state || {});   // no rollback, no re-derive
    let ok = true, detail = '';
    for (const n of NAMES) {
      const exp = ground(n, target);
      if (!eq(sabState[n], exp)) {
        ok = false;
        if (detail.length < 160) detail += `${n}: 显示=${JSON.stringify(sabState[n])} 应=${JSON.stringify(exp)}; `;
      }
    }
    return {ok, detail};
  }

  let jumpFails = 0, sabFails = 0, jumpRows = '';
  const shown = new Set([0, 1, 5, 9]);
  for (const [label, target] of seq) {
    const pure = checkPure(target);
    const sab = checkSabotage(target);
    if (!pure.ok) jumpFails++;
    if (!sab.ok) sabFails++;
    if ((!pure.ok || !sab.ok || shown.has(target)) && jumpRows.split('<tr>').length < 22) {
      jumpRows += `<tr><td>${esc(label)} ${target}</td>`
        + `<td class="${pure.ok ? 'pass' : 'fail'}">${pure.ok ? '一致' : '不一致'}</td>`
        + `<td class="${sab.ok ? 'pass' : 'fail'}">${sab.ok ? '一致' : '不一致'}</td>`
        + `<td class="sub">${esc(sab.ok ? (pure.detail || '—') : sab.detail.slice(0, 150))}</td></tr>`;
    }
  }
  h += `<div class="verdict ${jumpFails ? 'bad' : 'ok'}">`
    + `<b>纯函数重建：</b>${seq.length} 次跳转（正向 / 完全逆序 / 跳回 / 乱序回环），`
    + (jumpFails ? `${jumpFails} 次不一致。` : `0 次不一致——与顺序累积基准逐位相同。`)
    + `<br><b>对照组（故意做错的有状态播放器）</b>：同一批跳转里 <b>${sabFails} 次不一致</b>。`
    + (sabFails ? `对照组确实会翻车，说明上面那个 0 不是测试写错了。`
                : `⚠️ 对照组也全过了 —— 说明这组跳转没有区分力，上面的 0 不能当证据。`)
    + `</div>`;
  h += `<table><tr><th>跳转</th><th>纯函数 render(trace,i)</th><th>对照组：只记住上一个状态</th><th>说明</th></tr>${jumpRows}</table>`;

  // ---- C: what the static snapshot CANNOT reconstruct. This is the real finding.
  h += `<div style="margin:12px 0 6px" class="sub">C · 静态快照重建不出来的东西（本原型里真实存在的反例）</div>`;
  const derivable = ['m', 'l', 'acc', 'x_blk', 'O'];
  h += `<table><tr><th>量</th><th>能否从 ≤i 的最大写入重建</th><th>原因</th></tr>`;
  h += `<tr><td>${derivable.join(' / ')}</td><td class="pass">能</td>
    <td class="sub">每个都是「整量覆盖写」——写进去的就是完整的新值，没有隐藏累积。</td></tr>`;
  h += `<tr><td>visited / 已走过的路径</td><td class="pass">能</td>
    <td class="sub">就是 step 索引的序关系 (j &lt; i)，不用存。</td></tr>`;
  h += `<tr><td>修正因子 e^(m_old−m_new)</td><td class="pass">能（但只因为它被显式写进了 bindings）</td>
    <td class="sub">它可由 m_old、m_new 算出，但如果引擎想「高亮它」就得有地方取；trace 里以 \region{CORR} + step.regions 提供了。</td></tr>`;
  h += `<tr><td class="fail">x 的原始值</td><td class="fail">不能</td>
    <td class="sub"><b>真实缺口：</b>没有任何 step 写 x。若不补 tensors[].init，render(trace,0) 根本不知道 x 是什么。lint 的 A 段会报这条。</td></tr>`;
  h += `<tr><td class="fail">"上一次是谁写的这个张量"</td><td class="fail">不能</td>
    <td class="sub">能重建「值」，不能重建「来源步」——除非引擎自己扫一遍（本原型就是自己扫的，O(steps)）。若要显示 provenance 又不愿重扫，需要 trace 给 per-step 的写集。</td></tr>`;
  h += `</table>`;
  h += `<div class="verdict ${'ok'}" style="background:#1c2330;border-color:#38445a;color:#c6d2e2">`
    + `<b>结论：不需要反向操作。</b>本 trace 的全部状态量都是「整量覆盖写」，`
    + `回溯只是「取 ≤i 的最后一个写」，没有需要回滚的副作用。`
    + `出现需要反向操作的唯一情形是：某状态是<b>非幂等的累积</b>（例如 append 到列表、计数器 ++、引用计数）——`
    + `这时 step.state 记的必须是<b>累积后的新全量</b>（像本原型的 l/acc 那样），而不是 delta。`
    + `若某 lab 真实存在不可全量化的状态（KV cache 逐 token 追加、PagedAttention 的块池），`
    + `它要么在 trace 里按步记全量快照（代价大），要么那一类 lab 需要显式的反向操作。`
    + `</div>`;

  // ---- D: KaTeX perf, measured
  h += `<div style="margin:12px 0 6px" class="sub">D · KaTeX 性能实测（本页真跑；点上方「跑性能实测」重跑）</div>`;
  h += `<div id="perfout" class="sub">尚未运行。</div>`;
  h += `<pre id="perflog"></pre>`;

  $('tests').innerHTML = h;
  $('perfbtn').onclick = measureKatex;
  $('run').onclick = runTests;
  // run perf automatically too so a screenshot always has numbers
  measureKatex();
}

function measureKatex() {
  const out = $('perfout'), log = $('perflog');
  if (!out) return;
  const N = 60;
  const tr = currentTrace();
  const steps = tr.steps;

  // ---- warm every path first, so we time steady state not JIT warmup
  steps.forEach((s) => renderFull(s, state.tier));
  steps.forEach((s) => { const k = buildSkeleton(s, state.tier); patchSlots(k, s, state.tier); });
  const skels = steps.map((s) => buildSkeleton(s, state.tier));
  const time = (fn) => { const a = performance.now(); for (let k = 0; k < N; k++) fn(steps[k % steps.length], k); return (performance.now() - a) / N; };

  // ---- path 1: naive -- full KaTeX re-render + DOM mount per cursor move
  const fullPer = time((s) => {
    const d = document.createElement('div');
    d.innerHTML = renderFull(s, state.tier);
    d.remove();
  });

  // ---- path 2: skeleton already built, patch slot values only
  const patchPer = time((s, k) => patchSlots(skels[k % skels.length], s, state.tier));

  // ---- path 3: build the skeleton too (what a per-step player pays)
  const skelPer = time((s) => buildSkeleton(s, state.tier));

  // ---- path 4: the unavoidable DOM work -- tensor panel + DAG rebuild
  const tensPer = time((s, k) => renderTensors(tr, resolveState(tr, k % tr.steps.length), k % tr.steps.length, s, {}));
  const dagPer = time((s, k) => drawDag(tr, k % tr.steps.length, {layout: state.layout || layout(tr)}));

  // ---- how full render scales with formula length: is KaTeX ever the problem?
  const base = steps[Math.min(5, steps.length - 1)];
  const scale = [];
  for (const reps of [1, 2, 4, 8, 16, 32]) {
    const fat = {id: 'fat', formula: {}, bindings: base.bindings};
    for (const t of ['sym', 'idx', 'num']) {
      fat.formula[t] = Array.from({length: reps}, () => base.formula[t]).join(' \\;+\\; ');
    }
    const a = performance.now();
    for (let k = 0; k < 20; k++) renderFull(fat, 'num');
    scale.push({reps, ms: (performance.now() - a) / 20});
  }

  const ratio = fullPer / Math.max(patchPer, 1e-6);
  const domPer = tensPer + dagPer;
  log.textContent =
    `N=${N} 次，tier=${state.tier}，公式数=${steps.length}，node=${navigator.userAgent.match(/Chrome\/[\d.]+/)}\n\n` +
    `  A: 全量 renderToString + DOM 挂载   ${fullPer.toFixed(4)} ms/次   ← 朴素播放器每帧付这个\n` +
    `  B: 骨架复用 + 只重渲 slot 子公式    ${patchPer.toFixed(4)} ms/次   ← 拆分方案每帧付这个\n` +
    `  C: 构建一个骨架（每步一次）         ${skelPer.toFixed(4)} ms/次\n` +
    `  D: 重建张量面板（DOM）              ${tensPer.toFixed(4)} ms/次\n` +
    `  E: 重绘 DAG（innerHTML）            ${dagPer.toFixed(4)} ms/次\n\n` +
    `  A/B = ${ratio.toFixed(1)}×    但真正的大头是 D+E = ${domPer.toFixed(3)} ms/次，` +
    `而 A+B 加起来只有 ${(fullPer + patchPer).toFixed(3)} ms/次\n\n` +
    `  公式长度 vs 全量渲染（看 KaTeX 是否随公式线性增长）：\n` +
    scale.map((s) => `    ${String(s.reps + '×').padStart(4)} 长 → ${s.ms.toFixed(4)} ms`).join('\n');

  const B = 16.7;
  out.innerHTML = `KaTeX 全量渲染 <b>${fullPer.toFixed(3)} ms/帧</b>（60fps 预算 16.7 ms 的 `
    + `<b>${(fullPer / B * 100).toFixed(1)}%</b>），拆分后 <b>${patchPer.toFixed(4)} ms/帧</b>`
    + `（差 ${ratio.toFixed(0)}×）。<b>但两者都不是瓶颈</b>：张量面板 ${tensPer.toFixed(2)} ms + `
    + `DAG 重绘 ${dagPer.toFixed(2)} ms = <b>${domPer.toFixed(2)} ms/帧</b>，是 KaTeX 全量渲染的 `
    + `${(domPer / fullPer).toFixed(1)} 倍。`;
  window.__PERF = {fullPer, patchPer, skelPer, tensPer, dagPer, domPer, ratio, scale, N, tier: state.tier};
}

/* ============================================================ wiring */

function setTier(t) { state.tier = t; state.skelKey = ''; sync(); }
function setWhen(w) { state.when = w; sync(); }

function sync() {
  document.querySelectorAll('#tier button').forEach((b) =>
    b.classList.toggle('on', b.dataset.t === state.tier));
  document.querySelectorAll('#when button').forEach((b) =>
    b.classList.toggle('on', b.dataset.w === state.when));
  document.querySelectorAll('#e-straight,#e-arc').forEach((b) =>
    b.classList.toggle('on', (b.id === 'e-arc') === (EDGE_MODE === 'arc')));
  document.querySelectorAll('#e-seq').forEach((b) => b.classList.toggle('on', NG.seqEdges));
  setCursor(state.cursor);
}

function init() {
  const step = Number(new URLSearchParams(location.search).get('step') || 0);
  $('ttl').textContent = T.meta.title;
  $('sub').innerHTML = `trace 由 <code>make_trace.py</code> 用 numpy 真算（O = ${T.meta.reference.split('= ')[1]}）· <b>停在首步</b>（不自动播放）· 桌面优先`;
  $('scrub').max = T.steps.length - 1;

  $('b-next').onclick = () => setCursor(state.cursor + 1);
  $('b-prev').onclick = () => setCursor(state.cursor - 1);
  $('b-home').onclick = () => setCursor(0);
  $('b-end').onclick = () => setCursor(currentTrace().steps.length - 1);
  $('scrub').oninput = (e) => setCursor(Number(e.target.value));
  $('b-play').onclick = togglePlay;
  document.querySelectorAll('#tier button').forEach((b) => b.onclick = () => setTier(b.dataset.t));
  document.querySelectorAll('#when button').forEach((b) => b.onclick = () => setWhen(b.dataset.w));
  $('e-seq').onclick = () => { NG.seqEdges = !NG.seqEdges; state.layout = null; sync(); };
  $('e-straight').onclick = () => { EDGE_MODE = 'straight'; sync(); };
  $('e-arc').onclick = () => { EDGE_MODE = 'arc'; sync(); };
  $('inflate').onchange = (e) => {
    state.inflation = Number(e.target.value);
    state.layout = null; state.skelKey = '';
    setCursor(Math.min(state.cursor, currentTrace().steps.length - 1));
  };
  $('wrap').onchange = (e) => {
    NG.wrapAt = Number(e.target.value);
    state.layout = null;
    setCursor(state.cursor);
  };

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' && e.key !== ' ') return;
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); setCursor(state.cursor + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); setCursor(state.cursor - 1); }
    else if (e.key === 'Home') { e.preventDefault(); setCursor(0); }
    else if (e.key === 'End') { e.preventDefault(); setCursor(currentTrace().steps.length - 1); }
    else if (e.key === '1') setTier('sym');
    else if (e.key === '2') setTier('idx');
    else if (e.key === '3') setTier('num');
  });

  setCursor(step);
  runTests();
}

function togglePlay() {
  const b = $('b-play');
  if (state.playing) {
    clearInterval(state.timer); state.playing = false; b.textContent = '▶ 播放'; b.classList.remove('on');
    return;
  }
  state.playing = true; b.textContent = '❚❚ 暂停'; b.classList.add('on');
  state.timer = setInterval(() => {
    const n = currentTrace().steps.length;
    if (state.cursor >= n - 1) { togglePlay(); return; }
    setCursor(state.cursor + 1);
  }, 900);
}

window.addEventListener('DOMContentLoaded', init);
