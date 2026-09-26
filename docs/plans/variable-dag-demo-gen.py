#!/usr/bin/env python3
"""Demo lab page in the user's requested form.

Their spec:
  - 每个知识点基本一屏
  - DAG：节点是变量，边是一次计算
  - 二维矩阵的数据结构展示
  - 单步前进/后退，当前步高亮
  - 侧栏 LaTeX 公式，变量在图里与公式里的高亮互相对应
  - PC 与手机都能方便阅读

Two things this demo must get right that the first attempt botched:
  1. The formula strings use the engine's own DSL (`\\slot{X}` / `\\region{...}`).
     They must be EXPANDED before KaTeX sees them, or raw LaTeX leaks.
  2. The highlighter must key off `\\slot{}` names -- those ARE the variable
     references. Guessing identifiers with a regex was wrong.
"""
import json
import pathlib
import re

REPO = pathlib.Path("/home/shared/code/AIInfraGuide-labs")
TRACE = json.loads((REPO / "labs/traces/online-softmax-N8-B4-T1.json").read_text(encoding="utf-8"))


TENSOR_NAMES = list(TRACE["tensors"].keys())

def _vars_in(sym):
    """Which tensors does this binding's symbolic form mention?

    The trace does not carry this mapping, but it does not need to: `sym` is
    LaTeX written by the generator, so the names in it ARE the tensor names
    (\mathrm{acc}^{(j-1)} -> acc, m^{(j)} -> m, x^{(j)}_i -> x). This is what
    lets the formula panel and the graph highlight the same variable.
    """
    if not sym:
        return []
    plain = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", sym)   # \mathrm{acc} -> acc
    found = []
    for name in TENSOR_NAMES:
        # 词边界：避免 x 命中 x_blk 内部、m 命中 \mathrm 里的 m
        if re.search(r"(?<![A-Za-z_])" + re.escape(name) + r"(?![A-Za-z0-9_])", plain):
            found.append(name)
    return found

def _bindings(raw, tensors):
    """Flatten each binding to its numeric text plus the tensors it refers to."""
    out = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[k] = {"num": v, "sym": v, "vars": []}
        else:
            sym = v.get("sym") or ""
            out[k] = {
                "num": v.get("num") or v.get("idx") or sym,
                "sym": sym,
                "idx": v.get("idx") or sym,
                "vars": _vars_in(sym),
            }
    return out

NODES, EDGES = [], []
for name, spec in TRACE["tensors"].items():
    NODES.append({"id": name, "shape": spec.get("shape", []), "at": spec.get("at", ""),
                  "role": spec.get("role", ""), "note": spec.get("note", "")})
for i, s in enumerate(TRACE["steps"]):
    EDGES.append({
        "i": i, "id": s["id"], "title": s.get("title", ""), "op": s.get("op", ""),
        "reads": [t for t in (s.get("reads") or []) if t in TRACE["tensors"]],
        "writes": [t for t in (s.get("writes") or []) if t in TRACE["tensors"]],
        "formula": s.get("formula") or {},
        "bindings": _bindings(s.get("bindings") or {}, TRACE["tensors"]),
        "state": s.get("state") or {},          # 本步之后被改写的张量的新值
        "narration": s.get("narration", ""),
    })

DATA = {"nodes": NODES, "edges": EDGES}

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>交互实验室 · 变量 DAG（示例）</title>
<link rel="stylesheet" href="./AIInfraGuide/labs/assets/vendor/katex/katex.min.css">
<style>
:root{--bg:#fbfbfd;--fg:#1d1d1f;--sub:#6e6e73;--line:#e3e3e6;--card:#fff;
  --accent:#0071e3;--accent-soft:#e8f1fd;--lit:#ff9500;--lit-soft:#fff4e5;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;--r:12px}
@media (prefers-color-scheme:dark){:root{--bg:#0b0b0d;--fg:#f5f5f7;--sub:#98989d;
  --line:#2a2a2e;--card:#161618;--accent:#0a84ff;--accent-soft:#0a2a4a;--lit:#ff9f0a;--lit-soft:#3a2a10}}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--fg);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased;display:flex;flex-direction:column;height:100dvh;overflow:hidden}
header{flex:0 0 auto;display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;
  padding:.7rem 1rem;border-bottom:1px solid var(--line);background:var(--card)}
h1{font-size:.95rem;margin:0;font-weight:650}
.hd-sub{color:var(--sub);font-size:.78rem}
.hd-step{margin-left:auto;font:600 .78rem/1 var(--mono);color:var(--accent);
  background:var(--accent-soft);padding:.32rem .58rem;border-radius:99px;white-space:nowrap}
main{flex:1 1 auto;display:grid;grid-template-columns:minmax(0,1.15fr) minmax(360px,1fr);min-height:0}
@media (max-width:900px){main{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) minmax(0,1fr)}}
#dagwrap{position:relative;min-height:0;overflow:hidden;border-right:1px solid var(--line)}
@media (max-width:900px){#dagwrap{border-right:0;border-bottom:1px solid var(--line)}}
#dag{width:100%;height:100%;display:block}
aside{min-height:0;overflow:auto;padding:1rem 1.1rem;background:var(--card);
  -webkit-overflow-scrolling:touch}
.edge-op{font:600 .7rem/1 var(--mono);color:var(--sub);letter-spacing:.06em;text-transform:uppercase}
.edge-title{font-weight:650;font-size:.93rem;margin:.15rem 0 .7rem}
.fml{padding:.65rem .8rem;background:var(--bg);border:1px solid var(--line);
  border-radius:var(--r);font-size:.82rem}
/* 长公式横向滚动，不要折行 —— 折行会把数学结构拆散，比滚动更难读 */
.fml .katex-display{overflow-x:auto;overflow-y:hidden;padding:2px 0;margin:0}
.fml .katex-display>.katex{white-space:nowrap}
.narr{color:var(--sub);font-size:.84rem;margin:.65rem 0 0}
.io{display:flex;gap:1.2rem;flex-wrap:wrap;margin:.7rem 0 0;font-size:.82rem}
.io b{display:block;font:600 .68rem/1.6 var(--mono);color:var(--sub);letter-spacing:.05em}
.chip{display:inline-block;padding:.08rem .4rem;margin:.12rem .16rem .12rem 0;border-radius:6px;
  background:var(--bg);border:1px solid var(--line);font:600 .76rem/1.5 var(--mono);
  cursor:pointer;transition:background .12s,border-color .12s;user-select:none}
.chip:hover{border-color:var(--accent)}
.chip.on{background:var(--lit-soft);border-color:var(--lit);color:var(--lit)}
.kv{cursor:pointer;border-radius:4px;padding:0 .1em;transition:background .12s}
.kv:hover{background:var(--accent-soft)}
.kv.on{background:var(--lit-soft);box-shadow:0 0 0 1px var(--lit)}
/* 张量数值查看器 */
.tv{margin:.7rem 0 0;border:1px solid var(--line);border-radius:var(--r);overflow:hidden}
.tv-hd{display:flex;gap:.5rem;align-items:baseline;padding:.35rem .6rem;background:var(--bg);
  border-bottom:1px solid var(--line);font:600 .74rem/1.4 var(--mono)}
.tv-shp{color:var(--sub);font-weight:500}
.tv-at{margin-left:auto;color:var(--sub);font-weight:500;font-size:.7rem}
.tv .row{display:flex;flex-wrap:wrap;gap:2px;padding:3px 6px}
.cell{font:500 .72rem/1.5 var(--mono);padding:.1rem .35rem;border-radius:4px;
  background:var(--bg);border:1px solid var(--line);white-space:nowrap}
.cell.scalar{background:var(--lit-soft);border-color:var(--lit);color:var(--lit);font-weight:650}
footer{flex:0 0 auto;display:flex;align-items:center;gap:.6rem;padding:.6rem 1rem;
  border-top:1px solid var(--line);background:var(--card);
  padding-bottom:calc(.6rem + env(safe-area-inset-bottom))}
button{font:inherit;font-size:.84rem;padding:.4rem .75rem;border-radius:9px;cursor:pointer;
  border:1px solid var(--line);background:var(--bg);color:var(--fg);white-space:nowrap}
button:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
button:disabled{opacity:.35;cursor:default}
input[type=range]{flex:1 1 auto;min-width:60px;accent-color:var(--accent)}
.pos{font:600 .76rem/1 var(--mono);color:var(--sub);white-space:nowrap}
.n-box{fill:var(--card);stroke:var(--line);stroke-width:1.5;transition:stroke .15s,opacity .15s}
.n-lab{font:650 13px var(--mono);fill:var(--fg);transition:opacity .15s}
.n-shp{font:11px var(--mono);fill:var(--sub);transition:opacity .15s}
.n-box.on{stroke:var(--lit);stroke-width:2.5}
.n-box.off{opacity:.55}.n-lab.off{opacity:.5}.n-shp.off{opacity:.45}
/* 已走过的节点：介于两者之间，让「进度」看得出来 */
.n-box.past{stroke:var(--accent);stroke-width:2;opacity:.85}
.n-lab.past{opacity:.85}.n-shp.past{opacity:.7}
.e-path{stroke:var(--line);stroke-width:2;fill:none;transition:stroke .15s,opacity .15s}
.e-path.done{stroke:var(--accent);opacity:.6}
.e-path.off{opacity:.35}
.e-path.on{stroke:var(--lit);stroke-width:3.5}
.e-lab{font:600 10px var(--mono);fill:var(--sub)}
.e-lab.on{fill:var(--lit);font-weight:700}.e-lab.off{opacity:.4}
</style>
</head>
<body>
<header>
  <h1>Online Softmax · 变量图</h1>
  <span class="hd-sub">节点 = 变量 · 边 = 一次计算</span>
  <span class="hd-step" id="pos"></span>
</header>
<main>
  <div id="dagwrap"><svg id="dag"></svg></div>
  <aside id="side"></aside>
</main>
<footer>
  <button id="prev">← 上一步</button>
  <button id="next">下一步 →</button>
  <input type="range" id="range" min="0" value="0">
</footer>
<script src="./AIInfraGuide/labs/assets/vendor/katex/katex.min.js"></script>
<script src="./AIInfraGuide/labs/assets/engine/formula.js"></script>
<script>
const DATA = __DATA__;

/* ---------- 公式 DSL 展开 ------------------------------------------------
 * Trace formulas carry the engine's own macros. KaTeX must never see them:
 *   \slot{NAME}      -> that binding's value for the current tier
 *   \region{NAME}{..}-> the body, wrapped so the region can be highlighted
 * Order matters: \region bodies contain \slot.
 */
/* 公式渲染**直接复用引擎的 formula.js** —— 它的 expand()/buildSkeleton()/patch()
   是 240 行、经 7 个 lab 验证的代码，文件头还写明了我手搓时踩的三个坑。
   自己重写一遍只会重新发现同样的问题，且在宽度处理上还不如它。 */
const F = LabEngine.formula;
const FCACHE = {};

/* ---------- 布局：变量按产生顺序分层，同层竖排 ------------------------- */
const NW = 116, NH = 46, GAPX = 112, GAPY = 74, PAD = 26;
function layout(vertical){
  const layer = {}; DATA.nodes.forEach(n => layer[n.id] = 0);
  for (let pass=0; pass<8; pass++){
    DATA.edges.forEach(e => {
      const base = e.reads.length ? Math.max(...e.reads.map(r => layer[r] ?? 0)) : 0;
      e.writes.forEach(w => { if (layer[w] !== undefined) layer[w] = Math.max(layer[w], base+1); });
    });
  }
  /* 层号会因为 base+1 的累积而跳号（实测出现过 0,1,17,18,19）。按「实际出现过
     的层」重编号，否则按层号算宽度会得出 4526px 这种荒诞尺寸，图被压成一条线。 */
  const used = [...new Set(DATA.nodes.map(n => layer[n.id]))].sort((a,b)=>a-b);
  const rank = {}; used.forEach((L,i) => rank[L] = i);
  const bands = {};
  DATA.nodes.forEach(n => (bands[rank[layer[n.id]]] ??= []).push(n.id));
  const nBand = used.length;
  const nWide = Math.max(...Object.values(bands).map(b => b.length), 1);

  /* 选方向：让画布宽高比尽量接近面板，否则 preserveAspectRatio 会把图压扁成一条线。
     判据必须用「离目标比例的距离」，不能比大小 —— 横排 6.2:1 与竖排 0.65:1 相比，
     0.65 离面板的 1.0 更近，但 0.65 < 6.2 会让「谁大选另一个」这种写法选错。
     用对数距离，因为比例是对称的（2:1 与 1:2 一样偏）。 */
  const wH = nBand*NW + (nBand-1)*GAPX, hH = nWide*NH + (nWide-1)*GAPY;
  const wV = nWide*NW + (nWide-1)*GAPX, hV = nBand*NH + (nBand-1)*GAPY;
  const TARGET = 1.0;                    // 面板在这套布局里接近正方
  if (vertical === undefined){
    const distH = Math.abs(Math.log((wH/hH) / TARGET));
    const distV = Math.abs(Math.log((wV/hV) / TARGET));
    vertical = distV <= distH;
  }

  const pos = {};
  Object.keys(bands).forEach(b => bands[b].forEach((id, i) => {
    const L = +b;
    pos[id] = vertical
      ? {x: PAD + i*(NW+GAPX), y: PAD + L*(NH+GAPY), layer: L}
      : {x: PAD + L*(NW+GAPX), y: PAD + i*(NH+GAPY), layer: L};
  }));
  const xs = Object.values(pos).map(p=>p.x), ys = Object.values(pos).map(p=>p.y);
  return {pos, vertical, w: Math.max(...xs)+NW+PAD*2, h: Math.max(...ys)+NH+PAD*2, nBand};
}

/* ---------- 渲染 ------------------------------------------------------- */
const svg = document.getElementById('dag');
const side = document.getElementById('side');
const range = document.getElementById('range');
const posEl = document.getElementById('pos');
const L = layout();

/* 变量名 -> 安全的 css 类片段 */
const cls = id => 'v-' + id.replace(/[^A-Za-z0-9_]/g, '_');

function edgeGeom(e){
  const a = L.pos[e.reads[0]], b = L.pos[e.writes[0]];
  if (!a || !b) return null;
  let x1,y1,x2,y2;
  if (L.vertical){                       // 上入下出
    x1 = a.x + NW/2; y1 = a.y + NH;
    x2 = b.x + NW/2; y2 = b.y;
  } else {                               // 左入右出
    x1 = a.x + NW; y1 = a.y + NH/2;
    x2 = b.x;      y2 = b.y + NH/2;
  }
  const d = (L.vertical ? Math.abs(x1-x2) < 3 : Math.abs(y1-y2) < 3)
    ? `M${x1} ${y1} L${x2} ${y2}`
    : (L.vertical
        ? `M${x1} ${y1} C${x1} ${(y1+y2)/2} ${x2} ${(y1+y2)/2} ${x2} ${y2}`
        : `M${x1} ${y1} C${(x1+x2)/2} ${y1} ${(x1+x2)/2} ${y2} ${x2} ${y2}`);
  const lx = L.vertical ? (x1+x2)/2 + 8 : (x1+x2)/2;
  const ly = L.vertical ? (y1+y2)/2        : Math.min(y1,y2) - 6;
  return {d, lx, ly};
}

let pinned = new Set();
function paintPinned(){
  document.querySelectorAll('.kv,.chip').forEach(el =>
    el.classList.toggle('on', pinned.has(el.dataset.v)));
  DATA.nodes.forEach(n => {
    const g = svg.querySelector('.'+cls(n.id));
    if (!g) return;
    const on = pinned.has(n.id);
    g.querySelector('.n-box').classList.toggle('on', on);
  });
}

function render(step){
  cur = step;              // 写回全局：按钮/键盘/时间轴三条路共用它
  const e = DATA.edges[step] || {};
  const hot = new Set([...(e.reads||[]), ...(e.writes||[])]);

  /* --- SVG --- */
  let s = `<defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="var(--line)"/></marker>
    <marker id="ar-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="var(--lit)"/></marker>
  </defs>`;
  DATA.edges.forEach((x, i) => {
    const g = edgeGeom(x); if (!g) return;
    const st = i===step ? 'on' : (i<step ? 'done' : 'off');
    const mk = i===step ? 'url(#ar-on)' : 'url(#ar)';
    s += `<path class="e-path ${st}" d="${g.d}" marker-end="${mk}"/>`;
    s += `<text class="e-lab ${i===step?'on':(i>step?'off':'')}" x="${g.lx}" y="${g.ly}"
           text-anchor="middle">${(x.op||'').toUpperCase()}</text>`;
  });
  /* 节点三态：本步涉及（橙）/ 之前出现过（蓝）/ 还没到（淡）。
     之前只有「涉及 / 未涉及」两态且未涉及压到 0.3，整张图看上去几乎全灰，
     读者会以为只有一步。 */
  const pastSet = new Set();
  for (let k = 0; k < step; k++){
    (DATA.edges[k].reads||[]).forEach(t => pastSet.add(t));
    (DATA.edges[k].writes||[]).forEach(t => pastSet.add(t));
  }
  DATA.nodes.forEach(n => {
    const p = L.pos[n.id]; if (!p) return;
    const on = hot.has(n.id);
    const past = !on && pastSet.has(n.id);
    const st = on ? 'on' : (past ? 'past' : 'off');   // 'on' 不能省——选择器要它
    const shp = n.shape.length ? '['+n.shape.join('×')+']' : '标量';
    s += `<g class="${cls(n.id)}">
      <rect class="n-box ${st}" x="${p.x}" y="${p.y}" width="${NW}" height="${NH}" rx="10"/>
      <text class="n-lab ${st}" x="${p.x+11}" y="${p.y+20}">${n.id}</text>
      <text class="n-shp ${st}" x="${p.x+11}" y="${p.y+35}">${shp}${n.at?' · '+n.at:''}</text>
    </g>`;
  });
  svg.setAttribute('viewBox', `0 0 ${L.w} ${L.h}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.innerHTML = s;

  /* --- 侧栏公式：交给引擎 --- */
  let fmlNode = null, fmlHtml = '';
  try {
    const sk = F.buildSkeleton(e, 'num', 'g' + e.i + '-', FCACHE);
    F.patch(sk, e, 'num', FCACHE);
    fmlNode = sk.node;
    fmlHtml = sk.node.outerHTML;
  } catch(err){
    fmlHtml = '<span style="color:#c00;font:12px var(--mono)">公式渲染失败: '+
              (err && err.message) + '</span>';
  }


  /* --- 侧栏其余 --- */
  const shapeOf = id => {
    const n = DATA.nodes.find(x=>x.id===id);
    if (!n) return '';
    return n.shape.length ? '['+n.shape.join('×')+']' : '标量';
  };
  const chips = arr => (arr||[]).map(v =>
    `<span class="chip" data-v="${v}">${v} <b style="opacity:.55">${shapeOf(v)}</b></span>`
  ).join('') || '—';

  side.innerHTML = `
    <div class="edge-op">${(e.op||'').toUpperCase()}${e.id ? ' · '+e.id : ''}</div>
    <div class="edge-title">${e.title||''}</div>
    <div class="fml">${fmlHtml}</div>
    <div class="io">
      <div><b>读</b>${chips(e.reads)}</div>
      <div><b>写</b>${chips(e.writes)}</div>
    </div>
    <p class="narr">${e.narration||''}</p>
    ${(e.writes||[]).map(valueGrid).join('')}`;

  /* 公式里 \htmlClass 生成的 span 也参与联动 */
  side.querySelectorAll('[class^="region-"]').forEach(el=>{
    el.style.borderBottom = '2px solid var(--lit)';
  });

  side.querySelectorAll('[class*="region-"]').forEach(el =>
    el.style.borderBottom = '2px solid var(--lit)');

  posEl.textContent = (step+1) + ' / ' + DATA.edges.length;
  range.value = step;
  document.getElementById('prev').disabled = step===0;
  document.getElementById('next').disabled = step===DATA.edges.length-1;
  pinned.clear();
  try { history.replaceState(null,'','?step='+step); } catch(_){}
  paintPinned();
}

/* --- 双向高亮：点公式里的 slot、点 chip、点图节点，三处一起亮 --- */
document.addEventListener('click', ev => {
  const t = ev.target.closest('[data-v]');
  if (!t){ pinned.clear(); paintPinned(); return; }
  const v = t.dataset.v;
  pinned.has(v) ? pinned.delete(v) : pinned.add(v);
  paintPinned();
});

/* ---------- 张量数值查看器：按 shape 决定 1D 一行 / 2D 网格 ---------- */
function valueGrid(name){
  // 沿时间回溯，找这个张量最近一次被写入的值（纯函数重建，与引擎同一套规则）
  let val, fromStep = -1;
  for (let i = 0; i <= cur; i++){
    const st = DATA.edges[i].state;
    if (st && Object.prototype.hasOwnProperty.call(st, name)){ val = st[name]; fromStep = i; }
  }
  if (val === undefined){
    const n = DATA.nodes.find(x=>x.id===name);
    if (n && n.init !== undefined){ val = n.init; fromStep = -1; }
  }
  if (val === undefined) return '';
  const n = DATA.nodes.find(x=>x.id===name) || {};
  const shape = n.shape || [];
  const fmt = v => (typeof v === 'number')
      ? (Number.isInteger(v) ? String(v) : v.toFixed(4).replace(/0+$/,'').replace(/\.$/,''))
      : String(v);

  let body;
  if (!Array.isArray(val)){                                   // 标量
    body = `<span class="cell scalar">${fmt(val)}</span>`;
  } else if (shape.length <= 1 || !Array.isArray(val[0])){    // 1D 向量
    body = '<div class="row">' + val.map(v=>`<span class="cell">${fmt(v)}</span>`).join('') + '</div>';
  } else {                                                     // 2D 矩阵
    body = val.map(r => '<div class="row">' +
      r.map(v=>`<span class="cell">${fmt(v)}</span>`).join('') + '</div>').join('');
  }
  const where = fromStep < 0 ? '初值' : ('第 ' + (fromStep+1) + ' 步写入');
  return `<div class="tv"><div class="tv-hd"><span>${name}</span>
    <span class="tv-shp">${shape.length ? '['+shape.join('×')+']' : '标量'}</span>
    <span class="tv-at">${where}</span></div>${body}</div>`;
}

/* --- 控制 --- */
let cur;
range.addEventListener('input', () => render(+range.value));
cur = Math.max(0, Math.min(DATA.edges.length-1,
  +(new URLSearchParams(location.search).get('step')||0)));
range.max = DATA.edges.length-1;
/* cur 由 render() 维护，这里只负责「相对当前位置走一步」——
   之前写成 render(cur+1) 但 cur 从不更新，于是按钮点第一次之后就没反应。 */
document.getElementById('prev').onclick = () => render(Math.max(0, cur - 1));
document.getElementById('next').onclick = () => render(Math.min(DATA.edges.length - 1, cur + 1));
document.addEventListener('keydown', ev => {
  if (ev.key==='ArrowRight') render(Math.min(DATA.edges.length-1, cur+1));
  if (ev.key==='ArrowLeft')  render(Math.max(0, cur-1));
});
render(cur);
</script>
</body>
</html>
"""

out = pathlib.Path("/tmp/labs-preview/demo-dag.html")
out.write_text(HTML.replace("__DATA__", json.dumps(DATA, ensure_ascii=False)), encoding="utf-8")
print(f"wrote {out}  ({len(NODES)} 变量, {len(EDGES)} 边)")
