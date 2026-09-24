# 交互学习实验室 · 源码树

设计文档见 `docs/plans/interactive-labs.md`。这里是 lab 的**源码**；`public/labs/` 是构建产物。

## 布局

```
labs/
├── assets/            所有 lab 页面共享（方案 A：共享 assets 目录）
│   ├── engine/        通用播放器（见下）
│   │   └── views/     跨 lab 复用的专用视图：每类一个 .js + 一个同名 .css
│   └── vendor/katex/  本地 KaTeX（由 scripts/vendor-katex.mjs 生成，勿手改）
├── traces/            每个 lab 一个 Python 轨迹生成器（**不会**被发布）
└── pages/             每个 lab 一个自包含 HTML
```

## 引擎

`labs/assets/engine/` 是一份通用播放器，**不含任何 lab 专属逻辑**。lab 页面只提供 trace 与配置，
调用 `LabEngine.lab(trace, config)`。按加载顺序：

| 文件 | 职责 |
|---|---|
| `trace-model.js` | 契约：哨兵解析、`resolve()` 纯函数重建、`lint()` |
| `layout.js` | DAG 分层布局（顺序边 + 折行） |
| `formula.js` | KaTeX 三档渲染（`\slot` / `\region` 预处理） |
| `dag.js` | SVG DAG，增量改 class 重绘 |
| `tensors.js` | 张量检查器（网格 + 热力图） |
| `narrow.js` | 统一的窄屏降级组件 |
| `player.js` | 时间轴 / 深链 / 键盘 / 播放 —— 入口 `LabEngine.lab()` |
| `verify.js` | 自检：任意跳转对照实验、lint 对照实验 |
| `lab.css` | 全部样式（含亮/暗色） |

页面配置项见 `player.js` 顶部的 `DEFAULTS`。额外面板用 `panels: [{id, label, render(step, ctx)}]`，
引擎负责容器，`render` 返回 HTML 字符串。面板只格式化 trace 里的字段，**不做任何算法计算** ——
L00 的修正因子放大器与对照模式都挂在这一层。

**有一个坑**：`render` 返回后，引擎会执行 `host.innerHTML = 返回值`。所以面板里要挂**有状态**的
视图（例如显存账本，它的悬停高亮与宽度过渡都需要保留 DOM）时，不能把视图挂进 `[data-panel-body]`，
要挂到引擎建好后不再触碰的 `[data-panel="<id>"]` 上（追加一个兄弟容器）——否则每次重绘都会
把视图从文档里摘掉。L00 的 `render` 返回 HTML 字符串，没踩到这条；账本页是第一个有状态面板。

## 专用视图

`labs/assets/engine/views/` 放跨 lab 复用的视图（决策 #38：在引擎阶段一次性抽出，之后的 lab 只消费）。
每个视图是「一个 .js + 一个同名 .css」，**不改 `lab.css`** —— 那份样式随引擎冻结，而视图组件正是
所有 lab 都会碰的东西，让它去改共享样式表就等于让并行开发重新冲突。

| 文件 | 职责 |
|---|---|
| `ledger.js` / `ledger.css` | 显存账本：一维堆叠条 + 二维矩阵（多策略 × 多分项） |

**视图组件里不放显存公式。** 代价模型是 lab 内容：L05 算 KV Cache，L14 算 ZeRO 切分，
组件不应该认识其中任何一个的键名。所以公式由页面作为 `{segments, predict(cfg), properties}`
传入，`predict` 必须是**纯函数**（输入配置 → 输出字节数），滑杆联动才可能是重算而不是换图。

组件自带自检（`LabEngine.ledger.panel(host, trace, model)`），它跑三件事并只在三件全过时报绿：
逐步对拍（模型的预测 == trace 里 Python 从 `arr.size` 数出来的字节数）、缩放性质（公式的性质，
自洽但错误的公式过不了）、以及**对照组**（故意写错的模型必须被抓到，否则这组对拍没有区分力）。
和 `verify.js` 的任意跳转检查同一形状，理由也一样。

**账本契约 lint 有两份**：`labs/traces/kv_cache.py` 的 `lint_ledger()`（权威，写在 trace 前跑）
与 `ledger.js` 的 `lint()`（JS 移植，让页面内编辑 trace 也能立刻得到同样的判决）。**两边的规则
必须同步改，且每边都要有对应的破坏用例** —— 只有一侧有规则的规则，那一侧等于没测。
`trace-model.js` 是引擎的**基础**契约 lint（读写一致性、graph 覆盖、三档三元组、哨兵），
与本票新增的账本规则是两套；在 `kv_cache.json` 上两者都必须报 0。

**trace 由 `scripts/build-labs.mjs` 在构建期内联**：页面里写 `<!-- trace:NAME -->`，
构建时替换成 `labs/traces/NAME.json` 的内容（包在 `window.LabTraces.NAME` 里）。
这样页面与 JSON 不可能漂移 —— 它们就是同一份数据。JSON 缺失会直接构建失败。

**带参数滑杆的 lab 用 trace 集合**，写 `<!-- traces:SET -->`：构建时读
`labs/traces/SET.manifest.json`，把清单（`window.LabTraceSets.SET`）和它列出的**每一份**
trace 一起内联。清单由 trace 生成脚本自己写出，所以「有哪些配置」只在产生数据的地方定义一次 ——
滑杆位置不可能指向一份不存在的 trace，manifest 列了但没生成的也会直接构建失败。
（文件大小按 lab 数翻倍增长，但 [R02 实测](../docs/research/size-budget.md)：trace 相对 KaTeX
字体的体积是零头，压缩后更小。）

**验收脚本**（都在真 Chromium 里跑完整条清单并出截图到 `labs/pages/shots/`，截图入库，
审阅时不必自己跑一遍）：

```bash
npm run build:labs && python3 scripts/verify-labs.py           # 引擎：L00 页面
npm run build:labs && python3 scripts/verify-tiling-stage.py   # 内存层级舞台视图
npm run build:labs && python3 scripts/verify-l01-params.py     # L01 滑杆与 Roofline
npm run build:labs && python3 scripts/verify-ledger.py         # 显存账本视图组件
npm run build:labs && python3 scripts/verify-gantt.py          # 甘特视图 + L10
npm run build:labs && python3 scripts/verify-ring.py           # 环形拓扑视图 + L13
npm run build:labs && python3 scripts/verify-flash-attention.py # L06 旗舰
npm run build:labs && python3 scripts/verify-l05.py            # L05 KV Cache 旗舰
npm run build:labs && python3 scripts/verify-l04.py            # L04 Decoder Block
```

`verify-l04.py` covers L04's five acceptance criteria (#18): the residual add's
shape confirmation together with the counterexamples that make it mean something
(three candidate shapes that broadcast SILENTLY and three that raise, all six
run rather than described), the LayerNorm axis made decidable by drawing the
normalized array's row statistics, its column statistics and the same formula
reduced over the other axis, SwiGLU's three paths with every hop's dimensions on
screen, the Pre/Post-Norm contrast measured by torch autograd to depth 32, and
the parameter account recomputed off the sliders. It carries the same control
groups the other harnesses do — the parameter count may not move with the step,
Post-Norm's residual RMS must hold flat while Pre-Norm's climbs, the two
Pre/Post charts must share one vertical scale — plus a geometry control that
hand-builds the violations (an element pushed past its panel's right edge, a
cell grid squeezed away from its declared aspect ratio) and requires the same
predicates to flag each one.

`verify-l05.py` covers L05's five acceptance criteria (#19): the replay with the
cache growing frame by frame, the with/without-cache shape comparison, the
ledger component being consumed rather than re-implemented, the three parameter
sliders recomputing the ledger, and the Prefill/Decode Roofline. It carries the
same control groups the other harnesses do — the parameters segment may not move
with the context, the Roofline's reference point must be the same dot in every
configuration, and the two attention modes must be *identical* at prefill — plus
a geometry control that hand-builds the violations (a dot pushed out of the
frame, a label moved onto a neighbouring dot, a rectangle squeezed away from its
declared aspect ratio) and requires the same predicates to flag each one.

> ⚠️ **`npm run build:labs` 不是可选的，也不是一次性的。** 验收脚本驱动的是
> `public/labs/` 里的**暂存产物**，不是 `labs/` 源文件。合并或拉取之后不重跑它，
> 脚本就会对着**上一次构建的页面**断言：新加的参数集不在里面，报出来的却是
> `trace.steps must be an array` 或读不到某个 `LabTraceSets` 条目 —— 看起来像代码坏了，
> 实际是构建陈旧。**每次 `git merge` / `git pull` 之后先跑一次 `build:labs`。**

`verify-labs.py` 覆盖引擎本身的契约。其中「任意跳转」同时跑纯函数重建与**故意做错的有状态
对照组** —— 只有对照组确实失败，纯函数的「0 次不一致」才算数；两者都过时脚本会报「无区分力」
而不是通过。`verify-ledger.py` 覆盖视图组件：以 L05 的真实 trace 驱动，逐步核对渲染出的字节数、
检查二维矩阵的 ZeRO 切分、驱动滑杆验证重算的缩放关系，并读回页面自检的结论。

**`verify-labs.py` 里的 lint 对照组是 L00 专属的**：它是一组写死在 `verify.js` 里、针对 L00 那份
trace 的破坏（`delete tensors.x`、`delete steps[2].bindings.MOLD.idx` …）。换一份 trace 时其中
几条会变成空操作，于是 `LabEngine.verify.panel` 会打出一个与当前 trace 无关的红 verdict。
所以新 trace 的验收页应当直接调**通用的** `LabEngine.verify.runJumpCheck`，
而不是整个 `verify.panel`（账本页就是这么做的）。`verify.js` 里那组破坏值得改成从 trace 派生，
但那要动引擎文件，留给后续票。

`labs/pages/*.html` 会被 `scripts/build-labs.mjs` **拍平**拷进 `public/labs/`，再由 Astro 原样复制到 `dist/`。所以：

- 页面里的相对路径按**发布后的位置**写，不是按源码位置。`labs/pages/x.html` 发布后是 `public/labs/x.html`，与 `assets/` 同级 —— 引用 KaTeX 要写 `./assets/vendor/katex/…`，写成 `../assets/…` 会在构建期被脚本拦下并报错。
- `/labs` 索引页**不在**这里。它由 Astro 渲染（`src/pages/labs/index.astro`），依赖图数据在 `src/utils/labRoadmap.ts`。两处都会写 `dist/labs/index.html`，Astro 胜出，所以这里放 `pages/index.html` 只会是永不生效的死代码 —— 脚本会拒绝构建。

## 加一个 lab

1. 写 `traces/<name>.py`，跑出 `traces/<name>.json`（必须带与 PyTorch 参考实现的对拍断言，
   并跑一遍 lint；`online_softmax.py` 是范本——它同时演示了 lint 与 lint 的对照组）。
   有参数滑杆的话，为每个合法组合各出一份 trace，并让脚本自己写
   `traces/<name>.manifest.json`（`default` + `params` + `traces` + `labels`）。
2. 写 `pages/<name>.html`：放一个 `<!-- trace:<name> -->`（单份）或
   `<!-- traces:<name> -->`（整组）占位符，用 `./assets/`
   下的共享引擎与 KaTeX，末尾调 `LabEngine.lab(trace, config)`。页面本身不写算法逻辑。
3. 在 `src/utils/labRegistry.ts` 加一条映射，让对应教程正文顶部出现入口卡片；把 `published` 置 `true`。
4. 在 `src/utils/labRoadmap.ts` 把对应节点的「已上线」点亮（`labId` 已在表中）。
5. `npm run build`。构建期会校验：guide id 真实存在、依赖边不悬空、页面资源可解析。

## 两条不能破的约定

- **lab 相关改动只落在新文件里**（决策 D02）。不要改 `docs/guides/**/*.md` 的 frontmatter 或正文 —— fork 追着 upstream，碰正文会把合并冲突面放大到 21 篇。教程侧的入口卡片由 `src/utils/labRegistry.ts` 驱动。
- **lab 页面加 `data-pagefind-ignore`**（决策 N01），且必须挂在包住整个文档的元素上（`<body>`）。挂在 `<meta>` 上 Pagefind 不认，页面照样进索引。`/labs` 索引页反过来 —— **要**被索引。

## 常用命令

```bash
npm run build        # 完整构建（含 build:labs 阶段）
npm run build:labs   # 只做 labs/ -> public/labs/ 的拷贝与校验
npm run dev          # 先 build:labs 再起 dev server
npm run vendor:katex # katex 升版后重新 vendor（rules 见脚本头部注释）
```
