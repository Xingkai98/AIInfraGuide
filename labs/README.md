# 交互学习实验室 · 源码树

设计文档见 `docs/plans/interactive-labs.md`。这里是 lab 的**源码**；`public/labs/` 是构建产物。

## 布局

```
labs/
├── assets/            所有 lab 页面共享（方案 A：共享 assets 目录）
│   ├── engine/        通用播放器（见下）
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

**trace 由 `scripts/build-labs.mjs` 在构建期内联**：页面里写 `<!-- trace:NAME -->`，
构建时替换成 `labs/traces/NAME.json` 的内容（包在 `window.LabTraces.NAME` 里）。
这样页面与 JSON 不可能漂移 —— 它们就是同一份数据。JSON 缺失会直接构建失败。

**带参数滑杆的 lab 用 trace 集合**，写 `<!-- traces:SET -->`：构建时读
`labs/traces/SET.manifest.json`，把清单（`window.LabTraceSets.SET`）和它列出的**每一份**
trace 一起内联。清单由 trace 生成脚本自己写出，所以「有哪些配置」只在产生数据的地方定义一次 ——
滑杆位置不可能指向一份不存在的 trace，manifest 列了但没生成的也会直接构建失败。
（文件大小按 lab 数翻倍增长，但 [R02 实测](../docs/research/size-budget.md)：trace 相对 KaTeX
字体的体积是零头，压缩后更小。）

**引擎的验收脚本**：`npm run build:labs && python3 scripts/verify-labs.py`。
它在真 Chromium 里跑完整条验收清单并出截图到 `labs/pages/shots/`（截图入库，
审阅时不必自己跑一遍）。其中「任意跳转」同时跑纯函数重建与**故意做错的有状态对照组** ——
只有对照组确实失败，纯函数的「0 次不一致」才算数；两者都过时脚本会报「无区分力」而不是通过。

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
