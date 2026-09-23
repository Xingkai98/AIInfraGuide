# 交互学习实验室 · 源码树

设计文档见 `docs/plans/interactive-labs.md`。这里是 lab 的**源码**；`public/labs/` 是构建产物。

## 布局

```
labs/
├── assets/            所有 lab 页面共享（方案 A：共享 assets 目录）
│   ├── engine/        通用播放器（**尚未落地**，P0 引擎 ticket 建）
│   └── vendor/katex/  本地 KaTeX（由 scripts/vendor-katex.mjs 生成，勿手改）
├── traces/            每个 lab 一个 Python 轨迹生成器（**不会**被发布）
└── pages/             每个 lab 一个自包含 HTML
```

`labs/pages/*.html` 会被 `scripts/build-labs.mjs` **拍平**拷进 `public/labs/`，再由 Astro 原样复制到 `dist/`。所以：

- 页面里的相对路径按**发布后的位置**写，不是按源码位置。`labs/pages/x.html` 发布后是 `public/labs/x.html`，与 `assets/` 同级 —— 引用 KaTeX 要写 `./assets/vendor/katex/…`，写成 `../assets/…` 会在构建期被脚本拦下并报错。
- `/labs` 索引页**不在**这里。它由 Astro 渲染（`src/pages/labs/index.astro`），依赖图数据在 `src/utils/labRoadmap.ts`。两处都会写 `dist/labs/index.html`，Astro 胜出，所以这里放 `pages/index.html` 只会是永不生效的死代码 —— 脚本会拒绝构建。

## 加一个 lab

1. 写 `traces/<name>.py`，跑出 trace JSON（必须带与 PyTorch 参考实现的对拍断言）。
2. 写 `pages/<name>.html`，内联 trace，用 `./assets/` 下的共享引擎与 KaTeX。
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
