# 交互学习实验室 · 设计文档

> 状态：设计已定稿，尚未实现。
> 仓库：`Xingkai98/AIInfraGuide`（个人 fork，upstream 为 `caomaolufei/AIInfraGuide`）
> 分支：`feat/interactive-labs`（worktree：`../AIInfraGuide-labs`）

## 目标

把仓库现有教程里"读得懂但想不清楚"的算法，做成**可一步步回放的学习页面**。每页展示：

1. **执行 DAG** —— 整个计算流程的计算图，当前节点高亮，已走过的变暗，未到的虚化，可点击跳转。
2. **维度全景** —— 每一条数据边上标注形状（shape），张量检查器可逐层拆开。
3. **公式逐帧绑定** —— 每一步的 LaTeX 公式，并把符号**逐级代入**（符号式 → 代入索引 → 代入数值）。
4. **回放控制** —— 播放 / 单步 / 回退 / 拖动时间轴 / 调速，加参数滑杆。

核心承诺是**真实算法的回放**：页面上看到的每个数字都来自真实可运行的实现，不是手编的演示数据。

## 非目标

- 不做在线代码编辑器（仓库已链接 kkcocoa 的 CUDA 手撕算子练习平台，不重复）。
- 不做习题判题系统。
- 不改动现有教程正文的组织结构（只在教程顶部加实验室入口）。

---

## 一、引擎架构：播放器 + 轨迹分离

这是整个方案最关键的架构决定。

```
labs/
├── engine/              通用播放器（一份，所有 lab 复用）
│   ├── player.js        时间轴 / 步骤调度 / 键盘快捷键
│   ├── dag.js           SVG DAG 渲染 + 分层布局 + 高亮
│   ├── tensor.js        张量检查器（数值网格 + 热力图 + 维度括号）
│   ├── formula.js       KaTeX 包装 + 符号绑定
│   ├── views/           专用视图：ring / gantt / radix / ledger / tiling-stage
│   ├── lab.css
│   └── vendor/katex/    本地 KaTeX（不走 CDN）
├── traces/              轨迹生成器（Python）
│   ├── online_softmax.py
│   ├── flash_attention_v1.py
│   └── ...
└── pages/               每个 lab 一个自包含 HTML
    ├── 01-online-softmax.html
    └── ...
```

**约定：lab 页面本身不含算法逻辑，只含一份 trace JSON。** trace 由 `traces/*.py` 里的真实 PyTorch/NumPy 实现跑出来——脚本"每做一步就 dump 一帧"，记录张量形状、数值、以及本步对应的公式符号表，输出 JSON，构建时内联进 HTML。

三个好处：

- **保证真实性** —— 数字来自真跑，不会出现"为了好画而编一组假数据"。
- **数值可对拍** —— 每个 trace 脚本带断言，与 PyTorch 参考实现比对，CI 里跑，防止动画与算法脱节。
- **引擎与内容解耦** —— 新增 lab = 写一个 trace 脚本 + 一份视图配置，不用重写播放器。

### 轨迹数据模型（草案）

```jsonc
{
  "meta": { "title": "Online Softmax", "source": "docs/guides/模块二/5.2", "config": { "N": 8, "Bc": 4 } },
  "tensors": {
    "x":   { "shape": [8], "dtype": "fp32" },
    "Q_i": { "shape": [4, 16], "label": "SRAM" }
  },
  "graph": {
    "nodes": [ { "id": "step1.max", "op": "max", "phase": "统计量更新" } ],
    "edges": [ { "from": "x", "to": "step1.max", "tensor": "x_block" } ]
  },
  "steps": [
    {
      "id": "step1.max",
      "title": "第 1 块：更新运行最大值 m",
      "formula": "m^{(j)} = \\max\\left(m^{(j-1)},\\; \\max_i x^{(j)}_i\\right)",
      "bindings": { "j": 1, "m^{(j-1)}": "-\\infty", "\\max_i x^{(j)}_i": "0.83", "m^{(j)}": "0.83" },
      "reads": ["x_block"], "writes": ["m"],
      "state": { "m": 0.83, "l": 1.0 },
      "narration": "第一次进来时 m 是 -∞，所以直接取本块最大值。"
    }
  ]
}
```

通用视图组件四种：**DAG 画布 / 张量检查器 / 公式面板 / 时间轴**。每种 lab 再挂各自主视图。

---

## 二、内容大纲：21 个 Lab

统一用一个**小到能全画出来的示例模型**贯穿单机部分：

| 符号 | 含义 | 取值 |
|---|---|---|
| `N` | 序列长度 | 8 |
| `d_model` | 模型维度 | 64 |
| `H` | 注意力头数 | 4 |
| `d_head` | 每头维度 | 16 |
| `d_ff` | FFN 中间维度 | 176 |
| `vocab` | 词表大小 | 32 |
| `B` | 批大小 | 1 |

每一课都用同一套配置，读者的空间感能累积。

### Phase 0 · 引擎 + 样板

**L00 Online Softmax** — 对应教程 `5.2-CUDA Online Softmax实现` / `第2章-数学基础` §5.3–5.5

选它当样板：最短、最闭环，且是 FlashAttention 的必备前置。

- **回放主线**：标准三遍扫描（求最大值 → 求指数和 → 归一化）→ 改造为分块递推。每来一块：更新 `m` → 用修正因子 `e^(m_old - m_new)` 更新 `ℓ` 和 `O`。
- **维度重点**：一维输入 `x∈R^8` → 分块后 `x^(j)∈R^4` → 统计量 `m, ℓ` 是**标量** → 输出 `O∈R^4`。这三层维度是整个 lab 的教学重点。
- **公式**：三条递推式逐帧代入。
- **专用视图**：**修正因子放大器** —— 每步把 `e^(m_old - m_new)` 单独高亮，并显示"如果不修正，结果偏差多少"。
- **对照模式**：朴素 softmax（加 +1000 偏置直接 NaN）vs 三遍安全版 vs online 版，三条轨迹并排。
- **参数**：`N`、块大小 `B_c`、温度 `T`。

### Phase 1 · Transformer 计算链（单卡前向，5 个 Lab）

**L01 GEMM：索引、分块与访存** — 对应 `4.1-CUDA GEMM算子性能优化` + `第2章-数学基础` §3

- **回放主线**：先把 `C_ij = Σ_k A_ik B_kj` 展开成单个输出点的点积，再铺满整个 `C`，然后引入 tiling——三层循环 `(i,j)` 块循环 × `k` 块循环，每次迭代显示 `A_tile/B_tile` 从 global → shared → register 的搬运。
- **维度**：`A[8,8] × B[8,8] → C[8,8]`，`B_M=B_N=B_K=4`。
- **专用视图**：**内存层级舞台**（HBM/SMEM/REG 三层，数据块在层间移动）+ **访存计数器**（全局访存 vs 计算次数）。
- **参数**：`B_M/B_N/B_K` 滑杆 → 访存次数与算术强度实时重算，右侧 Roofline 图上的点跟着动。

**L02 Reduce：顺序 / 共享内存树 / Warp Shuffle** — 对应 `第3章 经典算子实现-Reduce`、`3.1-CUDA Reduce算子优化`

- **回放主线**：三种实现在同一份输入上并排跑。`x[16]` → 顺序归约 → 树形归约（每轮 `stride` 减半，活跃线程数减半）→ warp shuffle。
- **专用视图**：**线程阵列 + 归约树**同步动画 + 每步访存次数累计条 + bank conflict 热点着色。

**L03 Self-Attention 全流程** — 对应 `3.3-Self-Attention机制深入理解`、`Transformer架构快速入门` §3

- **回放主线**：`X --W_q,W_k,W_v--> Q,K,V` → reshape 分头 → `S = QK^T / √d_h` → 因果 mask → softmax → `O = PV` → 合并头 → `W_o`。
- **维度重点**：全程标注，尤其 `[8,64] → [4,8,16]` 这个拆头/并头的 reshape，以及 mask 为什么是 `[8,8]` 加到 `[H,8,8]` 上要广播。
- **专用视图**：注意力矩阵热力图（每个 head 一张小图）+ mask 遮盖可视化。
- **参数**：`N`、`H`、`d_h`、是否 causal、缩放系数。

**L04 Decoder Block：残差 / LayerNorm / SwiGLU** — 对应 `3.6-LayerNorm与残差连接`、`3.7-Transformer Decoder Block完整解析`

- **回放主线**：完整 block，`x → LN → MHA → +x → LN → FFN → +x`。
- **维度重点**：残差相加要求两侧 shape **完全相同**（直观展示为什么 `d_model` 一路不变）；LayerNorm 是**在 `d_model` 维上**对每个 token 独立归一化。
- **公式**：`LN(x) = γ·(x-μ)/√(σ²+ε) + β`；`SwiGLU(x) = W_down(SiLU(W_gate x) ⊙ W_up x)`。
- **对照**：Pre-Norm vs Post-Norm 的梯度尺度变化。
- **参数**：`d_model`、`d_ff`、norm 位置、激活函数。

**L05 KV Cache 与自回归生成** — 对应 `3.8-从Transformer到LLM自回归生成`、`1.1-LLM推理基础`

- **回放主线**：逐 token 生成。每个新 token：追加一行到 KV Cache → 只用最后一个 query 算 attention → 输出。
- **维度重点**：`K_cache` 从 `[1,H,d_h]` 长到 `[t,H,d_h]`；无 cache 时是 `[t,d]×[t,d]`，有 cache 时是 `[1,d]×[t,d]`——同一张图上把两次的形状摆出来。
- **专用视图**：**显存账本**（参数/梯度/优化器状态/KV Cache 堆叠条）+ Prefill/Decode 两阶段在 Roofline 上的位置迁移。
- **参数**：上下文长度、层数、精度（fp16 / fp8 / int8）→ 账本实时重算。

### Phase 2 · CUDA 算子回放

**L06 FlashAttention V1** ★ 旗舰 — 对应 `6.1-FlashAttention V1详解`

- **回放主线**：外循环 `i` 遍历 Q 块，内循环 `j` 遍历 K/V 块。每轮：载入 `K_j, V_j` → `S_ij = Q_i K_j^T` → 行最大值 → `P_ij = exp(S_ij - m_new)` → 用修正因子更新 `ℓ_i, O_i` → **`S, P` 从不落回 HBM**。
- **维度重点**：SRAM 里同时驻留 `Q_i, K_j, V_j, O_i` 各 `B×d`，加 `S_ij` 是 `B_r×B_c` —— 这就解释了公式 `B_c = ⌈M/4d⌉` 的来源。
- **专用视图**：**HBM / SRAM 双层舞台**，数据块在两层间搬运；旁边挂 IO 计数器，实时对比标准 Attention 的 `O(N^2)` 与 FA 的 `O(N^2d^2/M)`。
- **参数**：`N`、`d`、`B_r`、`B_c`、SRAM 容量 `M` → IO 次数与 SRAM 占用实时重算。
- **前置**：必须先看 L00。

**L07 FlashAttention V2 增量** — 对应 `6.2-FlashAttention V2详解`

- **回放主线**：不重跑全流程，而是**与 V1 并排做 diff**——循环顺序从 K 外 Q 内改成 Q 外 K 内、去掉了对 `O` 的两次 rescale、non-matmul FLOPs 下降。
- **专用视图**：A/B 双轨播放器（同一份数据、同一个步进）+ FLOP/IO 差异条。

**L08 Softmax 的算子视角**（可选）— 对应 `5.1-CUDA Softmax朴素实现优化`、`5.2-CUDA Online Softmax实现`

把 L00 的算法回放换成 kernel 视角：一个 block 内多个 warp 怎么分工处理一行、warp 内 shuffle 归约求 max/sum。若 L00 已讲透，可砍。

### Phase 3 · 推理引擎调度

**L09 PagedAttention** ★ — 对应 `2.1-PagedAttention`

- **回放主线**：三个请求陆续到达，每个按 `block_size=4` 分配逻辑块 → 空闲物理块池按需分配 → 共享前缀触发 CoW → 显存紧张触发抢占（swap 或 recompute）。
- **专用视图**：**Block Table 映射连线**（左边逻辑块序列，右边物理块池）+ 显存占用条 + **碎片对照**（连续分配 vs 分页）。
- **维度重点**：物理块是 `[block_size, H, d_h]` 三维张量，逻辑块数 `⌈len/block_size⌉`；Kernel 视角显示 `physical_block_number × stride + offset` 地址计算。
- **参数**：`block_size`（16/8/4）、请求到达序列、最大并发数、是否开 prefix cache。
- **素材**：教程里的 mermaid 图是现成雏形。

**L10 Continuous Batching 调度** ★ — 对应 `2.2-Continuous Batching`

- **回放主线**：**迭代级调度循环**。每个 tick 是一次完整调度：从 waiting 队列取请求（受 token budget 约束）→ running 队列各生成一步 → 处理完成/抢占 → 组成新 batch。
- **专用视图**：**甘特图** + 每 tick 的 batch 组成堆叠条 + 吞吐/延迟计数器。
- **对照模式**：Static Batching 甘特图（长尾请求拖死整批）vs Continuous Batching。
- **参数**：请求到达率、生成长度分布、`max_num_seqs`、token budget。

**L11 Chunked Prefill 与统一调度** — 对应 `2.4-Chunked Prefill 与统一调度`

- **回放主线**：长 prefill 被切成 chunk，与 decode 混排在同一个 tick 里，token budget 是唯一货币。
- **专用视图**：token budget 分配堆叠条 + ITL/TTFT 曲线实时绘制。
- **对照**：关掉 chunked prefill 时 ITL 的尖刺。

**L12 Prefix Cache 与 RadixAttention** — 对应 `2.3-Prefix Cache 与 RadixAttention`

- **回放主线**：多个请求的 prompt 进来 → 块哈希链 → 基数树插入/匹配/分裂 → 引用计数 → LRU 淘汰。
- **专用视图**：**基数树生长动画** + 块哈希链 + 命中高亮 + 淘汰时显存回收。
- **参数**：请求序列（控制前缀共享度）、缓存容量、淘汰策略。

### Phase 4 · 分布式训练

**L13 Ring AllReduce** ★ — 对应 `2.1 集合通信原语详解` / `6-集合通信基础` / `collective-communication-primer`

- **回放主线**：4 卡环形。第一阶段 ReduceScatter 共 `N-1` 步（每步每卡发给下游一个 chunk、从上游收一个 chunk 并累加）；第二阶段 AllGather 共 `N-1` 步。总计 `2(N-1)` 步，每步可单步看。
- **专用视图**：**环形拓扑 + 每卡缓冲区状态表 + 步骤时间轴 + 通信量累计器**，四者同步。
- **维度重点**：`S` 切成 `N` 个 chunk，每步传输 `S/N`；对比"朴素 `2(N-1)S`"与"Ring `2(N-1)S/N`"两条累计曲线。
- **参数**：卡数（2/4/8）、数据量 `S`、链路带宽 → 实时算耗时与 busbw。
- **对照**：Ring vs Tree。

**L14 ZeRO-1/2/3 显存账本** — 对应 `第5章-ZeRO系列`、`3.1-优化器原理与显存开销分析`

- **回放主线**：一个训练 step 的完整时间线（前向 → 反向 → 梯度归约 → 优化器步），每阶段展示每张卡显存条的变化。
- **专用视图**：**显存账本矩阵**——横轴 baseline/Z1/Z2/Z3，纵轴参数/梯度/优化器状态/激活；再加一条通信量代价曲线。
- **参数**：模型参数量、卡数、精度、是否 offload。

**L15 张量并行：Column / Row Parallel** — 对应 `第6章-张量并行与序列并行`

- **回放主线**：一个 MLP 层 → 列并行切 `W_1`（切输出维）→ 各卡算部分 → 激活 → 行并行切 `W_2`（切输入维）→ AllReduce 合并。
- **维度重点**：每卡 shape 逐帧变化，以及"为什么列并行后不用通信、行并行后必须 AllReduce"。
- **专用视图**：**多卡并排数据流**（rank0..rank3 各一列）+ 通信节点高亮 + 跨卡 shape 检查器。
- **参数**：TP 度、切分方式、GQA/MQA 下 KV 头怎么分。

**L16 流水线并行 1F1B** — 对应 `第7章-流水线并行`

- **回放主线**：4 stage × 8 micro-batch 的 1F1B 调度逐步展开。
- **专用视图**：**Gantt（时间 × stage）** + 气泡占比 + 激活显存峰值曲线。
- **对照**：GPipe（全前向再全反向，峰值高）vs 1F1B。
- **公式**：气泡率 `(P-1)/(M+P-1)`。

**L17 DDP 梯度分桶与通信重叠** — 对应 `4.1-数据并行详解`、`4.2-PyTorch 数据并行从原理到实战`

- **回放主线**：反向传播逐层产生梯度 → 按 bucket 聚合 → bucket 满即触发 AllReduce → 与后续层反向计算重叠。
- **专用视图**：反向时间轴 + bucket 状态条 + 计算/通信重叠 Gantt。
- **素材**：`public/images/dp_overlap1..3.svg` 是现成素材。
- **参数**：bucket 大小、卡数。

### Phase 5 · 进阶（部分需先补教程正文）

**L18 量化：FP16 → INT8/INT4** — 对应 `第4章-量化`（该章目前只有简介，需先补正文）

- **回放主线**：逐值量化 `x_int = round(x/s) + z`，再反量化看误差。对比 per-tensor / per-channel / per-group 三种粒度。
- **专用视图**：数值对照表 + 误差热力图 + 分布直方图（带裁剪/饱和曲线）。
- **参数**：bit 宽、粒度、对称/非对称、校准集。

**L19 投机解码** — 对应 `第5章-Speculative-Decoding`（需先补正文）

- **回放主线**：draft 模型生成 `K` 个 token 及分布 → target 模型一次前向验证 → 逐个接受/拒绝 → 拒绝时按修正分布重采样。
- **专用视图**：逐 token 的两组概率分布并排 + 接受判定 + 期望加速比计数器。
- **公式**：接受概率 `min(1, p/q)`、修正分布 `norm(max(0, p-q))`。

**L20 MoE 专家并行与 All-to-All** — 对应 `第10章-MoE并行`

- **回放主线**：token → gate 打分 → top-k 路由 → All-to-All 分发 → 专家计算 → All-to-All 收回 → 加权合并。
- **专用视图**：token–专家路由矩阵热力图 + 分发/回收通信动画 + 负载不均衡度（容量因子）。

**L21 PD 解耦架构** — 对应 `第7章-PD解耦架构`（需先补正文）

- **回放主线**：请求在 prefill 池与 decode 池之间的流转 + KV 传输。
- **专用视图**：两个资源池的队列联动 + KV 传输链路 + TTFT/TPOT 对比曲线。

---

## 三、横切设计

**维度标注规范（所有 lab 统一）**：张量一律 `名称 [d0, d1, ...] @ 位置` 三段式；张量检查器里维度和数值网格用括号对齐；任何一次 reshape/permute 都要画"维度搬家"的连线，不能只写结果。

**公式渐进绑定**：公式面板提供三档显示——符号式 → 代入索引的式子 → 代入数值的式子，用一个开关切换。这是"公式要 LaTeX 展示"落到实处的方式。

**Lab 依赖图**：`/labs` 索引页本身用一张 DAG 展示 21 个 lab 的前置关系（L00 → L06 → L07 是硬依赖链），让读者知道从哪进。

**数值对拍**：每个 `traces/*.py` 必须包含与 PyTorch 参考实现的断言，CI 里跑。这是"真实算法回放"承诺的兜底。

---

## 四、工程方案

**分支**：`feat/interactive-labs`，worktree 在 `../AIInfraGuide-labs`，从 `fork/main` 切出。最终 PR 提到个人 fork。

**自包含的取舍**：KaTeX 约 1MB 的 JS + 字体无法优雅塞进每一页。两个方案：

- **A（推荐）共享 assets 目录** —— 每个 lab 一个 HTML + 一个共享 `assets/`（vendor 的 KaTeX + 引擎），整包拷走即可离线打开，不发外部请求。再加一个 `--inline` 打包脚本，需要单文件分享时把 asset 内联进去。
- **B 每页真正单文件** —— 便于随手分享，代价是单页约 1.5MB。

**站点集成**：lab 页面放 `public/labs/`，加一个 Astro 的 `/labs` 索引页（含依赖 DAG），并在对应的教程正文顶部插一张"交互实验室"入口卡片。**不改动现有 content collection**，零回归风险。

**KaTeX vendoring**：KaTeX 目前只是 `rehype-katex` 的传递依赖，`node_modules` 里没装，需要显式 `npm i katex` 再 vendor 到 `labs/engine/vendor/`。

---

## 五、分期路线图

| 阶段 | 交付 | Lab |
|---|---|---|
| **P0** | 引擎 + 样板，验证抽象 | L00 |
| **P1** | 单卡 Transformer 计算链 | L01–L05 |
| **P2** | CUDA 算子回放（旗舰） | L06–L07 |
| **P3** | 推理引擎调度 | L09–L12 |
| **P4** | 分布式训练 | L13–L17 |
| **P5** | 进阶（部分需先补正文） | L18–L21 |

P0 必须先做且要扎实——L06 会用到 L00 的双层内存舞台，L13/L16 会用到 L01 引入的时间轴与甘特组件。引擎抽象若 P0 没验证好，P2 之后要返工。

---

## 六、待定的设计决策

1. **自包含程度** —— 共享 assets 目录（推荐共 A）还是每页真正单文件（B）？
2. **站点集成方式** —— `public/labs/` + Astro 索引页（推荐），还是完全独立于 Astro？
3. **21 个 Lab 的取舍** —— P0–P3（12 个）是否为最小完整闭环？P4/P5 是否本轮排入？
