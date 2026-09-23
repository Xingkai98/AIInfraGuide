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

### 轨迹数据模型

> **本节的字段是 P01 原型验证后的结果**，不是草案。原型（`prototype/`）用一份手写的 10 步 online softmax trace 真跑过，报告见 `docs/research/engine-prototype-findings.md`。标 **⚠️ 已修正** 的字段是原型发现的设计缺口。

```jsonc
{
  "meta": { "title": "Online Softmax", "source": "docs/guides/模块二/5.2", "config": { "N": 8, "Bc": 4 } },
  "tensors": {
    // ⚠️ 已修正：必须给 init。没有任何 step 写 x，但第一步就读它；
    // 没有 init，render(trace, 0) 从第一步起就是错的（纯函数重建的必要条件）
    "x":   { "shape": [8], "dtype": "fp32", "at": "HBM", "role": "input",
             "init": [0.83, -0.20, 1.40, 0.35, 0.12, -1.05, 2.10, 0.44] },
    "Q_i": { "shape": [4, 16], "label": "SRAM" }
  },
  "graph": {
    "nodes": [ { "id": "step1.max", "kind": "op", "phase": "统计量更新" } ],
    "edges": [ { "from": "x", "to": "step1.max", "tensor": "x_block" } ]
  },
  "steps": [
    {
      "id": "step1.max",
      "title": "第 1 块：更新运行最大值 m",
      "formula": "m^{(j)} = \\max\\left(\\slot{MOLD},\\; \\slot{BLKMAX}\\right)",
      "bindings": {
        // ⚠️ 已修正：slot 的值一律是 LaTeX 片段，且每档显式给出，不靠猜
        "MOLD":   { "sym": "m^{(j-1)}",     "idx": "m^{(0)}",  "num": "-\\infty" },
        "BLKMAX": { "sym": "\\max_i x^{(j)}_i", "idx": "\\max_i x^{(1)}_i", "num": "0.83" }
      },
      "reads": ["x"], "writes": ["m"],
      // ⚠️ 已修正：state = "这一步之后所有发生变化的张量值（含中间张量）"，
      // 且**记全量不记 delta** —— 这是"任意跳转不需要反向操作"的前提条件
      "state": { "m": "0.83", "l": "1.0" },
      "regions": { "CORR": "\\exp(\\slot{MOLD} - \\slot{MNEW})" },
      "narration": "第一次进来时 m 是 -∞，所以直接取本块最大值。"
    }
  ]
}
```

**P01 验证后新增/修正的契约要点**：

| 要点 | 说明 |
|---|---|
| **`tensors[].init`** | 已修正。每个被读的张量必须有初始值来源，否则纯函数重建从第一步就是错的。 |
| **±∞ / NaN 哨兵** | 已修正。JSON 表示不了 `-Infinity`。原型靠 Python `json.dumps` 输出裸 `-Infinity`（合法 JS、非法 JSON）绕过，正式版**必须定一个哨兵**（推荐字符串 `"-\infty"`）。 |
| **`step.state` 语义** | 已修正。写死为「这一步之后所有发生变化的张量值（含中间张量），记**全量**不记 delta」。 |
| **slot 值一律是 LaTeX** | 已修正。否则 trace 作者写 `1/2` 会拿到斜体分数。`-∞` 要写成 `-\infty`。 |
| **`kind` 字段** | 保留（原型实测：`comm` 虚线框紫点 / `state` 点线框绿点 / `op` 实线蓝点，视觉上确实可区分）。 |
| **`regions`** | 新增。公式中需要框出高亮的片段（如修正因子），用 `\region{...}` 标记。 |

**留给后续 lab 的两个真问题**（P0 不用管，但契约要给逃生舱）：

- **`kind: "alloc"`**（改外部资源、不改张量，如 PagedAttention 分配物理块）的 `state` 写什么？→ L09 定。
- **KV cache 逐 token 追加**若不想记全量快照（O(N²) 体积），需要 delta + 反向操作。→ L05/L09 定。契约**预留**可选的 `snapshot` / `inverse` 字段，**但现在不实现**——别为了 L05/L09 给所有 lab 增加复杂度。

**通用视图组件四种**：**DAG 画布 / 张量检查器 / 公式面板 / 时间轴**。每种 lab 再挂各自主视图。

**DAG 分层必须补「顺序边」**（P01 实测的硬结论）：只按数据依赖分层，`load` 这类「只读不写、无前驱」的步骤会卡在第 0 层，层宽随块数线性增长（10→74 节点时最宽层从 3 涨到 19），画布变成 8:1 的横条。补上合成的顺序边（`step i → i+1`，只参与分层、不画出来）后每层恰好 1 个节点。

**折行阈值的判断依据**（I01 落地时修正了 P01 的一处自相矛盾）：P01 报告正文写「≤10 节点单排可读」，但它自己引用的证据截图文件名是 `03-dag-10-seq-wrap4.png`——**wrap4 就是折行成 4 列**。按 I01 的实际几何（`w=78, xgap=104`）算，10 节点单排是 1752×63 ≈ **28:1**，比折行要防止的 8:1 还糟。所以**即使 10 层也要折**。I01 的做法不是硬编码阈值，而是**按面板自身宽高比自动选列数**（枚举候选列数、算画布形状、取最接近的），18 层时 50.9:1（单排）vs 3.85:1（折行）。

**lint 是 trace 的一部分**：原型的 `lint()` 规则（读写一致性、graph/steps 一致性、三档完整性、三元组形状、`state` 里的未声明张量、narration 混入 LaTeX）应搬到 Python 侧或 CI 里，`labs/traces/*.py` 输出 JSON 后立刻自查。原型验证过 lint 的有效性：干净 trace 报 0，故意破坏的副本报 11 个。

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

**公式渐进绑定**：公式面板提供三档显示——符号式 → 代入索引的式子 → 代入数值的式子。三档的数据在 trace 里**显式给出**（每个绑定值给 `{sym, idx, num}` 三元组），不靠约定从 key 里猜。这是"公式要 LaTeX 展示"落到实处的方式。

**Lab 依赖图**：`/labs` 索引页本身用一张 DAG 展示各 lab 的前置关系（L00 → L06 → L07 是硬依赖链），让读者知道从哪进。

**数值对拍**：每个 `labs/traces/*.py` 必须包含与 PyTorch 参考实现的断言。跑在**独立的 GitHub workflow**（`.github/workflows/trace-checks.yml`），在 PR 与 push 到 `feat/**` / `research/**` 时触发，只装 CPU 版 torch、只跑 `labs/traces/`，与部署 workflow 完全解耦。这是"真实算法回放"承诺的兜底。

**容差按精度定，不按惯例定**：`#37` 当初建议 `1e-5`（fp32 相对容差），但 I01 实测发现——trace 的递推与 torch 参考实现**都在 float64 下算**，两者之间唯一的误差来源是求和顺序，所以容差可以紧到 **`1e-12`**。这不只是"更严格更好"：在 4 位显示精度下，`1e-5` 会**掩盖一个恰好四舍五入到同一数字的错误公式**，而 `1e-12` 能抓到它。**新 trace 按自己的 dtype 定容差并写明理由**，不要照抄这个数。

**trace 的 lint 是对拍之外的第二道闸**：`online_softmax.py` 里除数值断言外，还会用引擎的契约规则自查（读写一致性、graph/steps 一致性、三档完整性、`state` 里的未声明张量、narration 混入 LaTeX 等）。**且它自带对照组**——同时 lint 一份故意破坏的副本，确认能抓到（I01 实测 10 种破坏全部抓到）。脚本在没有通过全部断言与 lint 时**拒绝写文件**。

**trace 不进 `labs/traces/` 之外的任何地方**：构建时由 `scripts/build-labs.mjs` 内联进页面（替换页面里的 trace 标记），所以 lab 页面是真正的"一个 HTML + 共享 assets"、零额外请求。`labs/traces/` 本身**从不暂存**——暂存只走 `labs/pages` 与 `labs/assets` 两个白名单，这条保证由构造给出，不是靠一条可能写错的排除规则（T02 的设计）。

**数据契约**：小张量（≤4096 元素）全量展开成嵌套数组；超出阈值的只记录变更切片。DAG 节点 = **算法步骤**（不是算子调用），带 `kind` 字段区分 `op` / `comm` / `alloc` / `state`，这样 PagedAttention 的"分配物理块"、Ring AllReduce 的"send-recv 累加"也能归一到同一套表示。

**共享视图组件**：四类跨 lab 复用的视图（显存账本 / 甘特 / 内存层级舞台 / 环形拓扑）在引擎阶段**一次性抽成通用组件**并定好接口，之后的 lab 只消费不修改。这是 17 个 lab 能真并行的前提——每个 lab 的文件集是 `traces/<name>.py` + `pages/<name>.html`，互不重叠。

**播放器交互模型**：

- **不自动播放**，加载时停在第一步。读者从教程正文点进来，需要先看清"这是什么东西"再让它动起来。
- **支持 `?step=N` 深链**。教程入口卡片可以精确指向某个教学时刻（如"修正因子第一次被触发的那一步"），而不只是链到开头。游标变化用 `history.replaceState` 更新 URL，不污染历史栈。
- **基本键盘支持**：空格播放/暂停（需 `preventDefault` 以免滚动页面）、左右箭头单步、Home/End 到首尾；DAG 节点可 Tab 聚焦、回车跳转。

**窄屏降级**：桌面优先。`< 1024px` 时走**统一**的降级组件（不是各 lab 各自实现响应式）：lab 标题与说明 + 静态降级内容（trace 第一步或总览图）+ 返回教程正文的链接。理由是 lab 的信息密度是刚性的——DAG 画布、张量热力图、公式面板、时间轴四者需要同时可见才有教学效果，375px 宽下强行排布只会两头不讨好。

**搜索**：lab 页面加 `data-pagefind-ignore`（正文很薄，真正内容在内联 trace 里，不是可读文本，进搜索只会稀释教程正文排名）。因此**入口卡片是读者发现 lab 的唯一路径，必须做得显眼**。`/labs` 索引页是真正的可读内容，**要**被索引。

---

## 四、工程方案

**分支**：`feat/interactive-labs`，worktree 在 `/home/shared/code/AIInfraGuide-labs`，从 `fork/main` 切出。最终 PR 提到个人 fork。

**自包含的取舍**（实测数据见 `docs/research/size-budget.md`，已推翻本文档此前的估算）：

主形式是**方案 A：共享 assets 目录**——每个 lab 一个 HTML + 一个共享 `labs/assets/`（vendor 的 KaTeX + 引擎）。整包拷走即可离线打开，不发外部请求。

选 A 的真正理由**不是体积而是缓存结构**：

| | 首访 | 第二个 lab 起 | 21 个 lab 累计 |
|---|---:|---:|---:|
| A 共享 assets | 263 KB | **5.7 KB** | **377 KB** |
| B 单文件 | 265 KB | 265 KB | 5.5 MB |

正确内联后 B 的首访并不比 A 重多少（0.6%），但 A 的后续页面只需付新的 HTML + trace（站点实测 `max-age=600` + `ETag`），B 每页都要重发引擎与字体。**读者看得越多，差距越大**（21 个 lab 时 14.7 倍）。

另配一个 `--inline` 打包脚本按需生成单文件分享版，**但必须用"聪明版"写法**：JS/CSS/trace 以**文本**内联，base64 **只用于字体**（对已压缩的 woff2，gzip 后 base64 仅 +1%；而对文本文件 base64 会破坏 gzip 的 LZ77 匹配，多花 50%）。朴素全 base64 版实测比正确版多 17%。

**目录契约**（T02 落地时细化，比本节早期版本的说法更好）：`labs/` 是**入库的源码树**，`public/labs/` 是**它的 gitignore 构建产物**，由 `scripts/build-labs.mjs` 作为 `npm run build` 与 `npm run dev` 的第一步生成。选这个拆分而不是把源码直接放 `public/` 下，有一个结构性理由：**`labs/traces/` 绝不能发布出去**，而这个布局让这条保证**由构造保证**，而不是靠一条可能写错的排除规则。附带好处是暂存时会先清空目标目录，删掉的 lab 页不会残留成陈旧产物。

**站点集成**：加一个 Astro 的 `/labs` 索引页（含依赖 DAG，数据在 `src/utils/labRoadmap.ts`），并在对应教程正文的 `<Content />` 之前插入入口卡片（`src/components/LabEntryCard.astro`）。**入口卡片通过 `src/utils/labRegistry.ts` 驱动，不由 frontmatter 字段驱动**——后者需要改动约 21 篇教程正文，会显著放大从 upstream 合并时的冲突面。实测结果：`docs/guides/` **零改动**，`GuideContent.astro` 只加了 10 行。

**两个只有检查产物才发现得出来的坑**（写在这里免得后面 17 个 lab 重踩）：

1. **`data-pagefind-ignore` 必须包在 `<body>` 上，加在 `<meta>` 标签上完全无效**——Pagefind 照样索引。构建日志看页数正常，要解压索引分片才能发现。
2. **暂存会把 `labs/pages/x.html` 拍平到 `public/labs/x.html`**，于是页面与 `assets/` 同级，`../assets/` 会 404 掉整个 KaTeX。正确写法是 `./assets/`。已加构建期资源解析检查并做了反向测试。

**guide id 的格式陷阱**：Astro 内容集合的 glob loader 会 **slug 化** id 且**含完整目录路径**——`5.2-CUDA Online Softmax实现.md` → `模块二-cuda编程与算子优化/52-cuda-online-softmax实现`。照文件名手写 key **不会报错，只会静默不渲染卡片**。`labRegistry.ts` 文件头写明了格式与获取真实 key 的方法（构建后读产物），并有断言让未知 guide id 直接构建失败。

**KaTeX vendoring**：`npm i katex` 后把 `dist/` 拷进 `labs/assets/vendor/katex/`，**保持原始目录结构**（`katex.min.css` 必须与 `fonts/` 同级，否则 CSS 里的相对字体路径会断）。**只带 woff2**（浏览器只取这一种；带上 ttf/woff 会让 vendor 目录胖 3 倍）；**字体取全量 20 个**，这是明确的保守取舍——接受首访从约 145 KB 涨到约 349 KB，换取不冒任何缺字形风险（`\bigoplus`、`\mathbb` 等生僻符号）。

**公式渲染**（⚠️ 本节此前写「拆成静态骨架 + 动态数值 span，只更新数值文本」，**P01 实测证明照做会在屏幕上打出 LaTeX 源码**）：

原因是**触发替换的恰恰是符号式公式**——`\slot{BLKMAX}` 在符号档要替换成 `\max_i x^{(j)}_i`，这本身就是 LaTeX，直接 `textContent` 会把反斜杠原样打到屏幕上。原型里真出现了。

**正确做法**（已验证）：骨架按「步骤 × 档位」构建一次并复用，每个 slot 的值**单独跑一次小 KaTeX 渲染**塞进 KaTeX 自己生成的 `<span>`。三个实测细节：`\htmlId` / `\htmlClass` 必须配 `trust: true`（否则静默不生成元素）；**不要用 KaTeX 的 macro 回调取双花括号参数**（KaTeX 交回的是反序词法 token，`\slot{MOLD}` 到手是 `"DLOM"`），改为自己写花括号配对的预处理；`\region` 的 body 里常嵌 `\slot`，预处理**必须递归**。

`\region` 高亮用 `\htmlClass` 即可，**不要用 Range / `surroundContents`**——跨 KaTeX 的 span 边界时必然抛 `InvalidStateError`（原型已排除这条死路）。

**性能不是这么做的理由**：实测全量渲染只要 **0.35 ms/帧**（60fps 预算的 2%），骨架复用 0.008 ms/帧——**两者都不是瓶颈**。真正的大头是 DOM（重绘 DAG 的 `innerHTML` 重建要 0.47 ms/帧，比 KaTeX 还大）。所以：**拆分方案值得做，但理由应该是它顺带解决了 `\region` 高亮**；若 `\region` 用别的方式实现，每帧全量重渲 KaTeX 是最简单且够快的方案。优化精力应花在 DOM 增量更新（改 class 而非重建，顺带解决节点焦点丢失）上。

---

## 五、分期路线图

| 阶段 | 交付 | Lab |
|---|---|---|
| **P0** | 引擎 + 样板，验证抽象 | L00 |
| **P1** | 单卡 Transformer 计算链 | L01–L05 |
| **P2** | CUDA 算子回放（旗舰） | L06–L07 |
| **P3** | 推理引擎调度 | L09–L12 |
| **P4** | 分布式训练 | L13–L17 |
| **P5** | 进阶（需先补正文，前置已立票） | L18–L21 |

P0 必须先做且要扎实——L06 会用到 L00 的双层内存舞台，L13/L16 会用到 L01 引入的时间轴与甘特组件。引擎抽象若 P0 没验证好，P2 之后要返工。

**本轮范围 = P0–P4 共 17 个 lab**（含分布式）。P5 三章（量化 / 投机解码 / PD 解耦）对应的教程章节目前只有「本章简介」、没有正文，补写正文已作为独立任务立票，并作为那三个 lab 的阻塞边。L20 MoE 有正文，不受影响。

---

## 六、素材复用（R01 盘点结论）

完整的 96 项素材盘点见 `docs/research/asset-inventory.md`。要点：

- **mermaid 远比静态图有用**：教程正文里 55 段 mermaid 中，43 段结构上可机械转成 DAG、20 段与 12 个 lab 直接相关。Mermaid 的 `graph` 语法本身就是「节点 + 边 + 标签」，与引擎的 `graph` 字段同构。
- **转换策略**：先手工转换 L09 的 Block Table 二分图与 L13 的环形/树拓扑，凝练成小解析器，验证「转出来的 DAG 真能被引擎用」之后再决定是否推广到其余 18 段。转换在**构建期离线执行**，不在 lab 页面里跑 mermaid。
- **最就绪的四个 lab：L09 / L13 / L04 / L17**，其中 L04 现在就能做出可看页面（两张纯示意图可直接嵌入 + 两段 Pre/Post-Norm mermaid 天然构成对照）。
- `dp_overlap1/2/3.svg` **不是三个并列方案，是同一个算法在三种 bucket 大小下的三帧**——正好是 L17 参数滑杆要扫的区间。
- **8 个 lab 零素材**（L03/L07/L15/L16/L18–L21），其中六个的素材缺口与正文缺口同源，应排在一起做。

---

## 七、已定的设计决策

三轮 batch grill 全部问完（完整索引见 wayfinder map）。本文档的早期版本列过三个「待定」问题，均已落定：

1. **自包含程度** → 方案 A 共享 assets + 智能 `--inline`。字体：**只 vendor woff2、但取全量 20 个**（不裁剪）——这是一个明确的保守取舍，见 §四。
2. **站点集成方式** → `labs/` 源码树（构建期暂存到 `public/labs/`）+ Astro `/labs` 索引页 + `labRegistry.ts` 驱动的入口卡片。
3. **Lab 取舍** → 本轮 P0–P4 共 17 个；P5 排后，其正文缺口已单独立票。
