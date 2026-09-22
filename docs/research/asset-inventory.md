# R01 · 交互学习实验室素材盘点

> 对应 ticket：issue #8「调研：仓库现有可视化素材盘点与复用边界」
> 设计依据：`docs/plans/interactive-labs.md` §二各 lab 的「专用视图」「素材」小节
> 盘点分支：`research/asset-inventory` · 日期：2026-09-22

## 盘点范围与方法

| 项 | 数量 | 方法 |
|---|---|---|
| `public/images/` 位图（PNG） | 36 | 逐张直接查看图像内容 |
| `public/images/` 矢量（SVG） | 4 | 读 XML 源码判断结构；并用 `cairosvg` 渲染成 PNG 后直接查看 |
| `public/images/` 其他 | 1（`caomaolufei_vx.jpeg`） | — |
| `docs/guides/` 正文内嵌 mermaid 代码块 | 55 | 脚本提取块体 + 前文标题上下文 |

合计 **96 个素材**。引用关系（某素材被哪篇教程引用）由 `grep` 确证。

**证据强度说明**：

- PNG 内容是我逐张看图得出的**直接观察**，不是推断。ticket 里"PNG 你无法直接看图内容"的预设不成立，所以本报告对 PNG 的描述可以按确证对待。
- `dp_overlap1/2/3.svg`、`dp_scaling.svg` 的源码没有 `<text>` 节点（文字被转成了字形轮廓 `<path>`），grep 读不出语义。我装了 `cairosvg` 把它们渲染成位图后直接查看，因此这四张图的语义同样是**确证**。
- 唯一属于**推断**的部分：某个素材"是否值得做成某 lab 的哪个专用视图"——这是教学价值的判断，不是事实。
- 与 ticket 描述的两处出入：SVG 实际是 **4 个**（ticket 说 5 个），另有 1 个 JPEG；`dp_scaling.svg` 未被 ticket 列出但确实存在。

**一个对结论影响很大的事实**：仓库**已经**在客户端渲染 mermaid——`src/layouts/Layout.astro:373` 从 jsDelivr CDN 加载 `mermaid@11`，把 `pre[data-language="mermaid"]` 转成 `<div class="mermaid">` 再 `mermaid.run()`。也就是说这 55 段 mermaid 在站点上**已经是活图**，同时它们的源码又是 Mermaid 自己就能解析的节点/边列表。**"已经渲染成图"和"结构可机械提取"这两个条件同时成立**，这是它们比 PNG 更适合作为 lab 素材的根本原因。

---

## 一、图片素材主表（41 项）

归类图例：**可直接复用** = 纯示意图，嵌进 lab 页面即可；**可机械转 DAG** = 节点与边可从素材里提取出来；**需重绘为交互版** = 有教学价值但现有形态是静态的，必须重做；**仅作参考** = 画得好但没有可提取结构，或超出 lab 范围。

### L00 Online Softmax

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无图片素材） | — | L00 | — | `5.2-CUDA Online Softmax实现.md` 无图，只有 1 段 mermaid（见 §二） |

### L01 GEMM：索引、分块与访存

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `gemm-1.png` | `public/images/`；引用：`4.1-CUDA GEMM算子性能优化.md:47` | L01 | **需重绘为交互版** | 内容：A 中一行、B 中一列、C 中一个红格高亮——即 `C_ij = Σ_k A_ik B_kj` 的单个输出点。这正是 L01 回放主线的**第一步**（"把 C_ij 展开成单个输出点的点积，再铺满整个 C"）。静态图无法展示 k 的遍历过程。 |
| `gemm-2.png` | 同上；引用：`4.1:164` | L01 | **需重绘为交互版** | 内容：Block 级 tiling——B 上一列 `Bn` 宽条沿 K 方向下移、A 上一行 `Bm` 高条沿 K 方向右移、C 上 `Bm×Bn` 红块。即"沿 K 以步长 BK 迭代搬运 A/B 子块"的过程。对应 L01 专用视图**内存层级舞台**，必须做成数据块在 HBM→SMEM 间移动的动画。 |
| `gemm-3.png` | 同上；引用：`4.1:343` | L01 | **需重绘为交互版** | 内容：Thread 级 tiling——`Bk=8` 的细条（A 侧 `8×8` 橙块、B 侧 `8` 宽蓝条）喂给 `Bm×Bn` checkerboard 中的绿色小块，块内标 `0`。对应 L01 三级搬运链的第三级（SMEM→REG）。与 `4.1:326` 的 mermaid 是同一件事，图更细。 |
| `gemm-4.png` | 同上；引用：`4.1:469` | L01 | 仅作参考 | 内容：warp 级分块，`Bm=128, Bn=128` 的 C 被切成 warp_1 独占右上 `64×32`、warp2~7 各占 `64×64`，左上 `32×64` 块内标出 0~31 的 lane 编号。解释了"为什么每个线程负责 8×8"的线程—warp 映射，但 L01 的参数面只有 `B_M/B_N/B_K`，不涉及 warp 形状。留作进阶或 L02 的线程阵列视图参考。 |
| `gemm-5.png` | 同上；引用：`4.1:639` | L01 | 仅作参考 | 内容：`Wn=64` 下 float4 访存的 quarter-warp 划分（4 行 × 8 列，lane 0~31），用于分析 bank conflict。属 L01 范围外的微架构细节。 |
| `gemm-6.png` | 同上；**孤儿素材**（`grep` 全文无引用） | L01 | 仅作参考 | 内容：`Wn=64 / Wm=32` 的 quarter-warp 线程映射，与 gemm-5 同族，展示另一种 Z-order 排布。当前未被任何教程引用。 |
| `gemm-7.png` | 同上；引用：`4.1:715` | L01 | 仅作参考 | 内容：`8×8` 计算区域拆成 4 个 `4×4` 子块（红框标出 quarter-warp 0~3），C 侧出现 4 个 `8×4` 分块。bank conflict 方案的第四种解法，超出 lab 范围。 |
| `gemm-8.png` | 同上；引用：`4.1:778` | L01 | 仅作参考 | 内容：方案三 Z-order 线程映射，C 侧 4 个 `8×8` 分块各自内部按 `0,2,4,6 / 1,3,5,7` 交织编号。同上。 |
| `gemm-C_tile_bolcks.png` | 同上；引用：`4.1:228` | L01 | 仅作参考 | 内容：`C_BLOCK_X/Y = 16` 的 C_BLOCK 阵列 + 右侧 `C_Tile(C_BLOCKS)` 的 `Tm=Tn=8` 分布，并用箭头标出"假设 c_thread_y=2, c_thread_x=3，线程写入每个 C_BLOCK 内相同位置 (2,3)"。解释了跨步写入模式，属实现细节。 |

### L02 Reduce：顺序 / 共享内存树 / Warp Shuffle

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `reduce-cuda-kernel-01.png` | `public/images/`；引用：`3.1-CUDA Reduce算子优化.md:39` | L02 | **需重绘为交互版** | 内容：一棵橙色的二叉树形规约图，8 个叶子（3,1,7,0,4,1,6,3）逐层汇聚到根 25。这张图就是 L02 专用视图**"线程阵列 + 归约树"**的静态版——但静态图看不到"每轮 stride 减半、活跃线程数减半"的时序。 |
| `reduce-cuda-kernel-02.png` | 同上；引用：`3.1:80` | L02 | **需重绘为交互版** | 内容：V0 交错寻址的四轮完整过程（Values 行 + Thread IDs 行成对出现，Step 1~4 / Stride 1,2,4,8）。每轮的 Values 数组都被完整画出来（10,1,8,-1,0,-2,3,5…→ 41），并在活跃 tid 上套橙圈。这是 lab 里"每步访存次数累计 + bank conflict 热点着色"要逐帧重放的东西。 |
| `reduce-cuda-kernel-03.png` | 同上；引用：`3.1:133` | L02 | **需重绘为交互版** | 内容：V1 消除 warp divergence 版，结构同 02，但活跃 Thread IDs 连续（0,1,2,3…），且用红框标出分化的边界。三个实现并排跑的对照正是 L02 的回放主线。 |
| `reduce-cuda-kernel-04.png` | 同上；引用：`3.1:208` | L02 | **需重绘为交互版** | 内容：V2 步长反转版，Stride 从 8→4→2→1，Thread IDs 始终是低位连续（0,1,2,3…）。bank conflict 热点着色所需的信息（每轮哪些 tid 访问哪些地址）在这张图里已经隐含，但需要重画成可交互的。 |

### L03 Self-Attention 全流程

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无素材） | — | L03 | — | 主要对应教程 `3.3-Self-Attention机制深入理解.md` 里 `img src` 与 `mermaid` 出现次数均为 **0**。L03 是 P1 里唯一零素材的 lab，attention 热力图、mask 可视化、拆头 reshape 全部要新建。 |

### L04 Decoder Block：残差 / LayerNorm / SwiGLU

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `decoder-blocks.png` | `public/images/`；引用：`Transformer架构快速入门.md:741` | L04 | **可直接复用** | 内容：宏观结构——`Token Embedding + RoPE` → `Decoder Block × 32`（内含 `LayerNorm → Masked MHA → Residual Add` 与 `LayerNorm → FFN (SwiGLU) → Residual Add`）→ `LayerNorm → LM Head → Softmax → Next Token`。这是一张干净的纯示意图，既是 L04 的入口概览，也是 L05 自回归生成的结构前提。 |
| `decode-only.png` | 同上；引用：`3.7 Transformer Decoder Block完整解析.md:65`, `Transformer架构快速入门.md:545` | L04 | **可直接复用** | 内容：Decoder-only block 微观图，虚线框内 `LayerNorm → Masked Self-Attention → +（残差）→ LayerNorm → FFN → +（残差）→ Output`，且 **MHA 与 FFN 两个框内直接写了逐步公式**（`(Q,K,V投影 → RoPE → QK^T/sqrt(d_k) → Mask → Softmax → PV → 输出投影)`、`(W_gate*x → Swish, W_up*x, 逐元素相乘 → W_down)`）。这是 Pre-Norm 版，与 L04 的公式与对照设计直接吻合。 |
| `decoder-block.png` | 同上；引用：`3.2 Transformer全貌及代码实现.md:143` | L04 | 仅作参考 | 内容：**Encoder-Decoder 架构下的 Decoder block**，含 `Multi-Head Cross-Attention`（标注 `Q ← 来自上一步输出 / K, V ← 来自 Encoder 输出`）。LLM 不用 cross-attention，所以结构本身不能复用到 L04，但"用箭头标出 Q/K/V 来源"的画法值得沿用。 |
| `encode-block.png` | 同上；引用：`3.2 Transformer全貌及代码实现.md:106` | — | 仅作参考 | 内容：Encoder block（无 mask、无 cross-attention），`输入 x:(N_src, d_model) → MHSA → + → LayerNorm → FFN → + → LayerNorm → 输出:(N_src, d_model)`。与 L04 无关，是 Post-Norm 结构的好例证。 |
| `Transformer.png` | 同上；引用：`3.2:62`, `Transformer架构快速入门.md:63` | L04 | 仅作参考 | 内容：Transformer 原始架构全图（左 Encoder 右 Decoder，带英文注释说明各子层）。历史对照用。 |
| `Decoder-only.png` | 同上；引用：`Transformer架构快速入门.md:76` | L04 / L05 | 仅作参考 | 内容：Decoder-only 堆叠图——`Input Token Vectors → Position Embedding → Decoder Block × N → Output Token Vectors`，右侧放大出单个 block 的内部（`Layer Norm → Masked Self-Attention → + → Layer Norm → FFN → +`）。与 `decode-only.png` 内容高度重合但注释更少、无公式，被后者完全覆盖，故不作首选。 |

### L05 KV Cache 与自回归生成

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无图片素材） | — | L05 | — | 主源 `3.8 从Transformer到LLM自回归生成深入理解.md` 的 `img src` 与 `mermaid` 计数均为 0。显存账本、KV Cache 增长条、Roofline 迁移全部要新建。`1.1-LLM推理基础.md` 有 3 段 mermaid 可复用（见 §二）。 |

### L06 FlashAttention V1 ★ 旗舰

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `flashattentionv1-0.png` | `public/images/`；引用：`6.1-FlashAttention V1详解.md:79` | L06 | **需重绘为交互版**（三合一） | 内容：**左**是内存层级金字塔，直接标了 L06 需要的全部数字——`GPU SRAM: 19 TB/s (20 MB)` / `GPU HBM: 1.5 TB/s (40 GB)` / `Main Memory (CPU DRAM): 12.8 GB/s (>1 TB)`；**中**是 FA 数据流，标出 `Outer Loop`（沿 K 列向右、沿 V 列向右）与 `Inner Loop`（沿 Q 列向下），以及 `Copy Block to SRAM`、`Compute Block on SRAM`、`Output to HBM` 四个动作，`Q: N×d` / `K^T: d×N` / `V: N×d` / `sm(QK^T)V: N×d` 四个张量形状齐全；**右**是 PyTorch vs FlashAttention 的 ms 堆叠柱（Matmul/Dropout/Softmax/Mask/Fused Kernel）。这三块恰好是 L06 的三件套（双层内存舞台 + IO 计数器的量纲 + 融合收益），但全部是静态的，必须重画成可播放、可调 `B_r/B_c/M` 的版本。**这是全仓库最有价值的单张图，也是最贵的一张。** |
| `flashattentionv1-1.png` | 同上；引用：`6.1:177` | L06 | **可直接复用**（兼交互版蓝本） | 内容：QKV 分块策略。左半 HBM 侧画出 Q/K/V 三列分块，每个块标 `B_r×d`（蓝）/`B_c×d`（绿）/`B_c×d`（橙），并把当前块 `Q_i`/`K_j`/`V_j` 高亮；旁注 `外层循环 j 固定 → 内层循环 i=1..T_r 遍历 Q`；右半 SRAM 侧画出四个驻留张量 `Q_i (B_r×d)` / `K_j (B_c×d)` / `V_j (B_c×d)` / `S_ij = Q_i·K_j^T (B_r×B_c)`，并用三条"加载"虚线连起来。**形状标注已经完整到可以直接进 lab 的维度标注体系**，作为静态示意图可立即嵌入；同时它是 HBM/SRAM 双层舞台的现成视觉蓝本。 |

### L07 FlashAttention V2 增量

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `flashattentionv3-1.png` | `public/images/`；**孤儿素材** | L07（弱相关） | 仅作参考 | 内容：Warpgroup 甘特图——Warpgroup 1 / 2 两行，绿(`GEMM0`)/红(`Softmax`)/橙(`GEMM1`) 三种块交错排列，横轴 time，虚线切分轮次。这是 **FA3** 的 warp specialization 调度，不是 V1/V2 的增量差异。但"用甘特表达 GEMM/Softmax 交错"的形式对 L07 的 FLOP/IO 差异条有借鉴价值。 |
| `flashattentionv3-2.png` | 同上；**孤儿素材** | L07（弱相关） | 仅作参考 | 内容：`WGMMA0` / `Softmax` / `WGMMA1` 三行 pingpong 调度甘特，块内标 `0,1,2,...,N-1` 轮次，明显的时间阶梯。同上。 |
| `flashattentionv3-3.png` | 同上；**孤儿素材** | — | 仅作参考 | 内容：WGMMA 寄存器布局表（`T0{d0,d1} T1{d0,d1}...` 与 `T0{a0,a1} T1{a2,a3}...` 两组，标注 "FP32 accumulator register WGMMA layout"、"FP8 operand A register WGMMA layout"）。纯微架构细节，与任何 lab 的抽象层级都差得远。 |
| （FA V2 正文素材） | — | L07 | — | `6.2-FlashAttention V2详解.md` 的 `img src` 与 `mermaid` 计数均为 **0**。L07 是"与 V1 并排做 diff"，A/B 双轨播放器完全依赖 L06 的引擎能力与 trace，不需要新素材。 |

### L08 Softmax 的算子视角（可选）

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `CUDA programming model.png` | `public/images/`；引用：`gpu-basics.md:244`, `1.2-CUDA编程模型.md:77`, `CUDA编程入门指南.md:120` | L08（若做） | 仅作参考 | 内容：Host/Device 分栏，Host 侧 Kernel 箭头指向 Device 侧 Grid（`Block (0,0)~(2,1)` 6 个块），再用红色虚线放大出 `Block (1,1)` 内的 `Thread (0,0)~(4,2)` 15 个线程。画的是**执行模型层级**，不是数据流。L08 要的"一个 block 内多 warp 分工处理一行"用不上它，但它对 L02/L08 的线程阵列视图是个好参照。 |
| `cuda内存模型.png` | 同上；引用：`1.3-CUDA内存模型.md:44` | L08 | 仅作参考 | 内容：`Grid → Block(0,0) → 共享内存/寄存器/Thread(0,0),(1,0)/本地内存`，左侧 CPU 通过双向箭头连到全局内存/常量内存/纹理内存。是内存**层级与作用域**图，非数据流。 |

### L09 PagedAttention ★

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无图片素材） | — | L09 | — | 该节完全依赖 mermaid（3 段，见 §二）——恰好是设计文档点名的"教程里的 mermaid 图是现成雏形"。 |

### L10–L12 推理引擎调度

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无图片素材） | — | L10/L11/L12 | — | 这三节的调度结构全部由 mermaid 承载（L10 三段、L11 两段、L12 一段，见 §二）。甘特图、基数树、token budget 堆叠条仍需新建。 |

### L13 Ring AllReduce ★

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无图片素材） | — | L13 | — | Ring/Tree 拓扑、overlap 时序全部是 mermaid（4 段，见 §二）。环形动画、缓冲区状态表、通信量累计曲线需新建。 |

### L14 ZeRO 显存账本

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无图片素材） | — | L14 | — | `第5章-ZeRO系列.md` 是 30 行本章概述 stub，无图无 mermaid。显存账本矩阵要全新建。 |

### L15 张量并行 / L16 流水线并行 1F1B

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无素材） | — | L15 / L16 | — | `第6章-张量并行与序列并行.md`（76 行）、`第7章-流水线并行.md`（79 行）都只有本章概述，`img src` 与 `mermaid` 计数均为 0。多卡并排数据流、1F1B 甘特图全部要新建，且需要先补正文。 |

### L17 DDP 梯度分桶与通信重叠

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `dp_diagram.png` | `public/images/`；引用：`4.1 数据并行详解.md:46` | L17 | 仅作参考 | 内容：DP 总体示意——顶部一排 `Model` 块，下面 `GPU 0/1/2` 三行各画 `Forward pass`（青）与 `Backward pass`（橙）两条时间线，再往下 `Gradients`（粉）与 `Optimization`（绿箭头）汇聚到 `Updated model`。alt 文本：*"数据并行：模型复制到多卡，各卡在不同数据微批上并行前反向，再同步梯度"*。是 L17 的**场景引入图**，但画的是"各卡平行、最后同步"的宏观概念，没有 bucket 粒度——比 `dp_overlap1/2/3` 粗一层。 |
| `dp_overlap1.svg` | `public/images/`；引用：`4.1 数据并行详解.md:157` | L17 | **需重绘为交互版**（★最高价值） | **已渲染确证**内容：一张两行甘特图——上轨 `GPU Computation:`，下轨 `GPU Communication:`。上轨为 `Forward`（3 个 teal 块 0/1/2）→ `Backward`（3 个 orange 块 2/1/0）→ `Optimizer`（虚线竖线分隔）→ 下一个 `Forward`；下轨是**一整条**紫色 `ALL` 块，横跨整个 Backward 区间——即"等整个反向传播算完再统一同步"。alt 文本也印证：*"朴素 DP：先完成整个反向传播，再统一同步梯度，通信期间 GPU 空闲"*。**这正是 L17 专用视图「计算/通信重叠 Gantt」要展示的第一帧**，但它是静态的、不可参数化。 |
| `dp_overlap2.svg` | 同上；引用：`4.1:166` | L17 | **需重绘为交互版**（★最高价值） | **已渲染确证**内容：同样两轨，但下轨变成**细密的小紫块序列**，每块内标 `2`（7 个）、`1`（6 个）、`0`（6 个），与上轨的 Backward 块**横向重叠**。即"bucket 极小 → 通信次数极多、每次只传一层，重叠完美但固定开销高"。 |
| `dp_overlap3.svg` | 同上；引用：`4.1:175` | L17 | **需重绘为交互版**（★最高价值） | **已渲染确证**内容：同样两轨，下轨是**3 个中等宽度**的紫块，分别标 `2`/`1`/`0`，与对应 Backward 块部分重叠。即"bucket 适中 → 每次凑满一桶再发，重叠部分损失但固定开销低"。 |
| `dp_scaling.svg` | 同上；引用：`4.1:190` | L17 | 仅作参考 | **已渲染确证**内容：matplotlib 双子图。左"Throughput Scaling with Data Parallelism"——横轴 DP 8/16/32/64/128/256 的柱状图（40000 → 16000 tokens/sec/GPU），标注性能跌幅 -6.3% / -6.0% / -12.0% / -15.0% / -40.6%；右"Memory Usage Scaling with Data Parallelism"——一条几乎水平的粉线（约 36.6 GB）。是**实测数据图**，不是可提取结构的示意图。它证明的是"加卡只摊薄计算、不摊薄显存"这个结论，作为正文插图更有价值。 |

> **关键发现**：`dp_overlap1/2/3.svg` 这三张图**不是三个并列的示意图，而是同一个算法在 `bucket_cap_mb` = 全部 / 极小 / 中等三种参数下的三帧**——上轨的计算块完全相同，只有下轨的通信块粒度在变。这正好就是 L17 参数滑杆（bucket 大小）要扫的区间。lab 的交互版可以直接把这三张图做成滑杆的三个锚点。颜色语义也已确定可沿用：teal = Forward、orange = Backward、purple = AllReduce Grads（右下角有图例）。

### L18–L21 进阶

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| （无素材） | — | L18/L19/L20/L21 | — | `第4章-量化.md`（33 行）、`第5章-Speculative-Decoding.md`（30 行）、`第10章-MoE并行.md`（64 行）、`第7章-PD解耦架构.md`（33 行）全部是本章简介 stub，无 `img src`、无 `mermaid`。四个 lab 的素材都要新建，且都需要先补正文。 |

### 跨 lab / 非 lab 素材

| 素材 | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `合并访存与非合并访存对比.png` | `public/images/`；引用：`1.3-CUDA内存模型.md`, `2.2-内存访问优化.md` | L01 / L02（辅助） | **可直接复用** | 内容：上下两栏对照。上"非合并访存"——线程束的 4 条箭头散落到 Arr 的 0,1,4,7,10 等分散格；下"合并访存"——4 条箭头聚到 Arr 的 4,5,6,7 四连格。纯示意图、语义自明、无需交互，在 L01 的访存计数器或 L02 的访存次数累计条旁边直接嵌即可。 |
| `GPU hardware architecture diagram.png` | `public/images/`；引用：`gpu-basics.md:77`, `2.3-Occupancy与资源分配.md` | — | 仅作参考 | 内容：A100 芯片平面图（GPC × 7、L2 Cache、HBM 控制器、NVLink/PCIe 标注）。硬件层级图，无数据流，且与任何 lab 的抽象层级不匹配。 |
| `AIInfra优化全流程解析.png` | `public/images/`；引用：`3.1 AI Infra工程师为什么必须懂Transformer.md:303` | — | 仅作参考 | 内容：一张手绘风全景图，把 [Tokenizer]→[Embedding]→`×32 Decoder Blocks`→[De-tokenizer] 主链与旁挂的优化项（`FlashAttention (tiling + online softmax)`、`PageAttention (K, V token)`、量化、投机采样、KV Cache 压缩、Continuous Batching 等）画在一起。信息密度极高，但节点位置是手工排布、无结构标注，提取不出可用的节点/边。适合做 `/labs` 索引页的装饰或全站导览。 |
| `理解AIInfra的职责边界.png` | `public/images/`；引用：`3.1 AI Infra工程师为什么必须懂Transformer.md:39` | — | 仅作参考 | 内容：四层职责分层图（应用层 / 算法与模型层 / AI Infra 层 / 硬件层），AI Infra 层内又分"推理与部署"、"分布式训练"、"CUDA 算子与编译"。同上，是概念分层而非数据流。 |
| `AIinfraGuideWeb.png` / `AIinfraGuideWeb1.png` / `AIinfraGuideWeb2.png` | `public/images/`；引用：`README.md:16,18,231` | — | 仅作参考 | 站点首页截图（"AI Infra Guide / 从零开始深入理解 AI Infra 的全栈核心技术"）。README 装饰用，与 lab 无关。 |
| `caomaolufei.png` | `public/images/`；引用：`src/pages/about/index.astro:52` | — | 仅作参考 | 作者头像。 |
| `caomaolufei_vx.jpeg` | `public/images/`；引用：`src/pages/about/index.astro:160` | — | 仅作参考 | 作者微信二维码。 |
| `wx_qun.png` | `public/images/`；引用：`README.md:310`, `src/pages/about/index.astro:142` | — | 仅作参考 | 微信群二维码。 |

---

## 二、mermaid 代码块盘点（55 段）

同 §一 的图例。`位置` 格式为 `文件:行号`（行号是 ```` ```mermaid ```` 那一行）。

### 与 lab 直接相关的 28 段

| 素材（节点摘要） | 位置 | 关联 lab | 归类 | 理由 |
|---|---|---|---|---|
| `Online Softmax 递推 → 分块 QK^T 增量更新(m,d) → 修正输出 O → FlashAttention`（4 节点链） | `模块二-CUDA编程与算子优化/5.2-CUDA Online Softmax实现.md:646` | L00 | 可机械转 DAG（浅） | 是 L00 完整算法回放的**顶层骨架**，4 个节点是纯线性链，可直接成为 DAG 布局的骨架线。但节点粒度太粗（缺张量形状、缺分块循环），实质内容是它背后的三条递推式，需要补。 |
| `Global Memory →(Block 协作加载 BM×BK / BK×BN)→ Shared Memory →(Thread 独立加载 TM+TN 个 float)→ Register →(TM×TN 次 FMA)→ 计算结果` | `模块二-CUDA编程与算子优化/4.1-CUDA GEMM算子性能优化.md:326` | L01 | **可机械转 DAG** | 边标签里**已经带了搬运量**（`BM×BK / BK×BN`、`TM+TN 个 float`、`TM×TN 次 FMA`）——这正好是 L01 专用视图「内存层级舞台」+「访存计数器」要的两个维度：位置 + 数据量。4 节点 3 边的结构与 L01 的"三级数据搬运链"叙述完全同构，机械提取即可。 |
| `Stage 0: 计算中 → Stage 1: 数据就绪 → Stage 2: 加载中 → Stage 3: 发射 LDG` | `模块二-CUDA编程与算子优化/4.1-CUDA GEMM算子性能优化.md:1012` | L01（弱） | 仅作参考 | CUTLASS 3~4 级流水线的状态环，与 L01 的 `B_M/B_N/B_K` 参数面无关。真要做流水线动画时再回来看。 |
| `256 线程/256 个值 → Warp 内规约(__shfl_down_sync, 8 Warp×32 线程) → 8 个中间值写入 Shared Memory → Warp 0 规约 → 最终结果` | `模块二-CUDA编程与算子优化/3.1-CUDA Reduce算子优化.md:483` | L02 | **可机械转 DAG** | 4 节点 3 边，**完整描述了 warp shuffle 版的两级归约**（Warp 内 → Warp 间），且节点上带了线程数/值数（256、8×32、8 个）。L02 回放主线三种实现里的第三种，可机械转。 |
| `Reduce 性能瓶颈 → {计算效率低/Warp Divergence, Shared Memory 冲突/Bank Conflict, 同步开销大, 访存效率低, GPU 利用不足} → {V1/V2/V4/V5/V6/V7/V3 各自的解法}`（14 节点归因树） | `模块二-CUDA编程与算子优化/3.1-CUDA Reduce算子优化.md:652` | L02（弱） | 仅作参考 | 这是一棵**优化手法归因树**，不是数据流。它把 V0~V7 的每个版本对应到它解决的瓶颈，价值在于给 L02 的"三种实现并排跑"提供解释文案，但节点不能进 DAG（模型里"计算图"和"优化手段"是两个层面）。不排除单独做成一张"优化地图"侧栏。 |
| Post-Norm：`Input(x) → SubLayer → Add → LayerNorm → Output`，另一条 `Input(x) → Add` | `模块一-前置知识/transformer/3.6 LayerNorm与残差连接深入理解.md:422` | L04 | **可机械转 DAG** | 5 节点、带一条跳跃边（`A → C`，即残差）。结构完整，可直接转为 L04 的 DAG。 |
| Pre-Norm：`Input(x) → LayerNorm → SubLayer → Add → Output`，另一条 `Input(x) → Add` | `模块一-前置知识/transformer/3.6 LayerNorm与残差连接深入理解.md:435` | L04 | **可机械转 DAG** | 同上的镜像版本。两段合起来**恰好就是 L04 的「Pre-Norm vs Post-Norm 对照」**——同样 5 个节点、同样的跳跃边，只有 norm 的位置不同。做成 A/B 双轨播放器只需把节点名映射到算子，diff 天然自带。**这是 55 段 mermaid 里唯一现成的"对照实验"素材。** |
| `输入 Prompt 500 tokens → Prefill 并行处理全部 token → 生成第 1 个 token → Decode step 1（输入 1 个 token）→ 生成第 2 个 token → Decode step 2 → … 直到 EOS` | `模块四-推理优化/第1章-LLM推理基础/1.1-LLM推理基础.md:56` | L05 | **可机械转 DAG** | 6 节点线性链，把 Prefill/Decode 两阶段的分界点标得很清楚，可作为 L05 主线的骨架。 |
| `低算术强度/Memory Bound(Decode 在这) → 平衡点 → 高算术强度/Compute Bound(Prefill 在这)` | `模块四-推理优化/第1章-LLM推理基础/1.1-LLM推理基础.md:211` | L05 | 仅作参考 | Roofline 的**口头描述**被画成了 3 节点链——但 L05 要的是真 Roofline 图（横轴算术强度、纵轴算力）以及 Prefill/Decode 两点在其上的**位置迁移**动画。mermaid 表达不了坐标轴，需要 `views/` 里新建一个 roofline 视图。 |
| `📥 Tokenize → Prefill(Compute Bound) → Decode Loop(Memory Bound) → Sampling → （回到 Decode Loop）`，另有 `Decode Loop → 📤 Detokenize` | `模块四-推理优化/第1章-LLM推理基础/1.1-LLM推理基础.md:255` | L05 | **可机械转 DAG** | 6 节点、**含一条回边**（`D → C`，Sampling 回到 Decode Loop）——这是目前唯一自带循环的 mermaid 链，正好对应 L05「逐 token 生成」的自回归主循环。且节点上标了 Compute/Memory Bound，可直接复用为 lab 的瓶颈着色。 |
| `逻辑块0(tok 0-3) --- 逻辑块1(tok 4-7) --- 逻辑块2(tok 8)`；`L0 -.Block Table.-> 物理块 #7`、`L1 → #2`、`L2 → #5` | `模块四-推理优化/第2章-推理引擎核心技术/2.1-PagedAttention.md:80` | L09 | **可机械转 DAG**（★★首选） | 6 个节点、5 条边，**是完整的二分图**：3 个逻辑块节点（`---` 表示连续）+ 3 个物理块节点 + 3 条跨侧映射边（边标签统一为 `Block Table`）。这是 Mermaid 自己就能解析的显式结构，提取成引擎的 `{nodes, edges}` 是纯机械变换。配套的 markdown 表格（`逻辑块 \| 物理块 \| 状态`：`0→#7 已满(4/4)`、`1→#2 已满(4/4)`、`2→#5 部分填充(1/4)`）可直接作为 trace 的初始 state。设计文档原话"教程里的 mermaid 图是现成雏形"指的就是它。 |
| `请求A 块表 → 共享物理块(ref_cnt=2)`、`请求B 块表 → 共享物理块`；`共享物理块 -.请求B 要写入.-> 触发 Copy-on-Write 复制出副本 → 请求B 私有块` | `模块四-推理优化/第2章-推理引擎核心技术/2.1-PagedAttention.md:153` | L09 | **可机械转 DAG** | 5 节点、4 条边，含 `ref_cnt=2` 这个状态量。但它是**一个瞬间的快照**——L09 的回放需要的是"ref_cnt: 2 → 触发 CoW → 复制 → ref_cnt 变为 1/1 → 块表改指"这个**时序**。mermaid 给的是静态拓扑，时序要靠 trace 补。 |
| `Scheduler(每步调度决策) →(allocate_slots/free)→ KVCacheManager(请求视角) →(get_new_blocks/free_blocks)→ BlockPool(全局视角)`；`BlockPool → FreeKVCacheBlockQueue`、`BlockPool → BlockHashToBlockMap` | `模块四-推理优化/第2章-推理引擎核心技术/2.1-PagedAttention.md:224` | L09（弱） | 仅作参考 | 5 节点的**软件模块分工图**（vLLM V1 的类职责）。边标签是方法名，不是张量。它回答"代码怎么组织"，而 lab 要回答"数据怎么流动"——两者抽象层级不同，不宜进 DAG。可作为 lab 页脚"真实实现对应关系"的注释。 |
| `Static Batching：请求A(20步完成) →(被迫等待480步)→ 整批结束`、`请求B(500步) → 整批结束`、`请求C(50步) →(被迫等待450步)→ 整批结束` | `模块四-推理优化/第2章-推理引擎核心技术/2.2-Continuous Batching.md:53` | L10 | 仅作参考（文案价值高） | 4 节点、3 条边，`subgraph` 名就叫「批内互相拖累」。但 L10 专用视图是**甘特图**，而这张图把"等待 480 步"写成了边标签——数字是对的（20/500/50），结构却表达不出时间轴。**数字本身可直接喂给对照模式的甘特数据。** |
| `步 t: A B C D 在批中 → A 完成退出拉入 E → 步 t+1: B C D E → C 完成退出拉入 F → 步 t+2: B D E F` | `模块四-推理优化/第2章-推理引擎核心技术/2.2-Continuous Batching.md:77` | L10 | **可机械转 DAG** | 5 节点的迭代级调度循环，`subgraph` 名「随退随补」。每步的 batch 组成（ABCD / BCDE / BDEF）是**显式写出来的**，可直接作为甘特图每 tick 的行数据。这是 L10 回放主线最直接的结构来源。 |
| `waiting 队列(优先级队列: FCFS/PRIORITY) →(schedule() 拉入)→ running 列表`；`running →(完成/抢占)→ 退出 or 打回 waiting`；`running →(token_budget 耗尽)→ 本步不再拉新` | `模块四-推理优化/第2章-推理引擎核心技术/2.2-Continuous Batching.md:135` | L10 | **可机械转 DAG** | 4 节点、4 条边，含一条**回边**（`running → 退出 or 打回 waiting`）——是调度器的状态机。正好是 L10 每 tick 循环的骨架，`token_budget` 也作为边标签出现了（对应 lab 的 token budget 参数）。 |
| `step t：4000-token Prefill + Decode×8 → 整步耗时被 Prefill 拉长，8 个 Decode 全部卡顿`（`subgraph` 名「长 Prefill 独占一步」） | `模块四-推理优化/第2章-推理引擎核心技术/2.4-Chunked Prefill 与统一调度.md:41` | L11 | 仅作参考（文案价值高） | 2 节点线性链，标题就是结论。L11 要的是 ITL/TTFT 曲线上的**尖刺**——这是曲线数据，mermaid 表达不了。数字（4000 token、Decode×8）可用于构造对照场景。 |
| `step t：Prefill块1(512) + Decode×8 → step t+1：Prefill块2(512) + Decode×8 → ... 每步 Decode 都平稳推进`（`subgraph` 名「Chunked Prefill：切块掺入」） | `模块四-推理优化/第2章-推理引擎核心技术/2.4-Chunked Prefill 与统一调度.md:61` | L11 | 仅作参考（同上） | 3 节点线性链。与上一条构成 L11 的"开/关 chunked prefill"对照，但两者都只是文案骨架。 |
| `新请求前缀分块 → 逐块计算哈希 → {缓存命中?} →(命中) 块表指向已有物理块/跳过 Prefill`、`→(未命中) 正常 Prefill/登记新块哈希`；两侧汇合 → 继续处理未命中部分 | `模块四-推理优化/第2章-推理引擎核心技术/2.3-Prefix Cache 与 RadixAttention.md:87` | L12 | **可机械转 DAG** | 6 节点、5 条边，**含一个菱形判断节点**（`C{"缓存命中?"}`）和两条带标签的分支边——这是 55 段里分支结构最完整的一段。L12 的"基数树匹配/分裂"动画可以从这个决策骨架直接生长出来。 |
| `rank 0 → rank 1 → rank 2 → rank 3 → rank 0`（4 节点环） | `模块一-前置知识/communication/collective-communication-primer.md:300` | L13 | **可机械转 DAG** | 4 节点、4 条边构成**有向环**——正是 Ring AllReduce 的拓扑本身。结构极简但完全正确，且 L13 **没有任何图片素材**，这段 mermaid 是它唯一的现成结构。`graph LR` 的线性布局需要换成环形布局（引擎的 `views/ring`）。 |
| Tree AllReduce：`rank 1 → rank 0(根)`、`rank 2 → rank 0`、`rank 3 → rank 1`、`rank 4 → rank 1`、`rank 5 → rank 2`、`rank 6 → rank 2` | `模块一-前置知识/communication/collective-communication-primer.md:336` | L13 | **可机械转 DAG** | 7 节点、6 条边的**二叉树**，是 Ring 的对照拓扑。节点数（7）与 L13 参数（卡数 2/4/8）不匹配，需要按 N 重新生成，但"树形归约"的父子规则是明确的。 |
| `participant GPU(计算)` / `participant NET(网络传输)`；`GPU→GPU: 反向传播：第 L 层梯度就绪`、`GPU→NET: 触发 AllReduce 第 L 层(异步)`、`GPU→GPU: 第 L-1 层梯度就绪`、`Note: 通信与计算并行进行`、`NET→GPU: 第 L 层 AllReduce 完成`、… | `模块一-前置知识/communication/collective-communication-primer.md:364` | L13 / L17 | **可机械转 DAG**（时序图） | 这是一个 **sequenceDiagram**，两个泳道（GPU 计算 / 网络传输）+ 5 条带方向的消息 + 1 条 Note。泳道结构**可直接映射成 L13 的甘特图两行**（计算轨/通信轨），消息顺序就是时间轴。这也是 L17 那三张 SVG 甘特的同构描述（一个是 SVG，一个是文本）。 |
| `rank0 → rank1 → rank2 → rank3 → rank0`（4 节点环） | `模块三-分布式训练/2.1 集合通信原语详解.md:265` | L13 | **可机械转 DAG** | 与 `primer.md:300` 完全同构的环，出现在「阶段二：All-Gather」小节。两处可合并成一个拓扑定义。 |
| `前向(算 loss) → 反向(loss.backward() 得到梯度 g) → 优化器(optimizer.step() 梯度+状态→更新参数) → 清零梯度(zero_grad()) → （回到前向）` | `模块三-分布式训练/3.1 优化器原理与显存开销分析.md:34` | L14 | **可机械转 DAG** | 4 节点、**含回边**的训练 step 循环，每个节点上标了对应的 API。L14 回放主线要走的"前向→反向→梯度归约→优化器步"四个阶段，骨架已经在这里了；显存条的逐阶段变化需要 trace 补。 |
| `主卡 GPU0(模型+输入 batch) → Scatter 把输入切分到各卡 → {GPU1/GPU2/GPU0 前向}` → Gather 输出汇总回主卡 → 主卡计算 loss + 反向 → 各卡梯度汇总回主卡求和 → 主卡更新参数` | `模块三-分布式训练/4.1 数据并行详解.md:101` | L17 | **可机械转 DAG** | 10 节点、9 条边，含 2 个汇聚节点（Gather 收 3 条、GG 收 3 条）。是 DP（非 DDP）的完整数据流拓扑——注意它是"主卡汇总"模型，与 L17 要讲的 DDP AllReduce 有区别，正好是 L17 的**对照基线**。 |
| `反向:Layer N 梯度就绪 → Bucket 1 AllReduce`、`反向:Layer N-1 → Bucket 2`、`反向:Layer N-2 → Bucket 3`；`AR1 -.同时进行.-> B2`、`AR2 -.同时进行.-> B3` | `模块三-分布式训练/4.1 数据并行详解.md:175` | L17 | **可机械转 DAG** | 6 节点、5 条边，其中 2 条是 `-.同时进行.->` 的**交叉重叠边**——这是唯一一段显式编码了"通信与下一层计算并行"的 mermaid。它和 `dp_overlap2/3.svg` 是同一件事的两种表达（文本 vs 图形），可互相校验。 |
| FSDP：`初始每卡只存 1/N 参数分片 → 前向 AllGather 拼出完整参数 → 用完整参数前向 → 立即释放非本地分片 → 反向再次 AllGather → 用完整参数算梯度 → ReduceScatter 梯度聚合并分片 → 释放完整参数 → 更新各卡只更新自己负责的 1/N 参数` | `模块三-分布式训练/4.1 数据并行详解.md:280` | L14 | **可机械转 DAG** | 9 节点线性链，每个节点都标了显存状态（"每卡只存 1/N"、"释放非本地分片"）——这恰好是 L14 显存账本要逐帧展示的东西。虽然写在 DDP 章节里，但内容是 ZeRO/FSDP 的，归属 L14。 |
| 决策树：`单卡装得下完整模型状态？ →(装得下) DDP` / `→(装不下) 是多机吗？ →(单机多卡) 差一点还是差很多？ → SHARD_GRAD_OP / FULL_SHARD` / `→(多机多卡) HYBRID_SHARD` → `单个层就装不下？ → 需叠加 TP/PP` | `模块三-分布式训练/4.1 数据并行详解.md:394` | L14 | 仅作参考 | 12 节点的**选型决策树**，判断节点与分支标签完整。但它回答"用户该选哪个方案"，不是算法数据流。适合做 `/labs` 索引页的导航 DAG，或 L14 的收尾小结，不进 lab 播放器。 |

### 与 lab 无关的 27 段

这些是环境搭建、CUDA 执行模型、工具链等章节的说明图，结构上大多是可机械转的，但没有对应的 lab，**不建议本轮投入**。列出以备将来取舍。

| 素材（节点摘要） | 位置 | 归类 |
|---|---|---|
| `Volta 2017 → Turing 2018 → Ampere 2020 → Hopper 2022 → Blackwell 2024`（GPU 架构代际链） | `模块一-前置知识/gpu/nvidia-gpu-evolution.md:38` | 可机械转 DAG |
| 同上 + 每代新特性作边标签（`+Tensor Core`、`+INT8 推理`、`+TF32/MIG/BF16`、`+FP8/Transformer Engine`、`+FP4/双芯封装/NVLink 5.0`） | `模块一-前置知识/gpu/nvidia-gpu-evolution.md:375` | 可机械转 DAG |
| nvcc 编译流程：`.cu 源文件 → nvcc 前端 → {设备代码(GPU) → ptxas→cubin, 主机代码(CPU) → g++/MSVC→.o} → 链接器 → 可执行文件` | `模块二-CUDA编程与算子优化/1.1-CUDA开发环境搭建.md:219` | 可机械转 DAG |
| **sequenceDiagram**：`CPU(Host)` / `GPU(Device)` 两泳道——初始化数据 → cudaMemcpy → `kernel<<<grid,block>>>()` 启动 → 数千线程并行执行 → cudaMemcpy 回传 → 后续处理 | `模块二-CUDA编程与算子优化/1.2-CUDA编程模型.md:35` | 可机械转 DAG（时序） |
| `线程 → {寄存器, 局部内存}`、`线程块 → 共享内存`、`Grid → {全局内存, 常量内存, 纹理内存}` | `模块二-CUDA编程与算子优化/1.3-CUDA内存模型.md:46` | 可机械转 DAG |
| 内存选型决策树：`数据是否被修改? →(只读) 所有线程读同一值? → 常量内存 / 全局内存+__ldg()`、`→(读写) 是否 Block 内共享? → 共享内存 / 全局内存(合并访问)` | `模块二-CUDA编程与算子优化/1.3-CUDA内存模型.md:410` | 可机械转 DAG |
| 五步流程：`1.分配 GPU 内存 → 2.CPU→GPU 传数据 → 3.启动 Kernel → 4.GPU→CPU 取结果 → 5.释放 GPU 内存` | `模块二-CUDA编程与算子优化/1.4-第一个实用Kernel.md:33` | 可机械转 DAG |
| warp 指令流水：`取指 → 译码 → 发射 → 执行 → 写回` | `模块二-CUDA编程与算子优化/2.1-Warp与执行模型.md:81` | 可机械转 DAG |
| 优化决策树：`分析 Kernel 性能 → Memory-Bound? → 检查全局内存访问模式 → 合并访问? → 检查共享内存 → Bank Conflict? → Padding/Swizzle → 向量化加载 → 已达带宽上限? → 优化完成 / 减少冗余访问`（12 节点） | `模块二-CUDA编程与算子优化/2.2-内存访问优化.md:497` | 可机械转 DAG |
| `SM 总资源池 → {Block 0→Warp 0-7→Thread 0-255, Block 1→Warp 8-15, Block N→Warp ...}` | `模块二-CUDA编程与算子优化/2.3-Occupancy与资源分配.md:88` | 可机械转 DAG |
| Occupancy 调优流程树：`测量 Kernel 性能 → 用 Nsight Compute 获取 Occupancy → Occupancy 是否瓶颈? → 提升 Occupancy → 限制因素? → {寄存器→减少临时变量/__launch_bounds__, 共享内存→减少 Tile 大小, Block 大小→调整 Block 维度} → 验证性能 → 性能是否提升? → 保持/回退`（13 节点） | `模块二-CUDA编程与算子优化/2.3-Occupancy与资源分配.md:378` | 可机械转 DAG |
| `Grid 级同步 → {Kernel 边界隐式同步, Cooperative Groups Grid Sync}`、`Block 级同步 → {__syncthreads(), block.sync()}` | `模块二-CUDA编程与算子优化/2.4-同步与原子操作.md:41` | 可机械转 DAG |
| `Warp 级同步 → {__syncwarp(), 隐式 lockstep(Pre-Volta)}`、`线程级 → {Memory Fence, 原子操作}`（注：这段上方缺 ` ```mermaid ` 围栏，是纯代码块形态） | `模块二-CUDA编程与算子优化/2.4-同步与原子操作.md:49` | 可机械转 DAG |
| 模型到 App 链路：`训练框架(PyTorch/TF) → 模型导出(torch.export/ONNX) → 图变换(常量折叠/算子融合/量化) → 后端编译或委托(CPU/GPU/NPU/DSP) → 端侧程序(.pte/.onnx/.tflite/.mlpackage) → Runtime → Android/iOS/C++ App` | `模块四-推理优化/第12章-端侧推理/12.1-端侧推理基础.md:110` | 可机械转 DAG |
| 图分区：`完整计算图 → 图分区器 → {支持子图交给 NPU, 不支持算子 CPU 执行} → 合并输出` | `模块四-推理优化/第12章-端侧推理/12.1-端侧推理基础.md:128` | 可机械转 DAG |
| 端侧框架选型决策树（按模型来源分 PyTorch / TensorFlow / 端侧 LLM / Apple 原生，12 节点） | `模块四-推理优化/第12章-端侧推理/12.1-端侧推理基础.md:430` | 可机械转 DAG |
| Decode 逐 Kernel 启动：`CPU 发射 Kernel1 → GPU 算(极快) → CPU 发射 Kernel2 → GPU 算(极快) → GPU 大量时间在等 CPU 发指令` | `模块四-推理优化/第2章-推理引擎核心技术/2.5-Attention 后端与图优化.md:38` | 仅作参考 |
| CUDA Graph 重放：`CPU 一次 replay → GPU 连续执行全部 Kernel → CPU 开销几乎归零` | `模块四-推理优化/第2章-推理引擎核心技术/2.5-Attention 后端与图优化.md:87` | 仅作参考 |
| 混合执行：`非 Attention 层(CUDA Graph 重放) → Attention(eager 执行) → 非 Attention 层 → Attention` | `模块四-推理优化/第2章-推理引擎核心技术/2.5-Attention 后端与图优化.md:113` | 仅作参考 |
| 三子图切分：`子图1(首层, Attention 之前) → 子图2(中间重复层) →（自环）→ 子图3(末层)` | `模块四-推理优化/第2章-推理引擎核心技术/2.5-Attention 后端与图优化.md:140` | 可机械转 DAG |
| vLLM 定位：`用户请求 → vLLM 推理引擎 → {PagedAttention KV Cache 管理→高显存利用率, Continuous Batching→高吞吐量, 模型执行 CUDA Kernels→低延迟}` | `模块四-推理优化/第3章-深入vLLM/vllm快速入门.md:46` | 仅作参考 |
| 五大并行策略：`单卡训练困境 → {装不下/显存瓶颈→切分训练状态(ZeRO/FSDP·TP·PP), 跑不完/算力瓶颈→复制模型并行(DP/DDP)} → 3D 并行组合` | `模块三-分布式训练/1.1 分布式训练总论.md:140` | 可机械转 DAG |
| 并行选型决策树：`要训多大模型? → 单卡装得下? → DDP / 切分后装得下? → FSDP/ZeRO / 瓶颈在哪? → {单层太大→TP, 层数太多→PP, 序列太长→SP/CP} → 3D 并行`（11 节点） | `模块三-分布式训练/1.1 分布式训练总论.md:184` | 可机械转 DAG |
| rank 概念：`Node0(node_rank=0){进程 rank=0/local_rank=0→GPU0, 进程 rank=1/local_rank=1→GPU1}`、`Node1(node_rank=1){进程 rank=2/local_rank=0→GPU0, 进程 rank=3/local_rank=1→GPU1}` | `模块三-分布式训练/1.2 环境搭建与分布式启动.md:82` | 可机械转 DAG |
| rendezvous 会合流程：`Node0 起 c10d store → Rendezvous 集合点`、`Node1 连接 store → 集合点`；`集合点 → 人齐了? →(是) 协商编号开始训练 / (否) 阻塞等待` | `模块三-分布式训练/1.2 环境搭建与分布式启动.md:301` | 可机械转 DAG |
| 多卡排障决策树（NCCL_DEBUG=INFO → 看日志 → 4 种症状分支 → 定位后关掉调试开关，12 节点） | `模块三-分布式训练/1.2 环境搭建与分布式启动.md:456` | 可机械转 DAG |
| DDP 五段式：`init_process_group → set_device → model→DDP → 训练循环+DistributedSampler → destroy_process_group` | `模块三-分布式训练/1.2 环境搭建与分布式启动.md:482` | 可机械转 DAG |

> **统计口径提示**：上表 27 段里有 **23 段属于"可机械转 DAG"**。它们结构清晰、提取无损，但**没有对应 lab**——本轮不建议投入。真正需要警惕的是把这些"可机械转"误当成"值得做"。

---

## 三、归类汇总

| 归类 | 图片 | mermaid | 合计 | 其中与 lab 相关 |
|---|---:|---:|---:|---:|
| **可直接复用** | 4 | 0 | 4 | 4 |
| **可机械转 DAG** | 0 | 43 | 43 | 20 |
| **需重绘为交互版** | 11 | 0 | 11 | 11 |
| **仅作参考** | 26 | 12 | 38 | 17 |
| **合计** | **41** | **55** | **96** | **52** |

（"与 lab 相关"指该素材在本报告主表里挂到了某个具体 lab。图片有 9 项、mermaid 有 4 段与任何 lab 无关——多为站点装饰、硬件平面图或环境搭建说明。）

按 lab 的素材覆盖度：

| Lab | 可用素材 | 状态 |
|---|---|---|
| L00 | 1 段 mermaid（浅） | 骨架有，内容全缺 |
| L01 | 3 张需重绘 + 6 张仅参考 + 1 段可转 DAG | **素材最厚的 lab**，但全需重画 |
| L02 | 4 张需重绘 + 1 段可转 DAG + 1 段仅参考 | 素材较厚，全需重画 |
| L03 | **0** | **零素材**，全新建 |
| L04 | 2 张可直接复用 + 3 张仅参考 + 2 段可转 DAG | **覆盖最好**，可立即开工 |
| L05 | 3 段 mermaid（1 段含回边，1 段浅，1 段仅参考） | 骨架有，显存账本/曲线全缺 |
| L06 ★ | 1 张需重绘（三合一，最贵）+ 1 张可直接复用 | 蓝本极好，重绘工作量大 |
| L07 | 0（3 张 FA3 孤儿图仅作参考） | 靠 L06 引擎，无需新素材 |
| L09 ★ | 3 段 mermaid（含 1 段二分图） | **结构最完整**，可直接机械转换 |
| L10 ★ | 3 段 mermaid（含 batch 组成数据） | 结构完整 |
| L11 | 2 段 mermaid（浅） | 文案骨架 |
| L12 | 1 段 mermaid（含判断分支） | 骨架好，树本身需重绘 |
| L13 ★ | 4 段 mermaid（环 / 树 / 时序） | **拓扑齐全**，零图片 |
| L14 | 2 段 mermaid（含 FSDP 链路） | 骨架有，账本矩阵需新建 |
| L15 / L16 | **0** | **零素材**，且需先补正文 |
| L17 | 3 张需重绘（三帧！）+ 2 张仅参考 + 3 段 mermaid | **素材与 lab 参数完美对齐** |
| L18–L21 | **0** | **零素材**，且需先补正文 |

---

## 四、最值得优先复用的 Top 5

排序依据：**关联 lab 的路线图优先级**（`docs/plans/interactive-labs.md` §五）× **复用就绪度**（是否已具备可机械提取的节点/边，或可直接嵌入）÷ **重绘成本**。

### 1. `2.1-PagedAttention.md:80` 的 Block Table 二分图 —— 用于 L09 ★

**为什么是第一**：它是 96 个素材里**唯一一个既已被渲染成图、其源码又是完整二分图定义、且对应 P3 旗舰 lab**的素材。3 个逻辑块节点之间用 `---` 表示"连续"，3 个物理块节点是散落的 `#7/#2/#5`，3 条 `-.Block Table.->` 边跨侧连接——**这正是 L09 专用视图「Block Table 映射连线」的字面定义**。设计文档点名"教程里的 mermaid 图是现成雏形"，说的就是它。

**复用路径**：Mermaid 的 `graph` 语法本身就是 `nodes + edges + labels`，写一个 30 行的解析器把 `L0["逻辑块0<br/>tok 0-3"]` 拆成 `{id, label}`、把 `-.Block Table.->` 拆成 `{from, to, label, style}`，即可直接产出引擎 `graph` 字段的 JSON。配套的 markdown 表格（`0→#7 已满(4/4)` 等）直接成为 trace 的初始 state。**零重绘、零推断，纯机械转换。**

### 2. `dp_overlap1.svg` + `dp_overlap2.svg` + `dp_overlap3.svg` —— 用于 L17

**为什么**：这三张图的语义我已经渲染确证，它们**不是三个并列方案，而是同一个算法在 bucket 大小三个取值下的三帧**——上轨的 `Forward(teal) / Backward(orange) / Optimizer` 完全相同，只有下轨紫色 AllReduce 块的**粒度**在变（整条 ALL → 24 个细块 → 3 个中块）。这恰好就是 L17 参数滑杆（`bucket_cap_mb`）要扫的区间，三张图直接成为滑杆的三个锚点；颜色语义（teal=Forward、orange=Backward、purple=AllReduce Grads，右下角自带图例）也已确定。

**代价可控**：因为源 SVG 是**扁平结构的 `<path>` 集合、没有嵌套 group**（三张图的 `<path>` 数分别是 54 / 105 / 60），所以不能像 mermaid 那样"机械提取"——但反过来，**重绘成本极低**：结构化程度高（就是两轨甘特 + 一排圆角矩形），布局规则明确（上轨块宽度对应层数、下轨块边界由 bucket 累积量决定），而且 `4.1 数据并行详解.md:175` 的 mermaid 提供了同构的文本描述可交叉验证。

### 3. `flashattentionv1-1.png` —— 用于 L06 ★ 旗舰

**为什么**：全仓库**形状标注最完整的一张图**——`B_r×d`（Q）、`B_c×d`（K/V）、`B_r×B_c`（S_ij）三种块形状逐块标出，HBM 侧与 SRAM 侧分开画，三条"加载"虚线把两层连起来，旁注 `外层循环 j 固定 → 内层循环 i=1..T_r 遍历 Q` 说明了循环嵌套顺序。**直接嵌入 lab 页面即可用**；更重要的价值是它同时是交互版「HBM/SRAM 双层舞台」的现成视觉蓝本——尺寸关系、颜色编码（Q 蓝 / K 绿 / V 橙 / S 黄）、分层布局都是现成的。

**注意**：同节的 `flashattentionv1-0.png` 是更有分量的那张（金字塔 + 数据流 + PyTorch/FA 性能对比三合一），但它必须重绘，且是 L06 里最贵的一项工作。建议的顺序是：先用 `v1-1.png` 把静态页立起来，再按 `v1-0.png` 的中栏重绘双层舞台。

### 4. `collective-communication-primer.md` 的 4 段 mermaid —— 用于 L13 ★

**为什么**：L13 是 P4 的旗舰 lab，但**图片素材为零**——它全部的现成结构都在 mermaid 里，而且四段各自解决一个问题：`:300` 给出 **Ring 拓扑**（4 节点有向环）、`:336` 给出 **Tree 对照拓扑**（7 节点二叉树，对应"Ring vs Tree"的对照需求）、`:364` 的 **sequenceDiagram** 给出"GPU 计算 / 网络传输"两泳道 + 5 条带时序的消息（可直接映射成 lab 要的甘特两轨）、`2.1 集合通信原语详解.md:265` 是 Ring 的重复定义（可合并）。

**复用路径**：环与树的拓扑可机械转成 DAG 的 `edges`；sequenceDiagram 的泳道 + 消息序列可机械转成甘特数据。L13 需要的"每卡缓冲区状态表"和"通信量累计器"仍要新建，但**拓扑与时序这两个最容易被画错的骨架已经有了**。

### 5. `3.6 LayerNorm与残差连接深入理解.md:422` + `:435` 的 Pre/Post-Norm 对照 —— 用于 L04

**为什么**：这是 55 段 mermaid 里**唯一现成的对照实验素材**。两段各 5 个节点、各含 1 条残差跳跃边，节点名完全相同（`Input(x) / SubLayer / Add / LayerNorm / Output`），**只有 norm 的位置不同**。L04 的「对照：Pre-Norm vs Post-Norm 的梯度尺度变化」正好需要一个 A/B 双轨播放器，而这里的 diff 是**天然自带的**——不需要手工构造两套数据，把两段 mermaid 各转成一个 DAG 就是 diff 的两侧。

**额外红利**：L04 同时是**素材覆盖最好的 lab**——`decoder-blocks.png`（宏观 32 层结构）与 `decode-only.png`（微观 Pre-Norm block，框内直接写了 MHA 和 FFN 的逐步公式 `(Q,K,V投影 → RoPE → ... → 输出投影)` 与 `(W_gate*x → Swish, W_up*x, 逐元素相乘 → W_down)`）都是**纯示意图、可直接嵌入**。也就是说 Top 5 里只有这一项是"现在就能做出可看页面"的。

---

## 五、给引擎实现的直接结论

1. **mermaid 解析器值得写**。55 段里有 **43 段可机械转 DAG**，其中 **20 段与 lab 直接相关**，覆盖 L00/L01/L02/L04/L05/L09/L10/L11/L12/L13/L14/L17 共 12 个 lab（L11 那 2 段偏文案，能提供的是场景参数而非拓扑）。Mermaid `graph` 语法的子集（`id["label"]`、`-->`、`-.text.->`、`-->|text|`、`subgraph`、`{...}` 菱形、`---`）足以覆盖其中绝大多数；建议**离线转换**（构建期跑一次脚本产出 JSON）而不是在 lab 页面里跑 mermaid——lab 页面承诺"不发外部请求"，而当前站点的 mermaid 是走 jsDelivr CDN 的。

2. **优先级应看"结构就绪度"而不是"图好不好看"**。按本报告的分类，L09/L13/L04/L17 是结构最就绪的四个；`AIInfra优化全流程解析.png` 和 `理解AIInfra的职责边界.png` 这两张信息密度最高的图，恰恰提取不出任何结构，属于"仅作参考"。

3. **有 8 个 lab 是零素材**：L03、L07、L15、L16、L18、L19、L20、L21。其中 L15/L16/L18/L19/L20/L21 对应的教程章节本身就是 30~80 行的"本章概述"stub——**素材缺口与正文缺口是同一个缺口**，这两件事应当排在一起做。

4. **有 4 个孤儿素材可以直接处置**：`flashattentionv3-1.png`、`flashattentionv3-2.png`、`flashattentionv3-3.png`、`gemm-6.png` 在 `grep` 全仓库后零引用。前三个是 FA3 的调度甘特与寄存器布局，对 L07 有参考价值但并非 V1/V2 的增量内容；`gemm-6.png` 与 `gemm-5.png` 同族且更弱。可以考虑在实现 L07 时清理，或明确留作进阶素材。

5. **`dp_overlap1/2/3.svg` 的三帧关系是本次盘点最有价值的单点发现**。它意味着 L17 的 bucket 滑杆不是"凭空设计的参数"，而是**仓库里已经有人用三张图讲过的同一件事**——交互版只是把这三帧之间的连续过渡补上。

---

### 附：可复现命令

```bash
# 图片清单与引用关系
ls public/images/
for f in $(ls public/images/); do grep -rn --include='*.md' -F "$f" docs/ src/ README.md; done

# SVG 渲染（源码无 <text> 节点，文字是字形轮廓，必须渲染后才能读）
pip install cairosvg
python3 -c "import cairosvg; cairosvg.svg2png(url='public/images/dp_overlap1.svg', write_to='/tmp/dp1.png', scale=1.6)"

# mermaid 块提取
python3 - <<'PY'
import pathlib
for p in sorted(pathlib.Path('docs/guides').rglob('*.md')):
    lines = p.read_text(encoding='utf-8').split('\n')
    for i, l in enumerate(lines):
        if l.strip().startswith('```mermaid'):
            print(f"{p}:{i+1}")
PY

# 孤儿素材
for f in $(ls public/images/); do
  [ "$(grep -rl --include='*.md' --include='*.astro' -F "$f" . | grep -v node_modules | grep -v '^./public/' | wc -l)" = 0 ] && echo "ORPHAN: $f"
done
```
