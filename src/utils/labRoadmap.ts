/**
 * Roadmap data for the `/labs` index page: every lab in this round, its phase,
 * and the hard prerequisites between them.
 *
 * Scope is P0–P4 (17 labs). P5 (quantization / speculative decoding / PD
 * disaggregation) is out of this round because those tutorials have no body
 * text yet; adding a lab here implies its tutorial exists.
 *
 * This is a separate file from `labRegistry.ts` on purpose. The registry is
 * wiring — "when rendering this tutorial, show this card" — while this is
 * editorial content for one page. They change for different reasons: a lab
 * gains a card the day it ships, but keeps its roadmap slot from day one.
 *
 * `requires` holds only the hard prerequisites the design doc states outright
 * (docs/plans/interactive-labs.md §三: "L00 → L06 → L07 是硬依赖链", and L06's
 * "前置：必须先看 L00"). It is deliberately not a full pedagogical DAG — the
 * phase lanes already convey reading order, and inventing edges the design doc
 * never asserted would tell readers to skip labs for no stated reason.
 *
 * Tutorial links are stored as (category, slugified id) rather than a URL, and
 * composed at render time. `guideId` values follow the same slugification as
 * `labRegistry.ts` — see the KEY FORMAT note in that file.
 */

import { CATEGORY_URL_PREFIX } from './moduleMetadata';
import type { CategorySlug } from './categories';

/** Phase from the design doc's roadmap (§五). */
export type LabPhase = 'P0' | 'P1' | 'P2' | 'P3' | 'P4';

export interface LabNode {
  /** Lab id from the design doc, e.g. `L00`. */
  labId: string;
  /** Display name, matching the design doc outline. */
  title: string;
  /** One line on the lab's subject, for the index card. */
  summary: string;
  phase: LabPhase;
  /** Lab ids the design doc states as hard prerequisites. Often empty. */
  requires?: string[];
  /** Slugified guide id of the tutorial this lab replays. */
  guideId?: string;
  /** Category the tutorial lives in; decides its URL prefix. */
  guideCategory?: CategorySlug;
}

export interface LabPhaseMeta {
  id: LabPhase;
  /** Heading for the phase lane on the index page. */
  label: string;
  /** What the phase delivers. */
  blurb: string;
}

export const LAB_PHASES: LabPhaseMeta[] = [
  {
    id: 'P0',
    label: '引擎 + 样板',
    blurb: '播放器、DAG 画布、张量检查器、公式面板、时间轴——一次建好，之后所有 lab 复用。',
  },
  {
    id: 'P1',
    label: 'Transformer 计算链',
    blurb: '单卡前向：GEMM、Reduce、Attention、Decoder Block、KV Cache 与自回归生成。',
  },
  {
    id: 'P2',
    label: 'CUDA 算子回放',
    blurb: 'FlashAttention 的 HBM / SRAM 双层舞台与 IO 计数，旗舰内容。',
  },
  {
    id: 'P3',
    label: '推理引擎调度',
    blurb: 'PagedAttention、Continuous Batching、Chunked Prefill、Prefix Cache。',
  },
  {
    id: 'P4',
    label: '分布式训练',
    blurb: '集合通信、显存账本、张量并行、流水线并行、梯度分桶与通信重叠。',
  },
];

export const LAB_ROADMAP: LabNode[] = [
  // P0
  {
    labId: 'L00',
    title: 'Online Softmax',
    summary: '标准三遍扫描改造为分块递推，修正因子 e^(m_old − m_new) 逐块更新 m、ℓ、O。',
    phase: 'P0',
    guideId: '模块二-cuda编程与算子优化/52-cuda-online-softmax实现',
    guideCategory: 'cuda-optimization',
  },

  // P1
  {
    labId: 'L01',
    title: 'GEMM：索引、分块与访存',
    summary: '从单个输出点的点积铺到整个 C，再引入 tiling 与 global → shared → register 的搬运。',
    phase: 'P1',
    guideId: '模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化',
    guideCategory: 'cuda-optimization',
  },
  {
    labId: 'L02',
    title: 'Reduce：顺序 / 共享内存树 / Warp Shuffle',
    summary: '三种归约在同一份输入上并排跑，线程阵列与归约树同步动画。',
    phase: 'P1',
    guideId: '模块二-cuda编程与算子优化/31-cuda-reduce算子优化',
    guideCategory: 'cuda-optimization',
  },
  {
    labId: 'L03',
    title: 'Self-Attention 全流程',
    summary: 'X → Q,K,V → 分头 → S = QKᵀ/√d_h → causal mask → softmax → O = PV → 合并头。',
    phase: 'P1',
    guideId: '模块一-前置知识/transformer/33-self-attention机制深入理解',
    guideCategory: 'prerequisites',
  },
  {
    labId: 'L04',
    title: 'Decoder Block：残差 / LayerNorm / SwiGLU',
    summary: '完整 block 走一遍，重点看残差为何要求两侧 shape 完全相同、LayerNorm 在哪个维度上做。',
    phase: 'P1',
    guideId: '模块一-前置知识/transformer/37-transformer-decoder-block完整解析',
    guideCategory: 'prerequisites',
  },
  {
    labId: 'L05',
    title: 'KV Cache 与自回归生成',
    summary: '逐 token 生成：K_cache 如何增长，以及有/无 cache 时 attention 形状的差别。',
    phase: 'P1',
    guideId: '模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解',
    guideCategory: 'prerequisites',
  },

  // P2
  {
    labId: 'L06',
    title: 'FlashAttention V1',
    summary: 'Q 块外循环 × K/V 块内循环，S、P 从不落回 HBM，IO 次数与标准 Attention 实时对比。',
    phase: 'P2',
    requires: ['L00'],
    guideId: '模块二-cuda编程与算子优化/61-flashattention-v1详解',
    guideCategory: 'cuda-optimization',
  },
  {
    labId: 'L07',
    title: 'FlashAttention V2 增量',
    summary: '与 V1 并排做 diff：循环顺序调换、去掉对 O 的两次 rescale、non-matmul FLOPs 下降。',
    phase: 'P2',
    requires: ['L06'],
    guideId: '模块二-cuda编程与算子优化/62-flashattention-v2详解',
    guideCategory: 'cuda-optimization',
  },

  // P3
  {
    labId: 'L09',
    title: 'PagedAttention',
    summary: '逻辑块到物理块的分配、共享前缀触发 CoW、显存紧张时抢占，Block Table 连线全程可见。',
    phase: 'P3',
    guideId: '模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention',
    guideCategory: 'inference-optimization',
  },
  {
    labId: 'L10',
    title: 'Continuous Batching 调度',
    summary: '迭代级调度循环：每个 tick 取请求、各生成一步、处理完成与抢占、组成新 batch。',
    phase: 'P3',
    guideId: '模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching',
    guideCategory: 'inference-optimization',
  },
  {
    labId: 'L11',
    title: 'Chunked Prefill 与统一调度',
    summary: '长 prefill 切成 chunk 与 decode 混排，token budget 是唯一货币。',
    phase: 'P3',
    guideId: '模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度',
    guideCategory: 'inference-optimization',
  },
  {
    labId: 'L12',
    title: 'Prefix Cache 与 RadixAttention',
    summary: '块哈希链、基数树插入与分裂、引用计数、LRU 淘汰时的显存回收。',
    phase: 'P3',
    guideId: '模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention',
    guideCategory: 'inference-optimization',
  },

  // P4
  {
    labId: 'L13',
    title: 'Ring AllReduce',
    summary: '4 卡环形，ReduceScatter 与 AllGather 各 N−1 步，每卡缓冲区状态与通信量同步累计。',
    phase: 'P4',
    guideId: '模块三-分布式训练/21-集合通信原语详解',
    guideCategory: 'distributed-training',
  },
  {
    labId: 'L14',
    title: 'ZeRO-1/2/3 显存账本',
    summary: '一个训练 step 的时间线上，每张卡的参数/梯度/优化器状态/激活各占多少。',
    phase: 'P4',
    guideId: '模块三-分布式训练/第5章-zero系列',
    guideCategory: 'distributed-training',
  },
  {
    labId: 'L15',
    title: '张量并行：Column / Row Parallel',
    summary: '列并行切输出维、行并行切输入维，以及为什么只有一个方向必须 AllReduce。',
    phase: 'P4',
    guideId: '模块三-分布式训练/第6章-张量并行与序列并行',
    guideCategory: 'distributed-training',
  },
  {
    labId: 'L16',
    title: '流水线并行 1F1B',
    summary: '4 stage × 8 micro-batch 的调度逐步展开，气泡占比与激活显存峰值曲线同步变化。',
    phase: 'P4',
    guideId: '模块三-分布式训练/第7章-流水线并行',
    guideCategory: 'distributed-training',
  },
  {
    labId: 'L17',
    title: 'DDP 梯度分桶与通信重叠',
    summary: '反向逐层产生梯度、按 bucket 聚合、bucket 满即 AllReduce，与后续层反向重叠。',
    phase: 'P4',
    guideId: '模块三-分布式训练/41-数据并行详解',
    guideCategory: 'distributed-training',
  },
];

/** Labs grouped by phase, in roadmap order. */
export function labsByPhase(): { meta: LabPhaseMeta; labs: LabNode[] }[] {
  return LAB_PHASES.map((meta) => ({
    meta,
    labs: LAB_ROADMAP.filter((lab) => lab.phase === meta.id),
  }));
}

/** Lab ids referenced by `requires` that are not themselves in the roadmap. */
export function danglingRequires(): string[] {
  const known = new Set(LAB_ROADMAP.map((lab) => lab.labId));
  const missing = new Set<string>();
  for (const lab of LAB_ROADMAP) {
    for (const req of lab.requires ?? []) {
      if (!known.has(req)) missing.add(`${lab.labId} → ${req}`);
    }
  }
  return [...missing].sort();
}

/** Site-root-relative URL of a lab's tutorial, or null when it has none. */
export function tutorialPath(lab: LabNode): string | null {
  if (!lab.guideId || !lab.guideCategory) return null;
  const prefix = CATEGORY_URL_PREFIX[lab.guideCategory];
  if (!prefix) return null;
  return `${prefix}/${lab.guideId}`;
}
