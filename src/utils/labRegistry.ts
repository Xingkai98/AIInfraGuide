/**
 * Map from a tutorial (guide) to the interactive lab that replays it.
 *
 * This file is the only place in the site that knows labs exist. Tutorial
 * markdown frontmatter is deliberately left alone: the fork tracks an upstream
 * repo, and adding a lab field to ~21 guides would put those files in permanent
 * conflict on every merge. Keeping the mapping here means an upstream merge
 * never touches it.
 *
 * Consequently the entry card rendered from this map is the *only* way readers
 * discover a lab — lab pages carry `data-pagefind-ignore`, so they never
 * surface in site search. The card is therefore meant to be prominent, not a
 * subtle aside.
 *
 * The map is keyed by guide, not by lab, because that is the direction the
 * renderer needs. Several guides may point at one lab page (a topic split
 * across two tutorials), so `labId` is not a unique key.
 *
 * Adding a lab = adding one entry here plus its page under `labs/pages/`.
 *
 * KEY FORMAT — read before editing
 * The keys are Astro content-collection guide ids, which the glob loader
 * **slugifies**: lowercased, `.` dropped, spaces turned into `-`. So the file
 * `5.2-CUDA Online Softmax实现.md` becomes `52-cuda-online-softmax实现`, and
 * `3.3 Self-Attention机制深入理解.md` becomes `33-self-attention机制深入理解`.
 * Guessing the key from the filename silently yields an entry that never
 * renders — there is no error, the card just never appears.
 *
 * To get a real key, ask the content layer rather than slugifying by hand:
 *   `npm run build` then read `dist/idprobe/index.html`,
 * or check the sidebar link for the tutorial in any built page.
 * `src/pages/labs/index.astro` asserts every key here resolves to a real guide,
 * so a typo fails the build instead of quietly dropping a card.
 */

export interface LabLink {
  /** Label for the deep link, shown on the entry card. */
  label: string;
  /**
   * Query string appended to the lab URL, e.g. `?step=3`. The player reads
   * `?step=N` to open at a given step, so a tutorial can point at a specific
   * teaching moment instead of just the beginning.
   */
  search?: string;
}

export interface LabEntry {
  /** Lab id from the design doc, e.g. `L00`. */
  labId: string;
  /** Display name, matching the design doc outline. */
  title: string;
  /** One line on what the reader gets to watch happen. */
  summary: string;
  /**
   * Path to the lab page relative to the site root — no `base` prefix, no
   * trailing slash. Callers run it through `getPath()` so `astro.config.mjs`'s
   * `base` is applied.
   */
  page: string;
  /** Lab ids that are prerequisites, rendered as a "先看 L00" hint. */
  requires?: string[];
  /** Deep links offered on the card. Falls back to a plain "打开实验室" link. */
  links?: LabLink[];
  /**
   * False while the lab is still a placeholder shell. Renders an "即将上线"
   * badge so the card never promises a replay that does not exist yet.
   */
  published?: boolean;
}

/** Base path for lab pages, relative to the site root. */
export const LAB_BASE = '/labs';

/**
 * The registry. Keys are slugified guide ids — see the header.
 */
export const LAB_REGISTRY: Record<string, LabEntry> = {
  '模块二-cuda编程与算子优化/52-cuda-online-softmax实现': {
    labId: 'L00',
    title: 'Online Softmax 逐步回放',
    summary:
      '把三遍扫描的 Safe Softmax 改造成分块递推，单步看 m、ℓ、O 如何用修正因子 e^(m_old − m_new) 逐块更新。',
    page: `${LAB_BASE}/00-online-softmax.html`,
    links: [
      { label: '从第一步开始' },
      // Step 7 is the second block's ℓ update, where the correction factor is
      // 0.5655 — the first block's factor is 0 by construction (no history to
      // rescale), so pointing at step 3 would show a reader a zero.
      { label: '跳到修正因子被触发的一步', search: '?step=7' },
    ],
    published: true,
  },
  '模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化': {
    labId: 'L01',
    title: 'GEMM 分块与数据搬运',
    summary:
      '从单个输出点的点积铺到整个 C，再引入 tiling：A、B 的 tile 沿 K 迭代，在 HBM → SMEM → REG 三层之间搬运。',
    page: `${LAB_BASE}/01-gemm-tiling.html`,
    links: [
      { label: '从第一步开始' },
      // Step 11 is the first HBM → SMEM load. Steps 0–9 are the single dot
      // product and the naive cost tally, where the stage's SMEM row is empty
      // by design — a reader pointed there first would see a missing layer
      // before seeing why it is missing.
      { label: '跳到第一次 tile 搬运', search: '?step=11' },
      // Step 12 is the first mma: it is the only frame where the whole chain is
      // on screen at once — the tiles just landed in SMEM, and the two rails
      // below show 8 elements crossing SMEM → REG and 16 FMA staying in REG.
      { label: '跳到 SMEM → REG 的外积累加', search: '?step=12' },
    ],
    published: true,
  },
  '模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching': {
    labId: 'L10',
    title: 'Continuous Batching 调度回放',
    summary:
      '每个 tick 一次完整调度：running 队列各生成一步、剩余预算拉新请求、完成即退随退随补；旁边是同一个算法的 Static Batching，两图共用一根时间轴。',
    page: `${LAB_BASE}/02-continuous-batching.html`,
    links: [
      { label: '从第一步开始' },
      // Tick 8 is the first tick where static holds a seat for nothing: R1, R3
      // and R4 have all generated their last token, and their batch still has
      // eleven ticks to run because R2 needs twenty. Steps 0–7 are the batch
      // filling up, where the two arms have not diverged yet and a reader would
      // see two identical charts.
      { label: '跳到槽位第一次开始空转', search: '?scenario=base&step=8' },
      // The burst scenario's continuous arm is the only place preemption fires
      // at all, and tick 12 is its first victim — the moment "显存耗尽时抢占"
      // stops being a sentence in the tutorial and becomes a red block.
      { label: '跳到 KV 不够、第一次抢占', search: '?scenario=burst&step=12' },
    ],
    published: true,
  },
  '模块三-分布式训练/21-集合通信原语详解': {
    labId: 'L13',
    title: 'Ring AllReduce 逐帧回放',
    summary:
      'N 卡环形：ReduceScatter 与 AllGather 各 N−1 步，每步看哪几条链路在传、每张卡的缓冲区里哪一块攒到了几份；通信量累计曲线与朴素中心化、Tree 两条对照共用一根轴。卡数可在 2 / 4 / 8 之间切换。',
    page: `${LAB_BASE}/03-ring-allreduce.html`,
    links: [
      { label: '从第一步开始' },
      // Step 3 is the LAST ReduceScatter step, and it is the only frame that
      // shows what the first phase bought: every rank holds exactly one
      // complete chunk, and a different one each. Every earlier step is a
      // half-built picture whose point is not legible yet.
      { label: '跳到 ReduceScatter 收尾（每卡各攒出一块）', search: '?n=4&step=3' },
      // 8 ranks, AllGather done: the widest ring on offer, and the frame where
      // the 8× gap between the hub's link and the ring's is at full height in
      // the accumulator.
      { label: '跳到 8 卡全部收齐', search: '?n=8&step=14' },
    ],
    published: true,
  },
};

/**
 * Look up the lab for a guide. Returns null when the tutorial has no lab,
 * which is the common case — callers must handle it.
 */
export function getLabForGuide(guideId: string): LabEntry | null {
  return LAB_REGISTRY[guideId] ?? null;
}
