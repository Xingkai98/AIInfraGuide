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
      { label: '跳到修正因子被触发的一步', search: '?step=3' },
    ],
    published: false,
  },
};

/**
 * Look up the lab for a guide. Returns null when the tutorial has no lab,
 * which is the common case — callers must handle it.
 */
export function getLabForGuide(guideId: string): LabEntry | null {
  return LAB_REGISTRY[guideId] ?? null;
}
