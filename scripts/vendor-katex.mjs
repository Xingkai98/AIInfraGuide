#!/usr/bin/env node
/**
 * Copies KaTeX out of node_modules into the lab's served asset tree.
 *
 * Run after `npm i katex` (or a version bump): `node scripts/vendor-katex.mjs`
 *
 * The vendor directory is checked into git on purpose — lab pages promise to
 * work offline with zero external requests, so KaTeX cannot come from a CDN.
 * Re-run this instead of hand-copying files, so the rules below stay enforced.
 *
 * Rules (decisions D06 / research R02):
 *   - Keep KaTeX's `dist/` layout verbatim. `katex.min.css` references fonts as
 *     `url(fonts/…)`, so the stylesheet and `fonts/` must stay siblings.
 *   - Ship every one of the 20 faces. Font subsetting was measured as viable
 *     (3–10 faces would cover the tutorials) but rejected: a missing glyph is a
 *     silent rendering failure, and the saving is not worth that risk.
 *   - Ship woff2 only. It is first in every `src:` list, so every target
 *     browser takes it and never requests the woff/ttf fallbacks — carrying
 *     those would triple the directory for zero requests.
 *   - Drop the non-minified bundles, the `font-display: swap` variant (only
 *     useful for pre-rendered static HTML, which is not how labs render), and
 *     `contrib/`. Labs call the KaTeX API directly.
 */
import { cp, mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const src = path.join(repoRoot, 'node_modules/katex');
const dest = path.join(repoRoot, 'labs/assets/vendor/katex');

const EXPECTED_FACES = 20;

if (!existsSync(path.join(src, 'dist'))) {
  console.error(`KaTeX not found at ${src}. Run \`npm i katex\` first.`);
  process.exit(1);
}

const { version } = JSON.parse(
  await import('node:fs/promises').then((fs) => fs.readFile(path.join(src, 'package.json'), 'utf8'))
);

await rm(dest, { recursive: true, force: true });
await mkdir(dest, { recursive: true });

for (const file of ['katex.min.css', 'katex.min.js', 'LICENSE']) {
  await cp(path.join(src, 'dist', file), path.join(dest, file)).catch(() =>
    cp(path.join(src, file), path.join(dest, file))
  );
}

const fontDir = path.join(dest, 'fonts');
await mkdir(fontDir, { recursive: true });
const allFonts = await readdir(path.join(src, 'dist/fonts'));
const woff2 = allFonts.filter((f) => f.endsWith('.woff2')).sort();
for (const font of woff2) {
  await cp(path.join(src, 'dist/fonts', font), path.join(fontDir, font));
}

if (woff2.length !== EXPECTED_FACES) {
  console.error(
    `Expected ${EXPECTED_FACES} woff2 faces, found ${woff2.length}. ` +
      `KaTeX ${version} changed its font set — revisit the vendoring rules before committing.`
  );
  process.exit(1);
}

await writeFile(
  path.join(dest, 'VERSION'),
  `${version}\n`,
  'utf8'
);

console.log(`Vendored KaTeX ${version} -> ${path.relative(repoRoot, dest)}`);
console.log(`  katex.min.css, katex.min.js, LICENSE, VERSION, fonts/ (${woff2.length} woff2)`);
