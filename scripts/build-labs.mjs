#!/usr/bin/env node
/**
 * Stages the lab source tree into `public/labs/`, which is what Astro copies
 * into `dist/` verbatim (Astro does not process `public/`).
 *
 * `labs/` is the checked-in source of truth:
 *   labs/assets/   engine JS/CSS + vendored KaTeX, shared by every lab page
 *   labs/pages/    one self-contained HTML per lab (NOT the /labs landing page,
 *                  which Astro renders from src/pages/labs/index.astro)
 *   labs/traces/   Python trace generators (never shipped; kept as provenance
 *                  for the JSON inlined into each page)
 *
 * `public/labs/` is a generated artifact and is gitignored. Keeping the two
 * apart buys three things:
 *   - the source tree stays in one place, rather than interleaved with
 *     `public/images` and the rest of the site's static blobs;
 *   - `labs/traces/` cannot be shipped by accident — only `labs/pages` and
 *     `labs/assets` are staged, and traces are excluded by construction;
 *   - a stale `public/labs/` can never linger, because the stage step wipes it
 *     first. A leftover HTML file from a deleted lab would otherwise keep being
 *     published forever.
 *
 * Run via `npm run build:labs`, which `npm run build` chains before Astro.
 */
import { cp, mkdir, readFile, readdir, rm, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const srcRoot = path.join(repoRoot, 'labs');
const destRoot = path.join(repoRoot, 'public/labs');

/** Source subdirectory -> destination subdirectory under public/labs/. */
const STAGED = [
  ['pages', '.'],
  ['assets', 'assets'],
];

await rm(destRoot, { recursive: true, force: true });
await mkdir(destRoot, { recursive: true });

let staged = 0;
for (const [from, to] of STAGED) {
  const fromPath = path.join(srcRoot, from);
  if (!existsSync(fromPath)) continue;
  const toPath = path.join(destRoot, to);
  await mkdir(toPath, { recursive: true });
  await cp(fromPath, toPath, { recursive: true });
  staged += await countFiles(fromPath);
}

console.log(`Staged ${staged} file(s) labs/ -> public/labs/`);

// The /labs landing page belongs to Astro (src/pages/labs/index.astro), not to
// this staged tree. Both would emit dist/labs/index.html, and Astro wins — so a
// static labs/pages/index.html is dead code that silently never ships and
// misleads whoever edits it next. Reject it loudly.
const shadowedIndex = path.join(destRoot, 'index.html');
if (existsSync(shadowedIndex)) {
  console.error(
    `labs/pages/index.html would be shadowed by src/pages/labs/index.astro — ` +
      `both write dist/labs/index.html and Astro wins. Edit the Astro page instead.`
  );
  process.exit(1);
}

// Pages move when staged (labs/pages/x.html -> public/labs/x.html), so a
// relative asset path written for the source layout silently 404s once served.
// Resolve every local href/src in each staged page against where it actually
// lands, and fail the build rather than ship a lab page with no CSS or engine.
for (const file of await readdir(path.join(srcRoot, 'pages')).catch(() => [])) {
  if (!file.endsWith('.html')) continue;
  const html = await readFile(path.join(destRoot, file), 'utf8');
  const missing = [];
  for (const [, url] of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    if (/^(?:[a-z]+:|\/\/|#|\?)/i.test(url)) continue; // external, anchor, query-only
    const target = path.resolve(destRoot, url.split(/[?#]/)[0]);
    if (!existsSync(target)) missing.push(url);
  }
  if (missing.length > 0) {
    console.error(
      `labs/pages/${file} references assets that do not exist after staging:\n` +
        missing.map((m) => `  ${m}`).join('\n') +
        `\nPaths are relative to the SERVED location (public/labs/<file>), not to labs/pages/.`
    );
    process.exit(1);
  }
}

async function countFiles(dir) {
  let total = 0;
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) total += await countFiles(path.join(dir, entry.name));
    else if (await stat(path.join(dir, entry.name)).then((s) => s.isFile())) total += 1;
  }
  return total;
}
