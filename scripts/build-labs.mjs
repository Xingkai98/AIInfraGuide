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
import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
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

// ---------------------------------------------------------------------------
// Inline traces into the pages that declare them.
//
// A lab page carries a `<!-- trace:NAME -->` marker where its trace belongs;
// this replaces the marker with the JSON from `labs/traces/NAME.json`. Two
// things this buys, both deliberate:
//
//   - The trace becomes part of the page rather than a fetch, so a lab is
//     genuinely "one HTML + shared assets" with no request of its own.
//   - Drift is impossible by construction. The page cannot show numbers that
//     differ from the JSON, because there is no second copy to disagree with.
//     (A lab must still be regenerated from its Python source for the numbers
//     to change, which is the property that keeps them real.)
//
// `labs/traces/` itself is still never staged: only the substituted result
// ships, and the generators and their JSON stay source-side.
const traceDir = path.join(srcRoot, 'traces');

// Anchored to its own line on purpose. An unanchored match would also hit the
// marker text wherever it appears in prose or inside a JS string literal —
// which is exactly what happened once: an error message that quoted the marker
// got the entire trace payload spliced into the middle of the string, breaking
// the page. A marker that is a real HTML comment stands alone on its line.
const TRACE_MARKER_RE = /^[ \t]*<!--\s*trace:([A-Za-z0-9_-]+)\s*-->[ \t]*$/gm;

let inlined = 0;
for (const file of await readdir(path.join(srcRoot, 'pages')).catch(() => [])) {
  if (!file.endsWith('.html')) continue;
  const pagePath = path.join(destRoot, file);
  let html = await readFile(pagePath, 'utf8');

  const names = [...html.matchAll(TRACE_MARKER_RE)].map((m) => m[1]);
  const duplicated = names.filter((n, i) => names.indexOf(n) !== i);
  if (duplicated.length > 0) {
    console.error(
      `labs/pages/${file} declares trace "${duplicated[0]}" more than once — ` +
        `each trace may be inlined only once per page.`
    );
    process.exit(1);
  }

  for (const name of names) {
    const jsonPath = path.join(traceDir, `${name}.json`);
    if (!existsSync(jsonPath)) {
      console.error(
        `labs/pages/${file} declares trace "${name}" but labs/traces/${name}.json does not exist.\n` +
          `Generate it with the matching labs/traces/*.py script.`
      );
      process.exit(1);
    }
    const json = JSON.parse(await readFile(jsonPath, 'utf8'));
    // `</` would close the <script> early if a trace ever contained it in a
    // string. Escaping the slash keeps the JSON byte-identical when parsed.
    const payload = JSON.stringify(json).replace(/<\//g, '<\\/');
    // Replace exactly this marker, identified by its own name — a bare global
    // replace would splice the same payload into every marker on the page.
    const markerRe = new RegExp(
      `^[ \\t]*<!--\\s*trace:${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*-->[ \\t]*$`,
      'm'
    );
    if (!markerRe.test(html)) {
      console.error(`labs/pages/${file}: could not locate the marker for "${name}"`);
      process.exit(1);
    }
    html = html.replace(
      markerRe,
      `<script>window.LabTraces=window.LabTraces||{};` +
        `window.LabTraces[${JSON.stringify(name)}]=${payload};</script>`
    );
    inlined += 1;
  }

  if (names.length > 0) {
    // Every declared trace must have been substituted. A surviving marker means
    // the page would ship a comment where its data should be, and would fail
    // only at runtime in the reader's browser.
    const leftover = [...html.matchAll(/<!--\s*trace:[A-Za-z0-9_-]+\s*-->/g)];
    if (leftover.length > 0) {
      console.error(
        `labs/pages/${file} still has ${leftover.length} unsubstituted trace marker(s): ` +
          leftover.map((m) => m[0]).join(', ')
      );
      process.exit(1);
    }
    await writeFile(pagePath, html);
  }
}
console.log(`Inlined ${inlined} trace(s) into staged pages`);

// A trace payload spliced into the middle of a <script> block would break the
// page at parse time, which no amount of downstream checking catches because it
// happens in the browser. Parse every inline script here instead, so a broken
// page fails the build rather than reaching a reader.
for (const file of await readdir(path.join(srcRoot, 'pages')).catch(() => [])) {
  if (!file.endsWith('.html')) continue;
  const html = await readFile(path.join(destRoot, file), 'utf8');
  const inline = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g;
  let match;
  let index = 0;
  while ((match = inline.exec(html))) {
    index += 1;
    try {
      // Parsed, not executed: `new Function` catches syntax errors without
      // needing a DOM or running any trace code.
      new Function(match[1]);
    } catch (err) {
      console.error(
        `labs/pages/${file}: inline <script> #${index} does not parse — ${err.message}\n` +
          `A trace inlined into the wrong place is the usual cause.`
      );
      process.exit(1);
    }
  }
}

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
