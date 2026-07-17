#!/usr/bin/env node
/**
 * Ingest adapter: harreman `easy_download` outputs -> app data contract JSON.
 *
 * The app never parses raw CSV. This script is the ONE place that knows the
 * on-disk harreman schema (see viz/docs/05_data_contract.md and the parent
 * project's DataForClaude/documentation/05_harreman_reference.md, produced by
 * metab_processing/). When new source formats appear (e.g. SpaceTravLR betadata),
 * add a sibling adapter that emits the SAME contract — do not change the app.
 *
 * Usage:
 *   node scripts/ingest.mjs <path> [--out public/data] [--id <id>] [--name <name>]
 *
 * <path> may be any of (auto-detected):
 *   - a harreman_outputs/ directory
 *   - an easy_download/ directory containing harreman_outputs/
 *   - a ROOT directory of datasets: <root>/<datasetName>/easy_download/harreman_outputs/
 *     (the eventual deploy layout — every <datasetName> becomes one dataset)
 *
 * Output tree:
 *   <out>/manifest.json
 *   <out>/<id>/dataset.json
 *   <out>/<id>/edges/<Tier>.<entityKind>.json
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, basename, resolve } from 'node:path';
import Papa from 'papaparse';

const SCHEMA_VERSION = 1;

// ---------- tiny helpers ----------
const isDir = (p) => existsSync(p) && statSync(p).isDirectory();
const num = (v) => {
  if (v === '' || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
};
const bool = (v) => String(v).trim().toLowerCase() === 'true';
const readCsv = (p) => {
  const text = readFileSync(p, 'utf8');
  const { data, errors } = Papa.parse(text, { header: true, skipEmptyLines: 'greedy' });
  if (errors.length) {
    // Papa reports per-row issues; surface the first so ingest stays honest.
    console.warn(
      `  ! CSV parse warnings in ${basename(p)}: ${errors[0].message} (row ${errors[0].row})`,
    );
  }
  return data;
};
const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));

// harreman writes bracketed filenames; keep them in one place.
const F = {
  network: 'harreman_network.json',
  globalM: '[ccc_results][cell_com_df_m].csv',
  globalGp: '[ccc_results][cell_com_df_gp_sig].csv',
  tierM: '[ct_ccc_results][cell_com_df_m].csv',
  tierGp: '[ct_ccc_results][cell_com_df_gp_sig].csv',
  metabSummary: join('summary', 'metabolite_summary.csv'),
  gpSummary: join('summary', 'gene_pair_summary.csv'),
};

const gpId = (g1, g2) => `${g1}__${g2}`;

// ---------- locate the harreman_outputs dir(s) ----------
function findHarremanRoot(dir) {
  // dir itself is a harreman_outputs dir?
  if (existsSync(join(dir, F.network))) return dir;
  // dir is an easy_download dir?
  if (isDir(join(dir, 'harreman_outputs'))) return join(dir, 'harreman_outputs');
  return null;
}

function discoverDatasets(inputPath) {
  const p = resolve(inputPath);
  if (!isDir(p)) throw new Error(`Not a directory: ${p}`);

  const direct = findHarremanRoot(p);
  if (direct) return [{ id: basename(resolve(p, '..')) || basename(p), root: direct }];

  // Treat as a root-of-datasets: <root>/<name>/easy_download/harreman_outputs
  const found = [];
  for (const name of readdirSync(p)) {
    const child = join(p, name);
    if (!isDir(child)) continue;
    const root = findHarremanRoot(child) || findHarremanRoot(join(child, 'easy_download'));
    if (root) found.push({ id: name, root });
  }
  if (found.length) return found;
  throw new Error(`No harreman outputs found under ${p} (looked for ${F.network}).`);
}

// ---------- tier discovery ----------
function discoverTiers(root) {
  const tierDirs = readdirSync(root)
    .filter((n) => /^Tier\w+$/i.test(n) && isDir(join(root, n)))
    .sort(); // Tier1 < Tier2 < Tier3 lexicographically for single digits
  return tierDirs;
}

// ---------- build one dataset ----------
function buildDataset(id, root, nameOverride) {
  console.log(`\n• dataset "${id}"  <- ${root}`);
  const network = readJson(join(root, F.network));
  const tierIds = discoverTiers(root);
  console.log(`  tiers: ${tierIds.join(', ') || '(none)'}`);

  // ---- metabolite entities from network + summary + global table ----
  const gpPerMetab = network.gp_per_metabolite || {};
  const metabSummary = existsSync(join(root, F.metabSummary))
    ? indexBy(readCsv(join(root, F.metabSummary)), 'metabolite')
    : {};
  const globalM = existsSync(join(root, F.globalM))
    ? indexBy(readCsv(join(root, F.globalM)), 'Metabolite')
    : {};

  const metaboliteEntities = Object.keys(gpPerMetab).map((name) => {
    const pairs = (gpPerMetab[name].gene_pair || []).map(([a, b]) => [a, b]);
    const sum = metabSummary[name] || {};
    const g = globalM[name] || {};
    const perTier = {};
    for (const t of tierIds) {
      perTier[t] = {
        nSigPairs: num(sum[`${t}_n_sig_pairs`]),
        tcellInvolved: sum[`${t}_tcell_involved`] != null ? bool(sum[`${t}_tcell_involved`]) : null,
        withinTcell: emptyToNull(sum[`${t}_within_Tcell`]),
        tcellInterfaces: emptyToNull(sum[`${t}_Tcell_interfaces`]),
        interactions: emptyToNull(sum[`${t}_interactions`]),
      };
    }
    return {
      id: name,
      name,
      kind: 'metabolite',
      nGenePairs: pairs.length || num(sum.n_gene_pairs) || 0,
      genePairs: pairs,
      globalSignificant:
        sum.global_significant != null ? bool(sum.global_significant) : bool(g.selected),
      globalFDR: num(sum.global_FDR_np) ?? num(g.FDR_np),
      nSigGenePairsGlobal: num(sum.n_sig_gene_pairs_global),
      perTier,
    };
  });

  // ---- gene-pair entities from network gp list + gp<->metabolite (many-to-many) ----
  const pairToMetabs = {};
  for (const [metab, obj] of Object.entries(gpPerMetab)) {
    for (const [a, b] of obj.gene_pair || []) {
      (pairToMetabs[gpId(a, b)] ||= new Set()).add(metab);
    }
  }
  const genePairEntities = (network.gp || []).map(([a, b]) => ({
    id: gpId(a, b),
    genes: [a, b],
    kind: 'gene_pair',
    metabolites: [...(pairToMetabs[gpId(a, b)] || [])].sort(),
  }));

  // ---- edges per (tier, entityKind) ----
  const tiers = [];
  const edgeBundles = []; // {filename, bundle}
  tierIds.forEach((t, i) => {
    const tdir = join(root, t);
    const mRows = existsSync(join(tdir, F.tierM)) ? readCsv(join(tdir, F.tierM)) : [];
    const gpRows = existsSync(join(tdir, F.tierGp)) ? readCsv(join(tdir, F.tierGp)) : [];

    const cellTypes = uniq([
      ...mRows.flatMap((r) => [r['Cell Type 1'], r['Cell Type 2']]),
      ...gpRows.flatMap((r) => [r['Cell Type 1'], r['Cell Type 2']]),
    ]).filter(Boolean);

    tiers.push({
      id: t,
      label: t,
      cellTypes,
      parentTier: i > 0 ? tierIds[i - 1] : null, // coarse->fine ordering assumption
      cellTypeParents: null, // no annotation crosswalk yet (see docs TODO)
    });

    edgeBundles.push({
      file: `${t}.metabolite.json`,
      bundle: bundleEdges(t, 'metabolite', cellTypes, mRows, (r) => r.metabolite),
    });
    edgeBundles.push({
      file: `${t}.gene_pair.json`,
      bundle: bundleEdges(t, 'gene_pair', cellTypes, gpRows, (r) => gpId(r['Gene 1'], r['Gene 2'])),
    });
  });

  const dataset = {
    id,
    name: nameOverride || `Harreman — ${id}`,
    source: 'harreman',
    entityKinds: ['metabolite', 'gene_pair'],
    tiers,
    entities: {
      metabolite: metaboliteEntities,
      gene_pair: genePairEntities,
    },
  };

  return { dataset, edgeBundles };
}

function bundleEdges(tier, entityKind, cellTypes, rows, keyFn) {
  const byEntity = {};
  for (const r of rows) {
    const key = keyFn(r);
    if (!key || r['Cell Type 1'] == null || r['Cell Type 2'] == null) continue;
    (byEntity[key] ||= []).push({
      source: r['Cell Type 1'],
      target: r['Cell Type 2'],
      scores: {
        C_p: num(r.C_p),
        Z: num(r.Z),
        Z_FDR: num(r.Z_FDR),
        C_np: num(r.C_np),
        FDR_np: num(r.FDR_np),
        selected: bool(r.selected),
      },
    });
  }
  return { tier, entityKind, cellTypes, byEntity };
}

// ---------- misc ----------
function indexBy(rows, key) {
  const out = {};
  for (const r of rows) if (r[key] != null) out[r[key]] = r;
  return out;
}
const uniq = (a) => [...new Set(a)];
const emptyToNull = (v) => (v === '' || v == null ? null : v);

// ---------- write ----------
function writeDataset(outDir, { dataset, edgeBundles }) {
  const dsDir = join(outDir, dataset.id);
  mkdirSync(join(dsDir, 'edges'), { recursive: true });
  writeFileSync(join(dsDir, 'dataset.json'), JSON.stringify(dataset));
  for (const { file, bundle } of edgeBundles) {
    writeFileSync(join(dsDir, 'edges', file), JSON.stringify(bundle));
  }
  const nMetab = dataset.entities.metabolite?.length ?? 0;
  const nGp = dataset.entities.gene_pair?.length ?? 0;
  console.log(
    `  wrote dataset.json (${nMetab} metabolites, ${nGp} gene pairs) + ${edgeBundles.length} edge files`,
  );
}

// ---------- main ----------
function main() {
  const args = process.argv.slice(2);
  if (!args.length || args[0].startsWith('-')) {
    console.error(
      'Usage: node scripts/ingest.mjs <path> [--out public/data] [--id <id>] [--name <name>]',
    );
    process.exit(1);
  }
  const inputPath = args[0];
  const getFlag = (f) => {
    const i = args.indexOf(f);
    return i >= 0 ? args[i + 1] : undefined;
  };
  const outDir = resolve(getFlag('--out') || 'public/data');
  const idOverride = getFlag('--id');
  const nameOverride = getFlag('--name');

  const datasets = discoverDatasets(inputPath);
  if (idOverride && datasets.length === 1) datasets[0].id = idOverride;

  mkdirSync(outDir, { recursive: true });
  const refs = [];
  for (const { id, root } of datasets) {
    const built = buildDataset(id, root, datasets.length === 1 ? nameOverride : undefined);
    writeDataset(outDir, built);
    refs.push({
      id: built.dataset.id,
      name: built.dataset.name,
      source: built.dataset.source,
      entityKinds: built.dataset.entityKinds,
      tierIds: built.dataset.tiers.map((t) => t.id),
    });
  }

  const manifest = { generatedAt: null, schemaVersion: SCHEMA_VERSION, datasets: refs };
  writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2));
  console.log(`\n✓ manifest.json written with ${refs.length} dataset(s) -> ${outDir}`);
}

main();
