/**
 * SpaceTravLR coefficients for the selected entity, as a per-cell-type heatmap.
 *
 * Rows are DIRECTED transporter pairs (`environment gene → cell gene`), columns are the target
 * genes the model scores. One block per cell type; a picker narrows to a single cell type or
 * shows every breakdown stacked.
 *
 * Encoding: color = sign (diverging `--val-*`), tint depth = log magnitude on one scale shared by
 * the whole view (so cell types are comparable), and the number is printed in every cell so the
 * color never carries the value alone. Near-zero coefficients are left untinted and marked ≈0
 * rather than given a visible floor — see `betaScale.ts` for why that distinction matters.
 *
 * Both directions of a pair are separate rows on purpose: `A→B` and `B→A` are different
 * coefficients, and merging them would invent a symmetry the model does not claim.
 */
import { useMemo, useState } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import {
  BETA_DECADES,
  betaKey,
  betaTooltip,
  formatBeta,
  formatMagnitude,
  groupBeta,
  makeBetaScale,
} from '@/data/betaScale';
import type { BetaRow, Entity } from '@/data/types';
import styles from './BetaPanel.module.css';

const ALL = '__all__';

/** Below this many cells a cell type's mean coefficient is noisy; flag it, don't hide it. */
const MIN_CELLS = 500;

/** The pair keys whose coefficients belong to this entity. */
function pairKeysFor(entity: Entity): string[] {
  if (entity.kind === 'gene_pair') return [betaKey(entity.genes[0], entity.genes[1])];
  // A metabolite is served by many transporter pairs (many-to-many); show all of them.
  return [...new Set(entity.genePairs.map(([a, b]) => betaKey(a, b)))];
}

export default function BetaPanel() {
  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const betaBundle = useVizStore((s) => s.betaBundle);
  const [cellTypeFilter, setCellTypeFilter] = useState<string>(ALL);

  const entity = useMemo<Entity | undefined>(() => {
    if (!dataset || !entityId) return undefined;
    const list =
      entityKind === 'metabolite' ? dataset.entities.metabolite : dataset.entities.gene_pair;
    return list?.find((e) => e.id === entityId);
  }, [dataset, entityId, entityKind]);

  // Every cell type this entity has coefficients for, in the bundle's order.
  const availableCellTypes = useMemo(() => {
    if (!entity || !betaBundle) return [];
    const keys = pairKeysFor(entity);
    const present = new Set(keys.flatMap((k) => betaBundle.byPair[k] ?? []).map((r) => r.cellType));
    return betaBundle.cellTypes.filter((ct) => present.has(ct));
  }, [entity, betaBundle]);

  // Cell-type names differ per tier, so a filter set at Tier3 is meaningless at Tier1. Fall back
  // to "All" rather than blanking the panel (and keep the <select> showing that fallback).
  const effectiveFilter = availableCellTypes.includes(cellTypeFilter) ? cellTypeFilter : ALL;
  const shown = useMemo(
    () => (effectiveFilter === ALL ? availableCellTypes : [effectiveFilter]),
    [effectiveFilter, availableCellTypes],
  );

  const blocks = useMemo(() => {
    if (!entity || !betaBundle) return [];
    return groupBeta(betaBundle.byPair, pairKeysFor(entity), shown);
  }, [entity, betaBundle, shown]);

  // ONE scale over every value on screen, so a cell in one cell type is comparable to a cell in
  // another. Built from the shown blocks, so narrowing to one cell type rescales to it.
  const scale = useMemo(
    () =>
      makeBetaScale(
        blocks.flatMap((b) => b.directions.flatMap((d) => Object.values(d.byGene).map((r) => r.mean))),
      ),
    [blocks],
  );

  if (!dataset?.hasBeta || !betaBundle) return null;
  if (!entity) return null;

  const genes = betaBundle.targetGenes;

  if (!availableCellTypes.length) {
    return (
      <section className={styles.wrap} aria-labelledby="beta-h">
        <h3 className={styles.title} id="beta-h">
          SpaceTravLR coefficients
        </h3>
        <div className={styles.none}>
          No SpaceTravLR coefficients for this{' '}
          {entityKind === 'metabolite' ? 'metabolite’s transporter pairs' : 'gene pair'} at{' '}
          {tier?.label}.
        </div>
      </section>
    );
  }

  return (
    <section className={styles.wrap} aria-labelledby="beta-h">
      <div className={styles.head}>
        <div>
          <h3 className={styles.title} id="beta-h">
            SpaceTravLR coefficients
          </h3>
          <div className={styles.subtitle}>
            How much each transporter pair moves the target genes, per cell type
          </div>
        </div>

        <label className={styles.filter}>
          <span className={styles.filterLabel}>Cell type</span>
          <select
            className="control"
            value={effectiveFilter}
            onChange={(e) => setCellTypeFilter(e.target.value)}
          >
            <option value={ALL}>All ({availableCellTypes.length})</option>
            {availableCellTypes.map((ct) => (
              <option key={ct} value={ct}>
                {ct}
              </option>
            ))}
          </select>
        </label>
      </div>

      <p className={styles.caveat}>
        <b>Direction is real here.</b> Each row is one direction: the <b>environment</b> gene is
        expressed by the neighboring cells, the <b>cell</b> gene by the cell being modelled. A pair
        listed both ways has two independent coefficients — they are never merged. Sign is the
        claim: positive means the interaction raises that target gene, negative lowers it.
      </p>

      <Legend scale={scale} />

      {blocks.map((block) => {
        const thin = block.nCells != null && block.nCells < MIN_CELLS;
        return (
          <div key={block.cellType} className={styles.block}>
            <div className={styles.blockHead}>
              <span className={styles.blockName}>
                {block.cellType}
                {thin && (
                  <span className={styles.thinTag} title="Few cells behind these means — noisier">
                    thin
                  </span>
                )}
              </span>
              <span className={styles.blockMeta}>
                {block.nCells == null ? '—' : block.nCells.toLocaleString()} cells ·{' '}
                {block.directions.length} direction{block.directions.length === 1 ? '' : 's'}
              </span>
            </div>

            <div
              className={styles.grid}
              style={{
                gridTemplateColumns: `minmax(190px, max-content) repeat(${genes.length}, 88px)`,
              }}
            >
              <span className={`${styles.colHead} ${styles.dirHead}`}>environment → cell</span>
              {genes.map((g) => (
                <span key={g} className={`${styles.colHead} ${styles.geneHead}`}>
                  {g}
                </span>
              ))}

              {block.directions.map((d) => (
                <Row key={d.id} env={d.env} cell={d.cell} byGene={d.byGene} genes={genes} scale={scale} />
              ))}
            </div>
          </div>
        );
      })}

      <p className={styles.foot}>
        Tint depth is log magnitude over the {BETA_DECADES} orders of magnitude below the strongest
        coefficient shown (|beta| {formatMagnitude(scale.max)}); one scale across every cell type
        here, so blocks are directly comparable. Anything weaker than{' '}
        {formatMagnitude(scale.floor)} is left blank and marked <b>≈0</b> — it is not given a
        minimum bar, because a negligible coefficient should read as nothing rather than as a small
        real effect. No significance test is applied; hover any cell for its mean, std and n. Rows
        marked <b>thin</b> come from fewer than {MIN_CELLS.toLocaleString()} cells.
      </p>
    </section>
  );
}

function Row({
  env,
  cell,
  byGene,
  genes,
  scale,
}: {
  env: string;
  cell: string;
  byGene: Record<string, BetaRow>;
  genes: string[];
  scale: ReturnType<typeof makeBetaScale>;
}) {
  return (
    <>
      <span className={styles.dir} title={`${env} expressed by the environment → ${cell} expressed by the cell`}>
        <span className={styles.envGene}>{env}</span>
        <span className={styles.arrow} aria-label="acts on">
          →
        </span>
        <span className={styles.cellGene}>{cell}</span>
      </span>

      {genes.map((g) => {
        const r = byGene[g];
        if (!r) {
          return (
            <span key={g} className={`${styles.cell} ${styles.missing}`} title={`No coefficient for ${g}`}>
              —
            </span>
          );
        }
        const v = r.mean;
        const negligible = scale.negligible(v);
        // Magnitude and sign ride separate channels: the tint depth never encodes the sign, and
        // the sign never changes the depth. A near-zero value gets no tint at all.
        const t = negligible ? 0 : scale.norm(v) * 50;
        const hue = (v ?? 0) < 0 ? 'var(--val-neg)' : 'var(--val-pos)';
        return (
          <span
            key={g}
            className={`${styles.cell} ${negligible ? styles.negligible : ''}`}
            style={negligible ? undefined : { background: `color-mix(in oklab, ${hue} ${t}%, var(--bg-canvas))` }}
            title={betaTooltip(r)}
          >
            {formatBeta(v, scale)}
          </span>
        );
      })}
    </>
  );
}

/** Diverging scale key. Mandatory whenever the encoding is on screen. */
function Legend({ scale }: { scale: ReturnType<typeof makeBetaScale> }) {
  return (
    <div className={styles.legend}>
      <span className={styles.legendEnd}>−{formatMagnitude(scale.max)}</span>
      <span className={styles.ramp} aria-hidden />
      <span className={styles.legendEnd}>+{formatMagnitude(scale.max)}</span>
      <span className={styles.legendNote}>
        lowers ← target gene → raises · blank = ≈0 (|beta| below {formatMagnitude(scale.floor)})
      </span>
    </div>
  );
}
