/**
 * Cell-type-major SpaceTravLR coefficient matrix. The single heatmap renderer for BOTH the
 * metabolite Environment view (BetaPanel) and the standalone view (SpaceTravlrView).
 *
 * Layout (Foster's REVISION 1): organize by CELL TYPE first — each cell type appears ONCE, and the
 * factor groups (metabolite pairs / LR / L-TF / TF / all-metabolic) are stacked beneath that one
 * cell-type label. Columns are the sorted UNION of target genes across all shown groups, in the
 * SAME position for every group and every cell type; where a group/feature has no coefficient for a
 * column, the slot is left BLANK (kept, never dropped or shifted).
 *
 * Encoding (see `betaScale.ts`): color = sign (diverging `--val-*`), tint depth = log magnitude on
 * each GROUP's OWN scale (channels span different orders of magnitude, so a shared scale would send
 * whole factors to the floor — per-group scaling is correctness, not styling), and the number is
 * printed in every cell so color never carries the value alone. Two honest "empties" are kept
 * distinct: BLANK = no coefficient measured; `≈0` = measured but negligible.
 */
import { Fragment } from 'react';
import {
  betaTooltip,
  formatBeta,
  formatMagnitude,
  groupChannel,
  makeBetaScale,
  memberDisplay,
} from '@/data/betaScale';
import type { BetaBlock } from '@/data/betaScale';
import type { BetaChannel, BetaRow } from '@/data/types';
import styles from './BetaMatrix.module.css';

/** Below this many cells a cell type's mean coefficient is noisy; flag it, don't hide it. */
const MIN_CELLS = 500;

/** One factor group stacked within each cell-type block. */
export interface BetaFactorGroup {
  /** Unique per group: a channel id, or `metab-primary` / `metab-all`. */
  key: string;
  /** The source channel — supplies rowHeader / kind / memberLabels / label. */
  channel: BetaChannel;
  /** Divider label (e.g. "This metabolite", "Metabolites", "Ligand–receptor"). */
  label: string;
  /** Rows scoped to what this group should show (a metabolite's pairs, or a whole channel). */
  rows: BetaRow[];
}

interface BetaMatrixProps {
  groups: BetaFactorGroup[];
  /** Ordered cell types to render as blocks (the caller applies the cell-type filter). */
  cellTypes: string[];
  /** If set, restrict columns to these genes (standalone gene chips); else all genes present. */
  allowedGenes?: string[];
}

interface PreparedGroup extends BetaFactorGroup {
  scale: ReturnType<typeof makeBetaScale>;
  byCellType: Map<string, BetaBlock>;
}

export default function BetaMatrix({ groups, cellTypes, allowedGenes }: BetaMatrixProps) {
  const ctSet = new Set(cellTypes);

  // Columns = sorted UNION of target genes across all shown groups (rows in the shown cell types),
  // intersected with allowedGenes when the caller restricts them.
  const geneUnion = new Set<string>();
  for (const g of groups) for (const r of g.rows) if (ctSet.has(r.cellType)) geneUnion.add(r.gene);
  let genesShown = [...geneUnion].sort();
  if (allowedGenes) {
    const allow = new Set(allowedGenes);
    genesShown = genesShown.filter((g) => allow.has(g));
  }
  const genesSet = new Set(genesShown);

  // Per group: its OWN scale over the shown cells/genes, and a cellType → block lookup.
  const prepared: PreparedGroup[] = groups.map((g) => {
    const scoped = g.rows.filter((r) => ctSet.has(r.cellType) && genesSet.has(r.gene));
    return {
      ...g,
      scale: makeBetaScale(scoped.map((r) => r.mean)),
      byCellType: new Map(groupChannel(scoped, cellTypes).map((b) => [b.cellType, b])),
    };
  });

  const cellTypesToShow = cellTypes.filter((ct) => prepared.some((g) => g.byCellType.has(ct)));

  if (!groups.length) return <div className={styles.none}>No factors selected.</div>;
  if (!genesShown.length) return <div className={styles.none}>No target genes to show.</div>;
  if (!cellTypesToShow.length) return <div className={styles.none}>No coefficients.</div>;

  const gridStyle = {
    gridTemplateColumns: `minmax(200px, max-content) repeat(${genesShown.length}, 84px)`,
  };

  return (
    <div className={styles.matrix}>
      <Legend />

      {/* ONE grid for the WHOLE matrix, so the feature-identity column and every gene column align
          across every cell-type block and factor group. Cell-type headers and factor dividers are
          full-width spanning rows within it; the gene-header row repeats per block (still inside the
          one grid, so it stays aligned) to keep column context while scrolling. */}
      <div className={styles.grid} style={gridStyle}>
        {cellTypesToShow.map((ct, i) => {
          // nCells is a property of the cell type at this tier — take it from any group's block.
          const anyBlock = prepared.find((g) => g.byCellType.has(ct))!.byCellType.get(ct)!;
          const thin = anyBlock.nCells != null && anyBlock.nCells < MIN_CELLS;

          return (
            <Fragment key={ct}>
              <div className={`${styles.blockHead} ${i > 0 ? styles.blockGap : ''}`}>
                <span className={styles.blockName}>
                  {ct}
                  {thin && (
                    <span className={styles.thinTag} title="Few cells behind these means — noisier">
                      thin
                    </span>
                  )}
                </span>
                <span className={styles.blockMeta}>
                  {anyBlock.nCells == null ? '—' : anyBlock.nCells.toLocaleString()} cells
                </span>
              </div>

              {/* Gene header row: an empty corner, then the union columns, fixed position. */}
              <span className={styles.colHead} />
              {genesShown.map((g) => (
                <span key={g} className={`${styles.colHead} ${styles.geneHead}`}>
                  {g}
                </span>
              ))}

              {prepared.map((group) => {
                const block = group.byCellType.get(ct);
                if (!block) return null;
                return (
                  <FactorRows key={group.key} group={group} block={block} genesShown={genesShown} />
                );
              })}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}

function FactorRows({
  group,
  block,
  genesShown,
}: {
  group: PreparedGroup;
  block: BetaBlock;
  genesShown: string[];
}) {
  const { channel, scale } = group;
  return (
    <>
      <span className={styles.divider}>
        <b>{group.label}</b> — {channel.rowHeader} · own scale |β| ≤ {formatMagnitude(scale.max)}
      </span>

      {block.directions.map((d) => {
        const featureTitle =
          channel.kind === 'pair'
            ? `${d.a} (${channel.memberLabels[0]}) → ${d.b} (${channel.memberLabels[1]})`
            : `${memberDisplay(d.a).text} (${channel.memberLabels[0]})`;
        return (
          <FeatureRow
            key={d.id}
            a={d.a}
            b={d.b}
            byGene={d.byGene}
            genesShown={genesShown}
            channel={channel}
            scale={scale}
            featureTitle={featureTitle}
          />
        );
      })}
    </>
  );
}

function FeatureRow({
  a,
  b,
  byGene,
  genesShown,
  channel,
  scale,
  featureTitle,
}: {
  a: string;
  b: string | null;
  byGene: Record<string, BetaRow>;
  genesShown: string[];
  channel: BetaChannel;
  scale: ReturnType<typeof makeBetaScale>;
  featureTitle: string;
}) {
  // A single-member metab value may be a pipe-joined group — render it readably (" · ") with a
  // tooltip listing the grouped metabolites. Plain names / gene names pass through unchanged.
  const aDisp = memberDisplay(a);
  return (
    <>
      <span className={styles.dir} title={featureTitle}>
        <span className={styles.gene} title={aDisp.title}>
          {aDisp.text}
        </span>
        {channel.kind === 'pair' && (
          <>
            <span className={styles.arrow} aria-label="acts on">
              →
            </span>
            <span className={styles.gene}>{b}</span>
          </>
        )}
      </span>

      {genesShown.map((g) => {
        const r = byGene[g];
        if (!r) {
          // No coefficient measured for this factor/gene: keep the slot, leave it blank.
          return <span key={g} className={styles.blank} title="no coefficient" />;
        }
        const v = r.mean;
        const negligible = scale.negligible(v);
        // Magnitude and sign ride separate channels: tint depth never encodes sign, and a near-zero
        // value gets no tint at all (but stays distinct from a blank — it reads `≈0`).
        const t = negligible ? 0 : scale.norm(v) * 50;
        const hue = (v ?? 0) < 0 ? 'var(--val-neg)' : 'var(--val-pos)';
        return (
          <span
            key={g}
            className={`${styles.cell} ${negligible ? styles.negligible : ''}`}
            style={negligible ? undefined : { background: `color-mix(in oklab, ${hue} ${t}%, var(--bg-canvas))` }}
            title={betaTooltip(r, channel)}
          >
            {formatBeta(v, scale)}
          </span>
        );
      })}
    </>
  );
}

/** One shared sign-only key — each group states its own magnitude range on its divider. */
function Legend() {
  return (
    <div className={styles.legend}>
      <span className={styles.legendEnd}>lowers</span>
      <span className={styles.ramp} aria-hidden />
      <span className={styles.legendEnd}>raises</span>
      <span className={styles.legendNote}>
        color = sign of the effect on the target gene · blank = no coefficient · ≈0 = measured but
        negligible · each factor group is on its own magnitude scale
      </span>
    </div>
  );
}
