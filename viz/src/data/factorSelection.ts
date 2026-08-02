/**
 * Selection model for the SpaceTravLR factor picker (rows, not columns): which individual factors
 * (matrix rows) are shown. The selection is ONE flat set of factor keys — a whole-group button is
 * just a bulk add/remove of that group's factors into this same set, so every factor stays
 * individually removable afterward.
 *
 * Kept in a plain module (not the `FactorPicker` component file) so both views and the picker share
 * these helpers without tripping react-refresh's "components only" rule.
 */
import type { BetaChannel, BetaChannelId, BetaRow } from './types';

/** Identity of one individual factor: `${channelId}::${a}::${b ?? ''}`. */
export type FeatureKey = string;

export const featureKey = (channelId: string, a: string, b: string | null): FeatureKey =>
  `${channelId}::${a}::${b ?? ''}`;

export interface FeatureOption {
  key: FeatureKey;
  channelId: BetaChannelId;
  a: string;
  b: string | null;
  /** Short feature label: `a → b` (pair) or `a` (single). */
  label: string;
}

/** Unique individual factors (matrix rows) of a channel, deduped by (a, b). */
export function channelFeatures(channel: BetaChannel): FeatureOption[] {
  const seen = new Set<string>();
  const out: FeatureOption[] = [];
  for (const r of channel.rows) {
    const key = featureKey(channel.id, r.a, r.b);
    if (seen.has(key)) continue;
    seen.add(key);
    const label = channel.kind === 'pair' ? `${r.a} → ${r.b}` : r.a;
    out.push({ key, channelId: channel.id, a: r.a, b: r.b, label });
  }
  return out;
}

/** The rows a channel contributes: those whose feature key is in the flat selection. */
export function selectedRows(channel: BetaChannel, selected: Set<FeatureKey>): BetaRow[] {
  return channel.rows.filter((r) => selected.has(featureKey(channel.id, r.a, r.b)));
}
