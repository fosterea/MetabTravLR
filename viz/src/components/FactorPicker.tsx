/**
 * Hybrid factor picker for the SpaceTravLR views. Selects arbitrary SUBSECTIONS of the coefficient
 * data as ONE flat set of individual factors (matrix rows: a transporter / ligand–receptor /
 * ligand–TF pair, or a single TF).
 *
 * Three affordances, all feeding the one selection set:
 *   1. Whole-group buttons — one per channel, showing none/some/all state. Click bulk-ADDs the
 *      group's factors (or removes them all when all are already selected). "Click the group to
 *      activate it, then ✕ out the ones you don't care about."
 *   2. A searchable "Add a specific factor…" combobox — matches EITHER member name; adds one factor.
 *      Results are ranked (exact → startsWith → substring) so a fully-typed name is never hidden
 *      behind the cap.
 *   3. Selected factors GROUPED INTO COLLAPSIBLE SECTIONS by channel — a big group reads as a tidy
 *      "Ligand–receptor · 104 selected" until expanded, then one removable chip per factor to prune.
 *
 * The selection set is owned by the parent; this component is controlled. `selectedRows(channel,
 * selected)` resolves a channel to the rows it contributes.
 */
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import type { BetaChannel } from '@/data/types';
import { channelFeatures, type FeatureKey, type FeatureOption } from '@/data/factorSelection';
import styles from './FactorPicker.module.css';

const MAX_OPTIONS = 40;
/** Sections with at most this many selected factors default to expanded. */
const AUTO_EXPAND_MAX = 8;

interface FactorPickerProps {
  channels: BetaChannel[];
  selected: Set<FeatureKey>;
  onChange: (next: Set<FeatureKey>) => void;
}

export default function FactorPicker({ channels, selected, onChange }: FactorPickerProps) {
  const [query, setQuery] = useState('');
  const [focused, setFocused] = useState(false);
  const [hi, setHi] = useState(0);
  const [overrides, setOverrides] = useState<Record<string, boolean>>({});
  const boxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const baseId = useId();
  const listboxId = `${baseId}-listbox`;
  const optionId = (i: number) => `${baseId}-opt-${i}`;

  // Expand/collapse overrides are per-session; once the parent clears the selection (entity/tier/
  // dataset reset) they'd otherwise linger, so re-adding a big group would show 100+ chips. Drop
  // them when nothing is selected, so defaults (collapsed for big groups) apply on the next add.
  useEffect(() => {
    if (selected.size === 0) setOverrides({});
  }, [selected]);

  // Per-channel unique features, and a flat list (with display labels) for the search.
  const byChannel = useMemo(
    () => channels.map((ch) => ({ channel: ch, features: channelFeatures(ch) })),
    [channels],
  );
  const allOptions = useMemo(
    () =>
      byChannel.flatMap(({ channel, features }) =>
        features.map((f) => ({ ...f, searchLabel: `${channel.label} · ${f.label}` })),
      ),
    [byChannel],
  );

  const q = query.trim().toLowerCase();
  const matches = useMemo(() => {
    if (!q) return [] as typeof allOptions;
    // Rank BEFORE capping so a fully-typed member name (exact/startsWith) is never truncated away.
    const scored: { f: (typeof allOptions)[number]; rank: number }[] = [];
    for (const f of allOptions) {
      if (selected.has(f.key)) continue; // already selected
      const a = f.a.toLowerCase();
      const b = (f.b ?? '').toLowerCase();
      let rank: number;
      if (a === q || b === q) rank = 0;
      else if (a.startsWith(q) || b.startsWith(q)) rank = 1;
      else if (a.includes(q) || b.includes(q) || `${a} → ${b}`.includes(q)) rank = 2;
      else continue;
      scored.push({ f, rank });
    }
    scored.sort((x, y) => x.rank - y.rank); // stable — preserves bundle order within a rank
    return scored.map((s) => s.f);
  }, [q, allOptions, selected]);

  const truncated = matches.length > MAX_OPTIONS;
  const shown = truncated ? matches.slice(0, MAX_OPTIONS) : matches;
  const open = focused && q.length > 0;

  // Keep the keyboard highlight in range as the option list changes.
  useEffect(() => {
    setHi((h) => (h >= shown.length ? 0 : h));
  }, [shown.length]);

  // Close the popover on an outside click.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setFocused(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const toggleGroup = (features: FeatureOption[]) => {
    const next = new Set(selected);
    const allSel = features.every((f) => selected.has(f.key));
    if (allSel) features.forEach((f) => next.delete(f.key));
    else features.forEach((f) => next.add(f.key));
    onChange(next);
  };
  const pin = (key: FeatureKey) => {
    const next = new Set(selected);
    next.add(key);
    onChange(next);
    setQuery('');
    setHi(0);
    inputRef.current?.focus();
  };
  const unpin = (key: FeatureKey) => {
    const next = new Set(selected);
    next.delete(key);
    onChange(next);
  };
  const clearChannel = (features: FeatureOption[]) => {
    const next = new Set(selected);
    features.forEach((f) => next.delete(f.key));
    onChange(next);
  };
  const toggleExpand = (id: string, def: boolean) =>
    setOverrides((o) => ({ ...o, [id]: !(o[id] ?? def) }));

  const onKeyDown = (e: ReactKeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHi((h) => Math.min(h + 1, shown.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHi((h) => Math.max(h - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const f = shown[hi];
      if (f) pin(f.key);
    } else if (e.key === 'Escape') {
      // Close by clearing the query but KEEP focus, so typing again re-opens the popover.
      setQuery('');
      setHi(0);
    }
  };

  return (
    <div className={styles.picker}>
      <div className={styles.groups} role="group" aria-label="Whole factor groups">
        <span className={styles.caption}>Factor groups</span>
        {byChannel.map(({ channel, features }) => {
          const total = features.length;
          const sel = features.reduce((n, f) => n + (selected.has(f.key) ? 1 : 0), 0);
          const state = sel === 0 ? 'none' : sel === total ? 'all' : 'some';
          return (
            <button
              key={channel.id}
              type="button"
              className={`${styles.group} ${state === 'all' ? styles.groupOn : ''} ${state === 'some' ? styles.groupPartial : ''}`}
              aria-pressed={state === 'all'}
              onClick={() => toggleGroup(features)}
              title={state === 'all' ? `Remove all ${channel.label}` : `Select all ${channel.label}`}
            >
              {channel.label}
              <span className={styles.count}>
                {sel}/{total}
              </span>
            </button>
          );
        })}
      </div>

      <div className={styles.comboWrap} ref={boxRef}>
        <input
          ref={inputRef}
          type="text"
          className={`control ${styles.search}`}
          placeholder="Add a specific factor…"
          role="combobox"
          aria-expanded={open}
          aria-controls={listboxId}
          aria-autocomplete="list"
          aria-activedescendant={open && shown.length ? optionId(hi) : undefined}
          value={query}
          onChange={(e) => {
            // Typing (re)opens the popover — don't rely on a focus change, which never fires after
            // an in-place Escape.
            setQuery(e.target.value);
            setHi(0);
          }}
          onFocus={() => setFocused(true)}
          onBlur={(e) => {
            // Click-safe: only close when focus leaves the whole combobox (tab-out / click-away).
            if (!boxRef.current?.contains(e.relatedTarget as Node)) setFocused(false);
          }}
          onKeyDown={onKeyDown}
        />
        {open && (
          <div className={styles.popover} role="listbox" id={listboxId}>
            {shown.length === 0 ? (
              <div className={styles.hint}>No matching factors.</div>
            ) : (
              <>
                {shown.map((f, i) => (
                  <button
                    key={f.key}
                    type="button"
                    id={optionId(i)}
                    role="option"
                    aria-selected={i === hi}
                    className={`${styles.option} ${i === hi ? styles.optionHi : ''}`}
                    onMouseEnter={() => setHi(i)}
                    onClick={() => pin(f.key)}
                  >
                    {f.searchLabel}
                  </button>
                ))}
                {truncated && <div className={styles.hint}>Keep typing to narrow…</div>}
              </>
            )}
          </div>
        )}
      </div>

      {byChannel.map(({ channel, features }) => {
        const selFeats = features.filter((f) => selected.has(f.key));
        if (!selFeats.length) return null;
        const def = selFeats.length <= AUTO_EXPAND_MAX;
        const openSec = overrides[channel.id] ?? def;
        return (
          <div key={channel.id} className={styles.section}>
            <div className={styles.sectionHead}>
              <button
                type="button"
                className={styles.sectionToggle}
                aria-expanded={openSec}
                onClick={() => toggleExpand(channel.id, def)}
              >
                <span className={styles.chevron} aria-hidden>
                  {openSec ? '▾' : '▸'}
                </span>
                {channel.label} · {selFeats.length} selected
              </button>
              <button
                type="button"
                className={styles.clear}
                onClick={() => clearChannel(features)}
                title={`Deselect all ${channel.label}`}
                aria-label={`Deselect all ${channel.label}`}
              >
                ✕
              </button>
            </div>
            {openSec && (
              <div className={styles.chips}>
                {selFeats.map((f) => (
                  <span key={f.key} className={styles.chip}>
                    {f.label}
                    <button
                      type="button"
                      className={styles.chipRemove}
                      onClick={() => unpin(f.key)}
                      title={`Remove ${f.label}`}
                      aria-label={`Remove ${f.label}`}
                    >
                      ✕
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
