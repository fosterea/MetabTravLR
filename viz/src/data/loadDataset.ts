/** Fetch helpers for the generated data contract (public/data/…). */
import type { Dataset, EdgeBundle, EntityKind, Manifest, NbhdBundle } from './types';

const base = import.meta.env.BASE_URL; // './' by config -> relative to current path

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${base}data/${path}`);
  if (!res.ok) throw new Error(`Failed to load data/${path}: ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const fetchManifest = () => getJson<Manifest>('manifest.json');

export const fetchDataset = (id: string) => getJson<Dataset>(`${id}/dataset.json`);

export const fetchEdgeBundle = (id: string, tier: string, kind: EntityKind) =>
  getJson<EdgeBundle>(`${id}/edges/${tier}.${kind}.json`);

export const fetchNbhdBundle = (id: string, tier: string) =>
  getJson<NbhdBundle>(`${id}/nbhd/${tier}.json`);
