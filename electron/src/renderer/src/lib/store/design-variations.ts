import { Store } from '@tanstack/store';
import { useStore } from '@tanstack/react-store';
import type { GenerateResult } from '@/lib/api/types';
import { audioUrl } from '@/lib/api/client';
import { createObjectUrl, revokeObjectUrl } from '@/lib/audio/object-url';

/** One candidate voice rendered from the same description with its own seed. */
export interface DesignVariation {
  seed: number;
  /** Served take, when the backend kept one; otherwise the object URL. */
  src: string;
  objectUrl: string | null;
  durationSeconds: number | null;
  /** The take itself, so choosing this candidate can show it as the output. */
  result: GenerateResult;
  text: string;
}

interface DesignVariationsState {
  /** The description the candidates were rendered from (see designVariationKey). */
  key: string;
  items: DesignVariation[];
}

export const designVariationsStore = new Store<DesignVariationsState>({ key: '', items: [] });

/** What makes the voice: a seed only reproduces a candidate with the same
 *  instruct and recipe, so candidates belong to this key. */
export function designVariationKey(instruct: string, recipe: unknown): string {
  return JSON.stringify([instruct, recipe ?? null]);
}

/** The candidates for ``key``; none when the description has changed since. */
export function useDesignVariations(key: string): DesignVariation[] {
  return useStore(designVariationsStore, (state) => (state.key === key ? state.items : NONE));
}

const NONE: DesignVariation[] = [];

export function clearDesignVariations(key = ''): void {
  for (const variation of designVariationsStore.state.items) revokeObjectUrl(variation.objectUrl);
  designVariationsStore.setState(() => ({ key, items: [] }));
}

export function addDesignVariation(result: GenerateResult, seed: number, text: string): void {
  const objectUrl = createObjectUrl(result.blob);
  const src = result.audioPath ? audioUrl(result.audioPath) : objectUrl;
  if (!src) return;
  designVariationsStore.setState((current) => ({
    ...current,
    items: [
      ...current.items,
      {
        seed: result.seed ?? seed,
        src,
        objectUrl,
        durationSeconds: result.durationSeconds,
        result,
        text,
      },
    ],
  }));
}
