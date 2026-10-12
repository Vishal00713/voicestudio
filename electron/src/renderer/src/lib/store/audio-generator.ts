import { Store } from '@tanstack/store';
import { useStore } from '@tanstack/react-store';

export interface GeneratedTake {
  id: string;
  audio_url: string;
  duration_s: number | null;
}

/** One tool's prompt and its latest run. Lives outside the page, so a render
 *  started before switching tools still lands when the user comes back. */
export interface GeneratorRun {
  prompt: string;
  busy: boolean;
  error: string | null;
  takes: { prompt: string; items: GeneratedTake[] } | null;
}

const EMPTY: GeneratorRun = { prompt: '', busy: false, error: null, takes: null };

export const audioGeneratorStore = new Store<Record<string, GeneratorRun>>({});

export function generatorRun(api: string): GeneratorRun {
  return audioGeneratorStore.state[api] ?? EMPTY;
}

export function useGeneratorRun(api: string): GeneratorRun {
  return useStore(audioGeneratorStore, (state) => state[api] ?? EMPTY);
}

export function patchGeneratorRun(api: string, patch: Partial<GeneratorRun>): void {
  audioGeneratorStore.setState((state) => ({
    ...state,
    [api]: { ...(state[api] ?? EMPTY), ...patch },
  }));
}
