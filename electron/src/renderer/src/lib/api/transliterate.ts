import { apiJson } from './client';

export interface TransliterateResult {
  text: string;
  changed: boolean;
}

/**
 * Rewrite romanized text (kya haal hai) in the language's own script
 * (क्या हाल है) with the backend's transliteration LLM skill. Rejects with an
 * ApiError: 503 when no LLM is set up, 502 when the rewrite failed its checks.
 */
export function transliterateScript(
  text: string,
  language: string,
  signal?: AbortSignal,
): Promise<TransliterateResult> {
  return apiJson<TransliterateResult>('/tools/transliterate', {
    method: 'POST',
    body: JSON.stringify({ text, language }),
    signal,
  });
}
