/**
 * Languages with their own script, where romanized text (Hinglish such as
 * "kya haal hai") is read with English pronunciation by every engine. Mirrors
 * `_TARGETS` in backend/services/transliteration.py.
 */
const NATIVE_SCRIPT_LANGUAGES = new Set([
  'Hindi',
  'Marathi',
  'Nepali',
  'Bengali',
  'Punjabi',
  'Gujarati',
  'Tamil',
  'Telugu',
  'Kannada',
  'Malayalam',
  'Urdu',
  'Arabic',
  'Standard Arabic',
  'Persian',
  'Russian',
  'Ukrainian',
  'Greek',
  'Thai',
]);

const TAG = /\[\[[^\]\n]*\]\]|\[[^[\]\n]{0,128}\]/g;
const LETTER = /\p{L}/u;
const LATIN = /[A-Za-z\u00C0-\u024F]/;

export function hasNativeScript(language: string): boolean {
  return NATIVE_SCRIPT_LANGUAGES.has(language);
}

/** True when most letters (outside [tags]) are Latin and there are enough to matter. */
export function looksRomanized(text: string): boolean {
  let letters = 0;
  let latin = 0;
  for (const ch of text.replace(TAG, ' ')) {
    if (!LETTER.test(ch)) continue;
    letters += 1;
    if (LATIN.test(ch)) latin += 1;
  }
  return letters >= 4 && latin / letters >= 0.6;
}

/** Offer the conversion: a native-script language with a romanized script. */
export function shouldOfferNativeScript(text: string, language: string): boolean {
  return hasNativeScript(language) && looksRomanized(text);
}
