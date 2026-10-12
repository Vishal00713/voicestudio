import languagesJson from './languages.json';

/** Display names the backend accepts as `language`; index 0 is 'Auto' (omitted on the wire). */
export const LANGUAGES: string[] = languagesJson;

export const POPULAR_LANGUAGES: string[] = [
  'English',
  'Spanish',
  'French',
  'German',
  'Italian',
  'Portuguese',
  'Russian',
  'Chinese',
  'Japanese',
  'Korean',
  'Arabic',
  'Hindi',
];

/** ElevenLabs-style cues the backend lowers for every engine (services/audio_tags.py). */
export const PERFORMANCE_TAGS: string[] = ['[whispers]', '[short pause]', '[long pause]'];

/** Expression tokens the engine renders as non-verbal sounds when inlined in the script. */
export const TAGS: string[] = [
  '[laughter]',
  '[sigh]',
  '[confirmation-en]',
  '[question-en]',
  '[question-ah]',
  '[question-oh]',
  '[question-ei]',
  '[question-yi]',
  '[surprise-ah]',
  '[surprise-oh]',
  '[surprise-wa]',
  '[surprise-yo]',
  '[dissatisfaction-hnn]',
];
