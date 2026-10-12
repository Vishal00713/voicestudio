import { describe, expect, it } from 'vitest';
import { hasNativeScript, looksRomanized, shouldOfferNativeScript } from './native-script';

describe('native script offer', () => {
  it('knows which languages have their own script', () => {
    expect(hasNativeScript('Hindi')).toBe(true);
    expect(hasNativeScript('Tamil')).toBe(true);
    expect(hasNativeScript('English')).toBe(false);
    expect(hasNativeScript('Auto')).toBe(false);
  });

  it('detects romanized text and ignores tags', () => {
    expect(looksRomanized('kya haal hai aap sab ka')).toBe(true);
    expect(looksRomanized('क्या हाल है आप सब का')).toBe(false);
    expect(looksRomanized('[pause 300ms] [laughter] है')).toBe(false);
    expect(looksRomanized('ok')).toBe(false);
    expect(looksRomanized('meeting में CEO ने plan बताया')).toBe(true);
  });

  it('offers only for a native-script language with romanized text', () => {
    expect(shouldOfferNativeScript('kya haal hai', 'Hindi')).toBe(true);
    expect(shouldOfferNativeScript('kya haal hai', 'English')).toBe(false);
    expect(shouldOfferNativeScript('क्या हाल है', 'Hindi')).toBe(false);
  });
});
