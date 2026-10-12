import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import '@/i18n';
import { ScriptPanel } from './script-panel';

const state = vi.hoisted(() => ({ text: 'kya haal hai aap sab ka', language: 'Hindi' }));
const setCloneSetting = vi.fn();
const transliterateScript = vi.fn();

vi.mock('@/lib/languages', () => ({
  LANGUAGES: ['Auto'],
  POPULAR_LANGUAGES: [],
  PERFORMANCE_TAGS: [],
  TAGS: [],
}));

vi.mock('@/lib/store/clone-settings', () => ({
  useCloneSetting: (key: 'text' | 'language') => state[key],
  setCloneSetting: (...args: unknown[]) => setCloneSetting(...args),
}));

vi.mock('@/lib/api/transliterate', () => ({
  transliterateScript: (...args: unknown[]) => transliterateScript(...args),
}));

beforeEach(() => {
  setCloneSetting.mockReset();
  transliterateScript.mockReset();
  state.text = 'kya haal hai aap sab ka';
  state.language = 'Hindi';
});

describe('ScriptPanel native script', () => {
  it('rewrites romanized Hindi in Devanagari', async () => {
    transliterateScript.mockResolvedValue({ text: 'क्या हाल है आप सब का', changed: true });
    render(<ScriptPanel />);
    fireEvent.click(screen.getByRole('button', { name: /native script/i }));
    await waitFor(() =>
      expect(setCloneSetting).toHaveBeenCalledWith('text', 'क्या हाल है आप सब का'),
    );
    expect(transliterateScript).toHaveBeenCalledWith('kya haal hai aap sab ka', 'Hindi');
  });

  it('is not offered for Devanagari text or a Latin-script language', () => {
    state.text = 'क्या हाल है आप सब का';
    const { unmount } = render(<ScriptPanel />);
    expect(screen.queryByRole('button', { name: /native script/i })).toBeNull();
    unmount();
    state.text = 'kya haal hai aap sab ka';
    state.language = 'English';
    render(<ScriptPanel />);
    expect(screen.queryByRole('button', { name: /native script/i })).toBeNull();
  });

  it('leaves the text alone when the conversion fails', async () => {
    transliterateScript.mockRejectedValue(new Error('boom'));
    render(<ScriptPanel />);
    fireEvent.click(screen.getByRole('button', { name: /native script/i }));
    await waitFor(() => expect(transliterateScript).toHaveBeenCalled());
    expect(setCloneSetting).not.toHaveBeenCalled();
  });
});
