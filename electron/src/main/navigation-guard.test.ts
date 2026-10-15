// @vitest-environment node
import { describe, expect, it, vi } from 'vitest';
import {
  attachNavigationGuards,
  isAllowedFrameUrl,
  isOwnUrl,
  openOutside,
  type NavigationGuardTarget,
} from './navigation-guard';

vi.mock('electron', () => ({
  shell: {
    openExternal: vi.fn().mockResolvedValue(undefined),
  },
}));

import { shell } from 'electron';

describe('isOwnUrl', () => {
  it('accepts valid app://voicestudio URLs', () => {
    expect(isOwnUrl('app://voicestudio')).toBe(true);
    expect(isOwnUrl('app://voicestudio/index.html')).toBe(true);
    expect(isOwnUrl('app://voicestudio/sub/path?query=1#hash')).toBe(true);
  });

  it('accepts configured dev URL', () => {
    const dev = 'http://localhost:5173';
    expect(isOwnUrl('http://localhost:5173/index.html', dev)).toBe(true);
    expect(isOwnUrl('http://localhost:5173/subpage', dev)).toBe(true);
  });

  it('rejects external or unapproved URLs', () => {
    expect(isOwnUrl('https://google.com')).toBe(false);
    expect(isOwnUrl('http://attacker.example.org')).toBe(false);
    expect(isOwnUrl('file:///etc/passwd')).toBe(false);
    expect(isOwnUrl('javascript:alert(1)')).toBe(false);
    expect(isOwnUrl('not-a-valid-url')).toBe(false);
  });
});

describe('isAllowedFrameUrl', () => {
  it('allows own URLs on both main frame and subframes', () => {
    expect(isAllowedFrameUrl('app://voicestudio/index.html', true)).toBe(true);
    expect(isAllowedFrameUrl('app://voicestudio/index.html', false)).toBe(true);
  });

  it('allows approved Google Forms embeds only in subframes', () => {
    expect(isAllowedFrameUrl('https://forms.gle/2PYCvd39hbwijzX37', false)).toBe(true);
    expect(isAllowedFrameUrl('https://docs.google.com/forms/d/e/viewform', false)).toBe(true);
    // Main frame navigation to external forms is strictly disallowed
    expect(isAllowedFrameUrl('https://forms.gle/2PYCvd39hbwijzX37', true)).toBe(false);
    expect(isAllowedFrameUrl('https://docs.google.com/forms/d/e/viewform', true)).toBe(false);
  });

  it('rejects unapproved external hosts in subframes', () => {
    expect(isAllowedFrameUrl('https://malicious.org/embed', false)).toBe(false);
    expect(isAllowedFrameUrl('http://forms.gle/insecure', false)).toBe(false);
    expect(isAllowedFrameUrl('file:///tmp/leak.html', false)).toBe(false);
    // Non-form Google services rejected in subframe
    expect(isAllowedFrameUrl('https://sites.google.com/view/phish', false)).toBe(false);
    expect(isAllowedFrameUrl('https://docs.google.com/document/d/123', false)).toBe(false);
    expect(isAllowedFrameUrl('https://drive.google.com/file', false)).toBe(false);
    // Google accounts authentication allowed for form login
    expect(isAllowedFrameUrl('https://accounts.google.com/ServiceLogin', false)).toBe(true);
  });
});

describe('openOutside', () => {
  it('delegates http, https, and mailto schemes to shell.openExternal', async () => {
    const mockOpen = vi.mocked(shell.openExternal);
    mockOpen.mockClear();

    await openOutside('https://example.com/docs');
    expect(mockOpen).toHaveBeenCalledWith('https://example.com/docs');

    mockOpen.mockClear();
    await openOutside('mailto:support@voicestudio.sh');
    expect(mockOpen).toHaveBeenCalledWith('mailto:support@voicestudio.sh');
  });

  it('ignores file and disallowed protocol schemes', async () => {
    const mockOpen = vi.mocked(shell.openExternal);
    mockOpen.mockClear();

    await openOutside('file:///bin/sh');
    await openOutside('javascript:alert(1)');
    await openOutside('invalid-uri');
    expect(mockOpen).not.toHaveBeenCalled();
  });
});

describe('attachNavigationGuards', () => {
  function makeMockTarget() {
    let windowOpenHandler: ((details: { url: string }) => { action: string }) | null = null;
    const listeners: Record<string, ((...args: any[]) => void)[]> = {};

    const target: NavigationGuardTarget = {
      setWindowOpenHandler: vi.fn((handler) => {
        windowOpenHandler = handler;
      }),
      on: vi.fn((event: string, listener: any) => {
        listeners[event] = listeners[event] || [];
        listeners[event].push(listener);
        return target;
      }),
    };

    return {
      target,
      getWindowOpenHandler: () => windowOpenHandler,
      emitWillNavigate: (event: any, url: string) => {
        for (const fn of listeners['will-navigate'] || []) fn(event, url);
      },
      emitWillFrameNavigate: (event: any) => {
        for (const fn of listeners['will-frame-navigate'] || []) fn(event);
      },
    };
  }

  it('denies window.open and forwards URL to openOutside', () => {
    const mockOpenOutside = vi.fn().mockResolvedValue(undefined);
    const { target, getWindowOpenHandler } = makeMockTarget();

    attachNavigationGuards(target, mockOpenOutside);

    const handler = getWindowOpenHandler();
    expect(handler).toBeDefined();

    const result = handler!({ url: 'https://example.com' });
    expect(result).toEqual({ action: 'deny' });
    expect(mockOpenOutside).toHaveBeenCalledWith('https://example.com');
  });

  it('guards top-level will-navigate', () => {
    const mockOpenOutside = vi.fn().mockResolvedValue(undefined);
    const { target, emitWillNavigate } = makeMockTarget();

    attachNavigationGuards(target, mockOpenOutside);

    // Own URL allowed without preventDefault
    const ownEvent = { preventDefault: vi.fn() };
    emitWillNavigate(ownEvent, 'app://voicestudio/index.html');
    expect(ownEvent.preventDefault).not.toHaveBeenCalled();
    expect(mockOpenOutside).not.toHaveBeenCalled();

    // External URL blocked and opened outside
    const externalEvent = { preventDefault: vi.fn() };
    emitWillNavigate(externalEvent, 'https://evil.com/phish');
    expect(externalEvent.preventDefault).toHaveBeenCalledOnce();
    expect(mockOpenOutside).toHaveBeenCalledWith('https://evil.com/phish');
  });

  it('guards frame-level will-frame-navigate', () => {
    const mockOpenOutside = vi.fn().mockResolvedValue(undefined);
    const { target, emitWillFrameNavigate } = makeMockTarget();

    attachNavigationGuards(target, mockOpenOutside);

    // Own frame URL allowed
    const ownFrameEvent = { preventDefault: vi.fn(), url: 'app://voicestudio/iframe.html', isMainFrame: false };
    emitWillFrameNavigate(ownFrameEvent);
    expect(ownFrameEvent.preventDefault).not.toHaveBeenCalled();
    expect(mockOpenOutside).not.toHaveBeenCalled();

    // Approved sponsor form embed allowed in subframe
    const sponsorFrameEvent = {
      preventDefault: vi.fn(),
      url: 'https://forms.gle/2PYCvd39hbwijzX37',
      isMainFrame: false,
    };
    emitWillFrameNavigate(sponsorFrameEvent);
    expect(sponsorFrameEvent.preventDefault).not.toHaveBeenCalled();
    expect(mockOpenOutside).not.toHaveBeenCalled();

    // Main frame navigation to external forms blocked and opened outside
    const mainFrameSponsorEvent = {
      preventDefault: vi.fn(),
      url: 'https://forms.gle/2PYCvd39hbwijzX37',
      isMainFrame: true,
    };
    emitWillFrameNavigate(mainFrameSponsorEvent);
    expect(mainFrameSponsorEvent.preventDefault).toHaveBeenCalledOnce();
    expect(mockOpenOutside).toHaveBeenCalledWith('https://forms.gle/2PYCvd39hbwijzX37');

    // Untrusted iframe navigation blocked without opening outside
    mockOpenOutside.mockClear();
    const externalFrameEvent = { preventDefault: vi.fn(), url: 'https://evil.com/iframe', isMainFrame: false };
    emitWillFrameNavigate(externalFrameEvent);
    expect(externalFrameEvent.preventDefault).toHaveBeenCalledOnce();
    expect(mockOpenOutside).not.toHaveBeenCalled();
  });
});
