import { shell } from 'electron';
import { isTrustedRenderer } from './trusted-renderer';

export function isOwnUrl(url: string, devUrl?: string): boolean {
  const dev = devUrl ?? process.env.ELECTRON_RENDERER_URL;
  return isTrustedRenderer(url, dev);
}

const ALLOWED_EXTERNAL_PROTOCOLS = new Set(['http:', 'https:', 'mailto:']);

export async function openOutside(url: string): Promise<void> {
  try {
    const parsed = new URL(url);
    if (ALLOWED_EXTERNAL_PROTOCOLS.has(parsed.protocol)) {
      await shell.openExternal(parsed.toString());
    }
  } catch {
    /* unparsable URLs are simply dropped */
  }
}

const ALLOWED_FORM_HOSTS = new Set(['forms.gle']);

export function isAllowedFrameUrl(
  url: string,
  isMainFrame: boolean = false,
  devUrl?: string,
): boolean {
  if (isOwnUrl(url, devUrl)) return true;
  if (isMainFrame) return false;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:') return false;
    if (ALLOWED_FORM_HOSTS.has(parsed.hostname)) return true;
    if (parsed.hostname === 'docs.google.com' && parsed.pathname.startsWith('/forms/')) return true;
    if (parsed.hostname === 'accounts.google.com') return true;
    return false;
  } catch {
    return false;
  }
}

export interface NavigationGuardTarget {
  setWindowOpenHandler(handler: (details: { url: string }) => { action: 'deny' | 'allow' }): void;
  on(event: 'will-navigate', listener: (event: { preventDefault: () => void }, url: string) => void): unknown;
  on(
    event: 'will-frame-navigate',
    listener: (event: { preventDefault: () => void; url: string; isMainFrame?: boolean }) => void,
  ): unknown;
}

export function attachNavigationGuards(
  contents: NavigationGuardTarget,
  openOutsideFn: (url: string) => Promise<void> = openOutside,
  devUrl?: string,
): void {
  contents.setWindowOpenHandler(({ url }) => {
    void openOutsideFn(url);
    return { action: 'deny' };
  });
  contents.on('will-navigate', (event, url) => {
    if (isOwnUrl(url, devUrl)) return;
    event.preventDefault();
    void openOutsideFn(url);
  });
  contents.on('will-frame-navigate', (event) => {
    if (isAllowedFrameUrl(event.url, event.isMainFrame, devUrl)) return;
    event.preventDefault();
    if (event.isMainFrame) {
      void openOutsideFn(event.url);
    }
  });
}
