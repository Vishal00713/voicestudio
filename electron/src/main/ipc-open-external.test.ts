// @vitest-environment node
import { beforeEach, describe, expect, it, vi } from 'vitest';

const native = vi.hoisted(() => ({
  handlers: new Map<string, (event: unknown, raw: unknown) => Promise<unknown>>(),
  listeners: new Map<string, (event: unknown) => void>(),
  openExternal: vi.fn(async () => {}),
  openPath: vi.fn(async () => ''),
}));

const owner = vi.hoisted(() => {
  const mainFrame = { url: 'app://voicestudio/index.html' };
  const webContents = { mainFrame };
  return {
    webContents,
    isMaximized: () => false,
    minimize: vi.fn(),
    unmaximize: vi.fn(),
    maximize: vi.fn(),
    close: vi.fn(),
  };
});

vi.mock('electron', () => ({
  app: { getPath: () => '/tmp', exit: vi.fn() },
  BrowserWindow: { fromWebContents: () => owner, getAllWindows: () => [owner] },
  dialog: {
    showSaveDialog: vi.fn(),
    showOpenDialog: vi.fn(),
  },
  ipcMain: {
    handle: (channel: string, handler: (event: unknown, raw: unknown) => Promise<unknown>) =>
      native.handlers.set(channel, handler),
    on: (channel: string, listener: (event: unknown) => void) =>
      native.listeners.set(channel, listener),
  },
  net: { fetch: vi.fn() },
  shell: {
    openExternal: native.openExternal,
    openPath: native.openPath,
    showItemInFolder: vi.fn(),
  },
  systemPreferences: {},
}));
vi.mock('./site-browser', () => ({ registerSiteBrowser: vi.fn() }));

import { CHANNELS, registerIpc } from './ipc';
import type { BackendSupervisor } from './backend';

const trustedEvent = { sender: owner.webContents, senderFrame: owner.webContents.mainFrame };
const untrustedEvent = {
  sender: owner.webContents,
  senderFrame: { url: 'https://malicious.example.com' },
};

describe('files:openExternal IPC handler', () => {
  beforeEach(() => {
    native.handlers.clear();
    native.openExternal.mockClear();
    native.openPath.mockClear();
    registerIpc(
      {
        subscribe: () => () => {},
        baseUrl: 'http://127.0.0.1:3900',
        requestHeaders: () => ({}),
      } as unknown as BackendSupervisor,
      () => owner as never,
    );
  });

  it('allows http, https, and mailto protocols from trusted frame', async () => {
    const handler = native.handlers.get(CHANNELS.filesOpenExternal)!;
    expect(handler).toBeDefined();

    await handler(trustedEvent, 'https://example.com/docs');
    expect(native.openExternal).toHaveBeenCalledWith('https://example.com/docs');

    await handler(trustedEvent, 'http://example.com/api');
    expect(native.openExternal).toHaveBeenCalledWith('http://example.com/api');

    await handler(trustedEvent, 'mailto:support@example.com');
    expect(native.openExternal).toHaveBeenCalledWith('mailto:support@example.com');
  });

  it('rejects file URLs to prevent local file execution', async () => {
    const handler = native.handlers.get(CHANNELS.filesOpenExternal)!;
    await expect(handler(trustedEvent, 'file:///etc/passwd')).rejects.toThrow(
      'Refusing to open file: URLs',
    );
    expect(native.openExternal).not.toHaveBeenCalled();
  });

  it('rejects non-allowlisted protocols', async () => {
    const handler = native.handlers.get(CHANNELS.filesOpenExternal)!;
    await expect(handler(trustedEvent, 'javascript:alert(1)')).rejects.toThrow(
      'Refusing to open javascript: URLs',
    );
    await expect(handler(trustedEvent, 'custom-scheme://test')).rejects.toThrow(
      'Refusing to open custom-scheme: URLs',
    );
  });

  it('rejects absolute paths and never delegates to shell.openPath', async () => {
    const handler = native.handlers.get(CHANNELS.filesOpenExternal)!;
    await expect(handler(trustedEvent, '/bin/sh')).rejects.toThrow();
    expect(native.openPath).not.toHaveBeenCalled();
  });

  it('rejects invocations from untrusted sender frames', async () => {
    const handler = native.handlers.get(CHANNELS.filesOpenExternal)!;
    await expect(handler(untrustedEvent, 'https://example.com')).rejects.toThrow(
      'Untrusted main-frame request',
    );
  });

  it('window:* listeners ignore untrusted frames without throwing uncaught exceptions', () => {
    const minimize = native.listeners.get(CHANNELS.windowMinimize)!;
    const toggleMaximize = native.listeners.get(CHANNELS.windowToggleMaximize)!;
    const close = native.listeners.get(CHANNELS.windowClose)!;

    expect(() => minimize(untrustedEvent)).not.toThrow();
    expect(owner.minimize).not.toHaveBeenCalled();

    expect(() => toggleMaximize(untrustedEvent)).not.toThrow();
    expect(owner.maximize).not.toHaveBeenCalled();

    expect(() => close(untrustedEvent)).not.toThrow();
    expect(owner.close).not.toHaveBeenCalled();
  });

  it('window:* listeners execute actions when invoked from trusted frame', () => {
    const minimize = native.listeners.get(CHANNELS.windowMinimize)!;
    const toggleMaximize = native.listeners.get(CHANNELS.windowToggleMaximize)!;
    const close = native.listeners.get(CHANNELS.windowClose)!;

    minimize(trustedEvent);
    expect(owner.minimize).toHaveBeenCalledOnce();

    toggleMaximize(trustedEvent);
    expect(owner.maximize).toHaveBeenCalledOnce();

    close(trustedEvent);
    expect(owner.close).toHaveBeenCalledOnce();
  });
});
