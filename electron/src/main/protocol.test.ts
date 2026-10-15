// @vitest-environment node
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileResponse, serveStatic } from './protocol';

describe('protocol response headers and security boundaries', () => {
  let tempDir: string;
  let sampleHtml: string;
  let sampleJs: string;

  beforeAll(async () => {
    tempDir = await mkdtemp(join(tmpdir(), 'protocol-test-'));
    sampleHtml = join(tempDir, 'index.html');
    sampleJs = join(tempDir, 'main.js');
    await writeFile(sampleHtml, '<!DOCTYPE html><html><body>Test</body></html>', 'utf-8');
    await writeFile(sampleJs, 'console.log("hello");', 'utf-8');
  });

  afterAll(async () => {
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true });
    }
  });

  it('fileResponse returns null for missing paths and non-files', async () => {
    expect(await fileResponse(join(tempDir, 'non-existent.txt'))).toBeNull();
    expect(await fileResponse(tempDir)).toBeNull();
  });

  it('fileResponse returns security headers for static files', async () => {
    const res = await fileResponse(sampleHtml);
    expect(res).not.toBeNull();
    expect(res?.status).toBe(200);
    expect(res?.headers.get('content-type')).toBe('text/html; charset=utf-8');
    expect(res?.headers.get('x-content-type-options')).toBe('nosniff');
    expect(res?.headers.get('x-frame-options')).toBe('DENY');
    expect(res?.headers.get('cross-origin-resource-policy')).toBe('same-origin');
    expect(res?.headers.get('cache-control')).toBe('no-cache');

    const jsRes = await fileResponse(sampleJs);
    expect(jsRes).not.toBeNull();
    expect(jsRes?.headers.get('content-type')).toBe('text/javascript; charset=utf-8');
    expect(jsRes?.headers.get('x-content-type-options')).toBe('nosniff');
    expect(jsRes?.headers.get('x-frame-options')).toBe('DENY');
    expect(jsRes?.headers.get('cross-origin-resource-policy')).toBe('same-origin');
  });

  it('serveStatic blocks null bytes and malformed paths with 400', async () => {
    const res = await serveStatic(tempDir, '/index%00.html');
    expect(res.status).toBe(400);
    expect(res.headers.get('x-content-type-options')).toBe('nosniff');
    expect(res.headers.get('x-frame-options')).toBe('DENY');

    const malformed = await serveStatic(tempDir, '/%E0%A4%A');
    expect(malformed.status).toBe(400);
    expect(malformed.headers.get('x-content-type-options')).toBe('nosniff');
  });

  it('serveStatic blocks path traversal with 403', async () => {
    const res = await serveStatic(tempDir, '/../../etc/passwd');
    expect(res.status).toBe(403);
    expect(res.headers.get('x-content-type-options')).toBe('nosniff');
    expect(res.headers.get('x-frame-options')).toBe('DENY');

    const encoded = await serveStatic(tempDir, '/%2e%2e/secret');
    expect(encoded.status).toBe(403);

    const backslash = await serveStatic(tempDir, '/..%5Csecret');
    expect(backslash.status).toBe(403);
  });

  it('serveStatic serves root index.html and assets with defensive headers', async () => {
    const res = await serveStatic(tempDir, '/');
    expect(res.status).toBe(200);
    expect(res.headers.get('x-content-type-options')).toBe('nosniff');
    expect(res.headers.get('x-frame-options')).toBe('DENY');

    const jsRes = await serveStatic(tempDir, '/main.js');
    expect(jsRes.status).toBe(200);
    expect(jsRes.headers.get('x-content-type-options')).toBe('nosniff');
  });

  it('serveStatic returns 404 with security headers on missing assets', async () => {
    const res = await serveStatic(tempDir, '/assets/missing.png');
    expect(res.status).toBe(404);
    expect(res.headers.get('x-content-type-options')).toBe('nosniff');
    expect(res.headers.get('x-frame-options')).toBe('DENY');
    expect(res.headers.get('cross-origin-resource-policy')).toBe('same-origin');
  });
});
