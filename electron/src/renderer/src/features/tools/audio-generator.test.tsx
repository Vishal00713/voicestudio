import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import '@/i18n';
import { apiJson } from '@/lib/api/client';
import { audioGeneratorStore } from '@/lib/store/audio-generator';
import { MusicGenerator, SoundEffects } from './audio-generator';

vi.mock('@/lib/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/client')>()),
  apiJson: vi.fn(),
}));
vi.mock('@/components/audio-preview-button', () => ({
  AudioPreviewButton: ({ src }: { src: string }) => <span data-testid="preview">{src}</span>,
}));
vi.mock('@/components/save-audio-button', () => ({ SaveAudioButton: () => null }));

const base = {
  error: null,
  license: 'CC-BY-NC-SA-4.0',
  download_bytes: 4_480_000_000,
  max_seconds: 20,
  max_variations: 4,
};

function renderPage(page = <SoundEffects />) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{page}</QueryClientProvider>);
}

describe('SoundEffects', () => {
  beforeEach(() => {
    vi.mocked(apiJson).mockReset();
    audioGeneratorStore.setState(() => ({}));
  });

  it('asks before downloading and starts the install only on click', async () => {
    vi.mocked(apiJson).mockImplementation(async (path) =>
      path === '/sfx/install'
        ? { ...base, ready: false, stage: 'installing' }
        : { ...base, ready: false, stage: 'idle' },
    );
    renderPage();
    const install = await screen.findByRole('button', { name: /Install sound effects/ });
    expect(screen.getByText(/4\.5 GB/)).toBeInTheDocument();
    expect(apiJson).not.toHaveBeenCalledWith('/sfx/install', expect.anything());
    fireEvent.click(install);
    await waitFor(() => expect(apiJson).toHaveBeenCalledWith('/sfx/install', { method: 'POST' }));
    expect(await screen.findByRole('button', { name: /Installing/ })).toBeDisabled();
  });

  it('generates the chosen number of takes and lists them', async () => {
    vi.mocked(apiJson).mockImplementation(async (path) =>
      path === '/sfx/generate'
        ? {
            prompt: 'Heavy rain on a tin roof',
            takes: [
              { id: 'a', audio_url: '/audio/a.wav', duration_s: 5 },
              { id: 'b', audio_url: '/audio/b.wav', duration_s: 5 },
            ],
          }
        : { ...base, ready: true, stage: 'ready' },
    );
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Heavy rain on a tin roof/ }));
    fireEvent.click(screen.getByRole('button', { name: '2' }));
    fireEvent.click(screen.getByRole('button', { name: /Generate sound effects/ }));
    expect(await screen.findAllByTestId('preview')).toHaveLength(2);
    const call = vi.mocked(apiJson).mock.calls.find(([path]) => path === '/sfx/generate');
    const body = call?.[1]?.body as FormData;
    expect(body.get('prompt')).toBe('Heavy rain on a tin roof');
    expect(body.get('variations')).toBe('2');
    expect(body.get('seconds')).toBe('5');
  });

  it('music talks to its own endpoints with music defaults', async () => {
    vi.mocked(apiJson).mockImplementation(async (path) =>
      path === '/music/generate'
        ? {
            prompt: 'Calm solo piano melody, gentle and emotional',
            takes: [{ id: 'm', audio_url: '/audio/m.wav', duration_s: 15 }],
          }
        : { ...base, ready: true, stage: 'ready', max_seconds: 30 },
    );
    renderPage(<MusicGenerator />);
    fireEvent.click(await screen.findByRole('button', { name: /Calm solo piano melody/ }));
    expect(screen.getByRole('button', { name: '1' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByRole('button', { name: /Generate music/ }));
    expect(await screen.findAllByTestId('preview')).toHaveLength(1);
    expect(apiJson).toHaveBeenCalledWith('/music/status');
    const call = vi.mocked(apiJson).mock.calls.find(([path]) => path === '/music/generate');
    const body = call?.[1]?.body as FormData;
    expect(body.get('seconds')).toBe('15');
    expect(body.get('variations')).toBe('1');
  });

  it('keeps a render that finishes after switching tools', async () => {
    let finish: (value: unknown) => void = () => {};
    vi.mocked(apiJson).mockImplementation((path) =>
      path === '/sfx/generate'
        ? new Promise((resolve) => (finish = resolve))
        : Promise.resolve({ ...base, ready: true, stage: 'ready' }),
    );
    const page = renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Heavy rain on a tin roof/ }));
    fireEvent.click(screen.getByRole('button', { name: /Generate sound effects/ }));
    await waitFor(() => expect(apiJson).toHaveBeenCalledWith('/sfx/generate', expect.anything()));
    page.unmount();
    finish({
      prompt: 'Heavy rain on a tin roof',
      takes: [{ id: 'a', audio_url: '/audio/a.wav', duration_s: 5 }],
    });
    renderPage();
    expect(await screen.findAllByTestId('preview')).toHaveLength(1);
    expect(screen.getByDisplayValue('Heavy rain on a tin roof')).toBeInTheDocument();
  });

  it('explains a failed status check instead of offering a download', async () => {
    vi.mocked(apiJson).mockRejectedValue(new Error('offline'));
    renderPage();
    expect(await screen.findByText(/Couldn't check this tool's status/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Install sound effects/ })).toBeNull();
    expect(screen.getByRole('button', { name: /Retry/ })).toBeInTheDocument();
  });

  it('offers no download when Commercial-safe mode refuses the model', async () => {
    vi.mocked(apiJson).mockResolvedValue({
      ...base,
      ready: false,
      stage: 'idle',
      commercial_blocked: true,
    });
    renderPage();
    expect(await screen.findByText(/Commercial-safe mode is on/)).toBeInTheDocument();
    expect(screen.getByText(/CC-BY-NC-SA-4\.0/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Install sound effects/ })).toBeNull();
  });
});
