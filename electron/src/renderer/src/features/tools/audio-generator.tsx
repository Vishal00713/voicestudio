import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { DownloadIcon, LoaderCircleIcon, RotateCwIcon, SparklesIcon } from 'lucide-react';
import { AudioPreviewButton } from '@/components/audio-preview-button';
import { PipelineFailure } from '@/components/pipeline-failure';
import { SaveAudioButton } from '@/components/save-audio-button';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { apiJson, audioUrl, describeError } from '@/lib/api/client';
import { generatorRun, patchGeneratorRun, useGeneratorRun } from '@/lib/store/audio-generator';

interface GeneratorStatus {
  ready: boolean;
  stage: 'idle' | 'installing' | 'downloading' | 'done' | 'ready' | 'error';
  error: string | null;
  license: string;
  download_bytes: number;
  max_seconds: number;
  /** Commercial-safe mode refuses this model, so it is not offered for download. */
  commercial_blocked?: boolean;
}

/** A local text → audio model behind `<api>/status|install|generate`. */
interface GeneratorKind {
  api: '/sfx' | '/music';
  /** i18n namespace holding the page's strings. */
  ns: 'soundEffects' | 'music';
  examples: readonly string[];
  variationChoices: readonly number[];
  defaultVariations: number;
  defaultSeconds: string;
  minSeconds: number;
  fileStem: string;
}

const SOUND_EFFECTS: GeneratorKind = {
  api: '/sfx',
  ns: 'soundEffects',
  examples: ['rain', 'door', 'crowd', 'whoosh', 'footsteps', 'thunder'],
  variationChoices: [1, 2, 3, 4],
  defaultVariations: 4,
  defaultSeconds: '5',
  minSeconds: 1,
  fileStem: 'sound-effect',
};

const MUSIC: GeneratorKind = {
  api: '/music',
  ns: 'music',
  examples: ['lofi', 'cinematic', 'bollywood', 'edm', 'piano', 'acoustic'],
  variationChoices: [1, 2, 3],
  defaultVariations: 1,
  defaultSeconds: '15',
  minSeconds: 5,
  fileStem: 'music',
};

function installing(stage: GeneratorStatus['stage'] | undefined) {
  return stage === 'installing' || stage === 'downloading';
}

/** Text → sound effects, rendered locally (ElevenLabs Sound Effects parity). */
export function SoundEffects() {
  return <AudioGenerator key="sfx" kind={SOUND_EFFECTS} />;
}

/** Text → music, rendered locally (ElevenLabs Music parity). */
export function MusicGenerator() {
  return <AudioGenerator key="music" kind={MUSIC} />;
}

function AudioGenerator({ kind }: { kind: GeneratorKind }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const statusKey = [kind.api, 'status'];
  const status = useQuery({
    queryKey: statusKey,
    queryFn: () => apiJson<GeneratorStatus>(kind.api + '/status'),
    refetchInterval: (query) => (installing(query.state.data?.stage) ? 2000 : false),
  });
  const { prompt, busy, error, takes } = useGeneratorRun(kind.api);
  const setPrompt = (value: string) => patchGeneratorRun(kind.api, { prompt: value });
  const setError = (value: string | null) => patchGeneratorRun(kind.api, { error: value });
  const [seconds, setSeconds] = useState(kind.defaultSeconds);
  const [variations, setVariations] = useState(kind.defaultVariations);
  const data = status.data;
  const maxSeconds = data?.max_seconds ?? 20;
  const duration = Number(seconds);
  const valid = Boolean(prompt.trim()) && duration >= kind.minSeconds && duration <= maxSeconds;
  const text = (key: string, options?: Record<string, unknown>) => t(`${kind.ns}.${key}`, options);

  const install = async () => {
    setError(null);
    try {
      client.setQueryData(
        statusKey,
        await apiJson<GeneratorStatus>(kind.api + '/install', { method: 'POST' }),
      );
    } catch (failure) {
      setError(describeError(failure));
    }
  };

  // The run is kept in a store, not in this component: switching tools
  // mid-render unmounts the page, and the takes must still be here on return.
  const generate = async () => {
    if (generatorRun(kind.api).busy || !valid) return;
    patchGeneratorRun(kind.api, { busy: true, error: null });
    try {
      const body = new FormData();
      body.set('prompt', prompt.trim());
      body.set('seconds', String(duration));
      body.set('variations', String(variations));
      const result = await apiJson<{
        prompt: string;
        takes: NonNullable<typeof takes>['items'];
      }>(kind.api + '/generate', { method: 'POST', body });
      patchGeneratorRun(kind.api, { takes: { prompt: result.prompt, items: result.takes } });
    } catch (failure) {
      patchGeneratorRun(kind.api, { error: describeError(failure) });
    } finally {
      patchGeneratorRun(kind.api, { busy: false });
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <div>
        <h2 className="text-lg font-medium">{text('title')}</h2>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">{text('description')}</p>
      </div>
      {error && <PipelineFailure fallback={error} onDismiss={() => setError(null)} />}
      {status.isPending ? (
        <p role="status" className="text-sm text-muted-foreground">
          {t('common.loading')}
        </p>
      ) : status.isError || !data ? (
        <PipelineFailure
          fallback={t('audioGenerator.status_failed')}
          action={
            <Button size="sm" variant="outline" onClick={() => void status.refetch()}>
              <RotateCwIcon />
              {t('common.retry')}
            </Button>
          }
        />
      ) : data.commercial_blocked ? (
        <p role="status" className="rounded-xl border border-border/60 bg-muted/20 p-4 text-sm">
          {t('audioGenerator.commercial_blocked', { license: data.license })}
        </p>
      ) : !data.ready ? (
        <section className="space-y-3 rounded-xl border border-border/60 bg-muted/20 p-4">
          <p className="text-sm">
            {text('install_body', {
              size: ((data?.download_bytes ?? 0) / 1e9).toFixed(1),
              license: data?.license ?? '',
            })}
          </p>
          {data?.stage === 'error' && data.error && <PipelineFailure fallback={data.error} />}
          <Button disabled={installing(data?.stage)} onClick={() => void install()}>
            {installing(data?.stage) ? (
              <LoaderCircleIcon className="animate-spin motion-reduce:animate-none" />
            ) : (
              <DownloadIcon />
            )}
            {installing(data?.stage)
              ? text(data?.stage === 'downloading' ? 'downloading' : 'installing')
              : text('install')}
          </Button>
        </section>
      ) : (
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            void generate();
          }}
        >
          <label className="block space-y-2 text-sm">
            <span>{text('prompt_label')}</span>
            <textarea
              className="min-h-28 w-full resize-y rounded-lg border border-input bg-transparent p-3 leading-7 outline-none focus-visible:ring-2 focus-visible:ring-ring"
              maxLength={500}
              disabled={busy}
              value={prompt}
              placeholder={text('prompt_placeholder')}
              onChange={(event) => setPrompt(event.target.value)}
            />
          </label>
          <div className="flex flex-wrap gap-1">
            {kind.examples.map((example) => (
              <Button
                key={example}
                type="button"
                size="xs"
                variant="ghost"
                disabled={busy}
                onClick={() => setPrompt(text('example_' + example))}
              >
                <SparklesIcon />
                {text('example_' + example)}
              </Button>
            ))}
          </div>
          <div className="flex flex-wrap items-end gap-4">
            <label className="space-y-2 text-sm">
              <span>{text('seconds_label', { max: maxSeconds })}</span>
              <Input
                className="w-28"
                type="number"
                min={kind.minSeconds}
                max={maxSeconds}
                step={0.5}
                disabled={busy}
                value={seconds}
                onChange={(event) => setSeconds(event.target.value)}
              />
            </label>
            <fieldset className="space-y-2 text-sm" disabled={busy}>
              <legend>{text('variations_label')}</legend>
              <div className="flex gap-1">
                {kind.variationChoices.map((count) => (
                  <Button
                    key={count}
                    type="button"
                    size="sm"
                    variant={variations === count ? 'secondary' : 'ghost'}
                    aria-pressed={variations === count}
                    onClick={() => setVariations(count)}
                  >
                    {count}
                  </Button>
                ))}
              </div>
            </fieldset>
            <Button type="submit" disabled={busy || !valid}>
              {busy ? (
                <LoaderCircleIcon className="animate-spin motion-reduce:animate-none" />
              ) : (
                <SparklesIcon />
              )}
              {text(busy ? 'generating' : 'generate')}
            </Button>
          </div>
        </form>
      )}
      {takes && (
        <section
          aria-label={text('results')}
          className="space-y-2 rounded-xl border border-border/60 p-3"
        >
          <h3 className="text-sm font-medium">{text('results')}</h3>
          <ul className="grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] gap-2">
            {takes.items.map((take, index) => {
              const file = take.audio_url.split('/').pop() ?? '';
              return (
                <li
                  key={take.id}
                  className="flex items-center gap-2 rounded-lg border border-border/60 p-2"
                >
                  <AudioPreviewButton src={audioUrl(file)} source={kind.ns + '-' + take.id} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">
                      {text('take', { number: index + 1 })}
                    </p>
                    <p className="truncate text-xs tabular-nums text-muted-foreground">
                      {take.duration_s == null ? takes.prompt : `${take.duration_s.toFixed(1)}s`}
                    </p>
                  </div>
                  <SaveAudioButton
                    url={audioUrl(file)}
                    suggestedName={`${kind.fileStem}-${file}`}
                  />
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
