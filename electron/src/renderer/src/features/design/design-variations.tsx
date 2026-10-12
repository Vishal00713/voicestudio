import { CheckIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { AudioPreviewButton } from '@/components/audio-preview-button';
import { Button } from '@/components/ui/button';
import { useDesignVariations } from '@/lib/store/design-variations';
import type { DesignVariation } from '@/lib/store/design-variations';
import { cn } from '@/lib/utils';

interface DesignVariationsProps {
  /** The seed the draft currently holds; its card shows as chosen. */
  seed: number;
  /** designVariationKey of the current description; candidates from another one are hidden. */
  recipeKey: string;
  disabled?: boolean;
  onChoose(variation: DesignVariation): void;
}

/** Candidate voices from one description: listen, compare, keep one. */
export function DesignVariations({ seed, recipeKey, disabled, onChoose }: DesignVariationsProps) {
  const { t } = useTranslation();
  const variations = useDesignVariations(recipeKey);
  if (variations.length === 0) return null;
  return (
    <section
      aria-label={t('designVariations.title')}
      className="space-y-2 rounded-xl border border-border/60 bg-muted/20 p-3"
    >
      <div className="space-y-0.5">
        <h2 className="text-sm font-medium">{t('designVariations.title')}</h2>
        <p className="text-xs text-muted-foreground">{t('designVariations.hint')}</p>
      </div>
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] gap-2">
        {variations.map((variation, index) => {
          const chosen = variation.seed === seed;
          return (
            <li
              key={variation.seed + ':' + index}
              className={cn(
                'flex items-center gap-2 rounded-lg border p-2',
                chosen ? 'border-primary/50 bg-primary/8' : 'border-border/60',
              )}
            >
              <AudioPreviewButton
                src={variation.src}
                fallbackSrc={variation.objectUrl ?? undefined}
                source={'design-variation-' + variation.seed}
              />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">
                  {t('designVariations.voice', { number: index + 1 })}
                </p>
                <p className="truncate text-xs tabular-nums text-muted-foreground">
                  {variation.durationSeconds == null
                    ? t('clone.seed_label') + ' ' + variation.seed
                    : `${variation.durationSeconds.toFixed(1)}s · ${variation.seed}`}
                </p>
              </div>
              <Button
                size="xs"
                variant={chosen ? 'secondary' : 'outline'}
                aria-pressed={chosen}
                disabled={disabled}
                onClick={() => onChoose(variation)}
              >
                {chosen && <CheckIcon />}
                {chosen ? t('designVariations.chosen') : t('designVariations.choose')}
              </Button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
