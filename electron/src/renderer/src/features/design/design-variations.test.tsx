import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import '@/i18n';
import type { GenerateResult } from '@/lib/api/types';
import {
  addDesignVariation,
  clearDesignVariations,
  designVariationKey,
} from '@/lib/store/design-variations';
import { DesignVariations } from './design-variations';

vi.mock('@/components/audio-preview-button', () => ({
  AudioPreviewButton: ({ src }: { src: string }) => <span data-testid="preview">{src}</span>,
}));

function take(id: string, seed: number): GenerateResult {
  return {
    blob: new Blob(),
    id,
    audioPath: id + '.wav',
    durationSeconds: 1,
    genTimeSeconds: 1,
    seed,
    routing: null,
    dropped: null,
  } as GenerateResult;
}

const WARM = designVariationKey('warm female', { gender: 'female' });

describe('DesignVariations', () => {
  beforeEach(() => {
    clearDesignVariations(WARM);
    addDesignVariation(take('a', 11), 11, 'Hello');
    addDesignVariation(take('b', 22), 22, 'Hello');
  });

  it('shows the candidates of the current description', () => {
    render(<DesignVariations seed={22} recipeKey={WARM} onChoose={() => {}} />);
    expect(screen.getAllByTestId('preview')).toHaveLength(2);
  });

  it('hides candidates once the description changes, since their seeds no longer match', () => {
    const edited = designVariationKey('raspy old male', { gender: 'male' });
    const { container } = render(
      <DesignVariations seed={22} recipeKey={edited} onChoose={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('hands back the chosen take, not just its seed', () => {
    const onChoose = vi.fn();
    render(<DesignVariations seed={22} recipeKey={WARM} onChoose={onChoose} />);
    fireEvent.click(screen.getByRole('button', { name: /Use this voice/ }));
    expect(onChoose).toHaveBeenCalledWith(
      expect.objectContaining({
        seed: 11,
        text: 'Hello',
        result: expect.objectContaining({ id: 'a' }),
      }),
    );
  });
});
