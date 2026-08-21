// Regression coverage for the Style-A loader fix (Codex #3, R2): the three
// forms that gate render on `!form` must surface a load failure as an error
// banner instead of spinning forever. A build/typecheck cannot prove this —
// only rendering the component with a failing initial GET does.
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { IntegrationSettingsForm } from './IntegrationSettings';
import { LLMSettingsForm } from './LLMSettings';
import { NotificationSettingsForm } from './NotificationSettings';

function resp(status: number): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => ({}),
  } as unknown as Response;
}

const FORMS = [
  { name: 'IntegrationSettingsForm', Comp: IntegrationSettingsForm },
  { name: 'LLMSettingsForm', Comp: LLMSettingsForm },
  { name: 'NotificationSettingsForm', Comp: NotificationSettingsForm },
] as const;

describe('Style-A Settings forms surface load errors instead of an endless spinner', () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  it.each(FORMS)(
    '$name renders the error banner and drops the spinner when the initial GET rejects',
    async ({ Comp }) => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('offline'));
      render(<Comp />);
      expect(await screen.findByText(/Load failed/)).toBeInTheDocument();
      expect(screen.queryByText(/Loading…/)).not.toBeInTheDocument();
    },
  );

  it.each(FORMS)(
    '$name renders the error banner when the initial GET returns 401 (session lost)',
    async ({ Comp }) => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(401));
      render(<Comp />);
      expect(await screen.findByText(/Load failed/)).toBeInTheDocument();
      expect(screen.queryByText(/Loading…/)).not.toBeInTheDocument();
    },
  );
});
