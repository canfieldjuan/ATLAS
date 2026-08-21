import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SettingsModal } from './SettingsModal';

// Stub the tab forms — they each fetch on mount and are irrelevant to the
// auth-gate logic under test here.
vi.mock('./VoiceSettings', () => ({ VoiceSettingsForm: () => <div>voice-form</div> }));
vi.mock('./EmailSettings', () => ({ EmailSettingsForm: () => <div>email-form</div> }));
vi.mock('./DailyIntelligenceSettings', () => ({ DailySettingsForm: () => <div>daily-form</div> }));
vi.mock('./NewsIntelligenceSettings', () => ({ IntelligenceSettingsForm: () => <div>news-form</div> }));
vi.mock('./LLMSettings', () => ({ LLMSettingsForm: () => <div>llm-form</div> }));
vi.mock('./NotificationSettings', () => ({ NotificationSettingsForm: () => <div>notif-form</div> }));
vi.mock('./IntegrationSettings', () => ({ IntegrationSettingsForm: () => <div>integrations-form</div> }));

function resp(status: number): Response {
  return { ok: status >= 200 && status < 300, status } as unknown as Response;
}

/** Route fetch by method: GET = the auth probe, POST = login, DELETE = logout. */
function routeFetch(probe: number, session: number) {
  globalThis.fetch = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    return Promise.resolve(resp(method === 'GET' ? probe : session));
  }) as typeof fetch;
}

describe('SettingsModal auth gate', () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  it('reveals the tabs when the probe reports authed', async () => {
    routeFetch(200, 200);
    render(<SettingsModal onClose={() => {}} />);

    expect(await screen.findByText('Voice Pipeline')).toBeInTheDocument();
    expect(screen.getByText('voice-form')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /lock/i })).toBeInTheDocument();
  });

  it('shows the login form when the probe says need-login', async () => {
    routeFetch(401, 200);
    render(<SettingsModal onClose={() => {}} />);

    expect(await screen.findByText('Unlock settings')).toBeInTheDocument();
    expect(screen.queryByText('Voice Pipeline')).not.toBeInTheDocument();
  });

  it('shows the not-configured banner on 503', async () => {
    routeFetch(503, 200);
    render(<SettingsModal onClose={() => {}} />);

    expect(
      await screen.findByText('Settings admin is not configured on the server.'),
    ).toBeInTheDocument();
  });

  it('shows the unreachable banner when the probe throws', async () => {
    globalThis.fetch = vi.fn(() => Promise.reject(new Error('offline'))) as typeof fetch;
    render(<SettingsModal onClose={() => {}} />);

    expect(
      await screen.findByText('Could not reach the settings admin API.'),
    ).toBeInTheDocument();
  });

  it('logs in from the login form and reveals the tabs', async () => {
    routeFetch(401, 200); // probe → need-login, session POST → success
    render(<SettingsModal onClose={() => {}} />);

    const input = await screen.findByPlaceholderText('Settings admin token');
    fireEvent.change(input, { target: { value: 'tok' } });
    fireEvent.click(screen.getByRole('button', { name: /unlock/i }));

    expect(await screen.findByText('Voice Pipeline')).toBeInTheDocument();
  });

  it('logs out and returns to the login form when the server confirms deletion', async () => {
    routeFetch(200, 200); // probe → authed, DELETE → 2xx (confirmed cleared)
    render(<SettingsModal onClose={() => {}} />);

    fireEvent.click(await screen.findByRole('button', { name: /lock/i }));

    expect(await screen.findByText('Unlock settings')).toBeInTheDocument();
    expect(screen.queryByText('Voice Pipeline')).not.toBeInTheDocument();
  });

  it('stays authed and shows a retryable error when logout is not confirmed', async () => {
    routeFetch(200, 500); // probe → authed, DELETE → non-2xx (cookie NOT cleared)
    render(<SettingsModal onClose={() => {}} />);

    fireEvent.click(await screen.findByRole('button', { name: /lock/i }));

    // The Lock did not clear the cookie, so we must not present a logged-out
    // state: surface a retryable error and keep the tabs.
    expect(await screen.findByText(/Couldn't lock/)).toBeInTheDocument();
    expect(screen.getByText('Voice Pipeline')).toBeInTheDocument();
    expect(screen.queryByText('Unlock settings')).not.toBeInTheDocument();
  });
});
