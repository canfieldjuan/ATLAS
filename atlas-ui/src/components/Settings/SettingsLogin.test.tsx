import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SettingsLogin } from './SettingsLogin';

function resp(status: number): Response {
  return { ok: status >= 200 && status < 300, status } as unknown as Response;
}

function typeToken(value: string) {
  fireEvent.change(screen.getByPlaceholderText('Settings admin token'), {
    target: { value },
  });
}

describe('SettingsLogin', () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  it('exchanges the token and calls onAuthed on success', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(200));
    const onAuthed = vi.fn();
    render(<SettingsLogin onAuthed={onAuthed} />);

    typeToken('good-token');
    fireEvent.click(screen.getByRole('button', { name: /unlock/i }));

    await waitFor(() => expect(onAuthed).toHaveBeenCalledTimes(1));
    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/v1/settings/session',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ token: 'good-token' }),
      }),
    );
  });

  it('shows "Invalid admin token." on 401 and does not authenticate', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(401));
    const onAuthed = vi.fn();
    render(<SettingsLogin onAuthed={onAuthed} />);

    typeToken('bad-token');
    fireEvent.click(screen.getByRole('button', { name: /unlock/i }));

    expect(await screen.findByText('Invalid admin token.')).toBeInTheDocument();
    expect(onAuthed).not.toHaveBeenCalled();
  });

  it('shows the not-configured message on 503', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(503));
    render(<SettingsLogin onAuthed={vi.fn()} />);

    typeToken('whatever');
    fireEvent.click(screen.getByRole('button', { name: /unlock/i }));

    expect(
      await screen.findByText('Settings admin is not configured on the server.'),
    ).toBeInTheDocument();
  });

  it('keeps the submit button disabled until a non-blank token is entered', () => {
    render(<SettingsLogin onAuthed={vi.fn()} />);
    const button = screen.getByRole('button', { name: /unlock/i });

    expect(button).toBeDisabled();
    typeToken('   '); // whitespace only
    expect(button).toBeDisabled();
    typeToken('real');
    expect(button).toBeEnabled();
  });
});
