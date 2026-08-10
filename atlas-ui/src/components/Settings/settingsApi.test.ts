import { describe, it, expect, vi, beforeEach } from 'vitest';
import { loginSettings, logoutSettings, probeSettingsAuth } from './settingsApi';

const SESSION_URL = '/api/v1/settings/session';
const PROBE_URL = '/api/v1/settings/notifications';

// Minimal Response stand-in: the helpers only read `.ok` and `.status`.
function resp(status: number): Response {
  return { ok: status >= 200 && status < 300, status } as unknown as Response;
}

describe('settingsApi', () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  describe('loginSettings', () => {
    it('200 → "ok", POSTing {token} as JSON to the session URL', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(200));
      expect(await loginSettings('secret-token')).toBe('ok');
      expect(globalThis.fetch).toHaveBeenCalledWith(
        SESSION_URL,
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: 'secret-token' }),
        }),
      );
    });

    it('401 → "invalid"', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(401));
      expect(await loginSettings('x')).toBe('invalid');
    });

    it('503 → "unavailable"', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(503));
      expect(await loginSettings('x')).toBe('unavailable');
    });

    it('any other non-2xx → "error"', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(500));
      expect(await loginSettings('x')).toBe('error');
    });

    it('network rejection → "error" (never throws)', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('offline'));
      expect(await loginSettings('x')).toBe('error');
    });
  });

  describe('probeSettingsAuth', () => {
    it('GETs the probe URL', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(200));
      await probeSettingsAuth();
      expect(globalThis.fetch).toHaveBeenCalledWith(PROBE_URL);
    });

    it('200 → "authed"', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(200));
      expect(await probeSettingsAuth()).toBe('authed');
    });

    it('401 → "need-login"', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(401));
      expect(await probeSettingsAuth()).toBe('need-login');
    });

    it('503 → "unavailable"', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(503));
      expect(await probeSettingsAuth()).toBe('unavailable');
    });

    it('any other non-2xx → "error"', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(500));
      expect(await probeSettingsAuth()).toBe('error');
    });

    it('network rejection → "error" (never throws)', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('offline'));
      expect(await probeSettingsAuth()).toBe('error');
    });
  });

  describe('logoutSettings', () => {
    it('DELETEs the session URL', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(resp(200));
      await logoutSettings();
      expect(globalThis.fetch).toHaveBeenCalledWith(SESSION_URL, { method: 'DELETE' });
    });

    it('swallows network errors (best-effort, resolves void)', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('offline'));
      await expect(logoutSettings()).resolves.toBeUndefined();
    });
  });
});
