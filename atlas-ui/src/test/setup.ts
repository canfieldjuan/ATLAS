// Vitest global setup: register jest-dom matchers (toBeInTheDocument, etc.)
// and reset the DOM + mocks between tests so cases stay isolated.
import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
