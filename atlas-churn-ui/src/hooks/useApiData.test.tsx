import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import useApiData from './useApiData'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe('useApiData', () => {
  beforeEach(() => {
    vi.useRealTimers()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('loads initially and uses the lightweight refreshing state after data is present', async () => {
    const first = deferred<number>()
    const second = deferred<number>()
    const fetcher = vi
      .fn<() => Promise<number>>()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)

    const { result } = renderHook(() => useApiData(fetcher, []))

    expect(result.current.loading).toBe(true)
    expect(result.current.refreshing).toBe(false)

    await act(async () => {
      first.resolve(1)
      await first.promise
    })

    await waitFor(() => {
      expect(result.current.data).toBe(1)
    })
    expect(result.current.loading).toBe(false)

    act(() => {
      result.current.refresh()
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.refreshing).toBe(true)

    await act(async () => {
      second.resolve(2)
      await second.promise
    })

    await waitFor(() => {
      expect(result.current.data).toBe(2)
    })
    expect(result.current.refreshing).toBe(false)
    expect(fetcher).toHaveBeenCalledTimes(2)
  })

  it('returns a refresh promise that resolves after refreshed data is applied', async () => {
    const first = deferred<number>()
    const second = deferred<number>()
    const fetcher = vi
      .fn<() => Promise<number>>()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)

    const { result } = renderHook(() => useApiData(fetcher, []))

    await act(async () => {
      first.resolve(1)
      await first.promise
    })

    await waitFor(() => {
      expect(result.current.data).toBe(1)
    })

    let refreshPromise: Promise<void> | null = null
    act(() => {
      refreshPromise = result.current.refresh()
    })

    expect(result.current.refreshing).toBe(true)

    await act(async () => {
      second.resolve(2)
      await refreshPromise
    })

    expect(result.current.data).toBe(2)
    expect(result.current.refreshing).toBe(false)
  })

  it('ignores stale responses after deps change', async () => {
    const first = deferred<string>()
    const second = deferred<string>()
    const fetcher = vi.fn((dep: string) => (dep === 'first' ? first.promise : second.promise))

    const { result, rerender } = renderHook(
      ({ dep }) => useApiData(() => fetcher(dep), [dep]),
      { initialProps: { dep: 'first' } },
    )

    rerender({ dep: 'second' })

    await act(async () => {
      second.resolve('newer')
      await second.promise
    })

    await waitFor(() => {
      expect(result.current.data).toBe('newer')
    })

    await act(async () => {
      first.resolve('older')
      await first.promise
    })

    expect(result.current.data).toBe('newer')
    expect(fetcher).toHaveBeenCalledTimes(2)
  })
})
