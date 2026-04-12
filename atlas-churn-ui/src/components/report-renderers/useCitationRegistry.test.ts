import { describe, expect, it } from 'vitest'
import { createCitationRegistry } from './useCitationRegistry'

describe('createCitationRegistry', () => {
  it('deduplicates witness registrations and preserves first metadata', () => {
    const registry = createCitationRegistry()

    const firstIndex = registry.register('witness-1', {
      companyName: 'Acme Corp',
      excerptSnippet: 'Pricing kept rising',
      source: 'g2',
    })
    const secondIndex = registry.register('witness-1', {
      companyName: 'Other Co',
      excerptSnippet: 'This should be ignored',
      source: 'reddit',
    })

    expect(firstIndex).toBe(1)
    expect(secondIndex).toBe(1)
    expect(registry.getIndex('witness-1')).toBe(1)
    expect(registry.getAll()).toEqual([
      {
        index: 1,
        witnessId: 'witness-1',
        companyName: 'Acme Corp',
        excerptSnippet: 'Pricing kept rising',
        source: 'g2',
      },
    ])
  })

  it('assigns incrementing indexes to distinct witnesses', () => {
    const registry = createCitationRegistry()

    expect(registry.register('witness-1')).toBe(1)
    expect(registry.register('witness-2', { companyName: 'Bravo' })).toBe(2)
    expect(registry.getAll().map((entry) => entry.index)).toEqual([1, 2])
  })
})
