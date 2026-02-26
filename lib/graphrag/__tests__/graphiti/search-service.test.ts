/**
 * Search Service Unit Tests
 * Tests for threshold filtering, deduplication, and search functionality
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { GraphitiSearchResult } from '../../graphiti/client';
import type { GraphRAGRetrievalMetadata } from '../../types';
import { log } from '@/lib/utils/logger';

// Use vi.hoisted to define mocks before they're hoisted
const {
  mockSearch,
  mockGetEntityEdges,
  mockClientInstance,
  mockRerank,
  mockCreateReranker,
  mockConfig,
} = vi.hoisted(() => {
  const mockSearch = vi.fn();
  const mockGetEntityEdges = vi.fn();
  const mockRerank = vi.fn();
  const mockCreateReranker = vi.fn(() => ({ rerank: mockRerank }));
  const mockClientInstance = {
    search: mockSearch,
    getEntityEdges: mockGetEntityEdges,
  };
  const mockConfig = {
    search: {
      topK: 10,
      searchMethod: 'hybrid',
      threshold: 0.7,
    },
    reranking: {
      enabled: false,
      type: 'heuristic',
      topK: 10,
    },
  };
  return {
    mockSearch,
    mockGetEntityEdges,
    mockClientInstance,
    mockRerank,
    mockCreateReranker,
    mockConfig,
  };
});

// Mock dependencies before importing
vi.mock('../../graphiti/client', () => ({
  getGraphitiClient: vi.fn(() => mockClientInstance),
}));

vi.mock('../../config', () => ({
  graphragConfig: mockConfig,
}));

vi.mock('../../reranking', () => ({
  createReranker: mockCreateReranker,
}));

vi.mock('@/lib/tracing/trace.service', () => ({
  traceService: {
    createChildSpan: vi.fn(),
    endTrace: vi.fn(),
  },
}));

vi.mock('@/lib/utils/logger', () => ({
  log: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

import { SearchService } from '../../graphiti/search-service';

describe('SearchService', () => {
  let searchService: SearchService;

  beforeEach(() => {
    vi.clearAllMocks();
    mockConfig.reranking.enabled = false;
    searchService = new SearchService();
  });

  describe('Threshold Filtering', () => {
    it('should filter out results below threshold', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'High confidence fact', score: 0.9, created_at: '2024-01-01' },
          { uuid: '2', name: 'rel2', fact: 'Low confidence fact', score: 0.3, created_at: '2024-01-01' },
          { uuid: '3', name: 'rel3', fact: 'Medium confidence fact', score: 0.7, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      // Should only include edges with score >= 0.7 (threshold)
      expect(result.sources.length).toBe(2);
      expect(result.sources.map(s => s.fact)).toContain('High confidence fact');
      expect(result.sources.map(s => s.fact)).toContain('Medium confidence fact');
      expect(result.sources.map(s => s.fact)).not.toContain('Low confidence fact');
    });

    it('should include edges exactly at threshold', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Exact threshold fact', score: 0.7, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.sources.length).toBe(1);
      expect(result.sources[0].fact).toBe('Exact threshold fact');
    });

    it('should return empty results when all below threshold', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Low fact 1', score: 0.5, created_at: '2024-01-01' },
          { uuid: '2', name: 'rel2', fact: 'Low fact 2', score: 0.3, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.sources.length).toBe(0);
      expect(result.context).toBe('');
    });

    it('should handle edges with undefined score', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'No score fact', created_at: '2024-01-01' },
          { uuid: '2', name: 'rel2', fact: 'High score fact', score: 0.9, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      // Undefined score defaults to 0, which is below threshold
      expect(result.sources.length).toBe(1);
      expect(result.sources[0].fact).toBe('High score fact');
    });
  });

  describe('Deduplication', () => {
    it('should deduplicate sources with same content prefix', async () => {
      // The deduplication uses first 100 chars normalized as fingerprint
      // These two facts have identical first 100 characters (they're identical)
      const mockResult: GraphitiSearchResult = {
        edges: [
          {
            uuid: '1',
            name: 'rel1',
            fact: 'The RTX 4090 GPU provides exceptional gaming performance with ray tracing capabilities',
            score: 0.9,
            created_at: '2024-01-01'
          },
          {
            uuid: '2',
            name: 'rel2',
            fact: 'The RTX 4090 GPU provides exceptional gaming performance with ray tracing capabilities',
            score: 0.85,
            created_at: '2024-01-01'
          },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('RTX 4090 specs', 'user-123');

      // Identical facts should be deduplicated
      expect(result.sources.length).toBe(1);
    });

    it('should keep sources with different content', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          {
            uuid: '1',
            name: 'rel1',
            fact: 'The RTX 4090 has 16384 CUDA cores',
            score: 0.9,
            created_at: '2024-01-01'
          },
          {
            uuid: '2',
            name: 'rel2',
            fact: 'The RTX 3090 has 10496 CUDA cores',
            score: 0.85,
            created_at: '2024-01-01'
          },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('GPU comparison', 'user-123');

      // Different facts should both be kept
      expect(result.sources.length).toBe(2);
    });

    it('should handle single source without deduplication', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Single fact', score: 0.9, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.sources.length).toBe(1);
    });

    it('should normalize whitespace during deduplication', async () => {
      // These facts differ only in whitespace - after normalization they should match
      const mockResult: GraphitiSearchResult = {
        edges: [
          {
            uuid: '1',
            name: 'rel1',
            fact: 'The  GPU   has  high   performance  and   excellent  cooling',
            score: 0.9,
            created_at: '2024-01-01'
          },
          {
            uuid: '2',
            name: 'rel2',
            fact: 'The GPU has high performance and excellent cooling',
            score: 0.85,
            created_at: '2024-01-01'
          },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('GPU performance', 'user-123');

      // Whitespace normalized, both become "the gpu has high performance and excellent cooling"
      expect(result.sources.length).toBe(1);
    });
  });

  describe('Rescore Loop', () => {
    it('should rescore results when top score is below threshold', async () => {
      mockConfig.reranking.enabled = true;
      mockRerank.mockResolvedValue([
        { originalIndex: 0, score: 0.9, text: 'Low fact 1', metadata: {} },
        { originalIndex: 1, score: 0.75, text: 'Low fact 2', metadata: {} },
      ]);
      searchService = new SearchService();

      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Low fact 1', score: 0.4, created_at: '2024-01-01' },
          { uuid: '2', name: 'rel2', fact: 'Low fact 2', score: 0.2, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(mockRerank).toHaveBeenCalledTimes(1);
      expect(result.sources.length).toBe(2);
      expect(result.sources[0].confidence).toBeCloseTo(0.9, 3);
      expect(result.sources[1].confidence).toBeCloseTo(0.75, 3);
    });

    it('should fall back to original scores when rescoring fails', async () => {
      mockConfig.reranking.enabled = true;
      mockRerank.mockRejectedValue(new Error('rescore failed'));
      searchService = new SearchService();

      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Low fact 1', score: 0.4, created_at: '2024-01-01' },
          { uuid: '2', name: 'rel2', fact: 'Low fact 2', score: 0.2, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(mockRerank).toHaveBeenCalledTimes(1);
      expect(log.error).toHaveBeenCalled();
      expect(result.sources.length).toBe(0);
      expect(result.context).toBe('');
    });

    it('should skip rescore when scores already exceed threshold', async () => {
      mockConfig.reranking.enabled = true;
      mockRerank.mockResolvedValue([
        { originalIndex: 0, score: 0.8, text: 'High fact', metadata: {} },
      ]);
      searchService = new SearchService();

      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'High fact', score: 0.8, created_at: '2024-01-01' },
          { uuid: '2', name: 'rel2', fact: 'Below threshold', score: 0.6, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(mockRerank).toHaveBeenCalledTimes(1);
      expect(mockRerank.mock.calls[0][1]).toHaveLength(1);
      expect(result.sources.length).toBe(1);
      expect(result.sources[0].fact).toBe('High fact');
    });
  });

  describe('Context Building', () => {
    it('should build context string from filtered edges', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          {
            uuid: '1',
            name: 'rel1',
            fact: 'First important fact',
            score: 0.9,
            source_description: 'Document A',
            created_at: '2024-01-01'
          },
          {
            uuid: '2',
            name: 'rel2',
            fact: 'Second important fact',
            score: 0.8,
            source_description: 'Document B',
            created_at: '2024-01-01'
          },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.context).toContain('Relevant information from knowledge graph');
      expect(result.context).toContain('First important fact');
      expect(result.context).toContain('Second important fact');
      expect(result.context).toContain('confidence');
    });

    it('should return empty context when no results pass threshold', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Low fact', score: 0.3, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.context).toBe('');
    });

    it('should include source descriptions in context', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          {
            uuid: '1',
            name: 'rel1',
            fact: 'Test fact',
            score: 0.9,
            source_description: 'Technical Documentation v2.0',
            created_at: '2024-01-01'
          },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.context).toContain('Technical Documentation v2.0');
    });
  });

  describe('Metadata', () => {
    it('should include correct metadata in response', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Fact 1', score: 0.9, created_at: '2024-01-01' },
          { uuid: '2', name: 'rel2', fact: 'Fact 2', score: 0.8, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.metadata.searchMethod).toBe('hybrid');
      expect(result.metadata.resultsCount).toBe(2);
      expect(result.metadata.queryTime).toBeGreaterThanOrEqual(0);
      const extendedMeta = result.metadata as GraphRAGRetrievalMetadata;
      expect(extendedMeta.graph_used).toBe(true);
      expect(extendedMeta.nodes_retrieved).toBe(2);
    });

    it('should calculate average relevance score', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Fact 1', score: 0.9, created_at: '2024-01-01' },
          { uuid: '2', name: 'rel2', fact: 'Fact 2', score: 0.8, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      // Average of 0.9 and 0.8 = 0.85
      const extendedMeta = result.metadata as GraphRAGRetrievalMetadata;
      expect(extendedMeta.context_relevance_score).toBeCloseTo(0.85, 2);
    });

    it('should set graph_used to false when no results', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      const extendedMeta = result.metadata as GraphRAGRetrievalMetadata;
      expect(extendedMeta.graph_used).toBe(false);
      expect(extendedMeta.nodes_retrieved).toBe(0);
    });
  });

  describe('Source Extraction', () => {
    it('should extract entity name from source node', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          {
            uuid: '1',
            name: 'HAS_FEATURE',
            fact: 'GPU has raytracing',
            score: 0.9,
            source_node: { name: 'RTX 4090', uuid: 'node-1' },
            created_at: '2024-01-01'
          },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('GPU features', 'user-123');

      expect(result.sources[0].entity).toBe('RTX 4090');
      expect(result.sources[0].relation).toBe('HAS_FEATURE');
    });

    it('should use Unknown when source node is missing', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'Orphan fact', score: 0.9, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.sources[0].entity).toBe('Unknown');
    });

    it('should preserve confidence scores in sources', async () => {
      const mockResult: GraphitiSearchResult = {
        edges: [
          { uuid: '1', name: 'rel1', fact: 'High confidence', score: 0.95, created_at: '2024-01-01' },
        ],
      };

      mockSearch.mockResolvedValue(mockResult);

      const result = await searchService.search('test query', 'user-123');

      expect(result.sources[0].confidence).toBe(0.95);
    });
  });

  describe('Citation Formatting', () => {
    it('should format citations correctly', () => {
      const sources = [
        { entity: 'Doc1', relation: 'rel1', fact: 'Fact 1', confidence: 0.9, sourceDescription: 'Source A' },
        { entity: 'Doc2', relation: 'rel2', fact: 'Fact 2', confidence: 0.8, sourceDescription: 'Source B' },
      ];

      const citations = searchService.formatCitations(sources);

      expect(citations).toContain('[1] Source A (90% confidence)');
      expect(citations).toContain('[2] Source B (80% confidence)');
    });

    it('should return empty string for no sources', () => {
      const citations = searchService.formatCitations([]);
      expect(citations).toBe('');
    });

    it('should use entity name when source description missing', () => {
      const sources = [
        { entity: 'EntityName', relation: 'rel1', fact: 'Fact', confidence: 0.9 },
      ];

      const citations = searchService.formatCitations(sources);

      expect(citations).toContain('EntityName');
    });
  });

  describe('Relevance Check', () => {
    it('should return true when sources above min confidence', () => {
      const result = {
        context: 'Some context',
        sources: [
          { entity: 'E1', relation: 'r1', fact: 'F1', confidence: 0.8 },
        ],
        metadata: { searchMethod: 'hybrid' as const, resultsCount: 1, queryTime: 100 },
      };

      expect(searchService.hasRelevantResults(result, 0.5)).toBe(true);
    });

    it('should return false when all sources below min confidence', () => {
      const result = {
        context: 'Some context',
        sources: [
          { entity: 'E1', relation: 'r1', fact: 'F1', confidence: 0.3 },
        ],
        metadata: { searchMethod: 'hybrid' as const, resultsCount: 1, queryTime: 100 },
      };

      expect(searchService.hasRelevantResults(result, 0.5)).toBe(false);
    });
  });
});
