/**
 * Search Service
 * Handles searching and retrieving information from knowledge graph
 */

import { getGraphitiClient, type GraphitiSearchParams, type GraphitiSearchResult } from './client';
import { graphragConfig } from '../config';
import type { SearchResult, SearchSource, GraphRAGRetrievalMetadata } from '../types';
import { traceService } from '@/lib/tracing/trace.service';
import type { TraceContext, RAGOutputData } from '@/lib/tracing/types';
import { log } from '@/lib/utils/logger';
import { createReranker, type IReranker } from '../reranking';
import { fallbackService } from '../service/fallback-service';
import { temporalClassifier } from '../utils/temporal-classifier';
import { expandQuery, shouldExpandQuery, getBestVariant } from '../utils/query-expansion';

// ============================================================================
// Search Service
// ============================================================================

export class SearchService {
  private client = getGraphitiClient();
  private reranker: IReranker | null = null;

  constructor() {
    // Initialize reranker if enabled
    if (graphragConfig.reranking?.enabled) {
      this.reranker = createReranker({
        type: graphragConfig.reranking.type,
        model: graphragConfig.reranking.model,
        topK: graphragConfig.reranking.topK,
        endpoint: graphragConfig.reranking.endpoint,
      });
      log.info('GraphRAG', 'Reranker initialized', {
        type: graphragConfig.reranking.type,
        topK: graphragConfig.reranking.topK,
      });
    }
  }

  /**
   * Search knowledge graph with hybrid search
   */
  async search(query: string, userId: string, parentContext?: TraceContext): Promise<SearchResult> {
    const startTime = Date.now();
    let retrievalContext: TraceContext | undefined;

    // Start trace for GraphRAG retrieval if parent context provided
    if (parentContext) {
      try {
        retrievalContext = await traceService.createChildSpan(
          parentContext,
          'graphrag.retrieve',
          'retrieval'
        );
        log.debug('GraphRAG', 'Started retrieval trace');
      } catch (traceErr) {
        log.error('GraphRAG', 'Failed to start retrieval trace', { error: traceErr });
      }
    }

    // Expand query to improve search recall
    let searchQuery = query;
    if (shouldExpandQuery(query)) {
      const expanded = expandQuery(query);
      searchQuery = getBestVariant(expanded);
      if (searchQuery !== query) {
        log.debug('GraphRAG', 'Query expanded', {
          original: query,
          expanded: searchQuery,
          variants: expanded.variants,
          transformations: expanded.transformationsApplied,
        });
      }
    }

    // Detect temporal intent from query
    const temporalIntent = temporalClassifier.detect(query);

    const params: GraphitiSearchParams = {
      query: searchQuery,
      group_ids: [userId],
      num_results: graphragConfig.search.topK,
    };

    // Apply temporal filters if detected
    if (temporalIntent.isHistorical !== undefined) {
      params.is_historical = temporalIntent.isHistorical;
    }
    if (temporalIntent.dateFrom) {
      params.date_from = temporalIntent.dateFrom;
    }
    if (temporalIntent.dateTo) {
      params.date_to = temporalIntent.dateTo;
    }
    if (temporalIntent.dataSourceType) {
      params.data_source_type = temporalIntent.dataSourceType;
    }

    log.debug('GraphRAG', 'Calling Graphiti search', {
      params,
      temporalIntent: Object.keys(temporalIntent).length > 0 ? temporalIntent : 'none',
    });

    const graphitiResult = await this.client.search(params);
    const edges = graphitiResult.edges ?? [];

    const threshold = graphragConfig.search.threshold;
    const initialTopScore = this.getMaxScore(edges);
    let edgesForFiltering = edges;
    let rescoreApplied = false;

    log.debug('GraphRAG', 'Graphiti search returned', {
      edgesCount: edges.length,
      firstEdgeScore: edges[0]?.score,
      topScore: initialTopScore,
    });

    if (this.reranker && edges.length > 0 && initialTopScore < threshold) {
      try {
        edgesForFiltering = await this.rescoreEdges(query, edges);
        rescoreApplied = true;
        log.debug('GraphRAG', 'Rescore loop applied', {
          threshold,
          initialTopScore,
          rescoredTopScore: this.getMaxScore(edgesForFiltering),
        });
      } catch (rescoreError) {
        log.error('GraphRAG', 'Rescore loop failed, using original scores', { error: rescoreError });
        edgesForFiltering = edges;
      }
    }

    // Apply threshold filtering from config
    const filteredEdges = edgesForFiltering.filter(
      edge => (edge.score || 0) >= threshold
    );

    log.debug('GraphRAG', 'Threshold filtering applied', {
      threshold,
      beforeCount: edgesForFiltering.length,
      afterCount: filteredEdges.length,
      rescoreApplied,
    });

    // Apply reranking if enabled (skip if rescore already used reranker output)
    let finalEdges = filteredEdges;
    if (this.reranker && filteredEdges.length > 0 && !rescoreApplied) {
      try {
        const candidates = filteredEdges.map(edge => ({
          text: edge.fact,
          score: edge.score || 0,
          metadata: {
            uuid: edge.uuid,
            sourceDescription: edge.source_description,
            createdAt: edge.created_at,
          },
        }));

        const reranked = await this.reranker.rerank(query, candidates);

        // Map reranked results back to edges
        finalEdges = reranked.map(r => filteredEdges[r.originalIndex]);

        log.debug('GraphRAG', 'Reranking applied', {
          before: filteredEdges.length,
          after: finalEdges.length,
          topScore: reranked[0]?.score,
        });
      } catch (rerankError) {
        log.error('GraphRAG', 'Reranking failed, using filtered edges', { error: rerankError });
        // Fall back to filtered edges on error
      }
    }

    // Build context and sources from final results
    const context = this.buildContextFromEdges(finalEdges);
    const rawSources = this.extractSourcesFromEdges(finalEdges);
    const sources = this.deduplicateSourcesByContent(rawSources);

    log.debug('GraphRAG', 'Sources deduplicated', {
      before: rawSources.length,
      after: sources.length
    });

    // Calculate relevance score from final edges (after reranking)
    const relevanceScores = finalEdges.map(e => e.score || 0);
    const avgRelevance = relevanceScores.length > 0
      ? relevanceScores.reduce((a, b) => a + b, 0) / relevanceScores.length
      : 0;

    const queryTime = Date.now() - startTime;

    // End retrieval trace with structured data
    if (retrievalContext) {
      try {
        const { truncateString, truncateRAGChunks } = await import('@/lib/tracing/trace-utils');

        const inputData = {
          query: truncateString(query, 500),
          searchMethod: graphragConfig.search.searchMethod,
          topK: graphragConfig.search.topK,
        };

        const outputData: RAGOutputData = {
          topChunks: truncateRAGChunks(
            finalEdges.map(edge => ({
              fact: edge.fact,
              score: edge.score || 0,
              sourceDescription: edge.source_description,
              entityName: edge.source_node?.name,
            })),
            5
          ),
          totalCandidates: edges.length, // Original count before filtering
          avgConfidence: avgRelevance,
        };

        // Calculate context tokens (approximate)
        const contextTokens = context.length / 4;

        await traceService.endTrace(retrievalContext, {
          endTime: new Date(),
          status: 'completed',
          inputData,
          outputData,
          ragContext: {
            contextTokens: Math.ceil(contextTokens),
            retrievalLatencyMs: queryTime,
            chunkDeduplicationCount: 0, // Graphiti handles deduplication internally
            cacheHitCount: 0, // No caching layer yet
          }
        });
        log.debug('GraphRAG', 'Ended retrieval trace successfully');
      } catch (traceErr) {
        log.error('GraphRAG', 'Failed to end retrieval trace', { error: traceErr });
      }
    }

    return {
      context,
      sources,
      metadata: {
        // Existing fields
        searchMethod: graphragConfig.search.searchMethod,
        resultsCount: finalEdges.length,
        queryTime,

        // New GraphRAG analytics fields
        graph_used: finalEdges.length > 0,
        nodes_retrieved: finalEdges.length,
        context_chunks_used: sources.length,
        retrieval_time_ms: queryTime,
        context_relevance_score: avgRelevance,
        answer_grounded_in_graph: sources.length > 0,
      } as GraphRAGRetrievalMetadata,
    };
  }

  /**
   * Search with custom options
   */
  async searchCustom(
    query: string,
    userId: string,
    options?: { topK?: number; threshold?: number }
  ): Promise<SearchResult> {
    const startTime = Date.now();

    const params: GraphitiSearchParams = {
      query,
      group_ids: [userId],
      num_results: options?.topK || graphragConfig.search.topK,
    };

    const graphitiResult = await this.client.search(params);

    // Apply threshold filtering (use option or config default)
    const threshold = options?.threshold ?? graphragConfig.search.threshold;
    const filteredEdges = graphitiResult.edges.filter(
      edge => (edge.score || 0) >= threshold
    );

    const context = this.buildContextFromEdges(filteredEdges);
    const rawSources = this.extractSourcesFromEdges(filteredEdges);
    const sources = this.deduplicateSourcesByContent(rawSources);

    // Calculate relevance score from filtered edges
    const relevanceScores = filteredEdges.map(e => e.score || 0);
    const avgRelevance = relevanceScores.length > 0
      ? relevanceScores.reduce((a, b) => a + b, 0) / relevanceScores.length
      : 0;

    const queryTime = Date.now() - startTime;

    return {
      context,
      sources,
      metadata: {
        // Existing fields
        searchMethod: 'hybrid',
        resultsCount: filteredEdges.length,
        queryTime,

        // New GraphRAG analytics fields
        graph_used: filteredEdges.length > 0,
        nodes_retrieved: filteredEdges.length,
        context_chunks_used: sources.length,
        retrieval_time_ms: queryTime,
        context_relevance_score: avgRelevance,
        answer_grounded_in_graph: sources.length > 0,
      } as GraphRAGRetrievalMetadata,
    };
  }

  /**
   * Get entity relationships
   */
  async getRelatedEntities(entityName: string, userId: string): Promise<SearchResult> {
    const startTime = Date.now();

    const graphitiResult = await this.client.getEntityEdges(entityName, [userId]);

    const context = this.buildContext(graphitiResult);
    const sources = this.extractSources(graphitiResult);

    return {
      context,
      sources,
      metadata: {
        searchMethod: 'hybrid',
        resultsCount: graphitiResult.edges.length,
        queryTime: Date.now() - startTime,
      },
    };
  }

  /**
   * Build context string from Graphiti results
   */
  private buildContext(result: GraphitiSearchResult): string {
    if (!result.edges.length) {
      return '';
    }

    return this.buildContextFromEdges(result.edges);
  }

  /**
   * Build context from edges
   */
  private buildContextFromEdges(edges: GraphitiSearchResult['edges']): string {
    const facts = edges.map((edge, index) => {
      const source = edge.source_description || 'Document';
      const confidence = edge.score ? ` (confidence: ${(edge.score * 100).toFixed(1)}%)` : '';
      return `${index + 1}. ${edge.fact}${confidence}\n   Source: ${source}`;
    });

    if (!facts.length) {
      return '';
    }

    return `Relevant information from knowledge graph:\n\n${facts.join('\n\n')}`;
  }

  /**
   * Extract sources from Graphiti results
   */
  private extractSources(result: GraphitiSearchResult): SearchSource[] {
    return this.extractSourcesFromEdges(result.edges);
  }

  /**
   * Extract sources from edges
   */
  private extractSourcesFromEdges(edges: GraphitiSearchResult['edges']): SearchSource[] {
    return edges.map(edge => ({
      entity: edge.source_node?.name || 'Unknown',
      relation: edge.name,
      fact: edge.fact,
      confidence: edge.score || 0,
      sourceDescription: edge.source_description,
    }));
  }

  /**
   * Deduplicate similar facts using content fingerprinting
   * Prevents token waste from redundant results
   */
  private deduplicateSourcesByContent(sources: SearchSource[]): SearchSource[] {
    if (sources.length <= 1) return sources;

    const unique: SearchSource[] = [];
    const seen = new Set<string>();

    for (const source of sources) {
      // Create fingerprint from first 100 chars normalized
      const fingerprint = source.fact
        .slice(0, 100)
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();

      if (!seen.has(fingerprint)) {
        unique.push(source);
        seen.add(fingerprint);
      }
    }

    return unique;
  }

  private getMaxScore(edges: GraphitiSearchResult['edges']): number {
    if (!edges.length) return 0;
    return edges.reduce((max, edge) => Math.max(max, edge.score || 0), 0);
  }

  private normalizeScore(score: number, edge?: GraphitiSearchResult['edges'][number]): number {
    if (!Number.isFinite(score)) {
      log.warn('GraphRAG', 'Non-finite rescore value, defaulting to 0', {
        score,
        edgeUuid: edge?.uuid ?? 'unknown',
        edgeFact: edge?.fact?.slice(0, 120) ?? 'unknown',
      });
      return 0;
    }
    return Math.min(1, Math.max(0, score));
  }

  private async rescoreEdges(
    query: string,
    edges: GraphitiSearchResult['edges']
  ): Promise<GraphitiSearchResult['edges']> {
    if (!this.reranker || edges.length === 0) {
      return edges;
    }

    const candidates = edges.map(edge => ({
      text: edge.fact,
      score: edge.score || 0,
      metadata: {
        uuid: edge.uuid,
        sourceDescription: edge.source_description,
        createdAt: edge.created_at,
      },
    }));

    const rescored = await this.reranker.rerank(query, candidates);

    return rescored.reduce<GraphitiSearchResult['edges']>((acc, result) => {
      const edge = edges[result.originalIndex];
      if (!edge) {
        log.warn('GraphRAG', 'Rescore result index out of bounds', {
          originalIndex: result.originalIndex,
          edgeCount: edges.length,
        });
        return acc;
      }
      acc.push({
        ...edge,
        score: this.normalizeScore(result.score, edge),
      });
      return acc;
    }, []);
  }

  /**
   * Format sources for citation
   */
  formatCitations(sources: SearchSource[]): string {
    if (!sources.length) {
      return '';
    }

    const citations = sources.map((source, index) => {
      const confidence = source.confidence > 0 
        ? ` (${(source.confidence * 100).toFixed(0)}% confidence)`
        : '';
      return `[${index + 1}] ${source.sourceDescription || source.entity}${confidence}`;
    });

    return `\n\nSources:\n${citations.join('\n')}`;
  }

  /**
   * Search with automatic fallback when results are insufficient
   */
  async searchWithFallback(
    query: string,
    userId: string,
    parentContext?: TraceContext
  ): Promise<SearchResult> {
    // Execute primary search
    const primaryResult = await this.search(query, userId, parentContext);

    // Check if fallback should be triggered
    if (!fallbackService.shouldTriggerFallback(primaryResult.sources.length)) {
      return primaryResult;
    }

    log.info('GraphRAG', 'Triggering fallback search', {
      primaryResults: primaryResult.sources.length,
      threshold: graphragConfig.fallback?.minResultsThreshold,
    });

    // Execute fallback
    const fallbackResult = await fallbackService.executeFallback({
      userId,
      query,
    });

    // Merge results
    const mergedSources = this.mergeAndDeduplicateSources([
      ...primaryResult.sources,
      ...fallbackResult.sources,
    ]);

    const mergedContext = [
      primaryResult.context,
      fallbackResult.context,
    ].filter(Boolean).join('\n\n---\n\n');

    return {
      context: mergedContext,
      sources: mergedSources,
      metadata: {
        ...primaryResult.metadata,
        fallbackUsed: true,
        fallbackStrategy: fallbackResult.strategy,
        fallbackResults: fallbackResult.sources.length,
      } as any,
    };
  }

  /**
   * Merge and deduplicate sources from multiple search methods
   */
  private mergeAndDeduplicateSources(sources: SearchSource[]): SearchSource[] {
    const seen = new Set<string>();
    return sources.filter(source => {
      const fingerprint = source.fact
        .slice(0, 100)
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();
      if (seen.has(fingerprint)) return false;
      seen.add(fingerprint);
      return true;
    });
  }

  /**
   * Check if search results are relevant
   */
  hasRelevantResults(result: SearchResult, minConfidence = 0.5): boolean {
    return result.sources.some(source => source.confidence >= minConfidence);
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

export const searchService = new SearchService();
