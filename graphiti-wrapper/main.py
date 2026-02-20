"""
Minimal FastAPI wrapper for graphiti-core
Exposes native graphiti-core API for Next.js client
"""
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.graphiti import AddBulkEpisodeResults
from graphiti_core.prompts.models import Message

from embedder_factory import EmbedderSettings, create_embedder
from sentiment_similarity import SentimentSimilarityAnalyzer
from llm_client_wrapper import create_retrying_llm_client
from ollama_llm_client import create_ollama_llm_client
from query_utils import (
    classify_query,
    expand_query,
    get_best_variant,
    should_expand_query,
    detect_temporal_intent,
    query_decomposer,
    SubQuery,
    DecomposedQuery,
)
from reranker import (
    HeuristicReranker,
    CrossEncoderReranker,
    create_reranker,
    deduplicate_by_content,
)
from fallback_service import fallback_service, FallbackResult
from graphrag_service import graphrag_service, EnhancedPrompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = None
    model_name: str | None = None
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Embedder configuration (optional, defaults to OpenAI)
    embedder_provider: str = "openai"
    embedder_model: str | None = None
    embedder_api_key: str | None = None
    embedder_base_url: str | None = None
    embedder_device: str = "cpu"
    embedder_batch_size: int = 32
    embedder_embedding_dim: int | None = None

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / '.env',
        extra='ignore',
    )


def get_settings():
    return Settings()


# ============================================================================
# DTOs (Data Transfer Objects)
# ============================================================================

class EpisodeRequest(BaseModel):
    name: str
    episode_body: str
    source_description: str
    reference_time: str  # ISO 8601
    group_id: str
    is_historical: bool = False
    data_source_type: str = "current"  # current, historical, training, archived
    ingestion_date: str | None = None  # ISO 8601, defaults to now


# --- Zep-compatible DTOs (used by Atlas RAGClient / MemoryClient / NightlySync) ---

class MessageItem(BaseModel):
    """A single message in a Zep-compatible POST /messages request."""
    content: str
    role_type: str = "user"  # user, assistant, system
    role: str | None = None  # speaker name
    source_description: str | None = None
    timestamp: str | None = None  # ISO 8601


class MessagesRequest(BaseModel):
    """POST /messages request -- batch of messages to ingest as episodes."""
    group_id: str = "atlas-conversations"
    messages: list[MessageItem]


class MessagesResponse(BaseModel):
    """POST /messages response."""
    success: bool
    episodes_created: int = 0
    episode_ids: list[str] = []
    message: str = ""


class ZepFact(BaseModel):
    """A single fact in the Zep search response format."""
    uuid: str = ""
    name: str = ""
    fact: str = ""
    valid_at: str | None = None
    invalid_at: str | None = None
    created_at: str | None = None
    expired_at: str | None = None


class ZepSearchRequest(BaseModel):
    """POST /search request -- Zep-compatible search."""
    query: str
    group_ids: list[str] | None = None
    max_facts: int = 5
    search_methods: list[str] = ["cosine_similarity"]
    reranker: str = "cross_encoder"


class ZepSearchResponse(BaseModel):
    """POST /search response -- Zep-compatible fact list."""
    facts: list[ZepFact] = []


class EpisodeResponse(BaseModel):
    episode_id: str
    entities_created: int = 0
    relations_created: int = 0


class SearchResult(BaseModel):
    edges: list[dict]
    nodes: list[dict] = []


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


class BulkEpisodeRequest(BaseModel):
    episodes: list[EpisodeRequest]
    group_id: str
    is_historical: bool = False
    data_source_type: str = "current"  # current, historical, training, archived


class BulkEpisodeResponse(BaseModel):
    total_episodes: int
    successful_episodes: int
    episode_ids: list[str]


class TemporalIntent(BaseModel):
    is_historical: bool | None = None
    data_source_type: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class TraversalRequest(BaseModel):
    start_entity_name: str
    relation_types: list[str] = []
    max_hops: int = 3
    direction: str = "both"  # outgoing, incoming, both
    group_id: str


class PathStep(BaseModel):
    entity: dict
    relation: dict | None = None


class GraphPath(BaseModel):
    steps: list[PathStep]
    length: int


class TraversalResponse(BaseModel):
    paths: list[GraphPath]


class ShortestPathRequest(BaseModel):
    start_entity: str
    end_entity: str
    group_id: str
    max_hops: int = 5


class ShortestPathResponse(BaseModel):
    path: GraphPath | None = None
    found: bool = False


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze for sentiment")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str  # positive, neutral, negative
    score: float  # -1 to +1
    confidence: float  # 0 to 1
    method: str  # semantic_similarity


# ============================================================================
# Graphiti Dependency
# ============================================================================

from fastapi import Header

async def get_graphiti(
    settings: Annotated[Settings, Depends(get_settings)],
    x_embedder_provider: str | None = Header(None, alias="X-Embedder-Provider"),
    x_embedder_base_url: str | None = Header(None, alias="X-Embedder-Base-Url"),
    x_embedder_model: str | None = Header(None, alias="X-Embedder-Model"),
    x_embedder_api_key: str | None = Header(None, alias="X-Embedder-Api-Key"),
):
    """Create and yield Graphiti client with optional per-request embedder config"""

    # Use per-request config if provided, otherwise fall back to settings
    provider = x_embedder_provider or settings.embedder_provider
    base_url = x_embedder_base_url or settings.embedder_base_url or settings.openai_base_url
    model = x_embedder_model or settings.embedder_model
    api_key = x_embedder_api_key or settings.embedder_api_key or settings.openai_api_key

    # Map 'runpod' provider to 'openai' since RunPod uses OpenAI-compatible API
    effective_provider = "openai" if provider == "runpod" else provider

    # Create custom embedder based on settings
    embedder_config = EmbedderSettings(
        provider=effective_provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        device=settings.embedder_device,
        batch_size=settings.embedder_batch_size,
        embedding_dim=settings.embedder_embedding_dim,
    )

    logger.info(
        "Creating embedder: provider=%s (effective=%s) model=%s base_url=%s",
        provider,
        effective_provider,
        embedder_config.model,
        base_url[:50] if base_url else None,
    )

    try:
        embedder = create_embedder(embedder_config)
    except Exception as e:
        logger.error("Failed to create embedder: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedder creation failed: {e}")

    # Create LLM client - use Ollama client if base_url points to Ollama
    is_ollama = settings.openai_base_url and ("11434" in settings.openai_base_url or "ollama" in settings.openai_base_url.lower())

    if is_ollama:
        llm_client = create_ollama_llm_client(
            model=settings.model_name or "hermes3:8b-q4",
            base_url=settings.openai_base_url,
        )
        logger.info("Created Ollama LLM client: model=%s base_url=%s", settings.model_name, settings.openai_base_url)
    else:
        llm_client = create_retrying_llm_client(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.model_name,
            max_retries=8,
            base_delay=5.0,
            max_delay=60.0,
        )
        logger.info("Created OpenAI LLM client with retry logic (max_retries=8, base_delay=5s)")

    client = Graphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        embedder=embedder,
        llm_client=llm_client,
    )

    try:
        yield client
    finally:
        await client.close()


GraphitiDep = Annotated[Graphiti, Depends(get_graphiti)]


# ============================================================================
# Sentiment Analyzer Dependency
# ============================================================================

sentiment_analyzer = None


async def get_sentiment_analyzer(settings: Annotated[Settings, Depends(get_settings)]):
    """Get or create sentiment analyzer singleton"""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        logger.info("Creating sentiment analyzer...")
        embedder_config = EmbedderSettings(
            provider=settings.embedder_provider,
            model=settings.embedder_model,
            api_key=settings.embedder_api_key or settings.openai_api_key,
            base_url=settings.embedder_base_url or settings.openai_base_url,
            device=settings.embedder_device,
            batch_size=settings.embedder_batch_size,
            embedding_dim=settings.embedder_embedding_dim,
        )
        embedder = create_embedder(embedder_config)
        sentiment_analyzer = SentimentSimilarityAnalyzer(embedder)
        await sentiment_analyzer.initialize()
        logger.info("Sentiment analyzer initialized successfully")
    return sentiment_analyzer


# ============================================================================
# Helper Functions
# ============================================================================

async def get_episode_temporal_metadata(driver, episode_uuid: str) -> dict | None:
    """Get temporal metadata from episode node"""
    try:
        result = await driver.execute_query(
            """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN e.is_historical as is_historical,
                   e.data_source_type as data_source_type,
                   e.ingestion_date as ingestion_date
            """,
            uuid=episode_uuid
        )
        if result.records and len(result.records) > 0:
            record = result.records[0]
            return {
                'is_historical': record['is_historical'],
                'data_source_type': record['data_source_type'],
                'ingestion_date': record['ingestion_date'],
            }
        return None
    except Exception as e:
        logger.warning(f"Failed to get temporal metadata for episode {episode_uuid}: {e}")
        return None


async def add_episode_metadata(
    driver,
    episode_uuid: str,
    is_historical: bool,
    data_source_type: str,
    ingestion_date: datetime,
) -> None:
    """Add custom temporal metadata to episode node in Neo4j"""
    try:
        logger.info(
            "Adding metadata to episode %s: is_historical=%s, type=%s, date=%s",
            episode_uuid,
            is_historical,
            data_source_type,
            ingestion_date,
        )

        query = """
        MATCH (e:Episodic {uuid: $uuid})
        SET e.is_historical = $is_historical,
            e.data_source_type = $data_source_type,
            e.ingestion_date = datetime($ingestion_date)
        RETURN e.uuid as uuid
        """
        result = await driver.execute_query(
            query,
            uuid=episode_uuid,
            is_historical=is_historical,
            data_source_type=data_source_type,
            ingestion_date=ingestion_date.isoformat(),
        )

        logger.info("Metadata added successfully for episode %s", episode_uuid)
    except Exception as e:
        logger.error("Failed to add metadata for episode %s: %s", episode_uuid, e)


async def analyze_query_temporal_intent(llm_client, query: str) -> dict:
    """
    Analyze search query for temporal intent using LLM
    Returns dict with detected filters: is_historical, data_source_type, date_from, date_to
    """
    try:
        prompt_text = f"""Analyze this search query for temporal intent: "{query}"

Determine if the user is looking for:
- Historical data (old/past/archived data)
- Current data (recent/today/latest data)
- Specific time period
- Specific data source type (current, historical, training, archived)

Examples:
Query: "show me historical training data"
Result: is_historical=true, data_source_type="historical"

Query: "recent updates"
Result: is_historical=false, data_source_type="current"

Query: "training data"
Result: is_historical=null, data_source_type=null"""

        logger.info("Analyzing query for temporal intent: %s", query[:100])

        messages = [Message(role="user", content=prompt_text)]

        llm_response = await llm_client.generate_response(
            messages,
            response_model=TemporalIntent,
        )

        logger.info("LLM temporal analysis response: %s", llm_response)

        result = {
            'is_historical': llm_response.get('is_historical'),
            'data_source_type': llm_response.get('data_source_type'),
            'date_from': llm_response.get('date_from'),
            'date_to': llm_response.get('date_to'),
        }

        logger.info("Detected temporal filters: %s", result)
        return result

    except Exception as e:
        logger.warning("Failed to analyze temporal intent: %s", e)
        return {
            'is_historical': None,
            'data_source_type': None,
            'date_from': None,
            'date_to': None,
        }


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Graphiti Wrapper API", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Preload the embedder model at startup to avoid meta tensor issues"""
    logger.info("Preloading embedder model at startup...")
    try:
        settings = Settings()
        embedder_config = EmbedderSettings(
            provider=settings.embedder_provider,
            model=settings.embedder_model,
            api_key=settings.embedder_api_key or settings.openai_api_key,
            base_url=settings.embedder_base_url or settings.openai_base_url,
            device=settings.embedder_device,
            batch_size=settings.embedder_batch_size,
            embedding_dim=settings.embedder_embedding_dim,
        )
        embedder = create_embedder(embedder_config)
        # Force model loading by running a test embedding
        test_result = await embedder.create("startup test")
        logger.info(f"Embedder preloaded successfully, test embedding dim: {len(test_result)}")
    except Exception as e:
        logger.error(f"Failed to preload embedder: {e}")
        # Don't fail startup, let it fail on first request instead


@app.get('/health')
async def health() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(status="healthy")


@app.get('/healthcheck')
async def healthcheck() -> HealthResponse:
    """Health check alias (Zep-compatible, used by Atlas clients)"""
    return HealthResponse(status="healthy")


@app.post('/messages')
async def post_messages(
    request: MessagesRequest,
    graphiti: GraphitiDep,
) -> MessagesResponse:
    """
    Zep-compatible message ingestion endpoint.

    Used by:
    - NightlyMemorySync (batch conversation sync)
    - RAGClient.add_messages() (real-time, if enabled)
    - MemoryClient.send_messages()
    - seed_business_data.py

    Each message becomes a graphiti episode with entity/relationship extraction.
    Individual message failures are logged but don't fail the batch.
    """
    logger.info(
        "POST /messages: group_id=%s, messages=%d",
        request.group_id,
        len(request.messages),
    )

    episode_ids: list[str] = []
    errors = 0

    for idx, msg in enumerate(request.messages):
        try:
            # Build episode body: "{role_type}: {content}"
            body = f"{msg.role_type}: {msg.content}" if msg.role_type else msg.content

            # Parse timestamp or use now
            if msg.timestamp:
                ref_time = datetime.fromisoformat(
                    msg.timestamp.replace("Z", "+00:00")
                )
            else:
                ref_time = datetime.now(timezone.utc)

            # Source description: use provided, or build from role info
            source_desc = (
                msg.source_description
                or f"atlas-{msg.role_type}"
            )

            result = await graphiti.add_episode(
                name=f"msg-{request.group_id}-{idx}",
                episode_body=body,
                source_description=source_desc,
                reference_time=ref_time,
                group_id=request.group_id,
                source=EpisodeType.message,
            )

            episode_uuid = (
                result.episode.uuid
                if hasattr(result, "episode")
                else str(result)
            )
            episode_ids.append(episode_uuid)

        except Exception as e:
            errors += 1
            logger.error(
                "Failed to ingest message %d/%d: %s",
                idx + 1,
                len(request.messages),
                e,
            )

    logger.info(
        "POST /messages complete: %d/%d episodes created (%d errors)",
        len(episode_ids),
        len(request.messages),
        errors,
    )

    return MessagesResponse(
        success=len(episode_ids) > 0 or len(request.messages) == 0,
        episodes_created=len(episode_ids),
        episode_ids=episode_ids,
        message=f"{len(episode_ids)} episodes created"
        + (f", {errors} errors" if errors else ""),
    )


@app.post('/search')
async def post_search(
    request: ZepSearchRequest,
    graphiti: GraphitiDep,
) -> ZepSearchResponse:
    """
    Zep-compatible search endpoint (POST).

    Used by:
    - RAGClient.search() (reranker="cross_encoder")
    - MemoryClient.search() (reranker="rrf")

    Coexists with the existing GET /search (different HTTP method).
    """
    logger.info(
        "POST /search: query=%s group_ids=%s max_facts=%d reranker=%s",
        request.query[:80],
        request.group_ids,
        request.max_facts,
        request.reranker,
    )

    try:
        group_ids = request.group_ids or []

        # Fetch more candidates than needed for reranking
        fetch_limit = request.max_facts * 3

        edges = await graphiti.search(
            query=request.query,
            group_ids=group_ids,
            num_results=fetch_limit,
        )

        logger.info("Raw search returned %d edges", len(edges))

        # Convert edges to fact dicts for reranking
        edge_dicts = []
        for edge in edges:
            edge_dicts.append({
                "uuid": edge.uuid,
                "name": edge.name or "",
                "fact": edge.fact or "",
                "valid_at": None,
                "invalid_at": None,
                "created_at": edge.created_at.isoformat() if edge.created_at else None,
                "expired_at": edge.expired_at.isoformat() if edge.expired_at else None,
                "score": edge.score if hasattr(edge, "score") and edge.score else 0.5,
            })

        # Apply reranker
        reranker_type = request.reranker.lower().replace("_", "-")
        if reranker_type in ("cross-encoder", "cross_encoder", "crossencoder"):
            reranker = create_reranker("cross-encoder", top_k=request.max_facts)
        elif reranker_type == "rrf":
            # RRF (Reciprocal Rank Fusion) -- use heuristic reranker as proxy
            reranker = create_reranker("heuristic", top_k=request.max_facts)
        else:
            reranker = None

        if reranker and edge_dicts:
            candidates = [
                {"text": e["fact"], "score": e["score"], "metadata": e}
                for e in edge_dicts
            ]
            reranked = reranker.rerank(request.query, candidates)
            edge_dicts = [r.metadata for r in reranked]
            logger.info("After %s reranking: %d facts", reranker_type, len(edge_dicts))
        else:
            edge_dicts = edge_dicts[: request.max_facts]

        # Convert to Zep fact format
        facts = [
            ZepFact(
                uuid=e.get("uuid", ""),
                name=e.get("name", ""),
                fact=e.get("fact", ""),
                valid_at=e.get("valid_at"),
                invalid_at=e.get("invalid_at"),
                created_at=e.get("created_at"),
                expired_at=e.get("expired_at"),
            )
            for e in edge_dicts
        ]

        logger.info("POST /search returning %d facts", len(facts))
        return ZepSearchResponse(facts=facts)

    except Exception as e:
        logger.error("POST /search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/embedder/info')
async def embedder_info(settings: Annotated[Settings, Depends(get_settings)]) -> dict:
    """Get current embedder configuration"""
    return {
        "provider": settings.embedder_provider,
        "model": settings.embedder_model or "default",
        "device": settings.embedder_device,
        "batch_size": settings.embedder_batch_size,
        "embedding_dim": settings.embedder_embedding_dim or "default",
    }


@app.post('/episodes')
async def add_episode(
    request: EpisodeRequest,
    graphiti: GraphitiDep
) -> EpisodeResponse:
    """Add an episode to the knowledge graph"""
    try:
        # Parse ISO 8601 timestamp
        reference_time = datetime.fromisoformat(request.reference_time.replace('Z', '+00:00'))

        # Add episode using graphiti-core's native API
        result = await graphiti.add_episode(
            name=request.name,
            episode_body=request.episode_body,
            source_description=request.source_description,
            reference_time=reference_time,
            group_id=request.group_id,
            source=EpisodeType.text,  # Default to text type
        )

        episode_uuid = result.episode.uuid if hasattr(result, 'episode') else str(result)
        logger.info(f"Added episode: {request.name} for group {request.group_id}")

        # Add custom temporal metadata
        ingestion_date = datetime.fromisoformat(request.ingestion_date.replace('Z', '+00:00')) \
            if request.ingestion_date else datetime.now()

        await add_episode_metadata(
            graphiti.driver,
            episode_uuid,
            request.is_historical,
            request.data_source_type,
            ingestion_date,
        )

        # Return response matching client expectations
        return EpisodeResponse(
            episode_id=episode_uuid,
            entities_created=0,  # graphiti-core doesn't return this
            relations_created=0,  # graphiti-core doesn't return this
        )
    except Exception as e:
        logger.error(f"Error adding episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/episodes/bulk')
async def add_episodes_bulk(
    request: BulkEpisodeRequest,
    graphiti: GraphitiDep
) -> BulkEpisodeResponse:
    """Add multiple episodes to the knowledge graph in bulk"""
    try:
        logger.info(f"Processing bulk upload: {len(request.episodes)} episodes for group {request.group_id}")

        # Convert EpisodeRequest objects to RawEpisode objects
        raw_episodes = []
        for ep in request.episodes:
            reference_time = datetime.fromisoformat(ep.reference_time.replace('Z', '+00:00'))
            raw_ep = RawEpisode(
                name=ep.name,
                content=ep.episode_body,
                source_description=ep.source_description,
                reference_time=reference_time,
                source=EpisodeType.text,
            )
            raw_episodes.append(raw_ep)

        logger.info(f"Converted {len(raw_episodes)} episodes to RawEpisode format")

        # Call graphiti's bulk add method
        result = await graphiti.add_episode_bulk(
            bulk_episodes=raw_episodes,
            group_id=request.group_id,
        )

        # Extract episode IDs from result (handle async/sync iterables)
        episode_ids = []
        if hasattr(result, 'episodes') and result.episodes:
            episodes = result.episodes
            # If episodes is a coroutine, await it
            if asyncio.iscoroutine(episodes):
                episodes = await episodes
            # If episodes is an async generator, collect items
            if hasattr(episodes, '__aiter__'):
                async for ep in episodes:
                    episode_ids.append(ep.uuid)
            else:
                # Regular iterable
                for ep in episodes:
                    episode_ids.append(ep.uuid)

        logger.info(f"Bulk upload complete: {len(episode_ids)} episodes added for group {request.group_id}")

        # Add custom temporal metadata to all episodes
        ingestion_date = datetime.now()
        for episode_uuid in episode_ids:
            await add_episode_metadata(
                graphiti.driver,
                episode_uuid,
                request.is_historical,
                request.data_source_type,
                ingestion_date,
            )

        logger.info(f"Added metadata to {len(episode_ids)} episodes: is_historical={request.is_historical}")

        return BulkEpisodeResponse(
            total_episodes=len(request.episodes),
            successful_episodes=len(episode_ids),
            episode_ids=episode_ids,
        )
    except Exception as e:
        logger.error(f"Error in bulk episode upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/search')
async def search(
    query: str,
    group_ids: str,  # Comma-separated
    graphiti: GraphitiDep,
    num_results: int = 10,
    is_historical: bool | None = None,
    data_source_type: str | None = None,
    date_from: str | None = None,  # ISO 8601
    date_to: str | None = None,  # ISO 8601
) -> SearchResult:
    """Search the knowledge graph with optional temporal filtering"""
    try:
        # Parse comma-separated group_ids
        group_id_list = [gid.strip() for gid in group_ids.split(',') if gid.strip()]

        # Auto-detect temporal intent if no explicit filters provided
        if all(x is None for x in [is_historical, data_source_type, date_from, date_to]):
            logger.info("No explicit filters provided, analyzing query for temporal intent")
            detected_filters = await analyze_query_temporal_intent(graphiti.llm_client, query)

            is_historical = detected_filters.get('is_historical')
            data_source_type = detected_filters.get('data_source_type')
            date_from = detected_filters.get('date_from')
            date_to = detected_filters.get('date_to')

            logger.info(
                "Auto-detected filters: is_historical=%s, type=%s, date_range=%s to %s",
                is_historical, data_source_type, date_from, date_to
            )

        # Search using graphiti-core's native API (get more results for filtering)
        search_limit = num_results * 3 if any([is_historical is not None, data_source_type, date_from, date_to]) else num_results
        edges = await graphiti.search(
            query=query,
            group_ids=group_id_list,
            num_results=search_limit,
        )

        logger.info(
            f"Search query: {query} | Found {len(edges)} edges | Filters: is_historical={is_historical}, "
            f"data_source_type={data_source_type}, date_range={date_from} to {date_to}"
        )

        # Convert EntityEdge objects to dicts and apply temporal filters
        edge_dicts = []
        for idx, edge in enumerate(edges):
            edge_dict = {
                'uuid': edge.uuid,
                'name': edge.name,
                'fact': edge.fact,
                'created_at': edge.created_at.isoformat() if edge.created_at else None,
                'expired_at': edge.expired_at.isoformat() if edge.expired_at else None,
            }

            # Add source and target nodes
            if hasattr(edge, 'source_node') and edge.source_node:
                edge_dict['source_node'] = {
                    'name': edge.source_node.name if hasattr(edge.source_node, 'name') else '',
                    'uuid': edge.source_node.uuid if hasattr(edge.source_node, 'uuid') else '',
                }

            if hasattr(edge, 'target_node') and edge.target_node:
                edge_dict['target_node'] = {
                    'name': edge.target_node.name if hasattr(edge.target_node, 'name') else '',
                    'uuid': edge.target_node.uuid if hasattr(edge.target_node, 'uuid') else '',
                }

            # Get metadata from episode and apply temporal filters
            should_include = True
            if hasattr(edge, 'episodes') and edge.episodes:
                try:
                    episode_uuid = edge.episodes[0] if isinstance(edge.episodes, list) else edge.episodes
                    result = await graphiti.driver.execute_query(
                        """MATCH (e:Episodic {uuid: $uuid})
                        RETURN e.source_description as source_description,
                               e.is_historical as is_historical,
                               e.data_source_type as data_source_type,
                               e.ingestion_date as ingestion_date""",
                        uuid=str(episode_uuid)
                    )
                    if result.records:
                        record = result.records[0]
                        if record['source_description']:
                            edge_dict['source_description'] = record['source_description']

                        # Apply temporal filters
                        if is_historical is not None and record['is_historical'] != is_historical:
                            should_include = False
                        if data_source_type and record['data_source_type'] != data_source_type:
                            should_include = False
                        if date_from and record['ingestion_date']:
                            ingestion_dt = record['ingestion_date']
                            # Convert Neo4j DateTime to Python datetime if needed
                            if hasattr(ingestion_dt, 'to_native'):
                                ingestion_dt = ingestion_dt.to_native()
                            # Ensure timezone-aware for comparison
                            if ingestion_dt.tzinfo is None:
                                ingestion_dt = ingestion_dt.replace(tzinfo=timezone.utc)
                            date_from_dt = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
                            # Ensure date_from_dt is also timezone-aware (LLM may return just date like '2025-12-12')
                            if date_from_dt.tzinfo is None:
                                date_from_dt = date_from_dt.replace(tzinfo=timezone.utc)
                            if ingestion_dt < date_from_dt:
                                should_include = False
                        if date_to and record['ingestion_date']:
                            ingestion_dt = record['ingestion_date']
                            # Convert Neo4j DateTime to Python datetime if needed
                            if hasattr(ingestion_dt, 'to_native'):
                                ingestion_dt = ingestion_dt.to_native()
                            # Ensure timezone-aware for comparison
                            if ingestion_dt.tzinfo is None:
                                ingestion_dt = ingestion_dt.replace(tzinfo=timezone.utc)
                            date_to_dt = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                            # Ensure date_to_dt is also timezone-aware (LLM may return just date like '2025-12-31')
                            if date_to_dt.tzinfo is None:
                                date_to_dt = date_to_dt.replace(tzinfo=timezone.utc)
                            if ingestion_dt > date_to_dt:
                                should_include = False
                except Exception as e:
                    logger.warning(f"Could not get metadata for edge {edge.uuid}: {e}")

            # Only include if passes temporal filters
            if not should_include:
                continue

            # Add score - use native score if available, otherwise generate synthetic score
            # Synthetic scores: rank-based, decreasing from 1.0 to 0.1
            if hasattr(edge, 'score') and edge.score is not None:
                edge_dict['score'] = edge.score
            else:
                # Generate synthetic relevance score based on position in results
                # First result = 1.0, last result approaches 0.1
                # Formula: 1.0 - (idx / total_results) * 0.9
                total_results = len(edges)
                synthetic_score = max(0.1, 1.0 - (idx / max(1, total_results)) * 0.9)
                edge_dict['score'] = synthetic_score

            edge_dicts.append(edge_dict)

            # Limit results after filtering
            if len(edge_dicts) >= num_results:
                break

        logger.info(f"After filtering: {len(edge_dicts)} edges returned")
        return SearchResult(edges=edge_dicts, nodes=[])
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class EnhancedSearchRequest(BaseModel):
    """Request model for enhanced search."""
    query: str
    group_ids: str
    num_results: int = 10
    threshold: float = 0.3
    enable_expansion: bool = True
    enable_reranking: bool = True
    reranker_type: str = "cross-encoder"
    enable_deduplication: bool = True
    enable_decomposition: bool = True
    enable_fallback: bool = True


class EnhancedSearchResult(BaseModel):
    """Result model for enhanced search."""
    edges: list[dict]
    nodes: list[dict] = []
    metadata: dict = {}


@app.post('/search/enhanced')
async def enhanced_search(
    request: EnhancedSearchRequest,
    graphiti: GraphitiDep
) -> EnhancedSearchResult:
    """
    Enhanced search with query processing pipeline.

    Pipeline: classify -> decompose -> expand -> search -> rerank -> dedupe
    """
    try:
        start_time = datetime.now()
        original_query = request.query

        classification = classify_query(original_query)
        if classification.should_skip_search:
            logger.info(
                "Skipping search for tool-specific query: %s",
                classification.reason
            )
            return EnhancedSearchResult(
                edges=[],
                nodes=[],
                metadata={
                    "skipped": True,
                    "reason": classification.reason,
                    "detected_pattern": classification.detected_pattern,
                },
            )

        decomposition_info = None
        sub_queries_to_search = [original_query]

        if request.enable_decomposition and query_decomposer.should_decompose(original_query):
            decomposed = query_decomposer.decompose(original_query)
            if decomposed.is_complex and len(decomposed.sub_queries) > 1:
                sub_queries_to_search = [sq.query for sq in decomposed.sub_queries]
                decomposition_info = {
                    "is_complex": True,
                    "reason": decomposed.complexity_reason,
                    "sub_query_count": len(decomposed.sub_queries),
                    "sub_queries": [
                        {"query": sq.query, "priority": sq.priority, "type": sq.query_type}
                        for sq in decomposed.sub_queries
                    ],
                }
                logger.info(
                    "Query decomposed: %s -> %d sub-queries (%s)",
                    original_query[:50],
                    len(decomposed.sub_queries),
                    decomposed.complexity_reason
                )

        expansion_info = None
        final_search_queries = []
        for sq in sub_queries_to_search:
            search_query = sq
            if request.enable_expansion and should_expand_query(sq):
                expanded = expand_query(sq)
                search_query = get_best_variant(expanded)
                if expansion_info is None:
                    expansion_info = {
                        "original": original_query,
                        "expanded": search_query,
                        "variants": expanded.variants,
                        "transformations": expanded.transformations_applied,
                    }
                logger.info("Query expanded: %s -> %s", sq[:30], search_query[:30])
            final_search_queries.append(search_query)

        temporal_intent = detect_temporal_intent(original_query)

        group_id_list = [g.strip() for g in request.group_ids.split(",") if g.strip()]

        per_query_limit = request.num_results * 3 if request.enable_reranking else request.num_results
        if len(final_search_queries) > 1:
            per_query_limit = max(5, per_query_limit // len(final_search_queries))

        all_edges = []
        for search_query in final_search_queries:
            edges = await graphiti.search(
                query=search_query,
                group_ids=group_id_list,
                num_results=per_query_limit,
            )
            all_edges.extend(edges)

        edges = all_edges
        logger.info("Raw search returned %d edges from %d queries", len(edges), len(final_search_queries))

        edge_dicts = []
        for idx, edge in enumerate(edges):
            edge_dict = {
                "uuid": edge.uuid,
                "name": edge.name,
                "fact": edge.fact,
                "score": edge.score if hasattr(edge, "score") and edge.score else 0.5,
                "created_at": edge.created_at.isoformat() if edge.created_at else None,
            }
            if hasattr(edge, "source_node") and edge.source_node:
                edge_dict["source_node"] = {
                    "name": getattr(edge.source_node, "name", ""),
                    "uuid": getattr(edge.source_node, "uuid", ""),
                }
            if hasattr(edge, "target_node") and edge.target_node:
                edge_dict["target_node"] = {
                    "name": getattr(edge.target_node, "name", ""),
                    "uuid": getattr(edge.target_node, "uuid", ""),
                }
            edge_dicts.append(edge_dict)

        filtered = [e for e in edge_dicts if e.get("score", 0) >= request.threshold]
        logger.info("After threshold %.2f: %d edges", request.threshold, len(filtered))

        fallback_info = None
        if request.enable_fallback and fallback_service.should_trigger_fallback(len(filtered)):
            logger.info("Triggering fallback search (only %d results)", len(filtered))
            group_id = group_id_list[0] if group_id_list else None
            fallback_result = await fallback_service.execute_fallback(
                query=original_query,
                group_id=group_id,
                limit=request.num_results,
            )
            if fallback_result.sources:
                for source in fallback_result.sources:
                    filtered.append({
                        "uuid": "",
                        "name": source.entity,
                        "fact": source.fact,
                        "score": source.confidence,
                        "source_description": source.source_description,
                        "is_fallback": True,
                    })
                fallback_info = {
                    "triggered": True,
                    "strategy": fallback_result.strategy,
                    "sources_added": len(fallback_result.sources),
                    "query_time_ms": fallback_result.query_time_ms,
                }
                logger.info("Fallback added %d sources", len(fallback_result.sources))

        reranker_used = None
        if request.enable_reranking and len(filtered) > 0:
            reranker = create_reranker(
                reranker_type=request.reranker_type,
                top_k=request.num_results,
            )
            if reranker is not None:
                candidates = [
                    {"text": e["fact"], "score": e["score"], "metadata": e}
                    for e in filtered
                ]
                reranked = reranker.rerank(original_query, candidates)
                reranker_used = request.reranker_type
                filtered = [r.metadata for r in reranked]
                logger.info("After %s reranking: %d edges", reranker_used, len(filtered))

        if request.enable_deduplication:
            before_count = len(filtered)
            filtered = deduplicate_by_content(filtered, content_key="fact")
            logger.info("After deduplication: %d -> %d", before_count, len(filtered))

        final_edges = filtered[:request.num_results]

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return EnhancedSearchResult(
            edges=final_edges,
            nodes=[],
            metadata={
                "query_time_ms": elapsed_ms,
                "original_query": original_query,
                "search_queries": final_search_queries,
                "decomposition": decomposition_info,
                "expansion": expansion_info,
                "temporal_intent": {
                    "is_historical": temporal_intent.is_historical,
                    "date_from": temporal_intent.date_from,
                    "date_to": temporal_intent.date_to,
                    "requires_latest": temporal_intent.requires_latest,
                },
                "results_count": len(final_edges),
                "raw_results_count": len(edges),
                "threshold": request.threshold,
                "reranking_enabled": request.enable_reranking,
                "reranker_type": reranker_used or request.reranker_type,
                "deduplication_enabled": request.enable_deduplication,
                "decomposition_enabled": request.enable_decomposition,
                "fallback_enabled": request.enable_fallback,
                "fallback": fallback_info,
            },
        )

    except Exception as e:
        logger.error("Enhanced search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class EnhancePromptRequest(BaseModel):
    """Request model for prompt enhancement."""
    query: str
    group_ids: str
    min_confidence: float = 0.3
    max_sources: int = 5
    max_context_length: int = 4000
    compress_context: bool = False
    include_metadata: bool = True


class EnhancePromptResponse(BaseModel):
    """Response model for prompt enhancement."""
    prompt: str
    context_used: bool
    sources: list = []
    metadata: dict = {}


@app.post('/enhance')
async def enhance_prompt(
    request: EnhancePromptRequest,
    graphiti: GraphitiDep
) -> EnhancePromptResponse:
    """
    Enhance a user prompt with document context.

    Performs search and injects relevant context into the prompt.
    """
    try:
        start_time = datetime.now()

        group_id_list = [g.strip() for g in request.group_ids.split(",") if g.strip()]

        edges = await graphiti.search(
            query=request.query,
            group_ids=group_id_list,
            num_results=request.max_sources * 2,
        )

        sources = []
        for edge in edges:
            sources.append({
                "entity": getattr(edge.source_node, "name", "") if hasattr(edge, "source_node") else "",
                "relation": edge.name if hasattr(edge, "name") else "",
                "fact": edge.fact if hasattr(edge, "fact") else "",
                "confidence": edge.score if hasattr(edge, "score") and edge.score else 0.5,
                "source_description": "",
            })

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        result = await graphrag_service.enhance_prompt(
            user_message=request.query,
            sources=sources,
            retrieval_time_ms=elapsed_ms,
            min_confidence=request.min_confidence,
            max_sources=request.max_sources,
            max_context_length=request.max_context_length,
            compress_context=request.compress_context,
            include_metadata=request.include_metadata,
        )

        return EnhancePromptResponse(
            prompt=result.prompt,
            context_used=result.context_used,
            sources=result.sources,
            metadata=result.metadata,
        )

    except Exception as e:
        logger.error("Enhance prompt error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/episodes/{episode_id}')
async def delete_episode(
    episode_id: str,
    graphiti: GraphitiDep
):
    """Delete an episode from the knowledge graph"""
    try:
        from graphiti_core.nodes import EpisodicNode

        # Get and delete the episodic node
        episode = await EpisodicNode.get_by_uuid(graphiti.driver, episode_id)
        await episode.delete(graphiti.driver)

        logger.info(f"Deleted episode: {episode_id}")
        return {"message": "Episode deleted", "success": True}
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    except Exception as e:
        logger.error(f"Error deleting episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/entities/{entity_name}/edges')
async def get_entity_edges(
    entity_name: str,
    group_ids: str,  # Comma-separated
    graphiti: GraphitiDep
) -> SearchResult:
    """Get edges for a specific entity"""
    try:
        # Parse comma-separated group_ids
        group_id_list = [gid.strip() for gid in group_ids.split(',') if gid.strip()]

        # Search for the entity by name
        edges = await graphiti.search(
            query=entity_name,
            group_ids=group_id_list,
            num_results=50,  # Get more results for entity-specific queries
        )

        # Filter edges that mention the entity in fact text or node names
        entity_lower = entity_name.lower()
        filtered_edges = []
        for edge in edges:
            if entity_lower in edge.fact.lower():
                filtered_edges.append(edge)
            elif hasattr(edge, 'source_node') and edge.source_node and entity_lower in edge.source_node.name.lower():
                filtered_edges.append(edge)
            elif hasattr(edge, 'target_node') and edge.target_node and entity_lower in edge.target_node.name.lower():
                filtered_edges.append(edge)

        logger.info(f"Entity edges for {entity_name}: Found {len(filtered_edges)} edges")

        # Convert to dicts (same as search endpoint)
        edge_dicts = []
        for idx, edge in enumerate(filtered_edges):
            edge_dict = {
                'uuid': edge.uuid,
                'name': edge.name,
                'fact': edge.fact,
                'created_at': edge.created_at.isoformat() if edge.created_at else None,
                'expired_at': edge.expired_at.isoformat() if edge.expired_at else None,
            }

            # Add source and target nodes
            if hasattr(edge, 'source_node') and edge.source_node:
                edge_dict['source_node'] = {
                    'name': edge.source_node.name if hasattr(edge.source_node, 'name') else '',
                    'uuid': edge.source_node.uuid if hasattr(edge.source_node, 'uuid') else '',
                }

            if hasattr(edge, 'target_node') and edge.target_node:
                edge_dict['target_node'] = {
                    'name': edge.target_node.name if hasattr(edge.target_node, 'name') else '',
                    'uuid': edge.target_node.uuid if hasattr(edge.target_node, 'uuid') else '',
                }

            # Get source_description from episode if edge has episode references
            if hasattr(edge, 'episodes') and edge.episodes:
                try:
                    episode_uuid = edge.episodes[0] if isinstance(edge.episodes, list) else edge.episodes
                    result = await graphiti.driver.execute_query(
                        "MATCH (e:Episodic {uuid: $uuid}) RETURN e.source_description as source_description",
                        uuid=str(episode_uuid)
                    )
                    if result.records and result.records[0]['source_description']:
                        edge_dict['source_description'] = result.records[0]['source_description']
                except Exception as e:
                    logger.warning(f"Could not get source_description for edge {edge.uuid}: {e}")

            # Add score - use native score if available, otherwise generate synthetic score
            if hasattr(edge, 'score') and edge.score is not None:
                edge_dict['score'] = edge.score
            else:
                # Generate synthetic relevance score based on position in results
                total_results = len(filtered_edges)
                synthetic_score = max(0.1, 1.0 - (idx / max(1, total_results)) * 0.9)
                edge_dict['score'] = synthetic_score

            edge_dicts.append(edge_dict)

        return SearchResult(edges=edge_dicts, nodes=[])
    except Exception as e:
        logger.error(f"Error getting entity edges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/traverse', response_model=TraversalResponse)
async def traverse_graph(
    request: TraversalRequest,
    graphiti: GraphitiDep
) -> TraversalResponse:
    """
    Traverse graph from starting entity following specified relation types.
    Returns paths found up to max_hops distance.
    """
    try:
        logger.info(
            "Traversal request: start=%s, max_hops=%d, direction=%s",
            request.start_entity_name,
            request.max_hops,
            request.direction
        )

        # Build direction clause for Cypher
        direction_clause = "-[r]-"
        if request.direction == "outgoing":
            direction_clause = "-[r]->"
        elif request.direction == "incoming":
            direction_clause = "<-[r]-"

        # Build relation type filter
        rel_filter = ""
        if request.relation_types:
            rel_types = "|".join(request.relation_types)
            rel_filter = f":{rel_types}"

        # Execute traversal query
        cypher = f"""
        MATCH path = (start:Entity {{name: $start_name, group_id: $group_id}})
        {direction_clause.replace('[r]', f'[r{rel_filter}*1..{request.max_hops}]')}
        (end:Entity)
        RETURN path
        LIMIT 100
        """

        result = await graphiti.driver.execute_query(
            cypher,
            start_name=request.start_entity_name,
            group_id=request.group_id
        )

        paths = []
        for record in result.records:
            path_data = record['path']
            formatted_path = format_neo4j_path(path_data)
            if formatted_path:
                paths.append(formatted_path)

        logger.info("Traversal found %d paths", len(paths))
        return TraversalResponse(paths=paths)

    except Exception as e:
        logger.error(f"Traversal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/shortest-path', response_model=ShortestPathResponse)
async def find_shortest_path(
    request: ShortestPathRequest,
    graphiti: GraphitiDep
) -> ShortestPathResponse:
    """
    Find shortest path between two entities.
    Returns the path if found, or None if no path exists.
    """
    try:
        logger.info(
            "Shortest path request: %s -> %s (max_hops=%d)",
            request.start_entity,
            request.end_entity,
            request.max_hops
        )

        cypher = """
        MATCH path = shortestPath(
            (start:Entity {name: $start_name, group_id: $group_id})
            -[*1..%d]-
            (end:Entity {name: $end_name, group_id: $group_id})
        )
        RETURN path
        """ % request.max_hops

        result = await graphiti.driver.execute_query(
            cypher,
            start_name=request.start_entity,
            end_name=request.end_entity,
            group_id=request.group_id
        )

        if not result.records:
            logger.info("No path found between %s and %s", request.start_entity, request.end_entity)
            return ShortestPathResponse(path=None, found=False)

        path_data = result.records[0]['path']
        formatted_path = format_neo4j_path(path_data)

        logger.info(
            "Found path of length %d between %s and %s",
            formatted_path.length if formatted_path else 0,
            request.start_entity,
            request.end_entity
        )

        return ShortestPathResponse(path=formatted_path, found=True)

    except Exception as e:
        logger.error(f"Shortest path error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def format_neo4j_path(path_data) -> GraphPath | None:
    """Format Neo4j path object into GraphPath response model."""
    try:
        if path_data is None:
            return None

        steps = []
        nodes = list(path_data.nodes)
        relationships = list(path_data.relationships)

        for i, node in enumerate(nodes):
            step = PathStep(
                entity={
                    'name': node.get('name', ''),
                    'uuid': node.get('uuid', ''),
                    'labels': list(node.labels) if hasattr(node, 'labels') else [],
                },
                relation=None
            )

            # Add relation info if not the last node
            if i < len(relationships):
                rel = relationships[i]
                step.relation = {
                    'type': rel.type if hasattr(rel, 'type') else '',
                    'fact': rel.get('fact', '') if hasattr(rel, 'get') else '',
                    'uuid': rel.get('uuid', '') if hasattr(rel, 'get') else '',
                }

            steps.append(step)

        return GraphPath(steps=steps, length=len(relationships))

    except Exception as e:
        logger.warning(f"Failed to format path: {e}")
        return None


@app.post('/analyze/sentiment', response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    analyzer: Annotated[SentimentSimilarityAnalyzer, Depends(get_sentiment_analyzer)]
):
    """
    Analyze sentiment using semantic similarity.
    Uses existing all-MiniLM-L6-v2 embedder with sentiment anchors.
    Returns sentiment (positive/neutral/negative), score, and confidence.
    """
    logger.info("Sentiment analysis request received for text length: %d", len(request.text))

    try:
        result = await analyzer.analyze(request.text)
        logger.info("Sentiment analysis completed: %s (score=%.4f, confidence=%.4f)",
                   result.get("sentiment"), result.get("score", 0), result.get("confidence", 0))
        return result
    except Exception as e:
        logger.error("Error in sentiment analysis endpoint: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
