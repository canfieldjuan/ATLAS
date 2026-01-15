#!/usr/bin/env python3
"""
RAG-based conversation learning.

Stores successful conversation patterns and retrieves them
during inference to improve responses WITHOUT retraining.

Benefits:
- No model retraining needed
- Updates in real-time
- Easy to add/remove examples
- Works with any model
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from atlas_brain.services.embedding.sentence_transformer import SentenceTransformerEmbedding
from atlas_brain.storage.database import init_database
from atlas_brain.storage.repositories.conversation import get_conversation_repo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationPatternStore:
    """
    Store and retrieve successful conversation patterns.
    
    Uses embeddings to find similar past interactions.
    """
    
    def __init__(
        self,
        embedding_service: Optional[SentenceTransformerEmbedding] = None,
        storage_path: Path = Path("data/conversation_patterns.json"),
    ):
        self.embedding = embedding_service or SentenceTransformerEmbedding()
        self.storage_path = storage_path
        self.patterns = []  # In-memory cache
        
        if not self.embedding.is_loaded:
            self.embedding.load()
    
    async def add_pattern(
        self,
        user_message: str,
        assistant_response: str,
        quality_score: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a successful conversation pattern.
        
        Args:
            user_message: User's input
            assistant_response: Successful assistant response
            quality_score: Quality rating (0-1)
            metadata: Additional context (intent, speaker, etc.)
        """
        # Create embedding of user message
        embedding = self.embedding.embed(user_message)
        
        pattern = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "quality_score": quality_score,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
            "added_at": datetime.now().isoformat(),
        }
        
        self.patterns.append(pattern)
        logger.info(f"Added pattern: {user_message[:50]}...")
    
    def find_similar_patterns(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.7,
    ) -> list[dict]:
        """
        Find similar conversation patterns.
        
        Args:
            query: User's current message
            top_k: Number of patterns to return
            min_similarity: Minimum cosine similarity
        
        Returns:
            List of similar patterns with similarity scores
        """
        import numpy as np
        from numpy.linalg import norm
        
        # Embed query
        query_embedding = self.embedding.embed(query)
        
        # Calculate similarities
        results = []
        for pattern in self.patterns:
            pattern_embedding = np.array(pattern["embedding"])
            
            # Cosine similarity
            similarity = np.dot(query_embedding, pattern_embedding) / (
                norm(query_embedding) * norm(pattern_embedding)
            )
            
            if similarity >= min_similarity:
                results.append({
                    "pattern": pattern,
                    "similarity": float(similarity),
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
    
    def build_context_from_patterns(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        Build context string from similar patterns.
        
        This gets injected into the prompt as examples.
        """
        similar = self.find_similar_patterns(query, top_k=top_k)
        
        if not similar:
            return ""
        
        context = "Here are similar past conversations:\n\n"
        
        for i, result in enumerate(similar, 1):
            pattern = result["pattern"]
            context += f"Example {i} (similarity: {result['similarity']:.2f}):\n"
            context += f"User: {pattern['user_message']}\n"
            context += f"Assistant: {pattern['assistant_response']}\n\n"
        
        return context


async def build_pattern_database_from_conversations(
    min_quality_score: float = 0.8,
    output_path: Path = Path("data/conversation_patterns.db"),
) -> None:
    """
    Build RAG database from existing conversations.
    
    Extracts high-quality patterns from conversation history.
    """
    repo = get_conversation_repo()
    pattern_store = ConversationPatternStore()
    
    # Get all conversations
    # You'd need to implement session iteration
    # For now, showing the concept
    
    logger.info("Building pattern database...")
    
    # Example: Process conversations
    # sessions = await repo.get_all_sessions()
    # for session in sessions:
    #     turns = await repo.get_history(session.id, limit=1000)
    #     
    #     # Extract user-assistant pairs
    #     for i in range(len(turns) - 1):
    #         if turns[i].role == "user" and turns[i+1].role == "assistant":
    #             # Calculate quality score based on:
    #             # - Response length
    #             # - User satisfaction (if tracked)
    #             # - Intent success
    #             quality = calculate_quality_score(turns[i], turns[i+1])
    #             
    #             if quality >= min_quality_score:
    #                 await pattern_store.add_pattern(
    #                     user_message=turns[i].content,
    #                     assistant_response=turns[i+1].content,
    #                     quality_score=quality,
    #                     metadata={
    #                         "intent": turns[i].intent,
    #                         "speaker_id": turns[i].speaker_id,
    #                     }
    #                 )
    
    logger.info(f"Pattern database built with {len(pattern_store.patterns)} patterns")


def calculate_quality_score(user_turn: dict, assistant_turn: dict) -> float:
    """
    Calculate quality score for a conversation pair.
    
    Factors:
    - Response relevance
    - Response length (not too short/long)
    - No error messages
    - Intent successfully handled
    """
    score = 1.0
    
    response = assistant_turn.get("content", "")
    
    # Too short (likely not helpful)
    if len(response) < 20:
        score *= 0.5
    
    # Error indicators
    error_phrases = ["sorry", "can't", "unable", "error", "failed"]
    if any(phrase in response.lower() for phrase in error_phrases):
        score *= 0.7
    
    # Intent success
    if user_turn.get("intent") and assistant_turn.get("metadata", {}).get("success"):
        score *= 1.2
    
    return min(score, 1.0)


async def example_rag_inference(user_query: str) -> str:
    """
    Example: Using RAG patterns during inference.
    """
    pattern_store = ConversationPatternStore()
    
    # Build context from similar patterns
    rag_context = pattern_store.build_context_from_patterns(user_query, top_k=3)
    
    # Build prompt with context
    system_prompt = f"""You are Atlas, a helpful AI assistant.

{rag_context}

Use these examples to inform your response, but adapt to the current context.
"""
    
    # Send to LLM with enriched context
    # ... (use your existing LLM service)
    
    return "Response informed by past successful patterns"


if __name__ == "__main__":
    # Build pattern database
    asyncio.run(build_pattern_database_from_conversations())
