"""
Identity management REST API.

Provides endpoints for viewing, adding, and removing identity embeddings
from the Brain master registry.  Changes are broadcast to all connected
edge nodes via WebSocket.
"""

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("atlas.api.identity")

router = APIRouter(prefix="/identity", tags=["identity"])


class IdentityCreate(BaseModel):
    name: str
    modality: str
    embedding: list[float]
    source_node: Optional[str] = None


@router.get("/")
async def list_identities():
    """List all identity embeddings in the master registry."""
    from ..storage.repositories.identity import get_identity_repo

    repo = get_identity_repo()
    identities = await repo.list_all()
    manifest = await repo.get_all_names()
    return {
        "identities": identities,
        "summary": {mod: len(names) for mod, names in manifest.items()},
    }


@router.get("/names")
async def get_identity_names():
    """Get {modality: [names]} manifest."""
    from ..storage.repositories.identity import get_identity_repo

    repo = get_identity_repo()
    return await repo.get_all_names()


@router.post("/")
async def create_identity(body: IdentityCreate):
    """Add or update an identity embedding.  Broadcasts to all connected edges."""
    from ..storage.repositories.identity import get_identity_repo
    from .edge.websocket import get_all_connections

    if body.modality not in ("face", "gait", "speaker"):
        raise HTTPException(400, f"Invalid modality: {body.modality}")

    repo = get_identity_repo()
    embedding = np.array(body.embedding, dtype=np.float32)
    await repo.upsert(body.name, body.modality, embedding, source_node=body.source_node)

    # Broadcast to all connected edges
    update_msg = {
        "type": "identity_update",
        "name": body.name,
        "modality": body.modality,
        "embedding": body.embedding,
        "source_node": "brain",
    }
    broadcast_count = 0
    for loc_id, conn in get_all_connections().items():
        try:
            await conn.send(update_msg)
            broadcast_count += 1
        except Exception as e:
            logger.warning("Failed to broadcast to %s: %s", loc_id, e)

    return {
        "status": "ok",
        "name": body.name,
        "modality": body.modality,
        "dim": len(body.embedding),
        "broadcast_to": broadcast_count,
    }


@router.delete("/{name}/{modality}")
async def delete_identity(name: str, modality: str):
    """Delete an identity embedding.  Broadcasts deletion to all connected edges."""
    from ..storage.repositories.identity import get_identity_repo
    from .edge.websocket import get_all_connections

    if modality not in ("face", "gait", "speaker"):
        raise HTTPException(400, f"Invalid modality: {modality}")

    repo = get_identity_repo()
    deleted = await repo.delete(name, modality)
    if not deleted:
        raise HTTPException(404, f"Identity not found: {modality}/{name}")

    # Broadcast deletion to all connected edges
    delete_msg = {
        "type": "identity_delete",
        "name": name,
        "modality": modality,
    }
    broadcast_count = 0
    for loc_id, conn in get_all_connections().items():
        try:
            await conn.send(delete_msg)
            broadcast_count += 1
        except Exception as e:
            logger.warning("Failed to broadcast delete to %s: %s", loc_id, e)

    return {
        "status": "deleted",
        "name": name,
        "modality": modality,
        "broadcast_to": broadcast_count,
    }


@router.delete("/{name}")
async def delete_person(name: str):
    """Delete all identity embeddings for a person across all modalities."""
    from ..storage.repositories.identity import get_identity_repo
    from .edge.websocket import get_all_connections

    repo = get_identity_repo()
    count = await repo.delete_person(name)
    if count == 0:
        raise HTTPException(404, f"No identities found for: {name}")

    # Broadcast deletion for all modalities
    broadcast_count = 0
    for modality in ("face", "gait", "speaker"):
        delete_msg = {
            "type": "identity_delete",
            "name": name,
            "modality": modality,
        }
        for loc_id, conn in get_all_connections().items():
            try:
                await conn.send(delete_msg)
                broadcast_count += 1
            except Exception as e:
                logger.warning("Failed to broadcast delete to %s: %s", loc_id, e)

    return {
        "status": "deleted",
        "name": name,
        "modalities_removed": count,
        "broadcast_to": broadcast_count,
    }
