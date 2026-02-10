# Identity Sync — Brain ↔ Edge Embedding Distribution

## Overview

Identity embeddings (face, gait, speaker) are synced between the Brain
master registry and all connected edge nodes over WebSocket.  Register once
on any node → Brain stores the master copy → all other nodes get it
automatically.

## Architecture

```
Brain (PostgreSQL: identity_embeddings table)
│
├── WS ←→ Edge Node 1 (office Pi)
│          local face_db/ gait_db/ speaker_db/ (.npy files)
│
├── WS ←→ Edge Node 2 (future)
│
└── WS ←→ Edge Node N (future)
```

## Sync Flow

1. **Edge connects** → sends `identity_sync_request` with `{modality: [names]}`
2. **Brain diffs** against `identity_embeddings` table → responds with `identity_sync`
   containing missing embeddings and names to delete
3. **Edge saves** `.npy` files and hot-reloads in-memory databases (no restart)
4. **Local registration** on an edge → pushed to Brain via `identity_register`
5. **Brain broadcasts** `identity_update` to all OTHER connected edges
6. **Periodic re-sync** every 5 minutes as a safety net

## WS Protocol

### Edge → Brain

| Message Type              | Fields                                    |
|---------------------------|-------------------------------------------|
| `identity_sync_request`   | `current: {face: [names], gait: [...], speaker: [...]}` |
| `identity_register`       | `name, modality, embedding: float[], node_id` |

### Brain → Edge

| Message Type       | Fields                                              |
|--------------------|-----------------------------------------------------|
| `identity_sync`    | `identities: {mod: {name: float[]}}, delete: {mod: [names]}` |
| `identity_update`  | `name, modality, embedding: float[], source_node`   |
| `identity_delete`  | `name, modality`                                    |

## Database

### Table: `identity_embeddings` (Migration 017)

```sql
CREATE TABLE identity_embeddings (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    modality VARCHAR(20) NOT NULL,     -- 'face', 'gait', 'speaker'
    embedding BYTEA NOT NULL,          -- pickle(numpy array)
    embedding_dim INTEGER NOT NULL,    -- 512 / 256 / 192
    source_node VARCHAR(100),          -- which edge registered it
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    UNIQUE(name, modality)
);
```

Embedding dimensions by modality:
- **face**: 512-dim (MobileFaceNet / ArcFace)
- **gait**: 256-dim (YOLOv8n-pose temporal features)
- **speaker**: 192-dim (CAM++ 3D-Speaker)

## REST API

All under `/api/v1/identity/`

| Method   | Endpoint                  | Description                         |
|----------|---------------------------|-------------------------------------|
| `GET`    | `/`                       | List all identities with metadata   |
| `GET`    | `/names`                  | Get `{modality: [names]}` manifest  |
| `POST`   | `/`                       | Add/update identity + broadcast     |
| `DELETE` | `/{name}/{modality}`      | Delete one identity + broadcast     |
| `DELETE` | `/{name}`                 | Delete all modalities for a person  |

### POST body

```json
{
  "name": "juan_canfield",
  "modality": "face",
  "embedding": [0.123, -0.456, ...],
  "source_node": "brain"
}
```

## Files

### Brain Side

| File | Description |
|------|-------------|
| `atlas_brain/storage/migrations/017_identity_sync.sql` | DB migration |
| `atlas_brain/storage/repositories/identity.py` | `IdentityRepository` — CRUD + diff_manifest |
| `atlas_brain/api/edge/websocket.py` | WS handlers: `identity_sync_request`, `identity_register` |
| `atlas_brain/api/identity.py` | REST API for identity management |
| `atlas_brain/api/__init__.py` | Router registration |
| `atlas_brain/storage/repositories/__init__.py` | Repo registration |

### Edge Side (Pi at `/opt/atlas-node/`)

| File | Description |
|------|-------------|
| `atlas_node/identity_sync.py` | `IdentitySyncManager` — sync logic, .npy management |
| `atlas_node/ws_client.py` | Bidirectional WS with handler dispatch |
| `atlas_node/config.py` | `IDENTITY_SYNC_ENABLED`, `IDENTITY_SYNC_INTERVAL`, `IDENTITY_WATCH_INTERVAL` |
| `atlas_node/main.py` | Wiring: sync manager + WS handlers |

## Testing

### Run migration
```bash
# Brain auto-runs migrations on startup, or manually:
psql -U atlas -d atlas_brain -f atlas_brain/storage/migrations/017_identity_sync.sql
```

### Restart Brain
```bash
# Brain server (uvicorn)
cd ~/Desktop/Atlas
# restart however you normally do (systemd, docker, manual)
```

### Verify sync
```bash
# Check Pi logs after Brain restart
ssh canfieldjuan@100.95.224.113 "journalctl -u atlas-node -n 20 --no-pager"

# Should see:
# Identity sync: sent request with 3 face, 0 gait, 0 speaker
# Identity sync: received 0 embeddings, 0 deletions
# (0 because the Brain DB is empty initially)
```

### Seed Brain DB from Pi's existing .npy files
After sync is running, register a face on the Pi:
```bash
python /opt/atlas-node/scripts/register_face.py juan_canfield --image /path/to/photo.jpg
```
The IdentitySyncManager's file watcher will detect the new .npy within 10 seconds
and push it to Brain. Brain stores it and any future edge nodes will receive it
on connect.

### REST API test
```bash
# List identities
curl http://localhost:8000/api/v1/identity/

# Get manifest
curl http://localhost:8000/api/v1/identity/names

# Delete an identity
curl -X DELETE http://localhost:8000/api/v1/identity/old_person/face
```
