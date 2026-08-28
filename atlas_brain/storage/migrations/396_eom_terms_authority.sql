-- atlas: atomic-bookkeeping
-- Immutable, bilingual EOM Terms releases. This migration stores no customer
-- acceptance and seeds no Terms content; later invitation/acceptance slices
-- reference the published version selected by the singleton pointer.

CREATE TABLE IF NOT EXISTS eom_terms_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    version_label VARCHAR(64) NOT NULL UNIQUE,
    status VARCHAR(16) NOT NULL DEFAULT 'draft',
    material_change BOOLEAN NOT NULL,
    documents JSONB NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    created_by_id BIGINT NOT NULL,
    created_by_name VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    published_by_id BIGINT,
    published_by_name VARCHAR(128),
    published_at TIMESTAMPTZ,
    CONSTRAINT ck_eom_terms_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_terms_version_label
        CHECK (version_label ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'),
    CONSTRAINT ck_eom_terms_status
        CHECK (status IN ('draft', 'published')),
    CONSTRAINT ck_eom_terms_documents_object
        CHECK (jsonb_typeof(documents) = 'object'),
    CONSTRAINT ck_eom_terms_content_hash
        CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_eom_terms_creator
        CHECK (created_by_id > 0 AND length(btrim(created_by_name)) > 0),
    CONSTRAINT ck_eom_terms_publication
        CHECK (
            (status = 'draft'
                AND published_by_id IS NULL
                AND published_by_name IS NULL
                AND published_at IS NULL)
            OR
            (status = 'published'
                AND published_by_id > 0
                AND length(btrim(published_by_name)) > 0
                AND published_at IS NOT NULL)
        )
);

CREATE TABLE IF NOT EXISTS eom_terms_current_version (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    version_id UUID NOT NULL UNIQUE
        REFERENCES eom_terms_versions(id) ON DELETE RESTRICT,
    selected_by_id BIGINT NOT NULL CHECK (selected_by_id > 0),
    selected_by_name VARCHAR(128) NOT NULL
        CHECK (length(btrim(selected_by_name)) > 0),
    selected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION protect_eom_terms_version()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        RAISE EXCEPTION 'EOM Terms version history is append-only';
    END IF;
    IF OLD.status = 'published' THEN
        RAISE EXCEPTION 'Published EOM Terms versions are immutable';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    IF NEW.status = 'draft' THEN
        RAISE EXCEPTION 'EOM Terms drafts cannot be edited; create a new version';
    END IF;
    IF NEW.id IS DISTINCT FROM OLD.id
       OR NEW.business_context_id IS DISTINCT FROM OLD.business_context_id
       OR NEW.version_label IS DISTINCT FROM OLD.version_label
       OR NEW.material_change IS DISTINCT FROM OLD.material_change
       OR NEW.documents IS DISTINCT FROM OLD.documents
       OR NEW.content_hash IS DISTINCT FROM OLD.content_hash
       OR NEW.created_by_id IS DISTINCT FROM OLD.created_by_id
       OR NEW.created_by_name IS DISTINCT FROM OLD.created_by_name
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'Publishing EOM Terms cannot rewrite draft content';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_protect_eom_terms_version
    ON eom_terms_versions;
CREATE TRIGGER trg_protect_eom_terms_version
    BEFORE UPDATE OR DELETE ON eom_terms_versions
    FOR EACH ROW EXECUTE FUNCTION protect_eom_terms_version();

DROP TRIGGER IF EXISTS trg_protect_eom_terms_version_truncate
    ON eom_terms_versions;
CREATE TRIGGER trg_protect_eom_terms_version_truncate
    BEFORE TRUNCATE ON eom_terms_versions
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_terms_version();

CREATE OR REPLACE FUNCTION require_published_eom_terms_current_version()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM eom_terms_versions AS version
        WHERE version.id = NEW.version_id
          AND version.status = 'published'
    ) THEN
        RAISE EXCEPTION 'Current EOM Terms version must be published';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_require_published_eom_terms_current_version
    ON eom_terms_current_version;
CREATE TRIGGER trg_require_published_eom_terms_current_version
    BEFORE INSERT OR UPDATE ON eom_terms_current_version
    FOR EACH ROW EXECUTE FUNCTION require_published_eom_terms_current_version();

CREATE OR REPLACE FUNCTION prevent_eom_terms_current_removal()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Current EOM Terms authority cannot be removed';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_terms_current_truncate
    ON eom_terms_current_version;
CREATE TRIGGER trg_prevent_eom_terms_current_truncate
    BEFORE TRUNCATE ON eom_terms_current_version
    FOR EACH STATEMENT EXECUTE FUNCTION prevent_eom_terms_current_removal();

DROP TRIGGER IF EXISTS trg_prevent_eom_terms_current_delete
    ON eom_terms_current_version;
CREATE TRIGGER trg_prevent_eom_terms_current_delete
    BEFORE DELETE ON eom_terms_current_version
    FOR EACH ROW EXECUTE FUNCTION prevent_eom_terms_current_removal();

CREATE INDEX IF NOT EXISTS idx_eom_terms_versions_created
    ON eom_terms_versions (created_at DESC, id DESC);
