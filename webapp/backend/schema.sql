-- GLOF inventory validation — PostgreSQL schema
-- Managed via pgAdmin4 (server + local).

CREATE TABLE IF NOT EXISTS lakes (
    lake_key        TEXT PRIMARY KEY,
    idx             INTEGER UNIQUE,
    area_name       TEXT NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    area_ha         DOUBLE PRECISION,
    model_score     DOUBLE PRECISION,
    dist_glacier_m  DOUBLE PRECISION,
    elev_mean       DOUBLE PRECISION,
    in_watchlist    BOOLEAN DEFAULT FALSE,
    known_glof      BOOLEAN DEFAULT FALSE,
    thumb           TEXT
);
CREATE INDEX IF NOT EXISTS lakes_area_idx     ON lakes (area_name);
CREATE INDEX IF NOT EXISTS lakes_watch_idx    ON lakes (in_watchlist);

CREATE TABLE IF NOT EXISTS reviewers (
    id          SERIAL PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS labels (
    id           SERIAL PRIMARY KEY,
    lake_key     TEXT NOT NULL REFERENCES lakes(lake_key) ON DELETE CASCADE,
    reviewer_id  INTEGER NOT NULL REFERENCES reviewers(id) ON DELETE CASCADE,
    is_real      BOOLEAN,
    feature_type TEXT,
    confidence   SMALLINT,
    note         TEXT,
    updated_at   TIMESTAMPTZ DEFAULT now(),
    UNIQUE (lake_key, reviewer_id)
);
CREATE INDEX IF NOT EXISTS labels_lake_idx ON labels (lake_key);
