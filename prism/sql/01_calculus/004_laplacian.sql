-- 004_laplacian.sql
-- Laplacian (normalized second derivative)
-- For 1D: d2y normalized by scale

CREATE OR REPLACE TABLE calculus_laplacian AS
SELECT
    c.*,
    d2y / NULLIF(STDDEV(d2y) OVER (PARTITION BY entity_id, signal_id, window_id), 0) AS laplacian
FROM calculus_kappa c;
