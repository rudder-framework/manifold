-- 001_correlation.sql
-- Compute pairwise correlations between signals
-- Pure SQL using window functions

CREATE OR REPLACE TABLE geometry_correlation AS
WITH signal_pairs AS (
    SELECT DISTINCT
        a.entity_id,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.window_id
    FROM calculus_output a
    JOIN calculus_output b ON a.entity_id = b.entity_id
        AND a.window_id = b.window_id
        AND a.signal_id < b.signal_id  -- Avoid duplicates and self-joins
),
paired_values AS (
    SELECT
        sp.entity_id,
        sp.signal_a,
        sp.signal_b,
        sp.window_id,
        a.I,
        a.y AS y_a,
        b.y AS y_b
    FROM signal_pairs sp
    JOIN calculus_output a ON sp.entity_id = a.entity_id
        AND sp.signal_a = a.signal_id
        AND sp.window_id = a.window_id
    JOIN calculus_output b ON sp.entity_id = b.entity_id
        AND sp.signal_b = b.signal_id
        AND sp.window_id = b.window_id
        AND a.I = b.I
    WHERE a.y IS NOT NULL AND b.y IS NOT NULL
)
SELECT
    entity_id,
    signal_a,
    signal_b,
    window_id,
    CORR(y_a, y_b) AS correlation,
    COUNT(*) AS n_pairs
FROM paired_values
GROUP BY entity_id, signal_a, signal_b, window_id;
