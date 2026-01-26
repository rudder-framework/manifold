-- 003_distance.sql
-- Compute Euclidean distance between signal pairs
-- DTW distance computed via PRISM engine

CREATE OR REPLACE TABLE geometry_distance AS
WITH paired_values AS (
    SELECT
        c.entity_id,
        c.signal_a,
        c.signal_b,
        c.window_id,
        a.y AS y_a,
        b.y AS y_b
    FROM geometry_correlation c
    JOIN calculus_output a ON c.entity_id = a.entity_id
        AND c.signal_a = a.signal_id
        AND c.window_id = a.window_id
    JOIN calculus_output b ON c.entity_id = b.entity_id
        AND c.signal_b = b.signal_id
        AND c.window_id = b.window_id
        AND a.I = b.I
    WHERE a.y IS NOT NULL AND b.y IS NOT NULL
)
SELECT
    entity_id,
    signal_a,
    signal_b,
    window_id,
    -- Euclidean distance (L2 norm of difference)
    SQRT(SUM(POWER(y_a - y_b, 2))) AS distance_euclidean,
    -- Manhattan distance (L1 norm)
    SUM(ABS(y_a - y_b)) AS distance_manhattan,
    -- Max distance (L-inf norm)
    MAX(ABS(y_a - y_b)) AS distance_chebyshev
FROM paired_values
GROUP BY entity_id, signal_a, signal_b, window_id;

-- DTW distance via PRISM engine (separate query due to array aggregation)
CREATE OR REPLACE TABLE geometry_dtw AS
SELECT
    gc.entity_id,
    gc.signal_a,
    gc.signal_b,
    gc.window_id,
    prism_dtw(
        ARRAY_AGG(a.y ORDER BY a.I),
        ARRAY_AGG(b.y ORDER BY b.I)
    ).distance AS distance_dtw
FROM geometry_correlation gc
JOIN calculus_output a ON gc.entity_id = a.entity_id
    AND gc.signal_a = a.signal_id
    AND gc.window_id = a.window_id
JOIN calculus_output b ON gc.entity_id = b.entity_id
    AND gc.signal_b = b.signal_id
    AND gc.window_id = b.window_id
    AND a.I = b.I
WHERE a.y IS NOT NULL AND b.y IS NOT NULL
GROUP BY gc.entity_id, gc.signal_a, gc.signal_b, gc.window_id;
