-- 002_covariance.sql
-- Compute pairwise covariance between signals

CREATE OR REPLACE TABLE geometry_covariance AS
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
    COVAR_POP(y_a, y_b) AS covariance,
    STDDEV(y_a) AS std_a,
    STDDEV(y_b) AS std_b
FROM paired_values
GROUP BY entity_id, signal_a, signal_b, window_id;
