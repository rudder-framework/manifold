-- 001_regime_detection.sql
-- Detect regime changes using statistical properties
-- Pure SQL using change point detection heuristics

CREATE OR REPLACE TABLE dynamics_regime AS
WITH stats_per_segment AS (
    -- Compute rolling statistics in small segments
    SELECT
        entity_id,
        signal_id,
        window_id,
        I,
        y,
        AVG(y) OVER (PARTITION BY entity_id, signal_id, window_id ORDER BY I ROWS BETWEEN 10 PRECEDING AND 10 FOLLOWING) AS local_mean,
        STDDEV(y) OVER (PARTITION BY entity_id, signal_id, window_id ORDER BY I ROWS BETWEEN 10 PRECEDING AND 10 FOLLOWING) AS local_std
    FROM calculus_output
    WHERE y IS NOT NULL
),
regime_changes AS (
    -- Detect significant changes in local statistics
    SELECT
        entity_id,
        signal_id,
        window_id,
        I,
        y,
        local_mean,
        local_std,
        LAG(local_mean) OVER w AS prev_mean,
        LAG(local_std) OVER w AS prev_std,
        CASE
            WHEN ABS(local_mean - LAG(local_mean) OVER w) > 2 * local_std THEN 1
            ELSE 0
        END AS is_regime_change
    FROM stats_per_segment
    WINDOW w AS (PARTITION BY entity_id, signal_id, window_id ORDER BY I)
)
SELECT
    entity_id,
    signal_id,
    window_id,
    I,
    y,
    local_mean,
    local_std,
    -- Assign regime IDs based on cumulative change points
    SUM(is_regime_change) OVER (PARTITION BY entity_id, signal_id, window_id ORDER BY I) AS regime_id
FROM regime_changes;
