-- 004_coupling.sql
-- Compute coupling strength and lead-lag relationships

CREATE OR REPLACE TABLE geometry_coupling AS
WITH lagged_correlations AS (
    -- Compute correlations at different lags
    SELECT
        c.entity_id,
        c.signal_a,
        c.signal_b,
        c.window_id,
        lag_offset,
        CORR(a.y, b_lagged.y) AS lagged_corr
    FROM geometry_correlation c
    CROSS JOIN (SELECT unnest(ARRAY[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) AS lag_offset) lags
    JOIN calculus_output a ON c.entity_id = a.entity_id
        AND c.signal_a = a.signal_id
        AND c.window_id = a.window_id
    JOIN calculus_output b_lagged ON c.entity_id = b_lagged.entity_id
        AND c.signal_b = b_lagged.signal_id
        AND c.window_id = b_lagged.window_id
    WHERE a.y IS NOT NULL AND b_lagged.y IS NOT NULL
    GROUP BY c.entity_id, c.signal_a, c.signal_b, c.window_id, lag_offset
),
optimal_lag AS (
    SELECT
        entity_id,
        signal_a,
        signal_b,
        window_id,
        lag_offset AS lag_optimal,
        lagged_corr AS max_lagged_corr
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY entity_id, signal_a, signal_b, window_id
                ORDER BY ABS(lagged_corr) DESC
            ) AS rn
        FROM lagged_correlations
    ) t
    WHERE rn = 1
)
SELECT
    entity_id,
    signal_a,
    signal_b,
    window_id,
    lag_optimal,
    max_lagged_corr AS coupling_strength,
    CASE
        WHEN lag_optimal > 0 THEN 'a_leads'
        WHEN lag_optimal < 0 THEN 'b_leads'
        ELSE 'synchronous'
    END AS lead_lag_direction
FROM optimal_lag;
