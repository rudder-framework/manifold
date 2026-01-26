-- 003_basin.sql
-- Compute basin of attraction metrics
-- Uses regime statistics to identify basins

CREATE OR REPLACE TABLE dynamics_basin AS
WITH regime_stats AS (
    SELECT
        entity_id,
        signal_id,
        window_id,
        regime_id,
        AVG(y) AS regime_mean,
        STDDEV(y) AS regime_std,
        MIN(y) AS regime_min,
        MAX(y) AS regime_max,
        COUNT(*) AS regime_duration
    FROM dynamics_regime
    GROUP BY entity_id, signal_id, window_id, regime_id
),
basin_analysis AS (
    SELECT
        entity_id,
        signal_id,
        window_id,
        regime_id,
        regime_mean,
        regime_std,
        regime_min,
        regime_max,
        regime_duration,
        -- Basin width (range within regime)
        regime_max - regime_min AS basin_width,
        -- Assign basin ID based on clustering of regime means
        NTILE(3) OVER (
            PARTITION BY entity_id, signal_id, window_id
            ORDER BY regime_mean
        ) AS basin_id
    FROM regime_stats
)
SELECT
    entity_id,
    signal_id,
    window_id,
    regime_id,
    basin_id,
    regime_mean AS basin_center,
    basin_width,
    regime_std AS basin_variance,
    regime_duration AS time_in_basin
FROM basin_analysis;
