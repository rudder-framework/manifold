-- 003_stationarity.sql
-- Compute stationarity metrics using SQL statistics
-- Pure SQL - no engine calls needed

CREATE OR REPLACE TABLE typology_stationarity AS
WITH signal_halves AS (
    SELECT
        entity_id,
        signal_id,
        window_id,
        CASE WHEN row_num <= total_rows / 2 THEN 'first_half' ELSE 'second_half' END AS half,
        y
    FROM (
        SELECT
            entity_id,
            signal_id,
            window_id,
            y,
            ROW_NUMBER() OVER (PARTITION BY entity_id, signal_id, window_id ORDER BY I) AS row_num,
            COUNT(*) OVER (PARTITION BY entity_id, signal_id, window_id) AS total_rows
        FROM calculus_output
        WHERE y IS NOT NULL
    ) t
),
half_stats AS (
    SELECT
        entity_id,
        signal_id,
        window_id,
        half,
        AVG(y) AS mean_y,
        STDDEV(y) AS std_y,
        VARIANCE(y) AS var_y
    FROM signal_halves
    GROUP BY entity_id, signal_id, window_id, half
)
SELECT
    f.entity_id,
    f.signal_id,
    f.window_id,
    f.mean_y AS mean_first_half,
    s.mean_y AS mean_second_half,
    f.std_y AS std_first_half,
    s.std_y AS std_second_half,
    -- Mean shift ratio
    ABS(f.mean_y - s.mean_y) / NULLIF(GREATEST(f.std_y, s.std_y), 0) AS mean_shift_ratio,
    -- Variance ratio
    f.var_y / NULLIF(s.var_y, 0) AS variance_ratio,
    -- Stationarity classification
    CASE
        WHEN ABS(f.mean_y - s.mean_y) / NULLIF(GREATEST(f.std_y, s.std_y), 0) > 1 THEN 'non_stationary_mean'
        WHEN f.var_y / NULLIF(s.var_y, 0) > 2 OR f.var_y / NULLIF(s.var_y, 0) < 0.5 THEN 'non_stationary_variance'
        ELSE 'stationary'
    END AS stationarity_class
FROM half_stats f
JOIN half_stats s ON f.entity_id = s.entity_id
    AND f.signal_id = s.signal_id
    AND f.window_id = s.window_id
    AND f.half = 'first_half'
    AND s.half = 'second_half';
