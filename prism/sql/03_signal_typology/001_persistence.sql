-- 001_persistence.sql
-- Compute persistence metrics using PRISM hurst engine
-- Requires: prism_hurst() UDF registered

CREATE OR REPLACE TABLE typology_persistence AS
SELECT
    entity_id,
    signal_id,
    window_id,
    -- Hurst exponent via PRISM engine
    prism_hurst(ARRAY_AGG(y ORDER BY I)) AS hurst_result
FROM calculus_output
WHERE y IS NOT NULL
GROUP BY entity_id, signal_id, window_id;

-- Unnest hurst results
CREATE OR REPLACE TABLE typology_persistence_flat AS
SELECT
    entity_id,
    signal_id,
    window_id,
    hurst_result.hurst AS hurst_rs,
    hurst_result.hurst_r2 AS hurst_r2,
    -- Persistence classification
    CASE
        WHEN hurst_result.hurst > 0.6 THEN 'persistent'
        WHEN hurst_result.hurst < 0.4 THEN 'antipersistent'
        ELSE 'neutral'
    END AS persistence_class
FROM typology_persistence;
