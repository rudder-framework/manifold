-- 002_stability.sql
-- Compute stability metrics using PRISM lyapunov engine
-- Requires: prism_lyapunov() UDF registered

CREATE OR REPLACE TABLE dynamics_stability AS
SELECT
    entity_id,
    signal_id,
    window_id,
    prism_lyapunov(ARRAY_AGG(y ORDER BY I)) AS lyapunov_result
FROM calculus_output
WHERE y IS NOT NULL
GROUP BY entity_id, signal_id, window_id;

-- Flatten lyapunov results
CREATE OR REPLACE TABLE dynamics_stability_flat AS
SELECT
    entity_id,
    signal_id,
    window_id,
    lyapunov_result.lyapunov_exponent AS lyapunov_exponent,
    lyapunov_result.is_chaotic AS is_chaotic,
    lyapunov_result.is_stable AS is_stable,
    lyapunov_result.is_critical AS is_critical,
    lyapunov_result.method AS lyapunov_method,
    -- Stability classification
    CASE
        WHEN lyapunov_result.lyapunov_exponent > 0.1 THEN 'unstable'
        WHEN lyapunov_result.lyapunov_exponent < -0.1 THEN 'stable'
        ELSE 'marginal'
    END AS stability_class
FROM dynamics_stability;
