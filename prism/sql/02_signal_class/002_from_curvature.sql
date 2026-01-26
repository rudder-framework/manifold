-- 002_from_curvature.sql
-- Classify based on curvature statistics
-- High curvature variance = nonlinear dynamics
-- Low curvature = linear/smooth

CREATE OR REPLACE TABLE signal_class_curvature AS
SELECT
    entity_id,
    signal_id,
    AVG(kappa) AS mean_curvature,
    STDDEV(kappa) AS std_curvature,
    MAX(kappa) AS max_curvature,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY kappa) AS median_curvature,
    CASE
        WHEN STDDEV(kappa) / NULLIF(AVG(kappa), 0) > 2 THEN 'nonlinear'
        WHEN STDDEV(kappa) / NULLIF(AVG(kappa), 0) < 0.5 THEN 'linear'
        ELSE 'mixed'
    END AS curvature_class
FROM calculus_output
WHERE kappa IS NOT NULL
GROUP BY entity_id, signal_id;
