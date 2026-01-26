-- 003_curvature.sql
-- Curvature: kappa = |d2y| / (1 + dy^2)^(3/2)

CREATE OR REPLACE TABLE calculus_kappa AS
SELECT
    c.*,
    CASE
        WHEN dy IS NULL OR d2y IS NULL THEN NULL
        ELSE ABS(d2y) / POWER(1 + POWER(dy, 2), 1.5)
    END AS kappa
FROM calculus_d2y c;
