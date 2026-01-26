-- 005_arc_length.sql
-- Cumulative arc length: integral of sqrt(1 + dy^2) dI

CREATE OR REPLACE TABLE calculus_staged AS
SELECT
    c.*,
    SUM(
        CASE
            WHEN dy IS NULL OR LAG(I) OVER w IS NULL THEN 0
            ELSE SQRT(1 + POWER(dy, 2)) * (I - LAG(I) OVER w)
        END
    ) OVER (PARTITION BY entity_id, signal_id, window_id ORDER BY I) AS arc_length,
    NOW() AS _computed_at
FROM calculus_laplacian c
WINDOW w AS (PARTITION BY entity_id, signal_id, window_id ORDER BY I);
