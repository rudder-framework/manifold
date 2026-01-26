-- 002_second_derivative.sql
-- Second derivative
-- d2y/dI2 = (dy[i+1] - dy[i-1]) / (I[i+1] - I[i-1])

CREATE OR REPLACE TABLE calculus_d2y AS
SELECT
    c.*,
    CASE
        WHEN LAG(c.I) OVER w IS NULL OR LEAD(c.I) OVER w IS NULL THEN NULL
        ELSE (LEAD(c.dy) OVER w - LAG(c.dy) OVER w) / NULLIF(LEAD(c.I) OVER w - LAG(c.I) OVER w, 0)
    END AS d2y
FROM calculus_dy c
WINDOW w AS (PARTITION BY entity_id, signal_id, window_id ORDER BY I);
