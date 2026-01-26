-- 001_first_derivative.sql
-- First derivative using central difference
-- dy/dI = (y[i+1] - y[i-1]) / (I[i+1] - I[i-1])

CREATE OR REPLACE TABLE calculus_dy AS
SELECT
    entity_id,
    signal_id,
    window_id,
    I,
    y,
    CASE
        WHEN LAG(I) OVER w IS NULL OR LEAD(I) OVER w IS NULL THEN NULL
        ELSE (LEAD(y) OVER w - LAG(y) OVER w) / NULLIF(LEAD(I) OVER w - LAG(I) OVER w, 0)
    END AS dy
FROM raw_signals
WINDOW w AS (PARTITION BY entity_id, signal_id, window_id ORDER BY I);
