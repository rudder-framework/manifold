-- 001_granger.sql
-- Compute Granger causality via PRISM engine
-- Requires: prism_granger() UDF registered

CREATE OR REPLACE TABLE causal_granger AS
WITH signal_pairs AS (
    SELECT DISTINCT
        a.entity_id,
        a.signal_id AS source_signal,
        b.signal_id AS target_signal,
        a.window_id
    FROM calculus_output a
    JOIN calculus_output b ON a.entity_id = b.entity_id
        AND a.window_id = b.window_id
        AND a.signal_id != b.signal_id
)
SELECT
    sp.entity_id,
    sp.source_signal,
    sp.target_signal,
    sp.window_id,
    prism_granger(
        ARRAY_AGG(a.y ORDER BY a.I),
        ARRAY_AGG(b.y ORDER BY b.I)
    ) AS granger_result
FROM signal_pairs sp
JOIN calculus_output a ON sp.entity_id = a.entity_id
    AND sp.source_signal = a.signal_id
    AND sp.window_id = a.window_id
JOIN calculus_output b ON sp.entity_id = b.entity_id
    AND sp.target_signal = b.signal_id
    AND sp.window_id = b.window_id
    AND a.I = b.I
WHERE a.y IS NOT NULL AND b.y IS NOT NULL
GROUP BY sp.entity_id, sp.source_signal, sp.target_signal, sp.window_id;

-- Flatten Granger results
CREATE OR REPLACE TABLE causal_granger_flat AS
SELECT
    entity_id,
    source_signal,
    target_signal,
    window_id,
    granger_result.f_statistic AS granger_f_stat,
    granger_result.p_value AS granger_p_value,
    granger_result.is_significant AS granger_significant
FROM causal_granger;
