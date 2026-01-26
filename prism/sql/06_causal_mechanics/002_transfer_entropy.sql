-- 002_transfer_entropy.sql
-- Compute transfer entropy via PRISM engine
-- Requires: prism_transfer_entropy() UDF registered

CREATE OR REPLACE TABLE causal_transfer_entropy AS
WITH signal_pairs AS (
    SELECT DISTINCT
        entity_id,
        source_signal,
        target_signal,
        window_id
    FROM causal_granger_flat
)
SELECT
    sp.entity_id,
    sp.source_signal,
    sp.target_signal,
    sp.window_id,
    prism_transfer_entropy(
        ARRAY_AGG(a.y ORDER BY a.I),
        ARRAY_AGG(b.y ORDER BY b.I)
    ) AS te_result
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

-- Flatten transfer entropy results
CREATE OR REPLACE TABLE causal_transfer_entropy_flat AS
SELECT
    entity_id,
    source_signal,
    target_signal,
    window_id,
    te_result.transfer_entropy AS transfer_entropy,
    te_result.normalized_te AS normalized_transfer_entropy
FROM causal_transfer_entropy;
