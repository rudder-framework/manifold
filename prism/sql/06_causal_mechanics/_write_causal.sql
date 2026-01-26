-- _write_causal.sql
-- WRITES: outputs/causal_mechanics.parquet

CREATE OR REPLACE TABLE causal_mechanics_output AS
SELECT
    entity_id,
    source_signal,
    target_signal,
    window_id,
    granger_f_stat,
    granger_p_value,
    transfer_entropy,
    causal_direction,
    causal_strength,
    source_role AS role,  -- Primary role column
    target_role,
    NOW() AS _computed_at
FROM causal_staged;

-- Validate: must have rows
SELECT CASE
    WHEN (SELECT COUNT(*) FROM causal_mechanics_output) = 0
    THEN error('FATAL: causal_mechanics_output has 0 rows')
END;

-- Write parquet
COPY causal_mechanics_output TO 'outputs/causal_mechanics.parquet' (FORMAT PARQUET);

-- Log write
INSERT INTO _write_log (file, rows, written_at)
SELECT 'causal_mechanics.parquet', COUNT(*), NOW()
FROM causal_mechanics_output;

-- Confirm
SELECT 'causal_mechanics.parquet written' AS status, COUNT(*) AS rows
FROM 'outputs/causal_mechanics.parquet';
