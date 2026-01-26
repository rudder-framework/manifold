-- _write_calculus.sql
-- WRITES: outputs/calculus.parquet
-- VALIDATES: row count, no nulls in key fields

-- Create final output
CREATE OR REPLACE TABLE calculus_output AS
SELECT
    entity_id,
    signal_id,
    window_id,
    I,
    y,
    dy,
    d2y,
    kappa,
    laplacian,
    arc_length,
    _computed_at
FROM calculus_staged;

-- Validate: must have rows
SELECT CASE
    WHEN (SELECT COUNT(*) FROM calculus_output) = 0
    THEN error('FATAL: calculus_output has 0 rows - nothing computed')
END;

-- Validate: key fields not null
SELECT CASE
    WHEN (SELECT COUNT(*) FROM calculus_output WHERE entity_id IS NULL) > 0
    THEN error('FATAL: calculus_output has NULL entity_id values')
END;

SELECT CASE
    WHEN (SELECT COUNT(*) FROM calculus_output WHERE I IS NULL) > 0
    THEN error('FATAL: calculus_output has NULL I (index) values')
END;

-- Write parquet
COPY calculus_output TO 'outputs/calculus.parquet' (FORMAT PARQUET);

-- Log write
INSERT INTO _write_log (file, rows, written_at)
SELECT
    'calculus.parquet',
    COUNT(*),
    NOW()
FROM calculus_output;

-- Confirm
SELECT 'calculus.parquet written' AS status, COUNT(*) AS rows
FROM 'outputs/calculus.parquet';
