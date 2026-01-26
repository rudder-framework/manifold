-- _write_typology.sql
-- WRITES: outputs/signal_typology.parquet

CREATE OR REPLACE TABLE signal_typology_output AS
SELECT * FROM typology_staged;

-- Validate: must have rows
SELECT CASE
    WHEN (SELECT COUNT(*) FROM signal_typology_output) = 0
    THEN error('FATAL: signal_typology_output has 0 rows')
END;

-- Validate: key fields not null
SELECT CASE
    WHEN (SELECT COUNT(*) FROM signal_typology_output WHERE entity_id IS NULL) > 0
    THEN error('FATAL: signal_typology_output has NULL entity_id values')
END;

-- Write parquet
COPY signal_typology_output TO 'outputs/signal_typology.parquet' (FORMAT PARQUET);

-- Log write
INSERT INTO _write_log (file, rows, written_at)
SELECT 'signal_typology.parquet', COUNT(*), NOW()
FROM signal_typology_output;

-- Confirm
SELECT 'signal_typology.parquet written' AS status, COUNT(*) AS rows
FROM 'outputs/signal_typology.parquet';
