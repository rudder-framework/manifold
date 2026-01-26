-- _write_signal_class.sql
-- WRITES: outputs/signal_class.parquet

CREATE OR REPLACE TABLE signal_class_output AS
SELECT * FROM signal_class_staged;

-- Validate: must have rows
SELECT CASE
    WHEN (SELECT COUNT(*) FROM signal_class_output) = 0
    THEN error('FATAL: signal_class_output has 0 rows')
END;

-- Validate: key fields not null
SELECT CASE
    WHEN (SELECT COUNT(*) FROM signal_class_output WHERE entity_id IS NULL) > 0
    THEN error('FATAL: signal_class_output has NULL entity_id values')
END;

SELECT CASE
    WHEN (SELECT COUNT(*) FROM signal_class_output WHERE signal_id IS NULL) > 0
    THEN error('FATAL: signal_class_output has NULL signal_id values')
END;

-- Write parquet
COPY signal_class_output TO 'outputs/signal_class.parquet' (FORMAT PARQUET);

-- Log write
INSERT INTO _write_log (file, rows, written_at)
SELECT 'signal_class.parquet', COUNT(*), NOW()
FROM signal_class_output;

-- Confirm
SELECT 'signal_class.parquet written' AS status, COUNT(*) AS rows
FROM 'outputs/signal_class.parquet';
