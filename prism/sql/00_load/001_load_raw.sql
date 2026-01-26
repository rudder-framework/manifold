-- 001_load_raw.sql
-- Loads and validates raw observations data
-- Expected columns: entity_id, signal_id, index (or I), value (or y)

-- Normalize column names if needed
CREATE OR REPLACE TABLE raw_signals AS
SELECT
    COALESCE(entity_id, 'default') AS entity_id,
    signal_id,
    COALESCE(I, "index", idx, time, t) AS I,
    COALESCE(y, value, val) AS y,
    0 AS window_id  -- Default window, can be overridden
FROM input_data;

-- Validate: must have data
SELECT CASE
    WHEN (SELECT COUNT(*) FROM raw_signals) = 0
    THEN error('FATAL: raw_signals has 0 rows - no data loaded')
END;

-- Validate: required columns exist and have data
SELECT CASE
    WHEN (SELECT COUNT(*) FROM raw_signals WHERE signal_id IS NULL) > 0
    THEN error('FATAL: raw_signals has NULL signal_id values')
END;

SELECT CASE
    WHEN (SELECT COUNT(*) FROM raw_signals WHERE I IS NULL) > 0
    THEN error('FATAL: raw_signals has NULL index (I) values')
END;

-- Log load stats
SELECT
    'raw_signals loaded' AS status,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT entity_id) AS entities,
    COUNT(DISTINCT signal_id) AS signals
FROM raw_signals;
