-- 001_from_units.sql
-- Classify signal type from metadata/units if available
-- intensive: temperature, pressure, concentration (independent of size)
-- extensive: mass, volume, energy (scales with size)
-- rate: flow, velocity, power (change over time)
-- state: position, level, count (instantaneous value)

CREATE OR REPLACE TABLE signal_class_units AS
SELECT DISTINCT
    entity_id,
    signal_id,
    CASE
        -- Intensive properties
        WHEN signal_id ILIKE '%temp%' OR signal_id ILIKE '%pressure%'
            OR signal_id ILIKE '%concentration%' OR signal_id ILIKE '%density%'
            OR signal_id ILIKE '%viscosity%' OR signal_id ILIKE '%conductivity%'
            THEN 'intensive'
        -- Extensive properties
        WHEN signal_id ILIKE '%mass%' OR signal_id ILIKE '%volume%'
            OR signal_id ILIKE '%energy%' OR signal_id ILIKE '%entropy%'
            OR signal_id ILIKE '%heat%' OR signal_id ILIKE '%work%'
            THEN 'extensive'
        -- Rate properties
        WHEN signal_id ILIKE '%flow%' OR signal_id ILIKE '%velocity%'
            OR signal_id ILIKE '%power%' OR signal_id ILIKE '%rate%'
            OR signal_id ILIKE '%speed%' OR signal_id ILIKE '%acceleration%'
            THEN 'rate'
        -- State properties
        WHEN signal_id ILIKE '%position%' OR signal_id ILIKE '%level%'
            OR signal_id ILIKE '%count%' OR signal_id ILIKE '%state%'
            OR signal_id ILIKE '%status%' OR signal_id ILIKE '%mode%'
            THEN 'state'
        ELSE 'unknown'
    END AS signal_class_from_units
FROM raw_signals;
