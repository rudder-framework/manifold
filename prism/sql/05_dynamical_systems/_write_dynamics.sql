-- _write_dynamics.sql
-- WRITES: outputs/dynamical_systems.parquet

CREATE OR REPLACE TABLE dynamical_systems_output AS
SELECT
    r.entity_id,
    r.signal_id,
    r.window_id,
    r.I,
    r.regime_id,
    a.basin_id,
    a.attractor_type,
    s.lyapunov_exponent AS regime_stability,
    s.stability_class,
    a.transition_frequency AS transition_prob,
    a.avg_distance_to_boundary AS time_to_boundary,
    NOW() AS _computed_at
FROM dynamics_regime r
LEFT JOIN dynamics_attractor a ON r.entity_id = a.entity_id
    AND r.signal_id = a.signal_id
    AND r.window_id = a.window_id
    AND r.regime_id = a.regime_id
LEFT JOIN dynamics_stability_flat s ON r.entity_id = s.entity_id
    AND r.signal_id = s.signal_id
    AND r.window_id = s.window_id;

-- Validate: must have rows
SELECT CASE
    WHEN (SELECT COUNT(*) FROM dynamical_systems_output) = 0
    THEN error('FATAL: dynamical_systems_output has 0 rows')
END;

-- Write parquet
COPY dynamical_systems_output TO 'outputs/dynamical_systems.parquet' (FORMAT PARQUET);

-- Log write
INSERT INTO _write_log (file, rows, written_at)
SELECT 'dynamical_systems.parquet', COUNT(*), NOW()
FROM dynamical_systems_output;

-- Confirm
SELECT 'dynamical_systems.parquet written' AS status, COUNT(*) AS rows
FROM 'outputs/dynamical_systems.parquet';
