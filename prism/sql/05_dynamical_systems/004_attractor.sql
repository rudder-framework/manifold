-- 004_attractor.sql
-- Classify attractor type based on dynamics
-- Uses stability, periodicity, and complexity metrics

CREATE OR REPLACE TABLE dynamics_attractor AS
SELECT
    r.entity_id,
    r.signal_id,
    r.window_id,
    r.regime_id,
    b.basin_id,
    s.lyapunov_exponent,
    s.stability_class,
    -- Attractor type classification
    CASE
        WHEN s.is_stable AND t.periodicity_class = 'periodic' THEN 'limit_cycle'
        WHEN s.is_stable AND t.periodicity_class != 'periodic' THEN 'fixed_point'
        WHEN s.is_chaotic THEN 'strange'
        WHEN s.is_critical THEN 'bifurcation'
        ELSE 'unknown'
    END AS attractor_type,
    -- Transition probability (based on regime changes)
    (SELECT COUNT(DISTINCT regime_id)::FLOAT / COUNT(*) FROM dynamics_regime dr
     WHERE dr.entity_id = r.entity_id AND dr.signal_id = r.signal_id
     AND dr.window_id = r.window_id) AS transition_frequency,
    -- Time to boundary estimate (based on distance from regime edges)
    AVG(LEAST(ABS(r.y - b.basin_center + b.basin_width/2),
              ABS(r.y - b.basin_center - b.basin_width/2))) AS avg_distance_to_boundary
FROM dynamics_regime r
JOIN dynamics_basin b ON r.entity_id = b.entity_id
    AND r.signal_id = b.signal_id
    AND r.window_id = b.window_id
    AND r.regime_id = b.regime_id
LEFT JOIN dynamics_stability_flat s ON r.entity_id = s.entity_id
    AND r.signal_id = s.signal_id
    AND r.window_id = s.window_id
LEFT JOIN typology_spectral_flat t ON r.entity_id = t.entity_id
    AND r.signal_id = t.signal_id
    AND r.window_id = t.window_id
GROUP BY r.entity_id, r.signal_id, r.window_id, r.regime_id,
         b.basin_id, b.basin_center, b.basin_width,
         s.lyapunov_exponent, s.stability_class, s.is_stable, s.is_chaotic, s.is_critical,
         t.periodicity_class;
