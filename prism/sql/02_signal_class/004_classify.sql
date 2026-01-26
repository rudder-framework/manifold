-- 004_classify.sql
-- Combine all classification sources into final signal_class

CREATE OR REPLACE TABLE signal_class_staged AS
SELECT
    u.entity_id,
    u.signal_id,
    -- Final class: prefer units-based, fall back to curvature-based
    CASE
        WHEN u.signal_class_from_units != 'unknown' THEN u.signal_class_from_units
        WHEN c.curvature_class = 'linear' THEN 'state'
        WHEN c.curvature_class = 'nonlinear' THEN 'rate'
        ELSE 'unknown'
    END AS signal_class,
    COALESCE(s.is_sparse, FALSE) AS is_sparse,
    COALESCE(s.is_monotonic, FALSE) AS is_monotonic,
    COALESCE(s.is_bounded, TRUE) AS is_bounded,
    -- Interpolation is valid if not sparse and not too many nulls
    CASE
        WHEN COALESCE(s.is_sparse, FALSE) = FALSE AND COALESCE(s.null_fraction, 0) < 0.1 THEN TRUE
        ELSE FALSE
    END AS interpolation_valid,
    -- Index dimension (placeholder - could come from metadata)
    'time' AS index_dimension,
    -- Curvature stats
    c.mean_curvature,
    c.std_curvature,
    c.curvature_class,
    -- Signal stats
    s.n_points,
    s.y_min,
    s.y_max,
    s.y_mean,
    s.y_std,
    NOW() AS _computed_at
FROM signal_class_units u
LEFT JOIN signal_class_curvature c USING (entity_id, signal_id)
LEFT JOIN signal_class_sparsity s USING (entity_id, signal_id);
