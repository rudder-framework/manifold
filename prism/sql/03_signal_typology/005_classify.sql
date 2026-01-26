-- 005_classify.sql
-- Combine all typology metrics into final classification

CREATE OR REPLACE TABLE typology_staged AS
SELECT
    c.entity_id,
    c.signal_id,
    c.window_id,
    -- From calculus (representative point per window)
    AVG(c.I) AS I_mean,
    AVG(c.y) AS y_mean,
    AVG(c.dy) AS dy_mean,
    AVG(c.d2y) AS d2y_mean,
    AVG(c.kappa) AS kappa_mean,
    -- From persistence
    p.hurst_rs,
    p.hurst_r2,
    p.persistence_class,
    -- From spectral
    sp.spectral_centroid,
    sp.spectral_bandwidth,
    sp.dominant_frequency,
    sp.spectral_rolloff,
    sp.periodicity_class,
    -- From stationarity
    st.mean_shift_ratio,
    st.variance_ratio,
    st.stationarity_class,
    -- From entropy
    e.sample_entropy,
    e.permutation_entropy,
    e.complexity_class,
    -- Overall behavioral classification
    CASE
        WHEN p.hurst_rs > 0.6 AND st.stationarity_class = 'non_stationary_mean' THEN 'trending'
        WHEN p.hurst_rs < 0.4 THEN 'mean_reverting'
        WHEN ABS(p.hurst_rs - 0.5) < 0.1 AND e.complexity_class = 'random' THEN 'random_walk'
        WHEN e.complexity_class = 'complex' AND sp.periodicity_class != 'periodic' THEN 'chaotic'
        ELSE 'indeterminate'
    END AS behavioral_class,
    NOW() AS _computed_at
FROM calculus_output c
LEFT JOIN typology_persistence_flat p USING (entity_id, signal_id, window_id)
LEFT JOIN typology_spectral_flat sp USING (entity_id, signal_id, window_id)
LEFT JOIN typology_stationarity st USING (entity_id, signal_id, window_id)
LEFT JOIN typology_entropy_flat e USING (entity_id, signal_id, window_id)
GROUP BY
    c.entity_id, c.signal_id, c.window_id,
    p.hurst_rs, p.hurst_r2, p.persistence_class,
    sp.spectral_centroid, sp.spectral_bandwidth, sp.dominant_frequency, sp.spectral_rolloff, sp.periodicity_class,
    st.mean_shift_ratio, st.variance_ratio, st.stationarity_class,
    e.sample_entropy, e.permutation_entropy, e.complexity_class;
