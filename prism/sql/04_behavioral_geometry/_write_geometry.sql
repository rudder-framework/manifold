-- _write_geometry.sql
-- WRITES: outputs/behavioral_geometry.parquet

CREATE OR REPLACE TABLE behavioral_geometry_output AS
SELECT
    c.entity_id,
    c.signal_a,
    c.signal_b,
    c.window_id,
    c.correlation,
    cv.covariance,
    d.distance_euclidean,
    d.distance_manhattan,
    dtw.distance_dtw,
    cp.coupling_strength,
    cp.lag_optimal,
    cp.lead_lag_direction,
    -- PCA loadings for signal_a (if available)
    pca.pc1_variance_ratio,
    pca.pc2_variance_ratio,
    -- Cluster assignments
    cl_a.cluster_id AS cluster_a,
    cl_b.cluster_id AS cluster_b,
    NOW() AS _computed_at
FROM geometry_correlation c
LEFT JOIN geometry_covariance cv USING (entity_id, signal_a, signal_b, window_id)
LEFT JOIN geometry_distance d USING (entity_id, signal_a, signal_b, window_id)
LEFT JOIN geometry_dtw dtw USING (entity_id, signal_a, signal_b, window_id)
LEFT JOIN geometry_coupling cp USING (entity_id, signal_a, signal_b, window_id)
LEFT JOIN geometry_pca_flat pca ON c.entity_id = pca.entity_id AND c.window_id = pca.window_id
LEFT JOIN geometry_clustering_flat cl_a ON c.entity_id = cl_a.entity_id
    AND c.signal_a = cl_a.signal_id AND c.window_id = cl_a.window_id
LEFT JOIN geometry_clustering_flat cl_b ON c.entity_id = cl_b.entity_id
    AND c.signal_b = cl_b.signal_id AND c.window_id = cl_b.window_id;

-- Validate: must have rows
SELECT CASE
    WHEN (SELECT COUNT(*) FROM behavioral_geometry_output) = 0
    THEN error('FATAL: behavioral_geometry_output has 0 rows')
END;

-- Write parquet
COPY behavioral_geometry_output TO 'outputs/behavioral_geometry.parquet' (FORMAT PARQUET);

-- Log write
INSERT INTO _write_log (file, rows, written_at)
SELECT 'behavioral_geometry.parquet', COUNT(*), NOW()
FROM behavioral_geometry_output;

-- Confirm
SELECT 'behavioral_geometry.parquet written' AS status, COUNT(*) AS rows
FROM 'outputs/behavioral_geometry.parquet';
