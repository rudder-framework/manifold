-- 006_clustering.sql
-- Compute clustering via PRISM engine
-- Requires: prism_clustering() UDF registered

CREATE OR REPLACE TABLE geometry_clustering AS
WITH signal_features AS (
    -- Create feature vector for each signal
    SELECT
        entity_id,
        signal_id,
        window_id,
        ARRAY[
            AVG(y),
            STDDEV(y),
            AVG(dy),
            STDDEV(dy),
            AVG(kappa)
        ] AS feature_vector
    FROM calculus_output
    WHERE y IS NOT NULL
    GROUP BY entity_id, signal_id, window_id
),
clustering_input AS (
    SELECT
        entity_id,
        window_id,
        ARRAY_AGG(feature_vector ORDER BY signal_id) AS feature_matrix,
        ARRAY_AGG(signal_id ORDER BY signal_id) AS signal_ids
    FROM signal_features
    GROUP BY entity_id, window_id
)
SELECT
    entity_id,
    window_id,
    signal_ids,
    prism_clustering(feature_matrix, 3) AS clustering_result  -- k=3 clusters
FROM clustering_input;

-- Flatten clustering results
CREATE OR REPLACE TABLE geometry_clustering_flat AS
SELECT
    entity_id,
    window_id,
    UNNEST(signal_ids) AS signal_id,
    UNNEST(clustering_result.labels) AS cluster_id,
    clustering_result.inertia AS clustering_inertia,
    clustering_result.n_clusters AS n_clusters
FROM geometry_clustering;
