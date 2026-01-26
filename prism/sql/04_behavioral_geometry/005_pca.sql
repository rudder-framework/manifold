-- 005_pca.sql
-- Compute PCA loadings via PRISM engine
-- Requires: prism_pca() UDF registered

CREATE OR REPLACE TABLE geometry_pca AS
WITH signal_matrix AS (
    -- Pivot signals into columns for PCA
    SELECT
        entity_id,
        window_id,
        I,
        ARRAY_AGG(y ORDER BY signal_id) AS signal_vector,
        ARRAY_AGG(signal_id ORDER BY signal_id) AS signal_ids
    FROM calculus_output
    WHERE y IS NOT NULL
    GROUP BY entity_id, window_id, I
),
pca_input AS (
    SELECT
        entity_id,
        window_id,
        ARRAY_AGG(signal_vector ORDER BY I) AS data_matrix,
        ANY_VALUE(signal_ids) AS signal_ids
    FROM signal_matrix
    GROUP BY entity_id, window_id
)
SELECT
    entity_id,
    window_id,
    signal_ids,
    prism_pca(data_matrix) AS pca_result
FROM pca_input;

-- Flatten PCA results
CREATE OR REPLACE TABLE geometry_pca_flat AS
SELECT
    entity_id,
    window_id,
    pca_result.explained_variance_ratio[1] AS pc1_variance_ratio,
    pca_result.explained_variance_ratio[2] AS pc2_variance_ratio,
    pca_result.components[1] AS pc1_loadings,
    pca_result.components[2] AS pc2_loadings,
    pca_result.n_components AS n_components
FROM geometry_pca;
