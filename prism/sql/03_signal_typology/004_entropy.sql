-- 004_entropy.sql
-- Compute entropy metrics using PRISM entropy engines
-- Requires: prism_sample_entropy() and prism_permutation_entropy() UDFs

CREATE OR REPLACE TABLE typology_entropy AS
SELECT
    entity_id,
    signal_id,
    window_id,
    -- Entropy via PRISM engines
    prism_sample_entropy(ARRAY_AGG(y ORDER BY I)) AS sample_entropy_result,
    prism_permutation_entropy(ARRAY_AGG(y ORDER BY I)) AS permutation_entropy_result
FROM calculus_output
WHERE y IS NOT NULL
GROUP BY entity_id, signal_id, window_id;

-- Unnest entropy results
CREATE OR REPLACE TABLE typology_entropy_flat AS
SELECT
    entity_id,
    signal_id,
    window_id,
    sample_entropy_result.sample_entropy AS sample_entropy,
    permutation_entropy_result.permutation_entropy AS permutation_entropy,
    permutation_entropy_result.normalized_entropy AS normalized_permutation_entropy,
    -- Complexity classification
    CASE
        WHEN sample_entropy_result.sample_entropy < 0.5 THEN 'ordered'
        WHEN sample_entropy_result.sample_entropy > 2.0 THEN 'random'
        ELSE 'complex'
    END AS complexity_class
FROM typology_entropy;
