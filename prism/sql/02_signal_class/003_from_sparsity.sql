-- 003_from_sparsity.sql
-- Detect sparse signals (many zeros or missing values)
-- Detect monotonic signals
-- Detect bounded signals

CREATE OR REPLACE TABLE signal_class_sparsity AS
SELECT
    entity_id,
    signal_id,
    COUNT(*) AS n_points,
    SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS zero_fraction,
    SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS null_fraction,
    -- Monotonic check: all dy same sign
    CASE
        WHEN SUM(CASE WHEN dy > 0 THEN 1 ELSE 0 END) = SUM(CASE WHEN dy IS NOT NULL THEN 1 ELSE 0 END) THEN TRUE
        WHEN SUM(CASE WHEN dy < 0 THEN 1 ELSE 0 END) = SUM(CASE WHEN dy IS NOT NULL THEN 1 ELSE 0 END) THEN TRUE
        ELSE FALSE
    END AS is_monotonic,
    -- Sparse check
    CASE
        WHEN SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) > 0.5 THEN TRUE
        ELSE FALSE
    END AS is_sparse,
    -- Bounded check (within 3 std of mean)
    CASE
        WHEN MAX(y) - MIN(y) < 6 * STDDEV(y) THEN TRUE
        ELSE FALSE
    END AS is_bounded,
    -- Statistics for downstream use
    MIN(y) AS y_min,
    MAX(y) AS y_max,
    AVG(y) AS y_mean,
    STDDEV(y) AS y_std
FROM calculus_output
GROUP BY entity_id, signal_id;
