-- 003_role_assignment.sql
-- Assign causal roles based on Granger and transfer entropy results
-- Roles: SOURCE, SINK, CONDUIT, ISOLATED

CREATE OR REPLACE TABLE causal_roles AS
WITH causal_stats AS (
    -- Aggregate incoming and outgoing causal relationships per signal
    SELECT
        entity_id,
        source_signal AS signal_id,
        window_id,
        COUNT(CASE WHEN g.granger_significant THEN 1 END) AS outgoing_granger,
        SUM(COALESCE(te.transfer_entropy, 0)) AS outgoing_te
    FROM causal_granger_flat g
    LEFT JOIN causal_transfer_entropy_flat te USING (entity_id, source_signal, target_signal, window_id)
    GROUP BY entity_id, source_signal, window_id
),
incoming_stats AS (
    SELECT
        entity_id,
        target_signal AS signal_id,
        window_id,
        COUNT(CASE WHEN g.granger_significant THEN 1 END) AS incoming_granger,
        SUM(COALESCE(te.transfer_entropy, 0)) AS incoming_te
    FROM causal_granger_flat g
    LEFT JOIN causal_transfer_entropy_flat te USING (entity_id, source_signal, target_signal, window_id)
    GROUP BY entity_id, target_signal, window_id
)
SELECT
    COALESCE(o.entity_id, i.entity_id) AS entity_id,
    COALESCE(o.signal_id, i.signal_id) AS signal_id,
    COALESCE(o.window_id, i.window_id) AS window_id,
    COALESCE(o.outgoing_granger, 0) AS outgoing_granger,
    COALESCE(i.incoming_granger, 0) AS incoming_granger,
    COALESCE(o.outgoing_te, 0) AS outgoing_te,
    COALESCE(i.incoming_te, 0) AS incoming_te,
    -- Role assignment
    CASE
        WHEN COALESCE(o.outgoing_granger, 0) > 0 AND COALESCE(i.incoming_granger, 0) = 0 THEN 'SOURCE'
        WHEN COALESCE(o.outgoing_granger, 0) = 0 AND COALESCE(i.incoming_granger, 0) > 0 THEN 'SINK'
        WHEN COALESCE(o.outgoing_granger, 0) > 0 AND COALESCE(i.incoming_granger, 0) > 0 THEN 'CONDUIT'
        ELSE 'ISOLATED'
    END AS role
FROM causal_stats o
FULL OUTER JOIN incoming_stats i ON o.entity_id = i.entity_id
    AND o.signal_id = i.signal_id
    AND o.window_id = i.window_id;

-- Causal strength for each pair
CREATE OR REPLACE TABLE causal_staged AS
SELECT
    g.entity_id,
    g.source_signal,
    g.target_signal,
    g.window_id,
    g.granger_f_stat,
    g.granger_p_value,
    te.transfer_entropy,
    -- Determine causal direction
    CASE
        WHEN g.granger_significant AND te.transfer_entropy > 0.1 THEN 'source_to_target'
        WHEN EXISTS (
            SELECT 1 FROM causal_granger_flat g2
            WHERE g2.entity_id = g.entity_id
            AND g2.source_signal = g.target_signal
            AND g2.target_signal = g.source_signal
            AND g2.window_id = g.window_id
            AND g2.granger_significant
        ) THEN 'bidirectional'
        WHEN NOT g.granger_significant AND COALESCE(te.transfer_entropy, 0) < 0.05 THEN 'none'
        ELSE 'weak'
    END AS causal_direction,
    -- Causal strength (combined metric)
    CASE
        WHEN g.granger_p_value < 0.001 THEN te.transfer_entropy * 2
        WHEN g.granger_p_value < 0.01 THEN te.transfer_entropy * 1.5
        WHEN g.granger_p_value < 0.05 THEN te.transfer_entropy
        ELSE te.transfer_entropy * 0.5
    END AS causal_strength,
    -- Role of source signal
    rs.role AS source_role,
    -- Role of target signal
    rt.role AS target_role
FROM causal_granger_flat g
LEFT JOIN causal_transfer_entropy_flat te USING (entity_id, source_signal, target_signal, window_id)
LEFT JOIN causal_roles rs ON g.entity_id = rs.entity_id
    AND g.source_signal = rs.signal_id
    AND g.window_id = rs.window_id
LEFT JOIN causal_roles rt ON g.entity_id = rt.entity_id
    AND g.target_signal = rt.signal_id
    AND g.window_id = rt.window_id;
