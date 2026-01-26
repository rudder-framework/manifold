-- 002_periodicity.sql
-- Compute spectral features using PRISM fft engine
-- Requires: prism_fft() UDF registered

CREATE OR REPLACE TABLE typology_spectral AS
SELECT
    entity_id,
    signal_id,
    window_id,
    -- FFT via PRISM engine
    prism_fft(ARRAY_AGG(y ORDER BY I)) AS fft_result
FROM calculus_output
WHERE y IS NOT NULL
GROUP BY entity_id, signal_id, window_id;

-- Unnest spectral results
CREATE OR REPLACE TABLE typology_spectral_flat AS
SELECT
    entity_id,
    signal_id,
    window_id,
    fft_result.centroid AS spectral_centroid,
    fft_result.bandwidth AS spectral_bandwidth,
    fft_result.dominant_freq AS dominant_frequency,
    fft_result.rolloff AS spectral_rolloff,
    fft_result.low_high_ratio AS low_high_ratio,
    fft_result.total_power AS total_power,
    -- Periodicity classification
    CASE
        WHEN fft_result.bandwidth < 0.05 THEN 'periodic'
        WHEN fft_result.low_high_ratio > 10 THEN 'low_frequency_dominant'
        WHEN fft_result.low_high_ratio < 0.1 THEN 'high_frequency_dominant'
        ELSE 'broadband'
    END AS periodicity_class
FROM typology_spectral;
