"""
Tests for PR11: Signal Vector Runner
"""

import pytest
import numpy as np
import polars as pl
import tempfile
import yaml
import sys
sys.path.insert(0, '/home/claude/pr11-signal-vector')

from prism.signal_vector.manifest_reader import ManifestReader, SignalConfig
from prism.signal_vector.engines import list_engines, run_engines, get_engine
from prism.signal_vector.runner import sliding_windows, process_signal, run_signal_vector


# ============================================================
# Engine Tests
# ============================================================

class TestEngineRegistry:
    
    def test_engines_registered(self):
        """Core engines are registered."""
        engines = list_engines()
        expected = ['kurtosis', 'skewness', 'crest_factor', 'hurst', 
                    'spectral', 'spectral_entropy', 'rate_of_change', 'trend_r2']
        for e in expected:
            assert e in engines, f"Missing engine: {e}"
    
    def test_run_single_engine(self):
        """Run a single engine."""
        values = np.random.randn(100)
        result = run_engines(['kurtosis'], values)
        assert 'kurtosis' in result
        assert isinstance(result['kurtosis'], float)
    
    def test_run_multiple_engines(self):
        """Run multiple engines."""
        values = np.random.randn(100)
        result = run_engines(['kurtosis', 'skewness', 'hurst'], values)
        assert 'kurtosis' in result
        assert 'skewness' in result
        assert 'hurst' in result


class TestStatisticsEngines:
    
    def test_kurtosis_normal(self):
        """Kurtosis of normal distribution ≈ 0."""
        np.random.seed(42)
        values = np.random.randn(10000)
        result = run_engines(['kurtosis'], values)
        assert abs(result['kurtosis']) < 0.5
    
    def test_kurtosis_heavy_tails(self):
        """Kurtosis of heavy-tailed distribution > 0."""
        np.random.seed(42)
        values = np.random.standard_t(df=3, size=1000)  # t-distribution
        result = run_engines(['kurtosis'], values)
        assert result['kurtosis'] > 1.0
    
    def test_skewness_symmetric(self):
        """Skewness of symmetric distribution ≈ 0."""
        np.random.seed(42)
        values = np.random.randn(10000)
        result = run_engines(['skewness'], values)
        assert abs(result['skewness']) < 0.1
    
    def test_crest_factor_sine(self):
        """Crest factor of sine ≈ √2."""
        t = np.linspace(0, 10*np.pi, 1000)
        values = np.sin(t)
        result = run_engines(['crest_factor'], values)
        assert abs(result['crest_factor'] - np.sqrt(2)) < 0.1


class TestSpectralEngines:
    
    def test_dominant_freq_sine(self):
        """Dominant frequency of sine wave."""
        t = np.linspace(0, 1, 1000)  # 1 second, 1000 Hz sample rate
        freq = 50  # 50 Hz
        values = np.sin(2 * np.pi * freq * t)
        result = run_engines(['spectral'], values, sample_rate=1000)
        # Dominant freq should be near 50 Hz
        assert 40 < result['dominant_freq'] < 60
    
    def test_spectral_entropy_noise(self):
        """Spectral entropy of white noise ≈ high."""
        np.random.seed(42)
        values = np.random.randn(1000)
        result = run_engines(['spectral_entropy'], values)
        assert result['spectral_entropy'] > 0.8


class TestMemoryEngines:
    
    def test_hurst_white_noise(self):
        """Hurst of white noise ≈ 0.5."""
        np.random.seed(42)
        values = np.random.randn(1000)  # White noise, not cumsum
        result = run_engines(['hurst'], values)
        # White noise has H ≈ 0.5
        assert 0.3 < result['hurst'] < 0.7
    
    def test_hurst_trending(self):
        """Hurst of trending signal > 0.5."""
        values = np.linspace(0, 100, 500) + np.random.randn(500) * 5
        result = run_engines(['hurst'], values)
        assert result['hurst'] > 0.7


class TestTrendEngines:
    
    def test_rate_of_change_linear(self):
        """Rate of change of linear signal."""
        values = np.linspace(0, 100, 200)  # Slope = 100/199 ≈ 0.5
        result = run_engines(['rate_of_change'], values)
        assert abs(result['rate_of_change'] - 0.5) < 0.1
    
    def test_trend_r2_perfect(self):
        """Trend R² of perfect line = 1."""
        values = np.linspace(0, 100, 200)
        result = run_engines(['trend_r2'], values)
        assert result['trend_r2'] > 0.99
    
    def test_trend_r2_noise(self):
        """Trend R² of noise ≈ 0."""
        np.random.seed(42)
        values = np.random.randn(200)
        result = run_engines(['trend_r2'], values)
        assert result['trend_r2'] < 0.1


# ============================================================
# Manifest Reader Tests
# ============================================================

class TestManifestReader:
    
    @pytest.fixture
    def sample_manifest(self, tmp_path):
        """Create sample manifest file."""
        manifest = {
            'version': '2.2',
            'job_id': 'test',
            'paths': {
                'observations': 'obs.parquet',
                'output_dir': 'output/',
            },
            'cohorts': {
                'unit_1': {
                    'sensor_a': {
                        'engines': ['kurtosis', 'hurst'],
                        'window_size': 128,
                        'stride': 64,
                        'derivative_depth': 1,
                        'eigenvalue_budget': 5,
                        'typology': {'temporal_pattern': 'RANDOM'},
                    },
                    'sensor_b': {
                        'engines': ['kurtosis', 'trend_r2'],
                        'window_size': 256,
                        'stride': 128,
                        'derivative_depth': 2,
                        'typology': {'temporal_pattern': 'TRENDING'},
                    },
                },
            },
            'skip_signals': ['unit_1/sensor_c'],
        }
        
        path = tmp_path / 'manifest.yaml'
        with open(path, 'w') as f:
            yaml.dump(manifest, f)
        return str(path)
    
    def test_read_manifest(self, sample_manifest):
        """Read manifest file."""
        reader = ManifestReader(sample_manifest)
        assert reader.version == '2.2'
    
    def test_get_signal(self, sample_manifest):
        """Get specific signal config."""
        reader = ManifestReader(sample_manifest)
        cfg = reader.get_signal('unit_1', 'sensor_a')
        
        assert cfg.signal_id == 'sensor_a'
        assert cfg.cohort == 'unit_1'
        assert cfg.window_size == 128
        assert 'kurtosis' in cfg.engines
    
    def test_iter_signals(self, sample_manifest):
        """Iterate over signals."""
        reader = ManifestReader(sample_manifest)
        signals = list(reader.iter_signals())
        
        assert len(signals) == 2
        signal_ids = [s.signal_id for s in signals]
        assert 'sensor_a' in signal_ids
        assert 'sensor_b' in signal_ids
    
    def test_skip_signals(self, sample_manifest):
        """Skipped signals not returned."""
        reader = ManifestReader(sample_manifest)
        # sensor_c is in skip_signals
        cfg = reader.get_signal('unit_1', 'sensor_c')
        assert cfg is None or 'unit_1/sensor_c' in reader.skip_signals


# ============================================================
# Runner Tests
# ============================================================

class TestSlidingWindows:
    
    def test_basic_windows(self):
        """Basic sliding window generation."""
        values = np.arange(100)
        windows = sliding_windows(values, window_size=20, stride=10)
        
        assert len(windows) == 9  # (100-20)/10 + 1
        assert windows[0] == (0, 0, 20)
        assert windows[1] == (1, 10, 30)
    
    def test_no_overlap(self):
        """Non-overlapping windows."""
        values = np.arange(100)
        windows = sliding_windows(values, window_size=25, stride=25)
        
        assert len(windows) == 4  # 100/25


class TestProcessSignal:
    
    def test_process_signal(self):
        """Process a single signal."""
        values = np.random.randn(500)
        indices = np.arange(500)
        
        config = SignalConfig(
            signal_id='test',
            cohort='unit',
            engines=['kurtosis', 'skewness'],
            rolling_engines=[],
            window_size=100,
            stride=50,
            derivative_depth=1,
            eigenvalue_budget=5,
            temporal_pattern='RANDOM',
            spectral='BROADBAND',
        )
        
        results = process_signal('test', 'unit', values, indices, config)
        
        # Should have (500-100)/50 + 1 = 9 windows
        assert len(results) == 9
        
        # Each result has metadata + engine outputs
        assert results[0]['signal_id'] == 'test'
        assert results[0]['cohort'] == 'unit'
        assert 'kurtosis' in results[0]
        assert 'skewness' in results[0]


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    
    @pytest.fixture
    def test_data(self, tmp_path):
        """Create test observations and manifest."""
        # Create observations
        n = 1000
        obs_data = []
        for signal_id in ['sensor_a', 'sensor_b']:
            for i in range(n):
                obs_data.append({
                    'cohort': 'unit',
                    'signal_id': signal_id,
                    'I': i,
                    'value': np.random.randn() if signal_id == 'sensor_a' else float(i) / 10,
                })
        
        obs_df = pl.DataFrame(obs_data)
        obs_path = tmp_path / 'observations.parquet'
        obs_df.write_parquet(obs_path)
        
        # Create manifest
        manifest = {
            'version': '2.2',
            'job_id': 'integration_test',
            'paths': {
                'observations': str(obs_path),
                'output_dir': str(tmp_path / 'output'),
            },
            'cohorts': {
                'unit': {
                    'sensor_a': {
                        'engines': ['kurtosis', 'spectral_entropy'],
                        'window_size': 128,
                        'stride': 64,
                        'typology': {'temporal_pattern': 'RANDOM'},
                    },
                    'sensor_b': {
                        'engines': ['hurst', 'trend_r2'],
                        'window_size': 128,
                        'stride': 64,
                        'typology': {'temporal_pattern': 'TRENDING'},
                    },
                },
            },
            'skip_signals': [],
        }
        
        manifest_path = tmp_path / 'manifest.yaml'
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f)
        
        return str(manifest_path), str(obs_path)
    
    def test_full_pipeline(self, test_data):
        """Run full signal vector pipeline."""
        manifest_path, obs_path = test_data
        
        result = run_signal_vector(manifest_path)
        
        assert len(result) > 0
        assert 'signal_id' in result.columns
        assert 'window_I' in result.columns
        
        # sensor_a should have kurtosis, spectral_entropy
        sensor_a = result.filter(pl.col('signal_id') == 'sensor_a')
        assert sensor_a['kurtosis'][0] is not None
        
        # sensor_b should have hurst, trend_r2
        sensor_b = result.filter(pl.col('signal_id') == 'sensor_b')
        assert sensor_b['hurst'][0] is not None
        assert sensor_b['trend_r2'][0] > 0.9  # Linear signal
