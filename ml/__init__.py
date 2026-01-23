"""
PRISM ML Module
===============

Machine learning pipelines for PRISM behavioral geometry features.

Entry Points:
    - ml.entry_points.train: Train ML models
    - ml.entry_points.predict: Generate predictions
    - ml.entry_points.features: Feature generation
    - ml.entry_points.ablation: Feature ablation studies
    - ml.entry_points.benchmark: Model benchmarking
    - ml.entry_points.baseline: Baseline comparisons

Data Structure:
    - ml/data/features/: Feature parquet files
    - ml/data/models/: Trained model pickles
    - ml/data/results/: Prediction results
"""

from pathlib import Path

ML_ROOT = Path(__file__).parent
DATA_DIR = ML_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
REPORTS_DIR = ML_ROOT / "reports"

__all__ = [
    "ML_ROOT",
    "DATA_DIR",
    "FEATURES_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "REPORTS_DIR",
]
