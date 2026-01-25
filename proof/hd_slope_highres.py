"""
High-Resolution HD Slope Analysis
==================================

Decomposes hd_slope into:
1. Per-sensor HD slopes (21 sensors)
2. Per-regime breakdown (6 regimes in FD002)
3. Multi-scale windows (5 to 100 cycles)
4. Second-order features (acceleration, curvature)
5. Dominant degradation channel identification
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CACHE_DIR = Path("/var/folders/2v/f2fc1dgd24x8rcn0l72b73sw0000gn/T/cmapss_data")
COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SENSOR_COLS = [f's{i}' for i in range(1, 22)]

# Multi-scale windows for high-res analysis
WINDOWS = [5, 10, 15, 20, 30, 50, 75, 100]

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SensorHDProfile:
    """Per-sensor healthy distance profile."""
    sensor_id: str
    slopes: Dict[int, float]      # window -> slope
    acceleration: float           # d²/dt²
    curvature: float             # change in slope direction
    contribution_pct: float       # % of total HD from this sensor
    regime_slopes: Dict[int, float]  # regime_id -> slope


@dataclass
class HighResHDProfile:
    """Complete high-resolution HD profile for an entity."""
    entity_id: int
    cycle: int

    # Aggregate
    hd_current: float
    hd_slopes: Dict[int, float]   # window -> slope
    hd_acceleration: float
    hd_curvature: float
    hd_jerk: float

    # Per-sensor breakdown
    sensor_profiles: Dict[str, SensorHDProfile]

    # Dominant channel
    dominant_sensor: str
    dominant_contribution: float

    # Regime context
    current_regime: int
    regime_hd: float


# ============================================================
# HIGH-RES HD COMPUTATION
# ============================================================

class HighResHDAnalyzer:
    """
    High-resolution healthy distance analysis.

    Computes:
    1. Per-sensor z-score trajectories
    2. Multi-scale slope analysis
    3. Second-order dynamics (acceleration, curvature)
    4. Dominant degradation channel identification
    """

    def __init__(self, n_regimes: int = 6):
        self.n_regimes = n_regimes
        self.regime_km = None
        self.regime_baselines = {}
        self.fitted = False

    def fit(self, train_df: pl.DataFrame) -> 'HighResHDAnalyzer':
        """Learn regime structure and baselines from training data."""
        print("=" * 60)
        print("HIGH-RES HD ANALYZER - FIT")
        print("=" * 60)

        # Step 1: Cluster operating conditions
        print(f"\n[1/2] Clustering into {self.n_regimes} regimes...")
        op_data = train_df.select(["op1", "op2"]).to_numpy()
        self.regime_km = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        self.regime_km.fit(op_data)

        regime_labels = self.regime_km.predict(op_data)
        train_df = train_df.with_columns(pl.Series("regime_id", regime_labels))

        # Show regime distribution
        for r in range(self.n_regimes):
            count = (regime_labels == r).sum()
            print(f"  Regime {r}: {count:,} samples")

        # Step 2: Compute per-regime baselines for each sensor
        print(f"\n[2/2] Computing per-regime baselines...")

        for regime in range(self.n_regimes):
            self.regime_baselines[regime] = {}
            regime_data = train_df.filter(pl.col("regime_id") == regime)

            # Sample from healthy portion (first 20% of each unit's life)
            healthy_samples = []
            for unit in regime_data["unit"].unique().to_list():
                unit_data = regime_data.filter(pl.col("unit") == unit).sort("cycle")
                n_healthy = max(1, len(unit_data) // 5)
                healthy_samples.append(unit_data.head(n_healthy))

            if healthy_samples:
                healthy_df = pl.concat(healthy_samples)

                for sensor in SENSOR_COLS:
                    vals = healthy_df[sensor].drop_nulls().to_numpy()
                    if len(vals) > 10:
                        self.regime_baselines[regime][sensor] = {
                            'mean': np.mean(vals),
                            'std': np.std(vals) + 1e-10,
                            'median': np.median(vals),
                            'q25': np.percentile(vals, 25),
                            'q75': np.percentile(vals, 75),
                        }

            print(f"  Regime {regime}: {len(self.regime_baselines[regime])} sensors baselined")

        self.fitted = True
        print("\n[OK] Fitting complete!")
        return self

    def compute_per_sensor_hd(
        self,
        row: dict,
        regime: int
    ) -> Dict[str, float]:
        """Compute z-score healthy distance for each sensor."""
        sensor_hd = {}
        baselines = self.regime_baselines.get(regime, {})

        for sensor in SENSOR_COLS:
            if sensor in baselines:
                val = row.get(sensor)
                if val is not None:
                    baseline = baselines[sensor]
                    # Z-score distance
                    z = abs(val - baseline['mean']) / baseline['std']
                    sensor_hd[sensor] = z

        return sensor_hd

    def compute_slopes(
        self,
        history: List[float],
        windows: List[int] = None
    ) -> Dict[int, float]:
        """Compute slopes over multiple windows."""
        windows = windows or WINDOWS
        slopes = {}
        n = len(history)

        for W in windows:
            if n >= W:
                recent = history[-W:]
                x = np.arange(W)
                try:
                    coeffs = np.polyfit(x, recent, 1)
                    slopes[W] = coeffs[0]
                except:
                    slopes[W] = 0.0
            else:
                slopes[W] = np.nan

        return slopes

    def compute_second_order(
        self,
        history: List[float],
        window: int = 30
    ) -> Tuple[float, float, float]:
        """Compute acceleration, curvature, jerk."""
        n = len(history)

        if n < window:
            return 0.0, 0.0, 0.0

        recent = history[-window:]

        # Acceleration: slope of slope
        # Split into thirds and compute slope of each
        third = window // 3
        slope1 = np.polyfit(np.arange(third), recent[:third], 1)[0]
        slope2 = np.polyfit(np.arange(third), recent[third:2*third], 1)[0]
        slope3 = np.polyfit(np.arange(third), recent[2*third:], 1)[0]

        acceleration = (slope3 - slope1) / (2 * third)

        # Curvature: change in slope direction
        curvature = slope3 - 2*slope2 + slope1

        # Jerk: rate of change of acceleration
        if n >= 2 * window:
            older = history[-2*window:-window]
            old_slopes = [
                np.polyfit(np.arange(third), older[:third], 1)[0],
                np.polyfit(np.arange(third), older[third:2*third], 1)[0],
                np.polyfit(np.arange(third), older[2*third:], 1)[0],
            ]
            old_acceleration = (old_slopes[2] - old_slopes[0]) / (2 * third)
            jerk = acceleration - old_acceleration
        else:
            jerk = 0.0

        return acceleration, curvature, jerk

    def analyze_entity(
        self,
        entity_df: pl.DataFrame,
    ) -> List[dict]:
        """
        Compute high-resolution HD features for a single entity.

        Returns one row per cycle with all high-res features.
        """
        rows = entity_df.sort("cycle").to_dicts()

        # History tracking
        aggregate_hd_history = []
        sensor_hd_histories = {s: [] for s in SENSOR_COLS}

        results = []

        for i, row in enumerate(rows):
            # Get regime
            op_data = np.array([[row['op1'], row['op2']]])
            regime = self.regime_km.predict(op_data)[0]

            # Compute per-sensor HD
            sensor_hd = self.compute_per_sensor_hd(row, regime)

            # Update histories
            for sensor, hd_val in sensor_hd.items():
                sensor_hd_histories[sensor].append(hd_val)

            # Aggregate HD (mean of all available sensors)
            if sensor_hd:
                agg_hd = np.mean(list(sensor_hd.values()))
            else:
                agg_hd = 0.0
            aggregate_hd_history.append(agg_hd)

            # Compute features
            result = {
                'unit': row['unit'],
                'cycle': row['cycle'],
                'regime_id': regime,
                'hd_current': agg_hd,
            }

            # 1. Aggregate HD slopes (multi-scale)
            agg_slopes = self.compute_slopes(aggregate_hd_history)
            for W, slope in agg_slopes.items():
                result[f'hd_slope_{W}'] = slope

            # 2. Per-sensor slopes (for key sensors, use window=30)
            for sensor in SENSOR_COLS:
                hist = sensor_hd_histories[sensor]
                if len(hist) >= 30:
                    slope = np.polyfit(np.arange(30), hist[-30:], 1)[0]
                    result[f'{sensor}_hd_slope_30'] = slope
                else:
                    result[f'{sensor}_hd_slope_30'] = np.nan

            # 3. Second-order features
            acc, curv, jerk = self.compute_second_order(aggregate_hd_history)
            result['hd_acceleration'] = acc
            result['hd_curvature'] = curv
            result['hd_jerk'] = jerk

            # 4. Dominant degradation channel
            if sensor_hd:
                sorted_sensors = sorted(sensor_hd.items(), key=lambda x: -x[1])
                result['dominant_sensor'] = sorted_sensors[0][0]
                result['dominant_hd'] = sorted_sensors[0][1]
                total_hd = sum(sensor_hd.values())
                if total_hd > 0:
                    result['dominant_contribution'] = sorted_sensors[0][1] / total_hd
                else:
                    result['dominant_contribution'] = 0

                # Top 3 contributors
                for j, (sensor, hd_val) in enumerate(sorted_sensors[:3]):
                    result[f'top{j+1}_sensor'] = sensor
                    result[f'top{j+1}_hd'] = hd_val

            # 5. Slope dispersion (are all sensors degrading uniformly?)
            sensor_slopes = []
            for sensor in SENSOR_COLS:
                hist = sensor_hd_histories[sensor]
                if len(hist) >= 20:
                    slope = np.polyfit(np.arange(20), hist[-20:], 1)[0]
                    sensor_slopes.append(slope)

            if sensor_slopes:
                result['slope_mean'] = np.mean(sensor_slopes)
                result['slope_std'] = np.std(sensor_slopes)
                result['slope_max'] = np.max(sensor_slopes)
                result['slope_min'] = np.min(sensor_slopes)
                result['slope_range'] = np.max(sensor_slopes) - np.min(sensor_slopes)

            results.append(result)

        return results

    def analyze_fleet(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Analyze all entities in fleet."""
        if not self.fitted:
            raise ValueError("Must call fit() first")

        print("\n" + "=" * 60)
        print("ANALYZING FLEET")
        print("=" * 60)

        all_results = []
        units = df["unit"].unique().to_list()

        for i, unit in enumerate(units):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(units)} entities...")

            unit_df = df.filter(pl.col("unit") == unit)
            results = self.analyze_entity(unit_df)
            all_results.extend(results)

        print(f"  Complete! {len(all_results):,} rows generated")

        return pl.DataFrame(all_results)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_slope_contributions(highres_df: pl.DataFrame) -> pl.DataFrame:
    """Analyze which sensors contribute most to degradation."""

    # Get sensor slope columns
    slope_cols = [c for c in highres_df.columns if c.endswith('_hd_slope_30')]

    # For each entity at end of life, identify top contributors
    last_rows = highres_df.group_by("unit").agg(
        pl.col("cycle").max().alias("last_cycle")
    )

    final_states = highres_df.join(last_rows, on="unit").filter(
        pl.col("cycle") == pl.col("last_cycle")
    )

    # Compute mean slope per sensor across fleet
    sensor_impact = {}
    for col in slope_cols:
        sensor = col.replace('_hd_slope_30', '')
        vals = final_states[col].drop_nulls().to_numpy()
        if len(vals) > 0:
            sensor_impact[sensor] = {
                'mean_slope': np.mean(vals),
                'std_slope': np.std(vals),
                'max_slope': np.max(vals),
                'positive_pct': (vals > 0).mean() * 100,
            }

    # Sort by mean slope
    sorted_impact = sorted(sensor_impact.items(), key=lambda x: -x[1]['mean_slope'])

    print("\n" + "=" * 60)
    print("SENSOR DEGRADATION CONTRIBUTION RANKING")
    print("=" * 60)
    print(f"{'Sensor':<10} {'Mean Slope':>12} {'Std':>10} {'Max':>10} {'%Positive':>10}")
    print("-" * 52)

    for sensor, stats in sorted_impact[:15]:
        print(f"{sensor:<10} {stats['mean_slope']:>12.6f} {stats['std_slope']:>10.6f} "
              f"{stats['max_slope']:>10.6f} {stats['positive_pct']:>9.1f}%")

    return pl.DataFrame([
        {'sensor': s, **stats} for s, stats in sorted_impact
    ])


def compare_catastrophic_vs_normal(
    highres_df: pl.DataFrame,
    catastrophic_units: List[int],
) -> None:
    """Compare high-res HD features between catastrophic and normal engines."""

    print("\n" + "=" * 60)
    print("CATASTROPHIC vs NORMAL - HIGH-RES HD COMPARISON")
    print("=" * 60)

    # Get final state for each unit
    last_rows = highres_df.group_by("unit").agg(
        pl.col("cycle").max().alias("last_cycle")
    )
    final = highres_df.join(last_rows, on="unit").filter(
        pl.col("cycle") == pl.col("last_cycle")
    )

    # Split
    cat_final = final.filter(pl.col("unit").is_in(catastrophic_units))
    norm_final = final.filter(~pl.col("unit").is_in(catastrophic_units))

    print(f"\nCatastrophic engines: {len(cat_final)}")
    print(f"Normal engines: {len(norm_final)}")

    # Compare key features
    features = [
        'hd_current',
        'hd_slope_30', 'hd_slope_50',
        'hd_acceleration', 'hd_curvature',
        'slope_mean', 'slope_std', 'slope_range',
    ]

    print(f"\n{'Feature':<20} {'Catastrophic':>15} {'Normal':>15} {'Ratio':>10}")
    print("-" * 60)

    for feat in features:
        if feat in final.columns:
            cat_val = cat_final[feat].mean()
            norm_val = norm_final[feat].mean()
            ratio = cat_val / norm_val if norm_val != 0 else float('inf')
            print(f"{feat:<20} {cat_val:>15.4f} {norm_val:>15.4f} {ratio:>10.2f}x")

    # Early detection - compare at 20% of life
    print("\n--- EARLY DETECTION (20% of life) ---")

    early_pct = 0.20
    early_results = []

    for unit in highres_df["unit"].unique().to_list():
        unit_df = highres_df.filter(pl.col("unit") == unit)
        n_cycles = len(unit_df)
        early_idx = int(n_cycles * early_pct)
        if early_idx > 0:
            early_row = unit_df[early_idx].to_dicts()[0]
            early_row['is_catastrophic'] = unit in catastrophic_units
            early_results.append(early_row)

    early_df = pl.DataFrame(early_results)
    cat_early = early_df.filter(pl.col("is_catastrophic"))
    norm_early = early_df.filter(~pl.col("is_catastrophic"))

    print(f"\n{'Feature':<20} {'Catastrophic':>15} {'Normal':>15} {'Ratio':>10}")
    print("-" * 60)

    for feat in ['hd_slope_30', 'hd_acceleration', 'slope_mean', 'slope_range']:
        if feat in early_df.columns:
            cat_val = cat_early[feat].mean()
            norm_val = norm_early[feat].mean()
            ratio = cat_val / norm_val if norm_val != 0 else float('inf')
            print(f"{feat:<20} {cat_val:>15.4f} {norm_val:>15.4f} {ratio:>10.2f}x")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run high-resolution HD analysis on FD002."""

    print("=" * 60)
    print("HIGH-RESOLUTION HD SLOPE ANALYSIS - FD002")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_pdf = pd.read_csv(CACHE_DIR / "train_FD002.txt", sep=r'\s+', header=None, names=COLS)
    train_df = pl.from_pandas(train_pdf)
    print(f"Loaded: {len(train_df):,} rows, {train_df['unit'].n_unique()} engines")

    # Initialize analyzer
    analyzer = HighResHDAnalyzer(n_regimes=6)

    # Fit on training data
    analyzer.fit(train_df)

    # Analyze fleet
    highres_df = analyzer.analyze_fleet(train_df)

    # Save results
    output_path = Path("/Users/jasonrudder/prism-mac/data/hd_highres.parquet")
    highres_df.write_parquet(output_path)
    print(f"\nSaved to {output_path}")

    # Analysis 1: Sensor contribution ranking
    sensor_impact = analyze_slope_contributions(highres_df)

    # Analysis 2: Catastrophic vs Normal comparison
    catastrophic_units = [25, 47, 107, 169]
    compare_catastrophic_vs_normal(highres_df, catastrophic_units)

    # Print sample of high-res features
    print("\n" + "=" * 60)
    print("SAMPLE HIGH-RES FEATURES (last cycle per unit)")
    print("=" * 60)

    sample = highres_df.filter(pl.col("unit").is_in([25, 47, 100, 200])).group_by("unit").agg(
        pl.col("cycle").max().alias("last_cycle")
    ).join(highres_df, on="unit").filter(pl.col("cycle") == pl.col("last_cycle"))

    print(sample.select([
        "unit", "cycle", "hd_current",
        "hd_slope_30", "hd_slope_50",
        "hd_acceleration", "dominant_sensor"
    ]))

    return highres_df


if __name__ == "__main__":
    highres_df = main()
