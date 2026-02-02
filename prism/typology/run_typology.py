"""
ORTHON Typology Pipeline Orchestrator

Connects the validation gate to Level 1 stationarity (and future Level 2+).
This is the single entry point for running typology on observations.

Pipeline:
    1. Validate observations (First Gate)
    2. Auto-repair if requested
    3. Level 1: Stationarity test per signal
    4. Level 2: Signal classification (future)
    5. Write typology.parquet

Usage:
    from prism.typology import run_typology
    result = run_typology("data/observations.parquet")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from prism.typology.data_validation import (
    validate_observations,
    ValidationResult,
    ValidationStatus,
)
from prism.typology.level1_stationarity import (
    test_stationarity,
    StationarityResult,
)


@dataclass
class TypologyResult:
    """Complete typology pipeline result."""

    # Pipeline status
    status: str  # "success", "validation_failed", "error"
    file_path: str

    # Validation
    validation: Optional[ValidationResult] = None

    # Level 1 results per signal
    level1_results: Dict[str, StationarityResult] = field(default_factory=dict)

    # Output path (if written)
    output_path: Optional[str] = None

    # Error message (if any)
    error: Optional[str] = None

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"ORTHON Typology Pipeline: {self.status.upper()}",
            f"File: {self.file_path}",
            "=" * 60,
        ]

        if self.validation:
            lines.append(f"\nValidation: {self.validation.status.value}")
            lines.append(f"  Signals: {self.validation.n_signals}")
            lines.append(f"  Rows: {self.validation.n_rows:,}")

        if self.level1_results:
            lines.append(f"\nLevel 1 Stationarity ({len(self.level1_results)} signals):")
            for sig_id, result in self.level1_results.items():
                lines.append(
                    f"  {sig_id}: {result.stationarity_type.value} "
                    f"(conf={result.confidence.value}, "
                    f"var_stable={result.variance_stable}, "
                    f"mean_stable={result.mean_stable})"
                )

        if self.output_path:
            lines.append(f"\nOutput: {self.output_path}")

        if self.error:
            lines.append(f"\nError: {self.error}")

        return "\n".join(lines)

    def to_dataframe(self):
        """Convert Level 1 results to a Polars DataFrame for typology.parquet."""
        import polars as pl

        if not self.level1_results:
            return None

        rows = []
        for signal_id, r in self.level1_results.items():
            row = {"signal_id": signal_id}
            row.update(r.to_dict())
            rows.append(row)

        return pl.DataFrame(rows)


def run_typology(
    file_path: str,
    output_dir: Optional[str] = None,
    auto_repair: bool = False,
    verbose: bool = True,
) -> TypologyResult:
    """
    Run the complete typology pipeline on an observations file.

    Pipeline:
        1. Validate observations (First Gate)
        2. Auto-repair if requested and validation failed
        3. Level 1: Stationarity test per signal
        4. Write typology.parquet to output_dir

    Args:
        file_path: Path to observations file (.parquet, .csv, .npz)
        output_dir: Where to write typology.parquet (default: same as input)
        auto_repair: Attempt to auto-repair validation failures
        verbose: Print progress

    Returns:
        TypologyResult with validation and Level 1 results
    """
    import polars as pl

    result = TypologyResult(
        status="error",
        file_path=file_path,
    )

    if verbose:
        print("=" * 60)
        print("ORTHON Typology Pipeline")
        print("=" * 60)
        print(f"\nInput: {file_path}")

    # --- 1. Validation Gate ---
    if verbose:
        print("\n[1/3] Validating observations...")

    validation = validate_observations(file_path, verbose=False)
    result.validation = validation

    if validation.has_failures():
        if auto_repair:
            if verbose:
                print("  Validation failed. Attempting auto-repair...")
            # Try repair
            from prism.typology.data_validation import repair_observations
            repaired_df, repair_result = repair_observations(file_path, verbose=verbose)

            if repaired_df is not None:
                # Re-validate the repaired data
                # For now, we'll trust the repair
                df = repaired_df
                if verbose:
                    print("  Auto-repair successful!")
            else:
                result.status = "validation_failed"
                result.error = "Validation failed and auto-repair could not fix issues"
                if verbose:
                    print(f"\n{result}")
                return result
        else:
            result.status = "validation_failed"
            result.error = "Validation failed. Set auto_repair=True to attempt fixes."
            if verbose:
                print(f"\nValidation FAILED:")
                print(validation)
            return result
    else:
        # Load the validated data
        path = Path(file_path)
        if path.suffix == '.parquet':
            df = pl.read_parquet(file_path)
        elif path.suffix == '.csv':
            df = pl.read_csv(file_path)
        else:
            result.status = "error"
            result.error = f"Unsupported file format for Level 1: {path.suffix}"
            return result

    if verbose:
        print(f"  Validation: {validation.status.value}")
        print(f"  Signals: {validation.n_signals}")
        print(f"  Rows: {validation.n_rows:,}")

    # --- 2. Level 1: Stationarity per signal ---
    if verbose:
        print(f"\n[2/3] Running Level 1 Stationarity ({validation.n_signals} signals)...")

    signal_ids = df['signal_id'].unique().to_list()

    for i, signal_id in enumerate(signal_ids):
        # Get signal values
        signal_df = df.filter(pl.col('signal_id') == signal_id).sort('I')
        values = signal_df['value'].to_numpy()

        # Run stationarity test
        l1_result = test_stationarity(values, verbose=False)
        result.level1_results[signal_id] = l1_result

        if verbose:
            status_char = "✓" if l1_result.is_stationary else "~"
            print(f"  {status_char} {signal_id}: {l1_result.stationarity_type.value}")

    # --- 3. Write typology.parquet ---
    if output_dir is None:
        output_dir = str(Path(file_path).parent)

    output_path = Path(output_dir) / "typology.parquet"

    if verbose:
        print(f"\n[3/3] Writing typology.parquet...")

    typology_df = result.to_dataframe()
    if typology_df is not None:
        typology_df.write_parquet(str(output_path))
        result.output_path = str(output_path)
        if verbose:
            print(f"  Written: {output_path}")

    result.status = "success"

    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)

    return result


def run_typology_on_array(
    values: np.ndarray,
    signal_id: str = "signal_0",
    verbose: bool = False,
) -> StationarityResult:
    """
    Run Level 1 typology on a raw numpy array.

    Convenience function for testing/benchmarking without file I/O.

    Args:
        values: 1D numpy array of signal values
        signal_id: Name for the signal
        verbose: Print results

    Returns:
        StationarityResult
    """
    return test_stationarity(values, verbose=verbose)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = run_typology(sys.argv[1], verbose=True)
    else:
        print("Usage: python -m prism.typology.run_typology <observations.parquet>")
