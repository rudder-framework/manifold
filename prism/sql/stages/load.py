"""
Load Stage Orchestrator

PURE: Loads 00_load.sql, creates base views.
NO computation. NO inline SQL.

Handles index column mapping from domain config or auto-detection.
"""

from pathlib import Path
from typing import Optional
import yaml

from .base import StageOrchestrator


class LoadStage(StageOrchestrator):
    """Load observations and create base view."""

    SQL_FILE = '00_load.sql'

    VIEWS = [
        'v_base',
        'v_schema_validation',
        'v_signal_inventory',
        'v_data_quality',
    ]

    DEPENDS_ON = []  # First stage, no dependencies

    # Default column mapping: source -> canonical
    # Extended to handle common real-world dataset conventions
    DEFAULT_INDEX_COLUMNS = {
        'timestamp': 'I',
        'index': 'I',
        'time': 'I',
        'cycle': 'I',        # NASA CMAPSS
        'sample': 'I',       # CWRU Bearing
        'sample_index': 'I',
        'depth': 'I',        # Well logging
        't': 'I',
    }

    DEFAULT_VALUE_COLUMNS = {
        'value': 'y',
        'y': 'y',
    }

    def __init__(self, conn, domain_config: Optional[dict] = None):
        """
        Initialize LoadStage.

        Args:
            conn: DuckDB connection
            domain_config: Optional domain configuration dict with index settings
        """
        super().__init__(conn)
        self.domain_config = domain_config or {}

    @classmethod
    def from_domain_file(cls, conn, domain_path: str = 'config/domain.yaml'):
        """Create LoadStage with domain config from file."""
        config_path = Path(domain_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return cls(conn, domain_config=config)
        return cls(conn)

    def _build_column_map(self, col_names: list[str]) -> dict[str, str]:
        """
        Build column mapping based on domain config and defaults.

        Priority:
        1. Domain config index.column setting
        2. Default index column names
        3. Default value column names
        """
        column_map = {}

        # Get index column from domain config
        index_config = self.domain_config.get('index', {})
        domain_index_col = index_config.get('column')

        # Map index column
        for col in col_names:
            # Domain config takes priority
            if domain_index_col and col == domain_index_col:
                column_map[col] = 'I'
            elif col in self.DEFAULT_INDEX_COLUMNS:
                column_map[col] = 'I'
            elif col in self.DEFAULT_VALUE_COLUMNS:
                column_map[col] = 'y'

        return column_map

    def load_observations(self, path: str) -> None:
        """
        Load observations parquet into database.

        Handles column renaming to canonical schema:
          entity_id, signal_id, I (index), y (value)

        Uses domain config for index column if available,
        otherwise auto-detects from common column names.

        PURE: Just column aliasing, no computation.
        """
        # Get actual columns from file
        cols = self.conn.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()
        col_names = [c[0] for c in cols]

        # Build column mapping
        column_map = self._build_column_map(col_names)

        # Build SELECT with renames
        select_parts = []
        for col in col_names:
            canonical = column_map.get(col, col)
            if canonical != col:
                select_parts.append(f'"{col}" AS {canonical}')
            else:
                select_parts.append(f'"{col}"')

        select_clause = ', '.join(select_parts)
        self.conn.execute(f"CREATE OR REPLACE TABLE observations AS SELECT {select_clause} FROM '{path}'")

    def get_row_count(self) -> int:
        """Return number of rows loaded."""
        return self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    def get_signal_count(self) -> int:
        """Return number of distinct signals."""
        return self.conn.execute("SELECT COUNT(DISTINCT signal_id) FROM observations").fetchone()[0]

    def get_index_range(self) -> tuple:
        """Return (min_I, max_I) range of index."""
        result = self.conn.execute("SELECT MIN(I), MAX(I) FROM observations").fetchone()
        return result

    def get_index_info(self) -> dict:
        """Return index dimension info from domain config."""
        index_config = self.domain_config.get('index', {})
        return {
            'dimension': index_config.get('dimension', 'unknown'),
            'unit': index_config.get('unit', 'unknown'),
            'sampling_rate': index_config.get('sampling_rate'),
        }
