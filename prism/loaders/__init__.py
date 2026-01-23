"""
PRISM Loaders
=============

Domain-agnostic data loaders with config-based column mapping.

The loader system keeps all domain-specific knowledge in YAML configs,
ensuring code never changes for a new domain.

Usage:
    from prism.loaders import DomainLoader

    loader = DomainLoader('path/to/domain.yaml')
    observations = loader.load_observations()

Config Format (domain.yaml):
    source: cmapss
    entity_column: unit_id
    signal_columns: [sensor_1, sensor_2, ...]
    time_column: cycle
    # OR for long-format data:
    signal_column: signal_id
    value_column: value
"""

from .domain_loader import DomainLoader, DomainConfig
from .schema_mapper import SchemaMapper

__all__ = [
    'DomainLoader',
    'DomainConfig',
    'SchemaMapper',
]
