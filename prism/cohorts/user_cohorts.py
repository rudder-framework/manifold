"""
User Cohorts Management
=======================

CRUD operations for user-defined cohorts.

Storage:
- cohorts.parquet: Cohort definitions
- cohort_members.parquet: Unit memberships

Usage:
    from prism.cohorts import create_cohort, add_to_cohort, list_cohorts

    # Create a cohort
    cohort = create_cohort("healthy_engines", ["FD002_U001", "FD002_U002"])

    # Add more members
    add_to_cohort(cohort.cohort_id, "FD002_U003")

    # List all cohorts
    for c in list_cohorts():
        print(f"{c.cohort_name}: {c.member_count} members")
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl

from .models import Cohort, CohortMember, CohortSummary


def _get_data_path() -> Path:
    """Get path to data directory."""
    from ..db.parquet_store import get_data_root
    return get_data_root()


def _get_cohorts_path() -> Path:
    """Get path to cohorts.parquet."""
    return _get_data_path() / "cohorts.parquet"


def _get_members_path() -> Path:
    """Get path to cohort_members.parquet."""
    return _get_data_path() / "cohort_members.parquet"


def _load_cohorts() -> pl.DataFrame:
    """Load cohorts DataFrame."""
    path = _get_cohorts_path()
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame({
        'cohort_id': pl.Series([], dtype=pl.Utf8),
        'cohort_name': pl.Series([], dtype=pl.Utf8),
        'cohort_type': pl.Series([], dtype=pl.Utf8),
        'description': pl.Series([], dtype=pl.Utf8),
        'created_at': pl.Series([], dtype=pl.Datetime),
        'updated_at': pl.Series([], dtype=pl.Datetime),
    })


def _load_members() -> pl.DataFrame:
    """Load cohort_members DataFrame."""
    path = _get_members_path()
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame({
        'cohort_id': pl.Series([], dtype=pl.Utf8),
        'unit_id': pl.Series([], dtype=pl.Utf8),
        'membership_type': pl.Series([], dtype=pl.Utf8),
        'added_at': pl.Series([], dtype=pl.Datetime),
        'added_by': pl.Series([], dtype=pl.Utf8),
    })


def _save_cohorts(df: pl.DataFrame) -> None:
    """Save cohorts DataFrame."""
    path = _get_cohorts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _save_members(df: pl.DataFrame) -> None:
    """Save cohort_members DataFrame."""
    path = _get_members_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


# =============================================================================
# PUBLIC API
# =============================================================================

def create_cohort(
    name: str,
    unit_ids: Optional[List[str]] = None,
    description: str = "",
    cohort_type: str = "user_defined",
) -> Cohort:
    """
    Create a new cohort.

    Args:
        name: Display name for the cohort
        unit_ids: Optional list of unit IDs to add as members
        description: Optional description
        cohort_type: "user_defined" or "discovered"

    Returns:
        Created Cohort object
    """
    now = datetime.now()

    # Create cohort
    cohort = Cohort(
        cohort_name=name,
        cohort_type=cohort_type,
        description=description,
        created_at=now,
        updated_at=now,
    )

    # Load and update cohorts
    cohorts_df = _load_cohorts()

    # Check for duplicate name
    if not cohorts_df.filter(pl.col('cohort_name') == name).is_empty():
        raise ValueError(f"Cohort '{name}' already exists")

    # Add new cohort
    new_row = pl.DataFrame([cohort.to_dict()])
    cohorts_df = pl.concat([cohorts_df, new_row], how="align")
    _save_cohorts(cohorts_df)

    # Add members if provided
    if unit_ids:
        for unit_id in unit_ids:
            add_to_cohort(cohort.cohort_id, unit_id)

    return cohort


def get_cohort(cohort_id: str) -> Optional[Cohort]:
    """
    Get a cohort by ID.

    Args:
        cohort_id: Cohort identifier

    Returns:
        Cohort object or None if not found
    """
    cohorts_df = _load_cohorts()
    filtered = cohorts_df.filter(pl.col('cohort_id') == cohort_id)

    if filtered.is_empty():
        return None

    row = filtered.row(0, named=True)
    return Cohort.from_dict(row)


def get_cohort_by_name(name: str) -> Optional[Cohort]:
    """
    Get a cohort by name.

    Args:
        name: Cohort name

    Returns:
        Cohort object or None if not found
    """
    cohorts_df = _load_cohorts()
    filtered = cohorts_df.filter(pl.col('cohort_name') == name)

    if filtered.is_empty():
        return None

    row = filtered.row(0, named=True)
    return Cohort.from_dict(row)


def list_cohorts(cohort_type: Optional[str] = None) -> List[CohortSummary]:
    """
    List all cohorts with member counts.

    Args:
        cohort_type: Optional filter by type ("user_defined" or "discovered")

    Returns:
        List of CohortSummary objects
    """
    cohorts_df = _load_cohorts()
    members_df = _load_members()

    if cohort_type:
        cohorts_df = cohorts_df.filter(pl.col('cohort_type') == cohort_type)

    if cohorts_df.is_empty():
        return []

    # Count members per cohort
    member_counts = (
        members_df
        .group_by('cohort_id')
        .agg(pl.count().alias('member_count'))
    )

    # Join
    result = cohorts_df.join(member_counts, on='cohort_id', how='left')
    result = result.with_columns(pl.col('member_count').fill_null(0))

    summaries = []
    for row in result.iter_rows(named=True):
        summaries.append(CohortSummary(
            cohort_id=row['cohort_id'],
            cohort_name=row['cohort_name'],
            cohort_type=row['cohort_type'],
            description=row.get('description', ''),
            member_count=row['member_count'],
            created_at=row['created_at'],
        ))

    return summaries


def add_to_cohort(
    cohort_id: str,
    unit_id: str,
    membership_type: str = "explicit",
    added_by: str = "user",
) -> CohortMember:
    """
    Add a unit to a cohort.

    Args:
        cohort_id: Cohort identifier
        unit_id: Unit to add
        membership_type: "explicit" or "derived"
        added_by: Who/what added the unit

    Returns:
        Created CohortMember object
    """
    members_df = _load_members()

    # Check if already a member
    existing = members_df.filter(
        (pl.col('cohort_id') == cohort_id) &
        (pl.col('unit_id') == unit_id)
    )
    if not existing.is_empty():
        # Return existing membership
        row = existing.row(0, named=True)
        return CohortMember.from_dict(row)

    # Create membership
    member = CohortMember(
        cohort_id=cohort_id,
        unit_id=unit_id,
        membership_type=membership_type,
        added_at=datetime.now(),
        added_by=added_by,
    )

    # Add to DataFrame
    new_row = pl.DataFrame([member.to_dict()])
    members_df = pl.concat([members_df, new_row], how="align")
    _save_members(members_df)

    # Update cohort timestamp
    _update_cohort_timestamp(cohort_id)

    return member


def remove_from_cohort(cohort_id: str, unit_id: str) -> bool:
    """
    Remove a unit from a cohort.

    Args:
        cohort_id: Cohort identifier
        unit_id: Unit to remove

    Returns:
        True if removed, False if wasn't a member
    """
    members_df = _load_members()

    # Check if exists
    existing = members_df.filter(
        (pl.col('cohort_id') == cohort_id) &
        (pl.col('unit_id') == unit_id)
    )
    if existing.is_empty():
        return False

    # Remove
    members_df = members_df.filter(
        ~((pl.col('cohort_id') == cohort_id) &
          (pl.col('unit_id') == unit_id))
    )
    _save_members(members_df)

    # Update cohort timestamp
    _update_cohort_timestamp(cohort_id)

    return True


def get_cohort_members(cohort_id: str) -> List[str]:
    """
    Get all unit IDs in a cohort.

    Args:
        cohort_id: Cohort identifier

    Returns:
        List of unit IDs
    """
    members_df = _load_members()
    filtered = members_df.filter(pl.col('cohort_id') == cohort_id)

    if filtered.is_empty():
        return []

    return filtered.select('unit_id').to_series().to_list()


def get_unit_cohorts(unit_id: str) -> List[Cohort]:
    """
    Get all cohorts a unit belongs to.

    Args:
        unit_id: Unit identifier

    Returns:
        List of Cohort objects
    """
    members_df = _load_members()
    cohorts_df = _load_cohorts()

    unit_memberships = members_df.filter(pl.col('unit_id') == unit_id)
    if unit_memberships.is_empty():
        return []

    cohort_ids = unit_memberships.select('cohort_id').to_series().to_list()

    cohorts = []
    for cid in cohort_ids:
        cohort = get_cohort(cid)
        if cohort:
            cohorts.append(cohort)

    return cohorts


def delete_cohort(cohort_id: str) -> bool:
    """
    Delete a cohort and all its memberships.

    Args:
        cohort_id: Cohort identifier

    Returns:
        True if deleted, False if didn't exist
    """
    cohorts_df = _load_cohorts()

    # Check if exists
    existing = cohorts_df.filter(pl.col('cohort_id') == cohort_id)
    if existing.is_empty():
        return False

    # Remove cohort
    cohorts_df = cohorts_df.filter(pl.col('cohort_id') != cohort_id)
    _save_cohorts(cohorts_df)

    # Remove all memberships
    members_df = _load_members()
    members_df = members_df.filter(pl.col('cohort_id') != cohort_id)
    _save_members(members_df)

    return True


def update_cohort(
    cohort_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[Cohort]:
    """
    Update cohort properties.

    Args:
        cohort_id: Cohort identifier
        name: New name (optional)
        description: New description (optional)

    Returns:
        Updated Cohort or None if not found
    """
    cohorts_df = _load_cohorts()

    # Check if exists
    existing = cohorts_df.filter(pl.col('cohort_id') == cohort_id)
    if existing.is_empty():
        return None

    # Build update expressions
    updates = [pl.col('updated_at').fill_null(datetime.now())]

    if name is not None:
        # Check for duplicate name
        duplicate = cohorts_df.filter(
            (pl.col('cohort_name') == name) &
            (pl.col('cohort_id') != cohort_id)
        )
        if not duplicate.is_empty():
            raise ValueError(f"Cohort '{name}' already exists")
        updates.append(
            pl.when(pl.col('cohort_id') == cohort_id)
            .then(pl.lit(name))
            .otherwise(pl.col('cohort_name'))
            .alias('cohort_name')
        )

    if description is not None:
        updates.append(
            pl.when(pl.col('cohort_id') == cohort_id)
            .then(pl.lit(description))
            .otherwise(pl.col('description'))
            .alias('description')
        )

    # Apply updates
    cohorts_df = cohorts_df.with_columns([
        pl.when(pl.col('cohort_id') == cohort_id)
        .then(pl.lit(datetime.now()))
        .otherwise(pl.col('updated_at'))
        .alias('updated_at')
    ])

    if name is not None:
        cohorts_df = cohorts_df.with_columns([
            pl.when(pl.col('cohort_id') == cohort_id)
            .then(pl.lit(name))
            .otherwise(pl.col('cohort_name'))
            .alias('cohort_name')
        ])

    if description is not None:
        cohorts_df = cohorts_df.with_columns([
            pl.when(pl.col('cohort_id') == cohort_id)
            .then(pl.lit(description))
            .otherwise(pl.col('description'))
            .alias('description')
        ])

    _save_cohorts(cohorts_df)

    return get_cohort(cohort_id)


def _update_cohort_timestamp(cohort_id: str) -> None:
    """Update cohort's updated_at timestamp."""
    cohorts_df = _load_cohorts()

    cohorts_df = cohorts_df.with_columns([
        pl.when(pl.col('cohort_id') == cohort_id)
        .then(pl.lit(datetime.now()))
        .otherwise(pl.col('updated_at'))
        .alias('updated_at')
    ])

    _save_cohorts(cohorts_df)
