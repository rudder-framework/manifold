"""
Cohort Models
=============

Dataclasses for cohort management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid


@dataclass
class Cohort:
    """
    A named grouping of units.

    Cohorts can be:
    - user_defined: Explicitly created by users (e.g., "healthy engines")
    - discovered: Emergent from state alignment analysis

    Attributes:
        cohort_id: Unique identifier
        cohort_name: Display name
        cohort_type: "user_defined" or "discovered"
        description: Optional description
        created_at: When the cohort was created
        updated_at: When the cohort was last modified
    """
    cohort_id: str = ""
    cohort_name: str = ""
    cohort_type: str = "user_defined"  # "user_defined" | "discovered"
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.cohort_id:
            self.cohort_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cohort_id': self.cohort_id,
            'cohort_name': self.cohort_name,
            'cohort_type': self.cohort_type,
            'description': self.description,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else str(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cohort":
        """Create Cohort from dictionary."""
        cohort = cls()
        cohort.cohort_id = data.get('cohort_id', '')
        cohort.cohort_name = data.get('cohort_name', '')
        cohort.cohort_type = data.get('cohort_type', 'user_defined')
        cohort.description = data.get('description', '')

        created = data.get('created_at')
        if isinstance(created, str):
            cohort.created_at = datetime.fromisoformat(created)
        elif isinstance(created, datetime):
            cohort.created_at = created

        updated = data.get('updated_at')
        if isinstance(updated, str):
            cohort.updated_at = datetime.fromisoformat(updated)
        elif isinstance(updated, datetime):
            cohort.updated_at = updated

        return cohort


@dataclass
class CohortMember:
    """
    A unit's membership in a cohort.

    Attributes:
        cohort_id: FK to cohort
        unit_id: Unit in the cohort
        membership_type: "explicit" (user added) or "derived" (from analysis)
        added_at: When the unit was added
        added_by: Who/what added the unit (user ID or "system")
    """
    cohort_id: str = ""
    unit_id: str = ""
    membership_type: str = "explicit"  # "explicit" | "derived"
    added_at: datetime = field(default_factory=datetime.now)
    added_by: str = "user"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cohort_id': self.cohort_id,
            'unit_id': self.unit_id,
            'membership_type': self.membership_type,
            'added_at': self.added_at.isoformat() if isinstance(self.added_at, datetime) else str(self.added_at),
            'added_by': self.added_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohortMember":
        """Create CohortMember from dictionary."""
        member = cls()
        member.cohort_id = data.get('cohort_id', '')
        member.unit_id = data.get('unit_id', '')
        member.membership_type = data.get('membership_type', 'explicit')
        member.added_by = data.get('added_by', 'user')

        added = data.get('added_at')
        if isinstance(added, str):
            member.added_at = datetime.fromisoformat(added)
        elif isinstance(added, datetime):
            member.added_at = added

        return member


@dataclass
class CohortSummary:
    """
    Summary view of a cohort with member count.

    Used for list views without loading all members.
    """
    cohort_id: str = ""
    cohort_name: str = ""
    cohort_type: str = "user_defined"
    description: str = ""
    member_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cohort_id': self.cohort_id,
            'cohort_name': self.cohort_name,
            'cohort_type': self.cohort_type,
            'description': self.description,
            'member_count': self.member_count,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
        }
