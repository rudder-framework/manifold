"""
PRISM Cohorts
=============

User-defined and discovered cohort management.

Cohorts are groupings of units (engines, bearings, etc.) for comparison.
Two types:
- user_defined: Explicitly created by users
- discovered: Emergent from state alignment analysis

Key distinction from previous "cohort" concept:
- Old: Muddied concept mixing entity groupings with behavioral patterns
- New: Clean separation between user cohorts (explicit) and discovered groupings (queries)
"""

from .models import Cohort, CohortMember
from .user_cohorts import (
    create_cohort,
    add_to_cohort,
    remove_from_cohort,
    list_cohorts,
    get_cohort_members,
    delete_cohort,
    get_cohort,
)

__all__ = [
    "Cohort",
    "CohortMember",
    "create_cohort",
    "add_to_cohort",
    "remove_from_cohort",
    "list_cohorts",
    "get_cohort_members",
    "delete_cohort",
    "get_cohort",
]
