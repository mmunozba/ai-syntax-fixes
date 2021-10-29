from enum import Enum


class FixType(Enum):
    """Enum with the different fix types for fixing syntax errors."""
    insert = 0
    delete = 1
    modify = 2
