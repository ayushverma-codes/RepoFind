"""Repository ingestion: safe snapshot, inventory, context and static analysis."""
from reviewharness.ingest.errors import (
    AcquisitionError,
    IngestError,
    PathEscapeError,
    RepositoryTooLargeError,
    UnsafeArchiveError,
)
from reviewharness.ingest.models import IngestResult
from reviewharness.ingest.pipeline import ingest_repository

__all__ = [
    "AcquisitionError",
    "IngestError",
    "IngestResult",
    "PathEscapeError",
    "RepositoryTooLargeError",
    "UnsafeArchiveError",
    "ingest_repository",
]
