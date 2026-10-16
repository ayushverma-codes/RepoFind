"""Exceptions raised by repository ingestion. All derive from IngestError."""


class IngestError(Exception):
    """Base class: the repository could not be safely ingested."""


class PathEscapeError(IngestError):
    """A path tried to leave the snapshot root (traversal, absolute path, symlink)."""


class UnsafeArchiveError(IngestError):
    """A ZIP archive is hostile or unsupported (traversal, duplicates, encryption)."""


class RepositoryTooLargeError(IngestError):
    """The repository exceeds the configured size or file-count limits."""


class AcquisitionError(IngestError):
    """The repository could not be fetched or read (missing path, git failure...)."""
