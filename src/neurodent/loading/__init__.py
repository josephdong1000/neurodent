"""Loading stage: discover recording files and organize them into recording objects.

Holds the single-recording loader (:class:`LongRecordingOrganizer`), the
per-animal organizer (:class:`AnimalOrganizer`), and file discovery. Depends only
on :mod:`neurodent.core` (shared helpers) and :mod:`neurodent.constants`.
"""

from .long_recording_organizer import (
    LongRecordingOrganizer,
    RecordingMetadata,
    split_recording,
)
from .discovery import FileDiscoverer, DiscoveredFile
from .animal_organizer import AnimalOrganizer

__all__ = [
    "LongRecordingOrganizer",
    "RecordingMetadata",
    "split_recording",
    "FileDiscoverer",
    "DiscoveredFile",
    "AnimalOrganizer",
]
