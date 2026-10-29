"""Abstract base class for track importers."""

from abc import ABC, abstractmethod
from pathlib import Path
from laptimesim.track.canonical import CanonicalTrack


class TrackImporterBase(ABC):
    """Base class for parsing and converting track files to CanonicalTrack representation."""

    @abstractmethod
    def load(self, file_path: str | Path) -> CanonicalTrack:
        """Parse track file and return a CanonicalTrack object."""
        ...
