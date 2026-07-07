"""File/item adapters and session grouping.

Mixin for :class:`~neurodent.loading.animal_organizer.AnimalOrganizer`.
"""

from __future__ import annotations

from pathlib import Path


class AoDiscoveryMixin:
    """Mixin: see module docstring."""

    def _get_item_name(self, item):
        """Helper to get a representative name for an item which could be a string, Path, list of strings, or DiscoveredFile."""
        from .discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            paths = item.get_path_list()
            if len(paths) > 1:
                return Path(paths[0]).name + "..."
            return Path(paths[0]).name if paths else "unknown"
        if isinstance(item, (list, tuple)):
            return Path(item[0]).name
        return Path(item).name

    def _get_item_key(self, item):
        """Return a unique key for the item, suitable for dict lookups across sessions.

        Unlike _get_item_name() which returns only the filename (e.g. 'file-0.bin'),
        this returns the full path, ensuring items with the same filename in different
        session directories get distinct keys.
        """
        from .discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            paths = item.get_path_list()
            return str(paths[0]) if paths else "unknown"
        if isinstance(item, (list, tuple)):
            return str(item[0])
        return str(item)

    def _is_item_file(self, item):
        """Helper to check if an item represents a file(s) rather than a directory."""
        from .discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            paths = item.get_path_list()
            return Path(paths[0]).is_file() if paths else False
        if isinstance(item, (list, tuple)):
            return Path(item[0]).is_file()
        return Path(item).is_file()

    @staticmethod
    def _get_context_path(item) -> Path:
        """Return a single Path from an item (str, Path, list, or DiscoveredFile)."""
        from .discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            return Path(item.get_path_list()[0])
        if isinstance(item, (list, tuple)):
            return Path(item[0])
        return Path(item)

    def _find_folder_by_name(
        self, folder_name: str, animalday_to_folders: dict
    ) -> Path:
        """Find folder path by name in the animalday groups."""
        for animalday, folders in animalday_to_folders.items():
            for folder in folders:
                if Path(folder).name == folder_name:
                    return Path(folder)

        available_names = []
        for folders in animalday_to_folders.values():
            available_names.extend([Path(f).name for f in folders])

        raise ValueError(
            f"Folder name '{folder_name}' not found. Available folders: {available_names}"
        )

    def _get_folders_for_animal(
        self, animal_id: str, animalday_to_folders: dict
    ) -> list:
        """Find all folder paths belonging to a specific animal ID."""
        matching_folders = []
        for animalday, folders in animalday_to_folders.items():
            if animalday.startswith(animal_id):
                matching_folders.extend(folders)
        return matching_folders

    def _items_have_index(self, items):
        """Check if items carry {index} metadata."""
        return (
            items
            and hasattr(items[0], "metadata")
            and "index" in getattr(items[0], "metadata", {})
        )

    def _session_sort_key(self, items):
        """Return sort-key function: use {index} metadata if available, else filename."""
        from .discovery import _natural_sort_key

        if self._items_have_index(items):
            return lambda f: _natural_sort_key(f.metadata["index"])
        return lambda f: _natural_sort_key(self._get_item_name(f))
