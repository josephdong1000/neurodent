import os
import re
import glob
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import warnings


class DiscoveredFile(os.PathLike):
    """Represents discovered file(s) with associated metadata.

    This unified class handles both single files and groups of files that must
    be loaded together. It replaces the previous dict/MultiFileGroup split behavior.

    Implements ``os.PathLike`` so that ``Path(discovered_file)`` works for
    single-file discoveries, and provides dict-style access (``obj["animal"]``,
    ``"path" in obj``) for backward compatibility with the old dict-based API.

    Attributes:
        path (str | None): Single file path (for single-pattern discoveries)
        paths (tuple[str, ...] | None): Multiple file paths (for multi-pattern discoveries)
        metadata (dict): Extracted metadata from pattern placeholders (e.g., {animal, session, index})

    Examples:
        Single file: DiscoveredFile(path="/data/A10/session1/001.rhd", metadata={"animal": "A10", "session": "session1", "index": "001"})
        Multiple files: DiscoveredFile(paths=("/data/A10/s1/data.bin", "/data/A10/s1/meta.csv"), metadata={"animal": "A10", "session": "s1"})
    """
    def __init__(self, path: str = None, paths: tuple[str, ...] = None, metadata: dict = None):
        if path is None and paths is None:
            raise ValueError("Either path or paths must be provided")
        if path is not None and paths is not None:
            raise ValueError("Cannot provide both path and paths")

        self.path = path
        self.paths = paths
        self.metadata = metadata or {}

    # -- os.PathLike protocol --------------------------------------------------

    def __fspath__(self) -> str:
        """Return the file-system path string.

        For single-file discoveries returns ``self.path``.
        For multi-file discoveries raises TypeError because the representation
        is ambiguous – use ``.paths`` or ``.get_path_list()`` instead.
        """
        if self.path is not None:
            return self.path
        if self.paths is not None:
            raise TypeError(
                "Multi-file DiscoveredFile cannot be converted to a single path. "
                "Use .paths or .get_path_list() instead."
            )
        raise TypeError("DiscoveredFile has no path to return")

    # -- dict-style backward compatibility ------------------------------------

    def __contains__(self, key):
        """Support ``'animal' in discovered_file`` for backward compat."""
        if key in ("path", "paths"):
            return getattr(self, key) is not None
        return key in self.metadata

    def __getitem__(self, key):
        """Support ``discovered_file['animal']`` for backward compat."""
        if key == "path":
            return self.path
        if key == "paths":
            return self.paths
        return self.metadata[key]

    @property
    def is_multi_file(self) -> bool:
        """Returns True if this represents multiple files that should be loaded together."""
        return self.paths is not None

    def get_path_list(self) -> List[str]:
        """Returns all paths as a list, whether single or multiple files."""
        if self.paths is not None:
            return list(self.paths)
        return [self.path] if self.path else []

    def __iter__(self):
        """Iterate over paths (useful for backward compatibility with MultiFileGroup)."""
        return iter(self.get_path_list())

    def __repr__(self):
        if self.is_multi_file:
            return f"DiscoveredFile(paths={self.paths}, metadata={self.metadata})"
        return f"DiscoveredFile(path={self.path!r}, metadata={self.metadata})"


# Deprecated: Keep MultiFileGroup for backward compatibility
class MultiFileGroup(DiscoveredFile):
    """Deprecated: Use DiscoveredFile instead.

    This class is maintained for backward compatibility but will be removed in a future version.
    """
    def __init__(self, paths: tuple[str, ...], metadata: dict):
        warnings.warn(
            "MultiFileGroup is deprecated. Use DiscoveredFile(paths=..., metadata=...) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(paths=paths, metadata=metadata)


class FileDiscoverer:
    """
    Utility class for discovering files based on wildcard pattern strings.
    Extracts metadata from paths based on named placeholders like {animal}, {session}, {index}.
    """

    def __init__(self, pattern: Union[str, Path, List[Union[str, Path]]]):
        if not pattern:
            raise ValueError("Pattern cannot be empty")

        # Ensure all patterns are strings, as users may pass pathlib.Path objects
        if isinstance(pattern, (str, Path)):
            self.patterns = [str(pattern)]
        else:
            self.patterns = [str(p) for p in pattern]

    def _pattern_to_regex_and_glob(self, pattern: str) -> Tuple[re.Pattern, str]:
        """
        Converts a format string to a regex and a glob pattern.
        Example:
            pattern: "/data/{animal}/{session}/{index}.rhd"
            glob: "/data/*/*/*.rhd"
            regex: "^/data/(?P<animal>[^/\\\\]+)/(?P<session>[^/\\\\]+)/(?P<index>[^/\\\\]+)\\.rhd$"
        """
        # Normalize path separators to forward slashes for consistent regex matching internally
        # We'll also normalize the actual paths when matching.
        pattern = pattern.replace("\\", "/")

        placeholders = re.findall(r"\{([^}]+)\}", pattern)
        glob_pattern = re.sub(r"\{[^}]+\}", "*", pattern)

        parts = re.split(r"\{[^}]+\}", pattern)
        regex_string = "^" + re.escape(parts[0])
        for i, placeholder in enumerate(placeholders):
            # Match anything except slashes and backslashes.
            # This ensures placeholders don't span multiple directories.
            regex_string += f"(?P<{placeholder}>[^/\\\\]+)"
            regex_string += re.escape(parts[i + 1])
        regex_string += "$"

        return re.compile(regex_string), glob_pattern

    def discover(self, **filter_kwargs) -> List["DiscoveredFile"]:
        """
        Discovers files matching patterns, returning a list of DiscoveredFile objects.
        Keyword args like `animal="A10"` can strictly filter the returned files.

        Returns:
            A list of DiscoveredFile objects.
            For single pattern: DiscoveredFile(path='...', metadata={'animal': 'A10', 'session': '1'})
            For multiple patterns: DiscoveredFile(paths=('..._data.bin', '..._meta.json'), metadata={'animal': 'A10', 'session': '1'})
        """
        is_single = len(self.patterns) == 1
        return_list = []

        # For multiple patterns, we need to discover each independently and then group.
        # grouping key -> tuple of paths
        all_discovered = [
            self._discover_single(p, **filter_kwargs) for p in self.patterns
        ]

        if is_single:
            # Convert dicts to DiscoveredFile objects
            for item in all_discovered[0]:
                path = item.pop("path")
                return_list.append(DiscoveredFile(path=path, metadata=item))
            return return_list

        # Grouping for multiple patterns.
        # Find the intersection of metadata keys across all found files.
        # We group by the metadata keys.
        grouped_results = {}

        # To group properly, we need to know the common keys.
        # But wait, each pattern should return the *same* metadata keys for a single recording?
        # Let's group by ALL metadata keys found in the first pattern's results.
        if not all_discovered[0]:
            return []

        keys_to_group = [k for k in all_discovered[0][0].keys() if k != "path"]

        # Build a dict: { tuple(metadata_values): [path1, path2, ...] }
        # We iterate through the patterns in order, adding paths to the corresponding group.
        # If a later pattern doesn't have a matching metadata group, it's skipped or creates an incomplete group.
        # We only return complete groups (having exactly len(patterns) paths).

        # Initialize groups with the first pattern
        for item in all_discovered[0]:
            key = tuple(item.get(k) for k in keys_to_group)
            grouped_results[key] = [item["path"]]

        # Add subsequent patterns
        for pattern_idx, pattern_results in enumerate(all_discovered[1:], 1):
            for item in pattern_results:
                key = tuple(item.get(k) for k in keys_to_group)
                if key in grouped_results:
                    # Make sure we don't add multiple from the same pattern if there's a duplicate?
                    # Append it if the list length matches pattern_idx
                    if len(grouped_results[key]) == pattern_idx:
                        grouped_results[key].append(item["path"])
                    else:
                        warnings.warn(
                            f"Duplicate or out-of-order match found for metadata {key} in pattern {self.patterns[pattern_idx]}"
                        )

        # Filter for complete groups and construct DiscoveredFile objects
        for key, paths in grouped_results.items():
            if len(paths) == len(self.patterns):
                metadata = {k: v for k, v in zip(keys_to_group, key)}
                return_list.append(DiscoveredFile(paths=tuple(paths), metadata=metadata))

        return return_list

    def _discover_single(self, pattern: str, **filter_kwargs) -> List[Dict]:
        """Discovers files for a single pattern."""
        regex, glob_str = self._pattern_to_regex_and_glob(pattern)

        # Check if pattern has any placeholders
        has_placeholders = bool(re.findall(r"\{([^}]+)\}", pattern))

        # Handle tilde and resolve to absolute just in case, though user can pass absolute.
        # For glob to work nicely with absolute/relative, we just pass the string.
        discovered_paths = glob.glob(glob_str, recursive=True)

        results = []
        for path in discovered_paths:
            if has_placeholders:
                # Normalize path for regex matching
                normalized_path = str(Path(path)).replace("\\", "/")

                match = regex.match(normalized_path)
                if match:
                    meta = match.groupdict()

                    # Apply filters
                    skip = False
                    for k, v in filter_kwargs.items():
                        if k in meta and meta[k] != v:
                            skip = True
                            break

                    if skip:
                        continue

                    meta["path"] = path
                    results.append(meta)
            else:
                # No placeholders - just return paths with no metadata
                results.append({"path": path})

        # Sort results by path for deterministic ordering (like old filepath_to_index)
        results.sort(key=lambda x: x["path"])
        return results
