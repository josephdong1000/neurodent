import re
import glob
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import warnings


class MultiFileGroup:
    """Files that must be loaded together as one recording unit.

    Created by FileDiscoverer when multiple patterns are provided.
    This wrapper distinguishes multi-file sessions (e.g., .bin + .csv that should
    be loaded together) from lists of single files (e.g., multiple .rhd files that
    should be concatenated).
    """
    def __init__(self, paths: tuple[str, ...], metadata: dict):
        self.paths = paths
        self.metadata = metadata  # {animal: ..., session: ..., etc.}

    def __iter__(self):
        return iter(self.paths)

    def __repr__(self):
        return f"MultiFileGroup(paths={self.paths}, metadata={self.metadata})"


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
            # Match anything except slashes. This ensures {animal} doesn't span multiple directories.
            regex_string += f"(?P<{placeholder}>[^/]+)"
            regex_string += re.escape(parts[i + 1])
        regex_string += "$"

        return re.compile(regex_string), glob_pattern

    def discover(self, **filter_kwargs) -> List[Union[Dict, "MultiFileGroup"]]:
        """
        Discovers files matching patterns, returning a list of dictionaries or MultiFileGroup objects.
        Keyword args like `animal="A10"` can strictly filter the returned files.

        Returns:
            A list of dicts (for single pattern) or MultiFileGroup objects (for multiple patterns).
            If a single pattern was provided, dicts look like:
                {'path': '...', 'animal': 'A10', 'session': '1'}
            If multiple patterns were provided, returns MultiFileGroup objects with grouped files:
                MultiFileGroup(paths=('..._data.bin', '..._meta.json'), metadata={'animal': 'A10', 'session': '1'})
        """
        is_single = len(self.patterns) == 1
        return_list = []

        # For multiple patterns, we need to discover each independently and then group.
        # grouping key -> tuple of paths
        all_discovered = [
            self._discover_single(p, **filter_kwargs) for p in self.patterns
        ]

        if is_single:
            return all_discovered[0]

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

        # Filter for complete groups and construct MultiFileGroup objects
        for key, paths in grouped_results.items():
            if len(paths) == len(self.patterns):
                metadata = {k: v for k, v in zip(keys_to_group, key)}
                return_list.append(MultiFileGroup(paths=tuple(paths), metadata=metadata))

        return return_list

    def _discover_single(self, pattern: str, **filter_kwargs) -> List[Dict]:
        """Discovers files for a single pattern."""
        regex, glob_str = self._pattern_to_regex_and_glob(pattern)

        # Handle tilde and resolve to absolute just in case, though user can pass absolute.
        # For glob to work nicely with absolute/relative, we just pass the string.
        discovered_paths = glob.glob(glob_str, recursive=True)

        results = []
        for path in discovered_paths:
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

        # Sort results by path for deterministic ordering (like old filepath_to_index)
        results.sort(key=lambda x: x["path"])
        return results
