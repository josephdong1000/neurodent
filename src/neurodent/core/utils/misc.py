"""Small standalone helpers (truncate parsing, stdout suppression)."""

import os
import sys


def parse_truncate(truncate: int | bool) -> int:
    """
    Parse the truncate parameter to determine how many characters to truncate.

    If truncate is a boolean, returns 10 if True and 0 if False.
    If truncate is an integer, returns that integer value directly.

    Args:
        truncate (int | bool): If bool, True=10 chars and False=0 chars.
                              If int, specifies exact number of chars.

    Returns:
        int: Number of characters to truncate (0 means no truncation)

    Raises:
        ValueError: If truncate is not a boolean or integer
    """
    if isinstance(truncate, bool):
        return 10 if truncate else 0
    elif isinstance(truncate, int):
        return truncate
    else:
        raise ValueError(f"Invalid truncate value: {truncate}")


class _HiddenPrints:
    """
    Context manager to suppress print output during code execution.

    This class provides a way to temporarily suppress print statements and other
    stdout output, which is useful when calling functions that produce unwanted
    console output.

    Args:
        silence (bool, optional): Whether to actually suppress output. Defaults to True.
            If False, acts as a no-op context manager.

    Examples:
        >>> with _HiddenPrints():
        ...     print("This won't be displayed")
        ...     some_noisy_function()
        >>> print("This will be displayed")
        This will be displayed

        >>> with _HiddenPrints(silence=False):
        ...     print("This will be displayed")
        This will be displayed
    """

    def __init__(self, silence: bool = True) -> None:
        self.silence = silence

    def __enter__(self):
        if self.silence:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silence:
            sys.stdout.close()
            sys.stdout = self._original_stdout
