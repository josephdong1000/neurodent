"""Example recordings bundled with the package.

Two rodent EEG excerpts, ``A10`` and ``F22``, backing the documentation tutorials
so the examples run from an installed package with no download. Each animal has a
60-second excerpt as a paired ColMajor ``.bin`` + Meta ``.csv``, which is what the
tutorials analyse, and a separate 5-second ``.edf`` used only to demonstrate
single-file loading of a standard format. See ``sample/PROVENANCE.md`` for their
origin.
"""

from importlib.resources import files
from pathlib import Path

__all__ = ["sample_dataset", "sample_pattern", "sample_edf"]


def sample_dataset() -> Path:
    """Return the directory holding the bundled example recordings.

    Returns
    -------
    Path
        Directory containing one subdirectory per animal (``A10``, ``F22``).
    """
    return Path(str(files("neurodent.data") / "sample"))


def sample_pattern() -> list[str]:
    """Return the ``AnimalOrganizer`` pattern pair for the bundled recordings.

    Returns
    -------
    list of str
        Two glob patterns with an ``{animal}`` placeholder, one for the
        ColMajor ``.bin`` file and one for its Meta ``.csv``.
    """
    root = sample_dataset()
    return [
        str(root / "{animal}" / "*_ColMajor.bin"),
        str(root / "{animal}" / "*_Meta.csv"),
    ]


def sample_edf(animal_id: str = "A10") -> Path:
    """Return the bundled single-file ``.edf`` recording for one animal.

    Parameters
    ----------
    animal_id : str
        Animal directory name, ``"A10"`` or ``"F22"``.

    Returns
    -------
    Path
        Path to the ``.edf`` file.
    """
    matches = sorted((sample_dataset() / animal_id).glob("*.edf"))
    if not matches:
        raise FileNotFoundError(f"No .edf recording bundled for animal {animal_id!r}")
    return matches[0]
