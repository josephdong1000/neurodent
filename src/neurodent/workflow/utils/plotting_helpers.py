"""WAR loading and plot-order/marker helpers for workflow figure scripts.

Split from the former ``workflow/utils.py`` module.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurodent.results import WindowAnalysisResult


def load_wars(
    parquet_paths: list[str | Path],
    json_paths: list[str | Path] | None = None,
) -> list["WindowAnalysisResult"]:
    """Load multiple WindowAnalysisResult objects from parquet/json file pairs.

    General-purpose utility for loading WAR files. Works with any list of paths,
    not tied to Snakemake.

    Args:
        parquet_paths: Paths to .parquet files containing WindowAnalysisResult data.
            For backward compatibility, legacy .pkl paths are also accepted — the
            loader will resolve the corresponding .parquet file next to them and
            fall back to the pickle only if the parquet is missing.
        json_paths: Optional paths to corresponding .json metadata files.
            If None, assumes json files are in the same directory as the parquet
            files with the same basename but .json extension.

    Returns:
        List of loaded WindowAnalysisResult objects.

    Raises:
        FileNotFoundError: If a parquet or json file does not exist.
        RuntimeError: If no WARs could be loaded.

    Example:
        Load WARs from explicit paths::

            from neurodent.workflow import load_wars

            wars = load_wars(
                parquet_paths=["data/animal1/war.parquet", "data/animal2/war.parquet"],
                json_paths=["data/animal1/war.json", "data/animal2/war.json"],
            )

        Load WARs with auto-detected json paths::

            wars = load_wars(parquet_paths=["data/animal1/war.parquet"])
            # Automatically looks for data/animal1/war.json
    """
    from neurodent.results import WindowAnalysisResult

    # If json_paths not provided, derive from parquet_paths
    if json_paths is None:
        json_paths = [Path(p).with_suffix(".json") for p in parquet_paths]

    if len(parquet_paths) != len(json_paths):
        raise ValueError(
            f"parquet_paths ({len(parquet_paths)}) and json_paths ({len(json_paths)}) "
            "must have the same length"
        )

    wars = []
    for parquet_path, json_path in zip(parquet_paths, json_paths):
        parquet_path = Path(parquet_path)
        json_path = Path(json_path)

        # Accept legacy .pkl input by swapping the suffix
        if parquet_path.suffix == ".pkl":
            parquet_path = parquet_path.with_suffix(".parquet")

        war = WindowAnalysisResult.load_parquet_and_json(
            folder_path=parquet_path.parent,
            parquet_name=parquet_path.name,
            json_name=json_path.name,
        )
        wars.append(war)

    if not wars:
        raise RuntimeError("No WARs were successfully loaded")

    return wars


def extend_plot_order_from_attr(wars, attr: str, base_order):
    """Extend a plot-order list with the values of *attr* observed on *wars*.

    Mirrors the dynamic-extension pattern used in EP plotting scripts:
    start from a base order (typically ``constants.DF_SORT_ORDER[attr]``)
    and append any values seen on the loaded WARs that aren't already in
    that base.  This keeps strict plot-order validation happy for datasets
    with non-default category values (e.g. arxrosa, where every animal has
    ``sex='Unknown'``).

    Args:
        wars: Iterable of objects exposing ``attr`` (typically
            :class:`WindowAnalysisResult` instances).
        attr: The attribute / column name to extend (``"genotype"``,
            ``"sex"``, ...).
        base_order: Starting list of category values; not mutated.

    Returns:
        list: ``list(base_order)`` with any newly-observed truthy values of
            ``getattr(war, attr)`` appended, preserving insertion order
            relative to *base_order*.

    Example::

        >>> from neurodent import constants
        >>> base = constants.DF_SORT_ORDER["sex"]   # ["Male", "Female"]
        >>> class W: pass
        >>> w1, w2 = W(), W()
        >>> w1.sex, w2.sex = "Male", "Unknown"
        >>> extend_plot_order_from_attr([w1, w2], "sex", base)
        ['Male', 'Female', 'Unknown']
    """
    order = list(base_order)
    seen = set(order)
    for war in wars:
        v = getattr(war, attr, None)
        if not v or v in seen:
            continue
        logging.info(f"Adding unknown {attr} '{v}' to plot order")
        order.append(v)
        seen.add(v)
    return order


def create_sex_marker_scale(df, plot_lib=None):
    """Build a seaborn-objects marker scale for the sex column of *df*.

    Preserves the canonical Female=circle (``"o"``), Male=square (``"s"``)
    mapping when those values are present, and assigns a diamond (``"D"``)
    fallback marker for any non-canonical sex value (e.g. arxrosa's
    ``"Unknown"``).

    Why this exists: ep_figures plots use ``so.Plot(..., marker="sex")``
    with a static ``so.Nominal(["o", "s"], order=["Female", "Male"])``
    scale. seaborn-objects **silently drops** any row whose sex value
    isn't listed in ``order``. Datasets with non-canonical sex (arxrosa)
    therefore produce blank plots with no error. This helper makes the
    scale's order/markers track what's actually in ``df``.

    Args:
        df: DataFrame with a ``"sex"`` column.
        plot_lib: Optional reference to ``seaborn.objects``. If ``None``,
            it's imported lazily so this util doesn't require seaborn at
            module-load time (useful for tests that don't render plots).

    Returns:
        A ``seaborn.objects.Nominal`` scale instance.

    Example::

        >>> import pandas as pd
        >>> df = pd.DataFrame({"sex": ["Female", "Male", "Female"]})
        >>> scale = create_sex_marker_scale(df)
        >>> scale.order
        ['Female', 'Male']
        >>> scale.values
        ['o', 's']
    """
    if plot_lib is None:
        import seaborn.objects as so
        plot_lib = so

    sex_marker_map = {"Female": "o", "Male": "s"}
    fallback_marker = "D"
    observed = list(df["sex"].dropna().unique())
    # Preserve canonical Female/Male ordering when present; append the rest.
    order = [s for s in ["Female", "Male"] if s in observed] + [
        s for s in observed if s not in ("Female", "Male")
    ]
    markers = [sex_marker_map.get(s, fallback_marker) for s in order]
    return plot_lib.Nominal(markers, order=order)
