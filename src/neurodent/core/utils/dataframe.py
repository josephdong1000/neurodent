"""NaN-aware aggregation and DataFrame ordering helpers."""

import itertools

from typing import Optional

import numpy as np
import pandas as pd

from neurodent import constants


def nanaverage(A: np.ndarray, weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute weighted average of an array, ignoring NaN values.

    This function computes a weighted average along the specified axis while
    properly handling NaN values by masking them out of the calculation.

    Args:
        A (np.ndarray): Input array containing the values to average.
        weights (np.ndarray): Array of weights corresponding to the values in A.
            Must be broadcastable with A along the specified axis.
        axis (int, optional): Axis along which to compute the average. Defaults to -1 (last axis).

    Returns:
        np.ndarray: Weighted average with NaN values properly handled. If all values
            along an axis are NaN, the result will be NaN for that position.

    Examples:
        >>> import numpy as np
        >>> A = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        >>> weights = np.array([1, 2, 1])
        >>> nanaverage(A, weights, axis=1)
        array([1.66666667, 5.        ])

    Note:
        Be careful with zero or negative weights as they may produce unexpected results.
        The function uses numpy's masked array functionality for robust NaN handling.
    """
    masked = np.ma.masked_array(A, np.isnan(A))
    avg = np.ma.average(masked, axis=axis, weights=weights)

    # Handle case where np.ma.average returns a scalar instead of masked array
    if np.ma.is_masked(avg):
        return avg.filled(np.nan)
    else:
        # avg is a scalar or regular array, convert to array and handle NaN
        result = np.asarray(avg)
        return np.where(np.isfinite(result), result, np.nan)


def nanmean_series_of_np(x: pd.Series, axis: int = 0) -> np.ndarray:
    """
    Efficiently compute NaN-aware mean of a pandas Series containing numpy arrays.

    This function is optimized for computing the mean across a Series where each element
    is a numpy array. It uses different strategies based on the size of the Series
    for optimal performance.

    Args:
        x (pd.Series): Series containing numpy arrays as elements.
        axis (int, optional): Axis along which to compute the mean. Defaults to 0.
            - axis=0: Mean across the Series elements (most common)
            - axis=1: Mean within each array element

    Returns:
        np.ndarray: Array containing the computed means with NaN values properly handled.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a Series of numpy arrays
        >>> arrays = [np.array([1.0, 2.0, np.nan]),
        ...           np.array([4.0, np.nan, 6.0]),
        ...           np.array([7.0, 8.0, 9.0])]
        >>> series = pd.Series(arrays)
        >>> nanmean_series_of_np(series)
        array([4. , 5. , 7.5])

    Performance Notes:
        - For Series with more than 1000 elements containing numpy arrays,
          uses `np.stack()` for better performance
        - Falls back to list conversion for smaller Series or mixed types
        - Handles shape mismatches gracefully by falling back to the slower method
    """
    # logging.debug(f"Unique shapes in x: {set(np.shape(item) for item in x)}")

    if len(x) > 1000:
        try:
            if isinstance(x.iloc[0], np.ndarray):
                xmean: np.ndarray = np.nanmean(np.stack(x.values, axis=0), axis=axis)
                return xmean
        except (ValueError, TypeError):
            pass

    xmean: np.ndarray = np.nanmean(np.array(list(x)), axis=axis)
    return xmean


def sort_dataframe_by_plot_order(df: pd.DataFrame, df_sort_order: Optional[dict] = None) -> pd.DataFrame:
    """
    Sort DataFrame columns according to predefined orders.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sort
    df_sort_order : dict
        Dictionary mapping column names to the order of the values in the column.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame

    Raises
    ------
    ValueError
        If df_sort_order is not a valid dictionary or contains invalid categories
    """
    if df_sort_order is None:
        df_sort_order = constants.DF_SORT_ORDER.copy()
    elif not isinstance(df_sort_order, dict):
        raise ValueError("df_sort_order must be a dictionary")

    if df.empty:
        return df.copy()

    for col, categories in df_sort_order.items():
        if not isinstance(categories, (list, tuple)):
            raise ValueError(f"Categories for column '{col}' must be a list or tuple")

    columns_to_sort = [col for col in df.columns if col in df_sort_order]
    df_sorted = df.copy()

    if not columns_to_sort:
        return df_sorted

    for col in columns_to_sort:
        categories = df_sort_order[col]

        # Check for values not in predefined categories
        unique_values = set(df_sorted[col].dropna().unique())
        missing_values = unique_values - set(categories)

        if missing_values:
            raise ValueError(
                f"Column '{col}' contains values not in sort order dictionary: {missing_values}. Add them to plot_order in ExperimentPlotter init."
            )

        # Filter categories to only include those that exist in the DataFrame
        existing_categories = [cat for cat in categories if cat in unique_values]

        df_sorted[col] = pd.Categorical(df_sorted[col], categories=existing_categories, ordered=True)

    df_sorted = df_sorted.sort_values(columns_to_sort)
    # REVIEW since "sex" is not inherently part of the pipeline (add ad-hoc), this could be a feature worth sorting
    # But this might mean rewriting the data loading pipeline, file-reading, etc.
    # Maybe a dictionary corresponding to animal/id -> sex would be good enough, instead of reading it in from filenames
    # which would be difficult since name conventions are not standardized

    return df_sorted


def _get_groupby_keys(df: pd.DataFrame, groupby: str | list[str]):
    """
    Get the unique values of the groupby variable.
    """
    return list(df.groupby(groupby).groups.keys())


def _get_pairwise_combinations(x: list):
    """
    Get all pairwise combinations of a list.
    """
    return list(itertools.combinations(x, 2))
