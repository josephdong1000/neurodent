"""Parquet/JSON serialization for :class:`WindowAnalysisResult` (issue #134)."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurodent import constants
from neurodent.core.utils import atomic_output_path, atomic_write_json, slugify


def _column_needs_encoding(col_name: str) -> bool:
    """Return True iff *col_name* is a WAR feature column whose cells hold
    nested values (list, dict, tuple, ndarray).

    Every feature column qualifies — LINEAR cells are stored as a list of C
    scalars per row, BAND cells as a dict-of-lists, HIST cells as a tuple of
    arrays, etc.  Non-feature columns (animal, animalday, timestamp, duration,
    endfile, …) are scalar per row and skip the nested-encoding path.

    Uses :data:`constants.FEATURE_TYPES` as the schema source of truth, so new
    ``FeatureType`` additions are picked up without changing this code.
    """
    return col_name in constants.FEATURE_TYPES

class WARSerializationMixin:
    """Mixin: see module docstring."""

    def save_parquet_and_json(
        self,
        folder: str | Path,
        make_folder=True,
        filename: str = None,
        slugify_filename=False,
        save_abbrevs=False,
    ):
        """Archive window analysis result into the folder specified, as a parquet and json file.

        The result DataFrame is saved as a Parquet file (stable across pandas
        versions).  Metadata (animal_id, channel_names, bad_channels_dict,
        lof_scores_dict, etc.) is written alongside as a JSON sidecar.

        Args:
            folder (str | Path): Destination folder to save results to
            make_folder (bool, optional): If True, create the folder if it doesn't exist. Defaults to True.
            filename (str, optional): Name of the file to save. Defaults to "war".
            slugify_filename (bool, optional): If True, slugify the filename (replace special characters). Defaults to False.
            save_abbrevs (bool, optional): If True, save the channel abbreviations as the channel names in the json file. Defaults to False.
        """
        import pyarrow.parquet as pq

        folder = Path(folder)
        if make_folder:
            folder.mkdir(parents=True, exist_ok=True)

        filename = "war" if filename is None else filename
        filename = slugify(filename) if slugify_filename else filename

        filepath = str(folder / filename)

        table, encoded_cols = WARSerializationMixin._df_to_arrow_table(self.result)
        # encoding_version=2: encoded_cols are native list/struct; absence/1 = legacy JSON.
        neurodent_meta = json.dumps(
            {"encoded_columns": encoded_cols, "encoding_version": 2}
        ).encode()
        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, b"neurodent": neurodent_meta}
        table = table.replace_schema_metadata(merged_meta)
        # Write to a temp sibling and atomically rename, so an interrupted write
        # (e.g. a killed SLURM job) never leaves a partial .parquet that a
        # downstream rule would read as a valid output.
        with atomic_output_path(filepath + ".parquet") as tmp_parquet:
            pq.write_table(
                table, str(tmp_parquet), compression="zstd", compression_level=4
            )
        del table
        logging.info(f"Saved WAR to {filepath + '.parquet'}")

        json_dict = {
            "animal_id": self.animal_id,
            "genotype": self.genotype,
            "sex": self.sex,
            "channel_names": (
                self.channel_abbrevs if save_abbrevs else self.channel_names
            ),
            "bad_channels_dict": self.bad_channels_dict,
            "suppress_short_interval_error": self.suppress_short_interval_error,
            "lof_scores_dict": self.lof_scores_dict.copy(),
        }

        atomic_write_json(filepath + ".json", json_dict, indent=2)
        logging.info(f"Saved WAR to {filepath + '.json'}")

    def save_pickle_and_json(self, *args, **kwargs):
        """Deprecated: use :meth:`save_parquet_and_json` instead.

        This alias is retained so external callers don't break immediately. It
        no longer writes a pickle file — only parquet + json. The name is
        misleading and will be removed in a future release.
        """
        import warnings

        warnings.warn(
            "save_pickle_and_json is deprecated and no longer writes a pickle file; "
            "use save_parquet_and_json instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.save_parquet_and_json(*args, **kwargs)

    class _NumpyEncoder(json.JSONEncoder):
        """JSON encoder that handles numpy types — used by the JSON fallback
        path for cells pyarrow can't infer a uniform schema for.
        """

        def default(self, o: Any) -> Any:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    _TUPLE_FIELD_PREFIX = "_t"  # tuple round-trip marker for _to/_from nested

    @staticmethod
    def _to_nested_python(v):
        """Convert numpy/dict/tuple cells to nested Python so pyarrow can
        infer native list/struct types.  Tuples become structs with keys
        ``_t0``, ``_t1``, … so heterogeneous-shape elements survive the
        round trip.
        """
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): WARSerializationMixin._to_nested_python(vv) for k, vv in v.items()}
        if isinstance(v, tuple):
            return {
                f"{WARSerializationMixin._TUPLE_FIELD_PREFIX}{i}": WARSerializationMixin._to_nested_python(x)
                for i, x in enumerate(v)
            }
        if isinstance(v, list):
            return [WARSerializationMixin._to_nested_python(x) for x in v]
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    @staticmethod
    def _canonicalise_band_dict(d: dict) -> dict:
        """Reorder *d* by ``constants.BAND_NAMES`` when it has any band-name keys.

        Pyarrow alphabetises struct fields on the read side of a parquet
        round-trip (``Table.to_pandas()``), so canonical-order band dicts
        written to disk come back as ``{"alpha", "beta", "delta", "gamma",
        "theta"}``. Best-fit reorder: any band-name keys present are
        promoted to the front in canonical (FREQ_BANDS insertion) order,
        any non-band keys are appended in their original order.
        Idempotent — an already-canonical dict round-trips to itself.
        Dicts with zero band-name overlap are returned unchanged.
        """
        band_set = set(constants.BAND_NAMES)
        if not (set(d.keys()) & band_set):
            return d
        band_keys = [b for b in constants.BAND_NAMES if b in d]
        other_keys = [k for k in d if k not in band_set]
        return {**{b: d[b] for b in band_keys}, **{k: d[k] for k in other_keys}}

    @staticmethod
    def _normalize_arrow_cell(v):
        """Convert pyarrow's ndarray-leafed cells back to plain Python lists,
        reconstruct ``_t0``/``_t1``/… structs as tuples, and canonicalise
        band-keyed dicts via :meth:`_canonicalise_band_dict`.
        """
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                return [WARSerializationMixin._normalize_arrow_cell(x) for x in v]
            return v.tolist()
        if isinstance(v, dict):
            prefix = WARSerializationMixin._TUPLE_FIELD_PREFIX
            keys = list(v.keys())
            if keys and all(k == f"{prefix}{i}" for i, k in enumerate(sorted(keys, key=lambda k: int(k[len(prefix):]) if k.startswith(prefix) and k[len(prefix):].isdigit() else -1))):
                ordered = sorted(keys, key=lambda k: int(k[len(prefix):]))
                return tuple(
                    WARSerializationMixin._normalize_arrow_cell(v[k]) for k in ordered
                )
            decoded = {k: WARSerializationMixin._normalize_arrow_cell(vv) for k, vv in v.items()}
            return WARSerializationMixin._canonicalise_band_dict(decoded)
        if isinstance(v, list):
            return [WARSerializationMixin._normalize_arrow_cell(x) for x in v]
        return v

    @staticmethod
    def _encode_df_for_parquet(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Return a copy of *df* with complex/object columns converted to
        nested Python structures (lists / dicts / scalars).

        Pyarrow can store these as native list/struct columns directly,
        without any JSON string intermediate.  Encoded column names are
        returned so the caller can stamp them into parquet schema metadata.

        Returns:
            (encoded_df, encoded_columns) — the modified DataFrame and the
            list of column names that were converted.
        """
        df_copy = df.copy()
        encoded_cols: list[str] = []
        for col in df_copy.columns:
            ser = df_copy[col]
            needs_encoding = False
            if ser.dtype == object:
                sample = ser.dropna().head(20)
                for v in sample:
                    if not isinstance(v, (str, int, float, bool, type(None))):
                        needs_encoding = True
                        break

            if needs_encoding:
                encoded_cols.append(col)
                df_copy[col] = ser.apply(WARSerializationMixin._to_nested_python)

        return df_copy, encoded_cols

    @staticmethod
    def _df_to_arrow_table(
        df: pd.DataFrame, encoded_cols: list[str] | None = None
    ):
        """Encode a DataFrame for parquet write. Shared by eager + streaming saves.

        Columns whose name is in :data:`constants.FEATURE_TYPES` with a non-LINEAR
        type are encoded as native nested pyarrow types (via ``_to_nested_python``
        + ``pa.array``) with a per-cell JSON fallback for shapes pyarrow can't
        infer.  tz-aware datetimes are normalized to UTC.  Pass an explicit
        ``encoded_cols`` list to override the schema-based detection (e.g. when
        round-tripping non-WAR DataFrames).

        Returns the table (without schema metadata stamped) and the list of
        columns that ended up encoded.
        """
        import pyarrow as pa

        encoded_out: list[str] = list(encoded_cols) if encoded_cols else []
        columns: dict[str, Any] = {}
        for col in df.columns:
            ser = df[col]
            needs_encoding = col in encoded_out or _column_needs_encoding(col)
            if needs_encoding and col not in encoded_out:
                encoded_out.append(col)
            if needs_encoding:
                nested = [WARSerializationMixin._to_nested_python(x) for x in ser]
                try:
                    columns[col] = pa.array(nested)
                except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError, AttributeError, TypeError, ValueError):
                    columns[col] = [
                        json.dumps(x, cls=WARSerializationMixin._NumpyEncoder, ensure_ascii=False)
                        for x in nested
                    ]
                del nested
            else:
                if ser.dtype == object:
                    non_null = ser.dropna()
                    if len(non_null) > 0 and isinstance(non_null.iloc[0], pd.Timestamp):
                        ser = pd.to_datetime(ser, errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(ser):
                    tz = getattr(getattr(ser, "dt", None), "tz", None)
                    if tz is not None:
                        if ser.isna().any():
                            ser = ser.dt.tz_convert("UTC").dt.tz_localize(None)
                        else:
                            ser = ser.dt.tz_convert("UTC")
                columns[col] = ser.to_numpy()
        return pa.table(columns), encoded_out

    @staticmethod
    def _try_load_json(v):
        """Legacy JSON-string decoder; identity on non-strings."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    @staticmethod
    def _decode_df_from_parquet(
        df: pd.DataFrame,
        encoded_cols: list[str],
        encoding_version: int = 1,
    ) -> pd.DataFrame:
        """Decode complex columns to plain Python.

        ``encoding_version`` is read from the parquet's ``neurodent.encoding_version``
        schema metadata at the call site:

        - 1 (or missing) → legacy JSON-string cells; decoded with :func:`json.loads`.
        - 2              → native nested cells (pa.list/struct/binary);
          normalized with :meth:`_normalize_arrow_cell`.
        """
        decoder = (
            WARSerializationMixin._normalize_arrow_cell
            if encoding_version >= 2
            else WARSerializationMixin._try_load_json
        )
        for col in encoded_cols:
            if col in df.columns:
                df[col] = df[col].apply(decoder)
        return df

    @classmethod
    def scan_parquet_and_json(cls, folder_path: str | Path, filename: str = "war"):
        """Open a WAR as a :class:`LazyWindowAnalysisResult` (no DataFrame materialised).

        The returned object mirrors the mutator API of ``WindowAnalysisResult``
        (``reorder_and_pad_channels``, ``add_unique_hash``, ``apply_filters``,
        ``aggregate_time_windows``) but records each call as a ``Transform``;
        :meth:`LazyWindowAnalysisResult.save_parquet_and_json` runs the chain
        against batched parquet reads.

        Args:
            folder_path: directory containing ``<filename>.parquet`` and
                ``<filename>.json``.  Matches the first positional of
                :meth:`load_parquet_and_json`.
            filename: stem shared by the two sidecar files.  Defaults to
                ``"war"`` (the convention used by every NeuRodent pipeline
                rule).

        Returns:
            LazyWindowAnalysisResult: streaming handle with the same mutator
            API as :class:`WindowAnalysisResult`.
        """
        from .streaming import LazyWindowAnalysisResult

        return LazyWindowAnalysisResult(folder_path, filename=filename)

    @classmethod
    def load_parquet_and_json(cls, folder_path=None, parquet_name=None, json_name=None, filename=None):
        """Load WindowAnalysisResult from folder.

        Reads ``war.parquet`` (the result DataFrame) plus ``war.json`` (the
        WAR metadata: animal_id, channel_names, bad_channels_dict, etc.).

        For backward compatibility, if the resolved parquet file does not
        exist but a matching ``.pkl`` file does, the loader falls back to
        reading the legacy pickle format. No pickle files are written.

        Args:
            folder_path (str, optional): Path of folder containing .parquet and .json files. Defaults to None.
            parquet_name (str, optional): Name of the parquet file. Can be just the filename (e.g. "war.parquet")
                or a path relative to folder_path (e.g. "subdir/war.parquet"). If None and folder_path is provided,
                expects exactly one .parquet file in folder_path. Defaults to None.
            json_name (str, optional): Name of the JSON file. Can be just the filename (e.g. "war.json")
                or a path relative to folder_path (e.g. "subdir/war.json"). If None and folder_path is provided,
                expects exactly one .json file in folder_path. Defaults to None.
            filename (str, optional): Shorthand stem shared by the parquet
                and JSON sidecars (i.e. ``<filename>.parquet`` +
                ``<filename>.json``). Matches the ``filename`` kwarg of
                :meth:`scan_parquet_and_json` so eager and lazy entry points have the
                same simple-case call shape. Ignored when ``parquet_name``
                or ``json_name`` is also provided. Defaults to None
                (auto-discovery).

        Raises:
            ValueError: folder_path does not exist
            ValueError: Expected exactly one parquet and one json file in folder_path (when parquet_name/json_name not specified)
            FileNotFoundError: Specified parquet_name or json_name not found

        Returns:
            result: WindowAnalysisResult object
        """
        if filename is not None:
            if parquet_name is None:
                parquet_name = f"{filename}.parquet"
            if json_name is None:
                json_name = f"{filename}.json"
        if folder_path is not None:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise ValueError(f"Folder path {folder_path} does not exist")

            if parquet_name is not None:
                # Handle parquet_name as either absolute path or relative to folder_path
                p = Path(parquet_name)
                parquet_path = p if p.is_absolute() else folder_path / parquet_name
                if not parquet_path.exists():
                    # Allow falling back to legacy pickle with the same stem
                    legacy_pkl = parquet_path.with_suffix(".pkl")
                    if not legacy_pkl.exists():
                        raise FileNotFoundError(
                            f"Parquet file not found: {parquet_path} (and no legacy pickle at {legacy_pkl})"
                        )
            else:
                pq_files = list(folder_path.glob("*.parquet"))
                if len(pq_files) == 1:
                    parquet_path = pq_files[0]
                elif len(pq_files) == 0:
                    # Legacy layout: fall back to a single pickle file
                    pkl_files = list(folder_path.glob("*.pkl"))
                    if len(pkl_files) != 1:
                        raise ValueError(
                            f"Expected exactly one parquet file in {folder_path}, found {len(pq_files)}"
                        )
                    parquet_path = pkl_files[0].with_suffix(".parquet")
                else:
                    raise ValueError(
                        f"Expected exactly one parquet file in {folder_path}, found {len(pq_files)}"
                    )

            if json_name is not None:
                # Handle json_name as either absolute path or relative to folder_path
                jp = Path(json_name)
                json_path = jp if jp.is_absolute() else folder_path / json_name
                if not json_path.exists():
                    raise FileNotFoundError(f"JSON file not found: {json_path}")
            else:
                # Prefer the JSON file that shares the parquet stem
                # (e.g. war.parquet → war.json).  This avoids false
                # positives from legacy sidecar files such as
                # *.parquet.meta.json that may coexist in the folder.
                json_path = parquet_path.with_suffix(".json")
                if not json_path.exists():
                    json_files = list(folder_path.glob("*.json"))
                    if len(json_files) != 1:
                        raise ValueError(
                            f"Expected exactly one json file in {folder_path}, found {len(json_files)}"
                        )
                    json_path = json_files[0]
        else:
            if parquet_name is None or json_name is None:
                raise ValueError(
                    "Either folder_path must be provided, or both parquet_name and json_name must be provided as absolute paths"
                )

            parquet_path = Path(parquet_name)
            json_path = Path(json_name)

            if not parquet_path.exists() and not parquet_path.with_suffix(".pkl").exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")

        data: pd.DataFrame
        if parquet_path.exists():
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(parquet_path)
                # Encoded-column list + encoding_version are stored in schema metadata
                encoded_cols: list[str] = []
                encoding_version: int = 1
                schema_meta = table.schema.metadata or {}
                if b"neurodent" in schema_meta:
                    nd_meta = json.loads(schema_meta[b"neurodent"])
                    encoded_cols = nd_meta.get("encoded_columns", [])
                    encoding_version = nd_meta.get("encoding_version", 1)
                else:
                    # Fallback: try legacy .parquet.meta.json sidecar file
                    legacy_meta_path = parquet_path.parent / (
                        parquet_path.name + ".meta.json"
                    )
                    if legacy_meta_path.exists():
                        with open(legacy_meta_path, "r") as mf:
                            pq_meta = json.load(mf)
                        encoded_cols = pq_meta.get("encoded_columns", [])

                # self_destruct + split_blocks: free Arrow buffers during
                # conversion and prevent giant BlockManager allocations.
                data = table.to_pandas(self_destruct=True, split_blocks=True)
                del table
                data = cls._decode_df_from_parquet(data, encoded_cols, encoding_version=encoding_version)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                legacy_pkl = parquet_path.with_suffix(".pkl")
                if not legacy_pkl.exists():
                    raise
                logging.warning(
                    f"Failed to load parquet WAR ({parquet_path}): {e}, falling back to legacy pickle"
                )
                with open(legacy_pkl, "rb") as f:
                    data = pd.read_pickle(f)
        else:
            # Parquet missing — try the legacy pickle fallback
            legacy_pkl = parquet_path.with_suffix(".pkl")
            logging.warning(
                f"Parquet WAR not found at {parquet_path}, loading legacy pickle at {legacy_pkl}"
            )
            with open(legacy_pkl, "rb") as f:
                data = pd.read_pickle(f)

        # Validate the JSON half of the pair explicitly: a partial/corrupt
        # sidecar (e.g. from an interrupted write) should fail with a clear,
        # actionable error so the WAR is regenerated, not crash with an opaque
        # JSONDecodeError downstream.
        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(
                f"WAR JSON sidecar {json_path} is missing or corrupt ({e}); "
                f"the parquet/JSON pair is incomplete and the WAR must be regenerated."
            ) from e
        # Back-compat: older WAR sidecars carry "assume_from_number"; the field was
        # removed (channel resolution is now exact-only), so drop it before construction.
        metadata.pop("assume_from_number", None)
        return cls(data, **metadata)

    @classmethod
    def load_pickle_and_json(cls, folder_path=None, pickle_name=None, json_name=None):
        """Deprecated: use :meth:`load_parquet_and_json` instead.

        This alias is retained so external callers don't break immediately.
        The loader already prefers parquet over pickle; this shim maps the
        old ``pickle_name`` argument to ``parquet_name`` (the parquet file
        will be resolved from the same stem).
        """
        import warnings

        warnings.warn(
            "load_pickle_and_json is deprecated; use load_parquet_and_json instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        parquet_name = None
        if pickle_name is not None:
            p = Path(pickle_name)
            parquet_name = str(p.with_suffix(".parquet"))
        return cls.load_parquet_and_json(
            folder_path=folder_path,
            parquet_name=parquet_name,
            json_name=json_name,
        )
