"""Streaming engine for ``WindowAnalysisResult``.

This module hosts the ``Transform`` protocol, the ``StreamContext`` carried
through transforms during execution, and ``LazyWindowAnalysisResult`` — a
streaming sibling of :class:`WindowAnalysisResult` that mirrors its mutator
API.  Concrete transforms (``ReorderAndPadChannels``, ``AddUniqueHash``,
``ApplyFilters``, ``AggregateTimeWindows``) are also defined here and shared
with the eager mutators so per-batch logic lives in exactly one place.

Usage::

    war = visualization.WindowAnalysisResult.scan_parquet_and_json(folder_path)
    war.reorder_and_pad_channels(target_channels, use_abbrevs=True)
    war.add_unique_hash(unique_hash_length)
    war.save_parquet_and_json(dst_folder)

Peak memory is bounded by ``batch_size`` rows for pass-through transforms.
Some transforms (e.g. cross-row filters) may materialise full-column stats and/or
a full ``(W, C)`` mask, so peak memory can scale with total WAR size in those cases.
"""

from __future__ import annotations

import abc
import gc
import json
import logging
import secrets
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import constants
from ..core.utils import resolve_channels, atomic_write_json, safe_unlink, slugify
from .feature_handlers import handler_for
from .filters import (
    FILTER_REGISTRY,
    ChannelInfo,
    FilterScope,
    compute_filter_mask,
    update_bad_channels_dict_from_config,
)

if TYPE_CHECKING:
    from .results import WindowAnalysisResult


# ---------------------------------------------------------------------------
# Context + Transform protocol
# ---------------------------------------------------------------------------


@dataclass
class StreamContext:
    """Mutable runtime context threaded through transforms.

    ``channel_info`` reflects the *current* channel ordering — transforms
    that reorder channels update it via :meth:`Transform.update_metadata` so
    later transforms see the new ordering. ``metadata`` is the JSON sidecar
    being assembled; ``n_rows_total`` is the total source row count
    (available to transforms that need a global denominator).
    """

    channel_info: ChannelInfo
    metadata: dict
    n_rows_total: int
    extra: dict = field(default_factory=dict)


class Transform(abc.ABC):
    """Composable per-batch transform used by both eager and lazy paths."""

    needs_pass1: bool = False
    required_columns_pass1: tuple[str, ...] = ()
    is_aggregating: bool = False

    def bind(self, ctx: StreamContext) -> None:
        """Called once before any pass1/apply.  Stashes ``ctx``."""
        self._ctx = ctx

    def pass1(self, df_stats: pd.DataFrame) -> None:
        """Optional pre-scan over cheap columns. Override when ``needs_pass1=True``."""
        return None

    @abc.abstractmethod
    def apply(self, batch_df: pd.DataFrame) -> pd.DataFrame | None:
        """Transform a batch. Return None for aggregating transforms."""

    def finalize(self) -> pd.DataFrame | None:
        """Emit the final aggregated DataFrame (aggregating transforms only)."""
        return None

    def update_metadata(self, metadata: dict) -> dict:
        """Apply metadata mutations (animal_id, bad_channels_dict, etc.)."""
        return metadata


# ---------------------------------------------------------------------------
# Concrete transforms
# ---------------------------------------------------------------------------


class ReorderAndPadChannels(Transform):
    """Reorder + pad channel axis to match ``target_channels``."""

    def __init__(self, target_channels: list[str] | None = None, use_abbrevs: bool = True):
        if target_channels is None:
            target_channels = list(constants.CHANNEL_ABBREVS)
        duplicates = [ch for ch in target_channels if target_channels.count(ch) > 1]
        if duplicates:
            raise ValueError(
                f"Target channels must be unique. Found duplicates: {duplicates}"
            )
        self.target_channels = list(target_channels)
        self.use_abbrevs = use_abbrevs

    def bind(self, ctx):
        super().bind(ctx)
        self.channel_map = {ch: i for i, ch in enumerate(self.target_channels)}
        ci = ctx.channel_info
        self.source_channels = list(ci.channel_abbrevs if self.use_abbrevs else ci.channel_names)
        valid = [ch for ch in self.source_channels if ch in self.channel_map]
        if not valid:
            warnings.warn(
                f"None of the channel names {self.source_channels} were found in target channels "
                f"{self.target_channels}. Is use_abbrevs correctly set?"
            )
        # Advance ctx.channel_info so subsequent transforms see the post-reorder
        # channel set during their own bind() / pass1().  channel_names becomes
        # target_channels; channel_abbrevs is re-derived from those names.
        from ..core.utils import resolve_channels as _abbrev
        ctx.channel_info = ChannelInfo(
            channel_names=list(self.target_channels),
            channel_abbrevs=_abbrev(list(self.target_channels)),
        )

    def apply(self, batch_df):
        for feature in batch_df.columns:
            if feature in constants.FEATURES:
                handler = handler_for(feature)
                batch_df[feature] = handler.reorder_pad(
                    batch_df[feature], self.channel_map, self.source_channels, self.target_channels
                )
        return batch_df

    def update_metadata(self, metadata):
        metadata = dict(metadata)
        metadata["channel_names"] = list(self.target_channels)
        return metadata


class AddUniqueHash(Transform):
    """Append a random hex suffix to ``animal_id`` (and rewrite ``animal`` / ``animalday`` cells)."""

    def __init__(self, nbytes: int | None = None):
        self.nbytes = nbytes
        self.original_animal_id: str | None = None
        self.new_animal_id: str | None = None

    def bind(self, ctx):
        super().bind(ctx)
        self.original_animal_id = ctx.metadata.get("animal_id", "") or ""
        self.new_animal_id = f"{self.original_animal_id}_{secrets.token_hex(self.nbytes)}"

    def apply(self, batch_df):
        if "animal" in batch_df.columns:
            batch_df["animal"] = self.new_animal_id
        if "animalday" in batch_df.columns:
            batch_df["animalday"] = batch_df["animalday"].str.replace(
                self.original_animal_id, self.new_animal_id
            )
        return batch_df

    def update_metadata(self, metadata):
        metadata = dict(metadata)
        metadata["animal_id"] = self.new_animal_id
        return metadata


class ApplyFilters(Transform):
    """Apply a filter_config via ``FILTER_REGISTRY``.

    Always uses a pass-1 stats scan so per-row + cross-row + post-mask
    filters share the same code path. The full ``(W, C)`` bool mask lives
    in this transform; per-batch :meth:`apply` slices it by row offset.
    """

    needs_pass1 = True

    def __init__(self, filter_config: dict, min_valid_channels: int = 3):
        self.filter_config = dict(filter_config) if filter_config else {}
        self.min_valid_channels = min_valid_channels
        cols: set[str] = set()
        for name in self.filter_config:
            spec = FILTER_REGISTRY.get(name)
            if spec is None:
                raise ValueError(
                    f"Unknown filter: {name}. Available: {sorted(FILTER_REGISTRY)}"
                )
            cols.update(spec.required_columns)
        # animalday is required for the bad_channels_dict bookkeeping after either reject_channels* filter.
        if "reject_channels" in self.filter_config or "reject_channels_by_session" in self.filter_config:
            cols.add("animalday")
        self.required_columns_pass1 = tuple(sorted(cols))
        self._mask: np.ndarray | None = None
        self._row_offset = 0
        self._animaldays: list[str] = []

    def pass1(self, df_stats):
        ctx = self._ctx
        self._mask = compute_filter_mask(
            df_stats, self.filter_config, ctx.channel_info, n_windows=ctx.n_rows_total
        )
        valid_per_window = np.sum(self._mask, axis=1)
        row_mask = valid_per_window >= self.min_valid_channels
        self._mask = self._mask & row_mask[:, np.newaxis]
        if "animalday" in df_stats.columns:
            self._animaldays = list(df_stats["animalday"].unique())

    def apply(self, batch_df):
        n = len(batch_df)
        batch_mask = self._mask[self._row_offset : self._row_offset + n]
        self._row_offset += n
        for feature in batch_df.columns:
            if feature in constants.FEATURES:
                handler = handler_for(feature)
                batch_df[feature] = handler.apply_mask(batch_df[feature], batch_mask)
        return batch_df

    def update_metadata(self, metadata):
        ctx = self._ctx
        existing = metadata.get("bad_channels_dict", {}) or {}
        updated = update_bad_channels_dict_from_config(
            existing, self.filter_config, ctx.channel_info, self._animaldays
        )
        metadata = dict(metadata)
        metadata["bad_channels_dict"] = updated
        return metadata


_SPECIAL_AGG_COLS = {"animalday", "isday", "duration", "endfile", "timestamp"}


class AggregateTimeWindows(Transform):
    """Streaming duration-weighted average aggregated by ``groupby`` keys.

    Single ``is_aggregating=True`` transform per chain (must be terminal).
    Per-batch :meth:`apply` updates per-group accumulators; :meth:`finalize`
    emits the small aggregated DataFrame.
    """

    is_aggregating = True

    def __init__(self, groupby: list[str] | str = ("animalday", "isday")):
        if isinstance(groupby, str):
            groupby = [groupby]
        groupby = list(groupby)
        if not all(col in ("animalday", "isday") for col in groupby):
            raise ValueError(
                f"groupby must be from ['animalday', 'isday']. Got {groupby}"
            )
        self.groupby = groupby
        self._group_state: dict[Any, dict[str, dict]] = {}
        self._constant_cols: dict[Any, dict[str, Any]] = {}
        self._first_timestamp: dict[Any, Any] = {}
        self._last_endfile: dict[Any, Any] = {}
        self._duration_sum: dict[Any, float] = {}
        self._features_seen: set[str] = set()
        self._non_groupby_pair: str | None = None
        self._column_order: list[str] | None = None

    def bind(self, ctx):
        super().bind(ctx)
        # The eager path produces a row for each groupby combo; ``isday`` / ``animalday``
        # not in groupby becomes None in the output row.
        for col in ("animalday", "isday"):
            if col not in self.groupby:
                self._non_groupby_pair = col

    def apply(self, batch_df):
        if self._column_order is None:
            self._column_order = list(batch_df.columns)

        non_feature = [c for c in batch_df.columns if c not in constants.FEATURES]
        feature_cols = [c for c in batch_df.columns if c in constants.FEATURES]

        # Group by the configured keys; iterate per group.
        grouped = batch_df.groupby(self.groupby, sort=False)
        for group_key, group_df in grouped:
            # Normalise to tuple keys so dicts have a stable hash type.
            key = group_key if isinstance(group_key, tuple) else (group_key,)

            # Constant columns: track unique values; raise if multi-valued.
            cc = self._constant_cols.setdefault(key, {})
            for col in non_feature:
                if col in self.groupby or col in _SPECIAL_AGG_COLS:
                    continue
                vals = group_df[col].unique()
                if len(vals) != 1:
                    raise ValueError(
                        f"Column {col} is not constant in group {group_key}"
                    )
                v = vals[0]
                if col in cc:
                    if cc[col] != v and not (pd.isna(cc[col]) and pd.isna(v)):
                        raise ValueError(
                            f"Column {col} is not constant in group {group_key}"
                        )
                else:
                    cc[col] = v

            # Special aggregated columns
            if "duration" in group_df.columns:
                self._duration_sum[key] = self._duration_sum.get(key, 0.0) + float(
                    group_df["duration"].sum()
                )
            if "timestamp" in group_df.columns and key not in self._first_timestamp:
                self._first_timestamp[key] = group_df["timestamp"].iloc[0]
            if "endfile" in group_df.columns:
                self._last_endfile[key] = group_df["endfile"].iloc[-1]

            # Feature accumulators
            durations = (
                group_df["duration"].to_numpy(dtype=float)
                if "duration" in group_df.columns
                else np.ones(len(group_df), dtype=float)
            )
            state = self._group_state.setdefault(key, {})
            for feature in feature_cols:
                self._features_seen.add(feature)
                handler = handler_for(feature)
                handler.accumulate(state.setdefault(feature, {}), group_df[feature], durations)

        return None  # aggregating transform — no per-batch output

    def finalize(self):
        rows = []
        for key, state in self._group_state.items():
            row: dict[str, Any] = {}
            for col_name, val in zip(self.groupby, key):
                row[col_name] = val
            if self._non_groupby_pair is not None:
                row[self._non_groupby_pair] = None
            for col, val in self._constant_cols.get(key, {}).items():
                row[col] = val
            if key in self._duration_sum:
                row["duration"] = self._duration_sum[key]
            if key in self._first_timestamp:
                row["timestamp"] = self._first_timestamp[key]
            if key in self._last_endfile:
                row["endfile"] = self._last_endfile[key]
            for feature in self._features_seen:
                if feature in state:
                    handler = handler_for(feature)
                    row[feature] = handler.finalize(state[feature])
            rows.append(row)

        df = pd.DataFrame(rows)
        # Preserve original column order where possible (other columns appended at the end).
        if self._column_order is not None:
            ordered = [c for c in self._column_order if c in df.columns]
            extras = [c for c in df.columns if c not in ordered]
            df = df[ordered + extras]
        return df.reset_index(drop=True)

    def update_metadata(self, metadata):
        metadata = dict(metadata)
        metadata["suppress_short_interval_error"] = True
        return metadata


# ---------------------------------------------------------------------------
# LazyWindowAnalysisResult — the streaming engine
# ---------------------------------------------------------------------------


class LazyWindowAnalysisResult:
    """Streaming sibling of :class:`WindowAnalysisResult`.

    Constructed via :meth:`WindowAnalysisResult.scan_parquet_and_json`. Mutator methods
    record :class:`Transform` instances; :meth:`save_parquet_and_json` runs
    the chain against batched parquet reads and writes the output without
    ever materialising the full WAR in memory.
    """

    def __init__(
        self,
        folder_path: str | Path,
        filename: str = "war",
    ):
        folder_path = Path(folder_path)
        self._src_folder = folder_path
        self._src_filename = filename
        self._src_parquet = folder_path / f"{filename}.parquet"
        self._src_json = folder_path / f"{filename}.json"
        self._metadata: dict = json.loads(self._src_json.read_text())
        # Channel info derived from JSON metadata; no DataFrame materialised yet.
        channel_names = list(self._metadata.get("channel_names") or [])
        channel_abbrevs = resolve_channels(channel_names)
        self._channel_info = ChannelInfo(
            channel_names=channel_names,
            channel_abbrevs=channel_abbrevs,
        )
        # Inspect the parquet schema for encoded-column metadata + row count.
        pf = pq.ParquetFile(self._src_parquet)
        schema_meta = pf.schema_arrow.metadata or {}
        encoded_cols: list[str] = []
        encoding_version: int = 1
        if b"neurodent" in schema_meta:
            nd_meta = json.loads(schema_meta[b"neurodent"])
            encoded_cols = nd_meta.get("encoded_columns", [])
            encoding_version = nd_meta.get("encoding_version", 1)
        self._encoded_columns = encoded_cols
        self._encoding_version = encoding_version
        self._n_rows_total = pf.metadata.num_rows if pf.metadata is not None else None
        self._available_columns = set(pf.schema_arrow.names)
        del pf  # release the file handle; we'll reopen during save
        self._pending: list[Transform] = []

    # ---- Metadata convenience (read-only access without loading the DataFrame) ----

    @property
    def metadata(self) -> dict:
        """The JSON sidecar metadata as a dict (read-only snapshot)."""
        return dict(self._metadata)

    @property
    def animal_id(self) -> str:
        return self._metadata.get("animal_id", "") or ""

    @property
    def path_safe_animal_id(self) -> str:
        """Slugified :attr:`animal_id` for filesystem paths.

        See :attr:`WindowAnalysisResult.path_safe_animal_id`.
        """
        return slugify(self.animal_id)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_info.channel_names)

    @property
    def channel_abbrevs(self) -> list[str]:
        return list(self._channel_info.channel_abbrevs)

    @property
    def lof_scores_dict(self) -> dict:
        return dict(self._metadata.get("lof_scores_dict") or {})

    @property
    def bad_channels_dict(self) -> dict:
        return dict(self._metadata.get("bad_channels_dict") or {})

    def get_bad_channels_by_lof_threshold(self, lof_threshold: float) -> dict:
        """Same as :meth:`WindowAnalysisResult.get_bad_channels_by_lof_threshold`,
        but reads from JSON metadata (no DataFrame materialisation)."""
        lof = self.lof_scores_dict
        if not lof:
            raise ValueError(
                "LOF scores not available in this WAR. Compute LOF scores first."
            )
        bad: dict[str, list[str]] = {}
        for animalday, lof_data in lof.items():
            if "lof_scores" in lof_data and "channel_names" in lof_data:
                scores = np.array(lof_data["lof_scores"])
                ch_names = lof_data["channel_names"]
                is_inlier = scores < lof_threshold
                bad[animalday] = [ch_names[i] for i in np.where(~is_inlier)[0]]
            else:
                raise ValueError(f"LOF scores not available for {animalday}")
        return bad

    # ---- Chained mutators (record transforms) ----

    def reorder_and_pad_channels(
        self, target_channels: list[str] | None = None, use_abbrevs: bool = True
    ) -> "LazyWindowAnalysisResult":
        # Mirror WindowAnalysisResult.reorder_and_pad_channels: a None target
        # defaults to the canonical channel list (constants.CHANNEL_ABBREVS).
        if target_channels is None:
            target_channels = list(constants.CHANNEL_ABBREVS)
        self._pending.append(ReorderAndPadChannels(target_channels, use_abbrevs))
        return self

    def add_unique_hash(
        self, nbytes: int | None = None
    ) -> "LazyWindowAnalysisResult":
        self._pending.append(AddUniqueHash(nbytes))
        return self

    def apply_filters(
        self, filter_config: dict | None = None, min_valid_channels: int = 3
    ) -> "LazyWindowAnalysisResult":
        if filter_config is None:
            filter_config = {
                "logrms_range": {"z_range": 3},
                "high_rms": {"max_rms": 500},
                "low_rms": {"min_rms": 50},
                "high_beta": {"max_beta_prop": 0.4},
                "reject_channels_by_session": {},
            }
        self._pending.append(ApplyFilters(filter_config, min_valid_channels))
        return self

    def aggregate_time_windows(
        self, groupby: list[str] | str = ("animalday", "isday")
    ) -> "LazyWindowAnalysisResult":
        self._pending.append(AggregateTimeWindows(groupby))
        return self

    # ---- Terminal: stream + write ----

    def save_parquet_and_json(
        self,
        folder: str | Path,
        *,
        filename: str = "war",
        batch_size: int = 5000,
    ) -> None:
        """Execute the pending transform chain against batched parquet reads.

        Pass-through chains write each transformed batch incrementally via
        ``pq.ParquetWriter``.  Aggregating chains (exactly one
        ``is_aggregating`` transform, must be terminal) fold batches into
        accumulators and write the small final DataFrame via
        :meth:`WindowAnalysisResult._df_to_arrow_table` once.
        """
        from .results import WindowAnalysisResult  # local import to avoid cycle

        dst_folder = Path(folder)
        dst_folder.mkdir(parents=True, exist_ok=True)
        dst_parquet = dst_folder / f"{filename}.parquet"
        dst_json = dst_folder / f"{filename}.json"
        # Stream the parquet to a unique temp sibling and only rename it into
        # place on success, so an interrupted/failed write never leaves a partial
        # .parquet that a downstream rule would treat as a valid output.
        tmp_parquet = dst_parquet.with_name(f"{dst_parquet.name}.{secrets.token_hex(8)}.tmp")

        ctx = StreamContext(
            channel_info=self._channel_info,
            metadata=dict(self._metadata),
            n_rows_total=self._n_rows_total or 0,
        )
        # Aggregating chains: enforce at most one terminal aggregator.
        aggregators = [t for t in self._pending if t.is_aggregating]
        if len(aggregators) > 1:
            raise ValueError(
                "Only one aggregating transform supported per chain"
            )
        if aggregators and self._pending[-1] is not aggregators[0]:
            raise ValueError("Aggregating transform must be the last in the chain")
        is_aggregating = bool(aggregators)

        for t in self._pending:
            t.bind(ctx)

        # PASS 1 — cheap pre-scan for transforms that need cross-row state.
        pass1_ts = [t for t in self._pending if t.needs_pass1]
        if pass1_ts:
            cols: set[str] = set()
            for t in pass1_ts:
                cols.update(t.required_columns_pass1)
            cols &= self._available_columns
            self._run_pass1(pass1_ts, list(cols), batch_size)

        # PASS 2 — streaming apply.
        writer: pq.ParquetWriter | None = None
        target_schema: pa.Schema | None = None
        out_encoded_cols: list[str] | None = None
        pq_file = pq.ParquetFile(self._src_parquet)
        for batch in pq_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas(self_destruct=True, split_blocks=True)
            df = WindowAnalysisResult._decode_df_from_parquet(
                df, self._encoded_columns, encoding_version=self._encoding_version
            )
            for t in self._pending:
                result = t.apply(df)
                if result is None:
                    df = None
                    break
                df = result
            if df is None:
                continue  # aggregating chain — nothing to write per batch

            table, batch_encoded = WindowAnalysisResult._df_to_arrow_table(
                df, out_encoded_cols
            )
            if writer is None:
                out_encoded_cols = batch_encoded
                out_meta = json.dumps(
                    {"encoded_columns": out_encoded_cols, "encoding_version": 2}
                ).encode()
                target_schema = table.schema.with_metadata({b"neurodent": out_meta})
                writer = pq.ParquetWriter(
                    str(tmp_parquet), target_schema, compression="zstd", compression_level=4
                )
                table = table.replace_schema_metadata({b"neurodent": out_meta})
            else:
                # Re-stamp metadata on subsequent batches (cast if pyarrow inferred a slightly different schema).
                if table.schema != target_schema.remove_metadata():
                    try:
                        table = table.cast(target_schema.remove_metadata())
                    except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError):
                        pass
                table = table.replace_schema_metadata(target_schema.metadata or {})
            writer.write_table(table)
            del df, table
            gc.collect()
        del pq_file

        if is_aggregating:
            final_df = aggregators[0].finalize()
            if final_df is None:
                final_df = pd.DataFrame()
            table, encoded_cols = WindowAnalysisResult._df_to_arrow_table(final_df)
            out_meta = json.dumps(
                {"encoded_columns": encoded_cols, "encoding_version": 2}
            ).encode()
            existing_meta = table.schema.metadata or {}
            merged_meta = {**existing_meta, b"neurodent": out_meta}
            table = table.replace_schema_metadata(merged_meta)
            pq.write_table(
                table, str(tmp_parquet), compression="zstd", compression_level=4
            )
        elif writer is not None:
            writer.close()

        # Commit the parquet: atomically rename temp → final if anything was
        # written. A non-aggregating chain over an empty source writes nothing,
        # in which case there is no temp file to commit (clean up if it exists).
        if is_aggregating or writer is not None:
            tmp_parquet.replace(dst_parquet)
        else:
            safe_unlink(tmp_parquet)

        # Compose output metadata via each transform's update hook.
        out_metadata = dict(self._metadata)
        for t in self._pending:
            out_metadata = t.update_metadata(out_metadata)

        atomic_write_json(dst_json, out_metadata, indent=2)

        logging.info(
            f"Lazy-saved WAR: {self._src_parquet} -> {dst_parquet} "
            f"(batch_size={batch_size}, transforms={[type(t).__name__ for t in self._pending]})"
        )

    def _run_pass1(
        self,
        pass1_ts: list[Transform],
        load_cols: list[str],
        batch_size: int,
    ) -> None:
        """Stream the cheap columns once to build per-transform cross-row state.

        For chained transforms (e.g. ``Reorder → ApplyFilters``), each
        pass1-eligible transform needs its cheap columns *as they would
        appear after the preceding pass-through transforms have run*. The
        engine reads each batch with the union of pass-1 columns, then
        replays the preceding pass-through transforms on the slice before
        accumulating into per-transform stats DataFrames.
        """
        from .results import WindowAnalysisResult

        if not load_cols:
            for t in pass1_ts:
                t.pass1(pd.DataFrame())
            return

        # Map each pass1 transform to its preceding pass-through transforms
        # (drop aggregating transforms — they don't shape per-row state).
        pre_for: dict[int, list[Transform]] = {}
        for t in pass1_ts:
            idx = self._pending.index(t)
            pre_for[id(t)] = [
                p for p in self._pending[:idx] if not p.is_aggregating
            ]

        pq_file = pq.ParquetFile(self._src_parquet)
        encoded_in_load = [c for c in self._encoded_columns if c in load_cols]
        chunks_for: dict[int, list[pd.DataFrame]] = {id(t): [] for t in pass1_ts}
        for batch in pq_file.iter_batches(batch_size=batch_size, columns=load_cols):
            df = batch.to_pandas(self_destruct=True, split_blocks=True)
            if encoded_in_load:
                df = WindowAnalysisResult._decode_df_from_parquet(
                    df, encoded_in_load, encoding_version=self._encoding_version
                )
            # Replay preceding pass-through transforms per pass1 target.  Each
            # target gets its own copy because transforms may mutate cells.
            for t in pass1_ts:
                df_for_t = df.copy() if pre_for[id(t)] else df
                for pre in pre_for[id(t)]:
                    df_for_t = pre.apply(df_for_t)
                chunks_for[id(t)].append(df_for_t)
        for t in pass1_ts:
            df_stats = (
                pd.concat(chunks_for[id(t)], ignore_index=True)
                if chunks_for[id(t)]
                else pd.DataFrame()
            )
            t.pass1(df_stats)
        del df_stats
        gc.collect()
