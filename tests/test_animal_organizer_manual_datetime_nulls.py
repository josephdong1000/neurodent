"""Regression suite for dict-with-null ``manual_datetime`` forward cumulation.

The pipeline supports three ``manual_datetime`` forms per animal:

- **dict** with per-session explicit start times,
- **list** of one datetime per file,
- **single value** as a global start.

The single-value form silently mis-orders sessions when ``_natural_sort_key``
disagrees with chronology (e.g. session markers ``[" ", "-", " 1 ", " 2 "]``
sort to ``[" ", " 1 ", " 2 ", "-"]`` but the dash is chronologically second).

To let users have one explicit anchor + cumulate forward without the
natural-sort fragility, the dict form now supports ``null`` values that
mean "cumulate forward from the previous explicit anchor's end time."
Dict insertion order (Python 3.7+ contract) is the sole canonical
chronological order — no separate ``session_order`` field, no conflict.

These tests pin down the new contract: forward cumulation, explicit
anchors as resets, no silent reconciliation when an explicit value
disagrees with cumulation, and loud errors for first-null / missing-key /
extra-key.  They also include the 1199-cohort marker pattern as a
concrete regression guard against the natural-sort failure mode.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from neurodent.visualization import results


# ─────────────────────────────────────────────────────────────────────────
# Fixtures — minimal AnimalOrganizer + mock LRO with fixed durations
# ─────────────────────────────────────────────────────────────────────────


def _mock_lro(duration_seconds: float):
    """Return a minimal mock LRO with the given fixed recording duration."""
    mock_lro = Mock()
    mock_lro.channel_names = ["ch1"]
    mock_lro.meta = Mock(f_s=1000, n_channels=1)
    mock_lro.file_durations = [duration_seconds]
    rec = Mock()
    rec.get_duration.return_value = duration_seconds
    mock_lro.LongRecording = rec
    mock_lro.file_end_datetimes = [None]
    return mock_lro


def _make_ao(durations_by_item: dict | None = None):
    """Construct a bare AnimalOrganizer with string sentinels as items.

    ``durations_by_item`` maps item string → seconds.  When unset, every
    item gets a default 3600 s (1 h) duration.
    """
    ao = object.__new__(results.AnimalOrganizer)
    ao.animal_id = "test_animal"
    # Items used in these tests are plain strings — _get_item_name and
    # _get_item_key both just return the string itself.
    ao._get_item_name = lambda item: item
    ao._get_item_key = lambda item: item
    ao._get_context_path = lambda item: "."
    # _resolve_timestamp_input parses ISO strings to datetimes; the real
    # method does more (path-based filename parsing etc.) but for strings
    # the parse path is all we need.
    ao._resolve_timestamp_input = lambda spec, _path: (
        datetime.fromisoformat(spec) if isinstance(spec, str) else spec
    )
    # _validate_timestamp_ordering is a method on the real class; bind it.
    ao._validate_timestamp_ordering = results.AnimalOrganizer._validate_timestamp_ordering
    # _items_have_index / _session_sort_key — single-element sessions, so
    # whatever ordering is fine; provide identity sort.
    ao._items_have_index = lambda items: False
    ao._session_sort_key = lambda items: (lambda f: f)
    ao._sort_lros_by_median_time = lambda pairs: pairs
    return ao, durations_by_item or {}


def _build_animalday(*session_specs):
    """Build animalday_to_items from ``(session_key, [item_names])`` tuples."""
    return {sess: list(items) for sess, items in session_specs}


def _patch_lro_durations(durations: dict):
    """Patch core.LongRecordingOrganizer to return mock LROs whose duration
    is looked up from *durations* by item name. Items missing from the dict
    default to 3600 s.
    """
    def _factory(item, **_kw):
        seconds = durations.get(item, 3600.0)
        return _mock_lro(seconds)
    return patch(
        "neurodent.visualization.results.core.LongRecordingOrganizer",
        side_effect=_factory,
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. Backward-compatibility: all-explicit dict behaves identically
# ─────────────────────────────────────────────────────────────────────────


class TestAllExplicitUnchanged:
    def test_all_explicit_dict_keeps_per_session_anchors(self):
        """Existing all-string dict behaves identically — per-session start
        times override any cumulation."""
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("_", ["a"]),
            ("_1_", ["b"]),
            ("_2_", ["c"]),
        )
        manual_datetime = {
            "_": "2015-02-24 09:56:19",
            "_1_": "2015-02-25 09:52:45",
            "_2_": "2015-02-26 09:48:33",
        }

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {}
            )

        # Each session uses its OWN explicit anchor — gaps are preserved.
        assert timeline["a"] == datetime(2015, 2, 24, 9, 56, 19)
        assert timeline["b"] == datetime(2015, 2, 25, 9, 52, 45)
        assert timeline["c"] == datetime(2015, 2, 26, 9, 48, 33)


# ─────────────────────────────────────────────────────────────────────────
# 2. Single anchor + rest null = forward cumulation
# ─────────────────────────────────────────────────────────────────────────


class TestSingleAnchorRestNullCumulates:
    def test_one_anchor_rest_null_chains_from_end_of_previous(self):
        """One explicit anchor, rest null → each later session starts at
        the previous session's last-file end time (contiguous chain).
        """
        ao, _ = _make_ao()
        # Each session has 2 items at 1 hour each → 2 h per session.
        animalday_to_items = _build_animalday(
            ("_", ["s0_f0", "s0_f1"]),
            ("_1_", ["s1_f0", "s1_f1"]),
            ("_2_", ["s2_f0", "s2_f1"]),
        )
        manual_datetime = {
            "_":   "2015-02-24 11:11:00",
            "_1_": None,
            "_2_": None,
        }

        with _patch_lro_durations({
            "s0_f0": 3600, "s0_f1": 3600,
            "s1_f0": 3600, "s1_f1": 3600,
            "s2_f0": 3600, "s2_f1": 3600,
        }):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # Session "_" starts at 11:11; 2 h of recording → ends at 13:11.
        assert timeline["s0_f0"] == datetime(2015, 2, 24, 11, 11, 0)
        assert timeline["s0_f1"] == datetime(2015, 2, 24, 12, 11, 0)
        # Session "_1_" cumulates from end of "_" → starts at 13:11.
        assert timeline["s1_f0"] == datetime(2015, 2, 24, 13, 11, 0)
        assert timeline["s1_f1"] == datetime(2015, 2, 24, 14, 11, 0)
        # Session "_2_" cumulates from end of "_1_" → starts at 15:11.
        assert timeline["s2_f0"] == datetime(2015, 2, 24, 15, 11, 0)
        assert timeline["s2_f1"] == datetime(2015, 2, 24, 16, 11, 0)


# ─────────────────────────────────────────────────────────────────────────
# 3. Partial anchors — explicit value resets cumulation
# ─────────────────────────────────────────────────────────────────────────


class TestPartialAnchorsResetCumulation:
    def test_three_sessions_middle_null_outer_explicit(self):
        """3 sessions: 1st and 3rd explicit, 2nd null.

            - Session 1 starts at its explicit anchor.
            - Session 2 cumulates from session 1's end.
            - Session 3 takes its explicit value as authoritative —
              does NOT continue from session 2's cumulated end.

        This is the most common partial-anchor case worth a direct test.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("S1", ["a"]),
            ("S2", ["b"]),
            ("S3", ["c"]),
        )
        manual_datetime = {
            "S1": "2015-02-24 10:00:00",
            "S2": None,                     # cumulate from end of S1
            "S3": "2015-02-26 08:00:00",    # explicit — overrides cumulation
        }

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # S1: explicit
        assert timeline["a"] == datetime(2015, 2, 24, 10, 0, 0)
        # S2: cumulates from end of S1 (11:00 on Feb 24, NOT Feb 26)
        assert timeline["b"] == datetime(2015, 2, 24, 11, 0, 0)
        # S3: explicit value taken as-is, ignoring what cumulation would give
        # (cumulated would be 12:00 on Feb 24; explicit says Feb 26 08:00)
        assert timeline["c"] == datetime(2015, 2, 26, 8, 0, 0)

        # Explicit confirmation that S3 was NOT cumulated from S2's end.
        cumulated_from_s2 = datetime(2015, 2, 24, 12, 0, 0)
        assert timeline["c"] != cumulated_from_s2, (
            "Sanity check: S3's explicit value should override what "
            "cumulation from S2's end would have produced."
        )

    def test_later_explicit_anchor_resets_chain(self):
        """{A: explicit, B: null, C: explicit, D: null} →
            - B cumulates from A's end
            - C overrides (does NOT continue from B)
            - D cumulates from C's end (NOT from B's cumulated state)
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("A", ["a"]),
            ("B", ["b"]),
            ("C", ["c"]),
            ("D", ["d"]),
        )
        manual_datetime = {
            "A": "2025-01-01 10:00:00",
            "B": None,                    # cumulate from end of A
            "C": "2025-06-15 08:00:00",   # explicit reset — far in the future
            "D": None,                    # cumulate from end of C
        }

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 7200, "d": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # A: explicit, 1 h
        assert timeline["a"] == datetime(2025, 1, 1, 10, 0, 0)
        # B: cumulate from A's end (11:00 on Jan 1)
        assert timeline["b"] == datetime(2025, 1, 1, 11, 0, 0)
        # C: explicit reset (Jun 15 08:00) — NOT continuing from B's end
        assert timeline["c"] == datetime(2025, 6, 15, 8, 0, 0)
        # D: cumulate from C's end (Jun 15 10:00 — C had 2 h duration)
        assert timeline["d"] == datetime(2025, 6, 15, 10, 0, 0)


# ─────────────────────────────────────────────────────────────────────────
# 4. Prefix backfill — first explicit anchor is in the middle of the dict
# ─────────────────────────────────────────────────────────────────────────


class TestPrefixBackfill:
    """When the first explicit anchor is NOT at index 0, prefix nulls are
    filled by working backward from the anchor (assuming contiguous
    recording).  After the anchor, current forward semantics apply.
    """

    def test_middle_anchor_backfills_prefix_and_forwards_suffix(self):
        """``{S1: null, S2: explicit, S3: null}`` →
            - S2 is the first explicit anchor.
            - S1 backfills: S1.start = S2.start - S1.total_duration.
            - S3 forwards: S3.start = S2.end.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("S1", ["a"]),  # 1 h
            ("S2", ["b"]),  # 1 h
            ("S3", ["c"]),  # 1 h
        )
        manual_datetime = {
            "S1": None,                    # backfill from S2
            "S2": "2015-02-24 12:00:00",   # first explicit anchor
            "S3": None,                    # forward from S2
        }

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # S1 should END exactly where S2 starts (contiguous backfill).
        assert timeline["a"] == datetime(2015, 2, 24, 11, 0, 0)
        assert timeline["b"] == datetime(2015, 2, 24, 12, 0, 0)
        assert timeline["c"] == datetime(2015, 2, 24, 13, 0, 0)

    def test_two_prefix_nulls_backfill_in_reverse_order(self):
        """``{S1: null, S2: null, S3: explicit, S4: null}`` →
            - S3 is the first explicit anchor.
            - S2 backfills from S3 (S2.start = S3.start - S2.duration).
            - S1 backfills from S2 (S1.start = S2.start - S1.duration).
            - S4 forwards from S3.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("S1", ["a"]),  # 1 h
            ("S2", ["b"]),  # 2 h (90 min + 30 min) — use single 2-hour item for simplicity
            ("S3", ["c"]),  # 1 h
            ("S4", ["d"]),  # 1 h
        )
        manual_datetime = {
            "S1": None,
            "S2": None,
            "S3": "2015-02-24 12:00:00",
            "S4": None,
        }

        with _patch_lro_durations({"a": 3600, "b": 7200, "c": 3600, "d": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # S3 explicit at 12:00.
        # S2 = S3 - 2h = 10:00.  S1 = S2 - 1h = 09:00.  S4 = S3 + 1h = 13:00.
        assert timeline["a"] == datetime(2015, 2, 24, 9, 0, 0)
        assert timeline["b"] == datetime(2015, 2, 24, 10, 0, 0)
        assert timeline["c"] == datetime(2015, 2, 24, 12, 0, 0)
        assert timeline["d"] == datetime(2015, 2, 24, 13, 0, 0)

    def test_pattern_k_u_u_u(self):
        """``{S1: k, S2: u, S3: u, S4: u}``: anchor at start, 3 forward
        cumulations. The basic single-anchor forward case extended to
        verify the chain holds over multiple consecutive nulls.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("S1", ["a"]),
            ("S2", ["b"]),
            ("S3", ["c"]),
            ("S4", ["d"]),
        )
        manual_datetime = {
            "S1": "2025-01-01 10:00:00",   # k
            "S2": None,                    # u
            "S3": None,                    # u
            "S4": None,                    # u
        }

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600, "d": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # Each session 1 h, chained forward from S1.
        assert timeline["a"] == datetime(2025, 1, 1, 10, 0, 0)
        assert timeline["b"] == datetime(2025, 1, 1, 11, 0, 0)
        assert timeline["c"] == datetime(2025, 1, 1, 12, 0, 0)
        assert timeline["d"] == datetime(2025, 1, 1, 13, 0, 0)

    def test_pattern_k_u_u_k(self):
        """``{S1: k, S2: u, S3: u, S4: k}``: anchor at start AND at end.

            - S1 explicit at the front (no backfill needed).
            - S2, S3 forward-cumulate from S1.
            - S4 explicit — overrides what cumulation would give.
            - Locks in that subsequent explicit anchors reset the chain
              even with multiple intervening nulls.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("S1", ["a"]),
            ("S2", ["b"]),
            ("S3", ["c"]),
            ("S4", ["d"]),
        )
        manual_datetime = {
            "S1": "2025-01-01 10:00:00",   # k
            "S2": None,                    # u
            "S3": None,                    # u
            "S4": "2025-06-15 08:00:00",   # k (resets — overrides cumulation)
        }

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600, "d": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # S1 explicit, S2 = S1+1h, S3 = S1+2h, S4 explicit (override).
        assert timeline["a"] == datetime(2025, 1, 1, 10, 0, 0)
        assert timeline["b"] == datetime(2025, 1, 1, 11, 0, 0)
        assert timeline["c"] == datetime(2025, 1, 1, 12, 0, 0)
        # S4 wins — would have been 13:00 on Jan 1 if cumulation chained.
        assert timeline["d"] == datetime(2025, 6, 15, 8, 0, 0)

    def test_pattern_u_u_u_k(self):
        """``{S1: u, S2: u, S3: u, S4: k}``: 3 prefix nulls backfill from
        the final anchor.

            - S4 is the first (and only) explicit anchor.
            - S3 = S4 - S3.duration
            - S2 = S3 - S2.duration
            - S1 = S2 - S1.duration
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("S1", ["a"]),
            ("S2", ["b"]),
            ("S3", ["c"]),
            ("S4", ["d"]),
        )
        manual_datetime = {
            "S1": None,                    # u
            "S2": None,                    # u
            "S3": None,                    # u
            "S4": "2025-01-01 13:00:00",   # k
        }

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600, "d": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # Working backward from S4 (13:00):
        # S3 starts at 12:00 (S4 - 1h)
        # S2 starts at 11:00 (S3 - 1h)
        # S1 starts at 10:00 (S2 - 1h)
        assert timeline["a"] == datetime(2025, 1, 1, 10, 0, 0)
        assert timeline["b"] == datetime(2025, 1, 1, 11, 0, 0)
        assert timeline["c"] == datetime(2025, 1, 1, 12, 0, 0)
        assert timeline["d"] == datetime(2025, 1, 1, 13, 0, 0)

    def test_pattern_u_u_k_u_u(self):
        """``{S1: u, S2: u, S3: k, S4: u, S5: u}``: 2 prefix nulls backfill,
        2 suffix nulls forward-fill, around a middle anchor.

            - S3 is the first explicit anchor.
            - S2 = S3 - S2.duration; S1 = S2 - S1.duration.
            - S4 = S3 + S3.duration; S5 = S4 + S4.duration.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("S1", ["a"]),
            ("S2", ["b"]),
            ("S3", ["c"]),
            ("S4", ["d"]),
            ("S5", ["e"]),
        )
        manual_datetime = {
            "S1": None,                    # u
            "S2": None,                    # u
            "S3": "2025-01-01 12:00:00",   # k
            "S4": None,                    # u
            "S5": None,                    # u
        }

        with _patch_lro_durations({
            "a": 3600, "b": 3600, "c": 3600, "d": 3600, "e": 3600,
        }):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # Centered on S3 = 12:00, each session 1 h.
        assert timeline["a"] == datetime(2025, 1, 1, 10, 0, 0)  # S1
        assert timeline["b"] == datetime(2025, 1, 1, 11, 0, 0)  # S2 (backfill)
        assert timeline["c"] == datetime(2025, 1, 1, 12, 0, 0)  # S3 (anchor)
        assert timeline["d"] == datetime(2025, 1, 1, 13, 0, 0)  # S4 (forward)
        assert timeline["e"] == datetime(2025, 1, 1, 14, 0, 0)  # S5 (forward)

    def test_arxrosa_1199_pattern_with_middle_anchor(self):
        """Realistic 1199-cohort pattern but with the explicit anchor on
        the ``-`` session (second by chronology, second by dict order):

            { " ": null, "-": explicit, " 1 ": null, " 2 ": null }

        Verifies the dash-anchored backfill works for the natural-sort-
        unsafe marker pattern.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            (" ",   ["s0"]),
            ("-",   ["s1"]),
            (" 1 ", ["s2"]),
            (" 2 ", ["s3"]),
        )
        manual_datetime = {
            " ":   None,
            "-":   "2016-01-19 20:15:41",   # first explicit
            " 1 ": None,
            " 2 ": None,
        }

        with _patch_lro_durations({"s0": 3600, "s1": 3600, "s2": 3600, "s3": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # " " backfills 1 h before "-": 19:15:41.
        # "-" explicit: 20:15:41.
        # " 1 " forward: 21:15:41.
        # " 2 " forward: 22:15:41.
        assert timeline["s0"] == datetime(2016, 1, 19, 19, 15, 41)
        assert timeline["s1"] == datetime(2016, 1, 19, 20, 15, 41)
        assert timeline["s2"] == datetime(2016, 1, 19, 21, 15, 41)
        assert timeline["s3"] == datetime(2016, 1, 19, 22, 15, 41)


# ─────────────────────────────────────────────────────────────────────────
# 5. Validation — all-null / missing / extra key all raise loudly
# ─────────────────────────────────────────────────────────────────────────


class TestValidationErrors:
    def test_all_null_dict_raises(self):
        """If every session is null → ValueError. At least one explicit
        anchor is required (either at the front or somewhere in the middle).
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("_", ["a"]), ("_1_", ["b"]), ("_2_", ["c"]),
        )
        manual_datetime = {"_": None, "_1_": None, "_2_": None}

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600}):
            with pytest.raises(ValueError, match="no explicit anchor"):
                ao._process_manual_datetimes(
                    manual_datetime, animalday_to_items, {"datetimes_are_start": True}
                )

    def test_missing_discovered_session_raises(self):
        """Dict missing a key the discovery found → ValueError listing the
        missing session.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("_", ["a"]), ("_1_", ["b"]), ("_2_", ["c"]),
        )
        # Dict missing "_2_" — discovery found it but config doesn't list it.
        manual_datetime = {"_": "2025-01-01 10:00:00", "_1_": None}

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600}):
            with pytest.raises(ValueError, match="Missing entries.*'_2_'"):
                ao._process_manual_datetimes(
                    manual_datetime, animalday_to_items, {"datetimes_are_start": True}
                )

    def test_dict_key_not_in_discovery_raises(self):
        """Dict key that wasn't discovered → ValueError (catches typos and
        stale config when animal data changes).
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(("_", ["a"]), ("_1_", ["b"]))
        # Dict has "_999_" — typo / stale entry.
        manual_datetime = {
            "_": "2025-01-01 10:00:00",
            "_1_": None,
            "_999_": "2025-01-02 10:00:00",  # not in discovery
        }

        with _patch_lro_durations({"a": 3600, "b": 3600}):
            with pytest.raises(ValueError, match="keys not in discovered sessions.*_999_"):
                ao._process_manual_datetimes(
                    manual_datetime, animalday_to_items, {"datetimes_are_start": True}
                )

    def test_null_with_end_times_raises(self):
        """Null cumulation requires datetimes_are_start=True (forward only).
        Mixing nulls with end-time mode → ValueError.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(("_", ["a"]), ("_1_", ["b"]))
        manual_datetime = {"_": "2025-01-01 12:00:00", "_1_": None}

        with _patch_lro_durations({"a": 3600, "b": 3600}):
            with pytest.raises(ValueError, match="null values.*datetimes_are_start is\\s+False"):
                ao._process_manual_datetimes(
                    manual_datetime, animalday_to_items, {"datetimes_are_start": False}
                )


# ─────────────────────────────────────────────────────────────────────────
# 5. Dict iteration order beats _natural_sort_key
# ─────────────────────────────────────────────────────────────────────────


class TestDictOrderRespectedNotNaturalSort:
    """Concrete regression guard against the natural-sort failure mode that
    motivated this whole feature.  ArxRosa-1199 cohort uses session markers
    ``[" ", "-", " 1 ", " 2 "]`` whose chronological order is
    ``[" ", "-", " 1 ", " 2 "]`` but ``_natural_sort_key`` orders them as
    ``[" ", " 1 ", " 2 ", "-"]`` (dash mis-placed by 2+ days).

    With the new dict-with-null contract, dict insertion order is the sole
    source of truth — the dash session lands in the correct chronological
    position regardless of natural-sort behaviour.
    """

    def test_arxrosa_1199_marker_pattern_uses_dict_order(self):
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            (" ", ["s0"]),
            ("-", ["s1"]),
            (" 1 ", ["s2"]),
            (" 2 ", ["s3"]),
        )
        # Dict insertion order matches chronological truth.
        manual_datetime = {
            " ":   "2016-01-19 12:15:41",
            "-":   None,
            " 1 ": None,
            " 2 ": None,
        }

        with _patch_lro_durations({"s0": 3600, "s1": 3600, "s2": 3600, "s3": 3600}):
            timeline = ao._process_manual_datetimes(
                manual_datetime, animalday_to_items, {"datetimes_are_start": True}
            )

        # Strict monotonic chain: s0 → s1 → s2 → s3, each 1h after the prior.
        assert timeline["s0"] == datetime(2016, 1, 19, 12, 15, 41)
        assert timeline["s1"] == datetime(2016, 1, 19, 13, 15, 41)
        assert timeline["s2"] == datetime(2016, 1, 19, 14, 15, 41)
        assert timeline["s3"] == datetime(2016, 1, 19, 15, 15, 41)

        # Critical regression check: the dash session ("-") lands BEFORE
        # the " 1 " session, even though _natural_sort_key would put it
        # LAST.  Insertion order is canonical.
        assert timeline["s1"] < timeline["s2"]

    def test_dash_after_naturally_sorted_keys_would_have_failed_pre_fix(self):
        """Same shape but using natural-sort directly would have produced
        the old broken ordering. Demonstrates *why* the fix matters.
        """
        from neurodent.core.discovery import _natural_sort_key

        keys = [" ", "-", " 1 ", " 2 "]
        natural_order = sorted(keys, key=_natural_sort_key)
        # The dash lands LAST under natural-sort.
        assert natural_order == [" ", " 1 ", " 2 ", "-"]
        # That contradicts chronology (dash is second). Under the new
        # contract, we never call _natural_sort_key on dict keys — we
        # iterate the dict directly.


# ─────────────────────────────────────────────────────────────────────────
# 6. Explicit anchor disagreeing with cumulation: silent override
# ─────────────────────────────────────────────────────────────────────────


class TestExplicitAnchorOverridesCumulationSilently:
    def test_explicit_value_disagrees_with_cumulated_no_warning(self):
        """{A: explicit, B: null, C: explicit-but-disagrees} →
            - B cumulates from A's end (e.g. 11:00)
            - C is explicitly set to 23:00, which disagrees with what
              cumulation would predict (12:00).
            - Pipeline silently uses C's explicit value.
            - No warning emitted by design — user owns the choice.
        """
        ao, _ = _make_ao()
        animalday_to_items = _build_animalday(
            ("A", ["a"]),
            ("B", ["b"]),
            ("C", ["c"]),
        )
        # Cumulation would put C at 12:00 (B ends at 12:00). User explicitly
        # sets C at 23:00.
        manual_datetime = {
            "A": "2025-01-01 10:00:00",
            "B": None,
            "C": "2025-01-01 23:00:00",
        }

        import warnings as _warnings

        with _patch_lro_durations({"a": 3600, "b": 3600, "c": 3600}):
            # Silent override: no warning emitted from our code path even
            # though the explicit C (23:00) disagrees with what cumulation
            # would have produced (12:00).
            with _warnings.catch_warnings(record=True) as record:
                _warnings.simplefilter("always")
                timeline = ao._process_manual_datetimes(
                    manual_datetime, animalday_to_items, {"datetimes_are_start": True}
                )
            # Filter for any warning that mentions cumulation/anchor/
            # disagreement (i.e., would come from THIS code path).  Anything
            # else is library noise we don't care about here.
            our_warnings = [
                w for w in record
                if "cumulation" in str(w.message).lower()
                or "anchor" in str(w.message).lower()
                or "disagree" in str(w.message).lower()
            ]
            assert our_warnings == []

        # C's explicit value wins — not the cumulated 12:00.
        assert timeline["a"] == datetime(2025, 1, 1, 10, 0, 0)
        assert timeline["b"] == datetime(2025, 1, 1, 11, 0, 0)
        assert timeline["c"] == datetime(2025, 1, 1, 23, 0, 0)
