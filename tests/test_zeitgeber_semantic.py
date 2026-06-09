"""Semantic correctness tests for the zeitgeber ZT-time contract.

The structural tests in ``tests/test_zeitgeber.py`` check that columns
exist and that helpers run without raising.  These tests check the
**meaning** of those columns, the kind of bug that's silent otherwise —
e.g. ``zt_minutes=0`` actually corresponding to ZT0 (lights-on) after the
default 6h shift, day/night labels matching the lights-on/lights-off
boundary without off-by-one, and the multi-day expansion preserving
labels across cycles.

These were added during the ``total_minutes → zt_minutes`` refactor that
also pulled the 48h row duplication out of the data layer and into
:func:`neurodent.core.zeitgeber.expand_zt_axis`.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from neurodent.core import zeitgeber


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _single_row_df_at_hour(hour: int, animal: str = "anim_test") -> pd.DataFrame:
    """Build a minimal DataFrame with one row at clock-time *hour*:00."""
    return pd.DataFrame(
        {
            "timestamp": [dt.datetime(2025, 1, 1, hour, 0)],
            "animal": [animal],
            "genotype": ["M_WT"],
            "feature1": [1.0],
        }
    )


def _multi_hour_df(animal: str = "anim_test") -> pd.DataFrame:
    """DataFrame covering all 24 hours of a single day at clock-time
    hh:00, for a single animal."""
    rows = [
        {
            "timestamp": dt.datetime(2025, 1, 1, h, 0),
            "animal": animal,
            "genotype": "M_WT",
            "feature1": float(h),
        }
        for h in range(24)
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# 1. zt_minutes semantic — 0 = lights-on, 720 = lights-off
# ─────────────────────────────────────────────────────────────────────────


class TestZTMinutesSemantic:
    """``zt_minutes`` must actually represent ZT after the pipeline runs."""

    def test_clock_6am_becomes_zt_zero_after_default_shift(self):
        """Recording at clock 06:00 → ZT 0:00 (lights-on) with shift_hours=6."""
        df = _single_row_df_at_hour(6)
        df = zeitgeber.add_zeitgeber_time_columns(df)
        processed = zeitgeber.run_zeitgeber_pipeline(df, zeitgeber_shift_hours=6)
        assert processed.iloc[0]["zt_minutes"] == 0

    def test_clock_5am_becomes_zt_23_after_default_shift(self):
        """Recording at clock 05:00 → ZT 23:00 (= 1380 min, one hour before
        lights-on the next day).  Catches a sign-flipped shift or
        double-application."""
        df = _single_row_df_at_hour(5)
        df = zeitgeber.add_zeitgeber_time_columns(df)
        processed = zeitgeber.run_zeitgeber_pipeline(df, zeitgeber_shift_hours=6)
        assert processed.iloc[0]["zt_minutes"] == 23 * 60

    def test_clock_6pm_becomes_zt_12_after_default_shift(self):
        """Recording at clock 18:00 → ZT 12:00 (= 720 min, lights-off)."""
        df = _single_row_df_at_hour(18)
        df = zeitgeber.add_zeitgeber_time_columns(df)
        processed = zeitgeber.run_zeitgeber_pipeline(df, zeitgeber_shift_hours=6)
        assert processed.iloc[0]["zt_minutes"] == 12 * 60

    def test_zt_zero_lands_in_day_phase(self):
        """End-to-end: ZT 0 should be labelled ``"Day"``."""
        df = _single_row_df_at_hour(6)
        df = zeitgeber.add_zeitgeber_time_columns(df)
        processed = zeitgeber.run_zeitgeber_pipeline(df, zeitgeber_shift_hours=6)
        assert processed.iloc[0]["zt_minutes"] == 0
        assert processed.iloc[0]["daynight"] == "Day"

    def test_zt_twelve_lands_in_night_phase(self):
        """End-to-end: ZT 12 (lights-off) should be labelled ``"Night"``."""
        df = _single_row_df_at_hour(18)
        df = zeitgeber.add_zeitgeber_time_columns(df)
        processed = zeitgeber.run_zeitgeber_pipeline(df, zeitgeber_shift_hours=6)
        assert processed.iloc[0]["zt_minutes"] == 720
        assert processed.iloc[0]["daynight"] == "Night"


# ─────────────────────────────────────────────────────────────────────────
# 2. Day / Night label correctness
# ─────────────────────────────────────────────────────────────────────────


class TestDaynightLabel:
    """The ``daynight`` column must match the lights-on/off split exactly."""

    @pytest.mark.parametrize(
        "zt_minutes, expected",
        [
            (0,    "Day"),    # ZT 0:00 — lights-on
            (1,    "Day"),    # ZT 0:01
            (360,  "Day"),    # ZT 6:00 — mid-light
            (719,  "Day"),    # ZT 11:59 — last minute of light
            (720,  "Night"),  # ZT 12:00 — lights-off (boundary)
            (721,  "Night"),  # ZT 12:01
            (1080, "Night"),  # ZT 18:00 — mid-dark
            (1439, "Night"),  # ZT 23:59 — last minute of dark
        ],
    )
    def test_label_at_boundary(self, zt_minutes, expected):
        """``zt_minutes < 720`` ↔ ``"Day"``; ``>= 720`` ↔ ``"Night"``."""
        labels = zeitgeber._compute_daynight(pd.Series([zt_minutes]))
        assert labels[0] == expected

    def test_label_attached_by_shift_to_zeitgeber_reference(self):
        """``shift_to_zeitgeber_reference`` must add the ``daynight``
        column when it adjusts ``zt_minutes``."""
        df = pd.DataFrame({"zt_minutes": [0, 60, 720, 1080]})
        out = zeitgeber.shift_to_zeitgeber_reference(df.copy(), shift_hours=0)
        assert "daynight" in out.columns
        # zt 0 → Day, 60 → Day, 720 → Night, 1080 → Night
        assert list(out["daynight"]) == ["Day", "Day", "Night", "Night"]

    def test_label_recomputed_when_transform_time_axis_shifts(self):
        """Calling ``transform_time_axis`` with a non-zero ``shift`` must
        recompute ``daynight`` so labels stay consistent with the new
        ``zt_minutes``."""
        df = pd.DataFrame(
            {
                "zt_minutes": [0, 720],   # was Day, Night
                "daynight": ["Day", "Night"],
            }
        )
        # Shift +12h: 0 → 720 (Night), 720 → 0 (Day after %1440).
        out = zeitgeber.transform_time_axis(df, shift=12)
        assert list(out["zt_minutes"]) == [720, 0]
        assert list(out["daynight"]) == ["Night", "Day"]


# ─────────────────────────────────────────────────────────────────────────
# 3. expand_zt_axis — semantic + structural
# ─────────────────────────────────────────────────────────────────────────


class TestExpandZTAxis:
    def test_n_days_one_is_passthrough(self):
        df = pd.DataFrame({"zt_minutes": [0, 60, 720]})
        out = zeitgeber.expand_zt_axis(df, n_days=1)
        assert len(out) == len(df)
        # Returned a copy, not the same object.
        assert out is not df

    @pytest.mark.parametrize("n_days", [2, 3, 5])
    def test_row_count_scales_with_n_days(self, n_days):
        df = pd.DataFrame({"zt_minutes": [0.0, 60.0, 720.0], "feature1": [1.0, 2.0, 3.0]})
        out = zeitgeber.expand_zt_axis(df, n_days=n_days)
        assert len(out) == len(df) * n_days

    def test_zt_minutes_offset_by_1440_per_cycle(self):
        """The i-th copy's ``zt_minutes`` is offset by ``1440 * i``."""
        df = pd.DataFrame({"zt_minutes": [0.0, 60.0], "feature1": [1.0, 2.0]})
        out = zeitgeber.expand_zt_axis(df, n_days=3)
        # First copy: [0, 60].  Second: [1440, 1500].  Third: [2880, 2940].
        assert sorted(out["zt_minutes"].tolist()) == [0.0, 60.0, 1440.0, 1500.0, 2880.0, 2940.0]

    def test_feature_values_preserved_across_copies(self):
        """For each original row, the duplicate at ``zt_minutes + 1440`` has
        identical feature values — the duplication is purely temporal."""
        df = pd.DataFrame(
            {
                "zt_minutes": [0.0, 360.0, 720.0],
                "feature1": [10.0, 20.0, 30.0],
                "feature2": [100.0, 200.0, 300.0],
            }
        )
        out = zeitgeber.expand_zt_axis(df, n_days=2)
        for orig_zt in [0.0, 360.0, 720.0]:
            row_orig = out[out["zt_minutes"] == orig_zt].iloc[0]
            row_copy = out[out["zt_minutes"] == orig_zt + 1440].iloc[0]
            assert row_orig["feature1"] == row_copy["feature1"]
            assert row_orig["feature2"] == row_copy["feature2"]

    def test_daynight_mirrors_first_day_in_expanded_range(self):
        """Day 2's daynight label mirrors day 1 at the same ZT-of-day:
        ZT 1:00 of day 2 → Day; ZT 13:00 of day 2 → Night."""
        df = pd.DataFrame({"zt_minutes": [60.0, 780.0]})  # ZT 1:00 (Day), ZT 13:00 (Night)
        out = zeitgeber.expand_zt_axis(df, n_days=3)
        # zt_minutes=60 (day 1, Day), 1500 (day 2, Day), 2940 (day 3, Day);
        # zt_minutes=780 (day 1, Night), 2220 (day 2, Night), 3660 (day 3, Night).
        for zt in [60.0, 1500.0, 2940.0]:
            label = out[out["zt_minutes"] == zt].iloc[0]["daynight"]
            assert label == "Day", f"zt_minutes={zt} should be Day, got {label!r}"
        for zt in [780.0, 2220.0, 3660.0]:
            label = out[out["zt_minutes"] == zt].iloc[0]["daynight"]
            assert label == "Night", f"zt_minutes={zt} should be Night, got {label!r}"

    @pytest.mark.parametrize("bad_n_days", [0, -1, -5])
    def test_rejects_invalid_n_days(self, bad_n_days):
        df = pd.DataFrame({"zt_minutes": [0.0]})
        with pytest.raises(ValueError, match="n_days must be"):
            zeitgeber.expand_zt_axis(df, n_days=bad_n_days)

    def test_missing_zt_minutes_returns_copy(self):
        """If the input lacks ``zt_minutes``, return a defensive copy
        unchanged.  This matches how other zeitgeber helpers behave on
        partial input."""
        df = pd.DataFrame({"feature1": [1.0, 2.0]})
        out = zeitgeber.expand_zt_axis(df, n_days=2)
        assert len(out) == len(df)
        assert out is not df


# ─────────────────────────────────────────────────────────────────────────
# 4. transform_time_axis — does NOT duplicate rows (regression guard)
# ─────────────────────────────────────────────────────────────────────────


class TestTransformTimeAxisNoDuplication:
    """After the refactor, ``transform_time_axis`` only does shift + sort
    + sex/gene enrichment.  Row count must equal input row count for any
    ``time_range`` — multi-day expansion has moved to
    :func:`expand_zt_axis`.  Pre-refactor these would have doubled."""

    @pytest.mark.parametrize("time_range", [(0, 24), (0, 48), (0, 72), (0, 168)])
    def test_no_duplication_regardless_of_time_range(self, time_range):
        df = pd.DataFrame(
            {
                "zt_minutes": [0.0, 360.0, 720.0, 1080.0],
                "genotype": ["M_WT"] * 4,
                "value": [1, 2, 3, 4],
            }
        )
        out = zeitgeber.transform_time_axis(df, time_range=time_range, shift=0)
        assert len(out) == len(df)
        assert out["zt_minutes"].max() < 1440


# ─────────────────────────────────────────────────────────────────────────
# 5. add_zeitgeber_time_columns — structural + emits zt_minutes (no
#    daynight yet, since it's pre-shift)
# ─────────────────────────────────────────────────────────────────────────


class TestAddZeitgeberTimeColumns:
    def test_emits_zt_minutes_column(self):
        df = _multi_hour_df()
        out = zeitgeber.add_zeitgeber_time_columns(df)
        assert "zt_minutes" in out.columns

    def test_does_not_emit_daynight_pre_shift(self):
        """``add_zeitgeber_time_columns`` populates ``zt_minutes`` with raw
        clock-time minutes; daynight semantics only make sense after the
        ZT shift, so the column is left for
        ``shift_to_zeitgeber_reference`` to add."""
        df = _multi_hour_df()
        out = zeitgeber.add_zeitgeber_time_columns(df)
        assert "daynight" not in out.columns

    def test_zt_minutes_is_numeric(self):
        df = _multi_hour_df()
        out = zeitgeber.add_zeitgeber_time_columns(df)
        assert pd.api.types.is_numeric_dtype(out["zt_minutes"])


# ─────────────────────────────────────────────────────────────────────────
# 6. Pipeline output is clean 24h with daynight (end-to-end)
# ─────────────────────────────────────────────────────────────────────────


class TestPipelineEndToEnd:
    def test_pipeline_output_is_24h(self):
        """``run_zeitgeber_pipeline`` returns a 24h dataframe — no row
        duplication."""
        df = _multi_hour_df()
        df = zeitgeber.add_zeitgeber_time_columns(df)
        out = zeitgeber.run_zeitgeber_pipeline(df)
        assert len(out) == len(df)
        assert out["zt_minutes"].max() < 1440

    def test_pipeline_emits_daynight_column(self):
        df = _multi_hour_df()
        df = zeitgeber.add_zeitgeber_time_columns(df)
        out = zeitgeber.run_zeitgeber_pipeline(df)
        assert "daynight" in out.columns
        assert set(out["daynight"].unique()).issubset({"Day", "Night"})

    def test_pipeline_daynight_matches_zt_minutes_row_by_row(self):
        """Every row's daynight label is consistent with its zt_minutes."""
        df = _multi_hour_df()
        df = zeitgeber.add_zeitgeber_time_columns(df)
        out = zeitgeber.run_zeitgeber_pipeline(df)
        for _, row in out.iterrows():
            expected = "Day" if row["zt_minutes"] < 720 else "Night"
            assert row["daynight"] == expected, (
                f"row with zt_minutes={row['zt_minutes']} got daynight="
                f"{row['daynight']!r}, expected {expected!r}"
            )


# ─────────────────────────────────────────────────────────────────────────
# 7. Channel-averaging contract for the four new features
#    (lognspike, logampvar, logpsdtotal, psdslope) added to the base config.
# ─────────────────────────────────────────────────────────────────────────


class TestNewBaseFeaturesChannelAveraging:
    """The base zeitgeber feature list in ``config/config.yaml`` was
    extended with ``lognspike``, ``logampvar``, ``logpsdtotal``, and
    ``psdslope``.  The first three are LINEAR and channel-average to a
    scalar with no new code; ``psdslope`` is LINEAR_2D and needs the
    new ``_extract_linear_2d_features`` helper to split into
    ``psdslope_slope`` + ``psdslope_intercept`` first.  These tests
    pin down the end-to-end channel-averaging contract for all four so
    that re-running the zeitgeber pipeline produces the new columns
    without anyone having to think about category dispatch.
    """

    def _make_war_with_new_features(self):
        """Build a minimal WAR carrying all four new features."""
        from neurodent.visualization import WindowAnalysisResult

        rng = np.random.default_rng(0)
        n_windows = 4
        channels = ["LMot", "RMot"]
        n_ch = len(channels)
        rows = []
        for i in range(n_windows):
            row = {
                "animal": "A1",
                "animalday": "A1 WT Jan-01-2023",
                "isday": True,
                "endfile": f"file_{i}.bin",
                "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=i),
                "duration": 60.0,
                # LINEAR features — one scalar per channel per row
                "lognspike": rng.random(n_ch).tolist(),
                "logampvar": rng.random(n_ch).tolist(),
                "logpsdtotal": rng.random(n_ch).tolist(),
                # LINEAR_2D — (n_channels, 2) per row: [slope, intercept]
                "psdslope": rng.random((n_ch, 2)).tolist(),
            }
            rows.append(row)
        return WindowAnalysisResult(
            result=pd.DataFrame(rows),
            animal_id="A1",
            genotype="WT",
            channel_names=channels,
            suppress_short_interval_error=True,
        )

    @pytest.mark.parametrize(
        "feature, expected_columns",
        [
            ("lognspike", ["lognspike"]),
            ("logampvar", ["logampvar"]),
            ("logpsdtotal", ["logpsdtotal"]),
            ("psdslope", ["psdslope_slope", "psdslope_intercept"]),
        ],
    )
    def test_channel_averaged_result_emits_scalar_columns(
        self, feature, expected_columns
    ):
        """Each new feature should produce its expected scalar column(s)
        after channel-averaging, with the source column dropped if it
        was multi-dimensional (psdslope)."""
        war = self._make_war_with_new_features()
        df_avg = war.get_channel_averaged_result(features=[feature])
        for col in expected_columns:
            assert col in df_avg.columns, (
                f"channel-averaged result missing expected column {col!r} "
                f"for feature {feature!r}"
            )
            # Each cell should be a scalar float (not a list/array).
            sample = df_avg[col].iloc[0]
            assert isinstance(sample, (float, int, np.floating, np.integer)), (
                f"{col!r} contains {type(sample).__name__}, expected scalar"
            )

    def test_psdslope_source_column_dropped(self):
        """LINEAR_2D ``psdslope`` must NOT survive channel-averaging — it's
        a 2-D source column that downstream aggregation can't handle."""
        war = self._make_war_with_new_features()
        df_avg = war.get_channel_averaged_result(features=["psdslope"])
        assert "psdslope" not in df_avg.columns

    def test_psdslope_components_are_channel_mean(self):
        """``psdslope_slope`` at row i should equal the mean of column 0
        of the original (n_ch, 2) array at row i; same for intercept."""
        war = self._make_war_with_new_features()
        original = war.result.copy()
        df_avg = war.get_channel_averaged_result(features=["psdslope"])
        for i in range(len(original)):
            arr = np.asarray(original["psdslope"].iloc[i])
            assert df_avg["psdslope_slope"].iloc[i] == pytest.approx(
                arr[:, 0].mean()
            )
            assert df_avg["psdslope_intercept"].iloc[i] == pytest.approx(
                arr[:, 1].mean()
            )
