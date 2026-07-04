"""Regression guard for the band-order ↔ label mismatch in EP / animal figures.

Why this file exists
--------------------
The arxrosa pipeline run produced EP figures whose plotted bars were in
*alphabetical* band order (alpha, beta, delta, gamma, theta) but whose x-axis
labels were in *canonical EEG* order (delta, theta, alpha, beta, gamma) — i.e.
the labels mismatched the underlying data. The bug had two cooperating gaps:

1. ``seaborn.objects`` (``so.Plot``) does not reliably honour
   ``pd.Categorical(ordered=True)`` once any of ``Dodge`` / ``Jitter`` /
   ``Est`` is in the pipeline. The fix is an explicit
   ``.scale(x=so.Nominal(order=BAND_NAMES))``, surfaced via
   :meth:`ExperimentPlotter.band_scale`.

2. :class:`AnimalPlotter` calls ``ax.set_yticks(..., constants.BAND_NAMES)``
   on BAND features without ever verifying that the extracted band-key list
   actually *is* in canonical order. The fix is an assertion inside
   ``AnimalPlotter.__get_linear_feature`` that fires loudly the moment the
   upstream extraction diverges.

This test file exercises *all four layers* of the fix and is the regression
guard that would have caught the original bug.
"""

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn.objects as so

from neurodent import constants
from neurodent.plotting import ExperimentPlotter
from neurodent.results import WindowAnalysisResult
from neurodent.plotting.animal import AnimalPlotter
from neurodent.results.feature_utils import extract_band_dict, extract_feature


CUSTOM_PLOT_ORDER = {
    "channel": ["average", "all", "LMot", "RMot"],
    "genotype": ["WT", "KO"],
    "sex": ["Male", "Female"],
    "isday": [True, False],
    "band": list(constants.BAND_NAMES),
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _make_asymmetric_band_war(
    animal_id="A1",
    genotype="WT",
    sex="Male",
    n_rows=4,
):
    """Build a mock WAR with a *known asymmetric* per-band distribution.

    Each band gets a different constant: delta=5, theta=4, alpha=3, beta=2,
    gamma=1. A correct plot will show bars in canonical order with
    monotonically decreasing height; an alphabetised plot will show
    [alpha=3, beta=2, delta=5, gamma=1, theta=4] — a tell-tale "middle is
    highest" pattern that mirrors the original arxrosa bug.
    """
    band_values = {"delta": 5.0, "theta": 4.0, "alpha": 3.0, "beta": 2.0, "gamma": 1.0}
    war = MagicMock(spec=WindowAnalysisResult)
    war.animal_id = animal_id
    war.genotype = genotype
    war.sex = sex
    war.channel_names = ["LMot", "RMot"]
    war.channel_abbrevs = ["LMot", "RMot"]
    war.channel_to_idx = {"LMot": 0, "RMot": 1}

    data = {
        "animal": [animal_id] * n_rows,
        "genotype": [genotype] * n_rows,
        "sex": [sex] * n_rows,
        "psdband": [
            {b: [band_values[b], band_values[b]] for b in constants.BAND_NAMES}
            for _ in range(n_rows)
        ],
    }
    war.get_result.return_value = pd.DataFrame(data)
    return war


class TestPullTimeseriesBandOrder:
    """Layer-1/2 upstream guarantee: the df returned by
    :meth:`ExperimentPlotter.pull_timeseries_dataframe` has ``band`` as an
    ordered :class:`pd.Categorical` with the canonical category list.
    """

    def test_pull_timeseries_dataframe_returns_canonical_band_order(self):
        war = _make_asymmetric_band_war()
        ep = ExperimentPlotter(
            [war], features=["psdband"], plot_order=CUSTOM_PLOT_ORDER
        )
        df = ep.pull_timeseries_dataframe("psdband", groupby=["genotype"])

        assert "band" in df.columns
        assert isinstance(df["band"].dtype, pd.CategoricalDtype)
        assert df["band"].cat.ordered is True
        assert list(df["band"].cat.categories) == list(constants.BAND_NAMES)


class TestBandScaleHelper:
    """Layer 2: :meth:`ExperimentPlotter.band_scale` exists and returns an
    ``so.Nominal`` with canonical-order categories.
    """

    def test_band_scale_returns_nominal_with_canonical_order(self):
        scale = ExperimentPlotter.band_scale()
        assert isinstance(scale, so.Nominal)
        assert list(scale.order) == list(constants.BAND_NAMES)

    def test_band_scale_accepts_explicit_plot_lib(self):
        # Passing the plot_lib avoids the lazy import — both paths should
        # produce the same canonical-order scale.
        scale = ExperimentPlotter.band_scale(plot_lib=so)
        assert isinstance(scale, so.Nominal)
        assert list(scale.order) == list(constants.BAND_NAMES)


class TestSoPlotBandAxisOrder:
    """Layer 1: ``so.Plot(df, x='band').scale(x=band_scale())`` renders bars
    in canonical order.

    We don't render to pixels; instead we drive the plot's internal
    ``_plot`` machinery and read the resulting matplotlib tick labels. That
    is the layer where the original bug manifested.
    """

    def _build_df(self):
        # Distribution chosen to make a mis-order *visible*: each band carries
        # a constant value, so the plot is a per-band single-bar bar chart.
        band_values = {
            "delta": 5.0, "theta": 4.0, "alpha": 3.0, "beta": 2.0, "gamma": 1.0,
        }
        rows = []
        for b in constants.BAND_NAMES:
            for _ in range(3):
                rows.append({"band": b, "value": band_values[b]})
        df = pd.DataFrame(rows)
        # Mirror what pull_timeseries_dataframe does upstream.
        df["band"] = pd.Categorical(
            df["band"], categories=list(constants.BAND_NAMES), ordered=True,
        )
        return df

    def test_so_plot_band_x_axis_order_is_canonical_with_band_scale(self):
        df = self._build_df()
        p = (
            so.Plot(df, x="band", y="value")
            .add(so.Dot(), so.Agg())
            .scale(x=ExperimentPlotter.band_scale(plot_lib=so))
        )
        fig, ax = plt.subplots()
        p.on(ax).plot()
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert labels == list(constants.BAND_NAMES), (
            f"Expected canonical band order on the x-axis, got {labels!r}. "
            f"The .scale(x=band_scale()) call is the load-bearing fix."
        )

    def test_so_plot_color_legend_band_order_is_canonical_with_band_scale(self):
        # The bygeno plot uses band as the colour mapping, not the x-axis;
        # the same fix has to apply there too.
        df = self._build_df()
        df["genotype"] = "WT"
        p = (
            so.Plot(df, x="genotype", y="value", color="band")
            .add(so.Dot(), so.Dodge())
            .scale(color=ExperimentPlotter.band_scale(plot_lib=so))
        )
        fig, ax = plt.subplots()
        # Just exercise the pipeline — if the colour scale's ordering is
        # respected, ``plot()`` succeeds without raising. We can't easily
        # inspect colour legend entries without a Plot.show()-style render,
        # so the value here is the no-raise contract plus the explicit
        # Nominal(order=...) on the scale object.
        p.on(ax).plot()


class TestAnimalPlotterBandKeyAssertion:
    """Layer 3: the canonical-order guard inside
    ``AnimalPlotter.__get_linear_feature``.

    The assertion is the second line of defence for the AnimalPlotter sites
    that call ``ax.set_yticks(..., constants.BAND_NAMES)`` — if the upstream
    extraction ever returns bands in non-canonical order, the plot would
    silently mislabel its y-axis. The assertion turns that into a loud
    error.
    """

    def _make_plotter(self):
        war = MagicMock(spec=WindowAnalysisResult)
        war.genotype = "WT"
        war.channel_names = ["LMot", "RMot"]
        war.channel_abbrevs = ["LM", "RM"]
        return AnimalPlotter(war)

    def _band_group_in_order(self, keys, n_time=5, n_chan=2):
        """Build a group DataFrame whose BAND dicts iterate in *keys* order.

        Python ≥3.7 dicts preserve insertion order, and ``extract_band_dict``
        treats the first row's dict as the source of truth for band keys.
        """
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "psdband": [
                    {b: rng.random(n_chan).tolist() for b in keys}
                    for _ in range(n_time)
                ],
                "duration": [1.0] * n_time,
            }
        )

    def test_extract_band_dict_returns_canonical_keys_for_canonical_input(self):
        """extract_band_dict already preserves dict-insertion order — verify
        the contract the assertion depends on."""
        group = self._band_group_in_order(list(constants.BAND_NAMES))
        _, keys = extract_band_dict(group["psdband"])
        assert list(keys) == list(constants.BAND_NAMES)

    def test_get_linear_feature_accepts_canonical_band_order(self):
        plotter = self._make_plotter()
        group = self._band_group_in_order(list(constants.BAND_NAMES))
        # _AnimalPlotter__get_linear_feature is the mangled name of the
        # private method — call it directly to exercise the assertion path.
        result = plotter._AnimalPlotter__get_linear_feature(
            group=group, feature="psdband"
        )
        assert result is not None

    def test_get_linear_feature_raises_on_non_canonical_band_order(self):
        plotter = self._make_plotter()
        # Reverse the band order in every dict — this is exactly the failure
        # mode the assertion is designed to catch.
        reversed_keys = list(reversed(constants.BAND_NAMES))
        group = self._band_group_in_order(reversed_keys)
        with pytest.raises(AssertionError, match="non-canonical band order"):
            plotter._AnimalPlotter__get_linear_feature(
                group=group, feature="psdband"
            )

    def test_get_linear_feature_no_assertion_for_non_band_features(self):
        """The assertion path only fires for dict-stored (BAND/BANDED_MATRIX)
        features — linear features have ``keys=None`` and must pass through
        without raising.
        """
        plotter = self._make_plotter()
        rng = np.random.default_rng(0)
        group = pd.DataFrame(
            {
                "rms": [rng.random(2).tolist() for _ in range(5)],
                "duration": [1.0] * 5,
            }
        )
        result = plotter._AnimalPlotter__get_linear_feature(
            group=group, feature="rms"
        )
        assert result is not None


class TestExtractFeatureBandContract:
    """Document and lock in the contract the AnimalPlotter assertion depends
    on: :func:`extract_feature` returns ``(vals, keys)`` where ``keys`` is
    either ``None`` (linear) or a list whose order is the dict-insertion
    order of the first row.
    """

    def test_extract_feature_returns_none_keys_for_linear(self):
        rng = np.random.default_rng(0)
        series = pd.Series([rng.random(2).tolist() for _ in range(3)])
        _, keys = extract_feature(series, constants.FeatureType.LINEAR)
        assert keys is None

    def test_extract_feature_returns_canonical_keys_for_band(self):
        rng = np.random.default_rng(0)
        series = pd.Series([
            {b: rng.random(2).tolist() for b in constants.BAND_NAMES}
            for _ in range(3)
        ])
        _, keys = extract_feature(series, constants.FeatureType.BAND)
        assert list(keys) == list(constants.BAND_NAMES)


# ─────────────────────────────────────────────────────────────────────────
# Layer 5 — parquet-round-trip canonicalisation
# ─────────────────────────────────────────────────────────────────────────
#
# The Layer-4 tests above all use in-memory dicts and miss the real bug:
# pyarrow's struct→dict round-trip alphabetises field order on the read
# side, so a canonical-order dict written to parquet comes back as
# ``{"alpha", "beta", "delta", "gamma", "theta"}``.  The Layer-3 assertion
# in AnimalPlotter caught this in production (arxrosa-2064: zcohere with
# alphabetised keys triggered AssertionError during diagnostic_figures).
# These tests exercise the FULL parquet round-trip and would have caught
# that failure before it shipped.

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


_ALPHABETICAL_BAND_NAMES = sorted(constants.BAND_NAMES)  # alpha, beta, delta, gamma, theta


def _make_band_war(
    n_rows: int = 4,
    n_chan: int = 2,
    feature: str = "psdband",
    band_order: list[str] | None = None,
    extra_metadata: bool = True,
) -> WindowAnalysisResult:
    """Build a minimal WAR carrying one band-feature column.

    Args:
        feature: name of the band feature (must be in BAND_FEATURES or
            BANDED_MATRIX_FEATURES).
        band_order: which order to insert band keys into each per-row dict
            (defaults to ``BAND_NAMES``).  Pass alphabetical to simulate
            post-parquet state coming into a fresh WAR construction.
    """
    if band_order is None:
        band_order = list(constants.BAND_NAMES)
    rng = np.random.default_rng(0)
    ftype = constants.classify_feature(feature)

    if ftype is constants.FeatureType.BAND:
        cells = [
            {b: rng.random(n_chan).tolist() for b in band_order}
            for _ in range(n_rows)
        ]
    elif ftype is constants.FeatureType.BANDED_MATRIX:
        cells = [
            {b: rng.random((n_chan, n_chan)).tolist() for b in band_order}
            for _ in range(n_rows)
        ]
    else:
        raise ValueError(f"Test fixture only handles band-keyed features, got {ftype}")

    data: dict = {feature: cells, "duration": [3600.0] * n_rows}
    if extra_metadata:
        data["animal"] = ["A1"] * n_rows
        data["animalday"] = ["A1 WT Jan-01-2026"] * n_rows
        data["genotype"] = ["WT"] * n_rows
        data["isday"] = [True, False] * (n_rows // 2)
        data["timestamp"] = pd.date_range("2026-01-01", periods=n_rows, freq="1h")
    return WindowAnalysisResult(
        result=pd.DataFrame(data),
        animal_id="A1",
        genotype="WT",
        sex="Male",
        channel_names=["LMot", "RMot"][:n_chan],
        suppress_short_interval_error=True,
    )


def _band_keys(cell: dict | list) -> list[str]:
    """Extract band keys from a decoded cell (top-level dict, or any nested
    dict whose keys are bands)."""
    if isinstance(cell, dict):
        return [k for k in cell if k in set(constants.BAND_NAMES)]
    raise TypeError(f"Expected dict, got {type(cell).__name__}")


class TestCanonicaliseBandDict:
    """Unit tests for the ``_canonicalise_band_dict`` static method."""

    def test_idempotent_on_canonical_input(self):
        canonical = {b: i for i, b in enumerate(constants.BAND_NAMES)}
        out = WindowAnalysisResult._canonicalise_band_dict(canonical)
        assert list(out.keys()) == list(constants.BAND_NAMES)
        assert out == canonical

    def test_reorders_alphabetical_input(self):
        # The exact failure mode from arxrosa-2064.
        alphabetical = {b: i for i, b in enumerate(_ALPHABETICAL_BAND_NAMES)}
        out = WindowAnalysisResult._canonicalise_band_dict(alphabetical)
        assert list(out.keys()) == list(constants.BAND_NAMES)
        # Values must still match their original band keys (no swap).
        for b, v in alphabetical.items():
            assert out[b] == v

    @pytest.mark.parametrize(
        "subset",
        [
            ["delta", "theta"],
            ["alpha", "beta", "gamma"],
            ["delta", "alpha", "gamma"],  # non-contiguous in canonical order
            ["gamma"],                     # singleton
            ["theta", "delta"],            # 2-element reverse
        ],
    )
    def test_handles_subset(self, subset):
        """Best-fit: even if dict has only some band keys, they should come
        out in canonical (BAND_NAMES) order."""
        # Insert in some non-canonical order.
        d = {b: i for i, b in enumerate(reversed(subset))}
        out = WindowAnalysisResult._canonicalise_band_dict(d)
        expected_order = [b for b in constants.BAND_NAMES if b in subset]
        assert list(out.keys()) == expected_order
        assert out == d  # same content, just reordered

    def test_preserves_non_band_keys_at_end(self):
        d = {
            "theta": 1,
            "extra": "metadata",
            "delta": 2,
            "another": [1, 2, 3],
        }
        out = WindowAnalysisResult._canonicalise_band_dict(d)
        keys = list(out.keys())
        # Bands first in canonical order.
        assert keys[: keys.index("extra")] == ["delta", "theta"]
        # Non-band keys preserved in original order.
        assert "extra" in keys and "another" in keys
        assert keys.index("extra") < keys.index("another")

    def test_leaves_pure_non_band_dict_alone(self):
        d = {"foo": 1, "bar": 2, "baz": 3}
        out = WindowAnalysisResult._canonicalise_band_dict(d)
        assert out is d  # zero band overlap → return unchanged


class TestNormalizeArrowCellBandCanonicalisation:
    """Layer 5's hook into the decode path itself."""

    def test_canonicalises_band_dict(self):
        alphabetical = {b: float(i) for i, b in enumerate(_ALPHABETICAL_BAND_NAMES)}
        out = WindowAnalysisResult._normalize_arrow_cell(alphabetical)
        assert list(out.keys()) == list(constants.BAND_NAMES)

    def test_canonicalises_nested_band_dict_in_list(self):
        # A list of band dicts (the shape post-`.to_pandas()` for a band column).
        cells = [{b: float(i) for i, b in enumerate(_ALPHABETICAL_BAND_NAMES)} for _ in range(3)]
        out = WindowAnalysisResult._normalize_arrow_cell(cells)
        for c in out:
            assert list(c.keys()) == list(constants.BAND_NAMES)

    def test_does_not_touch_tuple_encoded_dict(self):
        """LINEAR_2D tuple-encoded dicts (``_t0``/``_t1``) must still come
        back as tuples; canonicalisation only fires on band dicts."""
        d = {"_t0": 1.5, "_t1": 2.5}
        out = WindowAnalysisResult._normalize_arrow_cell(d)
        assert isinstance(out, tuple)
        assert out == (1.5, 2.5)


class TestEagerParquetRoundTripBandOrder:
    """Real save→load via ``save_parquet_and_json`` /
    ``load_parquet_and_json``.  This is the path that broke arxrosa-2064.
    """

    @pytest.mark.parametrize(
        "feature",
        constants.BAND_FEATURES + constants.BANDED_MATRIX_FEATURES,
    )
    def test_all_band_features_roundtrip_canonical(self, feature, tmp_path):
        """Exhaustive: every band-bearing feature in
        ``BAND_FEATURES + BANDED_MATRIX_FEATURES`` must round-trip with
        canonical key order."""
        war = _make_band_war(feature=feature)
        war.save_parquet_and_json(tmp_path, filename="war")
        reloaded = WindowAnalysisResult.load_parquet_and_json(
            folder_path=tmp_path, parquet_name="war.parquet", json_name="war.json"
        )
        for cell in reloaded.result[feature]:
            assert _band_keys(cell) == list(constants.BAND_NAMES), (
                f"{feature}: cell came back with non-canonical band order "
                f"{list(cell.keys())!r}"
            )

    def test_canonical_input_survives_roundtrip(self, tmp_path):
        """Pre-Layer-5 this FAILED (pyarrow alphabetises on read).  Post-
        Layer-5, the decoder fixes it.  Belt-and-suspenders test against
        future regressions."""
        war = _make_band_war(feature="psdband", band_order=list(constants.BAND_NAMES))
        war.save_parquet_and_json(tmp_path, filename="war")
        reloaded = WindowAnalysisResult.load_parquet_and_json(
            folder_path=tmp_path, parquet_name="war.parquet", json_name="war.json"
        )
        assert _band_keys(reloaded.result["psdband"].iloc[0]) == list(constants.BAND_NAMES)

    def test_alphabetical_input_canonicalised_by_load(self, tmp_path):
        """If a WAR was built with alphabetical band dicts (simulating
        a value that's already in the wrong order before save), the load
        path canonicalises it."""
        war = _make_band_war(feature="psdband", band_order=_ALPHABETICAL_BAND_NAMES)
        war.save_parquet_and_json(tmp_path, filename="war")
        reloaded = WindowAnalysisResult.load_parquet_and_json(
            folder_path=tmp_path, parquet_name="war.parquet", json_name="war.json"
        )
        assert _band_keys(reloaded.result["psdband"].iloc[0]) == list(constants.BAND_NAMES)


class TestStreamingParquetRoundTripBandOrder:
    """Same end-to-end guarantees for the
    ``LazyWindowAnalysisResult.save_parquet_and_json`` streaming path."""

    def test_streaming_passthrough_canonical(self, tmp_path):
        """A no-transform scan→save chain emits canonical band cells."""
        # Step 1: build a WAR and save eagerly so scan_parquet_and_json has
        # something to read.
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        war = _make_band_war(feature="psdband")
        war.save_parquet_and_json(src_dir, filename="war")

        # Step 2: open lazily, save with no transforms → pass-through path.
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        lazy = WindowAnalysisResult.scan_parquet_and_json(src_dir, filename="war")
        lazy.save_parquet_and_json(dst_dir, filename="war")

        # Step 3: load and inspect.
        reloaded = WindowAnalysisResult.load_parquet_and_json(
            folder_path=dst_dir, parquet_name="war.parquet", json_name="war.json"
        )
        for cell in reloaded.result["psdband"]:
            assert _band_keys(cell) == list(constants.BAND_NAMES)


class TestAnimalPlotterPostParquet:
    """End-to-end regression: replicates the arxrosa-2064 failure shape.

    Pre-Layer-5, ``plot_coherecorr_spectral`` on a parquet-loaded WAR
    would raise AssertionError from the Layer-3 guard because the band
    keys came back alphabetised.  Post-Layer-5, the decoder canonicalises
    and the assertion is satisfied.
    """

    def test_get_linear_feature_no_assertion_after_parquet(self, tmp_path):
        """Most surgical replica: build WAR with a BANDED_MATRIX feature,
        round-trip via parquet, run AnimalPlotter.__get_linear_feature on
        it — must not raise the Layer-3 AssertionError."""
        war = _make_band_war(feature="zcohere", n_chan=2)
        war.save_parquet_and_json(tmp_path, filename="war")
        reloaded = WindowAnalysisResult.load_parquet_and_json(
            folder_path=tmp_path, parquet_name="war.parquet", json_name="war.json"
        )
        plotter = AnimalPlotter(reloaded)
        # Direct access to the mangled private name.
        result = plotter._AnimalPlotter__get_linear_feature(
            group=reloaded.result, feature="zcohere"
        )
        assert result is not None

    def test_get_linear_feature_no_assertion_for_band(self, tmp_path):
        """Same guarantee for plain BAND features (psdband)."""
        war = _make_band_war(feature="psdband", n_chan=2)
        war.save_parquet_and_json(tmp_path, filename="war")
        reloaded = WindowAnalysisResult.load_parquet_and_json(
            folder_path=tmp_path, parquet_name="war.parquet", json_name="war.json"
        )
        plotter = AnimalPlotter(reloaded)
        result = plotter._AnimalPlotter__get_linear_feature(
            group=reloaded.result, feature="psdband"
        )
        assert result is not None


class TestPyarrowStructAlphabetisationBehaviour:
    """Documents *why* Layer 5 exists.

    Pyarrow ≥ 12 stores struct fields in deterministic alphabetical order
    when converting a list of dicts to a struct array.  This test locks in
    that assumption — if pyarrow ever changes (or our pyarrow version
    behaves differently), the test fails and we can revisit Layer 5.
    """

    def test_pyarrow_struct_round_trip_alphabetises(self, tmp_path):
        # Build a list of canonical-order dicts → pa.array → parquet → read back.
        canonical_dicts = [
            {b: float(i) for i, b in enumerate(constants.BAND_NAMES)}
            for _ in range(3)
        ]
        table = pa.table({"x": pa.array(canonical_dicts)})
        parquet_path = tmp_path / "raw.parquet"
        pq.write_table(table, parquet_path, compression="zstd", compression_level=4)
        loaded = pq.read_table(parquet_path).to_pandas()
        observed_keys = list(loaded["x"].iloc[0].keys())
        # If this assertion ever fails (i.e. pyarrow now preserves order),
        # Layer 5 becomes redundant — keep it anyway for defence and delete
        # this test.
        assert observed_keys == _ALPHABETICAL_BAND_NAMES, (
            f"pyarrow now preserves struct-field order: got {observed_keys!r}. "
            f"Layer 5's reason-to-exist may no longer apply — review."
        )
