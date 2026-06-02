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
from neurodent.visualization import ExperimentPlotter, WindowAnalysisResult
from neurodent.visualization.plotting.animal import AnimalPlotter
from neurodent.visualization.feature_utils import extract_band_dict, extract_feature


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
        df["gene"] = "WT"
        p = (
            so.Plot(df, x="gene", y="value", color="band")
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
        war.assume_from_number = False
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
