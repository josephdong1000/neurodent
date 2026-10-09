"""Averaging animaldays weights them by how much recording each one holds.

A row reaching :meth:`ExperimentPlotter.pull_timeseries_dataframe` is an animalday once the
WARs have been flattened, and animaldays are not the same length. An animal recorded on two
rigs in sequence can carry one 70 hour animalday beside five 18 hour ones, so an unweighted
mean gives 44 percent of that animal's data 17 percent of the weight.
"""

import warnings
from unittest.mock import MagicMock

import matplotlib

import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from neurodent import constants
from neurodent.core.utils import nanaverage_series_of_np, nanmean_series_of_np
from neurodent.plotting import ExperimentPlotter
from neurodent.results import WindowAnalysisResult

CHANNELS = ["LMot", "RMot"]
ABBREVS = ["LM", "RM"]

PLOT_ORDER = {
    # "average" is what collapse_channels=True names the channel; "all" is what an
    # uncollapsed matrix feature names it.
    "channel": ABBREVS + CHANNELS + ["average", "all"],
    "genotype": ["WT", "KO"],
    "sex": ["Male", "Female"],
    "isday": [True, False],
    "band": constants.BAND_NAMES,
}

LONG_DAY_SECONDS = 70 * 3600.0
SHORT_DAY_SECONDS = 18 * 3600.0


def _war(animal_id, values, durations, genotype="WT"):
    """A WAR whose flattened result has one row per animalday, with explicit durations."""
    war = MagicMock(spec=WindowAnalysisResult)
    war.animal_id = animal_id
    war.genotype = genotype
    war.sex = "Male"
    war.channel_names = CHANNELS
    war.channel_abbrevs = ABBREVS
    war.get_result.return_value = pd.DataFrame(
        {
            "animal": [animal_id] * len(values),
            "genotype": [genotype] * len(values),
            "sex": ["Male"] * len(values),
            "duration": list(durations),
            "rms": [list(row) for row in values],
        }
    )
    return war


def _one_long_five_short(**kwargs):
    """One 70 hour animalday reading 10, five 18 hour ones reading 0."""
    values = [[10.0, 10.0]] + [[0.0, 0.0]] * 5
    durations = [LONG_DAY_SECONDS] + [SHORT_DAY_SECONDS] * 5
    return _war("A1", values, durations, **kwargs)


def _pull(war, **kwargs):
    plotter = ExperimentPlotter([war], features=["rms"], plot_order=PLOT_ORDER)
    return plotter.pull_timeseries_dataframe(
        "rms", groupby=["animal", "genotype", "sex"], average_groupby=True, **kwargs
    )


class TestPullTimeseriesDataframeWeighting:
    def test_long_animalday_gets_its_share_of_the_weight(self):
        """The 70 hour day carries 43.75 percent of the animal's time, so 43.75 of the mean.

        The unweighted answer is 10/6 = 1.667, which is the number this change replaces.
        """
        expected = (LONG_DAY_SECONDS * 10.0) / (LONG_DAY_SECONDS + 5 * SHORT_DAY_SECONDS)
        assert expected == pytest.approx(4.375)

        df = _pull(_one_long_five_short())

        assert len(df) == len(ABBREVS)
        for value in df["rms"]:
            assert float(np.ravel(value)[0]) == pytest.approx(4.375)

    def test_opting_out_gives_the_unweighted_mean(self):
        df = _pull(_one_long_five_short(), weight_by_duration=False)

        for value in df["rms"]:
            assert float(np.ravel(value)[0]) == pytest.approx(10.0 / 6.0)

    def test_equal_durations_match_the_unweighted_mean(self):
        """Weighting must be a no-op when every row is the same length."""
        values = [[4.0, 8.0], [6.0, 2.0]]
        war = _war("A1", values, [SHORT_DAY_SECONDS] * 2)
        weighted = _pull(war)
        unweighted = _pull(
            _war("A1", values, [SHORT_DAY_SECONDS] * 2), weight_by_duration=False
        )

        np.testing.assert_allclose(
            np.stack([np.ravel(v) for v in weighted["rms"]]),
            np.stack([np.ravel(v) for v in unweighted["rms"]]),
        )

    def test_duration_is_not_a_column_of_the_result(self):
        """It is a weight, not a grouping key, and leaking it would melt into the channels."""
        df = _pull(_one_long_five_short())

        assert "duration" not in df.columns
        assert set(df["channel"]) == set(ABBREVS)

    def test_missing_duration_warns_and_falls_back(self):
        war = _one_long_five_short()
        war.get_result.return_value = war.get_result.return_value.drop(columns=["duration"])

        with pytest.warns(UserWarning, match="counts every row equally"):
            df = _pull(war)

        for value in df["rms"]:
            assert float(np.ravel(value)[0]) == pytest.approx(10.0 / 6.0)

    def test_no_warning_when_opted_out(self):
        war = _one_long_five_short()
        war.get_result.return_value = war.get_result.return_value.drop(columns=["duration"])

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _pull(war, weight_by_duration=False)


class TestEpFiguresSecondAggregation:
    """``generate_ep_figures`` pivots the pulled frame with ``aggfunc="mean"``.

    That is a second, unweighted aggregation sitting downstream of the weighted one, so it
    could undo the weighting. It does not, because the pull it is given already collapses to
    one row per pivot cell. These tests pin that, since the day it stops holding the
    weighting is silently lost again.
    """

    @staticmethod
    def _uneven_day_and_night():
        """One long animalday and two short ones, each split across day and night."""
        values, durations, isday = [], [], []
        for seconds, reading in [(LONG_DAY_SECONDS, 10.0), (SHORT_DAY_SECONDS, 0.0),
                                 (SHORT_DAY_SECONDS, 0.0)]:
            for day in (True, False):
                values.append([reading, reading])
                durations.append(seconds)
                isday.append(day)
        war = _war("A1", values, durations)
        war.get_result.return_value["isday"] = isday
        return war

    def _pull_as_ep_figures_does(self):
        plotter = ExperimentPlotter(
            [self._uneven_day_and_night()], features=["rms"], plot_order=PLOT_ORDER
        )
        return plotter.pull_timeseries_dataframe(
            feature="rms",
            groupby=["animal", "genotype", "sex", "isday"],
            collapse_channels=True,
            average_groupby=True,
        )

    def test_pivot_cells_hold_exactly_one_value(self):
        df = self._pull_as_ep_figures_does()

        sizes = df.groupby(
            ["animal", "genotype", "sex", "isday", "channel"], observed=True
        ).size()
        assert sizes.max() == 1, f"pivot_table would re-average these groups: {sizes}"

    def test_pivoted_value_is_still_the_weighted_one(self):
        df = self._pull_as_ep_figures_does()
        df["rms"] = df["rms"].apply(lambda v: float(np.ravel(v)[0]))

        pivoted = df.pivot_table(
            index=["animal", "genotype", "sex"],
            columns=["isday"],
            values="rms",
            aggfunc="mean",
            observed=True,
        )

        expected = (LONG_DAY_SECONDS * 10.0) / (LONG_DAY_SECONDS + 2 * SHORT_DAY_SECONDS)
        assert expected == pytest.approx(70.0 / 106.0 * 10.0)
        for value in np.ravel(pivoted.to_numpy()):
            assert value == pytest.approx(expected)


class TestReturnContainerMatchesUnweighted:
    """The weighted and unweighted averages must return the same kind of object.

    They did not: ``np.where`` yields a 0-d array for scalar rows where ``np.nanmean`` yields
    a numpy scalar, which made the averaged column object dtype and took the EP figure rule
    down with a PlotSpecError. The numbers were right the whole time, so only a dtype
    assertion catches it. Asserting on ``float(np.ravel(v)[0])`` does not.
    """

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([1.0, 3.0], id="scalars"),
            pytest.param([np.array([1.0, 2.0]), np.array([3.0, 4.0])], id="vectors"),
            pytest.param([np.zeros((2, 2)), np.full((2, 2), 4.0)], id="matrices"),
        ],
    )
    def test_container_type_matches(self, values):
        series = pd.Series(values)
        weights = pd.Series([3.0, 1.0])

        unweighted = nanmean_series_of_np(series)
        weighted = nanaverage_series_of_np(series, weights)

        assert type(weighted) is type(unweighted)
        assert np.ndim(weighted) == np.ndim(unweighted)

    def test_zero_weight_fallback_matches_too(self):
        """The fallback and the weighted path are two branches of one function."""
        series = pd.Series([1.0, 3.0])

        fallback = nanaverage_series_of_np(series, pd.Series([0.0, 0.0]))
        weighted = nanaverage_series_of_np(series, pd.Series([3.0, 1.0]))

        assert type(fallback) is type(weighted)

    def test_averaged_column_is_not_object_dtype(self):
        """The defect only shows as a dtype on the assembled frame."""
        df = _pull(_one_long_five_short())

        assert df["rms"].dtype != object, (
            "an object column reaches seaborn as unhashable arrays and the EP rule aborts"
        )
        assert df["rms"].dtype == np.dtype("float64")

    def test_unweighted_pull_has_the_same_dtype(self):
        weighted = _pull(_one_long_five_short())
        unweighted = _pull(_one_long_five_short(), weight_by_duration=False)

        assert weighted["rms"].dtype == unweighted["rms"].dtype


class TestDurationNamedInGroupby:
    """Grouping BY duration makes it a key, so the weighting must stand down, not crash."""

    def test_does_not_raise(self):
        war = _one_long_five_short()
        plotter = ExperimentPlotter([war], features=["rms"], plot_order=PLOT_ORDER)

        df = plotter.pull_timeseries_dataframe(
            "rms",
            groupby=["animal", "genotype", "sex", "duration"],
            average_groupby=True,
        )

        assert "duration" in df.columns
        assert df["rms"].dtype != object

    def test_groups_are_unweighted_within_themselves(self):
        """Every row in such a group shares one duration, so the means coincide."""
        war = _one_long_five_short()
        plotter = ExperimentPlotter([war], features=["rms"], plot_order=PLOT_ORDER)
        groupby = ["animal", "genotype", "sex", "duration"]

        weighted = plotter.pull_timeseries_dataframe(
            "rms", groupby=groupby, average_groupby=True
        )
        unweighted = plotter.pull_timeseries_dataframe(
            "rms", groupby=groupby, average_groupby=True, weight_by_duration=False
        )

        np.testing.assert_allclose(
            weighted.sort_values(groupby + ["channel"])["rms"].to_numpy(dtype=float),
            unweighted.sort_values(groupby + ["channel"])["rms"].to_numpy(dtype=float),
        )


class TestRenderPathsAgreeWithExportedNumbers:
    """The figures and the exported data must use the same weighting.

    ``generate_ep_heatmaps`` exports from a pull with ``average_groupby=True`` and then plots
    via ``plot_heatmap``, which pulls again with ``average_groupby=False`` and averages at
    render time in ``_plot_matrix``. Weighting only the first of those would publish a figure
    and a CSV that disagree for the same grouping.
    """

    @staticmethod
    def _matrix_war():
        """One long animalday reading 1.0 and two short ones reading 0.0, as 2x2 matrices."""
        war = MagicMock(spec=WindowAnalysisResult)
        war.animal_id = "A1"
        war.genotype = "WT"
        war.sex = "Male"
        war.channel_names = CHANNELS
        war.channel_abbrevs = ABBREVS
        readings = [1.0, 0.0, 0.0]
        durations = [LONG_DAY_SECONDS, SHORT_DAY_SECONDS, SHORT_DAY_SECONDS]
        war.get_result.return_value = pd.DataFrame(
            {
                "animal": ["A1"] * 3,
                "genotype": ["WT"] * 3,
                "sex": ["Male"] * 3,
                "duration": durations,
                "pcorr": [np.full((2, 2), r).tolist() for r in readings],
            }
        )
        return war

    def test_render_time_matrix_average_is_weighted(self):
        plotter = ExperimentPlotter(
            [self._matrix_war()], features=["pcorr"], plot_order=PLOT_ORDER
        )
        expected = LONG_DAY_SECONDS / (LONG_DAY_SECONDS + 2 * SHORT_DAY_SECONDS)
        assert expected == pytest.approx(70.0 / 106.0)

        # What the heatmap path renders: no averaging in the pull, averaged in _plot_matrix.
        unaveraged = plotter.pull_timeseries_dataframe(
            "pcorr", groupby=["animal"], average_groupby=False
        )
        assert "duration" in unaveraged.columns, (
            "_plot_matrix cannot weight what the pull did not carry"
        )
        rendered = nanaverage_series_of_np(
            unaveraged["pcorr"], unaveraged["duration"]
        )

        np.testing.assert_allclose(np.ravel(rendered), expected)

    def test_averaging_pull_drops_duration_but_render_pull_keeps_it(self):
        plotter = ExperimentPlotter(
            [self._matrix_war()], features=["pcorr"], plot_order=PLOT_ORDER
        )

        averaged = plotter.pull_timeseries_dataframe(
            "pcorr", groupby=["animal"], average_groupby=True
        )
        unaveraged = plotter.pull_timeseries_dataframe(
            "pcorr", groupby=["animal"], average_groupby=False
        )

        assert "duration" not in averaged.columns
        assert "duration" in unaveraged.columns

    def test_plot_catplot_renders_with_weighting_on(self):
        """The call shape docs/tutorials/visualization.ipynb makes, which must not raise."""
        plotter = ExperimentPlotter(
            [_one_long_five_short()], features=["rms"], plot_order=PLOT_ORDER
        )

        grid = plotter.plot_catplot(
            "rms", groupby=["animal", "genotype", "sex"], average_groupby=True
        )

        assert grid is not None


class TestNanaverageSeriesOfNp:
    def test_weighted_average_of_arrays(self):
        values = pd.Series([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        out = nanaverage_series_of_np(values, pd.Series([3.0, 1.0]))

        np.testing.assert_allclose(out, [1.5, 2.5])

    def test_nan_position_does_not_consume_its_rows_weight(self):
        """A channel missing from the long row must not inherit the long row's weight.

        Masking per row would give the second position 2.0; masking per position gives 4.0,
        which is the only value actually observed there.
        """
        values = pd.Series([np.array([1.0, np.nan]), np.array([2.0, 4.0])])
        out = nanaverage_series_of_np(values, pd.Series([9.0, 1.0]))

        np.testing.assert_allclose(out, [1.1, 4.0])

    def test_all_nan_position_is_nan(self):
        values = pd.Series([np.array([1.0, np.nan]), np.array([2.0, np.nan])])
        out = nanaverage_series_of_np(values, pd.Series([1.0, 1.0]))

        assert out[0] == pytest.approx(1.5)
        assert np.isnan(out[1])

    def test_zero_weights_fall_back_to_the_unweighted_mean(self):
        """A zero total says nothing about how to combine the rows, so do not divide by it."""
        values = pd.Series([np.array([1.0, 2.0]), np.array([3.0, 6.0])])
        out = nanaverage_series_of_np(values, pd.Series([0.0, 0.0]))

        np.testing.assert_allclose(out, [2.0, 4.0])

    def test_non_finite_weight_is_treated_as_zero(self):
        values = pd.Series([np.array([1.0]), np.array([5.0])])
        out = nanaverage_series_of_np(values, pd.Series([np.nan, 2.0]))

        np.testing.assert_allclose(out, [5.0])

    def test_length_mismatch_raises(self):
        values = pd.Series([np.array([1.0]), np.array([2.0])])

        with pytest.raises(ValueError, match="same length"):
            nanaverage_series_of_np(values, pd.Series([1.0]))

    def test_matrix_valued_rows_are_weighted_elementwise(self):
        values = pd.Series([np.zeros((2, 2)), np.full((2, 2), 4.0)])
        out = nanaverage_series_of_np(values, pd.Series([1.0, 3.0]))

        np.testing.assert_allclose(out, np.full((2, 2), 3.0))
