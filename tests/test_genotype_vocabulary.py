"""Tests for the config-driven genotype vocabulary: sort order, EP baseline, palettes.

These cover a class of defect the rest of the suite is structurally blind to. The pipeline
carries several genotype vocabularies that default to ``["WT", "KO"]``, and a dataset whose
genotypes are something else does not crash: it silently sorts differently, picks a
different EP difference-heatmap baseline, and reuses colours. Nothing asserts on any of
that today, so a change to the ordering mechanism can regress a shipped dataset's published
figures with the whole suite green.
"""

import copy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

from neurodent import constants
from neurodent.workflow.utils.config import apply_samples_config
from neurodent.workflow.utils.plotting_helpers import create_genotype_color_scale

CONFIG_DIR = Path(__file__).parent.parent / "config" / "datasets"
SHIPPED_CONFIGS = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))


@pytest.fixture(autouse=True)
def restore_constants():
    """Undo the global mutation apply_samples_config performs."""
    saved = (
        copy.deepcopy(constants.DF_SORT_ORDER),
        copy.deepcopy(constants.GENOTYPE_MAP),
        copy.deepcopy(constants.SEX_MAP),
        copy.deepcopy(constants.CHANNEL_MAP),
    )
    yield
    constants.DF_SORT_ORDER.clear()
    constants.DF_SORT_ORDER.update(saved[0])
    constants.GENOTYPE_MAP, constants.SEX_MAP = saved[1], saved[2]
    constants.set_channel_map(saved[3])


@pytest.fixture
def ep_heatmaps():
    spec = importlib.util.spec_from_file_location(
        "generate_ep_heatmaps",
        Path(__file__).parent.parent / "workflow" / "scripts" / "generate_ep_heatmaps.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPlotOrderIsOptIn:
    """``plot_order`` replaces only what a dataset declares."""

    def test_absent_leaves_module_default(self):
        default = list(constants.DF_SORT_ORDER["genotype"])
        apply_samples_config({"animals": []})
        assert constants.DF_SORT_ORDER["genotype"] == default

    def test_declared_order_is_installed(self):
        apply_samples_config({"plot_order": {"genotype": ["x/y", "+/y"]}})
        assert constants.DF_SORT_ORDER["genotype"] == ["x/y", "+/y"]

    def test_undeclared_columns_untouched(self):
        band = list(constants.DF_SORT_ORDER["band"])
        apply_samples_config({"plot_order": {"genotype": ["x/y"]}})
        assert constants.DF_SORT_ORDER["band"] == band

    def test_mutated_in_place_so_prior_importers_see_it(self):
        # set_channel_map mutates rather than rebinds for this reason; a module that did
        # `from ... import DF_SORT_ORDER` at import time must see the update.
        borrowed = constants.DF_SORT_ORDER
        apply_samples_config({"plot_order": {"genotype": ["a", "b"]}})
        assert borrowed["genotype"] == ["a", "b"]

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError, match="unknown column"):
            apply_samples_config({"plot_order": {"nonsense": ["a"]}})

    @pytest.mark.parametrize("order", [[], "notalist", None])
    def test_malformed_order_raises(self, order):
        with pytest.raises(ValueError, match="non-empty list"):
            apply_samples_config({"plot_order": {"genotype": order}})

    @pytest.mark.parametrize("config_name", SHIPPED_CONFIGS)
    def test_shipped_configs_keep_a_usable_order(self, config_name):
        """Every shipped dataset must end up with a non-empty genotype order.

        This is the regression guard for the rejected design. Deriving the order from
        GENOTYPE_MAP would have emptied it for the five shipped configs that declare no
        such map, and ``sort_dataframe_by_plot_order`` raises on any value absent from the
        order. An empty list here is a broken dataset.
        """
        samples = yaml.safe_load((CONFIG_DIR / config_name).read_text()).get("samples_data")
        if not samples:
            pytest.skip(f"{config_name} declares no samples_data")
        apply_samples_config(samples)
        assert constants.DF_SORT_ORDER["genotype"], config_name
        assert constants.DF_SORT_ORDER["sex"], config_name

    @pytest.mark.parametrize("config_name", SHIPPED_CONFIGS)
    def test_declared_plot_order_covers_the_genotypes_assigned(self, config_name):
        """A declared ``plot_order`` must cover every genotype that dataset assigns.

        Scoped to configs that opt in, because that is the actual contract. Only two of the
        genotype consumers (``generate_ep_figures`` and ``generate_ep_heatmaps``) call
        ``extend_plot_order_from_attr``, so a dataset without ``plot_order`` is carried by
        that runtime extension; one that declares a partial order is not, and
        ``sort_dataframe_by_plot_order`` raises on the first value it omits.
        """
        samples = yaml.safe_load((CONFIG_DIR / config_name).read_text()).get("samples_data")
        if not samples or not samples.get("plot_order", {}).get("genotype"):
            pytest.skip(f"{config_name} declares no genotype plot_order")
        assigned = {
            a["genotype"] for a in samples.get("animals", [])
            if not a.get("exclude") and a.get("genotype")
        }
        missing = assigned - set(samples["plot_order"]["genotype"])
        assert not missing, f"{config_name}: {sorted(missing)} assigned but not in plot_order"

    @pytest.mark.parametrize("config_name", SHIPPED_CONFIGS)
    def test_runtime_extension_covers_configs_that_do_not_opt_in(self, config_name):
        """Datasets without ``plot_order`` must still end up orderable at plot time.

        Four shipped configs assign genotypes absent from the ``["WT", "KO"]`` default
        (ap3b2_rhd, arx_parv, arx_rosa, sox5_bin), so they depend entirely on
        ``extend_plot_order_from_attr`` appending what it observes. This asserts that
        dependency holds, and is the guard against a change that empties the base order:
        extension appends to the base, so an empty base still works here but loses every
        canonical position, which is why the base must never be derived from a map that
        most datasets do not declare.
        """
        from neurodent.workflow.utils.plotting_helpers import extend_plot_order_from_attr

        samples = yaml.safe_load((CONFIG_DIR / config_name).read_text()).get("samples_data")
        if not samples or not samples.get("animals"):
            pytest.skip(f"{config_name} declares no animals")
        apply_samples_config(samples)
        assigned = sorted(
            {a["genotype"] for a in samples["animals"]
             if not a.get("exclude") and a.get("genotype")}
        )
        if not assigned:
            pytest.skip(f"{config_name} assigns no genotypes")

        wars = [type("W", (), {"genotype": g})() for g in assigned]
        extended = extend_plot_order_from_attr(wars, "genotype", constants.DF_SORT_ORDER["genotype"])
        assert not set(assigned) - set(extended), config_name


class TestDetermineBaselineKey:
    """The EP difference-heatmap baseline is what every difference is measured against."""

    def test_per_sex_mapping(self, ep_heatmaps):
        config = {"analysis": {"ep_heatmaps": {"baseline_genotype": {"Male": "x/y", "Female": "x/x"}}}}
        assert ep_heatmaps.determine_baseline_key(["+/y", "x/y"], "Male", config) == "x/y"
        assert ep_heatmaps.determine_baseline_key(["+/x", "x/x"], "Female", config) == "x/x"

    def test_scalar_applies_to_every_sex(self, ep_heatmaps):
        config = {"analysis": {"ep_heatmaps": {"baseline_genotype": "WT"}}}
        assert ep_heatmaps.determine_baseline_key(["WT", "KO"], "Male", config) == "WT"

    def test_configured_but_absent_raises(self, ep_heatmaps):
        """The defect this replaces: falling through to the first observed genotype.

        For a vocabulary that sorts the mutant first, that silently made the mutant the
        reference and inverted every map, with no error and a warning that could not fire.
        """
        config = {"analysis": {"ep_heatmaps": {"baseline_genotype": {"Male": "x/y"}}}}
        with pytest.raises(ValueError, match="not among the genotypes present"):
            ep_heatmaps.determine_baseline_key(["+/y", "+/+"], "Male", config)

    def test_falls_back_to_sort_order_when_unconfigured(self, ep_heatmaps):
        constants.DF_SORT_ORDER["genotype"] = ["WT", "KO"]
        assert ep_heatmaps.determine_baseline_key(["KO", "WT"], "Male", None) == "WT"

    def test_empty_bucket_is_none(self, ep_heatmaps):
        assert ep_heatmaps.determine_baseline_key([], "Male", None) is None

    def test_unconfigured_sex_falls_through_rather_than_raising(self, ep_heatmaps):
        """A mapping that omits a sex is a partial config, not an error."""
        constants.DF_SORT_ORDER["genotype"] = ["WT", "KO"]
        config = {"analysis": {"ep_heatmaps": {"baseline_genotype": {"Male": "WT"}}}}
        assert ep_heatmaps.determine_baseline_key(["WT", "KO"], "Female", config) == "WT"


class TestGenotypeColorScale:
    """Colours must be distinct per genotype, at any number of genotypes."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
    def test_every_genotype_gets_a_distinct_colour(self, n):
        genotypes = [f"g{i}" for i in range(n)]
        scale = create_genotype_color_scale(pd.DataFrame({"genotype": genotypes}))
        assert len(scale.values) == n
        assert len(set(scale.values)) == n, f"colour collision at {n} genotypes"
        assert list(scale.order) == genotypes

    def test_five_genotypes_do_not_collide(self):
        """The measured defect: a padded 6-entry cycle of 3 colours gave 5 genotypes 3."""
        df = pd.DataFrame({"genotype": ["+/y", "x/y", "+/x", "+/+", "x/x"]})
        scale = create_genotype_color_scale(df)
        assert len(set(scale.values)) == 5

    def test_plot_order_is_respected(self):
        df = pd.DataFrame({"genotype": ["KO", "WT"]})
        scale = create_genotype_color_scale(df, ["WT", "KO"])
        assert list(scale.order) == ["WT", "KO"]

    def test_values_outside_plot_order_are_kept(self):
        """seaborn-objects drops rows whose level is absent from ``order``."""
        df = pd.DataFrame({"genotype": ["WT", "KO", "Unknown"]})
        scale = create_genotype_color_scale(df, ["WT", "KO"])
        assert set(scale.order) == {"WT", "KO", "Unknown"}
