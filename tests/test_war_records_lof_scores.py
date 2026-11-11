"""A generated WAR records LOF scores, and its bad_channels_dict starts empty.

WAR generation applied the LOF threshold and wrote its picks into every WAR's
bad_channels_dict. The channel filters masked that stored dict by default, and the LOF
evaluation fell back to it as ground truth, so LOF's picks were scored against themselves.
"""

import datetime

import numpy as np
import pytest

si = pytest.importorskip("spikeinterface.core")

from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.loading import AnimalOrganizer, LongRecordingOrganizer  # noqa: E402

FS = 1000
CHANNELS = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud"]
START = datetime.datetime(2020, 1, 1, 8, 0, 0)


def _lro(n_channels, day, noisy_channel, seed):
    """Correlated channels with one much noisier than the rest."""
    rng = np.random.default_rng(seed)
    n_samples = 30 * FS
    shared = rng.standard_normal(n_samples) * 20
    data = shared[:, None] + rng.standard_normal((n_samples, n_channels)) * 2
    data[:, noisy_channel] = rng.standard_normal(n_samples) * 400
    rec = si.NumpyRecording(data.astype(np.float32), sampling_frequency=float(FS))
    lro = LongRecordingOrganizer(item=None, recording=rec.rename_channels(new_channel_ids=CHANNELS[:n_channels]))
    lro.file_durations = [n_samples / FS]
    lro.cumulative_file_durations = [n_samples / FS]
    lro.manual_datetimes = START + datetime.timedelta(days=day)
    lro.datetimes_are_start = True
    lro.finalize_file_timestamps()
    return lro


@pytest.mark.parametrize("n_channels, n_days", [(4, 1), (6, 3)])
# LOF scores are about 1 here, so a threshold of 0.5 picks every channel: picks that used to
# land in the WAR's bad_channels_dict
@pytest.mark.parametrize("threshold", [None, 0.5])
def test_the_war_starts_with_no_bad_channels_and_keeps_the_scores(n_channels, n_days, threshold):
    lros = [_lro(n_channels, day, noisy_channel=day % n_channels, seed=day) for day in range(n_days)]
    ao = AnimalOrganizer.from_lros(lros, animal_id="T", genotype="WT")
    az = AnimalAnalyzer(ao)

    az.compute_bad_channels(lof_threshold=threshold)
    war = az.compute_windowed_analysis(["rms"], window_s=5, multiprocess_mode="serial")

    assert set(war.bad_channels_dict) == set(ao.animaldays)
    assert all(channels == [] for channels in war.bad_channels_dict.values())
    assert set(war.lof_scores_dict) == set(ao.animaldays)
    for animalday, entry in war.lof_scores_dict.items():
        assert len(entry["lof_scores"]) == n_channels
        assert entry["channel_names"] == CHANNELS[:n_channels]
    if threshold is not None:
        # The picks stay on the analyzer, and the WAR gives the same ones from its scores
        assert az.bad_channels_dict == war.get_bad_channels_by_lof_threshold(threshold)
        assert any(az.bad_channels_dict.values())
