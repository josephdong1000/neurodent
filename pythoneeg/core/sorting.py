# Standard library imports
import os
import tempfile
from pathlib import Path

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import probeinterface as pi
from mountainsort5 import Scheme2SortingParameters, sorting_scheme2
from mountainsort5.util import create_cached_recording

# Local imports
from .core import _HiddenPrints


class MountainSortOrganizer:
    def __init__(self, recording, plot_probe=False, verbose=False, n_jobs: None | int = None) -> None:
        assert isinstance(recording, si.BaseRecording)
        self.recording = recording
        # self.notch_filter = 60 # Hz
        self.n_channels = recording.get_num_channels()
        self.verbose = verbose
        if n_jobs is not None:
            si.set_global_job_kwargs(n_jobs=n_jobs) # n_jobs = -1 => use all cores (may be slower)

        # Create a dummy probe
        linprobe = pi.generate_linear_probe(self.n_channels, ypitch=40)
        linprobe.set_device_channel_indices(self.recording.get_channel_ids())
        linprobe.set_contact_ids(self.recording.get_channel_ids())
        self.recording = self.recording.set_probe(linprobe)

        # Visualize
        if plot_probe:
            _, ax2 = plt.subplots(1, 1)
            plot_probe(linprobe, ax=ax2, with_device_index=True, with_contact_id=True)
            plt.show()

    def preprocess_recording(self, freq_min=100):
        rec_prep = spre.common_reference(self.recording)
        rec_prep = spre.scale(rec_prep, gain=10) # Scaling for whitening to work properly
        rec_prep = spre.whiten(rec_prep)
        rec_prep = spre.highpass_filter(rec_prep, freq_min=freq_min, ftype='bessel')
        # rec_prep = spre.bandpass_filter(rec_prep, freq_min=freq_min, freq_max=freq_max, ftype='bessel')
        # rec_preps = []
        # for i in range(self.n_channels):
        #     rec_preps.append(rec_prep.remove_channels(np.delete(np.arange(self.n_channels), i)))
        # rec_prep = rec_prep.remove_channels(np.arange(1, 8)) # Experimental, remove all channels except one
        # self.prep_recordings = rec_preps
        self.prep_recording = rec_prep

    def extract_spikes(self, snippet_T=0.1):
        snippet_samples = round(self.recording.sampling_frequency * snippet_T)
        temp_dir = Path(tempfile.gettempdir()) / os.urandom(24).hex()
        os.makedirs(temp_dir)
        sort_params = Scheme2SortingParameters(
            phase1_detect_channel_radius=1,
            detect_channel_radius=1,
            snippet_T1=snippet_samples,
            snippet_T2=snippet_samples,
            )

        recording_cached = create_cached_recording(self.prep_recording, folder=temp_dir)

        with _HiddenPrints(silence=not self.verbose):
            # Sort over all channels
            self.sorting = sorting_scheme2(
                recording=recording_cached,
                sorting_parameters=sort_params
            )
            # Sort on individual channels
            self.sortings = []
            for i in range(self.n_channels):
                sorting = sorting_scheme2(
                            recording=recording_cached.remove_channels(np.delete(np.arange(self.n_channels), i)),
                            sorting_parameters=sort_params
                        )
                self.sortings.append(sorting)

    def preprocess_final_recording(self, notch_freq=60):
        rec_prep = spre.notch_filter(self.recording, freq=notch_freq) # Get rid of mains hum
        # rec_prep = spre.highpass_filter(rec_prep, freq_min=60, ftype='bessel')

        rec_preps = []
        for i in range(self.n_channels):
            rec_preps.append(rec_prep.remove_channels(np.delete(np.arange(self.n_channels), i)))
        self.prep_final_recording = rec_prep
        self.prep_final_recordings = rec_preps

    def get_final_analyzer(self, folder=None):
        if folder is None:
            folder = Path(tempfile.gettempdir()) / os.urandom(24).hex()
            os.makedirs(folder)
        # self.get_final_sorting()
        sorting_analyzers = []
        for i,e in enumerate(self.sortings):
            sorting_analyzers.append(si.create_sorting_analyzer(e, self.prep_final_recordings[i],
                                                                # folder=folder,
                                                                sparse=False,
                                                                overwrite=True))
        sorting_analyzer = si.create_sorting_analyzer(self.sorting, self.prep_final_recording,
                                                    #   folder=folder, 
                                                      sparse=False,
                                                      overwrite=True)

        self.sorting_analyzer = sorting_analyzer
        self.sorting_analyzers = sorting_analyzers
        return sorting_analyzer, sorting_analyzers