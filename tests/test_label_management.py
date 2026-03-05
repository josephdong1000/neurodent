import logging
import warnings
import pytest
import numpy as np
from neurodent.core import LongRecordingOrganizer

def test_lro_label_merging_success():
    """Test that labels are correctly merged when there are no conflicts."""
    lro1 = LongRecordingOrganizer(item=None, recording=None)
    lro1.labels = {"animal": "A1", "genotype": "WT"}
    lro2 = LongRecordingOrganizer(item=None, recording=None)
    lro2.labels = {"day": "Jan-01-2023"}
    
    # Mocking necessary attributes for merge to work
    lro1.LongRecording = type('MockRecording', (), {'get_num_channels': lambda: 1, 'get_sampling_frequency': lambda: 1000})()
    lro2.LongRecording = lro1.LongRecording
    lro1.meta = type('MockMeta', (), {'n_channels': 1, 'dt_end': None})()
    lro2.meta = lro1.meta
    
    # Act
    lro1._update_metadata_after_merge(lro2)
    
    # Assert
    assert lro1.labels == {"animal": "A1", "genotype": "WT", "day": "Jan-01-2023"}

def test_lro_label_merging_conflict_warning():
    """Test that a warning is issued when labels conflict during merge."""
    lro1 = LongRecordingOrganizer(item=None, recording=None)
    lro1.labels = {"animal": "A1", "genotype": "WT"}
    lro2 = LongRecordingOrganizer(item=None, recording=None)
    lro2.labels = {"animal": "A1", "genotype": "KO"}
    
    # Mocking necessary attributes
    lro1.meta = type('MockMeta', (), {'n_channels': 1, 'dt_end': None})()
    lro2.meta = lro1.meta
    
    # Act & Assert
    with pytest.warns(UserWarning, match="Label conflict during merge for key 'genotype'"):
        lro1._update_metadata_after_merge(lro2)
    
    # Assert overwrite behavior (python dict update behavior preserved but warned)
    assert lro1.labels["genotype"] == "KO"

def test_lro_labels_inheritance_on_split():
    """Test that labels are inherited when an LRO is split."""
    parent_labels = {"animal": "A1", "genotype": "WT"}
    lro = LongRecordingOrganizer(item=None, recording=None)
    lro.labels = parent_labels
    
    # Mocking necessary attributes for split
    mock_rec = type('MockRecording', (), {
        'get_channel_ids': lambda self: ["Ch1", "Ch2"],
        'get_num_channels': lambda self: 2,
        'get_sampling_frequency': lambda self: 1000,
        'get_total_duration': lambda self: 10.0,
        'get_dtype': lambda self: np.float32,
        'select_channels': lambda self, channel_ids: self,
        'rename_channels': lambda self, new_channel_ids: self
    })()
    lro.LongRecording = mock_rec
    lro.channel_names = ["Ch1", "Ch2"]
    lro.meta = type('MockMeta', (), {'n_channels': 2, 'f_s': 1000})()
    
    # Act
    splits = lro.split({"Group1": ["Ch1"]})
    
    # Assert
    assert "Group1" in splits
    assert splits["Group1"].labels == parent_labels
    assert splits["Group1"].labels is not lro.labels  # Should be a copy
