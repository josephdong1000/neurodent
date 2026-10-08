"""Tests for LroMergeMixin.merge_many, the flat replacement for the pairwise merge fold.

Folding N recordings pairwise nests them N-1 deep, because SpikeInterface does not
flatten a concatenation and ``merge`` feeds its own output back in as the first operand.
Sample lookup then walks that nesting, so a full pass is O(N^2), and past roughly a
thousand files ``get_traces`` raises RecursionError against Python's default limit.
Pickling fails far earlier, near N=125, which breaks any recording handed to a worker.

Measured on 1500 real .rhd files: a flat concatenation reads frame 0 fine while the
pairwise fold raises RecursionError, with both reporting the same 180,000,000 frames.
"""

from unittest.mock import MagicMock, patch

import pytest

from neurodent.loading.lro_merge import LroMergeMixin


class _Lro(LroMergeMixin):
    """Minimal stand-in exposing only what merge_many touches."""

    def __init__(self, name, n_samples=100, channel_names=("a", "b")):
        self.item = name
        self.channel_names = list(channel_names)
        self.LongRecording = MagicMock()
        self.LongRecording.get_total_samples.return_value = n_samples
        self.meta = MagicMock(f_s=1000.0, n_channels=len(channel_names))
        self._update_metadata_after_merge = MagicMock()
        self._validate_merge_compatibility = MagicMock()


@pytest.fixture
def base():
    return _Lro("base")


def test_concatenates_once_regardless_of_count(base):
    """N recordings must produce ONE concatenate call, not N-1 nested ones."""
    others = [_Lro(f"o{i}") for i in range(9)]
    with patch("neurodent.loading.lro_merge.si") as mock_si:
        base.merge_many(others)

    assert mock_si.concatenate_recordings.call_count == 1
    passed = mock_si.concatenate_recordings.call_args[0][0]
    assert len(passed) == 10, "base plus all nine others in a single flat call"


def test_validates_every_entry(base):
    """Per-step compatibility checking must survive the flattening."""
    others = [_Lro(f"o{i}") for i in range(4)]
    with patch("neurodent.loading.lro_merge.si"):
        base.merge_many(others)

    assert base._validate_merge_compatibility.call_count == 4


def test_zero_sample_entry_is_excluded_but_its_metadata_is_kept(base):
    """A 0-sample tail contributes no samples yet still carries metadata."""
    live, dead = _Lro("live"), _Lro("dead", n_samples=0)
    with patch("neurodent.loading.lro_merge.si") as mock_si:
        base.merge_many([live, dead])

    passed = mock_si.concatenate_recordings.call_args[0][0]
    assert len(passed) == 2, "the 0-sample recording must not be concatenated"
    assert base._update_metadata_after_merge.call_count == 2, "but its metadata still counts"


def test_renames_channels_when_names_differ(base):
    """Differing raw names are renamed onto the base, as the pairwise merge did."""
    other = _Lro("other", channel_names=("x", "y"))
    with patch("neurodent.loading.lro_merge.si"):
        base.merge_many([other])

    other.LongRecording.rename_channels.assert_called_once_with(
        new_channel_ids=base.channel_names
    )
    assert other.channel_names == base.channel_names


def test_no_concatenation_when_nothing_to_merge(base):
    """An empty list leaves the recording untouched rather than rebuilding it."""
    original = base.LongRecording
    with patch("neurodent.loading.lro_merge.si") as mock_si:
        base.merge_many([])

    mock_si.concatenate_recordings.assert_not_called()
    assert base.LongRecording is original


def test_all_zero_sample_entries_skip_concatenation(base):
    """If every entry is empty there is nothing to concatenate, only metadata to fold."""
    others = [_Lro(f"d{i}", n_samples=0) for i in range(3)]
    with patch("neurodent.loading.lro_merge.si") as mock_si:
        base.merge_many(others)

    mock_si.concatenate_recordings.assert_not_called()
    assert base._update_metadata_after_merge.call_count == 3
