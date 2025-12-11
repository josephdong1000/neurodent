import spikeinterface.extractors as se
from neurodent import core, visualization
from datetime import datetime
from pathlib import Path

core.set_temp_directory('/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Yong/temp')

data_path = Path('/mnt/isilon/marsh_single_unit/MarshMountainSort/rhds/PortA-AP3B2homo-228-M-PortB-AP3B2wt-232-M-standardEEG 11-7-25_251107_123158')
#data_path = Path('/mnt/isilon/marsh_single_unit/MarshMountainSort/rhds')


# Create LongRecordingOrganizer
# mode options: 'bin', 'si' (SpikeInterface), 'mne', etc.
recording = core.LongRecordingOrganizer(
    base_folder_path=data_path,
    mode="si",  # Change based on your data format
    manual_datetimes=datetime(2015, 1, 16, 9, 51, 52),
    extract_func=se.read_intan,
    stream_id='0',
    input_type='files',
    file_pattern='*.rhd'
)


# Access recordings
record_out = recording.LongRecording

# Get basic properties
print(f"Sampling frequency: {record_out.get_sampling_frequency()} Hz")
print(f"Number of channels: {record_out.get_num_channels()}")
print(f"Channel IDs: {record_out.get_channel_ids()}")
print(f"Duration: {record_out.get_num_frames() / record_out.get_sampling_frequency()} seconds")


# Get channel metadata
channel_locations = record_out.get_channel_locations()
print(f"Channel locations: {channel_locations}")

'''
# After using se.read_intan(), the data type is unsigned, and spikeinterface is not able to process it further.
# we need to convert it to signed using unsigned_to_signed() in spikeinterface.preprocessing
# Maybe we need to add this in _apply_resampling??

dtype = recording.get_dtype()
if dtype == 'uint16':
     recording = spre.unsigned_to_signed(recording)

'''

# data_parent_folder = Path('/mnt/isilon/marsh_single_unit/MarshMountainSort')
# data_folder = 'rhds'
# animal_id = '501-502'

# ao = visualization.AnimalOrganizer(
#                 data_parent_folder / data_folder,
#                 animal_id,
#                 mode="nest",
#                 assume_from_number=True,
#                 skip_days=["bad"],
#                 lro_kwargs={"mode": "bin", "multiprocess_mode": "dask", "overwrite_rowbins": True},
#             )
