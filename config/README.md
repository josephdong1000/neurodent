# Configuration

This document describes the configuration parameters used by the NeuRodent Snakemake pipeline. Configuration is specified in `config/config.yaml`. A local override file `config/config.local.yaml` is also supported and, if present, will be loaded after the main config to override specific values.

## General usage

To run the pipeline, execute the following from the repository root:

```bash
snakemake --configfile config/config.yaml --cores <N>
```

Snakemake will automatically detect the main workflow definition at `workflow/Snakefile`.

To use conda for software deployment:

```bash
snakemake --configfile config/config.yaml --cores <N> --sdm conda
```

For cluster (SLURM) execution, see the [Snakemake cluster execution documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html).

### Dataset selection

The pipeline supports multiple datasets. To select a dataset, either:

- Set the `NEURODENT_DATASET` environment variable:
  ```bash
  NEURODENT_DATASET=mini_real snakemake --cores <N>
  ```
- Or change the `active_dataset` value in `config/config.yaml`.

Dataset-specific configurations are stored in `config/datasets/{dataset_name}.yaml`. These override any matching keys in the main `config/config.yaml`.

Available datasets can be found in the `config/datasets/` directory.

### Local configuration overrides

To customize settings without modifying `config/config.yaml`, create a `config/config.local.yaml` file. Any values in the local config will override the defaults. This file is not tracked by version control and is intended for site-specific settings (e.g., `temp_directory` paths).

---

## Configuration parameters

### `active_dataset`

- **Type:** string
- **Required:** no
- **Default:** `"sox5_bin"`
- **Description:** Name of the active dataset. Can be overridden by the `NEURODENT_DATASET` environment variable. The corresponding dataset config file must exist at `config/datasets/{active_dataset}.yaml`.

### `temp_directory`

- **Type:** string (path)
- **Required:** yes
- **Description:** Path to a temporary/scratch directory used for intermediate pipeline files.

### `samples`

#### `samples.samples_file`

- **Type:** string (path)
- **Required:** yes (typically set by dataset config)
- **Description:** Path to a JSON file containing sample metadata (data folder paths, animal IDs, joint session definitions, etc.). This is usually set in the dataset-specific config file (e.g., `config/datasets/sox5_bin.yaml`).

#### `samples.truncate_animals`

- **Type:** integer or null
- **Required:** no
- **Default:** `null`
- **Description:** Limit processing to the first N animals. Useful for testing. Set to `null` to process all animals.

#### `samples.quality_filter.exclude_unknown_genotypes`

- **Type:** boolean
- **Required:** no
- **Default:** `true`
- **Description:** Whether to exclude animals with unknown genotypes during quality filtering.

#### `samples.quality_filter.exclude_bad_animaldays`

- **Type:** boolean
- **Required:** no
- **Default:** `true`
- **Description:** Whether to exclude animal-days marked as "bad" during quality filtering.

### `analysis`

#### `analysis.war_generation`

Parameters for Windowed Analysis Result (WAR) generation. Note that `mode`, `file_pattern`, and some `lro_kwargs` are typically set by the active dataset config (see `config/datasets/`).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `mode` | string | no | dataset-specific | WAR generation mode (set by dataset config). |
| `file_pattern` | string | no | dataset-specific | Glob pattern for input data files (set by dataset config). |
| `day_sep` | string or null | no | `null` | Day separator for multi-day recordings. |
| `assume_from_number` | boolean | no | `true` | Whether to infer session ordering from file numbering. |
| `skip_sessions` | list of strings | no | `["*bad*"]` | Session patterns to skip during processing (glob-style matching). |
| `day_parse_kwargs.date_patterns` | list of [pattern, format] | no | see config | Regex patterns and date format strings for parsing dates from filenames. |
| `lro_kwargs.mode` | string | no | dataset-specific | LongRecordingOrganizer mode (set by dataset config). |
| `lro_kwargs.multiprocess_mode` | string | no | `"dask"` | Parallelization strategy (`"dask"` or `"serial"`). |
| `lro_kwargs.overwrite_rowbins` | boolean | no | `false` | Whether to overwrite existing row bins. |

#### `analysis.frequency_domain_spike_detection`

Parameters for frequency-domain spike detection (FDSAR).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `default_params.bp` | list of floats | no | `[3.0, 40.0]` | Bandpass filter range in Hz [low, high]. |
| `default_params.notch` | float | no | `60.0` | Notch filter frequency in Hz. |
| `default_params.notch_q` | float | no | `30.0` | Notch filter quality factor. |
| `default_params.freq_slices` | list of floats | no | `[10.0, 20.0]` | STFT energy slice frequencies in Hz. |
| `default_params.sneo_percentile` | float | no | `99.99` | SNEO threshold percentile (higher = fewer detections). |
| `default_params.cluster_gap_ms` | float | no | `80.0` | Minimum gap between detected spikes in ms. |
| `default_params.search_ms` | float | no | `160.0` | Spike refinement search window in ms. |
| `default_params.baseline_ms` | float | no | `500.0` | Baseline analysis window in ms. |
| `default_params.k_sigma` | float | no | `3.0` | Statistical significance threshold (sigma). |
| `default_params.smooth_window` | integer | no | `7` | Smoothing kernel size for spike refinement. |
| `default_params.vote_k` | integer | no | `2` | Minimum votes required across frequency bands. |
| `default_params.smooth_len` | integer | no | `5` | SNEO smoothing window length. |
| `multiprocess_mode` | string | no | `"dask"` | Parallelization strategy (`"dask"` or `"serial"`). |

##### `analysis.frequency_domain_spike_detection.spike_averaged_traces`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `tmin` | float | no | `-0.5` | Epoch start time in seconds relative to spike. |
| `tmax` | float | no | `0.5` | Epoch end time in seconds relative to spike. |
| `baseline` | list or null | no | `null` | Baseline correction period [tmin, tmax], or null for no correction. |
| `save_epochs` | boolean | no | `true` | Whether to save epoch data (.fif files). |
| `figure_format` | string | no | `"png"` | Output format for spike-averaged trace plots. |
| `dpi` | integer | no | `300` | Figure resolution in dots per inch. |

#### `analysis.standardization`

Parameters for WAR channel standardization (applied before filtering).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `channel_reorder` | list of strings | no | see config | Ordered list of channel abbreviations to standardize to. |
| `use_abbrevs` | boolean | no | `true` | Whether to use abbreviated channel names. |
| `add_unique_hash` | boolean | no | `false` | Whether to add a unique hash during standardization. |
| `unique_hash_length` | integer | no | `4` | Length of the unique hash if enabled. |

#### `analysis.fragment_filter_config`

Fragment (temporal artifact) filtering configuration. Each sub-key defines a filter step.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `logrms_range.z_range` | float | no | `3` | Z-score range for log-RMS based filtering. |
| `high_rms.max_rms` | float | no | `500` | Maximum RMS threshold; windows above this are rejected. |
| `low_rms.min_rms` | float | no | `50` | Minimum RMS threshold; windows below this are rejected. |
| `high_beta.max_beta_prop` | float | no | `0.4` | Maximum beta-band proportion; windows above are rejected. |

#### `analysis.channel_filter_config`

Channel (spatial artifact) filtering configuration.

##### Manual filtering (`channel_filter_config.manual`)

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `reject_channels_by_session` | boolean | no | `false` | Whether to use per-session bad channel lists. |
| `reject_channels` | list of strings | no | `["LHip", "RHip"]` | Channels to reject globally. |
| `min_valid_channels` | integer | no | `3` | Minimum number of valid channels required. |

##### LOF filtering (`channel_filter_config.lof`)

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `reject_lof_threshold` | float | no | `2.5` | LOF score threshold above which channels are considered bad. |
| `reject_channels` | list of strings | no | `[]` | Channels to reject globally (before LOF). |
| `min_valid_channels` | integer | no | `3` | Minimum number of valid channels required. |

#### `analysis.aggregation`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `groupby` | list of strings | no | `["animalday", "isday"]` | Grouping variables for time-window aggregation. |

#### `analysis.zeitgeber`

Zeitgeber (circadian) time processing parameters.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `features` | list of strings | no | see config | Features to extract for zeitgeber analysis. |
| `time_aggregation_minutes` | integer | no | `60` | Time window for aggregation in minutes (60 = hourly). |

#### `analysis.zeitgeber_plots`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `features` | list of strings | no | see config | Features to plot in temporal analysis. |
| `baseline_hours` | integer | no | `12` | Hours to use for baseline correction. |
| `exclude_from_baseline` | list of strings | no | `[]` | Features to exclude from baseline correction. |
| `figure_format` | string | no | `"png"` | Figure output format (png, tif, pdf, svg). |
| `data_format` | string | no | `"csv"` | Data export format (csv, pkl). |
| `dpi` | integer | no | `300` | Figure resolution. |
| `figsize` | list of integers | no | `[10, 20]` | Figure size [width, height]. |

#### `analysis.relfreq_plots`

Relative frequency distribution plot parameters.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `features` | list of strings | no | see config | Features to include in relative frequency plots. |
| `figure_format` | string | no | `"png"` | Figure output format. |
| `data_format` | string | no | `"csv"` | Data export format. |
| `dpi` | integer | no | `300` | Figure resolution. |

#### `analysis.ep_figures`

ExperimentPlotter statistical figure parameters.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `features` | list of strings | no | see config | Features to plot. |
| `exclude_features` | list or null | no | `null` | Features to exclude. |
| `figure_format` | string | no | `"png"` | Figure output format. |
| `data_format` | string | no | `"csv"` | Data export format. |
| `dpi` | integer | no | `300` | Figure resolution. |

#### `analysis.ep_heatmaps`

ExperimentPlotter heatmap parameters.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `matrix_features` | list of strings | no | see config | Matrix features to plot (coherence, correlation, etc.). |
| `baseline_type` | string | no | `"sex_specific"` | Baseline type: `"sex_specific"` or `"global"`. |
| `figure_format` | string | no | `"png"` | Figure output format. |
| `data_format` | string | no | `"pkl"` | Data export format. |
| `dpi` | integer | no | `300` | Figure resolution. |

#### `analysis.lof_evaluation`

LOF (Local Outlier Factor) accuracy evaluation parameters.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `threshold_range.min` | float | no | `0.5` | Minimum LOF threshold to test. |
| `threshold_range.max` | float | no | `4.0` | Maximum LOF threshold to test. |
| `threshold_range.step` | float | no | `0.05` | Step size for threshold testing. |
| `evaluation_channels` | list of strings | no | see config | Channels to evaluate. |
| `figure_format` | string | no | `"png"` | Figure output format. |
| `data_format` | string | no | `"csv"` | Data export format. |
| `dpi` | integer | no | `300` | Figure resolution. |

#### `analysis.filtering_comparison`

Parameters for comparing manual vs. LOF channel filtering.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `features_to_compare` | list of strings | no | see config | Features to compare between filtering methods. |
| `plot_types` | list of strings | no | see config | Types of comparison plots to generate. |

#### `analysis.figures`

Per-animal diagnostic figure parameters.

##### `figures.coherecorr_spectral`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `figsize` | list of integers | no | `[20, 5]` | Figure size [width, height]. |
| `score_type` | string | no | `"z"` | Score type for coherence/correlation plots. |

##### `figures.psd_histogram`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `figsize` | list of integers | no | `[10, 4]` | Figure size. |
| `avg_channels` | boolean | no | `true` | Whether to average across channels. |
| `plot_type` | string | no | `"loglog"` | Plot type for PSD histograms. |

##### `figures.psd_spectrogram`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `figsize` | list of integers | no | `[20, 4]` | Figure size. |
| `mode` | string | no | `"none"` | Spectrogram mode. |

##### `figures.temporal_heatmaps`

Configurable per-feature temporal heatmap parameters. Each feature key (e.g. `rms`, `psdslope`, `zpcorr`) accepts:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `figsize` | list of integers | no | varies | Figure size. |
| `cmap` | string | no | varies | Matplotlib colormap name. |
| `norm_type` | string | no | varies | Normalization type (`"fixed"` or `"centered"`). |
| `norm_params` | object | no | varies | Normalization parameters (`vmin`/`vmax` for fixed, `halfrange` for centered). |

### `cluster`

SLURM cluster resource configuration. Each rule name maps to resource limits.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `time` | string | no | varies | Maximum wall-clock time (e.g. `"6h"`, `"2h"`). |
| `mem_mb` | integer | no | varies | Memory allocation in MB. |
| `nodes` | integer | no | `1` | Number of nodes. |
| `threads` | integer | no | varies | Number of threads/CPUs. |
| `interface` | string or null | no | `null` | Network interface. |
| `retries` | integer | no | varies | Number of retries on failure. |

Rules with cluster configuration: `war_generation`, `split_joint_recordings`, `frequency_domain_spike_detection`, `spike_averaged_traces`, `war_quality_filter`, `diagnostic_figures`, `war_fragment_filter`, `war_flattening`, `war_zeitgeber`, `relfreq_plots`, `ep_figures`, `ep_heatmaps`, `zeitgeber_plots`, `lof_evaluation`, `notebook`, `war_standardize`, `war_channel_filtering`, `filtering_comparison`.

### `logging`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `level` | string | no | `"DEBUG"` | Logging level (DEBUG, INFO, WARNING, ERROR). |
| `format` | string | no | see config | Python logging format string. |
