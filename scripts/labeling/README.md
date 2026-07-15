# Artifact-labeling pipeline (issue #208)

Human ground-truth (channel, window) labels, then score detectors against them. Three pieces; see each
function's docstring for arguments.

- `render_context.py` — `render_lro()` renders one figure per window (all channels) → `images/` +
  `manifest.csv` + `geometry.json`. Works for any format the loader reads (rhd/EDF/NWB/bin).
- `build_rater_bundle.py` — packs a rendered set into a self-contained HTML rating zip.
- `neurodent.results.scoring` — `ingest` → `consensus` → `interrater` (Fleiss kappa) → `score_mask`.
  Lives in the library because it scores `FILTER_REGISTRY` detectors and owns the manifest schema.

## Flow

1. **Render** each recording's chosen windows with `render_lro(..., append=True)` to accumulate many
   animals into one bundle.
2. **Bundle:** `uv run python scripts/labeling/build_rater_bundle.py <render_dir>` → `rating_bundle_*.zip`.
3. **Rate:** send the zip. The rater opens `index.html` (offline, no install), types their name, and
   labels — every channel defaults to `clean`, so they touch only exceptions and press Next, then
   Export CSV. All raters label every window (full overlap), each in a different order.
4. **Score** the returned CSVs with `neurodent.results.scoring` (see docstrings).

**Footgun:** `FILTER_REGISTRY` masks are `True = KEEP`; `score_mask` wants `True = REJECT` — use
`score_keep_mask` for a filter's output. Selection is random only (scoring on a detector's output
flatters it); split tune/test by animal.

Tests: `uv run pytest tests/test_scoring.py tests/test_labeling_bundle.py`.

## The rubric (this lives only here)

- `clean` — normal EEG for that channel. Keep. The default.
- `bad` — artifact, not brain: movement, muscle, electrode pop/step, flatline, saturation. NOT plain
  60 Hz hum (already notched); only gross residual mains (120/180 Hz, visible in the PSD panel) counts.
- `event` — real but dramatic brain activity (epileptiform discharge, seizure). KEEP it — an event
  builds/sustains/stops, an artifact usually doesn't. Without this category a seizure has no answer but
  "bad", and the ground truth would call seizures artifacts.
- `unsure` — genuinely can't tell; use it rather than guess.

Scoring maps: reject = `bad`; keep = `clean`/`event` (`event` tagged so over-rejection is measurable);
`unsure` drops out.
