# Manuscript-protocol validation

## Scope and environment

Validation was completed on **2026-10-08** using the six prepared reference
Zarr datasets, one NVIDIA L20 GPU, Python 3.11, PyTorch 2.10.0+cu128, and
PyTorch Lightning 2.6.4. The test copy was isolated from the original
experiment workspace.

The regression suite passed **43 tests with no skips**. All **12 configurations**
completed real data loading, training, full validation, best-validation-state
restoration, and three-day full-grid prediction with exit code 0.

| Backbone | No EWC | Standard EWC | Seasonal EWC | Batch size | Training patch |
|----------|--------|--------------|--------------|------------|----------------|
| U-Net | Passed | Passed | Passed | 2 | 768 × 1280 |
| ConvLSTM | Passed | Passed | Passed | 3 | 384 × 384 |
| IAM4VP | Passed | Passed | Passed | 3 | 384 × 384 |
| Earthformer | Passed | Passed | Passed | 1 | 384 × 384 |

The configured architectures, spatial strides, patches per sample, batch sizes,
and bfloat16 precision were retained. The bounded smoke overrides were:

- one rolling window, forecasting 2019-01-01 through 2019-01-03;
- two epochs with one training batch per epoch, and the complete validation set;
- one offline training step and one Fisher batch per EWC condition;
- zero data-loader workers, two reading/compute threads, and no preloading or
  shared-memory caching.

All EWC banks were built afresh. Two normalizers, one for each normalization
mode, were fitted on the real 2017–2018 data and shared by runs with the same
mode. The tested source and configuration files were independently compared
with the publication tree using a **96-file SHA-256 manifest**, with no missing
files or mismatches.

## Verified protocol

For the first forecast:

| Component | Dates | Calendar days |
|-----------|-------|---------------|
| Historical target window | 2018-11-02–2018-12-31 | 60 |
| Training targets | 2018-11-02–2018-12-24 | 53 |
| Validation targets | 2018-12-25–2018-12-31 | 7 |
| Historical predictor sequence | 2018-12-27–2018-12-31 | 5 |
| GFS valid dates and forecast targets | 2019-01-01–2019-01-03 | 3 |

All runs recorded **51 training samples, 5 validation samples, and zero
overlapping target dates**, before spatial patch expansion. The 60-day target
pool is distinct from the five-step input sequence. Three-day label windows
cannot straddle the training/validation boundary.

All runs used **56 input channels**: 41 historical channels followed by 15
forecast-date GFS channels. Tests checked the lead-major channel order,
agreement between serial/threaded/preloaded access, exclusion of future ERA5
and LGHAP from the input, and rejection of incompatible normalization schemas.

Normalization metadata recorded 2017-01-01–2018-12-31, 56 feature names, and
three GFS forecast leads. Runs sharing a mode had identical fingerprints:

- min-max: `cc5793dd3073281a8a89d883ea1f88acd5f619c490dab8c173c9afee86d8dad2`
- z-score: `b65d87da0f40982556424390c3c070439bb6da097170b4bc4a707473b3dbca24`

Best-validation-state selection was checked with a real Lightning test whose
second epoch was deliberately worse than its first. In the real-data smoke,
all three Earthformer runs restored epoch 0; the other nine runs restored
epoch 1. Every run produced three finite daily MAE/RMSE records, 377,905 valid
pixels per date, and a 5,898,240-byte int16 prediction file.

No CUDA out-of-memory errors occurred. The largest five-second sampled GPU
memory usage was 42,735 MiB, in the IAM4VP No-EWC run. This is a sampled value,
not an instrumented absolute peak or a minimum hardware specification.

## Validation boundary

This establishes bounded runtime correctness, not numerical reproduction of
the published tables. Full-year, 20-epoch experiments were not rerun. The
first-window smoke does not trigger the online EWC update scheduled every
seven windows. The 123-cycle calendar and complete, single-contribution 2019
date coverage were checked by regression tests, not by a full-year run.

The GFS store exposes valid dates but lacks forecast-initialization metadata.
Future-channel alignment was tested; issuance-time availability cannot be
established from this store. Data preparation must supply forecasts available
at the corresponding prediction cutoff and retain their provenance.
