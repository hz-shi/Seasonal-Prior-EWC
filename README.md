# Seasonal-Prior-EWC

Official code for the paper **"Seasonal Prior-Guided Dynamic Forecasting of
Nonstationary Gridded PM2.5 Concentrations in China"** (accepted for publication in *Remote
Sensing*, MDPI; manuscript ID **remotesensing-4551408**).

This repository implements a rolling-retraining forecasting framework for
daily, gridded PM2.5 over China, together with **Seasonal-Prior EWC** — a
condition-aware Elastic Weight Consolidation regularizer that uses seasonal
anchors and an online prior update to mitigate catastrophic forgetting across
rolling windows.

## Highlights

- **Rolling-retraining framework.** The model is retrained on a sliding
  60-day window and predicts the next 3 days from a 5-day input sequence, with
  a 3-day step. The historical target window ends immediately before the first
  forecast day; its first 53 days are used for training and its latest 7 days
  for validation. The lowest validation prediction-loss state is restored
  before forecasting. The schedule retains **123 updating cycles** (input
  origins from 2018-12-27 to 2019-12-28): 122 cycles contribute to the 2019
  evaluation, while the final cycle forecasts dates in 2020.
- **Known-future meteorology.** GFS forecast covariates for all three target
  dates are explicitly included alongside the five historical predictor days.
- **Seasonal-Prior EWC.** A condition-aware EWC penalty built on **seasonal
  anchors** (four seasons: DJF / MAM / JJA / SON) with an **online prior
  update** that refreshes the anchor and importance estimates every 7 windows
  (`αθ = 0.02`, `αΩ = 0.05`). The EWC strength is unified at `λ = 3e-3`.
- **Four backbone models.** U-Net, ConvLSTM, Earthformer, and IAM4VP, all
  driven through a common adapter interface and the same data pipeline.

## Repository structure

```
.
├── configs/                     # Paper configs: 4 models × {no_ewc, standard_ewc, seasonal_ewc}
│   └── pm25_<model>_<variant>.toml
├── src/
│   ├── cli.py                   # Entry point: python -m src.cli --config <toml>
│   ├── data.py                  # Multi-source zarr reader + windowed Dataset/DataModule
│   ├── rolling.py               # Rolling retraining + 2019 prediction loop
│   ├── lightning_module.py      # PyTorch Lightning module (train/val/predict)
│   ├── normalization.py         # Input/target normalization
│   ├── visualize.py             # Result visualization helpers
│   ├── adapters/                # Model adapters (unet, convlstm, earthformer, iam4vp, phydnet)
│   ├── model/
│   │   ├── unet/                # U-Net backbone
│   │   ├── convlstm/            # ConvLSTM backbone
│   │   ├── iam4vp/              # IAM4VP backbone
│   │   ├── phydnet/             # PhyDNet backbone
│   │   └── earthformer/         # Vendored Amazon Earthformer (Apache-2.0)
│   └── reg/
│       ├── condition_aware_ewc.py   # Seasonal-Prior EWC (bank build, penalty, online update)
│       └── conditions.py            # Seasonal / monthly condition mapping
├── requirements.txt
├── tests/                        # Window, model-selection, normalization and GFS regression tests
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## Environment

- Python 3.11
- PyTorch 2.10.0 (CUDA 12.8 build)
- PyTorch Lightning 2.6.4
- CUDA 12.8

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data preparation

The pipeline reads **six zarr datasets**. Place them under `./data/` (or point
the config keys at their actual locations):

| Key         | Dataset | Role |
|-------------|---------|------|
| `year_path` | Static / land-surface variables (e.g. elevation, land cover) | Input |
| `eral_path` | ERA5 / ERA5-Land single-level reanalysis | Historical input |
| `erap_path` | ERA5 pressure-level reanalysis | Historical input |
| `gfs_path`  | GFS forecast fields | Input |
| `meic_path` | MEIC emission inventory | Input |
| `pm25_path` | LGHAP PM2.5 | **Target only** |

Notes on the data layout:

- Time coordinates must decode to calendar dates. Reference daily stores use
  **`"days since 2017-01-01"`**; the annual land-surface store uses
  **`"days since 2018-01-01"`**. Annual/monthly sources are mapped by calendar
  year/month rather than by a shared array index.
- The spatial grid is **768 × 1280** at **0.05°** resolution (`lat` × `lon`).
- The model consumes only the meteorological, emission, and static variables as
  **inputs**; the LGHAP PM2.5 field is used **only as the prediction target**
  (never as an input feature).
- With `future_gfs_enabled = true`, historical channels are followed by
  `gfs/lead1/<variable>`, `gfs/lead2/<variable>` and `gfs/lead3/<variable>`.
  These forecast-date fields are broadcast across the five historical time
  steps as conditional feature planes. The reference stores contain 41
  historical channels and 15 known-future GFS channels, giving 56 input
  channels; counts and names are inferred from the stores at runtime.
- **GFS availability:** supply forecasts issued no later than the prediction
  cutoff, aligned to their target valid dates. The current daily GFS store has
  no forecast-initialization metadata, so the reader can check valid-date
  alignment but cannot verify issuance-time availability. Record initialization
  times and forecast provenance during data preparation.

Data sources and citations:

- **LGHAP** (PM2.5 target) — DOI: [10.5281/zenodo.5652265](https://doi.org/10.5281/zenodo.5652265)
- **ERA5** (reanalysis) — DOI: [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47)
- **CNEMC** (China National Environmental Monitoring Centre, ground observations) — http://www.cnemc.cn/
- **MODIS / VIIRS / SRTM** — please obtain from their respective official
  distribution portals and cite the corresponding product documentation.

## Quick start

1. Install dependencies (see [Environment](#environment)).
2. Place the six zarr datasets under `./data/` (see
   [Data preparation](#data-preparation)).
3. Run a rolling-retraining forecast, e.g. U-Net with Seasonal-Prior EWC:

```bash
python -m src.cli --config configs/pm25_unet_seasonal_ewc.toml
```

Outputs are written under `experiments/<run>/`:

- the 2019 prediction **memmap** (daily gridded PM2.5),
- **per-day metrics** (MAE / RMSE and valid-pixel counts),
- training and rolling **logs**, including `rolling_window_protocols.jsonl`
  with the resolved target-date ranges and train/validation overlap checks.

### Exact first-cycle dates

For the forecast of **2019-01-01 through 2019-01-03**:

| Component | Dates | Length |
|-----------|-------|--------|
| Historical optimization/validation target window | 2018-11-02–2018-12-31 | 60 days |
| Training target dates | 2018-11-02–2018-12-24 | 53 days |
| Validation target dates | 2018-12-25–2018-12-31 | 7 days |
| Historical predictor sequence | 2018-12-27–2018-12-31 | 5 days |
| Known-future GFS valid dates / forecast target dates | 2019-01-01–2019-01-03 | 3 days |

The 60 days define the pool of historical **target dates**, not a 60-step model
input. Each sample contains five predictor days and three target days. Requiring
all three labels to stay within their own split gives 51 training samples and
5 validation samples, before spatial patch expansion. Predictor histories can
precede the first target date; training and validation **label dates** are
strictly disjoint. Model parameters carry over across cycles, while AdamW and
its cosine scheduler are recreated. The cosine period equals `max_epochs`.

All three regularization settings fit normalization statistics on historical
2017–2018 samples and hold them fixed in 2019. Statistics record their fit range
and input-channel schema; incompatible statistics and EWC banks must be rebuilt.

## Configurations

The `configs/` directory contains **12 paper configurations**, covering the four
backbones × three regularization settings:

| Backbone     | No EWC | Standard EWC | Seasonal EWC |
|--------------|--------|--------------|--------------|
| U-Net        | `pm25_unet_no_ewc.toml` | `pm25_unet_standard_ewc.toml` | `pm25_unet_seasonal_ewc.toml` |
| ConvLSTM     | `pm25_convlstm_no_ewc.toml` | `pm25_convlstm_standard_ewc.toml` | `pm25_convlstm_seasonal_ewc.toml` |
| Earthformer  | `pm25_earthformer_no_ewc.toml` | `pm25_earthformer_standard_ewc.toml` | `pm25_earthformer_seasonal_ewc.toml` |
| IAM4VP       | `pm25_iam4vp_no_ewc.toml` | `pm25_iam4vp_standard_ewc.toml` | `pm25_iam4vp_seasonal_ewc.toml` |

Shared EWC settings across the paper runs:

- EWC strength `λ = 3e-3`
- Online prior update every **7 windows**
- `αθ = 0.02`, `αΩ = 0.05`

Standard EWC uses one global anchor (`single`); Seasonal EWC uses four seasonal
anchors (`season4`). The paper configs use two data-loader workers, on-demand
reads, and disabled full-data preloading/SHM warm-up. Auxiliary parameter-drift
analysis is disabled by default so each configuration can run independently.

## Validation

All 12 configurations passed bounded real-data validation on an NVIDIA L20,
and the regression suite passed 43 tests without skips. See
[the validation record](docs/VALIDATION.md) for the exact scope, verified
first-cycle dates, input schema, and remaining validation boundaries.

Run the regression suite in the configured environment:

```bash
python -m unittest discover -s tests -v
```

For a bounded real-data smoke run (two epochs, one training batch per epoch,
one offline training step per condition, and one Fisher batch), use dedicated
output paths:

```bash
python -m src.cli --config configs/pm25_unet_seasonal_ewc.toml \
  --work_dir experiments/smoke/unet_seasonal_ewc \
  --norm_stats_path experiments/smoke/normalizer_stats.npz \
  --ca_ewc_bank_path experiments/smoke/unet_seasonal_ewc/smoke_bank.pt \
  --max_epochs 2 --dry_run_steps 1 --train_first_window_only true \
  --predict_start_date 2019-01-01 --predict_end_date 2019-01-03 \
  --ca_ewc_offline_max_steps_per_condition 1 --ca_ewc_fisher_batches 1 \
  --num_workers 0
```

This exercises data loading, bank construction, training, validation-state
selection and full-grid three-day prediction. Numerical reproduction of the
paper tables requires the full-year runs with the unshortened training and
Fisher settings, together with the matching data preparation and evaluation.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{shan2026seasonalprior,
  title   = {Seasonal Prior-Guided Dynamic Forecasting of Nonstationary
             Gridded PM2.5 Concentrations in China},
  author  = {Shi, Haoze and Wen, Fuling and Zhong, Guiliang and Jiang, Yang and Deng, Liangkun and Shan, Shihan and Yang, Xin and Tang, Hong},
  journal = {Remote Sensing},
  year    = {2026},
  note    = {Manuscript ID remotesensing-4551408 (in press)}
}
```

Repository: https://github.com/hz-shi/Seasonal-Prior-EWC

## License

The project code is released under the **MIT License** (see `LICENSE`).

`src/model/earthformer/` is vendored from **Amazon Science Earthformer** and is
licensed under the **Apache License 2.0**. See `THIRD_PARTY_NOTICES.md` for the
full license text and attribution.

## Acknowledgements

This work was supported by the PowerChina project **DJ-HXGG-2025-02**.
