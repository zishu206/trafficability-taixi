# Pixel-Level Vehicle Trafficability on the Taixi Plain

Code accompanying the manuscript *Pixel-Level Vehicle Trafficability on the Taixi Plain
Using Multi-Channel Remote-Sensing Indicators and CNN–Transformer Dual Models*.

The project performs pixel-level semantic segmentation of thirteen remote-sensing channels
over the western coastal plain (Taixi) to grade vehicle trafficability into three classes.
Two network families are implemented from scratch in PyTorch — a U-net (CNN) and a
SegFormer-B0 (Transformer) — together with the code that builds an independent mechanistic
label, runs random-forest and SVM baselines, and reproduces the analysis figures and tables.

## What the labels are (and are not)

Two labels are used, and the distinction matters when reading the metrics:

- **AHP weak label.** A three-level grade produced from the analytic hierarchy process over
  terrain, hydrology and land-use indicators. It is a rule-based weak label, so a high score
  measures pixel-level *reproduction* of that label rather than independently established truth.
- **Bekker–Wong mechanistic label.** A rated cone index (RCI) estimated from standalone
  OpenLandMap soil parameters under a terramechanics model and then split with Jenks natural
  breaks. It is an empirical reference, not an in-situ measurement. It shares the OpenLandMap
  product family with two input channels but draws on five bands absent from the network input;
  a linear regression of RCI on the input returns only R² = 0.4366, indicating that it carries
  information not determined by the input.

No in-situ cone, plate-loading or vehicle-bogging measurements have yet been obtained, so
neither label is treated as mechanical ground truth. The least-cost route-planning module is
defined conceptually but not executed and is left as future work.

## Repository layout

```
.
├── core/                       # Deep-learning pipeline
│   ├── config.py               # Paths, channel order, patch and training hyperparameters
│   ├── data_utils.py           # GeoTIFF reading, patching, spatial split, augmentation, Dataset
│   ├── unet.py                 # U-net (encoder–decoder with skip connections)
│   ├── segformer.py            # SegFormer-B0 (MiT stages + all-MLP decoder)
│   ├── metrics.py              # Confusion matrix, OA, Kappa, PA/UA, F1, ROC-AUC
│   ├── train.py                # Generic training entry point
│   ├── gpu_train.py            # Full-scale GPU training entry point
│   ├── common_train.py         # Common-domain training for both labels
│   ├── real_train.py           # Real-data training entry point
│   ├── smoke_test.py           # Synthetic end-to-end check (no GeoTIFF needed)
│   └── requirements.txt
├── analysis/                   # Baselines, diagnostics and figure scripts
├── soil/                       # Mechanistic-label construction and soil-mask checks
├── data/                       # Place input/derived rasters and arrays here (not redistributed)
├── outputs/                    # Weights, confusion matrices, reports and figures are written here
└── export_soil_physical_raw.js # Google Earth Engine export of the raw physical soil bands
```

## Environment

Python 3.10 or newer is recommended.

CPU-only setup:

```
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

With an NVIDIA GPU, install a CUDA-matched build of PyTorch from <https://pytorch.org>, then
install the remaining dependencies. CUDA and automatic mixed precision are enabled without code
changes when a GPU is available.

Verify the installation with the synthetic smoke test, which needs no real data:

```
python core/smoke_test.py
```

## Data

The repository contains code only; the large raster and array files are not redistributed here.
Place them under `data/` (see `data/README.md` for the expected file names). The remote-sensing
indicators come from Google Earth Engine and the soil physical properties from OpenLandMap; both
are public sources, and the GEE export script is included. Restricted third-party
engineering-geological materials are not included and must be requested from the original provider.

All paths are repository-relative. By default the scripts read from `data/` and write to
`outputs/`. The locations can be redirected without editing code through the `TRAFFIC_DATA` and
`TRAFFIC_OUT` environment variables.

## Reproducing the workflow

1. Build the mechanistic label (continuous RCI and three-class raster):
   ```
   python soil/build_mech_label.py
   ```
2. Prepare the common-domain tiles, then train both models on each label:
   ```
   python analysis/prep_common_domain.py
   python core/common_train.py --label both --model both
   ```
   For the full-scale GPU run:
   ```
   python core/gpu_train.py --model both
   ```
3. Train the random-forest and SVM baselines:
   ```
   python analysis/train_rf_svm.py
   ```
4. Run the diagnostics and plotting scripts in `analysis/` to reproduce the reported tables and
   figures (label consistency, linear explainability, spatial maps, convergence and RCI drivers).

Default hyperparameters match the manuscript: 256×256 patches, a 70/30 spatial split, Adam at
1e-4 with weight decay 1e-5, batch size 20 and 50 epochs for the generic entry point. The best
checkpoint is selected by validation macro-F1.

## Outputs

`outputs/` receives model checkpoints (`.pth`), confusion matrices (`.csv`), text reports
(`.txt`) and figures (`.png`). Confusion matrices are written with row and column totals and can
be opened directly in a spreadsheet.

## Citation

A citation entry will be added once the manuscript is assigned its article identifier. The
repository is archived on Zenodo for a stable, citable DOI.

## License

The code is released under the MIT License; see [LICENSE](LICENSE).
