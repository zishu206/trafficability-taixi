# Borehole supporting proxy validation

This folder contains the deterministic three-well borehole proxy check used in the English manuscript. The evidence supports near-surface lithology, grain-size and geophysical consistency checks. It does not provide CI, CBR, plate-bearing, vehicle-load or observed pass/fail validation.

## Files

- `borehole_proxy_validation_summary.csv`: per-well coverage, missing intervals and existing reference labels.
- `shallow_lithology_10m.csv`: official depth-resolved lithology labels from 0 to 10 m.
- `log_lithology_medians.csv`: median SP, 16-inch/64-inch resistivity and Gamma values by lithology; missing values remain missing.
- `three_well_existing_reference_labels.csv`: AHP and corrected-mechanistic labels from the existing 888-record compilation.
- `dongyuan_model_check.json`: model-class and probability check at the Dongyuan validation tile.
- `borehole_proxy_validation.json`: machine-readable complete summary.

The three well locations were already included in the project compilation. The added value is depth-resolved evidence, not a new independent validation set.
