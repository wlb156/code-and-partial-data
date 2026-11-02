# Code and Partial Data for Soil Organic Carbon Estimation

This repository contains Python scripts illustrating the spectral band selection, band ranking, and stacking methods used for soil organic carbon (SOC) estimation, together with a subset of the study data.

## Repository contents

| File | Description |
| --- | --- |
| `RFE.py` | Uses XGBoost-based recursive feature elimination (RFE) to select spectral bands. |
| `TOPSIS.py` | Evaluates candidate bands using five criteria, entropy weights, and TOPSIS ranking. |
| `stacking.py` | Combines 1D-CNN, XGBoost, SVR, and PLSR base models with a random forest meta-model, and reports prediction metrics. |
| `partial_data.xlsx` | Contains a subset of the soil organic carbon and spectral data used in the study. |

## Software requirements

The scripts use Python with `numpy`, `pandas`, `scikit-learn`, `xgboost`, `tensorflow`, and `openpyxl`.

## Data scope

`partial_data.xlsx` contains partial study data rather than the complete training and test datasets.
