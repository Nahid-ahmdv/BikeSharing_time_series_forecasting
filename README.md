# Forecasting Washington D.C. Bike-Share Rentals

I wanted to see just how much a fitted forecasting model could add to a simple average of matching hours from the previous four weeks. In this project, I forecast the next **48 hours of recorded rentals** across Capital Bikeshare and examine the size and consistency of improvements across forecast occasions.

The investigation covers target reconstruction, seasonal patterns, chronological model comparisons, weather availability, and prediction intervals. A strong seasonal benchmark gives the more flexible models a meaningful reference to improve upon.

- **Full analysis:** [Executed forecasting notebook](dc_bikeshare_demand_forecasting.ipynb)
- **Towards Data Science article:** Link coming soon.

## The forecasting task

At midnight, each procedure issues all 48 hourly predictions before any actual rental counts from that forecast window become available. Development-selection and final-evaluation origins are two calendar days apart.

I use the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset), which contains hourly and daily Capital Bikeshare rental records for 2011 and 2012. The target is system-wide **recorded rentals**. These records do not measure unmet requests caused by unavailable bikes, full docks, or service disruption.

Some hourly timestamps have no supplied rows. All 670 development days reconcile with the daily totals, supporting **159 inferred zeros** under the assumption that those totals account for all recorded rentals. The scoring target retains these zeros, including 83 hours within four retrospectively labelled disruption periods. A separate engineered lag-source series handles feature preparation without altering the scoring targets.

Forecast steps follow the supplied civil-time grid. Two nonexistent spring-forward positions are excluded from scoring, while repeated autumn hours cannot be distinguished using the available fields. A 48-step path therefore doesn't necessarily represent exactly 48 elapsed hours around a clock change.

## How I approached the comparison

I started with daily and weekly seasonal-naive forecasts and an untuned **four-week hour-of-week climatology**. I then compared Ridge regression, XGBoost, LightGBM, Prophet, and SARIMA on the same forecast occasions and targets.

The feature-based models use calendar information and historical summaries that can be supplied across the entire 48-step forecast path. Target-relative rental lags start at 48 hours.

Deployable weather features also use observations from 48 hours before each target timestamp, with forward filling restricted to past observations. A separate future-weather oracle experiment explores what happens when the model receives information that wouldn't be available at forecast issuance.

| Stage | Forecast paths | Target-date span in 2012 | Role |
|:---:|:---:|:---:|:---:|
| Early tuning | 22 | February 1 to April 26 | Select hyperparameters; later choose weather inclusion |
| Development selection | 92 | April 30 to October 30 | Compare frozen configurations |
| Final evaluation | 30 | November 1 to December 30 | Evaluate the selected pipeline and three benchmarks |

Tuning origins are four days apart, while development-selection and final-evaluation origins are two days apart. The tuning date span therefore includes gaps between forecast paths.

Development selection covers **4,416 hourly targets**, and final scoring covers **1,440**. The reserved final block extends through December 31, but its last 24 positions fall outside the complete 48-hour forecast paths and remain unscored.

Models are refitted at each origin using earlier observations. During final evaluation, earlier final-period counts can enter subsequent training fits as they become available, while the selected configuration remains fixed.

I use **MAE (mean absolute error)** as the primary selection metric, alongside pooled RMSE and MASE. MASE scales each forecast error using the mean absolute 24-hour difference in the training history available at its origin.

The development analysis also includes paired path-loss comparisons with Holm adjustment, sensitivity checks for serial dependence, and bootstrap ranking analyses.

## Main results

The selected pipeline is **LightGBM with lagged weather**, using an absolute-error objective, the raw rental target, and predictions clipped at zero.

| Procedure | Final MAE | Pooled RMSE | MASE |
|:---:|:---:|:---:|:---:|
| **LightGBM with lagged weather** | **51.882** | **86.042** | **0.811** |
| Four-week hour-of-week climatology | 64.534 | 109.906 | 1.009 |
| Weekly seasonal naive | 71.828 | 117.833 | 1.123 |
| Daily seasonal naive | 83.321 | 134.906 | 1.303 |

Across the 30 final forecast paths, LightGBM reduced mean absolute hourly error by **12.652 rentals**, approximately **19.6%**, relative to four-week climatology. It also had the lowest pooled RMSE among the four evaluated procedures.

The development comparison was much closer. The selected pipeline achieved an MAE of **48.590**, compared with **49.856** for climatology. Their paired difference remained statistically unresolved after multiple-comparison adjustment.

The larger final-period gap is encouraging, but it describes that particular historical block. I did not perform a final-period paired inferential test or calculate an uncertainty interval for the difference.

### Prediction intervals

I calibrated nominal **80% pointwise prediction intervals** using development forecast errors and froze the thresholds before final evaluation.

In the final period:

- **Per-horizon intervals (primary):** 78.7% coverage, mean width 145.1 rentals.
- **Pooled intervals (secondary):** 76.8% coverage, mean width 132.7 rentals.

Widths are measured after clipping interval bounds at zero.

These coverage rates concern individual hourly outcomes. Simultaneous coverage of all 48 hours in a forecast path was not evaluated.

The chronological, repeatedly refitted setting also does not establish the exchangeability assumptions needed for a standard conformal coverage guarantee. I therefore assess coverage empirically.

### Interpreting the evidence

The development comparison is part of an adaptive investigation: exploratory analysis across the development period informed the choice of features and candidate procedures.

An early data-quality check also revealed four inferred-zero outcomes and the pattern of absent timestamps in the final block before performance evaluation. I therefore describe that period as a **historical holdback with limited prior exposure**.

The notebook documents these qualifications, along with the limitations of transferring results from one bike-share system and two years of historical observations to other settings.

## Run the analysis

Use **Python 3.10**. The recorded execution used **Python 3.10.18**.

From the project root, the following commands create a virtual environment and launch JupyterLab on macOS or Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m ipykernel install --user \
  --name dc-bikeshare \
  --display-name "DC bikeshare (Python 3.10)"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m jupyterlab
```

Open [dc_bikeshare_demand_forecasting.ipynb](dc_bikeshare_demand_forecasting.ipynb), select **DC bikeshare (Python 3.10)**, restart the kernel, and run all cells in order.

Keep the `Data/` directory beside the notebook. The saved outputs can be inspected without rerunning the analysis.

The [requirements.txt](requirements.txt) file pins the recorded imported versions of the analysis packages and includes JupyterLab as a frontend convenience. The last fresh-kernel verification executed all 72 analytical code cells without errors and reproduced the headline results at their reported precision in the existing environment. A clean installation has not been tested; the notebook contains the execution record and documented numerical sensitivity.

For native-library installation issues, consult the official [LightGBM installation guide](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html) and [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html).

## Repository structure

```text
.
├── README.md
├── dc_bikeshare_demand_forecasting.ipynb
├── requirements.txt
├── .gitignore
└── Data/
    ├── hour.csv
    ├── day.csv
    └── Readme.txt
```

## Data and attribution

**Dataset:** Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository. [DOI: 10.24432/C5W894](https://doi.org/10.24432/C5W894).

The supplied [data documentation](Data/Readme.txt) also requests citation of Fanaee-T and Gama, *Event labeling combining ensemble detectors and background knowledge*. [DOI: 10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3).

The dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), which permits reuse, including commercial use, subject to its conditions. These include appropriate attribution, a license link, and identification of changes.

The reconstructed target and engineered features are transformations I made from the original data. The dataset license does not automatically apply to the project's code or unrelated third-party materials.

## Author

**Nahid Ahmadvand**

[LinkedIn](https://www.linkedin.com/in/nahid-ahmadvand-00382b365/) | [Website](https://www.nahidahmadvand.website/)

Questions, corrections, and ideas for extending the analysis are always welcome.
