# Electricity Load Forecasting

A day-ahead electricity load forecasting project built using the **UCI Individual Household Electric Power Consumption** dataset.

The project converts minute-level electricity consumption data into hourly observations, performs feature engineering, trains a machine learning model, compares it against a naive baseline, and predicts the next 24 hours of electricity load.

---

## Project Overview

The forecasting pipeline includes:

- Loading the raw semicolon-separated TXT dataset
- Handling missing values (`?` → NaN)
- Combining Date and Time into a datetime index
- Resampling minute-level readings into hourly data
- Creating lag, rolling window, and calendar features
- Performing chronological train/test split
- Training a Random Forest regression model
- Comparing performance with a naive baseline
- Forecasting the next 24 hours
- Visualizing predictions and feature importance

---

## Dataset

Dataset used:

**UCI Individual Household Electric Power Consumption Dataset**

Target variable:

```
Global_active_power
```

The original dataset contains approximately four years of minute-level electricity consumption measurements for a single household.

> **Note:**  
> The dataset is not included in this repository because it exceeds GitHub's file size limit.
>
> Download it from:
>
> https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Place the downloaded file in the project directory as:

```
household_power_consumption.txt
```

---

## Feature Engineering

The model uses only historical electricity load and calendar information.

### Lag Features

- Lag 1 hour
- Lag 2 hours
- Lag 24 hours
- Lag 48 hours
- Lag 168 hours

### Rolling Statistics

- Rolling mean (24 hours)
- Rolling standard deviation (24 hours)
- Rolling mean (168 hours)

### Calendar Features

- Year
- Month
- Day
- Hour
- Day of week
- Weekend indicator

---

## Baseline

A naive forecasting baseline is implemented using:

> **Same Hour Yesterday**

The value from 24 hours earlier is used as the prediction.

This provides a strong benchmark for evaluating the machine learning model.

---

## Machine Learning Model

The forecasting model uses:

**Random Forest Regressor**

Reasons for choosing Random Forest:

- Handles non-linear relationships
- Works well with engineered features
- Easy to interpret
- Fast to train
- Strong baseline for structured time-series forecasting

---

## Evaluation Metrics

Performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The Random Forest model is compared directly against the naive baseline.

---

## Project Structure

```
electricity-load-forecasting/
│
├── main.py
├── answers.md
├── requirements.txt
├── README.md
├── actual_vs_predicted.png
├── feature_importance.png
├── next_24_hour_forecast.csv
└── LICENSE
```

---

## Output Files

Running the project generates:

- `next_24_hour_forecast.csv`
- `actual_vs_predicted.png`
- `feature_importance.png`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sneha205985/electricity-load-forecasting.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from UCI and place:

```
household_power_consumption.txt
```

inside the project folder.

Run:

```bash
python main.py
```

---

## Results

The project:

- Forecasts the next 24 hours of electricity load
- Produces prediction visualizations
- Displays feature importance
- Reports MAE and RMSE for both the baseline and Random Forest model

---

## Assessment Notes

The written answers to the technical assessment questions are provided separately in:

```
answers.md
```

---

## License

This project is released under the MIT License.
