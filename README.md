# Electricity Load Forecasting

This project performs day-ahead electricity load forecasting using the UCI Individual Household Electric Power Consumption dataset.

## Dataset

The dataset used is:

`household_power_consumption.txt`

The target variable is:

`Global_active_power`

The original data is minute-level electricity consumption data. It is resampled into hourly data using the hourly mean.

## Method

The pipeline includes:

1. Load the dataset
2. Combine Date and Time into a datetime index
3. Handle missing values
4. Resample minute-level data into hourly data
5. Create lag, rolling and calendar features
6. Split the data chronologically
7. Train a Random Forest model
8. Compare with a naive baseline
9. Forecast the next 24 hours

## Features Used

- Lag 1 hour
- Lag 2 hours
- Lag 24 hours
- Lag 48 hours
- Lag 168 hours
- Rolling 24-hour mean
- Rolling 24-hour standard deviation
- Hour
- Day of week
- Month
- Weekend indicator

## Model

Random Forest Regressor was used because it is reliable, easy to explain, and works well with engineered time-series features.

## Baseline

The baseline used is:

Same hour yesterday.

This means the value from 24 hours ago is used as the prediction.

## Evaluation

The model is evaluated using:

- MAE
- RMSE

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt