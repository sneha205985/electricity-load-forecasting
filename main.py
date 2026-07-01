import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_FILE = "household_power_consumption.txt"
TARGET = "Global_active_power"


def load_data(file_path: str) -> pd.DataFrame:
    """Load semicolon-separated household power consumption TXT file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=";",
        na_values="?",
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime")
    df = df.sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample minute-level data into hourly values."""
    hourly = pd.DataFrame()

    hourly["Global_active_power"] = df["Global_active_power"].resample("h").mean()
    hourly["Global_reactive_power"] = df["Global_reactive_power"].resample("h").mean()
    hourly["Voltage"] = df["Voltage"].resample("h").mean()
    hourly["Global_intensity"] = df["Global_intensity"].resample("h").mean()

    hourly["Sub_metering_1"] = df["Sub_metering_1"].resample("h").sum()
    hourly["Sub_metering_2"] = df["Sub_metering_2"].resample("h").sum()
    hourly["Sub_metering_3"] = df["Sub_metering_3"].resample("h").sum()

    hourly = hourly.interpolate(method="time")
    hourly = hourly.dropna()

    return hourly


def create_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Create forecasting features using only past load and calendar features."""
    data = hourly.copy()

    data["lag_1"] = data[TARGET].shift(1)
    data["lag_2"] = data[TARGET].shift(2)
    data["lag_24"] = data[TARGET].shift(24)
    data["lag_48"] = data[TARGET].shift(48)
    data["lag_168"] = data[TARGET].shift(168)

    data["rolling_mean_24"] = data[TARGET].shift(1).rolling(24).mean()
    data["rolling_std_24"] = data[TARGET].shift(1).rolling(24).std()
    data["rolling_mean_168"] = data[TARGET].shift(1).rolling(168).mean()

    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day"] = data.index.day
    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    data = data[
        [
            TARGET,
            "lag_1",
            "lag_2",
            "lag_24",
            "lag_48",
            "lag_168",
            "rolling_mean_24",
            "rolling_std_24",
            "rolling_mean_168",
            "year",
            "month",
            "day",
            "hour",
            "day_of_week",
            "is_weekend",
        ]
    ]

    data = data.dropna()

    return data


def chronological_split(data: pd.DataFrame, test_size: float = 0.2):
    """Split data chronologically to avoid future data leakage."""
    split_index = int(len(data) * (1 - test_size))

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    return X_train, X_test, y_train, y_test, train, test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train Random Forest forecasting model."""
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=22,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred, name: str):
    """Calculate MAE and RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name}")
    print("-" * 35)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return mae, rmse


def print_model_summary(
    raw_data,
    hourly_data,
    feature_data,
    X_train,
    X_test,
    baseline_mae,
    baseline_rmse,
    model_mae,
    model_rmse,
    improvement
):
    """Print final model summary."""
    print("\n" + "=" * 45)
    print("MODEL SUMMARY")
    print("=" * 45)
    print(f"Raw dataset rows        : {len(raw_data):,}")
    print(f"Hourly samples          : {len(hourly_data):,}")
    print(f"Feature rows used       : {len(feature_data):,}")
    print(f"Training samples        : {len(X_train):,}")
    print(f"Testing samples         : {len(X_test):,}")
    print(f"Features used           : {X_train.shape[1]}")
    print(f"Baseline MAE            : {baseline_mae:.4f}")
    print(f"Baseline RMSE           : {baseline_rmse:.4f}")
    print(f"Random Forest MAE       : {model_mae:.4f}")
    print(f"Random Forest RMSE      : {model_rmse:.4f}")
    print(f"MAE improvement         : {improvement:.2f}%")
    print("Forecast horizon        : 24 hours")
    print("=" * 45)


def baseline_same_hour_yesterday(test: pd.DataFrame):
    """Naive baseline: predict using same hour from previous day."""
    return test["lag_24"]


def forecast_next_24_hours(
    model: RandomForestRegressor,
    hourly: pd.DataFrame,
    feature_columns
) -> pd.DataFrame:
    """Forecast the next 24 hours recursively using past load features."""
    history = hourly[[TARGET]].copy()
    forecasts = []

    future_dates = pd.date_range(
        start=history.index[-1] + pd.Timedelta(hours=1),
        periods=24,
        freq="h"
    )

    for future_time in future_dates:
        row = {
            "lag_1": history[TARGET].iloc[-1],
            "lag_2": history[TARGET].iloc[-2],
            "lag_24": history[TARGET].iloc[-24],
            "lag_48": history[TARGET].iloc[-48],
            "lag_168": history[TARGET].iloc[-168],
            "rolling_mean_24": history[TARGET].iloc[-24:].mean(),
            "rolling_std_24": history[TARGET].iloc[-24:].std(),
            "rolling_mean_168": history[TARGET].iloc[-168:].mean(),
            "year": future_time.year,
            "month": future_time.month,
            "day": future_time.day,
            "hour": future_time.hour,
            "day_of_week": future_time.dayofweek,
            "is_weekend": int(future_time.dayofweek in [5, 6]),
        }

        X_future = pd.DataFrame([row], index=[future_time])
        X_future = X_future[feature_columns]

        predicted_load = model.predict(X_future)[0]
        forecasts.append(predicted_load)

        new_row = pd.DataFrame(
            {TARGET: predicted_load},
            index=[future_time]
        )

        history = pd.concat([history, new_row])

    forecast_df = pd.DataFrame(
        {
            "datetime": future_dates,
            "predicted_load": forecasts
        }
    )

    return forecast_df


def plot_actual_vs_predicted(y_test, y_pred):
    """Save actual vs predicted plot."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index[:200], y_test.iloc[:200], label="Actual")
    plt.plot(y_test.index[:200], y_pred[:200], label="Predicted")
    plt.xlabel("Datetime")
    plt.ylabel("Global Active Power")
    plt.title("Actual vs Predicted Electricity Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    plt.close()


def plot_feature_importance(model, feature_names):
    """Save feature importance plot."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    importance.plot(kind="bar")
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()


def main():
    print("Loading data...")
    raw_data = load_data(DATA_FILE)

    print("Resampling minute-level data to hourly data...")
    hourly_data = resample_hourly(raw_data)

    print("Creating features...")
    feature_data = create_features(hourly_data)

    print("Splitting data chronologically...")
    X_train, X_test, y_train, y_test, train, test = chronological_split(feature_data)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating baseline and model...")
    baseline_predictions = baseline_same_hour_yesterday(test)
    model_predictions = model.predict(X_test)

    baseline_mae, baseline_rmse = evaluate(
        y_test,
        baseline_predictions,
        "Naive Baseline: Same Hour Yesterday"
    )

    model_mae, model_rmse = evaluate(
        y_test,
        model_predictions,
        "Random Forest Model"
    )

    improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
    print(f"\nMAE improvement over baseline: {improvement:.2f}%")

    print_model_summary(
        raw_data,
        hourly_data,
        feature_data,
        X_train,
        X_test,
        baseline_mae,
        baseline_rmse,
        model_mae,
        model_rmse,
        improvement
    )

    print("\nForecasting next 24 hours recursively...")
    forecast_df = forecast_next_24_hours(model, hourly_data, X_train.columns)

    print("\nNext 24-Hour Forecast")
    print("-" * 35)
    print(forecast_df.to_string(index=False))

    forecast_df.to_csv("next_24_hour_forecast.csv", index=False)

    plot_actual_vs_predicted(y_test, model_predictions)
    plot_feature_importance(model, X_train.columns)

    print("\nFiles created:")
    print("- next_24_hour_forecast.csv")
    print("- actual_vs_predicted.png")
    print("- feature_importance.png")
    print("\nAssessment questions are answered separately in answers.md")


if __name__ == "__main__":
    main()