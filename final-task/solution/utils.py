import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm.auto import tqdm


class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def pretty_info(df, name):
    print(f"{name}:")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Rows: {df.shape[0]}")
    print("-" * 50)


def plot_item_features(df):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df["date"], df["cnt"], label="Sales", color="tab:blue")

    # Plot green vertical lines for weekends (wday=6 or 7)
    weekend_mask = df["wday"].isin([6, 7])
    weekend_dates = df.loc[weekend_mask, "date"].values
    for i, date in enumerate(weekend_dates):
        ax.axvline(
            x=date,
            color="green",
            linewidth=1,
            alpha=0.5,
            label="Weekend" if i == 0 and "Weekend" not in ax.get_legend_handles_labels()[1] else "",
        )

    # Plot * at y=0 for event_name_1 not NaN
    event_mask = df["event_name_1"].notna()
    event_dates = df.loc[event_mask, "date"].values
    ax.scatter(
        event_dates,
        np.zeros(len(event_dates)),
        marker="*",
        color="red",
        s=120,
        label="Event" if "Event" not in ax.get_legend_handles_labels()[1] else "",
    )

    ax.set_xlabel("date")
    ax.set_ylabel("Sales Count")
    ax.set_title("Sales, Events, and Weekends")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_linear_trend(df, date_col="date", target_col="cnt") -> LinearRegression:
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[target_col].values

    lr = LinearRegression()
    lr.fit(X, y)
    trend = lr.predict(X)
    slope = lr.coef_[0]

    plt.figure(figsize=(14, 5))
    plt.plot(df[date_col], y, label="Sales Count")
    plt.plot(
        df[date_col],
        trend,
        label=f"Trend (Linear Regression)\nSlope: {slope:.4f}",
        color="red",
        linewidth=2,
    )
    plt.title("Sales Count and Linear Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales Count")
    plt.legend()
    plt.show()

    return lr


def test_trend(row):
    """
    Case 1: Both tests conclude that the series is not stationary - The series is not stationary
    Case 2: Both tests conclude that the series is stationary - The series is stationary
    Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
    Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
    """
    kpsstest = kpss(row, regression="c", nlags="auto")
    dftest = adfuller(row, autolag="AIC")
    nonstationary_kpss = kpsstest[1] < 0.05
    nonstationary_df = dftest[1] > 0.05
    return nonstationary_kpss | nonstationary_df, {"KPSS": kpsstest[1], "ADF": dftest[1]}


def save_metrics_to_json(model_name: str, fh: int, results: dict, file_path: Path):
    """
    Saves the metrics for each item and the average metrics to a JSON file.
    The structure of the JSON file is:
    {
        "fh": {
            "model_name": {
                "item_id": {
                    "best_params": {...},
                    "metrics": {...}
                },
                "AVERAGE": {
                    "metrics": {...}
                }
            }
        }
    }
    """
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            all_json_results = json.load(f)
    else:
        all_json_results = {}

    fh_str = str(fh)
    if fh_str not in all_json_results:
        all_json_results[fh_str] = {}
    if model_name not in all_json_results[fh_str]:
        all_json_results[fh_str][model_name] = {}

    for item_id in results:
        best_params, _, metrics = results[item_id]
        all_json_results[fh_str][model_name][item_id] = {
            "best_params": best_params,
            "metrics": metrics,
        }

    # Calculate average metrics
    metrics_list = [results[item_id][2] for item_id in results]
    if metrics_list:
        avg_metrics = {}
        metric_keys = metrics_list[0].keys()
        for key in metric_keys:
            vals = [m[key] for m in metrics_list if isinstance(m[key], (int, float))]
            avg_metrics[key] = float(np.mean(vals)) if vals else None
        all_json_results[fh_str][model_name]["AVERAGE"] = {
            "best_params": "",
            "metrics": avg_metrics,
        }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_json_results, f, ensure_ascii=False, indent=4)


def load_info_from_json(file_path: Path, fh: int, model_name: str, item_id: str) -> tuple | None:
    """
    Loads the metrics for a specific item from a JSON file.
    Returns a tuple of (best_params, metrics) or None if not found.
    """
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        all_json_results = json.load(f)

    fh_str = str(fh)
    if fh_str in all_json_results and model_name in all_json_results[fh_str]:
        model_results = all_json_results[fh_str][model_name]
        if item_id in model_results:
            return (
                model_results[item_id]["best_params"],
                model_results[item_id]["metrics"],
            )

    return None


def load_avg_metrics_from_json(file_path: Path, fh: int, model_name: str) -> dict | None:
    """
    Loads the average metrics for a specific model from a JSON file.
    Returns a dictionary of average metrics or None if not found.
    """
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        all_json_results = json.load(f)

    fh_str = str(fh)
    if fh_str in all_json_results and model_name in all_json_results[fh_str]:
        model_results = all_json_results[fh_str][model_name]
        if "AVERAGE" in model_results:
            return model_results["AVERAGE"]["metrics"]

    return None
