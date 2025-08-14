from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
)
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
)
from sktime.transformations.series.boxcox import BoxCoxTransformer


class SalesForecaster:
    """
    All-in-one sales forecasting class.
    """

    def prepare_data(self, data_folder: Path, store_id: str):
        """
        Prepares the data for forecasting:
        - Loads sales, dates, prices data from CSV files.
        - Selects data for the specified store.
        - Merges the data into a single DataFrame.
        - Make training, validation, and test splits of sizes (2 years, 1 year, 4 months).
        - Saves the splits to CSV files with names: <store_id>_<item_id>_train_val.csv and <store_id>_<item_id>_test.csv.
        """
        sales = pd.read_csv(data_folder / "shop_sales.csv")
        dates = pd.read_csv(data_folder / "shop_sales_dates.csv")
        prices = pd.read_csv(data_folder / "shop_sales_prices.csv")

        # Filter data for the specified store
        sales = sales[sales["store_id"] == store_id].drop(columns=["store_id"])
        cashback_cols = [col for col in dates.columns if col.startswith("CASHBACK_")]
        cols_to_drop = [col for col in cashback_cols if col != f"CASHBACK_{store_id}"]
        dates.drop(columns=cols_to_drop, inplace=True)
        dates.rename(columns={f"CASHBACK_{store_id}": "CASHBACK"}, inplace=True)
        prices = prices[prices["store_id"] == store_id].drop(columns=["store_id"])

        dates["date"] = pd.to_datetime(dates["date"])

        # Merge
        merged_df = pd.merge(sales, dates, on="date_id", how="left")
        merged_df = pd.merge(merged_df, prices, on=["wm_yr_wk", "item_id"], how="left")
        merged_df = merged_df.sort_values("date")

        # Train, validation, and test splits
        test_size = 4 * 30  # 4 months
        train_val_size = 2 * 365 + 1 * 365  # 2 years + 1 year

        test_start_date_id = merged_df["date_id"].max() - test_size
        train_val_start_date_id = test_start_date_id - train_val_size - 1

        test = merged_df[merged_df["date_id"] > test_start_date_id]
        train_val = merged_df[
            (merged_df["date_id"] > train_val_start_date_id) & (merged_df["date_id"] <= test_start_date_id)
        ]

        # Save splits (item_id: train_val, test)
        for item_id in sales.item_id.unique():
            item_train_val = train_val[train_val["item_id"] == item_id]
            item_test = test[test["item_id"] == item_id]

            item_train_val.to_csv(data_folder / f"{store_id}_{item_id}_train_val.csv", index=False)
            item_test.to_csv(data_folder / f"{store_id}_{item_id}_test.csv", index=False)

    def box_cox_transform(self, series: pd.Series):
        """
        Box-Cox transformation.
        """
        box_cox_transformer = BoxCoxTransformer()
        return box_cox_transformer.fit_transform(series.replace(0, 1e-6))

    def cross_validate_forecaster(self, forecaster, train_val: pd.DataFrame, fh: int, param_grid: dict):
        """
        Cross-validate the forecaster on the training and validation data.

        Uses ForecastingGridSearchCV and SMAPE as the scoring metric.

        Returns the fitted ForecastingGridSearchCV object and fitted on last 2 years of train_val data estimator.
        """
        if fh == 7:
            cv = SlidingWindowSplitter(fh=np.arange(1, 8), window_length=365 * 2, step_length=38)
        elif fh == 30:
            cv = SlidingWindowSplitter(fh=np.arange(1, 31), window_length=365 * 2, step_length=35)
        elif fh == 120:
            cv = SlidingWindowSplitter(fh=np.arange(1, 121), window_length=365 * 2, step_length=25)
        else:
            raise ValueError("Unsupported forecasting horizon. Use 7, 30, or 120.")

        gscv = ForecastingGridSearchCV(
            forecaster=forecaster,
            cv=cv,
            param_grid=param_grid,
            scoring=MeanAbsolutePercentageError(symmetric=True),
            verbose=3,
        )
        gscv.fit(train_val.cnt)

        # Fit the best forecaster on the last 2 years of train_val data
        last_2_years = train_val.tail(365 * 2)
        best_forecaster = gscv.best_forecaster_.clone()
        best_forecaster.fit(last_2_years.cnt)

        return gscv, best_forecaster

    def forecast(self, forecaster, fh: int) -> pd.Series:
        """
        Forecast the sales using the fitted forecaster.
        Uses different forecasting horizons:

        - 7 days
        - 30 days
        - 120 days

        """
        if fh == 7:
            forecast = forecaster.predict(fh=np.arange(1, 8))
        elif fh == 30:
            forecast = forecaster.predict(fh=np.arange(1, 31))
        elif fh == 120:
            forecast = forecaster.predict(fh=np.arange(1, 121))
        else:
            raise ValueError("Unsupported forecasting horizon. Use 7, 30, or 120.")

        return forecast

    def calculate_metrics(self, forecast: pd.Series, gt: pd.Series) -> dict[str, float]:
        """
        Calculates metrics (MAE, SMAPE, R2) for the forecasted series against the ground truth.
        """
        mae = MeanAbsoluteError()(gt, forecast)
        smape = MeanAbsolutePercentageError(symmetric=True)(gt, forecast)
        r2 = r2_score(gt, forecast)

        return {"MAE": mae, "SMAPE": smape, "R2": r2}

    def save_model(self, model, save_path: Path):
        """
        Saves the model to the specified path using joblib.
        """
        joblib.dump(model, save_path)

    def load_model(self, load_path: Path):
        """
        Loads the model from the specified path using joblib.
        """
        return joblib.load(load_path)
