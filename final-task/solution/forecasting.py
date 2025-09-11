import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.transformations.series.boxcox import BoxCoxTransformer
from tqdm.auto import tqdm
from utils import ProgressParallel


class SalesForecaster:
    """
    All-in-one sales forecasting class.
    """

    FH2STEP = {7: 89, 30: 83, 120: 61}

    def _select_data_for_store(self, sales: pd.DataFrame, dates: pd.DataFrame, prices: pd.DataFrame, store_id: int):
        """
        Selects data from sales, dates, prices tables for specified store.
        """
        sales = sales[sales["store_id"] == f"STORE_{store_id}"].drop(columns=["store_id"])
        sales["item_id"] = sales["item_id"].str.replace(f"STORE_{store_id}_", "", regex=False)

        cashback_cols = [col for col in dates.columns if col.startswith("CASHBACK_")]
        cols_to_drop = [col for col in cashback_cols if col != f"CASHBACK_STORE_{store_id}"]
        dates = dates.drop(columns=cols_to_drop)
        dates.rename(columns={f"CASHBACK_STORE_{store_id}": "cashback"}, inplace=True)

        prices = prices[prices["store_id"] == f"STORE_{store_id}"].drop(columns=["store_id"])
        prices["item_id"] = prices["item_id"].str.replace(f"STORE_{store_id}_", "", regex=False)

        return sales, dates, prices

    def _merge_data(self, sales: pd.DataFrame, dates: pd.DataFrame, prices: pd.DataFrame):
        """
        Merges sales, dates, and prices data into a single DataFrame.
        """
        merged_df = pd.merge(sales, dates, on="date_id", how="left")
        merged_df = pd.merge(merged_df, prices, on=["wm_yr_wk", "item_id"], how="left")

        return merged_df

    def apply_ema(self, series: pd.Series, alpha: float = 0.5):
        """
        Applies Exponential Moving Average (EMA) smoothing to the series.
        """
        return series.ewm(alpha=alpha).mean()

    def apply_imputation(self, series: pd.Series):
        """
        Interpolates zero values in the series using the weekly rolling mean window.
        """
        series = series.copy().astype(float)

        zero_mask = series == 0
        rolling_mean = series.replace(0, np.nan).rolling(window=7, min_periods=1, center=True).mean()
        series[zero_mask] = rolling_mean[zero_mask]

        # If any zeros remain (e.g., at edges), fill with forward/backward fill
        series = series.ffill().bfill()

        return series.round()

    def preprocess_data(
        self, sales: pd.DataFrame, dates: pd.DataFrame, prices: pd.DataFrame, store_id: str, cutoff_date_id: int = None
    ):
        """
        Prepares the data for forecasting:
        - Selects data for the specified store.
        - Merges the data into a single DataFrame.
        - Formatting columns by:
            - remove some unnecessary columns
            - renaming some columns
            - converting to needed types
        - (Optional) Deletes values older than cutoff date_id.
        - Applies OHE to categorical columns.
        - Asserts for NaNs in columns.
        """
        # Filter data for the specified store
        sales, dates, prices = self._select_data_for_store(sales, dates, prices, store_id)

        # Merge
        merged_df = self._merge_data(sales, dates, prices)

        # Formatting columns
        merged_df.drop(columns=["wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_name_2"], inplace=True)

        merged_df["cashback"] = pd.to_numeric(merged_df["cashback"], errors="raise")
        merged_df["sell_price"] = pd.to_numeric(merged_df["sell_price"], errors="raise")

        merged_df["date"] = pd.to_datetime(merged_df["date"], errors="raise")
        merged_df["day"] = merged_df["date"].dt.day
        merged_df["month"] = merged_df["date"].dt.month
        merged_df["year"] = merged_df["date"].dt.year
        merged_df.drop(columns=["date"], inplace=True)

        # (Optional) Delete values older than cutoff date_id
        if cutoff_date_id:
            merged_df = merged_df[merged_df["date_id"] > cutoff_date_id]

        # OHE for categorical columns
        categorical_cols = ["event_type_1", "event_type_2"]
        ohe = OneHotEncoder(sparse_output=False)
        ohe_encoded = ohe.fit_transform(merged_df[categorical_cols].fillna("NoEvent"))
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(categorical_cols))
        merged_df = pd.concat([merged_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        merged_df.drop(columns=categorical_cols, inplace=True)

        # Target
        merged_df.rename(columns={"cnt": "target"}, inplace=True)
        target_col = merged_df.pop("target")
        merged_df["target"] = target_col

        # Apply sorting by date_id
        merged_df["date_id"] = pd.to_numeric(merged_df["date_id"], errors="raise")
        merged_df = merged_df.sort_values(["date_id"]).reset_index(drop=True)

        # Asserts for NaNs in columns
        for col in merged_df.columns:
            assert not merged_df[col].isna().any(), f"NaNs in {col} column"

        return merged_df

    def make_data_splits(
        self,
        sales: pd.DataFrame,
        dates: pd.DataFrame,
        prices: pd.DataFrame,
        store_id: str,
        data_folder: Path,
        forecast_horizons: list = [7, 30, 120],
    ):
        """
        Prepares the data for forecasting:
        - Applies common preprocessing steps.
        - Makes (train + val) and test splits for each item_id in the store.
        - Saves the splits as CSV files in the specified folder.
        """
        train_val_size = 2 * 365 + 1 * 365  # 3 years

        for fh in tqdm(forecast_horizons, desc="Forecasting horizons"):
            cutoff_date_id = sales["date_id"].max() - train_val_size - fh
            data = self.preprocess_data(sales, dates, prices, store_id, cutoff_date_id=cutoff_date_id)
            for current_item_id in data.item_id.unique():
                item_data = data[data["item_id"] == current_item_id].copy()

                test_start_date_id = item_data["date_id"].max() - fh
                train_val_start_date_id = test_start_date_id - train_val_size - 1

                test = item_data[item_data["date_id"] > test_start_date_id]
                train_val = item_data[
                    (item_data["date_id"] > train_val_start_date_id) & (item_data["date_id"] <= test_start_date_id)
                ]

                # Save splits
                train_val.to_csv(data_folder / f"{store_id}_{current_item_id}_train_val_fh{fh}.csv", index=False)
                test.to_csv(data_folder / f"{store_id}_{current_item_id}_test_fh{fh}.csv", index=False)

    def post_process_predictions(self, series: pd.Series):
        """
        Post-processes the forecasted series:
        - Sets negative values to zero.
        - Rounds fractional values to the nearest integer.
        """
        series[series < 0] = 0
        series = series.round()
        return series

    def box_cox_transform(self, series: pd.Series):
        """
        Box-Cox transformation.
        """
        box_cox_transformer = BoxCoxTransformer()
        return box_cox_transformer.fit_transform(series.replace(0, 1e-6))

    def cross_validate_forecaster(
        self,
        forecaster,
        train_val: pd.DataFrame,
        fh: int,
        param_grid: dict,
        exog_cols: list,
        ema: float | None = None,
        zero_target_inputing: bool = False,
    ):
        """
        Custom grid search for forecaster with error skipping.

        - Tries all parameter combinations, skips those that fail on fit.
        - Parallelized using joblib (loky backend).
        - Uses SMAPE as the scoring metric.
        - Makes preprocessing for train split (zero imputation for high-seller items, EMA smoothing).
        - Makes postprocessing for predictions (set negatives to zero, round to int).
        - Fits the best forecaster on the last 2 years of train_val data.
        - Returns: best_params, best_score, best_forecaster, all_results (list of dicts)
        """
        cv = SlidingWindowSplitter(fh=np.arange(1, fh + 1), window_length=365 * 2, step_length=self.FH2STEP[fh])

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        def evaluate_params(param_tuple):
            params = dict(zip(param_names, param_tuple))
            try:
                local_forecaster = forecaster.set_params(**params)
                scores = []
                for train_idx, test_idx in cv.split(train_val):
                    y_train = train_val.target.iloc[train_idx]
                    y_test = train_val.target.iloc[test_idx]
                    X_train = train_val[exog_cols].iloc[train_idx] if exog_cols else None
                    X_test = train_val[exog_cols].iloc[test_idx] if exog_cols else None

                    if zero_target_inputing:
                        y_train = self.apply_imputation(y_train)
                    if ema:
                        y_train = self.apply_ema(y_train, alpha=ema)

                    forecaster_clone = local_forecaster.clone()
                    forecaster_clone.fit(y=y_train, X=X_train)

                    y_pred = forecaster_clone.predict(fh=np.arange(1, len(y_test) + 1), X=X_test)
                    y_pred = self.post_process_predictions(y_pred)

                    smape = MeanAbsolutePercentageError(symmetric=True)(y_test, y_pred)
                    scores.append(smape)
                if scores:
                    mean_score = np.mean(scores)
                    return {"params": params, "score": mean_score}
            except Exception as e:
                print(f"Skipping params {params} due to error: {str(e)}")
            return None

        results = ProgressParallel(n_jobs=-1, backend="loky")(
            delayed(evaluate_params)(param_tuple) for param_tuple in param_combinations
        )

        all_results = [r for r in results if r is not None]
        best_score = np.inf
        best_params = None
        for res in all_results:
            if res["score"] < best_score:
                best_score = res["score"]
                best_params = res["params"]

        best_forecaster = None
        # Fit best forecaster on last 2 years
        if best_params is not None:
            last_2_years = train_val.tail(365 * 2)

            y_last_2_years = last_2_years.target
            if zero_target_inputing:
                y_last_2_years = self.apply_imputation(y_last_2_years)
            if ema:
                y_last_2_years = self.apply_ema(y_last_2_years, alpha=ema)

            X_last_2_years = last_2_years[exog_cols] if exog_cols else None

            best_forecaster = forecaster.set_params(**best_params)
            best_forecaster.fit(y=y_last_2_years, X=X_last_2_years)

        return (best_params, best_score, all_results), best_forecaster

    def forecast(self, forecaster, fh: int, exog: None | pd.DataFrame) -> pd.Series:
        """
        Forecast the sales using the fitted forecaster for given forecasting horizon.
        """
        forecast = forecaster.predict(fh=np.arange(1, fh + 1), X=exog)
        forecast = self.post_process_predictions(forecast)
        return forecast

    def calculate_metrics(self, forecast: pd.Series, gt: pd.Series) -> dict[str, float]:
        """
        Calculates metrics (SMAPE, R2) for the forecasted series against the ground truth.
        """
        smape = MeanAbsolutePercentageError(symmetric=True)(gt, forecast)
        r2 = r2_score(gt, forecast)

        return {"SMAPE": smape, "R2": r2}

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
