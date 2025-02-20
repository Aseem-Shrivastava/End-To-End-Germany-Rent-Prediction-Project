from abc import ABC, abstractmethod
import mlflow
import os
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.config import MODELS_DIR
from src.logging_config import logger
from src.utils import save_object

from xgboost import XGBRegressor


class ModelBuilder(ABC):
    @abstractmethod
    def build_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        """
        Abstract method to build model.

        Parameters:
        X_train (pd.DataFrame)
        X_test (pd.DataFrame)
        y_train (pd.Series)
        y_test (pd.Series)

        Returns:
        Trained_model: Trained model based on the selected training strategy
        """
        pass


class HyperParameterTuned_ModelBuilder(ModelBuilder):
    def build_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        # Train and evaluate different models with cross-validation
        logger.info("Selecting model based on cross_val_score...")
        models = {
            "Linear Regression": LinearRegression(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge(),
            # "Decision Tree": DecisionTreeRegressor(),
            # "Random Forest": RandomForestRegressor(),
            # "Gradient Boosting": GradientBoostingRegressor(),
            "XGBRegressor": XGBRegressor(),
            # "SVR": SVR(),
        }

        metrics = {
            "MAE": make_scorer(mean_absolute_error),
            "MSE": make_scorer(mean_squared_error),
            "RMSE": make_scorer(
                lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            ),
            "R2": make_scorer(r2_score),
        }
        results = {}

        for model_name, model in models.items():
            results[model_name] = {}
            for metric_name, metric in metrics.items():
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=3,
                    scoring=metric,
                    error_score="raise",
                    n_jobs=-1,
                )
                results[model_name][metric_name] = scores.mean()

        # Converting the results dictionary into a dataframe to easily visualize and select the model with best f1 score
        results_df = pd.DataFrame(results).T
        print("\nModel scores on training dataset:")
        print(results_df)

        best_model = results_df["MAE"].idxmin()

        logger.success(
            "Model selection based on cross_val_score on training dataset completed"
        )

        # Saving the base model evaluation using MLflow
        mlflow.set_experiment("Testing Base Models")
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")

        for model_name in models:
            with mlflow.start_run(run_name=model_name):
                for metric_name, metric_value in results[model_name].items():
                    mlflow.log_metric(metric_name, metric_value)

        # Hyperparameter tuning the selected model
        logger.info(f"Hyperparameter tuning the {best_model} model...")

        # Hyperparameter grids for each model
        hyperparameter_grids = {
            "Linear Regression": {},  # No hyperparameters to tune
            "Lasso Regression": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
            "Ridge Regression": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
            "Decision Tree": {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["squared_error", "friedman_mse"],
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 6, 10],
            },
            "XGBRegressor": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
            "SVR": {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10],
                "epsilon": [0.01, 0.1, 0.2],
            },
        }

        param_grid = hyperparameter_grids.get(best_model, {})

        scoring = {
            "R2": "r2",
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "RMSE": "neg_root_mean_squared_error",
        }

        if param_grid:
            grid_search = GridSearchCV(
                estimator=models[best_model],
                param_grid=param_grid,
                scoring=scoring,
                refit="MAE",  # Optimizing for lowest MAE
                cv=3,
                verbose=1,
                n_jobs=-1,
            )

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            best_tuned_model = grid_search.best_estimator_

        else:
            best_params = {}
            best_tuned_model = models[best_model]  # No tuning, use default model

        print(f"\nBest Hyperparameters for {best_model}: {best_params}")

        # Evaluate the best-tuned model on the test dataset
        y_pred = best_tuned_model.predict(X_test)
        print("\nEvaluation of best-tuned model for test dataset:")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")

        # Save the best-tuned model as a .pkl file
        model_dir: Path = MODELS_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file_path = os.path.join(model_dir, "best_tuned_model.pkl")
        save_object(file_path=model_file_path, obj=best_tuned_model)

        print(f"Trained {best_model} model saved to {model_file_path}")

        logger.success(f"Hyperparameter-tuned {best_model} model saved")

        # Saving the hyperparameter-tuned model evaluation using MLflow
        mlflow.set_experiment("Hyperparameter-Tuned Model")
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")

        with mlflow.start_run():
            if param_grid:
                mlflow.log_params(best_params)
            else:
                mlflow.log_params(best_model)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2 Score", r2)

            if "XGB" in best_model:
                mlflow.xgboost.log_model(best_tuned_model, "hyperparameter-tuned model")
            else:
                mlflow.sklearn.log_model(best_tuned_model, "hyperparameter-tuned model")

        return best_tuned_model
