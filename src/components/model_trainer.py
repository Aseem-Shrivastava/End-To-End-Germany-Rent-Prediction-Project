from abc import ABC, abstractmethod
import os
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR
from src.logging_config import logger
from src.utils import save_object

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


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
        logger.info("Applying SMOTE to balance the training data...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        logger.success("Training data balanced using SMOTE.")

        # Train and evaluate different models with cross-validation
        logger.info("Selecting model based on cross_val_score...")
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            # "XGBClassifier": XGBClassifier(),
        }

        metrics = ["accuracy", "recall", "precision", "f1"]
        results = {}

        for name, model in models.items():
            results[name] = {}
            for metric in metrics:
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring=metric)
                results[name][metric] = scores.mean()

        # Converting the results dictionary into a dataframe to easily visualize and select the model with best f1 score
        results_df = pd.DataFrame(results).T
        print("\nModel scores on training dataset:")
        print(results_df)

        best_model = results_df["f1"].idxmax()

        logger.success(
            "Model selection based on cross_val_score on training dataset completed"
        )

        # Hyperparameter tuning the selected model
        logger.info(f"Hyperparameter tuning the {best_model} model...")

        # Define hyperparameter grids for each model
        hyperparameter_grids = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "saga"],
            },
            "Decision Tree": {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "XGBClassifier": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        }

        param_grid = hyperparameter_grids[best_model]

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }

        grid_search = GridSearchCV(
            estimator=models[best_model],
            param_grid=param_grid,
            scoring=scoring,
            refit="f1",
            cv=3,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_tuned_model = grid_search.best_estimator_

        print(f"\nBest Hyperparameters for {best_model}: {best_params}")

        # Evaluate the best-tuned model on the test dataset
        y_pred = best_tuned_model.predict(X_test)
        print("\nClassification Report of best-tuned model for test dataset:")
        print(classification_report(y_test, y_pred))

        # Save the best-tuned model as a .pkl file
        model_dir: Path = MODELS_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file_path = os.path.join(model_dir, "best_tuned_model.pkl")
        save_object(file_path=model_file_path, obj=best_tuned_model)

        print(f"Trained {best_model} model saved to {model_file_path}")

        logger.success(f"Hyperparameter-tuned {best_model} model saved")

        return best_tuned_model
