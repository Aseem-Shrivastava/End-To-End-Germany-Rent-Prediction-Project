from abc import ABC, abstractmethod
import os
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import ARTIFACTS_DIR
from src.logging_config import logger
from src.utils import save_object


class DataTransformation(ABC):
    @abstractmethod
    def apply_transformation(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Abstract method to apply data transformation.

        Parameters:
        X_train (pd.DataFrame): The X_train dataframe to transform.
        X_test (pd.DataFrame): The X_test dataframe to transform.

        Returns:
        X_train_transformed (pd.DataFrame): X_train dataframe with the applied transformations.
        X_test_transformed (pd.DataFrame): X_test dataframe with the applied transformations.
        """
        pass


class SimpleDataTransformation(DataTransformation):
    def apply_transformation(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        logger.info("X_train and X_test dataframes transformation started...")

        bool_columns = X_train.select_dtypes(include=["bool"]).columns.tolist()
        for col in bool_columns:
            X_train[col] = X_train[col].astype("Int64")
            X_test[col] = X_test[col].astype("Int64")

        numerical_continuous = X_train.select_dtypes(
            include=["float64"]
        ).columns.tolist()
        numerical_discrete = (
            X_train.select_dtypes(include=["int64"]).columns.tolist() + bool_columns
        )
        categorical_columns = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_continuous_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        numerical_discrete_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_cont", numerical_continuous_transformer, numerical_continuous),
                ("num_disc", numerical_discrete_transformer, numerical_discrete),
                ("cat", categorical_transformer, categorical_columns),
            ]
        )

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Saving preprocessor for later use (e.g. transformation of test data)
        artifact_dir: Path = ARTIFACTS_DIR
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_file_path = os.path.join(artifact_dir, "preprocessor.pkl")
        save_object(file_path=artifact_file_path, obj=preprocessor)
        logger.info("Saved preprocessing object.")

        feature_names = preprocessor.get_feature_names_out()

        X_train_transformed = pd.DataFrame(
            X_train_transformed.toarray(), columns=feature_names, index=X_train.index
        )
        X_test_transformed = pd.DataFrame(
            X_test_transformed.toarray(), columns=feature_names, index=X_test.index
        )

        logger.success("X_train and X_test dataframes transformation complete")

        return X_train_transformed, X_test_transformed
