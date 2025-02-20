from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logging_config import logger


# Abstract Base Class for Data Splitting Strategy
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass


# Concrete strategy for Simple Train-Test Split
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        df = df.dropna(axis=1, thresh=0.7 * df.shape[0])
        df = df.drop(
            columns=[
                "livingSpaceRange",
                "street",
                "description",
                "facilities",
                "geo_krs",
                "geo_plz",
                "scoutId",
                "regio1",
                "telekomUploadSpeed",
                "telekomTvOffer",
                "pricetrend",
                "houseNumber",
                "streetPlain",
                "regio3",
                "noRoomsRange",
                "picturecount",
                "geo_bln",
                "date",
                "baseRentRange",
                "baseRent",
                "serviceCharge",
            ]
        )

        df = df.dropna(subset=["totalRent"])

        # Function to cap outliers
        def cap_outlier(col):
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            return col.map(
                lambda x: lower_bound
                if x < lower_bound
                else (upper_bound if x > upper_bound else x)
            )

        numerical_continuous = df.select_dtypes(include=["float64"]).columns.tolist()

        df[numerical_continuous] = df[numerical_continuous].apply(cap_outlier)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logger.success("Simple splitting of data complete")
        return X_train, X_test, y_train, y_test


class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        df = df.dropna(axis=1, thresh=0.7 * df.shape[0])
        df = df.drop(
            columns=[
                "livingSpaceRange",
                "street",
                "description",
                "facilities",
                "geo_krs",
                "geo_plz",
                "scoutId",
                "regio1",
                "telekomUploadSpeed",
                "telekomTvOffer",
                "pricetrend",
                "houseNumber",
                "streetPlain",
                "regio3",
                "noRoomsRange",
                "picturecount",
                "geo_bln",
                "date",
                "baseRentRange",
                "baseRent",
                "serviceCharge",
            ]
        )

        # Function to cap outliers
        def cap_outlier(col):
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            return col.map(
                lambda x: lower_bound
                if x < lower_bound
                else (upper_bound if x > upper_bound else x)
            )

        numerical_continuous = df.select_dtypes(include=["float64"]).columns.tolist()

        df[numerical_continuous] = df[numerical_continuous].apply(cap_outlier)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        logger.success("Stratified splitting of data complete")
        return X_train, X_test, y_train, y_test


# Context Class for Data Splitting
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific data splitting strategy.

        Parameters:
        strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logger.info("Switching data splitting strategy.")
        self.set_strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Executes the data splitting using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logger.info("Splitting data using the selected strategy...")
        return self._strategy.split_data(df, target_column)
