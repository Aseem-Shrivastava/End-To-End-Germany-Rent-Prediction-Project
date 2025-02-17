from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Univariate Analysis Strategy
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass


# Concrete Strategy for Numerical Features
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a KDE plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays KDE plot.
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df, x=feature, fill=True)
        plt.title(f"Probability Distribution of the feature: {feature.title()}")
        plt.xlabel(feature.title())
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


# Concrete Strategy for Categorical Features
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a bar plot showing the count of each category.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=feature)
        plt.xticks(rotation=45)
        plt.xlabel(feature.title())
        plt.ylabel("Count")
        plt.title(f"Distribution of the feature: {feature.title()}")
        plt.legend()
        plt.show()


# Context Class that uses a UnivariateAnalysisStrategy
# This class allows you to switch between different univariate analysis strategies
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature)
