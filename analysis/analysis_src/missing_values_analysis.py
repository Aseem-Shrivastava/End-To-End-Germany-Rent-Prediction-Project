from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Missing Values Analysis
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

        @abstractmethod
        def identify_missing_values(self, df: pd.DataFrame):
            """
            Identifies missing values in the dataframe.

            Parameters:
            df (pd.DataFrame): The dataframe to be analyzed.

            Returns:
            None: This method should print the percentage of missing values for each column.
            """
            pass

        @abstractmethod
        def visualize_missing_values(self, df: pd.DataFrame):
            """
            Visualizes missing values in the dataframe.

            Parameters:
            df (pd.DataFrame): The dataframe to be visualized.

            Returns:
            None: This method should create a visualization (e.g., heatmap) of missing values.
            """
            pass


# Concrete Class for Missing Values Identification
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the percentage of missing values for each column in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values percentage to the console.
        """
        print("\nMissing Values Percentage by Column:")
        missing_values_percentage = df.isnull().sum(axis=0) * 100 / df.shape[0]
        print(round(missing_values_percentage, 2))

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
