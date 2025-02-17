from abc import ABC, abstractmethod
import os
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, EXTRACTED_DATA_DIR
from src.logging_config import logger
import zipfile


# Abstract class for data ingestor
class DataIngestor(ABC):
    output_dir: Path = EXTRACTED_DATA_DIR

    @abstractmethod
    def ingest(self, filepath: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Concrete class for zip ingestor
class ZipDataIngestor(DataIngestor):
    def ingest(self, filepath: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        if not filepath.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(self.output_dir)

        extracted_files = os.listdir(self.output_dir)
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError(
                "Multiple CSV files found. Please specify which one to use."
            )

        csv_file_path = os.path.join(self.output_dir, csv_files[0])
        df = pd.read_csv(csv_file_path)
        logger.success("Zipped data ingestion complete.")
        return df


# Implement a Factory to create DataIngestors and subsequently return a DataFrame
class DataIngestorFactory:
    @staticmethod
    def data_ingestor(filepath: str):
        logger.info("Data ingestion started...")
        file_extension = os.path.splitext(filepath)[1]
        if file_extension == ".csv":
            df = pd.read_csv(filepath)
            logger.success("Data ingestion complete.")
            return df
        elif file_extension == ".zip":
            return ZipDataIngestor().ingest(filepath)
        else:
            logger.error(f"No ingestor available for file extension: {file_extension}")
            raise ValueError(
                f"No ingestor available for file extension: {file_extension}"
            )


if __name__ == "__main__":
    input_dir: Path = RAW_DATA_DIR
    raw_files = os.listdir(input_dir)
    input_files = [f for f in raw_files if f not in [".gitkeep", ".DS_Store"]]
    if len(input_files) == 0:
        raise FileNotFoundError("No file found in the raw data.")
    if len(input_files) > 1:
        raise ValueError("Multiple files found. Please specify which one to use.")
    filepath = os.path.join(input_dir, input_files[0])
    DataIngestorFactory.data_ingestor(filepath)
