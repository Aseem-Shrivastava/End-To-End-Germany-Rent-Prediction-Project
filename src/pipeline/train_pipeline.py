from src.components.data_ingestion import DataIngestorFactory
from src.components.data_splitter import DataSplitter, StratifiedTrainTestSplitStrategy
from src.components.data_transformation import SimpleDataTransformation
from src.components.model_trainer import HyperParameterTuned_ModelBuilder


def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    df = DataIngestorFactory.data_ingestor(
        "data/raw/healthcare-dataset-stroke-data.csv"
    )

    # Data Splitting Step
    data_splitter = DataSplitter(StratifiedTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = data_splitter.split(df, target_column="stroke")

    # Data Transformation Step
    data_transformer = SimpleDataTransformation()
    X_train_transformed, X_test_transformed = data_transformer.apply_transformation(
        X_train, X_test
    )

    # Model Builder Step
    model_builder = HyperParameterTuned_ModelBuilder()
    model = model_builder.build_model(
        X_train_transformed, X_test_transformed, y_train, y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
