import pandas as pd
from pathlib import Path
from src.config import MODELS_DIR, ARTIFACTS_DIR
from src.logging_config import logger
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, df: pd.DataFrame):
        """
        Predict method first transforms the custom test data and then outputs model prediction.

        Parameters:
        df (pd.DataFrame): Custom test dataframe.

        Returns:
        preds (pd.Series): Model prediction.
        """
        try:
            logger.info("Prediction started...")
            model_path: Path = MODELS_DIR / "best_tuned_model.pkl"
            preprocessor_path: Path = ARTIFACTS_DIR / "preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_transformed = preprocessor.transform(df)
            feature_names = preprocessor.get_feature_names_out()
            data_transformed = pd.DataFrame(data_transformed, columns=feature_names)

            preds = model.predict(data_transformed)
            logger.success("Prediction completed.")
            return preds

        except Exception as e:
            print(f"An error occurred: {e}")


class CustomData:
    def __init__(
        self,
        gender: str,
        age: float,
        hypertension: int,
        heart_disease: int,
        ever_married: str,
        work_type: str,
        residence_type: str,
        avg_glucose_level: float,
        bmi: float,
        smoking_status: str,
    ):
        self.gender = gender
        self.age = age
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.ever_married = ever_married
        self.work_type = work_type
        self.residence_type = residence_type
        self.avg_glucose_level = avg_glucose_level
        self.bmi = bmi
        self.smoking_status = smoking_status

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "hypertension": [self.hypertension],
                "heart_disease": [self.heart_disease],
                "ever_married": [self.ever_married],
                "work_type": [self.work_type],
                "Residence_type": [self.residence_type],
                "avg_glucose_level": [self.avg_glucose_level],
                "bmi": [self.bmi],
                "smoking_status": [self.smoking_status],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            print(f"An error occurred: {e}")
