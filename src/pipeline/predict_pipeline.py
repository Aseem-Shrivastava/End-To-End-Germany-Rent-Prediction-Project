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

            # Convert to array only if sparse matrix
            if hasattr(data_transformed, "toarray"):
                data_transformed = data_transformed.toarray()

            preds = model.predict(data_transformed)
            logger.success("Prediction completed.")
            return preds

        except Exception as e:
            print(f"An error occurred: {e}")


class CustomData:
    def __init__(
        self,
        regio2: str,
        typeOfFlat: str,
        livingSpace: float,
        noRooms: int,
        hasKitchen: int,
        cellar: int,
        balcony: int,
        lift: int,
        garden: int,
        floor: int,
        heatingType: str,
        firingTypes: str,
        newlyConst: int,
        yearConstructed: int,
        yearConstructedRange: int,
        condition: str,
    ):
        self.regio2 = regio2
        self.heatingType = heatingType
        self.newlyConst = newlyConst
        self.balcony = balcony
        self.yearConstructed = yearConstructed
        self.firingTypes = firingTypes
        self.hasKitchen = hasKitchen
        self.cellar = cellar
        self.yearConstructedRange = yearConstructedRange
        self.livingSpace = livingSpace
        self.condition = condition
        self.lift = lift
        self.typeOfFlat = typeOfFlat
        self.noRooms = noRooms
        self.floor = floor
        self.garden = garden

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "regio2": [self.regio2],
                "typeOfFlat": [self.typeOfFlat],
                "livingSpace": [self.livingSpace],
                "noRooms": [self.noRooms],
                "hasKitchen": [self.hasKitchen],
                "cellar": [self.cellar],
                "balcony": [self.balcony],
                "lift": [self.lift],
                "garden": [self.garden],
                "floor": [self.floor],
                "heatingType": [self.heatingType],
                "firingTypes": [self.firingTypes],
                "newlyConst": [self.newlyConst],
                "yearConstructed": [self.yearConstructed],
                "yearConstructedRange": [self.yearConstructedRange],
                "condition": [self.condition],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            print(f"An error occurred: {e}")
