import dill
import os


# Function to save important files at a custom location
def save_object(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)


# Function to load a saved file
def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)
