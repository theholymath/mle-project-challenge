# Inference script for the model
from fastapi import FastAPI, Request
from pydantic import BaseModel, validator
import joblib
import numpy as np
import pandas as pd
import pathlib
import logging
import json

# local imports
from config import SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION, OUTPUT_DIR
from create_model import load_data

# set up logging
# Configure the root logger to write to a file
logging.basicConfig(filename="app.log", level=logging.INFO)

# Create a custom logger
logger = logging.getLogger(__name__)

# Create a FileHandler
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)

# Create a formatter and set the format for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


SALES_COLUMN_SELECTION = [val for val in SALES_COLUMN_SELECTION if val != "price"]
output_dir = pathlib.Path(OUTPUT_DIR)

# Load the trained model
model = joblib.load(output_dir / "model.joblib")

# set up app
app = FastAPI()


# Define the Pydantic model for the request body
class InputData(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the ML Model API"}


@app.post("/predict/")
async def predict(request: Request, data: InputData):
    logger.info("Prediction endpoint accessed")
    result = await request.json()

    logger.info("Received Data: %s", result)
    print("Received Data:\n", result, "\n")

    features = pd.DataFrame(result, index=[0])
    features.to_csv("temp.csv", index=False)

    # merge data using code provided and return features ready for prediction
    features, _ = load_data("temp.csv", DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # # Log the variable data
    logger.info("Merged request data:\n %s", features)
    print("\nMerged request data:\n", features, "\n")

    prediction = model.predict(features)
    logger.info("Prediction: %s", prediction[0])
    print("PREDICTION", prediction)

    return {"predictied price": prediction[0]}
