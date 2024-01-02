import json
import pathlib
import pickle
import joblib
from typing import List
from typing import Tuple

import pandas
import numpy as np
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(
        sales_path, usecols=sales_column_selection, dtype={"zipcode": str}
    )
    demographics = pandas.read_csv(demographics_path, dtype={"zipcode": str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    # Remove the target variable from the dataframe, features will remain
    print(merged_data.head())
    y = merged_data.pop("price")
    x = merged_data

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42, test_size=0.2
    )

    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(), neighbors.KNeighborsRegressor()
    ).fit(x_train, y_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", "wb"))
    json.dump(list(x_train.columns), open(output_dir / "model_features.json", "w"))

    # Save the trained model using joblin to the output dir as well
    joblib.dump(model, output_dir / "model.joblib")

    # Output test data for use in testing
    x_train.to_csv(output_dir / "x_train.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    x_test.to_csv(output_dir / "x_test.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    # predict model on test data and output predictions and perfomance metrics
    y_pred = model.predict(x_test)
    pandas.DataFrame(y_pred).to_csv(output_dir / "y_pred.csv", index=False)
    pandas.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_csv(
        output_dir / "y_test_y_pred.csv", index=False
    )

    # evalute model performance

    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("VarScore:", metrics.explained_variance_score(y_test, y_pred))


if __name__ == "__main__":
    main()
