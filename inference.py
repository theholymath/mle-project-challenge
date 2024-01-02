# inference script for the model
# use argparse for input arguments and data

# load the pickled model
# load the JSON list of features
# load the input data
# call the predict method

# Path: test_inference.py
import json
import pickle
import pathlib

import pandas

# load pickeld model
prince_model = pickle.load(open("model.pkl", "rb"))

# load JSON list of features
model_features = json.load(open("model_features.json", "r"))

# load input data


if __name__ == "__main__":
    # load input data
    x_test = pandas.read_csv("x_test.csv")

    # call predict method
    y_pred = prince_model.predict(x_test[model_features])

    # output predictions
    pandas.DataFrame(y_pred).to_csv("y_pred.csv", index=False)
    pandas.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_csv(
        "y_test_y_pred.csv", index=False
    )
