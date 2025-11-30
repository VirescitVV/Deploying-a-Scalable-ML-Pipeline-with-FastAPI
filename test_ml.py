import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_model_with_predict():
    """
    Answers: “If any ML functions return the expected type of result”
    Train_model should return an object that looks like an ML model.
    It must at least implement a .predict() method.
    """
    X = np.array([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6],
                  [0.7, 0.8]])
    y = np.array([0, 1, 0, 1])

    model = train_model(X, y)

    assert hasattr(model, "predict")


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics_perfect_predictions():
    """
    Answers: “If the computing metrics functions return the expected value”
    When predictions are perfect, precision, recall, and F1 should all be 1.0.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


# TODO: implement the third test. Change the function name and input as needed
def test_process_data_output_types_and_lengths():
    """
    Answers: “If the training and test datasets have the expected size or data type”
    process_data should return numpy arrays for X and y with the
    same number of rows as the input DataFrame.
    """
    df = pd.DataFrame({
        "age": [25, 40, 35],
        "workclass": ["Private", "Self-emp-not-inc", "Private"],
        "salary": ["<=50K", ">50K", "<=50K"],
    })

    categorical_features = ["workclass"]

    X, y, encoder, lb = process_data(
        df,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]

    assert encoder is not None
    assert lb is not None
