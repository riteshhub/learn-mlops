from sklearn.metrics import roc_auc_score, f1_score
import json
import os
import tarfile
import pandas as pd
import xgboost
import joblib

if __name__ == "__main__":
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    print("Extracting model from path: {}".format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    print("Loading model")
    model = joblib.load("xgboost-model")

    print("Loading test input data")
    test_data = os.path.join("/opt/ml/processing/test", "test.csv")

    df = pd.read_csv(test_data, header=None)
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    predictions = model.predict(X_test)

    print("constructing metrics")
    auc = roc_auc_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    report_dict = {
        "classification_metrics": {
            "auc": {"value": auc},
            "f1": {"value": f1}, 
        },
    }

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))