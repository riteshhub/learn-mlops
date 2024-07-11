import argparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder 
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="sample.csv")
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    input_data_path = os.path.join("/opt/ml/processing/input", args.filename)
    df = pd.read_csv(input_data_path)
    df.drop('Loan_ID', axis=1, inplace=True)

    num_cols = df.drop('Loan_Status', axis=1).select_dtypes(include=['float64','int64']).columns
    cat_cols = df.drop('Loan_Status', axis=1).select_dtypes(include='object').columns

    num_pipe = Pipeline(steps=[
        ('num_cols_impute', SimpleImputer(strategy='mean')),
        ('num_cols_scale', StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ('cat_cols_impute', SimpleImputer(strategy='most_frequent')),
        ('cat_cols_encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocess = ColumnTransformer(transformers=[
        ('num_col_trans', num_pipe, num_cols),
        ('cat_col_trans', cat_pipe, cat_cols)
    ], remainder='passthrough')

    final_pipe = Pipeline(steps=[
        ('preprocess', preprocess)
    ])

    le = LabelEncoder()
    df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

    y = df.pop('Loan_Status')
    X_pre = final_pipe.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    train_output_path = os.path.join("/opt/ml/processing/train", "train.csv")
    validation_output_path = os.path.join("/opt/ml/processing/validation", "validation.csv")
    test_output_path = os.path.join("/opt/ml/processing/test", "test.csv")

    print("Saving training dataset to {}".format(train_output_path))
    pd.DataFrame(train).to_csv(train_output_path, header=False, index=False)

    print("Saving validation dataset to {}".format(validation_output_path))
    pd.DataFrame(validation).to_csv(validation_output_path, header=False, index=False)

    print("Saving test dataset to {}".format(test_output_path))
    pd.DataFrame(test).to_csv(test_output_path, header=False, index=False)