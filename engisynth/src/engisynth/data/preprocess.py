import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median"))]
    )
    cat_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("oh", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols)]
    )

