# models/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class SurvivalAnalysisPreprocessor:
    """Handles preprocessing for survival analysis"""

    def __init__(self):
        self.preprocessor = None
        self.feature_names = []
        self.numerical_features = []
        self.categorical_features = []

    def fit_transform(self, X_train):
        self.numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, self.numerical_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )

        X_transformed = self.preprocessor.fit_transform(X_train)

        ohe = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = ohe.get_feature_names_out(self.categorical_features)

        self.feature_names = self.numerical_features + list(cat_names)

        return X_transformed

    def transform(self, X):
        return self.preprocessor.transform(X)

