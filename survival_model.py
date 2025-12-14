# models/survival_models.py

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index

class XGBoostCoxSurvivalModel:
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1):
        self.model = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.baseline_hazard = None
        self.baseline_survival = None
        self.time_points = None

    def fit(self, X, time, event):
        labels = time.copy()
        labels[event == 0] = -labels[event == 0]

        dtrain = xgb.DMatrix(X, label=labels)

        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "tree_method": "hist",
            "base_score": 0.5,
        }

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False,
        )

        self._compute_baseline_hazard(X, time, event)

    def _compute_baseline_hazard(self, X, time, event):
        dmatrix = xgb.DMatrix(X)
        risk_scores = self.model.predict(dmatrix)

        order = np.argsort(time)
        sorted_time = time.iloc[order]
        sorted_event = event.iloc[order]
        sorted_risk = np.exp(risk_scores[order])

        unique_times = np.unique(sorted_time[sorted_event == 1])
        baseline_hazard = []

        for t in unique_times:
            n_events = np.sum((sorted_time == t) & (sorted_event == 1))
            at_risk = sorted_time >= t
            risk_sum = np.sum(sorted_risk[at_risk])
            h0 = n_events / risk_sum if risk_sum > 0 else 0
            baseline_hazard.append(h0)

        cumulative_hazard = np.cumsum(baseline_hazard)
        self.baseline_survival = np.exp(-cumulative_hazard)
        self.time_points = unique_times

    def predict_survival_function(self, X, times):
        dmatrix = xgb.DMatrix(X)
        risk_scores = self.model.predict(dmatrix)
        risk_factors = np.exp(risk_scores)

        baseline_interp = np.interp(
            times,
            self.time_points,
            self.baseline_survival,
            left=1.0,
            right=self.baseline_survival[-1],
        )

        return np.array([baseline_interp ** rf for rf in risk_factors])

    def concordance_index(self, X, time, event):
        dmatrix = xgb.DMatrix(X)
        risk_scores = self.model.predict(dmatrix)
        return concordance_index(time, -risk_scores, event)
