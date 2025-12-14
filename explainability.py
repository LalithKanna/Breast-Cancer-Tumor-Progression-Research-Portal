import shap
import numpy as np
import xgboost as xgb

class SurvivalExplainer:
    """
    SHAP explainer for XGBoost Cox survival models.
    Explains contribution to log-hazard (risk score).
    """

    def __init__(self, survival_model, feature_names, background_data):
        """
        survival_model: XGBoostCoxSurvivalModel
        background_data: numpy array (N, F)
        """
        self.model = survival_model
        self.feature_names = feature_names

        # Use independent masker with background
        masker = shap.maskers.Independent(background_data)

        self.explainer = shap.Explainer(
            self._predict_margin,
            masker=masker,
            feature_names=self.feature_names
        )

    def _predict_margin(self, X):
        dmatrix = xgb.DMatrix(X)
        return self.model.model.predict(dmatrix, output_margin=True)

    def explain_individual(self, x):
        x = x.reshape(1, -1)

        shap_values = self.explainer(x).values[0]

        idx = np.argsort(np.abs(shap_values))[::-1][:15]

        explanation = {
            "interpretation": "Feature contributions to log-hazard (risk score)",
            "features": []
        }

        for i in idx:
            explanation["features"].append({
                "name": self.feature_names[i],
                "value": float(x[0, i]),
                "shap_value": float(shap_values[i]),
                "risk_multiplier": float(np.exp(shap_values[i])),
                "effect": (
                    "Increases risk"
                    if shap_values[i] > 0
                    else "Decreases risk"
                )
            })

        return explanation
