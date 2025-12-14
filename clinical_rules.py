class ClinicalTestRecommender:
    def recommend_tests(self, shap_explanation, risk_trajectory, current_features):
        return {
            "monitoring": [],
            "short_term": [],
            "immediate": []
        }
