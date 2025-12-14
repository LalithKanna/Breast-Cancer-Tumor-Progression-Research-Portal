"""
Breast Cancer Survival Analysis System - Training Pipeline
Uses Cox Proportional Hazards Models for Time-to-Event Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import xgboost as xgb
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')


class SurvivalAnalysisPreprocessor:
    """Handles preprocessing for survival analysis"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = []
        self.numerical_features = []
        self.categorical_features = []
    
    def fit_transform(self, X_train):
        """Fit preprocessor and transform training data"""
        # Identify feature types
        self.numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessing pipelines
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine into ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ]
        )
        
        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X_train)
        
        # Get feature names
        ohe = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_names = ohe.get_feature_names_out(self.categorical_features)
        self.feature_names = self.numerical_features + list(cat_names)
        
        return X_transformed
    
    def transform(self, X):
        """Transform new data"""
        return self.preprocessor.transform(X)


class XGBoostCoxSurvivalModel:
    """XGBoost-based Cox Proportional Hazards Model"""
    
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1):
        self.model = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.baseline_hazard = None
        self.baseline_survival = None
        self.time_points = None
    
    def fit(self, X, time, event):
        """
        Fit XGBoost Cox model
        X: feature matrix (already transformed)
        time: time to event or censoring (positive values)
        event: event indicator (1 = event occurred, 0 = censored)
        """
        # Create labels for survival:cox: positive time for events, negative for censored
        labels = time.copy()
        labels[event == 0] = -labels[event == 0]  # Negative for censored
        
        # Create DMatrix with proper labels
        dtrain = xgb.DMatrix(X, label=labels)
        
        # Parameters
        params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': self.max_depth,
        'learning_rate': self.learning_rate,
        'tree_method': 'hist',
        'base_score': 0.5  # ← Add this line
    }
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        
        # Compute baseline hazard (Breslow estimator) - keep your existing logic
        self._compute_baseline_hazard(X, time, event)
        
        return self
    def _compute_baseline_hazard(self, X, time, event):
        """Compute baseline hazard function using Breslow estimator"""
        # Get risk scores
        dmatrix = xgb.DMatrix(X)
        risk_scores = self.model.predict(dmatrix)
        
        # Sort by time
        order = np.argsort(time)
        sorted_time = time.iloc[order] if isinstance(time, pd.Series) else time[order]
        sorted_event = event.iloc[order] if isinstance(event, pd.Series) else event[order]
        sorted_risk = np.exp(risk_scores[order])
        
        # Get unique event times
        unique_times = np.unique(sorted_time[sorted_event == 1])
        
        # Breslow estimator
        baseline_hazard = []
        
        for t in unique_times:
            # Number of events at time t
            n_events = np.sum((sorted_time == t) & (sorted_event == 1))
            
            # Risk set at time t
            at_risk = sorted_time >= t
            risk_sum = np.sum(sorted_risk[at_risk])
            
            # Baseline hazard increment
            h0 = n_events / risk_sum if risk_sum > 0 else 0
            baseline_hazard.append(h0)
        
        # Cumulative baseline hazard
        cumulative_hazard = np.cumsum(baseline_hazard)
        
        # Baseline survival function
        baseline_survival = np.exp(-cumulative_hazard)
        
        self.time_points = unique_times
        self.baseline_hazard = np.array(baseline_hazard)
        self.baseline_survival = baseline_survival
    
    def predict_survival_function(self, X, times=None):
        """
        Predict survival probability at given times
        Returns: survival probabilities for each sample at each time point
        """
        if times is None:
            times = self.time_points
        
        # Get risk scores
        dmatrix = xgb.DMatrix(X)
        risk_scores = self.model.predict(dmatrix)
        risk_factors = np.exp(risk_scores)
        
        # Interpolate baseline survival to requested times
        baseline_surv_interp = np.interp(
            times,
            self.time_points,
            self.baseline_survival,
            left=1.0,
            right=self.baseline_survival[-1]
        )
        
        # Individual survival: S(t) = S0(t)^exp(X*beta)
        survival_probs = np.array([
            baseline_surv_interp ** rf for rf in risk_factors
        ])
        
        return survival_probs
    
    def predict_hazard_ratio(self, X):
        """Predict hazard ratio (exp(linear predictor))"""
        dmatrix = xgb.DMatrix(X)
        risk_scores = self.model.predict(dmatrix)
        return np.exp(risk_scores)
    
    def concordance_index(self, X, time, event):
        """Calculate concordance index (C-index)"""
        dmatrix = xgb.DMatrix(X)
        risk_scores = self.model.predict(dmatrix)
        return concordance_index(time, -risk_scores, event)


class SurvivalExplainer:
    """
    XGBoost-native SHAP explainer for Cox survival models.
    Explains contribution to log-hazard.
    """

    def __init__(self, model, feature_names):
        self.model = model.model  # xgboost Booster
        self.feature_names = feature_names

    def explain_global(self, X):
        """
        Global importance = mean |SHAP| over samples
        """
        dmatrix = xgb.DMatrix(X)
        shap_values = self.model.predict(dmatrix, pred_contribs=True)

        # Remove bias term (last column)
        shap_values = shap_values[:, :-1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(
            sorted(
                zip(self.feature_names, mean_abs),
                key=lambda x: x[1],
                reverse=True
            )
        )

    def explain_individual(self, x):
        """
        Explain one patient's hazard risk
        """
        dmatrix = xgb.DMatrix(x.reshape(1, -1))
        shap_values = self.model.predict(dmatrix, pred_contribs=True)[0]

        bias = shap_values[-1]
        shap_values = shap_values[:-1]

        idx = np.argsort(np.abs(shap_values))[::-1][:15]

        explanation = {
            "prediction_type": "log_hazard (Cox)",
            "base_hazard": float(np.exp(bias)),
            "features": []
        }

        for i in idx:
            explanation["features"].append({
                "feature": self.feature_names[i],
                "value": float(x[i]),
                "shap_value": float(shap_values[i]),
                "hazard_multiplier": float(np.exp(shap_values[i])),
                "direction": (
                    "increases risk" if shap_values[i] > 0
                    else "decreases risk"
                )
            })

        return explanation

class ClinicalTestRecommender:
    """Rule-based clinical test recommendation system"""
    
    def __init__(self):
        self.test_mapping = {
            'molecular_markers': {
                'features': ['ER Status', 'PR Status', 'HER2 Status', 'Pam50 + Claudin-low subtype'],
                'tests': ['IHC for ER/PR', 'HER2 FISH confirmation', 'PAM50 gene expression assay']
            },
            'pathology': {
                'features': ['Neoplasm Histologic Grade', 'Cellularity', 'Tumor Other Histologic Subtype'],
                'tests': ['Core needle biopsy', 'Ki-67 proliferation index', 'Pathology review']
            },
            'imaging': {
                'features': ['Tumor Size', 'Tumor Stage', 'Lymph nodes examined positive'],
                'tests': ['Mammography', 'Breast MRI', 'CT chest/abdomen/pelvis', 'Bone scan']
            },
            'genomics': {
                'features': ['Mutation Count', '3-Gene classifier subtype', 'Integrative Cluster'],
                'tests': ['Oncotype DX', 'MammaPrint', 'FoundationOne CDx', 'Comprehensive genomic profiling']
            }
        }
    
    def recommend_tests(self, shap_explanation, risk_trajectory, current_features):
        """
        Generate test recommendations based on:
        - High-impact features from SHAP
        - Rising risk trajectory
        - Current feature values
        """
        recommendations = {
            'immediate': [],
            'short_term': [],
            'monitoring': []
        }
        
        # Extract top influential features
        top_features = [f['name'] for f in shap_explanation['features'][:10]]
        
        # Check each test category
        for category, info in self.test_mapping.items():
            relevant_features = [f for f in top_features if any(marker in f for marker in info['features'])]
            
            if relevant_features:
                for test in info['tests']:
                    # Determine priority based on SHAP values
                    max_shap = max([f['shap_value'] for f in shap_explanation['features'] 
                                   if f['name'] in relevant_features], default=0)
                    
                    if abs(max_shap) > 0.5:
                        priority = 'immediate'
                    elif abs(max_shap) > 0.2:
                        priority = 'short_term'
                    else:
                        priority = 'monitoring'
                    
                    recommendations[priority].append({
                        'test': test,
                        'category': category,
                        'reason': f"High impact on risk from {', '.join(relevant_features)}",
                        'shap_magnitude': float(abs(max_shap))
                    })
        
        # Add time-based recommendations if risk is rising
        if len(risk_trajectory) > 1:
            risk_increase = risk_trajectory[-1] - risk_trajectory[0]
            if risk_increase > 0.1:  # 10% increase
                recommendations['immediate'].append({
                    'test': 'Comprehensive re-staging workup',
                    'category': 'surveillance',
                    'reason': f'Risk trajectory shows {risk_increase*100:.1f}% increase',
                    'shap_magnitude': None
                })
        
        return recommendations


class SurvivalImprovementAnalyzer:
    """Counterfactual analysis for survival improvement"""
    
    def __init__(self, survival_model, preprocessor, feature_names):
        self.survival_model = survival_model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
    
    def analyze_interventions(self, X_original, original_features, times=[12, 36, 60]):
        """
        Simulate interventions and quantify survival benefit
        """
        interventions = []
        
        # Define modifiable features and their interventions
        modifiable_features = {
            'Chemotherapy': {'from': 'No', 'to': 'Yes', 'name': 'chemotherapy'},
            'Hormone Therapy': {'from': 'No', 'to': 'Yes', 'name': 'hormone_therapy'},
            'Radio Therapy': {'from': 'No', 'to': 'Yes', 'name': 'radiotherapy'}
        }
        
        # Baseline survival
        baseline_survival = self.survival_model.predict_survival_function(X_original, times)[0]
        
        # Test each intervention
        for feature, intervention in modifiable_features.items():
            if feature in original_features:
                current_value = original_features[feature]
                
                # Only suggest if not already receiving treatment
                if str(current_value).lower() in ['no', '0', 'false']:
                    # Modify feature
                    modified_features = original_features.copy()
                    modified_features[feature] = intervention['to']
                    
                    # Transform
                    modified_df = pd.DataFrame([modified_features])
                    X_modified = self.preprocessor.transform(modified_df)
                    
                    # Predict new survival
                    modified_survival = self.survival_model.predict_survival_function(X_modified, times)[0]
                    
                    # Calculate benefit
                    benefits = [(mod - base) * 100 for mod, base in zip(modified_survival, baseline_survival)]
                    
                    interventions.append({
                        'intervention': intervention['name'],
                        'feature': feature,
                        'baseline_survival': baseline_survival.tolist(),
                        'modified_survival': modified_survival.tolist(),
                        'benefit_at_1yr': benefits[0] if len(benefits) > 0 else 0,
                        'benefit_at_3yr': benefits[1] if len(benefits) > 1 else 0,
                        'benefit_at_5yr': benefits[2] if len(benefits) > 2 else 0,
                        'time_points': times
                    })
        
        return interventions


class BreastCancerSurvivalPredictor:
    """Main survival analysis system"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
        # Models
        self.os_model = None  # Overall survival
        self.rfs_model = None  # Relapse-free survival
        
        # Preprocessing
        self.preprocessor = SurvivalAnalysisPreprocessor()
        
        # Explainability
        self.os_explainer = None
        self.rfs_explainer = None
        
        # Clinical tools
        self.test_recommender = ClinicalTestRecommender()
        self.improvement_analyzer = None
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.X_train_transformed = None
        self.X_test_transformed = None
        
    def load_and_prepare_data(self):
        """Load data and prepare for survival analysis"""
        print("=" * 80)
        print("LOADING METABRIC DATASET FOR SURVIVAL ANALYSIS")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"\nDataset shape: {self.df.shape}")
        
        # Define feature columns (exclude outcomes)
        outcome_columns = [
            'Overall Survival (Months)',
            'Overall Survival Status',
            'Relapse Free Status (Months)',
            'Relapse Free Status',
            "Patient's Vital Status",
            'Patient ID'
        ]
        
        feature_columns = [col for col in self.df.columns if col not in outcome_columns]
        
        print(f"Feature columns: {len(feature_columns)}")
        
        # Prepare data
        X_full = self.df[feature_columns].copy()
        
        time_os = self.df['Overall Survival (Months)']
        event_os = self.df['Overall Survival Status'].apply(
            lambda x: 1 if str(x).lower() in ['deceased', '1:deceased', '1'] else 0
        )
        
        time_rfs = self.df['Relapse Free Status (Months)']
        event_rfs = self.df['Relapse Free Status'].apply(
            lambda x: 1 if str(x).lower() in ['recurred', '1:recurred', '1'] else 0
        )
        
        # Create valid mask
        valid_os = time_os.notna() & event_os.notna()
        valid_rfs = time_rfs.notna() & event_rfs.notna()
        valid_idx = valid_os & valid_rfs
        
        print(f"\nValid samples: {valid_idx.sum()}")
        print(f"Overall Survival - Events: {event_os[valid_idx].sum()} ({event_os[valid_idx].mean()*100:.1f}%)")
        print(f"Relapse-Free Survival - Events: {event_rfs[valid_idx].sum()} ({event_rfs[valid_idx].mean()*100:.1f}%)")
        
        # Filter everything using the same boolean mask
        X = X_full[valid_idx].reset_index(drop=True)  # Important: reset index!
        
        self.time_os = time_os[valid_idx].values
        self.event_os = event_os[valid_idx].values
        self.time_rfs = time_rfs[valid_idx].values
        self.event_rfs = event_rfs[valid_idx].values
        
        # Now split - this will give clean sequential indices
        self.X_train, self.X_test = train_test_split(
            X, test_size=0.2, random_state=42, stratify=event_os[valid_idx]  # optional: stratify
        )
        
        # Get boolean masks for train/test split
        train_indices = self.X_train.index
        test_indices = self.X_test.index
        
        # Assign train/test splits using the new clean indices
        self.time_os_train = self.time_os[train_indices]
        self.event_os_train = self.event_os[train_indices]
        self.time_os_test = self.time_os[test_indices]
        self.event_os_test = self.event_os[test_indices]
        
        self.time_rfs_train = self.time_rfs[train_indices]
        self.event_rfs_train = self.event_rfs[train_indices]
        self.time_rfs_test = self.time_rfs[test_indices]
        self.event_rfs_test = self.event_rfs[test_indices]
        
        # Fit preprocessor on training data only
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)
        
        print(f"\nTransformed features: {len(self.preprocessor.feature_names)}")
        print(f"Training set: {self.X_train_transformed.shape}")
        print(f"Test set: {self.X_test_transformed.shape}")
        
        return self.X_train, self.X_test
    
    def train_survival_models(self):
        """Train Cox survival models"""
        print("\n" + "=" * 80)
        print("TRAINING SURVIVAL MODELS")
        print("=" * 80)
        
        # Overall Survival Model
        print("\n" + "=" * 60)
        print("1. OVERALL SURVIVAL MODEL")
        print("=" * 60)
        
        self.os_model = XGBoostCoxSurvivalModel(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1
        )
        
        self.os_model.fit(
            self.X_train_transformed,
            pd.Series(self.time_os_train),
            pd.Series(self.event_os_train)
        )
        
        c_index_train = self.os_model.concordance_index(
            self.X_train_transformed,
            pd.Series(self.time_os_train),
            pd.Series(self.event_os_train)
        )
        
        c_index_test = self.os_model.concordance_index(
            self.X_test_transformed,
            pd.Series(self.time_os_test),
            pd.Series(self.event_os_test)
        )
        
        print(f"Training C-index: {c_index_train:.4f}")
        print(f"Test C-index: {c_index_test:.4f}")
        
        # Relapse-Free Survival Model
        print("\n" + "=" * 60)
        print("2. RELAPSE-FREE SURVIVAL MODEL")
        print("=" * 60)
        
        self.rfs_model = XGBoostCoxSurvivalModel(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1
        )
        
        self.rfs_model.fit(
            self.X_train_transformed,
            pd.Series(self.time_rfs_train),
            pd.Series(self.event_rfs_train)
        )
        
        c_index_train_rfs = self.rfs_model.concordance_index(
            self.X_train_transformed,
            pd.Series(self.time_rfs_train),
            pd.Series(self.event_rfs_train)
        )
        
        c_index_test_rfs = self.rfs_model.concordance_index(
            self.X_test_transformed,
            pd.Series(self.time_rfs_test),
            pd.Series(self.event_rfs_test)
        )
        
        print(f"Training C-index: {c_index_train_rfs:.4f}")
        print(f"Test C-index: {c_index_test_rfs:.4f}")
        
        print("\n✓ Survival models trained successfully!")
    
    def build_explainers(self):
        """Build SHAP explainers for both models"""
        print("\n" + "=" * 80)
        print("BUILDING SHAP EXPLAINERS")
        print("=" * 80)
        
        self.os_explainer = SurvivalExplainer(
        self.os_model,
        self.preprocessor.feature_names
        )

        self.rfs_explainer = SurvivalExplainer(
        self.rfs_model,
        self.preprocessor.feature_names
        )

        
        print("✓ SHAP explainers built successfully!")
    
    def build_improvement_analyzer(self):
        """Build survival improvement analyzer"""
        self.improvement_analyzer = SurvivalImprovementAnalyzer(
            self.os_model,
            self.preprocessor,
            self.preprocessor.feature_names
        )
        
        print("✓ Survival improvement analyzer built!")
    
    def save_all_artifacts(self, output_dir='models/'):
        """Save all models and components"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("SAVING ALL ARTIFACTS")
        print("=" * 80)
        
        # Save models
        joblib.dump(self.os_model, f'{output_dir}os_survival_model.pkl')
        joblib.dump(self.rfs_model, f'{output_dir}rfs_survival_model.pkl')
        print("✓ Saved survival models")
        
        # Save preprocessor
        joblib.dump(self.preprocessor, f'{output_dir}preprocessor.pkl')
        print("✓ Saved preprocessor")
        
        # Save explainers
        joblib.dump(self.os_explainer, f'{output_dir}os_explainer.pkl')
        joblib.dump(self.rfs_explainer, f'{output_dir}rfs_explainer.pkl')
        print("✓ Saved explainers")
        
        # Save test recommender
        joblib.dump(self.test_recommender, f'{output_dir}test_recommender.pkl')
        print("✓ Saved test recommender")
        
        # Save improvement analyzer
        joblib.dump(self.improvement_analyzer, f'{output_dir}improvement_analyzer.pkl')
        print("✓ Saved improvement analyzer")
        
        print(f"\nAll artifacts saved to {output_dir}")


def main():
    """Main training pipeline"""
    print("\n" + "=" * 80)
    print("BREAST CANCER SURVIVAL ANALYSIS SYSTEM")
    print("Cox Proportional Hazards Models with XGBoost")
    print("=" * 80)
    
    # Initialize
    data_path = r'K:\HackRush\archive (4)\Breast Cancer METABRIC.csv'  # Update with your path
    predictor = BreastCancerSurvivalPredictor(data_path)
    
    # Load and prepare
    predictor.load_and_prepare_data()
    
    # Train models
    predictor.train_survival_models()
    
    # Build explainers
    predictor.build_explainers()
    
    # Build improvement analyzer
    predictor.build_improvement_analyzer()
    
    # Save everything
    predictor.save_all_artifacts()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review C-index metrics above")
    print("2. Run Streamlit app for predictions")
    print("3. Generate survival curves and explanations")
    print("=" * 80)


if __name__ == "__main__":
    main()