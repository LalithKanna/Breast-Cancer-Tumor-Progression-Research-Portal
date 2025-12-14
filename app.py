from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import torch
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

from transformer_model import TumorProgressionTransformer
from survival_model import XGBoostCoxSurvivalModel
from preprocessing import SurvivalAnalysisPreprocessor
from explainability import SurvivalExplainer
from clinical_rules import ClinicalTestRecommender
from improvement import SurvivalImprovementAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['TEMPLATES_AUTO_RELOAD'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Add timeout protection
import signal
from functools import wraps

def timeout_handler(signum, frame):
    raise TimeoutError("Request processing timeout")

def with_timeout(seconds=120):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set alarm for timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel alarm
            return result
        return wrapper
    return decorator

# ======================================================
# Utilities
# ======================================================

def align_patient_schema(patient_df, preprocessor):
    expected_cols = (
        preprocessor.numerical_features +
        preprocessor.categorical_features
    )
    aligned = patient_df.copy()
    for col in expected_cols:
        if col not in aligned.columns:
            if col in preprocessor.numerical_features:
                aligned[col] = np.nan
            else:
                aligned[col] = "Unknown"
    return aligned[expected_cols]

def load_sequence_from_bytes(file_bytes):
    try:
        seq = np.load(io.BytesIO(file_bytes), allow_pickle=True)
        if seq.ndim != 3:
            raise ValueError("Expected shape (N, T, F)")
        return torch.tensor(seq, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Invalid sequence file format: {str(e)}")

def create_survival_plot(times, survival, title):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, survival, linewidth=2, color='#1f77b4', marker='o', markersize=6)
        ax.set_xlabel("Months", fontsize=12)
        ax.set_ylabel("Survival Probability", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([times[0] - 2, times[-1] + 2])
        
        # Add percentage labels
        for i, (t, s) in enumerate(zip(times, survival)):
            ax.annotate(f'{s:.1%}', xy=(t, s), xytext=(0, 10), 
                       textcoords='offset points', ha='center', fontsize=9)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        raise

def create_shap_plot(df_exp):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = df_exp["shap_value"].apply(
            lambda x: "#d62728" if x > 0 else "#2ca02c"
        )
        bars = ax.barh(df_exp["name"], df_exp["shap_value"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (Impact on Mortality Hazard)", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title("Feature Contributions to Mortality Risk", fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else width
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}',
                   ha='left' if width > 0 else 'right',
                   va='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Error creating SHAP plot: {str(e)}")
        raise

def estimate_risk_time(times, survival_probs, threshold):
    for t, s in zip(times, survival_probs):
        if (1 - s) >= threshold:
            return int(t)
    return None

def simulate_improvement(model, X_original, feature_names, shap_df, times, max_features=3):
    improvements = []
    candidates = shap_df[
        shap_df["shap_value"] > 0
    ].sort_values("shap_value", ascending=False).head(max_features)
    
    for _, row in candidates.iterrows():
        feature = row["name"]
        if feature not in feature_names:
            continue
        
        idx = feature_names.index(feature)
        X_cf = X_original.copy()
        X_cf[0, idx] *= 0.9
        
        original_surv = model.predict_survival_function(X_original, times)[0]
        new_surv = model.predict_survival_function(X_cf, times)[0]
        gain = float(new_surv[-1] - original_surv[-1])
        
        improvements.append({
            "Feature": feature,
            "Hypothetical Adjustment": "10% reduction",
            "5-Year Survival Gain": f"{gain:+.2%}"
        })
    
    return improvements

# ======================================================
# Load models
# ======================================================

def load_models():
    print("Loading models...", flush=True)
    models = {}
    
    try:
        # Transformer
        transformer = TumorProgressionTransformer(
            feature_dim=20,
            d_model=128,
            n_heads=4,
            n_layers=4
        )
        transformer.load_state_dict(
            torch.load("models/tumor_progression_transformer.pt", map_location="cpu")
        )
        transformer.eval()
        models["transformer"] = transformer
        print("✓ Transformer loaded", flush=True)
        
        # Survival models
        models["os_model"] = joblib.load("models/os_survival_model.pkl")
        print("✓ OS model loaded", flush=True)
        
        models["rfs_model"] = joblib.load("models/rfs_survival_model.pkl")
        print("✓ RFS model loaded", flush=True)
        
        # Preprocessor
        preprocessor = joblib.load("models/preprocessor.pkl")
        models["preprocessor"] = preprocessor
        print("✓ Preprocessor loaded", flush=True)
        
        # SHAP background
        background_df = pd.DataFrame({
            col: [np.nan] if col in preprocessor.numerical_features else ["Unknown"]
            for col in preprocessor.numerical_features + preprocessor.categorical_features
        })
        background_X = preprocessor.transform(background_df)
        
        models["os_explainer"] = SurvivalExplainer(
            models["os_model"],
            preprocessor.feature_names,
            background_X
        )
        print("✓ Explainer loaded", flush=True)
        
        models["test_recommender"] = joblib.load("models/test_recommender.pkl")
        print("✓ Test recommender loaded", flush=True)
        
        models["improvement_analyzer"] = joblib.load("models/improvement_analyzer.pkl")
        print("✓ Improvement analyzer loaded", flush=True)
        
        print("All models loaded successfully!", flush=True)
        return models
    except Exception as e:
        print(f"ERROR loading models: {str(e)}", flush=True)
        raise

# Load models at startup (only once)
MODELS = None
try:
    MODELS = load_models()
except Exception as e:
    print(f"CRITICAL: Failed to load models: {str(e)}")
    sys.exit(1)

# ======================================================
# Routes
# ======================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    import time
    start_time = time.time()
    print("\n" + "="*70, flush=True)
    print(f"=== New analysis request received at {time.strftime('%H:%M:%S')} ===", flush=True)
    print("="*70, flush=True)
    
    try:
        # Validate files
        if 'sequence_file' not in request.files or 'patient_file' not in request.files:
            print("ERROR: Missing files in request", flush=True)
            return jsonify({'error': 'Both sequence and patient files are required'}), 400
        
        seq_file = request.files['sequence_file']
        patient_file = request.files['patient_file']
        
        if seq_file.filename == '' or patient_file.filename == '':
            print("ERROR: Empty file names", flush=True)
            return jsonify({'error': 'Empty file names'}), 400
        
        print(f"Files received: {seq_file.filename}, {patient_file.filename}", flush=True)
        
        # Load sequence data
        print("Loading sequence data...", flush=True)
        seq_bytes = seq_file.read()
        X_seq = load_sequence_from_bytes(seq_bytes)
        print(f"Sequence shape: {X_seq.shape}", flush=True)
        
        # Load patient data
        print("Loading patient data...", flush=True)
        patient_df = pd.read_csv(patient_file)
        print(f"Patient data shape: {patient_df.shape}", flush=True)
        
        # Transformer prediction
        print("Running transformer prediction...", flush=True)
        with torch.no_grad():
            predictions = MODELS["transformer"](X_seq).numpy()[0]
            death_prob, survival_prob, recurrence_prob = predictions
        print(f"Predictions: death={death_prob:.3f}, survival={survival_prob:.3f}, recurrence={recurrence_prob:.3f}", flush=True)
        
        # Prepare patient data
        print("Preparing patient data...", flush=True)
        patient_df_aligned = align_patient_schema(patient_df, MODELS["preprocessor"])
        X_proc = MODELS["preprocessor"].transform(patient_df_aligned)
        print(f"Processed data shape: {X_proc.shape}", flush=True)
        
        # Survival predictions
        print("Generating survival curves...", flush=True)
        times = np.array([12, 24, 36, 48, 60])
        os_surv = MODELS["os_model"].predict_survival_function(X_proc, times)[0]
        rfs_surv = MODELS["rfs_model"].predict_survival_function(X_proc, times)[0]
        print(f"OS survival: {os_surv}", flush=True)
        print(f"RFS survival: {rfs_surv}", flush=True)
        
        # Create plots
        print("Creating plots...", flush=True)
        os_plot = create_survival_plot(times, os_surv, "Overall Survival")
        rfs_plot = create_survival_plot(times, rfs_surv, "Relapse-Free Survival")
        
        # Risk estimates
        print("Calculating risk estimates...", flush=True)
        risk_data = []
        for label, t in [("1 Year", 12), ("2 Years", 24), ("5 Years", 60)]:
            idx = np.where(times == t)[0][0]
            risk_data.append({
                "time_horizon": label,
                "survival_prob": f"{os_surv[idx]:.2%}",
                "death_prob": f"{1 - os_surv[idx]:.2%}",
                "recurrence_prob": f"{1 - rfs_surv[idx]:.2%}"
            })
        
        # Risk timing
        print("Calculating risk timing...", flush=True)
        early_risk = estimate_risk_time(times, os_surv, 0.25)
        median_risk = estimate_risk_time(times, os_surv, 0.50)
        
        # Explainability
        print("Generating SHAP explanations...", flush=True)
        explanation = MODELS["os_explainer"].explain_individual(X_proc[0])
        df_exp = pd.DataFrame(explanation["features"])
        
        required_cols = ["name", "value", "shap_value"]
        for col in required_cols:
            if col not in df_exp.columns:
                return jsonify({'error': f'Missing explainability field: {col}'}), 500
        
        df_exp["shap_value"] = df_exp["shap_value"].astype(float)
        df_exp["abs_shap"] = df_exp["shap_value"].abs()
        df_exp = df_exp.sort_values("abs_shap", ascending=False).head(10)
        
        print("Creating SHAP plot...", flush=True)
        shap_plot = create_shap_plot(df_exp.sort_values("shap_value", ascending=True))
        
        feature_contributions = df_exp[["name", "value", "shap_value"]].to_dict('records')
        for item in feature_contributions:
            item["shap_value_formatted"] = f"{item['shap_value']:+.4f}"
        
        # Improvement analysis
        print("Running improvement analysis...", flush=True)
        improvements = simulate_improvement(
            model=MODELS["os_model"],
            X_original=X_proc,
            feature_names=MODELS["preprocessor"].feature_names,
            shap_df=df_exp,
            times=times
        )
        
        # Prepare response
        response = {
            'success': True,
            'transformer_predictions': {
                'death_prob': f"{death_prob:.2%}",
                'survival_prob': f"{survival_prob:.2%}",
                'recurrence_prob': f"{recurrence_prob:.2%}"
            },
            'survival_plots': {
                'os_plot': os_plot,
                'rfs_plot': rfs_plot
            },
            'risk_estimates': risk_data,
            'risk_timing': {
                'early_risk': f"{early_risk//12} years" if early_risk else "Not reached within 5 years",
                'early_risk_status': 'warning' if early_risk else 'success',
                'median_risk': f"{median_risk//12} years" if median_risk else "Not reached within 5 years",
                'median_risk_status': 'error' if median_risk else 'success'
            },
            'explainability': {
                'features': feature_contributions,
                'shap_plot': shap_plot
            },
            'improvements': improvements
        }
        
        print("Analysis completed successfully!", flush=True)
        elapsed = time.time() - start_time
        print(f"Total processing time: {elapsed:.2f} seconds", flush=True)
        print("="*70, flush=True)
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR during analysis: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'models_loaded': MODELS is not None})

if __name__ == '__main__':
    print("="*60)
    print("Starting Flask Application")
    print("Server: http://localhost:5000")
    print("="*60)
    
    # IMPORTANT: Disable reloader to prevent interruptions during analysis
    app.run(
        debug=False,  # Changed to False to prevent restarts
        host='0.0.0.0', 
        port=5000,
        use_reloader=False,  # Completely disable reloader
        threaded=True  # Enable threading for better performance

    )
