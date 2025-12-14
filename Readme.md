# 🧠 Breast Cancer Tumor Progression & Survival Analysis System

![Python](https://img.shields.io/badge/Python-3.9%2B-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/Flask-backend-000000?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-transformer-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-cox%20survival-0074A6?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-data--processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-preprocessing-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/Research-Only-critical?style=for-the-badge)

> **⚠️ CRITICAL DISCLAIMER**
> 
> This project is strictly for **research and educational purposes only**. It is **NOT** a clinical decision-support system and must **NEVER** be used for:
> - Medical diagnosis
> - Treatment recommendations
> - Clinical decision-making
> - Patient care decisions
> 
> All predictions are model-based estimates for research purposes only and require extensive validation before any clinical consideration.

---

## 📘 Table of Contents

- [Overview](#-overview)
- [System Architecture](#️-system-architecture)
- [Key Features](#-key-features)
- [Data Requirements](#-data-requirements)
- [Model Components](#-model-components)
- [Training Pipeline](#️-training-pipeline)
- [Web Application](#-web-application)
- [API Endpoints](#-api-endpoints)
- [Explainability & Counterfactuals](#-explainability--counterfactuals)
- [Installation & Setup](#️-installation--setup)
- [Running the System](#️-running-the-system)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Evaluation Metrics](#-evaluation-metrics)
- [Limitations & Research Notes](#️-limitations--research-notes)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🩺 Overview

Breast cancer progression is inherently **temporal and heterogeneous**, requiring sophisticated analytical approaches that can capture:

- **Temporal dynamics** of tumor evolution
- **Individual patient variability** in disease trajectory
- **Complex interactions** between clinical and molecular features
- **Risk stratification** for personalized treatment planning

This project addresses these challenges by implementing an integrated machine learning system that:

1. **Models tumor evolution** over time using a **Transformer-based neural network**
2. **Predicts survival outcomes** including:
   - Overall Survival (OS)
   - Relapse-Free Survival (RFS)
3. **Provides interpretable results** through:
   - Individual-level feature importance (SHAP values)
   - Counterfactual analysis for intervention planning
4. **Delivers insights** through an intuitive **Flask-based web research portal**

### Research Context

Traditional survival models often treat patient data as static snapshots, missing critical temporal patterns in disease progression. This system combines:

- **Deep learning** for temporal pattern recognition
- **Classical survival analysis** for risk quantification
- **Explainable AI** for clinical interpretability

---

## ⚙️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          Longitudinal Tumor Sequences (.npy)                │
│               Shape: (N, T, F)                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Transformer Progression Model                       │
│         - Multi-head self-attention                         │
│         - Temporal feature aggregation                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│    Aggregate Risk Predictions                               │
│    - Death probability                                      │
│    - Survival probability                                   │
│    - Recurrence probability                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│       Static Patient Clinical Data (.csv)                   │
│       - Demographics                                        │
│       - Clinical features                                   │
│       - Treatment history                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Preprocessing Pipeline                              │
│         - Imputation (median/mode)                          │
│         - Scaling (standardization)                         │
│         - Encoding (one-hot)                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│      XGBoost Cox Survival Models                            │
│      - Overall Survival (OS)                                │
│      - Relapse-Free Survival (RFS)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ Survival Curves  │    │ SHAP Values      │
│ - 1-year         │    │ - Feature impact │
│ - 2-year         │    │ - Risk drivers   │
│ - 5-year         │    │ - Protectives    │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│      Counterfactual Survival Analysis                       │
│      - Intervention simulation                              │
│      - Survival improvement estimation                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Web Research Portal                              │
│            - Interactive visualizations                     │
│            - Tabular reports                                │
│            - Export capabilities                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

### 1. 🧬 Tumor Progression Modeling

- **Architecture**: Transformer-based neural network with multi-head self-attention
- **Input**: Longitudinal tumor feature sequences
- **Processing**: Captures temporal dependencies and progression patterns
- **Output**: Aggregate risk probabilities
  - Death risk
  - Survival probability
  - Recurrence likelihood

**Technical Details:**
- Position encoding for temporal awareness
- Multi-head attention for feature interaction modeling
- Fully connected layers for risk aggregation

### 2. ⏱️ Survival Analysis

- **Algorithm**: XGBoost with Cox Proportional Hazards objective
- **Models**: 
  - Overall Survival (OS) - Time to death from any cause
  - Relapse-Free Survival (RFS) - Time to disease recurrence
- **Predictions**: Survival probabilities at clinically relevant timepoints
  - 1-year survival
  - 2-year survival
  - 5-year survival

**Technical Details:**
- Breslow baseline hazard estimation
- Partial likelihood maximization
- Concordance Index (C-index) optimization

### 3. 🔍 Explainability

- **Method**: Native XGBoost SHAP (SHapley Additive exPlanations)
- **Granularity**: Individual patient-level feature contributions
- **Interpretation**: 
  - Positive SHAP values → Increased mortality risk
  - Negative SHAP values → Protective effect
- **Visualization**: Interactive SHAP bar plots and summary statistics

**What SHAP Values Tell You:**
- Which features most strongly affect a patient's risk
- Whether each feature increases or decreases risk
- Relative importance of different clinical factors

### 4. 🔮 Counterfactual Analysis

- **Purpose**: Simulate hypothetical clinical interventions
- **Methodology**: 
  - Identify top risk-driving features for each patient
  - Simulate 10% reduction in modifiable risk factors
  - Re-calculate survival probabilities
- **Output**: Estimated survival improvement (percentage points)

**Use Cases:**
- Treatment prioritization
- Intervention planning
- Risk factor modification strategies

### 5. 🌐 Web-Based Research Portal

- **Interface**: Intuitive file upload and analysis dashboard
- **Capabilities**:
  - Upload tumor sequences and patient data
  - Generate interactive survival plots
  - View tabular risk estimates
  - Access explainability dashboards
  - Export results for further analysis

---

## 📊 Data Requirements

### 1. Longitudinal Tumor Sequence (`.npy` format)

**Structure:**
```python
Shape: (N, T, F)
  N: Number of samples (typically 1 for single patient analysis)
  T: Number of time steps (temporal observations)
  F: Number of tumor features per time step
```

**Example Features:**
- Tumor size measurements
- Biomarker levels (e.g., Ki-67, ER, PR, HER2)
- Cellularity
- Mitotic count
- Gene expression profiles

**Format Requirements:**
- NumPy binary format (`.npy`)
- Float32 or Float64 data type
- No missing values (or pre-imputed)

**Example Creation:**
```python
import numpy as np

# Example: 1 patient, 10 time steps, 50 features
sequence = np.random.randn(1, 10, 50).astype(np.float32)
np.save('patient_sequence.npy', sequence)
```

### 2. Patient Clinical Snapshot (`.csv` format)

**Structure:**
- Single row representing one patient
- Columns aligned with training schema

**Required/Recommended Columns:**
- **Demographics**: Age at diagnosis, ethnicity
- **Tumor Characteristics**: 
  - Tumor size
  - Grade
  - Stage
  - Histological type
- **Biomarkers**: ER status, PR status, HER2 status
- **Treatment**: Chemotherapy, hormone therapy, radiotherapy
- **Outcomes** (if available): 
  - Overall Survival (months)
  - Overall Survival Status (0=alive, 1=deceased)
  - Relapse Free Survival (months)
  - Relapse Free Status (0=no relapse, 1=relapse)

**Example:**
```csv
Age_at_diagnosis,Tumor_size,Grade,ER_status,PR_status,HER2_status,Chemotherapy,Overall_Survival_months,Overall_Survival_status
55,2.5,3,Positive,Positive,Negative,Yes,60,0
```

**Handling Missing Data:**
- Missing features are automatically imputed
- Numerical: Median imputation
- Categorical: Mode imputation

### 3. Training Dataset

**Recommended Source:**
- **METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium)
- Publicly available through cBioPortal
- Contains ~2,000 breast cancer patients with comprehensive clinical and genomic data

**Alternative Sources:**
- TCGA (The Cancer Genome Atlas) - Breast cancer cohort
- Custom institutional datasets (requires proper de-identification)

**Essential Training Columns:**
- Patient identifier
- Survival time (months)
- Survival status (event indicator)
- Clinical features (matching inference schema)

---

## 🧩 Model Components

### Transformer Model

**Architecture:**
```
Input: (batch, sequence_length, features)
  ↓
Positional Encoding
  ↓
Multi-Head Self-Attention (4-8 heads)
  ↓
Feed-Forward Network
  ↓
Layer Normalization
  ↓
Aggregation (mean pooling)
  ↓
Fully Connected Layers
  ↓
Output: (death_prob, survival_prob, recurrence_prob)
```

**Key Parameters:**
- Embedding dimension: 128-256
- Number of attention heads: 4-8
- Number of transformer layers: 2-4
- Dropout rate: 0.1-0.3

**Training:**
- Loss function: Cross-entropy (multi-class)
- Optimizer: Adam (learning rate: 1e-4)
- Batch size: 32-64
- Epochs: 50-100 with early stopping

### Survival Models

**XGBoost Cox Model Specifications:**

**Overall Survival (OS) Model:**
```python
XGBSurvival(
    objective='survival:cox',
    eval_metric='concordance',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Relapse-Free Survival (RFS) Model:**
- Same architecture as OS model
- Trained on different endpoint (recurrence instead of death)

**Baseline Hazard Estimation:**
- Method: Breslow estimator
- Computed from training data
- Used for absolute risk prediction

### Preprocessing Pipeline

**Numerical Features:**
1. **Imputation**: Median strategy
2. **Scaling**: StandardScaler (zero mean, unit variance)

**Categorical Features:**
1. **Imputation**: Most frequent category
2. **Encoding**: One-hot encoding (drop first category to avoid multicollinearity)

**Pipeline Structure:**
```python
ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), categorical_features)
    ]
)
```

---

## 🏗️ Training Pipeline

The training pipeline orchestrates the entire model development process:

### Pipeline Steps

1. **Data Loading & Validation**
   - Load training dataset
   - Verify required columns
   - Check data quality

2. **Feature Engineering**
   - Create derived features
   - Handle categorical variables
   - Transform numerical distributions

3. **Train-Test Split**
   - Stratified split by survival status
   - Typical ratio: 80% train, 20% test
   - Ensures balanced representation

4. **Preprocessing**
   - Fit transformers on training data only
   - Apply transformations to both sets

5. **Model Training**
   - Train OS Cox model
   - Train RFS Cox model
   - Validate on holdout set

6. **Explainer Construction**
   - Build SHAP TreeExplainer
   - Pre-compute expected values

7. **Artifact Serialization**
   - Save trained models
   - Persist preprocessor
   - Store explainers

### Running the Pipeline

```bash
python training_metabrick.py --data_path data/metabric.csv \
                            --output_dir models/ \
                            --test_size 0.2 \
                            --random_seed 42
```

**Command-line Arguments:**
- `--data_path`: Path to training CSV
- `--output_dir`: Directory for model artifacts
- `--test_size`: Fraction for test set (default: 0.2)
- `--random_seed`: Reproducibility seed (default: 42)

### Output Artifacts

Saved in `models/` directory:

```
models/
├── os_model.xgb              # Overall Survival Cox model
├── rfs_model.xgb             # Relapse-Free Survival Cox model
├── preprocessor.pkl          # Fitted preprocessing pipeline
├── shap_explainer.pkl        # SHAP TreeExplainer
├── improvement_analyzer.pkl  # Counterfactual analyzer
└── training_metadata.json    # Feature names, shapes, statistics
```

### Validation Metrics

The pipeline reports:
- **C-index** (Concordance Index): Discrimination ability
- **Brier Score**: Calibration at 1, 2, 5 years
- **Integrated Brier Score**: Overall calibration
- **Feature importance**: Top contributing variables

---

## 🌐 Web Application

### Frontend

**Technologies:**
- HTML5 for structure
- CSS3 for styling (responsive design)
- Vanilla JavaScript for interactivity

**Features:**
- Drag-and-drop file upload
- Real-time validation
- Progress indicators
- Interactive result exploration
- Responsive layout (desktop/tablet/mobile)

**User Workflow:**
1. Upload tumor sequence (`.npy`)
2. Upload patient data (`.csv`)
3. Click "Analyze"
4. View comprehensive results dashboard

### Backend

**Framework:** Flask (Python microframework)

**Core Capabilities:**
- Model loading and inference
- Dynamic plot generation (Matplotlib)
- Base64 image encoding for web display
- JSON response formatting
- Error handling and logging

**Key Routes:**
- `/` - Main application page
- `/analyze` - POST endpoint for analysis
- `/health` - Health check
- `/static/` - Static assets (CSS, JS, images)

**Performance Considerations:**
- Models loaded once at startup
- Caching for repeated analyses
- Async processing for large datasets (optional)

---

## 🔌 API Endpoints

### `POST /analyze`

**Purpose:** Execute complete survival analysis pipeline

**Request Format:**
```http
POST /analyze HTTP/1.1
Content-Type: multipart/form-data

sequence_file: [binary .npy file]
patient_file: [CSV file]
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/analyze \
  -F "sequence_file=@patient_sequence.npy" \
  -F "patient_file=@patient_data.csv"
```

**Response Format:**
```json
{
  "status": "success",
  "transformer_predictions": {
    "death_probability": 0.35,
    "survival_probability": 0.55,
    "recurrence_probability": 0.10
  },
  "survival_curves": {
    "os_curve": "data:image/png;base64,iVBORw0KG...",
    "rfs_curve": "data:image/png;base64,iVBORw0KG..."
  },
  "risk_estimates": {
    "os_1_year": 0.92,
    "os_2_year": 0.85,
    "os_5_year": 0.68,
    "rfs_1_year": 0.88,
    "rfs_2_year": 0.78,
    "rfs_5_year": 0.65
  },
  "risk_timing": {
    "highest_risk_period": "12-24 months",
    "risk_trajectory": "increasing"
  },
  "shap_explanation": {
    "top_risk_factors": [
      {"feature": "Tumor_size", "shap_value": 0.45},
      {"feature": "Grade", "shap_value": 0.32},
      {"feature": "Age_at_diagnosis", "shap_value": 0.18}
    ],
    "protective_factors": [
      {"feature": "ER_status_Positive", "shap_value": -0.25},
      {"feature": "Chemotherapy_Yes", "shap_value": -0.15}
    ],
    "plot": "data:image/png;base64,iVBORw0KG..."
  },
  "counterfactual_improvement": {
    "baseline_5y_survival": 0.68,
    "improved_5y_survival": 0.75,
    "absolute_improvement": 0.07,
    "relative_improvement": 10.3,
    "modified_features": ["Tumor_size", "Grade"]
  }
}
```

**Error Responses:**
```json
{
  "status": "error",
  "message": "Invalid file format: sequence_file must be .npy",
  "code": 400
}
```

### `GET /health`

**Purpose:** System health check

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

---

## 🧠 Explainability & Counterfactuals

### SHAP Value Interpretation

**What are SHAP Values?**

SHAP (SHapley Additive exPlanations) values are a game-theoretic approach to explain model predictions. Each feature receives a value indicating its contribution to the prediction.

**Interpretation Guide:**

| SHAP Value | Meaning | Clinical Interpretation |
|------------|---------|------------------------|
| +0.5 | Strong positive | Feature strongly increases mortality risk |
| +0.1 | Weak positive | Feature slightly increases risk |
| 0.0 | Neutral | Feature has no impact on risk |
| -0.1 | Weak negative | Feature slightly decreases risk (protective) |
| -0.5 | Strong negative | Feature strongly decreases mortality risk |

**Example:**
```python
Feature: Tumor_size
SHAP Value: +0.45

Interpretation: This patient's tumor size contributes substantially 
to increased mortality risk. The model predicts that without this 
feature, the patient's hazard would be e^(-0.45) ≈ 0.64x lower.
```

**Aggregation:**
- **Base value**: Average log-hazard in training population
- **SHAP sum**: Sum of all feature contributions
- **Final prediction**: Base value + SHAP sum = individual log-hazard

### Counterfactual Analysis

**Concept:**
"What would happen to this patient's survival if we could modify their risk factors?"

**Methodology:**

1. **Identify Top Risk Drivers**
   - Select features with highest positive SHAP values
   - Focus on potentially modifiable factors

2. **Simulate Intervention**
   - Reduce risk factors by 10% (or specified amount)
   - Keep protective factors unchanged

3. **Recompute Survival**
   - Generate new predictions with modified features
   - Calculate difference in 5-year survival probability

4. **Report Improvement**
   - Absolute improvement (percentage points)
   - Relative improvement (%)

**Example Output:**
```
Baseline 5-year survival: 68%

Top modifiable risk factors:
1. Tumor size (current: 3.5 cm)
2. Grade (current: 3)

Simulated intervention:
- Reduce tumor size to 3.15 cm (-10%)
- Reduce grade effect by 10%

Improved 5-year survival: 75%
Absolute gain: +7 percentage points
Relative gain: +10.3%
```

**Limitations:**
- Assumes factors are independently modifiable
- Does not account for biological feasibility
- Simplified causal model
- Should inform further research, not direct treatment

---

## 🛠️ Installation & Setup

### System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+ recommended)
- macOS 10.15+
- Windows 10+ (with WSL recommended)

**Hardware:**
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: 2GB for dependencies and models

**Software:**
- Python 3.9 or higher
- pip (Python package manager)
- Git (for cloning repository)

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/breast-cancer-survival-analysis.git
cd breast-cancer-survival-analysis
```

#### 2. Create Virtual Environment (Recommended)

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt contents:**
```
flask==3.0.0
numpy==1.24.3
pandas==2.0.3
torch==2.0.1
xgboost==2.0.0
scikit-learn==1.3.0
lifelines==0.27.7
matplotlib==3.7.2
seaborn==0.12.2
shap==0.42.1
joblib==1.3.1
scipy==1.11.1
```

#### 4. Download Training Data (Optional)

**METABRIC Dataset:**
```bash
# Visit cBioPortal
# https://www.cbioportal.org/study/summary?id=brca_metabric


```

#### 5. Train Models

```bash
python training_metabric.py 
```

**Expected output:**
```
Loading data...
Preprocessing features...
Training Overall Survival model...
  C-index: 0.72
Training Relapse-Free Survival model...
  C-index: 0.69
Building SHAP explainers...
Saving artifacts...
✓ Training complete! Models saved to models/
```

#### 6. Verify Installation

```bash
python -c "import torch, xgboost, sklearn, lifelines; print('All dependencies installed successfully!')"
```

---

## ▶️ Running the System

### Development Mode

**Start Flask server:**
```bash
python app.py
```

**Expected output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

**Access application:**
- Open browser to `http://localhost:5000`
- Upload files and run analysis

