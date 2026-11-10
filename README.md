# PMGSY Scheme Classifier

**Automatic classification of PMGSY road/bridge projects into their correct scheme using Machine Learning, deployed via IBM Watson Machine Learning (Lite tier).**

---

## Problem Statement

The Pradhan Mantri Gram Sadak Yojana (PMGSY) runs multiple schemes (PMGSY-I, PMGSY-II, RCPLWEA, PMGSY-III, MNREGA). Manually tagging thousands of projects into the correct scheme is error-prone and unscalable. This application uses a Random Forest classifier to automatically predict the scheme from a project's physical and financial characteristics.

---

## Features

| Category | Details |
|----------|---------|
| **ML Model** | Random Forest (300 estimators, balanced class weights) |
| **Input Features** | Road length, bridge length, DPR cost, revised cost, expenditure, population, terrain, surface, state, year |
| **Target** | `PMGSY_SCHEME` (5 classes) |
| **IBM Cloud** | Watson Machine Learning Lite — model storage + online scoring endpoint |
| **UI** | Streamlit — Data Explorer · Train · Predict · Deploy |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure IBM Cloud credentials (optional — for cloud deployment)
```bash
cp .env.example .env
# Edit .env and fill in:
# IBM_API_KEY, IBM_WML_URL, IBM_WML_SPACE_ID
```

### 3. Train the model
```bash
# Using synthetic data (default)
python train.py

# Using the AI Kosh CSV
python train.py --csv path/to/pmgsy_data.csv
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## IBM Cloud Setup (Lite — Free Tier)

Only **3 credentials** are required:

| Credential | Where to find it |
|-----------|-----------------|
| `IBM_API_KEY` | IBM Cloud → Manage → Access (IAM) → API keys → Create |
| `IBM_WML_URL` | Watson ML service → Credentials (region URL) |
| `IBM_WML_SPACE_ID` | Watson Studio → Deployment Spaces → Space GUID |

**Cost:** IBM WML Lite tier is free. The model is small and stays well within free capacity unit limits.

---

## Project Structure

```
prblm35/
├── app.py              # Streamlit application (main UI)
├── data_utils.py       # Data loading, feature definitions, preprocessing
├── model_utils.py      # Model training, evaluation, local inference
├── ibm_wml_utils.py    # IBM Watson ML deployment & scoring
├── train.py            # CLI training script
├── requirements.txt
├── .env.example        # Credential template
└── README.md
```

---

## Data Source

AI Kosh Dataset — Pradhan Mantri Gram Sadak Yojana (PMGSY)  
🔗 https://aikosh.indiaai.gov.in/web/datasets/details/pradhan_mantri_gram_sadak_yojna_pmgsy.html

If the CSV is not available, the app uses a built-in **synthetic dataset** (8,000 records) that exactly mirrors the column structure and realistic value distributions of the official data.

---

## Model Performance (Synthetic Data Baseline)

| Metric | Value |
|--------|-------|
| Accuracy | ~97–99% |
| Weighted F1 | ~0.97–0.99 |
| Inference | < 5ms per record |

Results on the real AI Kosh dataset will vary based on data quality and label consistency.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Streamlit Application                    │
│   ┌────────────┐   ┌────────────┐   ┌────────────────┐  │
│   │ Data       │   │ Train      │   │ Predict        │  │
│   │ Explorer   │   │ RF Model   │   │ (Local / WML)  │  │
│   └────────────┘   └─────┬──────┘   └───────┬────────┘  │
└─────────────────────────┼───────────────────┼────────────┘
                          │ deploy            │ score
                  ┌───────▼───────────────────▼────────┐
                  │   IBM Watson Machine Learning       │
                  │   (Lite — Free Tier)                │
                  │   REST Scoring Endpoint             │
                  └────────────────────────────────────┘
```
