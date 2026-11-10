"""
PMGSY Scheme Classification — Streamlit Application
IBM Cloud Lite + Watson Machine Learning Integration

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PMGSY Scheme Classifier",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1a472a 0%, #2d6a4f 50%, #1b4332 100%);
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 { color: #ffffff; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #d8f3dc; font-size: 1rem; margin: 5px 0 0; }
    .metric-card {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .scheme-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .ibm-badge {
        background: #0530ad;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🛣️ PMGSY Scheme Classifier</h1>
        <p>Pradhan Mantri Gram Sadak Yojana · Automatic Scheme Classification using ML
        &nbsp;&nbsp;<span class="ibm-badge">IBM Watson ML</span></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: IBM Credentials + Navigation ────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80,
    )
    st.markdown("### ⚙️ IBM Cloud Configuration")
    st.caption("Watson Machine Learning (Lite tier)")

    ibm_api_key = st.text_input(
        "IBM API Key *",
        type="password",
        value=os.getenv("IBM_API_KEY", ""),
        help="From IBM Cloud → Manage → Access (IAM) → API keys",
    )
    ibm_wml_url = st.text_input(
        "WML Instance URL *",
        value=os.getenv("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com"),
        help="Watson ML service URL (region-specific)",
    )
    ibm_space_id = st.text_input(
        "Deployment Space ID *",
        type="password",
        value=os.getenv("IBM_WML_SPACE_ID", ""),
        help="Watson Studio → Deployment Spaces → Space GUID",
    )

    # Persist to environment for downstream modules
    os.environ["IBM_API_KEY"] = ibm_api_key
    os.environ["IBM_WML_URL"] = ibm_wml_url
    os.environ["IBM_WML_SPACE_ID"] = ibm_space_id

    wml_ready = bool(ibm_api_key and ibm_space_id)
    if wml_ready:
        st.success("✅ IBM credentials configured")
    else:
        st.warning("⚠️ Enter IBM credentials to enable cloud deployment")

    st.divider()
    st.markdown("### 🗂️ Navigation")
    page = st.radio(
        "Select page",
        ["🏠 Overview", "📊 Data Explorer", "🤖 Train Model", "🔍 Predict", "☁️ IBM WML Deploy"],
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None
if "train_results" not in st.session_state:
    st.session_state.train_results = None
if "deployment_info" not in st.session_state:
    st.session_state.deployment_info = None
if "df_loaded" not in st.session_state:
    st.session_state.df_loaded = None


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 0 — Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("## About the Project")
        st.markdown(
            """
            The **Pradhan Mantri Gram Sadak Yojana (PMGSY)** is India's flagship rural
            road connectivity programme, providing all-weather roads to unconnected
            habitations.

            This application uses **Machine Learning** to automatically classify a road or
            bridge construction project into its correct **PMGSY scheme** based on physical
            and financial characteristics — eliminating manual tagging errors and scaling
            government reporting.

            #### 🗂️ Supported Schemes
            | Scheme | Period | Focus |
            |--------|--------|-------|
            | **PMGSY-I** | 2000–2012 | Basic rural connectivity |
            | **PMGSY-II** | 2013–2019 | Upgradation of existing roads |
            | **RCPLWEA** | 2016–present | LWE-affected areas |
            | **PMGSY-III** | 2019–present | Consolidation & upgradation |
            | **MNREGA** | 2006–present | Labour-intensive rural works |

            #### 🔬 Features Used
            Physical: road length, bridge length, terrain type, surface type  
            Financial: DPR cost, revised cost, expenditure  
            Administrative: state, sanction year, project type
            """
        )

    with col2:
        st.markdown("## Workflow")
        steps = [
            ("1️⃣", "Load / Generate Data", "PMGSY project records"),
            ("2️⃣", "Preprocess", "Encode + Scale features"),
            ("3️⃣", "Train RF Model", "300-estimator Random Forest"),
            ("4️⃣", "Evaluate", "Accuracy · F1 · Confusion Matrix"),
            ("5️⃣", "Deploy", "IBM Watson ML (optional)"),
            ("6️⃣", "Predict", "Online or local inference"),
        ]
        for icon, title, desc in steps:
            st.markdown(
                f"""
                <div style="background:#f8fafc;border-left:4px solid #2d6a4f;
                     padding:10px 14px;border-radius:6px;margin-bottom:8px;">
                  <strong>{icon} {title}</strong><br>
                  <span style="color:#6b7280;font-size:0.85rem;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### 📁 Data Source")
    st.info(
        "**AI Kosh Dataset** — Pradhan Mantri Gram Sadak Yojana (PMGSY)\n\n"
        "🔗 https://aikosh.indiaai.gov.in/web/datasets/details/pradhan_mantri_gram_sadak_yojna_pmgsy.html\n\n"
        "Upload your downloaded CSV in the **Data Explorer** tab, or use the built-in "
        "synthetic dataset (8,000 records) that mirrors the official data structure."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")

    from data_utils import load_data, ALL_FEATURES, TARGET, SCHEME_LABELS

    uploaded = st.file_uploader(
        "Upload PMGSY CSV (from AI Kosh) — or leave blank to use synthetic data",
        type=["csv"],
    )

    csv_path = None
    if uploaded:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.read())
            csv_path = tmp.name

    with st.spinner("Loading data…"):
        try:
            df = load_data(csv_path)
            st.session_state.df_loaded = df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    src = "uploaded CSV" if csv_path else "synthetic dataset"
    st.success(f"✅ Loaded {len(df):,} records from {src}")

    # ── Key metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Projects", f"{len(df):,}")
    c2.metric("Features", len(ALL_FEATURES))
    c3.metric("Schemes", df[TARGET].nunique())
    c4.metric("States", df["STATE_NAME"].nunique())

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Distributions", "🔥 Correlation"])

    with tab1:
        st.dataframe(df.head(200), use_container_width=True)
        st.caption(f"Showing first 200 of {len(df):,} rows")

    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            scheme_counts = df[TARGET].value_counts().reset_index()
            scheme_counts.columns = ["Scheme", "Count"]
            fig = px.bar(
                scheme_counts, x="Scheme", y="Count", color="Scheme",
                title="Projects by PMGSY Scheme",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.pie(
                scheme_counts, names="Scheme", values="Count",
                title="Scheme Distribution (%)",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            fig3 = px.box(
                df, x=TARGET, y="ROAD_LENGTH_KM", color=TARGET,
                title="Road Length by Scheme",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            fig4 = px.box(
                df, x=TARGET, y="DPR_COST_LAKH", color=TARGET,
                title="DPR Cost (Lakh) by Scheme",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        col_e, col_f = st.columns(2)
        with col_e:
            terrain_scheme = (
                df.groupby(["TERRAIN_TYPE", TARGET])
                .size()
                .reset_index(name="Count")
            )
            fig5 = px.bar(
                terrain_scheme, x="TERRAIN_TYPE", y="Count",
                color=TARGET, barmode="stack",
                title="Terrain Type vs Scheme",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col_f:
            fig6 = px.histogram(
                df, x="SANCTION_YEAR", color=TARGET,
                title="Sanction Year Distribution",
                barmode="stack",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            st.plotly_chart(fig6, use_container_width=True)

    with tab3:
        from data_utils import NUMERIC_FEATURES
        num_df = df[NUMERIC_FEATURES + [TARGET]].copy()
        # Encode target for correlation
        from sklearn.preprocessing import LabelEncoder
        num_df[TARGET] = LabelEncoder().fit_transform(num_df[TARGET])
        corr = num_df.corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Train Model
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Train Model":
    st.markdown("## 🤖 Train Classification Model")

    from data_utils import load_data, ALL_FEATURES, TARGET
    from model_utils import train_model

    st.info(
        "The model is a **Random Forest Classifier** (300 estimators) trained on "
        "physical and financial project attributes to predict the PMGSY scheme."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        n_estimators = st.slider("Number of trees", 100, 500, 300, 50)
        use_uploaded = st.checkbox("Use uploaded CSV (if available)")
        csv_path = None
        if use_uploaded and st.session_state.df_loaded is not None:
            csv_path = "__session__"

        train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)

    with col2:
        st.markdown(
            """
            **Algorithm:** Random Forest (ensemble of decision trees)  
            **Train / Test split:** 80% / 20% stratified  
            **Preprocessing:** Label encoding + StandardScaler  
            **Evaluation metrics:** Accuracy · Weighted F1 · Confusion Matrix
            """
        )

    if train_btn:
        with st.spinner("Training model… (this takes ~15–30 seconds)"):
            try:
                if use_uploaded and st.session_state.df_loaded is not None:
                    # Use already-loaded df via a monkey-patch
                    import data_utils as du
                    _orig_load = du.load_data
                    du.load_data = lambda _=None: st.session_state.df_loaded
                    results, artifacts = train_model()
                    du.load_data = _orig_load
                else:
                    results, artifacts = train_model()

                st.session_state.train_results = results
                st.session_state.artifacts = artifacts
                st.success("✅ Model trained and saved successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

    if st.session_state.train_results:
        results = st.session_state.train_results

        st.divider()
        st.markdown("### 📊 Evaluation Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{results['accuracy']:.2%}")
        c2.metric("Weighted F1", f"{results['f1_weighted']:.4f}")
        c3.metric("Schemes", len(results["scheme_labels"]))

        tab1, tab2, tab3 = st.tabs(["🔢 Confusion Matrix", "📋 Class Report", "🌲 Feature Importance"])

        with tab1:
            cm = results["confusion_matrix"]
            labels = results["scheme_labels"]
            fig_cm = px.imshow(
                cm,
                x=labels, y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title="Confusion Matrix",
            )
            fig_cm.update_layout(width=600, height=500)
            st.plotly_chart(fig_cm, use_container_width=True)

        with tab2:
            rpt = results["classification_report"]
            rpt_df = pd.DataFrame(rpt).transpose().round(4)
            st.dataframe(rpt_df, use_container_width=True)

        with tab3:
            feat_imp = results["feature_importances"]
            fi_df = pd.DataFrame(feat_imp, columns=["Feature", "Importance"])
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                title="Feature Importances (Random Forest)",
                color="Importance",
                color_continuous_scale="Greens",
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Predict
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Predict":
    st.markdown("## 🔍 Predict PMGSY Scheme")

    from model_utils import load_artifacts, predict_local
    from data_utils import STATE_NAMES, PROJECT_TYPES, SURFACE_TYPES, TERRAIN_TYPES, SCHEME_LABELS

    # Try to load artifacts
    if st.session_state.artifacts is None:
        try:
            st.session_state.artifacts = load_artifacts()
            st.info("ℹ️ Loaded previously trained model from disk.")
        except FileNotFoundError:
            st.warning("⚠️ No trained model found. Please go to **Train Model** first.")
            st.stop()

    artifacts = st.session_state.artifacts

    st.markdown("### Enter Project Details")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Physical Characteristics**")
            road_len = st.number_input("Road Length (km)", 0.0, 100.0, 5.0, 0.1)
            bridge_len = st.number_input("Bridge Length (m)", 0.0, 500.0, 0.0, 5.0)
            num_packages = st.number_input("Number of Packages", 1, 10, 1)
            connected_hab = st.number_input("Connected Habitations", 1, 20, 1)
            hab_pop = st.number_input("Habitation Population", 50, 10000, 500, 50)

        with c2:
            st.markdown("**Financial Details**")
            dpr_cost = st.number_input("DPR Cost (Lakh ₹)", 0.0, 5000.0, 50.0, 5.0)
            revised_cost = st.number_input("Revised Cost (Lakh ₹)", 0.0, 5000.0, 55.0, 5.0)
            expenditure = st.number_input("Expenditure (Lakh ₹)", 0.0, 5000.0, 40.0, 5.0)

        with c3:
            st.markdown("**Administrative Details**")
            state = st.selectbox("State", sorted(STATE_NAMES))
            proj_type = st.selectbox("Project Type", PROJECT_TYPES)
            surface = st.selectbox("Surface Type", SURFACE_TYPES)
            terrain = st.selectbox("Terrain Type", TERRAIN_TYPES)
            sanction_year = st.selectbox(
                "Sanction Year", [str(y) for y in range(2000, 2026)]
            )

        submitted = st.form_submit_button("🔮 Classify Scheme", type="primary", use_container_width=True)

    if submitted:
        input_dict = {
            "ROAD_LENGTH_KM": road_len,
            "BRIDGE_LENGTH_MTR": bridge_len,
            "DPR_COST_LAKH": dpr_cost,
            "REVISED_COST_LAKH": revised_cost,
            "EXPENDITURE_LAKH": expenditure,
            "HABITATION_POPULATION": hab_pop,
            "CONNECTED_HABITATION": connected_hab,
            "NUM_PACKAGES": num_packages,
            "STATE_NAME": state,
            "PROJECT_TYPE": proj_type,
            "SURFACE_TYPE": surface,
            "TERRAIN_TYPE": terrain,
            "SANCTION_YEAR": sanction_year,
        }

        with st.spinner("Classifying…"):
            result = predict_local(input_dict, artifacts)

        scheme = result["predicted_scheme"]
        probs = result["probabilities"]

        SCHEME_COLORS = {
            "PMGSY-I": "#2d6a4f",
            "PMGSY-II": "#1e6091",
            "RCPLWEA": "#9b2226",
            "PMGSY-III": "#7209b7",
            "MNREGA": "#e85d04",
        }
        color = SCHEME_COLORS.get(scheme, "#374151")

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        st.markdown(
            f"""
            <div style="background:{color}20;border:2px solid {color};
                 border-radius:12px;padding:20px 30px;text-align:center;margin:10px 0;">
              <div style="font-size:1rem;color:#6b7280;">Predicted PMGSY Scheme</div>
              <div style="font-size:2.4rem;font-weight:800;color:{color};">{scheme}</div>
              <div style="font-size:0.9rem;color:#6b7280;margin-top:6px;">
                Confidence: {probs[scheme]:.1%}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability bar chart
        prob_df = pd.DataFrame(
            list(probs.items()), columns=["Scheme", "Probability"]
        ).sort_values("Probability", ascending=True)

        fig_prob = px.bar(
            prob_df, x="Probability", y="Scheme", orientation="h",
            title="Classification Probabilities",
            color="Probability",
            color_continuous_scale=[[0, "#e0f2fe"], [1, color]],
            text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
        )
        fig_prob.update_traces(textposition="outside")
        fig_prob.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── Batch prediction ──
    st.divider()
    st.markdown("### 📦 Batch Prediction")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        from data_utils import ALL_FEATURES
        missing = [c for c in ALL_FEATURES if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = []
            for _, row in batch_df.iterrows():
                r = predict_local(row.to_dict(), artifacts)
                preds.append(r["predicted_scheme"])
            batch_df["PREDICTED_SCHEME"] = preds
            st.dataframe(batch_df, use_container_width=True)
            csv_out = batch_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv_out, "pmgsy_predictions.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — IBM WML Deploy
# ─────────────────────────────────────────────────────────────────────────────
elif page == "☁️ IBM WML Deploy":
    st.markdown("## ☁️ IBM watsonx.ai Deployment")

    from ibm_wml_utils import is_wml_configured, deploy_model_to_wml, score_via_wml

    st.markdown(
        """
        Deploy the trained model to **IBM watsonx.ai** (Lite tier) for
        scalable, persistent online scoring via REST API.

        ### Prerequisites
        1. Create a free IBM Cloud account at [cloud.ibm.com](https://cloud.ibm.com)
        2. Provision **Watson Machine Learning** (Lite plan — free)
        3. Create a **Deployment Space** in IBM Watson Studio
        4. Enter your credentials in the sidebar
        """
    )

    # Status panel
    col1, col2, col3 = st.columns(3)
    col1.metric("IBM API Key", "✅ Set" if os.getenv("IBM_API_KEY") else "❌ Missing")
    col2.metric("WML URL", "✅ Set" if os.getenv("IBM_WML_URL") else "❌ Missing")
    col3.metric("Space ID", "✅ Set" if os.getenv("IBM_WML_SPACE_ID") else "❌ Missing")

    st.divider()

    # ── Deploy button ──
    col_d, col_s = st.columns(2)

    with col_d:
        st.markdown("### 🚀 Deploy Model")
        if st.button("Deploy to IBM WML", type="primary", disabled=not is_wml_configured()):
            if st.session_state.artifacts is None and not os.path.exists("pmgsy_artifacts.pkl"):
                st.error("Please train the model first (Train Model page).")
            else:
                with st.spinner("Deploying to IBM Watson ML… (may take 1–3 minutes)"):
                    try:
                        info = deploy_model_to_wml()
                        st.session_state.deployment_info = info
                        st.success("✅ Model deployed successfully!")
                    except Exception as e:
                        st.error(f"Deployment failed: {e}")

        if st.session_state.deployment_info:
            info = st.session_state.deployment_info
            st.markdown("**Deployment Details:**")
            st.code(
                f"Model UID:      {info['model_uid']}\n"
                f"Deployment ID:  {info['deployment_id']}\n"
                f"Scoring URL:    {info['scoring_url']}"
            )

    with col_s:
        st.markdown("### 🔮 Score via WML")
        deployment_id_input = st.text_input(
            "Deployment ID",
            value=st.session_state.deployment_info["deployment_id"]
            if st.session_state.deployment_info
            else "",
        )
        sample_payload = st.text_area(
            "Feature values (comma-separated, same order as ALL_FEATURES)",
            value="5.0,0.0,50.0,55.0,40.0,500,1,1,0,0,0,0,2015",
            height=80,
        )
        if st.button("Score via WML API", disabled=not is_wml_configured()):
            try:
                features = [float(x.strip()) for x in sample_payload.split(",")]
                result = score_via_wml(features, deployment_id_input)
                st.success(f"Predicted scheme: **{result['predicted_scheme']}**")
                st.json(result["raw_response"])
            except Exception as e:
                st.error(f"Scoring failed: {e}")

    st.divider()

    # ── Architecture diagram ──
    st.markdown("### 🏗️ Architecture")
    st.markdown(
        """
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
        """
    )

    st.markdown(
        """
        **Cost:** IBM WML Lite tier provides free capacity unit-hours per month.
        The Random Forest model is lightweight and well within the free tier limits.

        **Security:** Only the three mandatory credentials (API key, WML URL, Space ID)
        are required. No data is stored in IBM Cloud — only the serialised model artefact.
        """
    )


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:40px;border:none;border-top:1px solid #e5e7eb;">
    <p style="text-align:center;color:#9ca3af;font-size:0.8rem;">
      PMGSY Scheme Classifier · Built with Streamlit &amp; IBM Watson ML ·
      Data source: AI Kosh / IndiaAI
    </p>
    """,
    unsafe_allow_html=True,
)
