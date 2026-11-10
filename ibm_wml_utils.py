"""
IBM watsonx.ai integration for PMGSY classifier.
Uses ibm-watsonx-ai (the successor to ibm-watson-machine-learning),
compatible with Python 3.13/3.14 on IBM Cloud Lite.

Only 3 mandatory credentials required: IBM_API_KEY, IBM_WML_URL, IBM_WML_SPACE_ID.
"""

import os
import joblib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Required IBM Cloud credentials ──────────────────────────────────────────
IBM_API_KEY    = os.getenv("IBM_API_KEY", "")
IBM_WML_URL    = os.getenv("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com")
IBM_WML_SPACE_ID = os.getenv("IBM_WML_SPACE_ID", "")

DEPLOYMENT_NAME = "pmgsy-scheme-classifier"
MODEL_NAME      = "PMGSY Scheme Classification Model"


def _get_client():
    """Authenticate and return an ibm_watsonx_ai APIClient."""
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        creds = Credentials(url=IBM_WML_URL, api_key=IBM_API_KEY)
        client = APIClient(credentials=creds, space_id=IBM_WML_SPACE_ID)
        return client
    except ImportError:
        raise RuntimeError(
            "ibm-watsonx-ai package not installed. "
            "Run: pip install ibm-watsonx-ai"
        )
    except Exception as e:
        raise RuntimeError(f"IBM watsonx.ai authentication failed: {e}")


def deploy_model_to_wml(artifacts_path: str = "pmgsy_artifacts.pkl") -> dict:
    """
    Deploy the trained scikit-learn model to IBM watsonx.ai.
    Returns deployment metadata: model_uid, deployment_id, scoring_url.
    """
    if not IBM_API_KEY or not IBM_WML_SPACE_ID:
        raise ValueError(
            "IBM_API_KEY and IBM_WML_SPACE_ID must be set in .env to deploy."
        )

    client = _get_client()
    artifacts = joblib.load(artifacts_path)
    clf = artifacts["clf"]

    # ── Resolve software spec ──
    sw_spec_id = client.software_specifications.get_id_by_name(
        "runtime-24.1-py3.11"
    )

    model_props = {
        client.repository.ModelMetaNames.NAME: MODEL_NAME,
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.4",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
    }

    # Serialise the classifier to a temp file for upload
    import tempfile, pickle
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(clf, tmp)
        tmp_path = tmp.name

    try:
        model_details = client.repository.store_model(
            model=tmp_path,
            meta_props=model_props,
        )
    finally:
        os.unlink(tmp_path)

    model_uid = client.repository.get_model_id(model_details)

    # ── Create online deployment ──
    deploy_props = {
        client.deployments.ConfigurationMetaNames.NAME: DEPLOYMENT_NAME,
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    deployment_details = client.deployments.create(
        artifact_uid=model_uid, meta_props=deploy_props
    )
    deployment_id  = client.deployments.get_id(deployment_details)
    scoring_url    = client.deployments.get_scoring_href(deployment_details)

    return {
        "model_uid":      model_uid,
        "deployment_id":  deployment_id,
        "scoring_url":    scoring_url,
    }


def score_via_wml(input_features: list, deployment_id: str) -> dict:
    """
    Score a single record through the WML online endpoint.
    input_features: list of pre-encoded numeric values in ALL_FEATURES order.
    """
    client = _get_client()
    payload = {"input_data": [{"values": [input_features]}]}
    response = client.deployments.score(deployment_id, payload)
    predictions = response["predictions"][0]
    predicted_label = predictions["values"][0][0]
    return {"predicted_scheme": predicted_label, "raw_response": predictions}


def is_wml_configured() -> bool:
    """Returns True only when all three mandatory IBM credentials are present."""
    return bool(IBM_API_KEY and IBM_WML_SPACE_ID)
