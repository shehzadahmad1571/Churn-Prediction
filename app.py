from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

NUMERIC_FIELDS = {
    "SeniorCitizen": {
        "label": "Senior citizen",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")],
        "minimum": 0,
        "maximum": 1,
        "integer": True,
    },
    "tenure": {
        "label": "Tenure (months)",
        "kind": "number",
        "minimum": 0,
        "maximum": 72,
        "step": 1,
        "integer": True,
    },
    "MonthlyCharges": {
        "label": "Monthly charges ($)",
        "kind": "number",
        "minimum": 0,
        "maximum": 200,
        "step": "0.01",
    },
    "TotalCharges": {
        "label": "Total charges ($)",
        "kind": "number",
        "minimum": 0,
        "maximum": 20000,
        "step": "0.01",
    },
}

CATEGORICAL_FIELDS = {
    "gender": [("Female", "Female"), ("Male", "Male")],
    "Partner": [("No", "No"), ("Yes", "Yes")],
    "Dependents": [("No", "No"), ("Yes", "Yes")],
    "PhoneService": [("No", "No"), ("Yes", "Yes")],
    "MultipleLines": [
        ("No phone service", "No phone service"),
        ("No", "No"),
        ("Yes", "Yes"),
    ],
    "InternetService": [
        ("DSL", "DSL"),
        ("Fiber optic", "Fiber optic"),
        ("No", "No internet service"),
    ],
    "OnlineSecurity": [
        ("No internet service", "No internet service"),
        ("No", "No"),
        ("Yes", "Yes"),
    ],
    "OnlineBackup": [
        ("No internet service", "No internet service"),
        ("No", "No"),
        ("Yes", "Yes"),
    ],
    "DeviceProtection": [
        ("No internet service", "No internet service"),
        ("No", "No"),
        ("Yes", "Yes"),
    ],
    "TechSupport": [
        ("No internet service", "No internet service"),
        ("No", "No"),
        ("Yes", "Yes"),
    ],
    "StreamingTV": [
        ("No internet service", "No internet service"),
        ("No", "No"),
        ("Yes", "Yes"),
    ],
    "StreamingMovies": [
        ("No internet service", "No internet service"),
        ("No", "No"),
        ("Yes", "Yes"),
    ],
    "Contract": [
        ("Month-to-month", "Month-to-month"),
        ("One year", "One year"),
        ("Two year", "Two year"),
    ],
    "PaperlessBilling": [("No", "No"), ("Yes", "Yes")],
    "PaymentMethod": [
        ("Bank transfer (automatic)", "Bank transfer (automatic)"),
        ("Credit card (automatic)", "Credit card (automatic)"),
        ("Electronic check", "Electronic check"),
        ("Mailed check", "Mailed check"),
    ],
}

FIELD_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

FIELD_LABELS = {
    "gender": "Gender",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}


def load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run the model-export cell in churn_prediction.ipynb first."
        )

    bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "threshold", "feature_names", "model_name"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError(f"{MODEL_PATH} is not a valid churn-model bundle.")
    if bundle["feature_names"] != FIELD_ORDER:
        raise ValueError("Model features do not match the web form field order.")
    return bundle


MODEL_BUNDLE = load_model_bundle()
MODEL = MODEL_BUNDLE["model"]
THRESHOLD = float(MODEL_BUNDLE["threshold"])

FORM_FIELDS: list[dict[str, Any]] = []
for field_name in FIELD_ORDER:
    if field_name in NUMERIC_FIELDS:
        FORM_FIELDS.append(
            {"name": field_name, **NUMERIC_FIELDS[field_name]}
        )
    elif field_name in CATEGORICAL_FIELDS:
        FORM_FIELDS.append(
            {
                "name": field_name,
                "label": FIELD_LABELS[field_name],
                "kind": "select",
                "choices": CATEGORICAL_FIELDS[field_name],
            }
        )


def parse_customer_form(
    form: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    customer: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for field_name, rules in NUMERIC_FIELDS.items():
        raw_value = form.get(field_name, "").strip()
        if not raw_value:
            errors[field_name] = "This field is required."
            continue
        try:
            value = float(raw_value)
        except ValueError:
            errors[field_name] = "Enter a valid number."
            continue
        if not math.isfinite(value):
            errors[field_name] = "Enter a finite number."
            continue
        if value < rules["minimum"] or value > rules["maximum"]:
            errors[field_name] = (
                f"Enter a value from {rules['minimum']} to {rules['maximum']}."
            )
            continue
        if rules.get("integer") and not value.is_integer():
            errors[field_name] = "Enter a whole number."
            continue
        customer[field_name] = int(value) if rules.get("integer") else value

    for field_name, choices in CATEGORICAL_FIELDS.items():
        value = form.get(field_name, "").strip()
        allowed_values = {choice[0] for choice in choices}
        if value not in allowed_values:
            errors[field_name] = "Choose one of the listed options."
        else:
            customer[field_name] = value

    if not errors:
        if customer["PhoneService"] == "No":
            if customer["MultipleLines"] != "No phone service":
                errors["MultipleLines"] = (
                    "Choose 'No phone service' when phone service is No."
                )
        elif customer["MultipleLines"] == "No phone service":
            errors["MultipleLines"] = (
                "Choose No or Yes when phone service is available."
            )

        internet_add_ons = (
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        )
        if customer["InternetService"] == "No":
            for field_name in internet_add_ons:
                if customer[field_name] != "No internet service":
                    errors[field_name] = (
                        "Choose 'No internet service' when Internet service is No."
                    )
        else:
            for field_name in internet_add_ons:
                if customer[field_name] == "No internet service":
                    errors[field_name] = (
                        "This option is only valid when Internet service is No."
                    )

    return customer, errors


app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        values={},
        errors={},
        result=None,
        form_error=None,
    )


@app.post("/predict")
def predict():
    customer, errors = parse_customer_form(request.form)
    if errors:
        return (
            render_template(
                "index.html",
                fields=FORM_FIELDS,
                values=request.form,
                errors=errors,
                result=None,
                form_error="Please correct the highlighted fields.",
            ),
            400,
        )

    input_frame = pd.DataFrame(
        [[customer[name] for name in FIELD_ORDER]],
        columns=FIELD_ORDER,
    )
    churn_probability = float(MODEL.predict_proba(input_frame)[0, 1])
    is_churn = churn_probability >= THRESHOLD
    result = {
        "is_churn": is_churn,
        "label": "Likely to churn" if is_churn else "Likely to stay",
        "probability": churn_probability,
        "threshold": THRESHOLD,
    }

    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        values=request.form,
        errors={},
        result=result,
        form_error=None,
    )


if __name__ == "__main__":
    app.run(debug=False)
