"""Streamlit dashboard for CDR Sentinel."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import (  # noqa: E402
    build_insights,
    load_bundle,
    load_financial_data,
    predict_test,
)

st.set_page_config(
    page_title="CDR Sentinel | Financial Resilience",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] { border: 1px solid #e7eaf0;
        border-radius: 10px; padding: 12px 14px; }
      .small-note { color: #64748b; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_financial():
    return load_financial_data()


@st.cache_data
def get_insights():
    return build_insights(get_bundle())


st.sidebar.title("CDR Sentinel")
st.sidebar.caption("Financial Resilience Stress Test")
page = st.sidebar.radio(
    "Explore",
    ["Overview", "District explorer", "Model evaluation", "Data quality"],
)
st.sidebar.divider()
st.sidebar.caption(
    "RBI financial aggregates + NFHS social indicators. "
    "Predictions are exploratory, not lending or investment advice."
)

try:
    bundle = get_bundle()
except Exception as exc:
    st.error("The model could not be prepared from the supplied datasets.")
    st.exception(exc)
    st.stop()

if page == "Overview":
    st.title("Can social indicators help explain district-level CDR?")
    st.write(
        "CDR Sentinel combines district-level banking aggregates from RBI "
        "with social and health indicators from NFHS. It builds a CDR target, "
        "matches district names, trains a regression model and reports where "
        "the model's predictions differ from observed values."
    )

    metrics = bundle.metrics
    cards = st.columns(4)
    cards[0].metric("Validation RMSE", f'{metrics["rmse"]:.2f}')
    cards[1].metric("Validation MAE", f'{metrics["mae"]:.2f}')
    cards[2].metric("Validation R²", f'{metrics["r2"]:.3f}')
    cards[3].metric("Matched districts", f'{metrics["training_rows"]:,}')

    st.subheader("How the pipeline works")
    st.markdown(
        """
        **1. Load** RBI quarterly financial data and NFHS district indicators  
        **2. Clean** headers, numeric values, names and missing values  
        **3. Match** district records using exact and state-aware fuzzy matching  
        **4. Calculate** CDR from total credit and deposits  
        **5. Train** a model using NFHS features only  
        **6. Evaluate** against a held-out validation set and inspect model errors
        """
    )

    st.info(
        "Interpretation note: a predicted-minus-actual gap is a model residual. "
        "It can identify districts for further investigation, but it does not "
        "prove that a district is underserved or that a social indicator causes CDR."
    )

elif page == "District explorer":
    st.title("District financial explorer")
    data = get_financial()
    query = st.text_input(
        "Search district or state",
        placeholder="Type a district or state name",
    )
    if query:
        mask = (
            data["district"].astype(str).str.contains(query, case=False, na=False)
            | data["state"].astype(str).str.contains(query, case=False, na=False)
        )
        data = data[mask]

    display = data[
        ["state", "district", "total_deposits", "total_credit", "cdr"]
    ].rename(columns={
        "state": "State",
        "district": "District",
        "total_deposits": "Total deposits (₹ crore)",
        "total_credit": "Total credit (₹ crore)",
        "cdr": "CDR (%)",
    })
    st.dataframe(
        display.sort_values("CDR (%)", ascending=False),
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Download district financial summary",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="district_financial_summary.csv",
        mime="text/csv",
    )

elif page == "Model evaluation":
    st.title("Model evaluation")
    st.caption(
        "Metrics below use an 80/20 random holdout with a fixed random seed. "
        "They are a local validation estimate, not a competition leaderboard score."
    )
    comparison = pd.DataFrame(bundle.model_comparison)
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.subheader("Most useful features in the validation set")
    importance = pd.DataFrame(bundle.feature_importance)
    if importance.empty:
        st.warning("No positive permutation-importance scores were found.")
    else:
        importance = importance.rename(columns={
            "feature": "NFHS feature",
            "importance": "Permutation importance",
        })
        st.bar_chart(importance.set_index("NFHS feature"))
    st.caption(
        "Permutation importance measures how validation performance changes "
        "when a feature is shuffled. It is not evidence of causation."
    )

    insights = get_insights()
    st.subheader("Districts for further investigation")
    st.write(
        "These are the five largest positive prediction residuals among matched "
        "districts. They are research candidates, not confirmed underserved markets."
    )
    st.dataframe(
        pd.DataFrame(insights["underserved_candidates"]),
        hide_index=True,
        use_container_width=True,
    )

elif page == "Data quality":
    st.title("Data quality and district matching")
    summary = bundle.merge_summary
    a, b, c, d = st.columns(4)
    a.metric("NFHS training rows", summary["social_rows"])
    b.metric("Rows with CDR target", summary["matched_rows"])
    c.metric("Exact matches", summary["exact_matches"])
    d.metric("Fuzzy matches", summary["fuzzy_matches"])

    st.write(
        f'Unmatched rows: **{summary["unmatched_rows"]}** · '
        f'Average matching score: **{summary["average_match_score"]}** · '
        f'Fuzzy threshold: **{summary["threshold"]}**'
    )
    audit_path = ROOT / "artifacts" / "district_match_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        st.dataframe(audit, hide_index=True, use_container_width=True)
        st.download_button(
            "Download district matching audit",
            data=audit.to_csv(index=False).encode("utf-8"),
            file_name="district_match_audit.csv",
            mime="text/csv",
        )
    else:
        st.info("The matching audit is created the first time the model is trained.")

    st.subheader("Generate challenge submission")
    st.write(
        "The output contains exactly one column, `Predicted_CDR`, and preserves "
        "the row order of social_test.csv."
    )
    if st.button("Generate submission.csv"):
        try:
            submission = predict_test(bundle)
            st.download_button(
                "Download submission.csv",
                data=submission.to_csv(index=False).encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
            )
            st.success(f"Prepared {len(submission)} predictions.")
            st.dataframe(submission.head(10), hide_index=True)
        except Exception as exc:
            st.error(f"Could not generate predictions: {exc}")
