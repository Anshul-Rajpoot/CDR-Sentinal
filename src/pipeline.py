"""Data preparation and modelling utilities for CDR Sentinel.

The target is built from RBI credit/deposit totals. Only NFHS numeric
features are passed to the model; financial amounts are never predictors.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
FINANCIAL_FILE = ROOT / "financial_data.xlsx"
SOCIAL_TRAIN_FILE = ROOT / "social_train.xlsx"
SOCIAL_TEST_FILE = ROOT / "social_test.csv"


def normalize_name(value: Any) -> str:
    """Normalize names for matching without changing the source value."""
    if pd.isna(value):
        return ""
    text = str(value).casefold().strip()
    text = re.sub(r"\b(district|dist)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_state(value: Any) -> str:
    """Normalize common state-name variants used by the supplied files."""
    name = normalize_name(value)
    aliases = {
        "andaman nicobar islands": "andaman and nicobar islands",
        "nct of delhi": "delhi",
        "national capital territory of delhi": "delhi",
        "orissa": "odisha",
        "uttaranchal": "uttarakhand",
        "maharastra": "maharashtra",
        "jammu kashmir": "jammu and kashmir",
    }
    return aliases.get(name, name)


def parse_number(value: Any) -> float:
    """Convert numeric-looking strings (commas/currency marks included)."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").replace("₹", "").strip()
    if text in {"", "-", "—", "NA", "N/A"}:
        return np.nan
    return pd.to_numeric(text, errors="coerce")


def load_financial_data(path: Path = FINANCIAL_FILE) -> pd.DataFrame:
    """Read the RBI workbook's multi-row header and consolidate quarters.

    The supplied workbook stores district metadata on row 6 and quarter
    labels on row 7 (zero-based indexing). Each district's available quarterly
    deposits and credit are summed, as specified in the project brief.
    """
    if not path.exists():
        raise FileNotFoundError(f"RBI workbook not found: {path}")

    raw = pd.read_excel(path, header=None)
    if raw.shape[0] < 8:
        raise ValueError("RBI workbook does not contain the expected header/data rows.")

    header_row = raw.iloc[5].fillna("").astype(str).str.strip()
    district_candidates = [i for i, value in enumerate(header_row) if value.casefold() == "district"]
    state_candidates = [i for i, value in enumerate(header_row) if value.casefold() in {"state or uts", "state/ut", "state"}]
    if not district_candidates:
        raise ValueError("Could not find the 'District' column in the RBI workbook.")
    district_col = district_candidates[0]
    state_col = state_candidates[0] if state_candidates else None

    measure_row = raw.iloc[6].fillna("").astype(str).str.strip().str.casefold()
    deposit_cols = [i for i, value in enumerate(measure_row) if value == "deposit"]
    credit_cols = [i for i, value in enumerate(measure_row) if value == "credit"]
    if not deposit_cols or not credit_cols:
        raise ValueError("Could not find quarterly Deposit and Credit columns in RBI workbook.")

    data = pd.DataFrame({
        "district": raw.iloc[7:, district_col].astype("string").str.strip(),
        "state": raw.iloc[7:, state_col].astype("string").str.strip() if state_col is not None else "",
        "total_deposits": raw.iloc[7:, deposit_cols].apply(lambda col: col.map(parse_number)).sum(axis=1, min_count=1),
        "total_credit": raw.iloc[7:, credit_cols].apply(lambda col: col.map(parse_number)).sum(axis=1, min_count=1),
    })
    data["district_key"] = data["district"].map(normalize_name)
    data["state_key"] = data["state"].map(normalize_state)
    data = data[data["district_key"].ne("") & ~data["district_key"].isin({"nan", "all india"})].copy()

    # Keep the original geography key visible for audits; aggregate duplicate
    # district records only within the same state.
    data = (
        data.groupby(["state_key", "district_key"], as_index=False, dropna=False)
        .agg(
            district=("district", "first"),
            state=("state", "first"),
            total_deposits=("total_deposits", "sum"),
            total_credit=("total_credit", "sum"),
        )
    )
    data["cdr"] = np.where(
        data["total_deposits"] > 0,
        100 * data["total_credit"] / data["total_deposits"],
        np.nan,
    )
    data.loc[~np.isfinite(data["cdr"]), "cdr"] = np.nan
    return data


def load_social_data(path: Path) -> pd.DataFrame:
    """Load NFHS features from CSV or Excel and standardize numeric columns."""
    if not path.exists():
        raise FileNotFoundError(f"NFHS file not found: {path}")

    if path.suffix.casefold() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path, sheet_name=0)

    frame = frame.dropna(axis=1, how="all").copy()
    required = {"District Names", "State/UT"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"NFHS file is missing required columns: {sorted(missing)}")

    frame["district_key"] = frame["District Names"].map(normalize_name)
    frame["state_key"] = frame["State/UT"].map(normalize_state)

    # Source columns remain untouched in the displayed data. Only model
    # candidate columns are converted to numeric.
    protected = {"District Names", "State/UT", "district_key", "state_key"}
    for column in frame.columns:
        if column not in protected:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def match_districts(
    social: pd.DataFrame,
    financial: pd.DataFrame,
    threshold: int = 88,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match NFHS districts to RBI districts using state-aware fuzzy matching.

    Exact matches are preferred. Fuzzy matches are restricted to the same
    normalized state when possible. Each RBI district can be used once to
    avoid many-to-one accidental matches. All decisions are retained in an
    audit table.
    """
    fin = financial.reset_index(drop=True).copy()
    fin["financial_id"] = fin.index
    used_ids: set[int] = set()
    match_rows: list[dict[str, Any]] = []

    # Build candidate lists by state for quicker, more reliable comparisons.
    state_groups = {
        state: group for state, group in fin.groupby("state_key", dropna=False)
    }
    all_choices = fin["district_key"].tolist()

    # Match exact keys first, globally but prefer same-state candidates.
    for _, row in social.iterrows():
        district_key = row["district_key"]
        state_key = row["state_key"]
        same_state = state_groups.get(state_key, fin.iloc[0:0])
        candidate_pool = same_state if not same_state.empty else fin

        exact = candidate_pool[
            (candidate_pool["district_key"] == district_key)
            & (~candidate_pool["financial_id"].isin(used_ids))
        ]
        if not exact.empty:
            candidate = exact.iloc[0]
            used_ids.add(int(candidate["financial_id"]))
            match_rows.append({
                "district_key": district_key,
                "state_key": state_key,
                "financial_id": int(candidate["financial_id"]),
                "financial_district": candidate["district"],
                "match_score": 100,
                "match_type": "exact",
            })
            continue

        available = candidate_pool[~candidate_pool["financial_id"].isin(used_ids)]
        if available.empty:
            available = fin[~fin["financial_id"].isin(used_ids)]
        if available.empty:
            match_rows.append({
                "district_key": district_key, "state_key": state_key,
                "financial_id": np.nan, "financial_district": None,
                "match_score": 0, "match_type": "unmatched",
            })
            continue

        choices = available["district_key"].tolist()
        result = process.extractOne(district_key, choices, scorer=fuzz.token_set_ratio)
        if result and result[1] >= threshold:
            chosen_key, score, position = result
            candidate = available.iloc[position]
            used_ids.add(int(candidate["financial_id"]))
            match_rows.append({
                "district_key": district_key,
                "state_key": state_key,
                "financial_id": int(candidate["financial_id"]),
                "financial_district": candidate["district"],
                "match_score": int(score),
                "match_type": "fuzzy",
            })
        else:
            match_rows.append({
                "district_key": district_key, "state_key": state_key,
                "financial_id": np.nan, "financial_district": None,
                "match_score": int(result[1]) if result else 0,
                "match_type": "unmatched",
            })

    audit = pd.DataFrame(match_rows)
    merged = social.merge(audit, on=["district_key", "state_key"], how="left", validate="one_to_one")
    fin_target = fin[["financial_id", "cdr", "total_credit", "total_deposits", "district", "state"]].rename(
        columns={
            "cdr": "actual_cdr",
            "total_credit": "rbi_total_credit",
            "total_deposits": "rbi_total_deposits",
            "district": "rbi_district",
            "state": "rbi_state",
        }
    )
    merged = merged.merge(fin_target, on="financial_id", how="left", validate="many_to_one")
    summary = {
        "social_rows": int(len(social)),
        "matched_rows": int(merged["actual_cdr"].notna().sum()),
        "exact_matches": int((audit["match_type"] == "exact").sum()),
        "fuzzy_matches": int((audit["match_type"] == "fuzzy").sum()),
        "unmatched_rows": int((audit["match_type"] == "unmatched").sum()),
        "average_match_score": round(float(audit["match_score"].mean()), 1) if len(audit) else 0.0,
        "threshold": int(threshold),
    }
    return merged, {"summary": summary, "audit": audit}


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select only numeric NFHS features; never include RBI-derived fields."""
    protected = {
        "actual_cdr", "rbi_total_credit", "rbi_total_deposits",
        "match_score", "financial_id",
    }
    features = []
    for column in frame.columns:
        if column in protected or column.startswith("rbi_"):
            continue
        if column in {"district_key", "state_key"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            features.append(column)
    return features


def make_model() -> Pipeline:
    """A compact, interpretable preprocessing + tree-ensemble pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )),
    ])


@dataclass
class ModelBundle:
    model: Pipeline
    features: list[str]
    metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]]
    merge_summary: dict[str, Any]
    model_comparison: list[dict[str, Any]]


def train_model() -> ModelBundle:
    """Train and evaluate the model using only matched NFHS training rows."""
    financial = load_financial_data()
    social = load_social_data(SOCIAL_TRAIN_FILE)
    merged, match_info = match_districts(social, financial)

    train_data = merged.dropna(subset=["actual_cdr"]).copy()
    if len(train_data) < 20:
        raise ValueError(
            f"Only {len(train_data)} districts matched RBI targets. "
            "Review the district matching audit before training."
        )

    features = get_feature_columns(train_data)
    if not features:
        raise ValueError("No numeric NFHS feature columns were found.")

    X = train_data[features].replace([np.inf, -np.inf], np.nan)
    y = train_data["actual_cdr"].astype(float)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    candidates = {
        "Median baseline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="median")),
        ]),
        "Ridge regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Extra Trees": make_model(),
    }
    comparison = []
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        prediction = candidate.predict(X_valid)
        comparison.append({
            "model": name,
            "mae": round(float(mean_absolute_error(y_valid, prediction)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_valid, prediction))), 3),
            "r2": round(float(r2_score(y_valid, prediction)), 3),
        })

    best_name = min(comparison, key=lambda row: row["rmse"])["model"]
    # Keep Extra Trees as the final estimator for this project unless a
    # simpler baseline wins by RMSE. This makes model choice explicit.
    best_model = candidates[best_name]
    valid_prediction = best_model.predict(X_valid)
    metrics = {
        "selected_model": best_name,
        "mae": round(float(mean_absolute_error(y_valid, valid_prediction)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_valid, valid_prediction))), 3),
        "r2": round(float(r2_score(y_valid, valid_prediction)), 3),
        "training_rows": int(len(train_data)),
        "train_split_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "feature_count": int(len(features)),
        "validation_method": "80/20 random holdout; random_state=42",
        "target": "100 * total RBI credit / total RBI deposits",
    }

    # Explain the selected model on held-out rows, then refit on all matched
    # training data for inference. For non-tree winners, permutation still works.
    importance = permutation_importance(
        best_model, X_valid, y_valid, n_repeats=5, random_state=42, n_jobs=1
    )
    ranked = np.argsort(importance.importances_mean)[::-1][:10]
    feature_importance = [
        {
            "feature": features[int(index)],
            "importance": round(float(importance.importances_mean[index]), 5),
        }
        for index in ranked if importance.importances_mean[index] > 0
    ]
    best_model.fit(X, y)

    bundle = ModelBundle(
        model=best_model,
        features=features,
        metrics=metrics,
        feature_importance=feature_importance,
        merge_summary=match_info["summary"],
        model_comparison=comparison,
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps({
            "metrics": metrics,
            "feature_importance": feature_importance,
            "merge_summary": match_info["summary"],
            "model_comparison": comparison,
        }, indent=2),
        encoding="utf-8",
    )
    match_info["audit"].to_csv(ARTIFACT_DIR / "district_match_audit.csv", index=False)
    return bundle


def load_bundle(force_retrain: bool = False) -> ModelBundle:
    path = ARTIFACT_DIR / "model.joblib"
    if force_retrain or not path.exists():
        return train_model()
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            return train_model()
        return bundle
    except Exception:
        # Rebuild stale artifacts after dependency or code changes.
        return train_model()


def predict_test(bundle: ModelBundle | None = None) -> pd.DataFrame:
    """Return the exact one-column format required by the challenge."""
    bundle = bundle or load_bundle()
    test = load_social_data(SOCIAL_TEST_FILE)
    missing = sorted(set(bundle.features).difference(test.columns))
    if missing:
        raise ValueError(f"Test data is missing model features: {missing[:10]}")
    X_test = test[bundle.features].replace([np.inf, -np.inf], np.nan)
    predictions = bundle.model.predict(X_test)
    return pd.DataFrame({"Predicted_CDR": np.round(predictions, 3)})


def build_insights(bundle: ModelBundle | None = None) -> dict[str, Any]:
    """Create explanatory tables for the dashboard, with caveats."""
    bundle = bundle or load_bundle()
    financial = load_financial_data()
    social = load_social_data(SOCIAL_TRAIN_FILE)
    merged, match_info = match_districts(social, financial)
    matched = merged.dropna(subset=["actual_cdr"]).copy()
    X = matched[bundle.features].replace([np.inf, -np.inf], np.nan)
    matched["predicted_cdr"] = bundle.model.predict(X)
    matched["gap"] = matched["predicted_cdr"] - matched["actual_cdr"]

    # These are candidates for follow-up analysis, not definitive proof that
    # a district is underserved or that social factors cause lending levels.
    candidates = matched.sort_values("gap", ascending=False).head(5)
    underserved = candidates[[
        "District Names", "State/UT", "actual_cdr", "predicted_cdr", "gap"
    ]].rename(columns={
        "District Names": "district",
        "State/UT": "state",
        "actual_cdr": "actual_cdr",
    }).round(2)

    return {
        "top_factors": bundle.feature_importance[:5],
        "model_comparison": bundle.model_comparison,
        "underserved_candidates": underserved.to_dict("records"),
        "merge_summary": match_info["summary"],
    }
