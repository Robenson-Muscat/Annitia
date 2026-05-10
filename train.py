import re
import warnings

import numpy as np
import pandas as pd
#import skrub
import sksurv
import umap

from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore")


from preprocessing import apply_final_feature_engineering


TRAIN_PATH = "data/processed_wide/train.csv"
TEST_PATH = "data/processed_wide/val_test.csv"
OUTPUT_PATH = "submission_annitia.csv"

# Columns that must not be used as features
TARGET_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]

ID_COLS = ["patient_id_anon", "trustii_id"]

MAX_MISSING_RATE = 0.50

BIOMARKERS = [
    "BMI",
    "alt",
    "ast",
    "bilirubin",
    "chol",
    "ggt",
    "gluc_fast",
    "plt",
    "triglyc",
    "aixp_aix_result_BM_3",
    "fibrotest_BM_2",
    "fibs_stiffness_med_BM_1",
]



def make_rsf_pipeline():
    """Random Survival Forest with regularization."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            #("imp", IterativeImputer()),
            (
                "rsf",
                RandomSurvivalForest(
                    n_estimators=100,
                    min_samples_leaf=20,
                    min_samples_split=40,
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=26,
                ),
            ),
        ]
    )


def cv_rsf(X, y, n_splits=5, random_state=26):
    """Cross-validated concordance index for a survival pipeline."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = make_rsf_pipeline()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        ci = concordance_index_censored(
            y_test[y.dtype.names[0]],  # event
            y_test[y.dtype.names[1]],  # time
            preds,
        )[0]

        scores.append(ci)

    return np.mean(scores), np.std(scores)
    


def prepare_survival_targets_robust(
    df: pd.DataFrame,
    outcome: str = "hepatic",
    baseline_col: str = "Age_v1",
    last_observed_col: str = "last_observed_age",
    min_time: float = 1e-1,
    drop_nonpositive_times: bool = True,
    return_report: bool = False,
):
    """
    Build a sksurv-compatible structured array for one survival endpoint.

    Rules:
      - missing event -> drop row
      - event == 1 and event time missing -> drop row
      - event == 0 -> censored at last_observed_age - Age_v1
      - negative times -> drop row if drop_nonpositive_times=True
      - zero times -> clamped to min_time
    """
    df = df.copy()

    if outcome == "hepatic":
        event_col = "evenements_hepatiques_majeurs"
        time_col = "evenements_hepatiques_age_occur"
        event_name = "Hepatic_event"
    elif outcome == "death":
        event_col = "death"
        time_col = "death_age_occur"
        event_name = "Death"
    else:
        raise ValueError(f"outcome must be 'hepatic' or 'death', got {outcome!r}")

    required_cols = [event_col, time_col, baseline_col, last_observed_col]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    n0 = len(df)

    # Track raw inconsistencies before filtering
    mask_negative_followup = (
        df[last_observed_col].notna()
        & df[baseline_col].notna()
        & (df[last_observed_col] - df[baseline_col] < 0)
    )

    mask_neg_censor = (
        (df[event_col] == 0)
        & df[last_observed_col].notna()
        & df[baseline_col].notna()
        & (df[last_observed_col] - df[baseline_col] < 0)
    )

    mask_neg_event_time = (
        (df[event_col] == 1)
        & df[time_col].notna()
        & df[baseline_col].notna()
        & (df[time_col] - df[baseline_col] < 0)
    )

    # Keep only rows where the event indicator is known
    mask_event_known = df[event_col].notna()
    df = df.loc[mask_event_known].copy()

    # For event rows, the event time must be known
    is_event = df[event_col].astype(int) == 1
    mask_event_time_known = (~is_event) | df[time_col].notna()
    df = df.loc[mask_event_time_known].copy()

    # Recompute event indicator after filtering
    is_event = df[event_col].astype(int) == 1

    # Survival time
    time_values = np.where(
        is_event,
        df[time_col] - df[baseline_col],
        df[last_observed_col] - df[baseline_col],
    ).astype(float)

    # Remove invalid times
    if drop_nonpositive_times:
        mask_valid_time = time_values >= 0
        df = df.loc[mask_valid_time].copy()
        time_values = time_values[mask_valid_time]

    # Remove non-finite times
    mask_finite = np.isfinite(time_values)
    df = df.loc[mask_finite].copy()
    time_values = time_values[mask_finite]

    # Clamp zeros / tiny values
    time_values = np.maximum(time_values, min_time)

    is_event = (df[event_col].astype(int) == 1).to_numpy()

    y = Surv.from_arrays(
        event=is_event.astype(bool),
        time=time_values,
        name_event=event_name,
        name_time="Time_years",
    )

    if return_report:
        report = {
            "n_initial": n0,
            "n_after_event_known": int(mask_event_known.sum()),
            "n_after_event_time_known": len(df),
            "n_final": len(df),
            "dropped_total": n0 - len(df),
            "n_negative_followup_raw": int(mask_negative_followup.sum()),
            "n_negative_censor_time_raw": int(mask_neg_censor.sum()),
            "n_negative_event_time_raw": int(mask_neg_event_time.sum()),
            "outcome": outcome,
        }
        return df.reset_index(drop=True), y, report

    return df.reset_index(drop=True), y




def build_feature_matrix(df, keep_cols=None):
    """
    Build a numeric feature matrix, excluding:
      - target columns
      - ID columns
      - Age_v* columns
      - last_observed_age

    If keep_cols is provided, only those columns are retained.
    """
    age_v_cols = [c for c in df.columns if c.startswith("Age_v")]
    leakage_cols = ["last_observed_age"]
    drop_cols = TARGET_COLS + ID_COLS + age_v_cols + leakage_cols

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include="number")

    if keep_cols is not None:
        keep_cols = [c for c in keep_cols if c in X.columns]
        X = X[keep_cols]

    return X


def run_pipeline():
    """
    End-to-end training and prediction workflow.

    Assumes apply_final_feature_engineering(df, task=...) is already defined.
    """
    df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Feature engineering
    train_df_hep = apply_final_feature_engineering(df, task="hep")
    train_df_death = apply_final_feature_engineering(df, task="death")

    # Target processing
    df_hep, y_hep = prepare_survival_targets_robust(train_df_hep, outcome="hepatic")
    df_death, y_death = prepare_survival_targets_robust(train_df_death, outcome="death")

    # Feature matrices
    X_hep_raw = build_feature_matrix(df_hep)
    X_death_raw = build_feature_matrix(df_death)

    print(f"Raw feature count: {X_hep_raw.shape[1]}")

    # Missing-rate filter fitted on train and applied consistently
    missing_rate_hep = X_hep_raw.isna().mean()
    missing_rate_death = X_death_raw.isna().mean()

    keep_hep = missing_rate_hep[missing_rate_hep <= MAX_MISSING_RATE].index.tolist()
    keep_death = missing_rate_death[missing_rate_death <= MAX_MISSING_RATE].index.tolist()

    X_hep_aln = X_hep_raw[keep_hep]
    X_death_aln = X_death_raw[keep_death]

    hep_events = int((df_hep["evenements_hepatiques_majeurs"] == 1).sum())
    death_events = int((df_death["death"] == 1).sum())

    print(
        f"Hepatic features after missing filter: {len(keep_hep)} "
        f"(EPV = {hep_events}/{len(keep_hep)} = {hep_events / max(len(keep_hep), 1):.2f})"
    )
    print(
        f"Death features after missing filter: {len(keep_death)} "
        f"(EPV = {death_events}/{len(keep_death)} = {death_events / max(len(keep_death), 1):.2f})"
    )

    # Cross-validation
    print("Fitting Death — RSF (CV)...")
    ci_rsf_death_mean, ci_rsf_death_std = cv_rsf(X_death_aln.astype("float32"), y_death)
    print(f"  RSF C-index (CV): {ci_rsf_death_mean:.4f} ± {ci_rsf_death_std:.4f}")

    print("Fitting Hepatic — RSF (CV)...")
    ci_rsf_hep_mean, ci_rsf_hep_std = cv_rsf(X_hep_aln.astype("float32"), y_hep)
    print(f"  RSF C-index (CV): {ci_rsf_hep_mean:.4f} ± {ci_rsf_hep_std:.4f}")

    # Final models
    print("Training Death Model...")
    model_death = make_rsf_pipeline()
    model_death.fit(X_death_aln, y_death)

    print("Training Hepatic Model...")
    model_hep = make_rsf_pipeline()
    model_hep.fit(X_hep_aln, y_hep)

    # Optional exploration
    # from skrub import TableReport
    # TableReport(X_hep_aln)
    # TableReport(X_death_aln)

    # Test feature engineering
    test_df_hep = apply_final_feature_engineering(test_df, task="hep")
    test_df_death = apply_final_feature_engineering(test_df, task="death")

    X_pred_raw_hep = build_feature_matrix(test_df_hep)
    X_pred_raw_death = build_feature_matrix(test_df_death)

    # Align columns to training matrices
    X_pred_hep = X_pred_raw_hep[keep_hep]
    X_pred_death = X_pred_raw_death[keep_death]

    # Predictions
    preds_hep = model_hep.predict(X_pred_hep)
    preds_death = model_death.predict(X_pred_death)

    submission = pd.DataFrame(
        {
            "trustii_id": test_df["trustii_id"].values,
            "risk_hepatic_event": preds_hep,
            "risk_death": preds_death,
        }
    )

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(submission)} predictions to {OUTPUT_PATH}")
    print(submission.head())

    return submission


if __name__ == "__main__":
    run_pipeline()