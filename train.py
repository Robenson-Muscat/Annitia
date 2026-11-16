import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from preprocessing import apply_final_feature_engineering

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "data/processed_wide/train.csv"
TEST_PATH = "data/processed_wide/val_test.csv"
OUTPUT_PATH = "submission_annitia.csv"

TARGET_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]

ID_COLS = ["patient_id_anon", "trustii_id"]

MAX_MISSING_RATE = 0.5

RANDOM_STATE = 26

OUTER_SPLITS = 4
INNER_SPLITS = 3

N_ITER_SEARCH = 30

N_JOBS = -1


# ============================================================
# FEATURE SELECTION
# ============================================================

class MissingRateFilter(BaseEstimator, TransformerMixin):
    """
    Remove features with a missing rate above the threshold.

    The filter is fitted independently inside each CV training fold.
    """

    def __init__(self, threshold=0.50):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        missing_rates = X.isna().mean()

        self.selected_features_ = missing_rates[missing_rates <= self.threshold].index.tolist()

        # Safety: retain at least one feature
        if len(self.selected_features_) == 0:
            self.selected_features_ = [missing_rates.idxmin()]

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        return X.reindex(columns=self.selected_features_, fill_value=np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


class MissingRateFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.50):
        self.threshold = threshold

    def fit(self, X, y=None):

        X = pd.DataFrame(X).copy()

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        missing_rates = X.isna().mean()

        self.selected_features_ = missing_rates[
            (missing_rates <= self.threshold) &
            (missing_rates < 1.0)
        ].index.tolist()

        if len(self.selected_features_) == 0:
            raise ValueError("No features available after missing-rate filtering.")

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        X = X.reindex(columns=self.selected_features_)

        return X

    def get_feature_names_out(self, input_features=None):

        return np.asarray(self.selected_features_, dtype=object)


# ============================================================
# SURVIVAL TARGET CONSTRUCTION
# ============================================================

def prepare_survival_targets_robust(
    df: pd.DataFrame,
    outcome: str = "hepatic",
    baseline_col: str = "Age_v1",
    last_observed_col: str = "last_observed_age",
    min_time: float = 1e-1,
    drop_nonpositive_times: bool = True,
    return_report: bool = False,
):

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
        raise ValueError(f"Unknown outcome: {outcome}")

    required_cols = [
        event_col,
        time_col,
        baseline_col,
        last_observed_col,]

    missing_required = [c for c in required_cols if c not in df.columns]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    n_initial = len(df)

    # Event indicator must be known
    df = df.loc[df[event_col].notna()].copy()

    # Event time must be known for observed events
    is_event = df[event_col].astype(int) == 1

    df = df.loc[
        (~is_event) | df[time_col].notna()
    ].copy()

    is_event = df[event_col].astype(int) == 1

    # Calculate follow-up time
    time_values = np.where(
        is_event,
        df[time_col] - df[baseline_col],
        df[last_observed_col] - df[baseline_col],
    ).astype(float)

    # Remove negative times
    if drop_nonpositive_times:
        valid = time_values >= 0

        df = df.loc[valid].copy()
        time_values = time_values[valid]

    # Remove non-finite times
    valid = np.isfinite(time_values)

    df = df.loc[valid].copy()
    time_values = time_values[valid]

    # Avoid zero survival times
    time_values = np.maximum(time_values, min_time)

    is_event = (df[event_col].astype(int) == 1).to_numpy()

    y = Surv.from_arrays(
        event=is_event.astype(bool),
        time=time_values,
        name_event=event_name,
        name_time="Time_years")

    df = df.reset_index(drop=True)

    if return_report:

        report = {
            "n_initial": n_initial,
            "n_final": len(df),
            "dropped_total": n_initial - len(df),
            "n_events": int(is_event.sum()),
            "outcome": outcome,
        }

        return df, y, report

    return df, y


# ============================================================
# FEATURE MATRIX
# ============================================================

def build_feature_matrix(df):

    age_v_cols = [c for c in df.columns if c.startswith("Age_v")]

    leakage_cols = ["last_observed_age"]

    drop_cols = (
        TARGET_COLS
        + ID_COLS
        + age_v_cols
        + leakage_cols)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    X = X.select_dtypes(include="number")

    return X.astype("float32")


# ============================================================
# RSF PIPELINE
# ============================================================

def make_rsf_pipeline():

    return Pipeline(
        steps=[
            ("missing_filter", MissingRateFilter(threshold=MAX_MISSING_RATE)),

            ("imputer", SimpleImputer(strategy="median")), 

            ("rsf", RandomSurvivalForest(n_estimators=300, n_jobs=1, random_state=RANDOM_STATE)),
        ])


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

def get_rsf_param_distributions():

    return {
        "rsf__n_estimators": [
            200,
            300,
            500,
        ],

        "rsf__min_samples_leaf": [
            5,
            10,
            15,
            20,
            30,
        ],

        "rsf__min_samples_split": [
            10,
            20,
            30,
            #40,
            60,
        ],

        "rsf__max_features": [
            "sqrt",
            "log2",
            0.3,
            #0.5,
            #0.7,
        ],

        "rsf__max_depth": [
            None,
            5,
            10,
            15,
            20,
        ],

        "rsf__max_leaf_nodes": [
            None,
            10,
            20,
            40,
        ],
    }


# ============================================================
# SURVIVAL SCORING
# ============================================================

def survival_c_index(estimator, X, y):

    predictions = estimator.predict(X)

    event_name = y.dtype.names[0]
    time_name = y.dtype.names[1]

    ci = concordance_index_censored(y[event_name], y[time_name],predictions)[0]

    return float(ci)


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================

def nested_cv_rsf(
    X,
    y,
    outcome_name="Outcome",
    outer_splits=OUTER_SPLITS,
    inner_splits=INNER_SPLITS,
):

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE,)

    outer_scores = []
    best_params_per_fold = []

    print("\n" + "=" * 65)
    print(f"NESTED CROSS-VALIDATION: {outcome_name}")
    print("=" * 65)

    selected_features_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):

        print(f"\nOuter fold {fold}/{outer_splits}" )

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[test_idx]

        y_train = y[train_idx]
        y_valid = y[test_idx]

        # Inner CV
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE + fold,)

        search = RandomizedSearchCV(
            
            estimator=make_rsf_pipeline(),

            param_distributions=get_rsf_param_distributions(),

            n_iter=N_ITER_SEARCH,

            scoring=survival_c_index,

            cv=inner_cv,

            refit=True,

            n_jobs=N_JOBS,

            random_state=RANDOM_STATE + fold,

            verbose=0,

            error_score="raise",
        )

        # Hyperparameter optimization on outer training data only
        search.fit(X_train, y_train)

        # Best model fitted on the complete outer training fold
        best_model = search.best_estimator_

        selected_features = best_model.named_steps["missing_filter"].selected_features_
        
        print(f"Number of selected features: {len(selected_features)}")
        selected_features_per_fold.append(set(selected_features))

        # Evaluation on untouched outer validation fold
        predictions = best_model.predict(X_valid)

        ci = concordance_index_censored(
            y_valid[y.dtype.names[0]],
            y_valid[y.dtype.names[1]],
            predictions,
        )[0]

        outer_scores.append(ci)

        best_params_per_fold.append(search.best_params_)

        print(f"Inner best C-index: {search.best_score_:.4f}")

        print(f"Outer C-index:  {ci:.4f}")

        print(f"Best parameters:  {search.best_params_}")


    all_features = set.union(*selected_features_per_fold)

    feature_stability = pd.DataFrame({
        "feature": list(all_features),
        "selection_frequency": [
            sum(
                feature in fold_features
                for fold_features in selected_features_per_fold
            ) / outer_splits
            for feature in all_features
        ]
    })
    
    feature_stability = feature_stability.sort_values("selection_frequency", ascending=False)
    
    print("\nFeature selection stability:")
    print(feature_stability.to_string(index=False))

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores, ddof=1)

    print("\n" + "-" * 65)
    print(f"{outcome_name} - NESTED CV RESULTS")
    print("-" * 65)

    print(f"Outer C-index: "
        f"{mean_score:.4f} ± {std_score:.4f}")

    print(f"Individual folds: "
        f"{np.round(outer_scores, 4)}")

    return {
        "mean": mean_score,
        "std": std_score,
        "scores": outer_scores,
        "best_params": best_params_per_fold,
    }


# ============================================================
# FINAL MODEL TRAINING
# ============================================================

def train_final_rsf(X, y, outcome_name="Outcome"):

    print("\n" + "=" * 65)
    print(f"FINAL MODEL TRAINING: {outcome_name}")
    print("=" * 65)

    inner_cv = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=make_rsf_pipeline(),

        param_distributions=get_rsf_param_distributions(),

        n_iter=N_ITER_SEARCH,

        scoring=survival_c_index,

        cv=inner_cv,

        refit=True,

        n_jobs=N_JOBS,

        random_state=RANDOM_STATE,

        verbose=1,

        error_score="raise",
    )

    search.fit(X, y)

    print(f"\nBest inner CV C-index: "
        f"{search.best_score_:.4f}")

    print(f"Best parameters: "
        f"{search.best_params_}")

    # Already refitted on all available training data
    final_model = search.best_estimator_

    return final_model, search


# ============================================================
# COMPLETE WORKFLOW
# ============================================================

def train_validation_pipeline():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("Loading datasets...")

    df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("Feature engineering...")

    train_df_hep = apply_final_feature_engineering(df.copy(),task="hep")
    train_df_death = apply_final_feature_engineering(df.copy(), task="death")
    
    test_df_hep = apply_final_feature_engineering(test_df.copy(), task="hep")
    test_df_death = apply_final_feature_engineering(test_df.copy(), task="death")

    # --------------------------------------------------------
    # Survival targets
    # --------------------------------------------------------

    print("Preparing survival targets...")

    df_hep, y_hep, report_hep = prepare_survival_targets_robust(train_df_hep, outcome="hepatic", return_report=True,)
    df_death, y_death, report_death = prepare_survival_targets_robust(train_df_death, outcome="death",return_report=True)

    print("\nHepatic target report:")
    print(report_hep)

    print("\nDeath target report:")
    print(report_death)

    # --------------------------------------------------------
    # Feature matrices
    # --------------------------------------------------------

    X_hep = build_feature_matrix(df_hep)
    X_death = build_feature_matrix(df_death)

    X_pred_hep = build_feature_matrix(test_df_hep)
    X_pred_death = build_feature_matrix(test_df_death)

    # Align train and test columns.
    # Missing columns are filled with NaN and handled by
    # the imputer inside the pipeline.
    X_pred_hep = X_pred_hep.reindex(
        columns=X_hep.columns)

    X_pred_death = X_pred_death.reindex(
        columns=X_death.columns)

    print("\nFeature matrix dimensions:")
    print(f"Hepatic: {X_hep.shape}")
    print(f"Death:   {X_death.shape}")

    # --------------------------------------------------------
    # 5. Event statistics
    # --------------------------------------------------------

    hep_events = int(y_hep[y_hep.dtype.names[0]].sum())

    death_events = int(y_death[y_death.dtype.names[0]].sum())

    print("\nEvent statistics:")

    print(f"Hepatic events: {hep_events}")

    print(f"Death events:   {death_events}")

    # --------------------------------------------------------
    # Nested CVs: hepatic endpoint and death endpoint
    # --------------------------------------------------------

    nested_hep = nested_cv_rsf(X=X_hep, y=y_hep, outcome_name="Hepatic")
    nested_death = nested_cv_rsf(X=X_death, y=y_death, outcome_name="Death")

    # --------------------------------------------------------
    # Finals models: hepatic and death
    # --------------------------------------------------------

    model_hep, search_hep = train_final_rsf(X=X_hep, y=y_hep, outcome_name="Hepatic",)
    model_death, search_death = train_final_rsf(X=X_death, y=y_death, outcome_name="Death",)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    preds_hep = model_hep.predict(X_pred_hep)

    preds_death = model_death.predict(X_pred_death)

    # --------------------------------------------------------
    # Submission

    submission = pd.DataFrame({
            "trustii_id": test_df["trustii_id"].values,

            "risk_hepatic_event": preds_hep,

            "risk_death": preds_death})

    submission.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(f"Hepatic nested CV: " 
          f"{nested_hep['mean']:.4f} "
          f"± {nested_hep['std']:.4f}")

    print(f"Death nested CV:   "
        f"{nested_death['mean']:.4f} "
        f"± {nested_death['std']:.4f}")

    print(f"\nSubmission saved: {OUTPUT_PATH}")

    print(submission.head())

    return {
        "submission": submission,

        "model_hep": model_hep,

        "model_death": model_death,

        "nested_cv_hep": nested_hep,

        "nested_cv_death": nested_death,

        "search_hep": search_hep,

        "search_death": search_death,
    }


if __name__ == "__main__":

    results = train_validation_pipeline()

