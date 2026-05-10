import re
import warnings

import numpy as np
import pandas as pd
#import skrub
import sksurv

from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance


warnings.filterwarnings("ignore")

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


def create_change_scores(df, baseline_visit=1):
    df = df.copy()

    longitudinal_cols = [col for col in df.columns if "_v" in col]
    prefixes = sorted(set(col.rsplit("_v", 1)[0] for col in longitudinal_cols))

    for prefix in prefixes:
        baseline_col = f"{prefix}_v{baseline_visit}"
        if baseline_col not in df.columns:
            continue

        visit_cols = [col for col in df.columns if col.startswith(f"{prefix}_v")]

        for col in visit_cols:
            if col == baseline_col:
                continue

            visit_num = col.split("_v")[-1]
            new_col = f"change_{prefix}_v{visit_num}"

            baseline = df[baseline_col]
            current = df[col]

            df[new_col] = np.where(
                baseline.notna() & current.notna() & (baseline != 0),
                (current - baseline) / baseline * 100.0,
                np.nan,
            )

    return df


def compute_patient_slopes_theilsen(df, biomarkers=BIOMARKERS, max_visit=22):
    df = df.copy()

    def theil_sen_slope(x, y):
        slopes = []
        n = len(x)

        for i in range(n):
            for j in range(i + 1, n):
                if x[j] != x[i]:
                    slopes.append((y[j] - y[i]) / (x[j] - x[i]))

        if len(slopes) == 0:
            return np.nan

        return np.median(slopes)

    for biom in biomarkers:
        values = []
        ages = []

        for v in range(1, max_visit + 1):
            val_col = f"{biom}_v{v}"
            age_col = f"Age_v{v}"

            if val_col in df.columns and age_col in df.columns:
                values.append(df[val_col])
                ages.append(df[age_col])

        if len(values) < 2:
            df[f"slope_ts_{biom}"] = np.nan
            continue

        values = pd.concat(values, axis=1)
        ages = pd.concat(ages, axis=1)

        slopes = []
        for i in range(len(df)):
            x = ages.iloc[i].to_numpy(dtype=float)
            y = values.iloc[i].to_numpy(dtype=float)

            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]

            if len(x) < 2:
                slopes.append(np.nan)
            else:
                slopes.append(theil_sen_slope(x, y))

        df[f"slope_ts_{biom}"] = slopes

    return df


def count_non_na_longitudinal(df):
    df = df.copy()

    longitudinal_cols = [col for col in df.columns if "_v" in col]
    prefixes = sorted(set(col.rsplit("_v", 1)[0] for col in longitudinal_cols))

    for prefix in prefixes:
        cols = [col for col in df.columns if col.startswith(f"{prefix}_v")]
        if cols:
            df[f"n_{prefix}"] = df[cols].notna().sum(axis=1)

    return df


def add_first_last_visits(df, biomarkers=BIOMARKERS, max_visit=22):
    df = df.copy()

    for biom in biomarkers:
        biom_cols = [
            f"{biom}_v{i}"
            for i in range(1, max_visit + 1)
            if f"{biom}_v{i}" in df.columns
        ]

        if not biom_cols:
            df[f"first_{biom}"] = np.nan
            df[f"last_{biom}"] = np.nan
            continue

        data = df[biom_cols]
        mask = data.notna()
        visit_numbers = np.array([int(col.split("_v")[1]) for col in biom_cols])

        first_visit = mask.apply(
            lambda row: visit_numbers[row.values].min() if row.any() else 0,
            axis=1,
        )
        last_visit = mask.apply(
            lambda row: visit_numbers[row.values].max() if row.any() else 0,
            axis=1,
        )

        df[f"first_{biom}"] = first_visit
        df[f"last_{biom}"] = last_visit

    return df


def add_visit_metrics(df, max_visit=22):
    df = df.copy()

    for v in range(1, max_visit + 1):
        age_col = f"Age_v{v}"
        prev_age_col = f"Age_v{v - 1}"

        if v == 1:
            df[f"age_diff_v{v}"] = np.nan
        elif age_col in df.columns and prev_age_col in df.columns:
            df[f"age_diff_v{v}"] = np.where(
                df[age_col].notna() & df[prev_age_col].notna(),
                df[age_col] - df[prev_age_col],
                np.nan,
            )
        else:
            df[f"age_diff_v{v}"] = np.nan

    return df


def add_patient_summary_metrics(df, biomarkers=BIOMARKERS, max_visit=22):
    df = df.copy()

    age_cols = [
        f"Age_v{i}"
        for i in range(1, max_visit + 1)
        if f"Age_v{i}" in df.columns
    ]

    if age_cols:
        age_diffs = [
            df[age_cols[i]] - df[age_cols[i - 1]]
            for i in range(1, len(age_cols))
        ]
        if age_diffs:
            df["mean_age_diff"] = pd.concat(age_diffs, axis=1).mean(axis=1, skipna=True)
        else:
            df["mean_age_diff"] = np.nan

        df["age_range"] = (
            df[age_cols].max(axis=1, skipna=True) - df[age_cols].min(axis=1, skipna=True)
        )
    else:
        df["mean_age_diff"] = np.nan
        df["age_range"] = np.nan

    longitudinal_cols = [col for col in df.columns if "_v" in col]
    n_measures = []

    for v in range(1, max_visit + 1):
        visit_cols = [col for col in longitudinal_cols if col.endswith(f"_v{v}")]
        if visit_cols:
            n_measures.append(df[visit_cols].notna().sum(axis=1))

    if n_measures:
        df["mean_n_measures"] = pd.concat(n_measures, axis=1).mean(axis=1, skipna=True)
    else:
        df["mean_n_measures"] = np.nan

    for biom in biomarkers:
        biom_cols = [
            f"{biom}_v{i}"
            for i in range(1, max_visit + 1)
            if f"{biom}_v{i}" in df.columns
        ]

        if not biom_cols:
            continue

        df[f"min_{biom}"] = df[biom_cols].min(axis=1, skipna=True)
        df[f"max_{biom}"] = df[biom_cols].max(axis=1, skipna=True)
        df[f"{biom}_range"] = (df[f"max_{biom}"] - df[f"min_{biom}"]).fillna(0)

        diffs = [
            df[biom_cols[i]] - df[biom_cols[i - 1]]
            for i in range(1, len(biom_cols))
        ]
        if diffs:
            df[f"mean_{biom}_diff"] = pd.concat(diffs, axis=1).mean(axis=1, skipna=True)
        else:
            df[f"mean_{biom}_diff"] = np.nan

    return df


def add_exam_within_2y_count(df, age_prefix="Age_v", out_col="n_exams_within_2y"):
    df = df.copy()

    age_cols = [c for c in df.columns if c.startswith(age_prefix)]
    if not age_cols:
        raise ValueError(f"No columns found with prefix '{age_prefix}'")

    age_cols = sorted(age_cols, key=lambda x: int(x.split("_v")[-1]))

    def count_gaps(row):
        ages = row[age_cols].dropna().astype(float).values
        if len(ages) < 2:
            return 0

        gaps = np.diff(np.sort(ages))
        return int(np.sum(gaps <= 2))

    df[out_col] = df.apply(count_gaps, axis=1)
    return df


def create_metabolic_syndrome(df, visits=range(1, 23)):
    df = df.copy()
    ms_cols = []

    for v in visits:
        bmi_col = f"BMI_v{v}"
        gluc_col = f"gluc_fast_v{v}"
        trig_col = f"triglyc_v{v}"

        if not all(c in df.columns for c in [bmi_col, gluc_col, trig_col]):
            continue

        crit_bmi = (df[bmi_col] >= 30).astype(float)
        crit_gluc = (df[gluc_col] >= 5.6).astype(float)
        crit_trig = (df[trig_col] >= 1.5).astype(float)

        crit_htn = (
            (df["Hypertension"] == 1).astype(float)
            if "Hypertension" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        crit_t2dm = (
            (df["T2DM"] == 1).astype(float)
            if "T2DM" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        crit_dyslip = (
            (df["Dyslipidaemia"] == 1).astype(float)
            if "Dyslipidaemia" in df.columns
            else pd.Series(0.0, index=df.index)
        )

        col_name = f"metabolic_syndrome_v{v}"
        df[col_name] = crit_bmi + crit_gluc + crit_trig + crit_htn + crit_t2dm + crit_dyslip
        ms_cols.append(col_name)

    if ms_cols:
        df["metabolic_syndrome_max"] = df[ms_cols].max(axis=1)
        df["metabolic_syndrome_mean"] = df[ms_cols].mean(axis=1)
        df = df.drop(columns=ms_cols)

    return df




def create_alt_ast_ratio(df, max_visit=21, drop_originals=True):
    df = df.copy()

    for v in range(1, max_visit + 1):
        alt_col = f"alt_v{v}"
        ast_col = f"ast_v{v}"
        new_col = f"alt_ast_v{v}"

        if alt_col in df.columns and ast_col in df.columns:
            df[new_col] = np.where(
                df[alt_col].notna() & df[ast_col].notna() & (df[ast_col] != 0),
                df[alt_col] / df[ast_col],
                np.nan,
            )
        else:
            df[new_col] = np.nan

    if drop_originals:
        cols_to_drop = [
            col
            for col in df.columns
            if col.startswith("alt_v") or col.startswith("ast_v")
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


def create_alt_ast_max(df, max_visit=21):
    df = df.copy()
    ratio_cols = []

    for v in range(1, max_visit + 1):
        alt_col = f"alt_v{v}"
        ast_col = f"ast_v{v}"
        new_col = f"alt_ast_v{v}"

        if alt_col in df.columns and ast_col in df.columns:
            df[new_col] = np.where(
                df[alt_col].notna() & df[ast_col].notna() & (df[ast_col] != 0),
                df[alt_col] / df[ast_col],
                np.nan,
            )
        else:
            df[new_col] = np.nan

        ratio_cols.append(new_col)

    df["alt_ast_max"] = df[ratio_cols].max(axis=1, skipna=True) if ratio_cols else np.nan
    return df


def create_cirrhosis_proxy(df, visits=range(1, 23)):
    df = df.copy()
    cirr_cols = []

    for v in visits:
        bili_col = f"bilirubin_v{v}"
        plt_col = f"plt_v{v}"
        ggt_col = f"ggt_v{v}"
        stiff_col = f"fibs_stiffness_med_BM_1_v{v}"
        fibro_col = f"fibrotest_BM_2_v{v}"

        available = [c for c in [bili_col, plt_col, ggt_col, stiff_col, fibro_col] if c in df.columns]
        if len(available) < 3:
            continue

        score = pd.Series(0.0, index=df.index)

        if bili_col in df.columns:
            score += (df[bili_col] > 20).astype(float)
        if plt_col in df.columns:
            score += (df[plt_col] < 150).astype(float)
        if ggt_col in df.columns:
            score += (df[ggt_col] > 60).astype(float)
        if stiff_col in df.columns:
            score += (df[stiff_col] > 10).astype(float)
        if fibro_col in df.columns:
            score += (df[fibro_col] > 0.7).astype(float)

        col_name = f"cirrhosis_score_v{v}"
        df[col_name] = score
        cirr_cols.append(col_name)

    if cirr_cols:
        df["cirrhosis_score_max"] = df[cirr_cols].max(axis=1)
        df = df.drop(columns=cirr_cols)

    return df


def add_fibrotest_stage_features(
    df,
    visits=range(1, 23),
    score_prefix="fibrotest_BM_2",
    keep_visit_columns=False,
):
    df = df.copy()
    coarse_cols = []

    def _coarse_stage(x):
        if pd.isna(x):
            return 0
        if x < 0.21:
            return 0
        if x < 0.31:
            return 1
        if x < 0.58:
            return 2
        if x < 0.74:
            return 3
        return 4

    for v in visits:
        col = f"{score_prefix}_v{v}"
        if col not in df.columns:
            continue

        coarse_col = f"fibrosis_stage_coarse_v{v}"
        df[coarse_col] = df[col].apply(_coarse_stage)
        coarse_cols.append(coarse_col)

    score_cols = [
        f"{score_prefix}_v{v}"
        for v in visits
        if f"{score_prefix}_v{v}" in df.columns
    ]

    if score_cols:
        df["fibrotest_max"] = df[score_cols].max(axis=1, skipna=True).fillna(0)

    if coarse_cols:
        df["fibrosis_stage_max"] = df[coarse_cols].max(axis=1)
        df["fibrosis_F3_or_more"] = (df[coarse_cols] >= 3).sum(axis=1)
        df["fibrosis_F4_any"] = (df[coarse_cols] == 4).any(axis=1).astype(int)

    if not keep_visit_columns:
        df = df.drop(columns=coarse_cols, errors="ignore")

    return df


def extract_last_observed_age(df):
    df = df.copy()

    age_cols = sorted(
        [c for c in df.columns if re.match(r"Age_v\d+", c)],
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )

    if not age_cols:
        df["num_visits"] = 0
        df["last_observed_age"] = np.nan
        return df

    age_matrix = df[age_cols].to_numpy(dtype=float)
    mask_age = ~np.isnan(age_matrix)

    df["num_visits"] = mask_age.sum(axis=1)

    last_idx_age = np.where(
        mask_age.any(axis=1),
        age_matrix.shape[1] - 1 - np.argmax(mask_age[:, ::-1], axis=1),
        -1,
    )

    df["last_observed_age"] = np.where(
        last_idx_age >= 0,
        age_matrix[np.arange(len(df)), last_idx_age],
        np.nan,
    )

    return df


def extract_last_available_values(df, remove=True, n_last=5):
    df = df.copy()

    age_cols = sorted(
        [c for c in df.columns if re.match(r"Age_v\d+", c)],
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )

    if "num_visits" not in df.columns:
        if age_cols:
            df["num_visits"] = df[age_cols].notna().sum(axis=1)
        else:
            df["num_visits"] = 0

    if "last_observed_age" not in df.columns:
        if age_cols:
            age_matrix = df[age_cols].to_numpy(dtype=float)
            mask_age = ~np.isnan(age_matrix)
            last_idx_age = np.where(
                mask_age.any(axis=1),
                age_matrix.shape[1] - 1 - np.argmax(mask_age[:, ::-1], axis=1),
                -1,
            )
            df["last_observed_age"] = np.where(
                last_idx_age >= 0,
                age_matrix[np.arange(len(df)), last_idx_age],
                np.nan,
            )
        else:
            df["last_observed_age"] = np.nan

    visit_cols_to_drop = list(age_cols) if remove else []

    for biom in BIOMARKERS:
        biom_cols = sorted(
            [c for c in df.columns if re.match(fr"{re.escape(biom)}_v\d+", c)],
            key=lambda x: int(re.findall(r"\d+", x)[0]),
        )

        if not biom_cols:
            continue

        biom_matrix = df[biom_cols].to_numpy(dtype=float)
        reversed_matrix = np.flip(biom_matrix, axis=1)

        last_values = [[] for _ in range(n_last)]

        for i in range(reversed_matrix.shape[0]):
            row = reversed_matrix[i]
            vals = row[~np.isnan(row)][:n_last]
            vals = list(vals) + [np.nan] * (n_last - len(vals))

            for j in range(n_last):
                last_values[j].append(vals[j])

        for j in range(n_last):
            col_name = f"{biom}_last" if j == 0 else f"{biom}_last_minus{j}"
            df[col_name] = last_values[j]

        if remove:
            visit_cols_to_drop.extend(biom_cols)

    df = df.drop(columns=visit_cols_to_drop, errors="ignore")

    event_cols = [
        "evenements_hepatiques_majeurs",
        "evenements_hepatiques_age_occur",
        "death",
        "death_age_occur",
    ]
    event_cols = [c for c in event_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in event_cols]
    df = df[other_cols + event_cols]

    return df


def extract_last_available_values(df, remove = True):
        df = df.copy()

        # -----------------------------
            # 1. Age columns
        # -----------------------------
        age_cols = sorted(
            [c for c in df.columns if re.match(r"Age_v\d+", c)],
            key=lambda x: int(re.findall(r"\d+", x)[0])
        )

        age_matrix = df[age_cols].to_numpy()
        mask_age = ~np.isnan(age_matrix)

        # num_visits
        #df["num_visits"] = mask_age.sum(axis=1)

        # last observed age (indépendant maintenant)
        rev_idx_age = np.argmax(mask_age[:, ::-1], axis=1)
        last_idx_age = age_matrix.shape[1] - 1 - rev_idx_age
        last_idx_age[df["num_visits"] == 0] = -1

        #df["last_observed_age"] = np.where(
            #last_idx_age >= 0,
            #age_matrix[np.arange(len(df)), last_idx_age],
            #np.nan
        #)

        # -----------------------------
        # 2. Biomarkers
        # -----------------------------
        biomarkers = [
            "BMI", "alt", "ast", "bilirubin", "chol", "ggt",
            "gluc_fast", "plt", "triglyc",
            "aixp_aix_result_BM_3",
            "fibrotest_BM_2",
            "fibs_stiffness_med_BM_1"
        ]

        visit_cols_to_drop = []


        N_LAST = 5 # 

        for biom in biomarkers:
            biom_cols = sorted(
                [c for c in df.columns if re.match(fr"{biom}_v\d+", c)],
                key=lambda x: int(re.findall(r"\d+", x)[0])
            )

            if not biom_cols:
                continue

            biom_matrix = df[biom_cols].to_numpy()

            reversed_matrix = np.flip(biom_matrix, axis=1)

            last_values = [[] for _ in range(N_LAST)]

            for i in range(reversed_matrix.shape[0]):
                row = reversed_matrix[i]
                vals = row[~np.isnan(row)][:N_LAST]  # 👈 prend les N dernières non-NaN

                # padding
                vals = list(vals) + [np.nan] * (N_LAST - len(vals))

                for j in range(N_LAST):
                    last_values[j].append(vals[j])
            if remove:        


                for j in range(N_LAST):
                    if j == 0:
                        col_name = f"{biom}_last"
                    else:
                        col_name = f"{biom}_last_minus{j}"

                    df[col_name] = last_values[j]

            visit_cols_to_drop.extend(biom_cols)

        # -----------------------------
        # 3. Drop longitudinal columns
        # -----------------------------
        df.drop(columns=visit_cols_to_drop, inplace=True, errors="ignore")

        # ---------------------------
        # -----------------------------
        event_cols = [
            "evenements_hepatiques_majeurs",
            "evenements_hepatiques_age_occur",
            "death",
            "death_age_occur"
        ]

        event_cols = [c for c in event_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in event_cols]

        df = df[other_cols + event_cols]

        return df


def apply_final_feature_engineering(df, task="hep", dummy_cols=None):
    """
    Apply the full feature-engineering pipeline to a single DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    task : {"hep", "death"}
        Selects the feature-engineering path.
    dummy_cols : list[str] or None
        Columns to one-hot encode before returning the final DataFrame.

    Returns
    -------
    pandas.DataFrame
        Feature-engineered DataFrame.
    """
    df = df.copy()
    task = task.lower().strip()

    if task == "hep":
        df = add_patient_summary_metrics(df)
        df = add_first_last_visits(df)
        df = add_visit_metrics(df)
        df = create_metabolic_syndrome(df)
        df = create_alt_ast_ratio(df)
        df = add_fibrotest_stage_features(df)
        df = create_cirrhosis_proxy(df)
        df = extract_last_observed_age(df)
        df = extract_last_available_values(df, remove=False)

    elif task == "death":
        df = add_patient_summary_metrics(df)
        df = count_non_na_longitudinal(df)
        df = create_change_scores(df)
        df = compute_patient_slopes_theilsen(df)
        df = add_patient_summary_metrics(df)
        df = add_first_last_visits(df)
        df = add_visit_metrics(df)
        df = add_exam_within_2y_count(df)
        df = add_fibrotest_stage_features(df)
        df = create_cirrhosis_proxy(df)
        df = extract_last_observed_age(df)
        df = extract_last_available_values(df, remove=True)

    else:
        raise ValueError("task must be either 'hep' or 'death'")

    if dummy_cols is None:
        dummy_cols = [
            "gender",
            "T2DM",
            "Hypertension",
            "Dyslipidaemia",
            "bariatric_surgery",
        ]

    dummy_cols = [c for c in dummy_cols if c in df.columns]
    if dummy_cols:
        df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)

    return df

