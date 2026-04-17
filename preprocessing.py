import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import SimpleImputer, IterativeImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold, GridSearchCV

    return IterativeImputer, KFold, Pipeline


@app.cell
def _():
    import skrub
    import sksurv

    return (skrub,)


@app.cell
def _():
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest

    return RandomSurvivalForest, Surv, concordance_index_censored


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np


    import re

    return np, re


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _():
    import lifelines

    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Load data
    """)
    return


@app.cell
def _():
    TRAIN_PATH    = 'data/processed_wide/train.csv'
    TEST_PATH = 'data/processed_wide/val_test.csv'
    OUTPUT_PATH   = 'submission_0304.csv'
    return TEST_PATH, TRAIN_PATH


@app.cell
def _():
    # Columns that must NOT be used as features
    TARGET_COLS = [
        'evenements_hepatiques_majeurs',
        'evenements_hepatiques_age_occur',
        'death',
        'death_age_occur',
    ]
    ID_COLS = ['patient_id_anon', 'trustii_id']
    return ID_COLS, TARGET_COLS


@app.cell
def _(TEST_PATH, TRAIN_PATH, pd):
    df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return (df,)


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df, skrub):
    from skrub import TableReport
    skrub.set_config(max_plot_columns=50)
    TableReport(df)
    return (TableReport,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Feature engineering


    Conservation des colonnes correspondant au dernier examen

    Evaluation par le TableReport

    Evaluation d'un one hot encoding sur les variables T2DM, Hypertension, dyslipidaemia, batriatric_surgery

    Supression des colonnes ayant un taux elevé de colonnes manquantes (>50%) et imputation par la médiane (ou Iterative Imputer)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.0 Création de la colonne ALT/ AST
    """)
    return


@app.cell
def _(np):

    def create_alt_ast_ratio(df, max_visit=21):
        df = df.copy()
    
        for v in range(1, max_visit + 1):
            alt_col = f'alt_v{v}'
            ast_col = f'ast_v{v}'
            new_col = f'alt_ast_v{v}'
        
            if alt_col in df.columns and ast_col in df.columns:
                df[new_col] = np.where(
                    df[alt_col].notna() & df[ast_col].notna(),
                    df[alt_col] / df[ast_col],
                    np.nan
                )
            else:
                # Si une des colonnes n'existe pas → colonne remplie de NA
                df[new_col] = np.nan
    
        # Drop toutes les colonnes ALT et AST
        cols_to_drop = [col for col in df.columns if col.startswith('alt_v') or col.startswith('ast_')]
        #df = df.drop(columns=cols_to_drop)
    
        return df

    return (create_alt_ast_ratio,)


@app.cell
def _(create_alt_ast_ratio, df):
    df1 = create_alt_ast_ratio(df, max_visit=21)
    return (df1,)


@app.cell
def _(TableReport, df1):
    TableReport(df1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Création des variables slopes, change_scores et rolling statisics
    """)
    return


@app.cell
def _():
    BIOMARKERS = [
            "BMI", "alt", "ast", "bilirubin", "chol", "ggt",
            "gluc_fast", "plt", "triglyc",
            "aixp_aix_result_BM_3",
            "fibrotest_BM_2",
            "fibs_stiffness_med_BM_1"]
    return (BIOMARKERS,)


@app.cell
def _(np):
    def create_change_scores(df, baseline_visit=1):
        df = df.copy()
    
        # Colonnes longitudinales
        long_cols = [col for col in df.columns if "_v" in col]
    
        # Préfixes (BMI, alt, etc.)
        prefixes = set(col.rsplit("_v", 1)[0] for col in long_cols)
    
        for prefix in prefixes:
            baseline_col = f"{prefix}_v{baseline_visit}"
        
            if baseline_col not in df.columns:
                continue
        
            visit_cols = [col for col in df.columns if col.startswith(prefix + "_v")]
        
            for col in visit_cols:
                visit_num = col.split("_v")[-1]
            
                if col == baseline_col:
                    continue
            
                new_col = f"change_{prefix}_v{visit_num}"
            
                df[new_col] = np.where(
                    df[col].notna() & df[baseline_col].notna(),
                    (df[col] - df[baseline_col]) / df[baseline_col] * 100, 
                    #(df[col] - df[baseline_col]) / df[baseline_col].std(),
                    #df[col] - df[baseline_col],
                    np.nan
                )
    
        return df

    return (create_change_scores,)


@app.cell
def _(BIOMARKERS, np, pd):


    def compute_patient_slopes(df, biomarkers = BIOMARKERS, max_visit=22):
        df = df.copy()
    
        for biom in biomarkers:
            values = []
            ages = []
        
            # Construire matrices (patients x visites)
            for v in range(1, max_visit+1):
                val_col = f"{biom}_v{v}"
                age_col = f"Age_v{v}"
            
                if val_col in df.columns and age_col in df.columns:
                    values.append(df[val_col])
                    ages.append(df[age_col])
        
            if len(values) < 2:
                df[f"slope_{biom}"] = np.nan
                continue
        
            values = pd.concat(values, axis=1)
            ages = pd.concat(ages, axis=1)
        
            slopes = []
        
            for i in range(len(df)):
                x = ages.iloc[i].values
                y = values.iloc[i].values
            
                mask = ~np.isnan(x) & ~np.isnan(y)
            
                if mask.sum() < 2:
                    slopes.append(np.nan)
                else:
                    slope = np.polyfit(x[mask], y[mask], 1)[0]
                    slopes.append(slope)
        
            df[f"slope_{biom}"] = slopes
    
        return df

    def compute_patient_slopes_theilsen(df, biomarkers = BIOMARKERS, max_visit=22):
        df = df.copy()
    
        def theil_sen_slope(x, y):
            slopes = []
            n = len(x)
        
            for i in range(n):
                for j in range(i+1, n):
                    if x[j] != x[i]:
                        slopes.append((y[j] - y[i]) / (x[j] - x[i]))
        
            if len(slopes) == 0:
                return np.nan
        
            return np.median(slopes)
    
        for biom in biomarkers:
            values = []
            ages = []
        
            for v in range(1, max_visit+1):
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
                x = ages.iloc[i].values
                y = values.iloc[i].values
            
                mask = ~np.isnan(x) & ~np.isnan(y)
                x = x[mask]
                y = y[mask]
            
                if len(x) < 2:
                    slopes.append(np.nan)
                else:
                    slopes.append(theil_sen_slope(x, y))
        
            df[f"slope_ts_{biom}"] = slopes
    
        return df

    return (compute_patient_slopes_theilsen,)


@app.function
def count_non_na_longitudinal(df):
    df = df.copy()
    
    # Identifier toutes les variables longitudinales (suffixe _vX)
    long_cols = [col for col in df.columns if "_v" in col]
    
    # Extraire les "préfixes" (ex: BMI, alt, ast, etc.)
    prefixes = set(col.rsplit("_v", 1)[0] for col in long_cols)
    
    for prefix in prefixes:
        cols = [col for col in df.columns if col.startswith(prefix + "_v")]
        
        # Compter les valeurs non NA par ligne
        df[f"n_{prefix}"] = df[cols].notna().sum(axis=1)
    
    return df


@app.cell
def _(BIOMARKERS, np):

    def add_first_last_visits(df, biomarkers = BIOMARKERS, max_visit=22):

        df = df.copy()
    
        for biom in biomarkers:
            biom_cols = [f"{biom}_v{i}" for i in range(1, max_visit+1) if f"{biom}_v{i}" in df.columns]
        
            if not biom_cols:
                df[f"first_{biom}"] = np.nan
                df[f"last_{biom}"] = np.nan
                continue
        
            # DataFrame temporaire
            data = df[biom_cols]
        
            # Booléen : valeur présente ou non
            mask = data.notna()
        
            # Numéros de visite
            visit_numbers = np.array([int(col.split("_v")[1]) for col in biom_cols])
        
            # Première visite
            first_visit = mask.apply(
                lambda row: visit_numbers[row.values].min() if row.any() else np.nan,
                axis=1
            )
        
            # Dernière visite
            last_visit = mask.apply(
                lambda row: visit_numbers[row.values].max() if row.any() else np.nan,
                axis=1
            )
        
            df[f"first_{biom}"] = first_visit
            df[f"last_{biom}"] = last_visit
    
        return df

    return


@app.cell
def _(np):

    def add_visit_metrics(df, max_visit=22):
        df = df.copy()
    
        # Toutes les colonnes longitudinales
        long_cols = [col for col in df.columns if "_v" in col]
    
        for v in range(1, max_visit + 1):
            # Colonnes de la visite v
            visit_cols = [col for col in long_cols if col.endswith(f"_v{v}")]
        
            if visit_cols:
                # Nombre de mesures non NA à cette visite
                df[f"n_measures_v{v}"] = df[visit_cols].notna().sum(axis=1)
            else:
                df[f"n_measures_v{v}"] = np.nan
        
            # Différence d'âge
            age_col = f"Age_v{v}"
            prev_age_col = f"Age_v{v-1}"
        
            if v == 1:
                df[f"age_diff_v{v}"] = np.nan
            else:
                if age_col in df.columns and prev_age_col in df.columns:
                    df[f"age_diff_v{v}"] = np.where(
                        df[age_col].notna() & df[prev_age_col].notna(),
                        df[age_col] - df[prev_age_col],
                        np.nan
                    )
                else:
                    df[f"age_diff_v{v}"] = np.nan
    
        return df

    return (add_visit_metrics,)


@app.cell
def _(BIOMARKERS, np, pd):

    def add_patient_summary_metrics(df, biomarkers = BIOMARKERS, max_visit=22):
        df = df.copy()
    
        # =========================
        # 1. AGE (global)
        # =========================
        age_cols = [f"Age_v{i}" for i in range(1, max_visit+1) if f"Age_v{i}" in df.columns]
    
        age_diffs = []
        for i in range(1, len(age_cols)):
            age_diffs.append(df[age_cols[i]] - df[age_cols[i-1]])
    
        if age_diffs:
            df["mean_age_diff"] = pd.concat(age_diffs, axis=1).mean(axis=1, skipna=True)
        else:
            df["mean_age_diff"] = np.nan
    
        df["min_age"] = df[age_cols].min(axis=1, skipna=True)
        df["max_age"] = df[age_cols].max(axis=1, skipna=True)
        df["age_range"] = df["max_age"] - df["min_age"]
    
        # =========================
        # 2. COMPLETUDE (global)
        # =========================
        long_cols = [col for col in df.columns if "_v" in col]
    
        n_measures = []
        for v in range(1, max_visit+1):
            visit_cols = [col for col in long_cols if col.endswith(f"_v{v}")]
            if visit_cols:
                n_measures.append(df[visit_cols].notna().sum(axis=1))
    
        if n_measures:
            df["mean_n_measures"] = pd.concat(n_measures, axis=1).mean(axis=1, skipna=True)
        else:
            df["mean_n_measures"] = np.nan
    
        # =========================
        # 3. BIOMARKERS (nouveau)
        # =========================
        for biom in biomarkers:
            biom_cols = [f"{biom}_v{i}" for i in range(1, max_visit+1) if f"{biom}_v{i}" in df.columns]
        
            if not biom_cols:
                continue
        
            # --- min / max / range ---
            df[f"min_{biom}"] = df[biom_cols].min(axis=1, skipna=True)
            df[f"max_{biom}"] = df[biom_cols].max(axis=1, skipna=True)
            df[f"{biom}_range"] = df[f"max_{biom}"] - df[f"min_{biom}"]
        
            # --- différences entre visites ---
            diffs = []
            for i in range(1, len(biom_cols)):
                diffs.append(df[biom_cols[i]] - df[biom_cols[i-1]])
        
            if diffs:
                df[f"mean_{biom}_diff"] = pd.concat(diffs, axis=1).mean(axis=1, skipna=True)
            else:
                df[f"mean_{biom}_diff"] = np.nan
    
        return df

    return (add_patient_summary_metrics,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Conservation des colonnes correspondant au dernier examen
    """)
    return


@app.cell
def _():

    MAX_MISSING_RATE = 0.50
    return (MAX_MISSING_RATE,)


@app.cell
def _(np, re):

    def extract_last_available_values(df):
        df = df.copy()

        # -----------------------------
        # 1. Colonnes âge
        # -----------------------------
        age_cols = sorted(
            [c for c in df.columns if re.match(r"Age_v\d+", c)],
            key=lambda x: int(re.findall(r"\d+", x)[0])
        )

        age_matrix = df[age_cols].to_numpy()
        mask_age = ~np.isnan(age_matrix)

        # num_visits
        df["num_visits"] = mask_age.sum(axis=1)

        # last observed age (indépendant maintenant)
        rev_idx_age = np.argmax(mask_age[:, ::-1], axis=1)
        last_idx_age = age_matrix.shape[1] - 1 - rev_idx_age
        last_idx_age[df["num_visits"] == 0] = -1

        df["last_observed_age"] = np.where(
            last_idx_age >= 0,
            age_matrix[np.arange(len(df)), last_idx_age],
            np.nan
        )

        # -----------------------------
        # 2. Biomarqueurs
        # -----------------------------
        biomarkers = [
            "BMI", "alt", "ast", "bilirubin", "chol", "ggt",
            "gluc_fast", "plt", "triglyc",
            "aixp_aix_result_BM_3",
            "fibrotest_BM_2",
            "fibs_stiffness_med_BM_1"
        ]

        #visit_cols_to_drop = age_cols.copy()
        visit_cols_to_drop = []


        N_LAST = 10  # 👈 nombre de valeurs à garder

        for biom in biomarkers:
            biom_cols = sorted(
                [c for c in df.columns if re.match(fr"{biom}_v\d+", c)],
                key=lambda x: int(re.findall(r"\d+", x)[0])
            )

            if not biom_cols:
                continue

            biom_matrix = df[biom_cols].to_numpy()

            # inversion pour partir de la fin
            reversed_matrix = np.flip(biom_matrix, axis=1)

            last_values = [[] for _ in range(N_LAST)]

            for i in range(reversed_matrix.shape[0]):
                row = reversed_matrix[i]
                vals = row[~np.isnan(row)][:N_LAST]  # 👈 prend les N dernières non-NaN

                # padding si moins de N valeurs
                vals = list(vals) + [np.nan] * (N_LAST - len(vals))

                for j in range(N_LAST):
                    last_values[j].append(vals[j])

            # création des colonnes
            for j in range(N_LAST):
                if j == 0:
                    col_name = f"{biom}_last"
                else:
                    col_name = f"{biom}_last_minus{j}"

                df[col_name] = last_values[j]

            visit_cols_to_drop.extend(biom_cols)


        # -----------------------------
        # 3. Drop colonnes longitudinales
        # -----------------------------
        df.drop(columns=visit_cols_to_drop, inplace=True, errors="ignore")

        # -----------------------------
        # 4. Colonnes événements à la fin
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

    return (extract_last_available_values,)


@app.cell
def _(df):
    for column in df.columns:
        print(column)
    return


@app.cell
def _(
    add_patient_summary_metrics,
    add_visit_metrics,
    compute_patient_slopes_theilsen,
    create_change_scores,
    df,
    extract_last_available_values,
    pd,
):


    train_df = pd.get_dummies(extract_last_available_values(add_visit_metrics(add_patient_summary_metrics(compute_patient_slopes_theilsen(create_change_scores(count_non_na_longitudinal(df)))))), columns=['gender', "T2DM",'Hypertension', 'Dyslipidaemia','bariatric_surgery'], drop_first=True)

    return (train_df,)


@app.cell
def _(train_df):
    train_df.columns
    return


@app.cell
def _(TableReport, train_df):
    TableReport(train_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Target Processing

    `prepare_survival_targets()` converts the 4 raw target columns in `train.csv` into a `sksurv`-compatible structured array.

    ### Survival time formula
    - **Event patients** : `age_at_event − Age_v1` (time from first visit to event)
    - **Censored patients** : `last_observed_age − Age_v1` (time from first visit to last follow-up)

    ### Filtering
    - **Hepatic** : to explain
    - **Death** : to explain
    """)
    return


@app.cell
def _(Surv, np, pd):


    def prepare_survival_targets_custom(
        df: pd.DataFrame,
        outcome: str = "hepatic",
        baseline_col: str = "Age_v1",
        last_observed_col: str = "last_observed_age",
        min_time: float = 1e-3,
        drop_nonpositive_times: bool = True,
        return_report: bool = False,
    ):
        df = df.copy()

        if outcome == "hepatic":
            event_col = "evenements_hepatiques_majeurs"
            time_col = "evenements_hepatiques_age_occur"
            event_name = "Hepatic_event"

            # Event must be known
            df = df[df[event_col].notna()].copy()
            is_event = df[event_col].astype(int) == 1

            # If event=1 → time must exist
            valid = (~is_event) | df[time_col].notna()
            df = df[valid].copy()

        elif outcome == "death":
            event_col = "death"
            time_col = "death_age_occur"
            event_name = "Death"

            # 🔥 NOUVELLE RÈGLE
            # event NaN → 0 (censuré)
            df[event_col] = df[event_col].fillna(0)

            is_event = df[event_col].astype(int) == 1

            # Si event=1 → temps obligatoire
            valid = (~is_event) | df[time_col].notna()
            df = df[valid].copy()

        else:
            raise ValueError(f"outcome must be 'hepatic' or 'death'")

        # Recompute after filtering
        is_event = df[event_col].astype(int) == 1

        # Compute time
        time_values = np.where(
            is_event,
            df[time_col] - df[baseline_col],
            df[last_observed_col] - df[baseline_col],
        ).astype(float)

        # Clean invalid times
        if drop_nonpositive_times:
            mask_valid = time_values > 0
            df = df.loc[mask_valid].copy()
            time_values = time_values[mask_valid]

        # Remove NaN / inf
        mask_finite = np.isfinite(time_values)
        df = df.loc[mask_finite].copy()
        time_values = time_values[mask_finite]

        # Clamp small values
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
                "n_final": len(df),
                "event_rate": is_event.mean(),
                "outcome": outcome,
            }
            return df.reset_index(drop=True), y, report

        return df.reset_index(drop=True), y

    return (prepare_survival_targets_custom,)


@app.cell
def _(prepare_survival_targets_custom, train_df):

    # --- Sanity check ---
    df_hep,   y_hep   = prepare_survival_targets_custom(train_df, outcome='hepatic')
    df_death, y_death = prepare_survival_targets_custom(train_df, outcome='death')


    return df_death, df_hep, y_death, y_hep


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Feature Matrix
    """)
    return


@app.cell
def _(ID_COLS, MAX_MISSING_RATE, TARGET_COLS, df_death, df_hep):
    def build_feature_matrix(df, keep_cols=None):
        """
        Build numeric feature matrix, excluding:
          - Target columns (would cause leakage)
          - ID columns
          - Age_v* columns (used only for censoring time derivation)
          - 'last_observed_age' (derived from Age_v* — leakage risk)
        If keep_cols is provided, only those columns are returned (for
        consistent alignment across train / val / test).
        """
        age_v_cols   = [c for c in df.columns if c.startswith('Age_v')]
        leakage_cols = ['last_observed_age']
        drop_cols    = TARGET_COLS + ID_COLS + age_v_cols + leakage_cols
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        X = X.select_dtypes(include='number')
        if keep_cols is not None:
            X = X[[c for c in keep_cols if c in X.columns]]
        return X


    # ── Build raw matrices ────────────────────────────────────────────────────
    X_hep_raw   = build_feature_matrix(df_hep)
    X_death_raw = build_feature_matrix(df_death)
    #X_pred_raw  = build_feature_matrix(test_df)

    print(f'Raw feature count: {X_hep_raw.shape[1]}')

    # ── Missing-rate filter (fitted on train, applied everywhere) ─────────────
    # Features with > MAX_MISSING_RATE missing in train collapse to a constant
    # after median imputation → the model memorises the imputed pattern instead
    # of a real signal (a form of overfitting).  Removing them first dramatically
    # reduces feature/event ratio (261 → ~19) which is the primary cause of the
    # train/val gap.
    missing_rate_hep   = X_hep_raw.isna().mean()
    missing_rate_death = X_death_raw.isna().mean()

    keep_hep   = missing_rate_hep[missing_rate_hep   <= MAX_MISSING_RATE].index.tolist()
    keep_death = missing_rate_death[missing_rate_death <= MAX_MISSING_RATE].index.tolist()

    # Align to val_test (prediction set must have the same columns)
    #keep_hep   = [c for c in keep_hep   if c in X_pred_raw.columns]
    #keep_death = [c for c in keep_death if c in X_pred_raw.columns]

    # Final aligned matrices
    X_hep_aln   = X_hep_raw[keep_hep]
    X_death_aln = X_death_raw[keep_death]


    print(f'Hepatic features after missing filter : {len(keep_hep)}  '
          f'(EPV = {int((df_hep["evenements_hepatiques_majeurs"]==1).sum())}/{len(keep_hep)} = '
          f'{(df_hep["evenements_hepatiques_majeurs"]==1).sum()/len(keep_hep):.2f})')
    print(f'Death   features after missing filter : {len(keep_death)}  '
          f'(EPV = {int((df_death["death"]==1).sum())}/{len(keep_death)} = '
          f'{(df_death["death"]==1).sum()/len(keep_death):.2f})')
    return X_death_aln, X_hep_aln


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Train Models & Evaluate

    Two model types per outcome:
    - **Elastic Net Cox** (`CoxnetSurvivalAnalysis`, l1_ratio=0.9) — LASSO-style sparsity, alpha tuned by 5-fold CV
    - **Random Survival Forest** — `min_samples_leaf=20` to reduce overfitting on rare events
    """)
    return


@app.cell
def _(
    IterativeImputer,
    KFold,
    Pipeline,
    RandomSurvivalForest,
    concordance_index_censored,
    np,
):
    def make_rsf_pipeline():
        """RSF with min_samples_leaf=20 to reduce overfitting on rare events."""
        return Pipeline([
            #('imp', SimpleImputer(strategy='median')),
            #('imp', IterativeImputer(ExtraTreesRegressor(n_estimators=50, random_state=0))),
            ('imp', IterativeImputer()),
            ('rsf', RandomSurvivalForest(
                n_estimators=100,
                min_samples_leaf=20,   # was 10 — larger leaves → less variance
                min_samples_split=40,  # require more samples before splitting
                max_features='sqrt',
                n_jobs=1,
                random_state=26,
            )),
        ])



    def cv_rsf(X, y, n_splits=5, random_state=26):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        scores = []

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = make_rsf_pipeline()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            ci = concordance_index_censored(
                y_test[y.dtype.names[0]],   # event
                y_test[y.dtype.names[1]],   # time
                preds
            )[0]

            scores.append(ci)

        return np.mean(scores), np.std(scores)

    return (cv_rsf,)


@app.cell
def _(X_death_aln, cv_rsf, y_death):
    print('Fitting Death — RSF (CV)...')
    ci_rsf_death_mean, ci_rsf_death_std = cv_rsf(X_death_aln, y_death)

    print(f'  RSF  C-index (CV): {ci_rsf_death_mean:.4f} ± {ci_rsf_death_std:.4f}')
    return


@app.cell
def _(X_hep_aln, cv_rsf, y_hep):
    print('Fitting Hepatic — RSF (CV)...')
    ci_rsf_hep_mean, ci_rsf_hep_std = cv_rsf(X_hep_aln, y_hep)

    #print(f'  RSF  C-index (CV): {ci_rsf_hep_mean:.4f} ± {ci_rsf_hep_std:.4f}')
    return ci_rsf_hep_mean, ci_rsf_hep_std


@app.cell
def _(ci_rsf_hep_mean, ci_rsf_hep_std):
    print(f'  RSF  C-index (CV): {ci_rsf_hep_mean:.4f} ± {ci_rsf_hep_std:.4f}')
    return


@app.cell
def _():
    # Les 10 dernières colonnes conservés,  nombre de mesurements par biomarqueur, Iterative Imputer et prepare_survival_targets_custom et change_score

    # 
    #La meilleure option est : 0.9191 et 0.8945 de death
    #La death monte à 0.9446 si on rajojute la fonction add_patient_summary_metrics (pas extended) et add_visit_metrics
    # Par contre hepatic est à 0.9299 avec l'extended et enlever le truc de slopes absolument pour garder cette performance

    # A tester l'impact du slope (avec theilsen directement) sur le risque hepatic
    # La mort augmente un peu avec ALT/AST mais avec une variance plus élévée
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Alignement des colonnes du test_df
    """)
    return


@app.cell
def _():
    """

    X_pred_raw  = build_feature_matrix(test_df_tr)



    print(f'Raw feature count: {X_pred_raw.shape[1]}')


    # Final aligned matrices


    X_pred_hep  = X_pred_raw[keep_hep]
    X_pred_death= X_pred_raw[keep_death]
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Predictions on val_test.csv
    """)
    return


@app.cell
def _():
    # Use RSF predictions (more robust on tabular survival data with rare events)
    """

    pred_hep   = rsf_hep.predict(X_pred_hep)
    pred_death = rsf_death.predict(X_pred_death)

    submission = pd.DataFrame({
        'trustii_id':         test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hep,
        'risk_death':         pred_death,
    })

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved {len(submission)} predictions → {OUTPUT_PATH}')
    print(submission.head())
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unused functions
    """)
    return


@app.cell
def _():
    """
    import pandas as pd
    import numpy as np

    import pandas as pd
    import numpy as np

    def add_first_last_visits(df, biomarkers, max_visit=22):
        df = df.copy()
    
        for biom in biomarkers:
            biom_cols = [f"{biom}_v{i}" for i in range(1, max_visit+1) if f"{biom}_v{i}" in df.columns]
        
            if not biom_cols:
                df[f"first_{biom}"] = 0
                df[f"last_{biom}"] = 0
                continue
        
            data = df[biom_cols]
            mask = data.notna()
        
            visit_numbers = np.array([int(col.split("_v")[1]) for col in biom_cols])
        
            def get_first(row):
                if not row.any():
                    return 0
                return visit_numbers[row.values].min()
        
            def get_last(row):
                if not row.any():
                    return 0
                return visit_numbers[row.values].max()
        
            df[f"first_{biom}"] = mask.apply(get_first, axis=1)
            df[f"last_{biom}"] = mask.apply(get_last, axis=1)
    
        return df
    """
    return


@app.cell
def _():
    """
    Une slope plus robuste à implémenter

    import numpy as np
    import pandas as pd

    def compute_patient_slopes_theilsen(df, biomarkers, max_visit=22):
        df = df.copy()
    
        def theil_sen_slope(x, y):
            slopes = []
            n = len(x)
        
            for i in range(n):
                for j in range(i+1, n):
                    if x[j] != x[i]:
                        slopes.append((y[j] - y[i]) / (x[j] - x[i]))
        
            if len(slopes) == 0:
                return np.nan
        
            return np.median(slopes)
    
        for biom in biomarkers:
            values = []
            ages = []
        
            for v in range(1, max_visit+1):
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
                x = ages.iloc[i].values
                y = values.iloc[i].values
            
                mask = ~np.isnan(x) & ~np.isnan(y)
                x = x[mask]
                y = y[mask]
            
                if len(x) < 2:
                    slopes.append(np.nan)
                else:
                    slopes.append(theil_sen_slope(x, y))
        
            df[f"slope_ts_{biom}"] = slopes
    
        return df
        """
    return


@app.cell
def _():
    """
    import pandas as pd
    import numpy as np

    def add_visit_metrics(df, max_visit=22):
        df = df.copy()
    
        # Toutes les colonnes longitudinales
        long_cols = [col for col in df.columns if "_v" in col]
    
        for v in range(1, max_visit + 1):
            # Colonnes de la visite v
            visit_cols = [col for col in long_cols if col.endswith(f"_v{v}")]
        
            if visit_cols:
                # Nombre de mesures non NA à cette visite
                df[f"n_measures_v{v}"] = df[visit_cols].notna().sum(axis=1)
            else:
                df[f"n_measures_v{v}"] = np.nan
        
            # Différence d'âge
            age_col = f"Age_v{v}"
            prev_age_col = f"Age_v{v-1}"
        
            if v == 1:
                df[f"age_diff_v{v}"] = np.nan
            else:
                if age_col in df.columns and prev_age_col in df.columns:
                    df[f"age_diff_v{v}"] = np.where(
                        df[age_col].notna() & df[prev_age_col].notna(),
                        df[age_col] - df[prev_age_col],
                        np.nan
                    )
                else:
                    df[f"age_diff_v{v}"] = np.nan
    
        return df
        """
    return


@app.cell
def _():
    # Pour les extract_last availables values

    # Pour la dernière 
    """
    for biom in biomarkers:
            biom_cols = sorted(
                [c for c in df.columns if re.match(fr"{biom}_v\d+", c)],
                key=lambda x: int(re.findall(r"\d+", x)[0])
            )

            if not biom_cols:
                continue

            biom_matrix = df[biom_cols].to_numpy()
            mask = ~np.isnan(biom_matrix)

            # dernière valeur non-NA (indépendante)
            rev_idx = np.argmax(mask[:, ::-1], axis=1)
            last_idx = biom_matrix.shape[1] - 1 - rev_idx

            # gérer lignes sans aucune valeur
            has_value = mask.any(axis=1)
            last_idx[~has_value] = -1

            df[f"{biom}_last"] = np.where(
                last_idx >= 0,
                biom_matrix[np.arange(len(df)), last_idx],
                np.nan
            )

            visit_cols_to_drop.extend(biom_cols)

    """

    """
    # Pour les 3 dernières 

    for biom in biomarkers:
            biom_cols = sorted(
                [c for c in df.columns if re.match(fr"{biom}_v\d+", c)],
                key=lambda x: int(re.findall(r"\d+", x)[0])
            )

            if not biom_cols:
                continue

            biom_matrix = df[biom_cols].to_numpy()
            reversed_matrix = np.flip(biom_matrix, axis=1)

            last1, last2, last3 = [], [], []

            for i in range(reversed_matrix.shape[0]):
                row = reversed_matrix[i]
                vals = row[~np.isnan(row)][:3]

                vals = list(vals) + [np.nan] * (3 - len(vals))  # padding

                last1.append(vals[0])
                last2.append(vals[1])
                last3.append(vals[2])

            df[f"{biom}_last"] = last1
            df[f"{biom}_last_minus1"] = last2
            df[f"{biom}_last_minus2"] = last3

            visit_cols_to_drop.extend(biom_cols)

    """

    # Pour les 10 derniers :
    """

    N_LAST = 10  # 👈 nombre de valeurs à garder

    for biom in biomarkers:
        biom_cols = sorted(
            [c for c in df.columns if re.match(fr"{biom}_v\d+", c)],
            key=lambda x: int(re.findall(r"\d+", x)[0])
        )

        if not biom_cols:
            continue

        biom_matrix = df[biom_cols].to_numpy()

        # inversion pour partir de la fin
        reversed_matrix = np.flip(biom_matrix, axis=1)

        last_values = [[] for _ in range(N_LAST)]

        for i in range(reversed_matrix.shape[0]):
            row = reversed_matrix[i]
            vals = row[~np.isnan(row)][:N_LAST]  # 👈 prend les N dernières non-NaN

            # padding si moins de N valeurs
            vals = list(vals) + [np.nan] * (N_LAST - len(vals))

            for j in range(N_LAST):
                last_values[j].append(vals[j])

        # création des colonnes
        for j in range(N_LAST):
            if j == 0:
                col_name = f"{biom}_last"
            else:
                col_name = f"{biom}_last_minus{j}"

            df[col_name] = last_values[j]

        visit_cols_to_drop.extend(biom_cols)

    """
    return


@app.cell
def _():
    """

    def extract_last_visit_vectorized(df):
        df = df.copy()

        # -----------------------------
        # 1. Colonnes âge triées
        # -----------------------------
        age_cols = sorted(
            [c for c in df.columns if re.match(r"Age_v\d+", c)],
            key=lambda x: int(re.findall(r"\d+", x)[0]))
        age_matrix = df[age_cols].to_numpy()

        # -----------------------------
        # 2. Mask + num_visits
        # -----------------------------
        mask = ~np.isnan(age_matrix)
        df["num_visits"] = mask.sum(axis=1)

        # -----------------------------
        # 3. Dernière visite (index)
        # -----------------------------
        # reverse + argmax
        rev_idx = np.argmax(mask[:, ::-1], axis=1)
        last_idx = age_matrix.shape[1] - 1 - rev_idx
        # si aucune visite → mettre NaN
        last_idx[df["num_visits"] == 0] = -1

        # -----------------------------
        # 4. last_observed_age
        # -----------------------------
        df["last_observed_age"] = np.where(
            last_idx >= 0,
            age_matrix[np.arange(len(df)), last_idx],
            np.nan)

        # -----------------------------
        # 5. Biomarqueurs
        # -----------------------------
        biomarkers = [
            "BMI", "alt", "ast", "bilirubin", "chol", "ggt",
            "gluc_fast", "plt", "triglyc",
            "aixp_aix_result_BM_3",
            "fibrotest_BM_2",
            "fibs_stiffness_med_BM_1"]

        visit_cols_to_drop = age_cols.copy()
        #visit_cols_to_drop = []

        for biom in biomarkers:
            biom_cols = sorted(
                [c for c in df.columns if re.match(fr"{biom}_v\d+", c)],
                key=lambda x: int(re.findall(r"\d+", x)[0]))

            if not biom_cols:
                continue

            biom_matrix = df[biom_cols].to_numpy()

            df[f"{biom}_last"] = np.where(
                last_idx >= 0,
                biom_matrix[np.arange(len(df)), last_idx],
                np.nan
            )
            visit_cols_to_drop.extend(biom_cols)

        # -----------------------------
        # 6. Drop toutes les colonnes _v*
        # -----------------------------

        df.drop(columns=visit_cols_to_drop, inplace=True, errors="ignore")

        # -----------------------------
        # 7. Mettre les colonnes events à la fin
        # -----------------------------
        event_cols = [
            "evenements_hepatiques_majeurs",
            "evenements_hepatiques_age_occur",
            "death",
            "death_age_occur"
        ]

        # garder seulement celles qui existent
        event_cols = [c for c in event_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in event_cols]
        df = df[other_cols + event_cols]

        return df


    """
    return


@app.cell
def _(Surv, np, pd):


    def prepare_survival_targets_robust(
        df: pd.DataFrame,
        outcome: str = "hepatic",
        baseline_col: str = "Age_v1",
        last_observed_col: str = "last_observed_age",
        min_time: float = 1e-3,
        drop_nonpositive_times: bool = True,
        return_report: bool = False,
    ):
        """
        Build a sksurv-compatible structured array for one survival endpoint.

        Rules:
          - event missing -> drop row
          - event == 1 and event time missing -> drop row  (Peut être prendre last_observed_age - Age_v1 ou predire selon les donnes)
          - event == 0 -> censored at last_observed_age - Age_v1
          - negative times -> drop row if drop_nonpositive_times=True
          - zero times -> clamped to min_time

        Parameters
        ----------
        df : pd.DataFrame
            Must contain baseline_col, last_observed_col, and the target columns.
        outcome : str
            "hepatic" or "death".
        return_report : bool
            If True, also returns a dict with filtering counts.

        Returns
        -------
        df_valid : pd.DataFrame
        y : structured np.ndarray
        report : dict (optional)
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

        # Keep only rows where the event indicator is known
        mask_event_known = df[event_col].notna()
        df = df.loc[mask_event_known].copy()
        n1 = len(df)

        # Convert to binary event indicator
        is_event = df[event_col].astype(int) == 1

        # For event rows, the event time must be known
        mask_event_time_known = (~is_event) | df[time_col].notna()
        df = df.loc[mask_event_time_known].copy()
        n2 = len(df)

        # Recompute event indicator after filtering
        is_event = df[event_col].astype(int) == 1

        # Survival time
        time_values = np.where(
            is_event,
            df[time_col] - df[baseline_col],
            df[last_observed_col] - df[baseline_col],
        ).astype(float)

        # Remove obviously invalid times
        if drop_nonpositive_times:
            mask_valid_time = time_values > 0
            df = df.loc[mask_valid_time].copy()
            time_values = time_values[mask_valid_time]
        else:
            # Keep them but clamp after
            pass

        # If any still negative due to numerical issues, drop them
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
                "n_after_event_known": n1,
                "n_after_event_time_known": n2,
                "n_final": len(df),
                "dropped_total": n0 - len(df),
                "outcome": outcome,
            }
            return df.reset_index(drop=True), y, report

        return df.reset_index(drop=True), y

    return


@app.cell
def _(Surv, np, pd):
    def prepare_survival_targets_custom_v2(
        df: pd.DataFrame,
        outcome: str = "hepatic",
        baseline_col: str = "Age_v1",
        last_observed_col: str = "last_observed_age",
        min_time: float = 1e-3,
        drop_nonpositive_times: bool = True,
        return_report: bool = False,
    ):
        """
        Prépare les cibles de survie ANNITIA
        Args:
            df: DataFrame d'entrée, une ligne/patient
            outcome: 'hepatic' ou 'death'
            baseline_col: nom de la colonne d'âge de baseline
            last_observed_col: nom de la colonne d'âge dernière obs.
            min_time: temps minimal admissible (>0)
            drop_nonpositive_times: dropper les temps <= 0
            return_report: si True, renvoie aussi un rapport de stats
        Returns:
            df_surv: DataFrame prêt pour modeling
            y: objet Surv (scikit-survival)
            (optionnel: rapport dict)
        """
        df = df.copy()

        # Map outcome
        outcome_map = {
            "hepatic": {
                "event_col": "evenements_hepatiques_majeurs",
                "time_col": "evenements_hepatiques_age_occur",
                "event_name": "Hepatic_event",
            },
            "death": {
                "event_col": "death",
                "time_col": "death_age_occur",
                "event_name": "Death"
            }
        }
        if outcome not in outcome_map:
            raise ValueError(f"outcome must be in {list(outcome_map)}")

        event_col = outcome_map[outcome]['event_col']
        time_col = outcome_map[outcome]['time_col']
        event_name = outcome_map[outcome]['event_name']

        # Remplace les event NaN -> 0 (censuré)
        df[event_col] = df[event_col].fillna(0)

        # Sécurité type (int)
        df[event_col] = df[event_col].astype(int)

        is_event = df[event_col] == 1

        # Si event=1, time_col obligatoire
        valid = (~is_event) | df[time_col].notna()
        exclu_idx = df.index[~valid].tolist()
        df = df[valid].copy()

        # Calcul du temps: event ou censure
        time_values = np.where(
            is_event,
            df[time_col] - df[baseline_col],
            df[last_observed_col] - df[baseline_col],
        ).astype(float)

        # Suppression temps non-positifs/illégaux
        mask_valid = time_values > 0 if drop_nonpositive_times else np.ones_like(time_values, dtype=bool)
        mask_finite = np.isfinite(time_values)
        mask = mask_valid & mask_finite

        exclu_time_idx = df.index[~mask].tolist()
        df = df.loc[mask].copy()
        time_values = time_values[mask]
        is_event = is_event.loc[mask].to_numpy()

        # Clamp min value
        time_values = np.maximum(time_values, min_time)

        y = Surv.from_arrays(
            event=is_event.astype(bool),
            time=time_values,
            name_event=event_name,
            name_time="Time_years",
        )

        if return_report:
            report = {
                "n_final": len(df),
                "event_rate": float(is_event.mean()),
                "time_min": float(np.min(time_values)),
                "time_max": float(np.max(time_values)),
                "excluded_missing_time": exclu_idx,
                "excluded_bad_time": exclu_time_idx,
                "outcome": outcome,
            }
            return df.reset_index(drop=True), y, report

        return df.reset_index(drop=True), y

    return


@app.cell
def _(IterativeImputer, np, pd):

    from sklearn.ensemble import RandomForestRegressor



    def _impute_missing_death_age_with_rf(
        df: pd.DataFrame,
        event_col: str = "death",
        age_col: str = "death_age_occur",
        baseline_col: str = "Age_v1",
        last_observed_col: str = "last_observed_age",
        exclude_cols=None,
        random_state: int = 42,
        n_estimators: int = 300,
        max_iter: int = 10,
        min_complete_events: int = 10,
    ):
        """
        Impute missing death_age_occur only among patients with death == 1,
        using IterativeImputer + RandomForestRegressor.
        """
        df = df.copy()

        if exclude_cols is None:
            exclude_cols = []

        # death must be known to use this row
        mask_event_known = df[event_col].notna()
        death_mask = pd.Series(False, index=df.index)
        death_mask.loc[mask_event_known] = df.loc[mask_event_known, event_col].astype(int) == 1

        # Need enough observed event times to learn a usable imputation model
        known_event_time = death_mask & df[age_col].notna()
        missing_event_time = death_mask & df[age_col].isna()

        if missing_event_time.sum() == 0:
            return df

        if known_event_time.sum() < min_complete_events:
            # Not enough signal to impute reliably
            return df

        # Use numeric predictors only, excluding targets / IDs / leakage columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        predictor_cols = [
            c for c in numeric_cols
            if c not in {event_col, age_col}
            and c not in set(exclude_cols)
        ]

        # Make sure baseline and last observed age are included if numeric
        for c in [baseline_col, last_observed_col]:
            if c in df.columns and c not in predictor_cols and pd.api.types.is_numeric_dtype(df[c]):
                predictor_cols.append(c)

        # Impute only on the subset death == 1
        cols_for_impute = predictor_cols + [age_col]
        sub = df.loc[death_mask, cols_for_impute].copy()

        # If the target has no missing values in the subset, nothing to do
        if sub[age_col].isna().sum() == 0:
            return df

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )

        imputer = IterativeImputer()
            #estimator=rf,
            #max_iter=max_iter,
            #random_state=random_state,
            #initial_strategy="median",
            #skip_complete=True,
            #sample_posterior=False,
        #)

        arr = imputer.fit_transform(sub)
        sub_imp = pd.DataFrame(arr, columns=cols_for_impute, index=sub.index)

        # Clip to plausible range: after baseline, and not after last observed age
        if baseline_col in sub_imp.columns:
            sub_imp[age_col] = np.maximum(sub_imp[age_col], sub_imp[baseline_col] + 1e-3)

        if last_observed_col in sub_imp.columns:
            sub_imp[age_col] = np.minimum(sub_imp[age_col], sub_imp[last_observed_col])

        # Write back only missing death ages for death == 1
        to_fill = death_mask & df[age_col].isna()
        df.loc[to_fill, age_col] = sub_imp.loc[to_fill, age_col].values

        return df

    

    return


@app.cell
def _(Surv, np, pd):

    def prepare_survival_targets_death_imputed(
        df: pd.DataFrame,
        outcome: str = "death",
        baseline_col: str = "Age_v1",
        last_observed_col: str = "last_observed_age",
        min_time: float = 1e-3,
    ):
        """
        Survival target preparation with:
          - hepatic: same rules as before
          - death: missing age_occur is imputed for death == 1 using RF IterativeImputer
        """
        df = df.copy()

        if outcome == "death":
            event_col = "death"
            age_col = "death_age_occur"
            event_name = "Death"

            # Conservative rule: drop unknown event labels and impossible event times
            df = df[df[event_col].notna()].copy()
            is_event = df[event_col].astype(int) == 1

            valid = (~is_event) | df[age_col].notna()
            df = df[valid].copy()
            is_event = df[event_col].astype(int) == 1

        elif outcome == "hepatic":
            event_col = "evenements_hepatiques_majeurs"
            age_col = "evenements_hepatiques_age_occur"
            event_name = "Hepatic_event"


            # Drop unknown death labels first
            df = df[df[event_col].notna()].copy()

            # Impute missing death_age_occur only for death == 1
            df = _impute_missing_death_age_with_rf(
                df,
                event_col=event_col,
                age_col=age_col,
                baseline_col=baseline_col,
                last_observed_col=last_observed_col,
                exclude_cols=["patient_id_anon"],
                random_state=42,
                n_estimators=100,
                max_iter=50,
                min_complete_events=50,
            )

            is_event = df[event_col].astype(int) == 1

            # After imputation, if some event rows are still missing age, drop them
            valid = (~is_event) | df[age_col].notna()
            df = df[valid].copy()
            is_event = df[event_col].astype(int) == 1

        else:
            raise ValueError(f"outcome must be 'hepatic' or 'death', got {outcome!r}")

        # Survival time
        time_values = np.where(
            is_event,
            df[age_col] - df[baseline_col],
            df[last_observed_col] - df[baseline_col],
        ).astype(float)

        # Remove invalid or non-positive times
        valid_time = np.isfinite(time_values) & (time_values > 0)
        df = df.loc[valid_time].copy()
        time_values = time_values[valid_time]
        is_event = (df[event_col].astype(int) == 1).to_numpy()

        time_values = np.maximum(time_values, min_time)

        y = Surv.from_arrays(
            event=is_event.astype(bool),
            time=time_values,
            name_event=event_name,
            name_time="Time_years",
        )

        return df.reset_index(drop=True), y

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluer l'importance du fibrotest
    """)
    return


@app.cell
def _():

    """
    def extract_high_fibrotest(df, threshold=0.6):
        df = df.copy()
    
        # Colonnes fibrotest
        fibro_cols = [col for col in df.columns if col.startswith("fibrotest_BM_2_v")]
    
    
        # Condition : au moins une valeur > threshold
        mask = (df[fibro_cols] > threshold).any(axis=1)
    
        # Sous-table
        high_fibrotest_df = df[mask]
    
        return high_fibrotest_df

    def extract_high_stiffness(df, threshold=0.6):
        df = df.copy()
    
        # Colonnes fibrotest
        fibro_cols = [col for col in df.columns if col.startswith("fibs_stiffness_med_BM_1_v")]
    
    
        # Condition : au moins une valeur > threshold
        mask = (df[fibro_cols] > threshold).any(axis=1)
    
        # Sous-table
        high_stiffness_df = df[mask]
    
        return high_stiffness_df


    high_fibrotest_df = extract_high_fibrotest(df, threshold=0.6)
    high_stiffness_df = extract_high_stiffness(df, threshold=14)


    TableReport(high_stiffness_df)


    import matplotlib.pyplot as plt

    def plot_death_vs_last_age(high_fibrotest_df):
        df = high_fibrotest_df.copy()
    
        # 1. Colonnes d'âge
        age_cols = [col for col in df.columns if col.startswith("Age_v")]
    
        # Trier correctement (v1, v2, ..., v22)
        age_cols = sorted(age_cols, key=lambda x: int(x.split("v")[1]))
    
        # 2. Dernier âge non NA par patient
        df["last_age"] = df[age_cols].apply(
            lambda row: row.dropna().iloc[-1] if row.notna().any() else np.nan,
            axis=1
        )
    
        # 3. Filtrer patients avec death_age_occur non manquant
        df = df[df["death_age_occur"].notna()]
    
        # 4. Calcul différence
        df["death_minus_last_age"] = df["death_age_occur"] - df["last_age"]
    
        # 5. Histogramme
        plt.figure(figsize=(8,5))
        plt.hist(df["death_minus_last_age"].dropna(), bins=30)
        plt.axvline(df["death_minus_last_age"].median(), linestyle="--")
        plt.xlabel("Death age - Last observed age")
        plt.ylabel("Number of patients")
        plt.title("Distribution of time between last visit and death")
        plt.show()
    
        return df

    result_df_stiffness = plot_death_vs_last_age(high_stiffness_df)
    result_df_fibroscan = plot_death_vs_last_age(high_fibrotest_df)

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
