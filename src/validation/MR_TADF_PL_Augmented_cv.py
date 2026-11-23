"""Evaluate the augmented PL-peak regressor by five-fold cross-validation."""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Full-dataset CV variant: all eligible molecules enter five-fold CV.
# Model settings are inherited unchanged; performance is evaluated by OOF predictions.

# Missing-value policy: targets are never imputed; predictors retain NaN.
# Set True to compare a small, prespecified set of missingness indicators.
# Indicators use only each row's availability, never targets or fold statistics.
ADD_MISSING_INDICATORS = False
MISSING_INDICATOR_FEATURES = (
    "S1",
    "T1",
    "dEST",
    "∆EST",
    "ΔEST",
    "f",
    "taup",
    "τp",
    "HOMO",
    "LUMO",
)


def prepare_missing_features(frame):
    result = frame.replace([np.inf, -np.inf], np.nan).copy()
    if ADD_MISSING_INDICATORS:
        for name in MISSING_INDICATOR_FEATURES:
            if name in result.columns:
                result[name + "__missing"] = result[name].isna().astype(int)
    return result


def training_median_frame(frame):
    """Fit on this training fold only; discard columns with no observed training value."""

    from sklearn.impute import SimpleImputer

    clean = frame.replace([float("inf"), -float("inf")], float("nan"))
    usable = clean.columns[clean.notna().any(axis=0)]
    if len(usable) == 0:
        raise ValueError("No observed predictor values in the training fold.")
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(
        imputer.fit_transform(clean[usable]), columns=usable, index=frame.index
    )


warnings.filterwarnings("ignore")


USE_PLQY = False


def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {}

    # Molecular descriptors
    features["MolWt"] = Descriptors.MolWt(mol)
    features["TPSA"] = Descriptors.TPSA(mol)
    features["NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
    features["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
    features["NumHDonors"] = Descriptors.NumHDonors(mol)
    features["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
    features["RingCount"] = Descriptors.RingCount(mol)
    features["FractionCSP3"] = Descriptors.FractionCSP3(mol)

    # Conjugation descriptors
    features["NumConjugatedBonds"] = sum(
        int(b.GetIsConjugated()) for b in mol.GetBonds()
    )

    # Heteroatom counts
    atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    features["Count_B"] = atom_symbols.count("B")
    features["Count_N"] = atom_symbols.count("N")
    features["Count_O"] = atom_symbols.count("O")
    features["Count_S"] = atom_symbols.count("S")
    features["Count_F"] = atom_symbols.count("F")

    # Derived 2D descriptors
    n_atoms = mol.GetNumAtoms()
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    sp2_like_atoms = sum(
        1 for a in mol.GetAtoms() if a.GetHybridization().name in ["SP2", "SP"]
    )

    features["AromaticAtomFrac"] = aromatic_atoms / (n_atoms + 1e-6)
    features["SP2AtomFrac"] = sp2_like_atoms / (n_atoms + 1e-6)
    features["HeteroAtomFrac"] = (
        atom_symbols.count("B")
        + atom_symbols.count("N")
        + atom_symbols.count("O")
        + atom_symbols.count("S")
        + atom_symbols.count("F")
    ) / (n_atoms + 1e-6)

    # Morgan FP.
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    fp_array = np.array(list(fp), dtype=int)
    for i, bit in enumerate(fp_array):
        features[f"FP_{i}"] = bit

    return features


def find_existing_col(possible_names, columns):
    for name in possible_names:
        if name in columns:
            return name
    return None


def make_feature_selection_model(seed=42):
    return LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.8,
        reg_lambda=1.5,
        random_state=seed,
        verbosity=-1,
    )


def make_final_model(seed=42):
    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.04,
        num_leaves=12,
        max_depth=4,
        min_child_samples=25,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=seed,
        verbosity=-1,
    )


def run_five_fold_random_cv(X, y, smiles, top_n=30):
    """Random five-fold CV on the supplied dataset; feature selection stays within each fold."""
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    smiles = pd.Series(smiles).reset_index(drop=True)

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    fold_rows = []
    selected_feature_rows = []

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        X_fit = X.iloc[fit_idx]
        X_val = X.iloc[val_idx]
        y_fit = y[fit_idx]
        y_val = y[val_idx]

        fold_fs = make_feature_selection_model(seed=42 + fold)
        fold_fs.fit(X_fit, y_fit)
        fold_importance = pd.DataFrame(
            {
                "feature": X_fit.columns,
                "importance": fold_fs.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        fold_features = fold_importance.head(top_n)["feature"].tolist()

        for rank, feature in enumerate(fold_features, start=1):
            selected_feature_rows.append(
                {
                    "fold": fold,
                    "rank": rank,
                    "feature": feature,
                }
            )

        fold_model = make_final_model(seed=42 + fold)
        fold_model.fit(X_fit[fold_features], y_fit)
        pred_fit = fold_model.predict(X_fit[fold_features])
        pred_val = fold_model.predict(X_val[fold_features])
        oof_pred[val_idx] = pred_val
        oof_fold[val_idx] = fold

        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(fit_idx),
                "n_validation": len(val_idx),
                "train_MAE_nm": mean_absolute_error(y_fit, pred_fit),
                "train_RMSE_nm": np.sqrt(mean_squared_error(y_fit, pred_fit)),
                "train_R2": r2_score(y_fit, pred_fit),
                "validation_MAE_nm": mean_absolute_error(y_val, pred_val),
                "validation_RMSE_nm": np.sqrt(mean_squared_error(y_val, pred_val)),
                "validation_R2": r2_score(y_val, pred_val),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(
        [
            {
                "metric": "MAE_nm",
                "mean": fold_df["validation_MAE_nm"].mean(),
                "std": fold_df["validation_MAE_nm"].std(ddof=1),
                "overall_OOF": mean_absolute_error(y, oof_pred),
            },
            {
                "metric": "RMSE_nm",
                "mean": fold_df["validation_RMSE_nm"].mean(),
                "std": fold_df["validation_RMSE_nm"].std(ddof=1),
                "overall_OOF": np.sqrt(mean_squared_error(y, oof_pred)),
            },
            {
                "metric": "R2",
                "mean": fold_df["validation_R2"].mean(),
                "std": fold_df["validation_R2"].std(ddof=1),
                "overall_OOF": r2_score(y, oof_pred),
            },
        ]
    )
    oof_df = pd.DataFrame(
        {
            "dataset_row": np.arange(len(y)),
            "fold": oof_fold,
            "Smiles": smiles,
            "actual_PL_peak": y,
            "oof_predicted_PL_peak": oof_pred,
            "residual": oof_pred - y,
        }
    )
    selected_feature_df = pd.DataFrame(selected_feature_rows)
    return fold_df, summary_df, oof_df, selected_feature_df


def plot_oof_diagnostics(oof_df, save_path):
    """Plot held-out predictions pooled across the five CV folds."""
    actual = oof_df["actual_PL_peak"].to_numpy()
    predicted = oof_df["oof_predicted_PL_peak"].to_numpy()
    mae = mean_absolute_error(actual, predicted)
    rmse_value = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(actual, predicted, alpha=0.7)
    limits = [
        min(actual.min(), predicted.min()) - 5,
        max(actual.max(), predicted.max()) + 5,
    ]
    axes[0].plot(limits, limits, "k--")
    axes[0].set(
        xlabel="Experimental PL peak (nm)",
        ylabel="OOF predicted PL peak (nm)",
        title=f"Random five-fold CV (n={len(actual)})",
    )
    axes[0].text(
        0.05,
        0.95,
        f"MAE = {mae:.2f} nm\nRMSE = {rmse_value:.2f} nm\nR² = {r2:.3f}",
        transform=axes[0].transAxes,
        va="top",
    )
    axes[1].scatter(actual, predicted - actual, alpha=0.7)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(
        xlabel="Experimental PL peak (nm)",
        ylabel="OOF residual (nm)",
        title="OOF residuals",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prettify_axis(ax, grid_axis=None, labelsize=11):
    """Use a closed box frame for publication-style consistency."""
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")
    ax.tick_params(width=1.0, length=4, labelsize=labelsize)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.30)
    else:
        ax.grid(False)


def format_feature_label(feature: str) -> str:
    """Format feature names with consistent case, Greek symbols and subscripts."""
    mapping = {
        "S1": r"$S_1$",
        "T1": r"$T_1$",
        "dEST": r"$\Delta E_{\mathrm{ST}}$",
        "∆EST": r"$\Delta E_{\mathrm{ST}}$",
        "ΔEST": r"$\Delta E_{\mathrm{ST}}$",
        "taup": r"$\tau_{\mathrm{p}}$",
        "τp": r"$\tau_{\mathrm{p}}$",
        "log_taup": r"log$_{10}(\tau_{\mathrm{p}})$",
        "f": r"$f$",
        "HOMO": "HOMO",
        "LUMO": "LUMO",
        "HL_gap": r"$E_{\mathrm{LUMO}}-E_{\mathrm{HOMO}}$",
        "S1_T1_gap": r"$S_1-T_1$",
        "orbital_center": "Orbital center",
        "f_div_dE": r"$f/\Delta E_{\mathrm{ST}}$",
        "f_times_dE": r"$f\times\Delta E_{\mathrm{ST}}$",
        "PLQY": "PLQY",
        "MolWt": "Molecular weight",
        "ExactMolWt": "Exact molecular weight",
        "TPSA": "TPSA",
        "MolLogP": "MolLogP",
        "MolMR": "MolMR",
        "BertzCT": "BertzCT",
        "NumAromaticRings": "Number of aromatic rings",
        "NumAliphaticRings": "Number of aliphatic rings",
        "NumSaturatedRings": "Number of saturated rings",
        "NumHAcceptors": "Number of H acceptors",
        "NumHDonors": "Number of H donors",
        "NumRotatableBonds": "Number of rotatable bonds",
        "RingCount": "Ring count",
        "FractionCSP3": r"Fraction Csp$^3$",
        "NumConjugatedBonds": "Number of conjugated bonds",
        "NumAromaticBonds": "Number of aromatic bonds",
        "ConjugatedBondFrac": "Conjugated bond fraction",
        "AromaticBondFrac": "Aromatic bond fraction",
        "RotatableBondFrac": "Rotatable bond fraction",
        "AromaticAtomFrac": "Aromatic atom fraction",
        "SP2AtomFrac": r"sp$^2$ atom fraction",
        "HeteroAtomFrac": "Heteroatom fraction",
        "FusedRingPairCount": "Fused ring pair count",
        "AromaticRingSystemCount": "Aromatic ring system count",
        "BridgeheadAtomCount": "Bridgehead atom count",
        "SpiroAtomCount": "Spiro atom count",
        "NumAromaticHeterocycles": "Number of aromatic heterocycles",
        "NumAromaticCarbocycles": "Number of aromatic carbocycles",
    }
    if feature in mapping:
        return mapping[feature]
    if feature.startswith("Count_"):
        return "Count " + feature.replace("Count_", "")
    if feature.endswith("_frac") and len(feature) > 5:
        return feature.replace("_frac", " fraction")
    if feature.startswith("fg_"):
        return feature.replace("fg_", "FG: ").replace("_", " ")
    if feature.startswith("motif_match_"):
        return "Motif match " + feature.replace("motif_match_", "")
    if feature.startswith("motif_count_"):
        return "Motif count " + feature.replace("motif_count_", "")
    if feature.startswith("motif_sim_"):
        return "Motif similarity " + feature.replace("motif_sim_", "")
    if feature.startswith("motif_any_"):
        return "Any motif " + feature.replace("motif_any_", "")
    if feature.startswith("motif_max_sim_"):
        return "Max motif similarity " + feature.replace("motif_max_sim_", "")
    if feature.startswith("has_"):
        return feature.replace("has_", "Has ").replace("_", " ")
    return feature.replace("_", " ")


def main():
    file_path = str(PROJECT_DIR / "data" / "tadf_dataset.xlsx")
    df = pd.read_excel(file_path)
    df.columns = [c.strip() for c in df.columns]

    OUTPUT_DIR = (
        PROJECT_DIR
        / "results"
        / (
            "plpeak_expensive_outputs_cv_full"
            + ("_indicators" if ADD_MISSING_INDICATORS else "")
        )
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SMILES_COL = "Smiles"
    TARGET_COL = "PL-peak"
    ENV_COL = "Solution"

    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()

    if ENV_COL in df.columns:
        df[ENV_COL] = df[ENV_COL].fillna("Unknown").astype(str).str.strip()
    else:
        df[ENV_COL] = "Unknown"

    print("Original data size:", len(df))

    # Restrict the target to 400–600 nm, inclusive.
    df = df[(df[TARGET_COL] >= 400) & (df[TARGET_COL] <= 600)].copy()
    print("400–600 nm data size:", len(df))

    feature_rows = []
    valid_idx = []

    for idx, smi in enumerate(df[SMILES_COL]):
        feat = featurize(smi)
        if feat is not None:
            feature_rows.append(feat)
            valid_idx.append(idx)

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    X_struct = pd.DataFrame(feature_rows)

    print("Valid molecules:", len(df_valid))
    print("Structural feature shape before FP filtering:", X_struct.shape)

    fp_cols = [c for c in X_struct.columns if c.startswith("FP_")]
    desc_cols = [c for c in X_struct.columns if not c.startswith("FP_")]

    # Retain fingerprint bits present in at least 3% of rows.
    fp_freq = X_struct[fp_cols].mean(axis=0)
    keep_fp_cols = fp_freq[fp_freq >= 0.03].index.tolist()

    X_struct = pd.concat(
        [X_struct[desc_cols].copy(), X_struct[keep_fp_cols].copy()], axis=1
    )

    print("FP before filtering:", len(fp_cols))
    print("FP after filtering :", len(keep_fp_cols))
    print("Structural feature shape after FP filtering:", X_struct.shape)

    # Encode Solution as the sole environment feature.

    X_env = pd.get_dummies(
        df_valid[[ENV_COL]], columns=[ENV_COL], prefix=[ENV_COL], drop_first=False
    )

    print("Environment feature shape:", X_env.shape)

    candidate_map = {
        "S1": ["S1"],
        "T1": ["T1"],
        "dEST": ["∆EST", "ΔEST", "dEST"],
        "taup": ["τp", "tp", "taup"],
        "f": ["f"],
        "LUMO": ["LUMO"],
        "HOMO": ["HOMO"],
        "PLQY": ["PLQY"],
    }

    resolved_cols = {}
    for key, names in candidate_map.items():
        resolved_cols[key] = find_existing_col(names, df_valid.columns)

    print("\nResolved electronic columns:")
    for k, v in resolved_cols.items():
        print(f"{k}: {v}")

    for key, col in resolved_cols.items():
        if col is not None:
            df_valid[key] = pd.to_numeric(df_valid[col], errors="coerce")
        else:
            df_valid[key] = np.nan

    df_valid["HL_gap"] = df_valid["LUMO"] - df_valid["HOMO"]
    df_valid["orbital_center"] = (df_valid["HOMO"] + df_valid["LUMO"]) / 2

    # Exclude nonpositive values before the logarithmic transform.
    df_valid["f_div_dE"] = df_valid["f"] / (df_valid["dEST"] + 1e-6)
    df_valid["f_times_dE"] = df_valid["f"] * df_valid["dEST"]

    # Log-transform the prompt lifetime.
    df_valid["log_taup"] = np.log10(df_valid["taup"] + 1e-6)

    elec_feature_cols = [
        "S1",
        "T1",
        "dEST",
        "f",
        "LUMO",
        "HOMO",
        "HL_gap",
        "orbital_center",
        "f_div_dE",
        "f_times_dE",
        "log_taup",
    ]

    if USE_PLQY:
        elec_feature_cols.append("PLQY")

    X_elec = prepare_missing_features(df_valid[elec_feature_cols])
    print("Electronic feature shape:", X_elec.shape)

    X_all = pd.concat(
        [
            X_struct.reset_index(drop=True),
            X_env.reset_index(drop=True),
            X_elec.reset_index(drop=True),
        ],
        axis=1,
    )

    y = df_valid[TARGET_COL].values

    print("Final feature matrix shape:", X_all.shape)
    print("Using PLQY:", USE_PLQY)

    # Use the curated input without automatic outlier removal.

    df_valid_clean = df_valid.reset_index(drop=True)
    X_all_clean = X_all.reset_index(drop=True)
    y_clean = df_valid_clean[TARGET_COL].values

    print(
        "Using manually cleaned dataset; no automatic outlier detection is performed."
    )
    print("Cleaned size:", len(df_valid_clean))

    # Full-dataset random five-fold CV with no independent holdout

    X_train = X_all_clean
    y_train = y_clean
    smiles_train = df_valid_clean[SMILES_COL].astype(str)
    print("Full dataset CV size:", len(X_train))

    print("\n===== Random five-fold cross-validation on full dataset =====")
    cv_folds_df, cv_summary_df, cv_oof_df, cv_features_df = run_five_fold_random_cv(
        X_train,
        y_train,
        smiles_train,
        top_n=30,
    )
    print(cv_folds_df.to_string(index=False))
    print("\nCross-validation summary (mean +/- SD):")
    print(cv_summary_df.to_string(index=False))

    cv_folds_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_random_cv_folds_expensive.csv"), index=False
    )
    cv_summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_random_cv_summary_expensive.csv"),
        index=False,
    )
    cv_oof_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_random_cv_oof_predictions_expensive.csv"),
        index=False,
    )
    cv_features_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_random_cv_selected_features_expensive.csv"),
        index=False,
    )

    # Feature selection on the full dataset after CV

    model_fs = make_feature_selection_model(seed=42)

    model_fs.fit(X_train, y_train)

    importance_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": model_fs.feature_importances_}
    ).sort_values("importance", ascending=False)

    TOP_N = 30
    selected_features = importance_df.head(TOP_N)["feature"].tolist()

    print("\nTop selected features:")
    print(selected_features)

    # Fit the final screening model on all data after OOF evaluation

    X_train_sel = X_train[selected_features].copy()
    model = make_final_model(seed=42)
    model.fit(X_train_sel, y_train)
    tag = "with_PLQY" if USE_PLQY else "without_PLQY"

    # Unified publication plotting style

    DPI = 300
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "ytick.direction": "out",
            "xtick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    COLOR_TRAIN = "#D55E5E"  # muted red
    COLOR_TEST = "#2C7FB8"  # paper-style blue
    COLOR_BLUE = "#2C7FB8"
    COLOR_BLUE_DARK = "#1F5E8C"
    COLOR_GREY = "#5A5A5A"

    # Larger fonts used specifically for SHAP and feature-importance figures.
    INTERPRET_TICK_SIZE = 14
    INTERPRET_LABEL_SIZE = 16
    INTERPRET_TITLE_SIZE = 17
    INTERPRET_COLORBAR_LABEL_SIZE = 15

    def enlarge_shap_fonts(fig, ax, xlabel):
        """Increase beeswarm feature names, axis text, and colorbar text."""
        ax.set_xlabel(xlabel, fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
        ax.tick_params(axis="both", labelsize=INTERPRET_TICK_SIZE)
        for extra_ax in fig.axes:
            if extra_ax is ax:
                continue
            extra_ax.tick_params(labelsize=INTERPRET_TICK_SIZE)
            extra_ax.xaxis.label.set_size(INTERPRET_COLORBAR_LABEL_SIZE)
            extra_ax.yaxis.label.set_size(INTERPRET_COLORBAR_LABEL_SIZE)

    def save_figure(fig, save_path: str):
        fig.tight_layout()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)

        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

        pdf_path = str(save_path).rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

        plt.close(fig)

    def plot_feature_importance(importance_sel: pd.DataFrame, save_path: str, top_n=20):
        top_imp = (
            importance_sel.sort_values("importance", ascending=True).tail(top_n).copy()
        )
        top_imp["feature_label"] = top_imp["feature"].apply(format_feature_label)

        fig, ax = plt.subplots(figsize=(8.8, max(6.2, 0.36 * len(top_imp) + 1.6)))
        ax.barh(
            top_imp["feature_label"],
            top_imp["importance"],
            color=COLOR_BLUE,
            edgecolor=COLOR_BLUE_DARK,
            linewidth=0.4,
        )
        ax.set_xlabel("Feature importance", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
        ax.set_ylabel("Feature", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
        ax.set_title(
            f"Top {len(top_imp)} feature importances",
            fontsize=INTERPRET_TITLE_SIZE,
            pad=10,
        )
        prettify_axis(ax, grid_axis="x", labelsize=INTERPRET_TICK_SIZE)
        save_figure(fig, save_path)

    def feature_group_name(feature: str) -> str:
        if feature.startswith("MORGAN"):
            return "Morgan fingerprint"
        if feature.startswith("MACCS"):
            return "MACCS keys"
        if feature.startswith("RDKFP"):
            return "RDKit fingerprint"
        if feature.startswith("PATTERN"):
            return "Pattern fingerprint"
        if feature.startswith(f"{ENV_COL}_"):
            return "Environment"
        if feature.startswith("fg_"):
            return "Functional group"
        if feature.startswith("motif_match_") or feature.startswith("motif_count_"):
            return "Exact A/B/C motif match"
        if feature.startswith("motif_sim_") or feature.startswith("motif_max_sim_"):
            return "A/B/C motif similarity"
        if (
            feature.startswith("motif_any_")
            or feature.startswith("motif_n_")
            or feature.startswith("motif_total_count_")
        ):
            return "A/B/C motif summary"
        if feature.startswith("has_B"):
            return "MR core proxy"
        if feature in [
            "S1",
            "T1",
            "dEST",
            "f",
            "HOMO",
            "LUMO",
            "taup",
            "log_taup",
            "HL_gap",
            "S1_T1_gap",
            "orbital_center",
            "f_div_dE",
            "f_times_dE",
        ]:
            return "Electronic descriptor"
        if (
            feature.startswith("core_")
            or feature.startswith("Core_")
            or feature.startswith("Motif_")
            or feature.startswith("motif_")
        ):
            return "Core/motif label"
        return "2D descriptor"

    def plot_feature_group_importance(importance_sel: pd.DataFrame, save_path: str):
        df = importance_sel.copy()
        df["group"] = df["feature"].apply(feature_group_name)
        group_imp = (
            df.groupby("group", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(8.8, max(5.6, 0.44 * len(group_imp) + 1.6)))
        ax.barh(
            group_imp["group"],
            group_imp["importance"],
            color=COLOR_BLUE,
            edgecolor=COLOR_BLUE_DARK,
            linewidth=0.4,
        )
        ax.set_xlabel(
            "Summed feature importance", fontsize=INTERPRET_LABEL_SIZE, labelpad=8
        )
        ax.set_ylabel("Feature group", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
        ax.set_title("Feature group importance", fontsize=INTERPRET_TITLE_SIZE, pad=10)
        prettify_axis(ax, grid_axis="x", labelsize=INTERPRET_TICK_SIZE)
        save_figure(fig, save_path)

    def plot_shap_summary(
        model, X_for_shap: pd.DataFrame, save_path: str, max_display=20
    ):
        """Create a SHAP beeswarm summary with the same closed-frame style."""
        try:
            import shap
        except Exception as exc:
            print(
                f"SHAP is not installed or cannot be imported. Skipped SHAP plot: {exc}"
            )
            return

        X_plot = X_for_shap.copy()
        if len(X_plot) > 300:
            X_plot = X_plot.sample(300, random_state=RANDOM_STATE)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_plot)
            display_names = [format_feature_label(c) for c in X_plot.columns]
            X_display = X_plot.copy()
            X_display.columns = display_names

            plt.figure(figsize=(8.8, 7.2), dpi=DPI)
            shap.summary_plot(
                shap_values, X_display, max_display=max_display, show=False
            )
            ax = plt.gca()
            fig = plt.gcf()
            enlarge_shap_fonts(fig, ax, "SHAP value / nm")
            prettify_axis(ax, labelsize=INTERPRET_TICK_SIZE)
            fig.tight_layout()
            if not os.path.isabs(save_path):
                save_path = os.path.join(OUTPUT_DIR, save_path)
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            fig.savefig(str(save_path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"SHAP plot failed and was skipped: {exc}")

    diag_name = f"oof_diagnostics_{tag}.png"
    plot_oof_diagnostics(cv_oof_df, os.path.join(OUTPUT_DIR, diag_name))

    importance_sel = pd.DataFrame(
        {"feature": selected_features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    imp_name = f"feature_importance_top20_{tag}.png"
    group_imp_name = f"feature_group_importance_{tag}.png"
    shap_name = f"shap_summary_{tag}.png"

    plot_feature_importance(importance_sel, imp_name, top_n=20)
    plot_feature_group_importance(importance_sel, group_imp_name)
    plot_shap_summary(model, X_train_sel, shap_name, max_display=20)

    print("\nSaved figures:")
    print(f"1. {os.path.join(OUTPUT_DIR, diag_name)}")
    print(f"2. {os.path.join(OUTPUT_DIR, imp_name)}")
    print(f"3. {os.path.join(OUTPUT_DIR, group_imp_name)}")
    print(f"4. {os.path.join(OUTPUT_DIR, shap_name)}  (if SHAP is installed)")
    print(
        f"5. {os.path.join(OUTPUT_DIR, 'five_fold_random_cv_oof_predictions_expensive.csv')}"
    )
    print(f"6. {os.path.join(OUTPUT_DIR, 'final_pl_peak_expensive.pkl')}")
    print(f"7. {os.path.join(OUTPUT_DIR, 'train_features_columns.csv')}")

    # Save the feature order explicitly for libraries that omit feature_names_in_.

    import joblib

    model_bundle = {
        "model": model,
        "selected_features": [str(c) for c in selected_features],
        "all_feature_columns": [str(c) for c in X_all_clean.columns],
        "metadata": {
            "task": "PL-peak regression",
            "feature_set": "expensive",
            "target_col": TARGET_COL,
            "smiles_col": SMILES_COL,
            "environment_col": ENV_COL,
            "use_plqy": USE_PLQY,
            "pl_peak_range_nm": [400, 600],
            "data_cleaning": "manual removal before running script",
            "n_cleaned": int(len(df_valid_clean)),
            "n_selected_features": int(len(selected_features)),
            "model_type": "LGBMRegressor",
            "random_state": 42,
            "validation": "shuffled five-fold KFold on all valid molecules; no independent holdout",
            "cv_n_splits": 5,
            "cv_n_samples": int(len(y_clean)),
            "final_fit_n_samples": int(len(y_clean)),
        },
    }

    joblib.dump(model_bundle, os.path.join(OUTPUT_DIR, "final_pl_peak_expensive.pkl"))

    pd.DataFrame({"feature": selected_features}).to_csv(
        os.path.join(OUTPUT_DIR, "train_features_columns.csv"), index=False
    )


if __name__ == "__main__":
    main()
