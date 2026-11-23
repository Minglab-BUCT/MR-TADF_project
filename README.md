# MR-TADF machine learning and scaffold-based generation

This repository contains models for the emission properties of multi-resonance thermally activated delayed fluorescence (MR-TADF) molecules, external validation on a separate set of molecules, and a scaffold-based candidate generation and screening workflow.

## Environment

The scripts were checked with Python 3.10.18. The requirements file records the direct dependencies installed in that environment. Run the commands from the repository root.

```sh
python -m pip install -r requirements.txt
```

Use `MPLBACKEND=Agg` for headless runs. The screening script uses the SA_Score implementation and fragment scores distributed with RDKit Contrib.

## Project layout

| Directory | Contents |
| --- | --- |
| `src/models/` | Six Python training scripts |
| `src/validation/` | PL-peak cross-validation and external validation |
| `src/generation/` | Scaffold generation, structural filtering, and property screening |
| `data/` | Training and external validation workbooks, scaffold definitions, and substitution patterns |
| `models/` | Six saved model files for validation and screening |
| `reference results/` | Archived generation, structural filtering, and property-screening tables |

## Models

| Study task | Feature set | Final estimator | Training script |
| --- | --- | --- | --- |
| PL-peak wavelength regression | Low-cost | LightGBM regressor | `src/models/MR_TADF_PL_lowcost.py` |
| PL-peak wavelength regression | Augmented | LightGBM regressor | `src/models/MR_TADF_PL_Augmented.py` |
| High-PLQY classification | Low-cost | Random forest classifier | `src/models/MR_TADF_PLQY_lowcost.py` |
| High-PLQY classification | Augmented | Random forest classifier | `src/models/MR_TADF_PLQY_Augmented.py` |
| Narrow-FWHM classification | Low-cost | Random forest classifier | `src/models/MR_TADF_FWHM_lowcost.py` |
| Narrow-FWHM classification | Augmented | Random forest classifier | `src/models/MR_TADF_FWHM_Augmented.py` |

Run a training script with `python src/models/<script.py>`. The training scripts preserve input row order and use a fixed seed of 42 for the 80/20 random split. PLQY and FWHM stratify the random split by class and also report five-fold scaffold-grouped evaluation. The positive classes are PLQY ≥ 80% and FWHM ≤ 30 nm, with both boundary values included.

Low-cost models derive molecular descriptors, fingerprints, and, where used, core or fragment descriptors from SMILES. Some scripts also use solvent or other categorical information. Augmented models combine structural predictors with script-specific physical and electronic columns from the workbook, such as excited-state energies, oscillator strength, prompt lifetime, and frontier orbital energies. The structural features, accepted column aliases, and selected feature order vary by script.

## Inputs and saved models

| File | Role |
| --- | --- |
| `data/tadf_dataset.xlsx` | Training, cross-validation, and generation source workbook. The first sheet contains `Molecules`, `Smiles`, `Solution`, `PL-peak`, `PLQY`, `FWHM`, and the available physical features, including `S1`, `T1`, `∆EST`, `τp`, `f`, `HOMO`, and `LUMO`. The generator uses the molecule and SMILES columns and retains available source properties in the learned-pattern table. |
| `data/core-structurexlsx.xlsx` | Core and fragment workbook with the original `core-type` and `smiles` columns. |
| `data/test molecules.xlsx` | External validation workbook. The scripts read the first sheet, including the original `Molecules`, `Smiles`, `Solution`, experimental targets, and available physical features. |
| `reference results/generated_candidates_50000.csv` | Supplied archive of 50,000 generated unique candidates, including the descriptors used by the structural filter. |
| `data/learned_substitution_patterns.csv` and `reference results/learned_substitution_patterns.csv` | Copies of the substitution patterns learned in the archived generation run. |
| `reference results/generated_candidates_10681.csv` | Supplied archive of 10,681 candidates that passed the first-stage structural filter. Use this table as the input to reproduce property screening on the archived library. |
| `reference results/screened_candidates_1972.csv` | Archived property-screening results, ordered by the screening rank. |

Keep workbook headers and row order when reproducing the supplied runs. Load model pickle files from trusted sources, as loading a pickle can execute code.

## PL-peak five-fold cross-validation

```sh
python src/validation/MR_TADF_PL_lowcost_cv.py
python src/validation/MR_TADF_PL_Augmented_cv.py
```

Both scripts use shuffled five-fold cross-validation with seed 42 on eligible rows and select features within each training fold. Each writes fold metrics, summary metrics, out-of-fold predictions, and per-fold selected features. With the supplied dataset, the augmented script writes those tables before reaching an existing undefined `RANDOM_STATE` reference in SHAP plotting. That error interrupts later outputs.

## External validation

```sh
python src/validation/test-molecules.py --model-dir models
```

The command loads the six saved models in `models/` and writes predictions, metrics, feature alignment reports, the FWHM evaluation inclusion table, and figures. To validate newly trained exports using the script’s default paths, run the six training scripts first and omit `--model-dir`.

The current `.gitignore` excludes `.pkl` files, including the local copies in `models/`.

| Model | Saved filename |
| --- | --- |
| PL-peak low-cost | `final_pl_peak_lowcost.pkl` |
| PL-peak augmented | `final_pl_peak_expensive.pkl` |
| PLQY low-cost | `final_plqy_rf_lowcost.pkl` |
| PLQY augmented | `final_plqy_rf_expensive.pkl` |
| FWHM low-cost | `final_fwhm_rf_lowcost.pkl` |
| FWHM augmented | `final_fwhm_rf_expensive.pkl` |

## Generation and screening

To generate a new library and screen it with the saved low-cost models, run the following commands.

```sh
python src/generation/scaffold_based_generator.py
python src/generation/filter_molecules.py --output filtered_candidates.csv
python src/generation/scaffold_based_screening.py \
  --input filtered_candidates.csv \
  --plpeak_model models/final_pl_peak_lowcost.pkl \
  --plqy_model models/final_plqy_rf_lowcost.pkl \
  --fwhm_model models/final_fwhm_rf_lowcost.pkl
```

The generator learns substitutions from the source workbook, recombines fragments on A/B cores, validates structures, and deduplicates canonical SMILES. The default source is the 393-row workbook `data/tadf_dataset.xlsx`. The supplied 50,000-candidate archive was generated from a separate 390-row source. Use the archived candidate tables to reproduce filtering and screening on that library.

The structural filter uses the descriptors recorded in the generated CSV. It retains novel structures with molecular weight 350–1200 Da, 25–90 heavy atoms, 4–14 aromatic rings, logP ≤ 15, at most 12 rotatable bonds, and at most five modifications. The bounds are inclusive. To filter the archived library, run `python src/generation/filter_molecules.py --input "reference results/generated_candidates_50000.csv"`.

Property screening uses the three low-cost training exports by default. Use `--plpeak_model`, `--plqy_model`, and `--fwhm_model` to supply existing saved files. Screening retains candidates with predicted PL-peak in 430–490 nm inclusive, both classifier labels positive and their corresponding probabilities at least 0.50, and SA score strictly below 6. It ranks survivors using the PLQY and FWHM probabilities, wavelength, and SA score and exports the top 50 separately. To screen the archived shortlist, use the screening command above with `--input "reference results/generated_candidates_10681.csv"` in place of `--input filtered_candidates.csv`.
