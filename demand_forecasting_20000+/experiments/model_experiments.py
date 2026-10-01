# ============================================================
# DEMAND FORECASTING - MODEL EXPERIMENTS
# ============================================================
#
# PURPOSE
# -------
# 1. Load and clean data
# 2. Create features
# 3. Prepare train/test data
# 4. Train candidate models
# 5. Compare candidate models
# 6. Select the model for tuning
# 7. SAVE THE EXACT TRAINED MODEL USED FOR PREDICTION
#
# Important:
# The selected model saved here is the SAME model object that
# produced the baseline predictions and baseline metrics.
#
# ============================================================

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT MODULES
# ============================================================

from src.data_cleaning import load_and_clean_data
from src.eda import run_eda

from src.evaluation import (
    calculate_generalization,
    calculate_metrics,
    select_best_model,
)

from src.feature_engineering import (
    create_features,
    prepare_model_data,
)

from src.models import (
    predict_model,
    train_selected_model,
)

from src.plots import (
    plot_model_comparison_mae,
    plot_model_comparison_r2,
    plot_model_comparison_rmse,
    plot_train_test_mae,
    plot_train_test_rmse,
)


# ============================================================
# DIRECTORIES
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales_demand.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

EDA_DIR = (
    OUTPUT_DIR
    / "eda"
)

PLOTS_DIR = (
    OUTPUT_DIR
    / "plots"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

EXPERIMENTS_DIR = (
    PROJECT_ROOT
    / "experiments"
)

EVIDENCE_DIR = (
    EXPERIMENTS_DIR
    / "evidence"
)


# Create directories
for directory in [
    OUTPUT_DIR,
    EDA_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    EVIDENCE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DEMAND FORECASTING - MODEL EXPERIMENTS")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset:")
print(DATA_PATH)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


# ============================================================
# STEP 1 - LOAD AND CLEAN DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: LOAD AND CLEAN DATA")
print("=" * 70)

df = load_and_clean_data(
    DATA_PATH
)

print(
    "\nData shape:",
    df.shape
)


# ============================================================
# STEP 2 - EDA
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

run_eda(
    df,
    EDA_DIR
)

print(
    "\nEDA completed."
)


# ============================================================
# STEP 3 - FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 70)

df_features = create_features(
    df
)

print(
    "\nFeature-engineered shape:",
    df_features.shape
)

print(
    "\nFeature columns:"
)

for column in df_features.columns:
    print(
        "-",
        column
    )


# ============================================================
# STEP 4 - PREPARE MODEL DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: PREPARE MODEL DATA")
print("=" * 70)

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    feature_names,
    train_df,
    test_df,
) = prepare_model_data(
    df_features
)


print(
    "\nTraining shape:",
    X_train.shape
)

print(
    "Testing shape:",
    X_test.shape
)

print(
    "\nTraining dates:"
)

print(
    train_df["Date"].min(),
    "to",
    train_df["Date"].max()
)

print(
    "\nTesting dates:"
)

print(
    test_df["Date"].min(),
    "to",
    test_df["Date"].max()
)


# ============================================================
# STEP 5 - CANDIDATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: CANDIDATE MODELS")
print("=" * 70)

model_names = [
    "Random Forest - Original",
    "XGBoost - Original",
    "CatBoost - Original",
    "XGBoost - Log Target",
    "CatBoost - Log Target",
]

for model_name in model_names:

    print(
        "-",
        model_name
    )


# ============================================================
# STEP 6 - TRAIN CANDIDATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: TRAIN CANDIDATE MODELS")
print("=" * 70)

trained_models = {}
test_predictions = {}
train_predictions = {}
target_types = {}


for model_name in model_names:

    print("\n" + "-" * 70)
    print(
        "Training:",
        model_name
    )
    print("-" * 70)

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model, target_type = train_selected_model(
        model_name,
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # TEST PREDICTION
    # --------------------------------------------------------

    test_pred = np.asarray(
        predict_model(
            model,
            target_type,
            X_test,
        )
    )

    # --------------------------------------------------------
    # TRAIN PREDICTION
    # --------------------------------------------------------

    train_pred = np.asarray(
        predict_model(
            model,
            target_type,
            X_train,
        )
    )

    # --------------------------------------------------------
    # STORE EVERYTHING
    # --------------------------------------------------------

    trained_models[
        model_name
    ] = model

    test_predictions[
        model_name
    ] = test_pred

    train_predictions[
        model_name
    ] = train_pred

    target_types[
        model_name
    ] = target_type

    print(
        "Training completed."
    )


# ============================================================
# STEP 7 - TEST METRICS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: TEST SET EVALUATION")
print("=" * 70)


metrics_list = []

for model_name in model_names:

    metrics = calculate_metrics(
        y_test,
        test_predictions[model_name],
        model_name,
    )

    metrics_list.append(
        metrics
    )


metrics_df = pd.DataFrame(
    metrics_list
)


print(
    "\nCandidate model metrics:"
)

print(
    metrics_df.to_string(
        index=False
    )
)


metrics_path = (
    OUTPUT_DIR
    / "model_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)


# ============================================================
# STEP 8 - GENERALIZATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: TRAIN VS TEST GENERALIZATION")
print("=" * 70)


generalization_list = []

for model_name in model_names:

    generalization = calculate_generalization(
        y_train,
        train_predictions[model_name],
        y_test,
        test_predictions[model_name],
        model_name,
    )

    generalization_list.append(
        generalization
    )


generalization_df = pd.DataFrame(
    generalization_list
)


print(
    "\nGeneralization results:"
)

print(
    generalization_df.to_string(
        index=False
    )
)


generalization_path = (
    OUTPUT_DIR
    / "generalization_results.csv"
)

generalization_df.to_csv(
    generalization_path,
    index=False,
)


# ============================================================
# STEP 9 - MODEL COMPARISON PLOTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9: MODEL COMPARISON PLOTS")
print("=" * 70)


plot_model_comparison_rmse(
    metrics_df,
    PLOTS_DIR
    / "01_model_comparison_rmse.png",
)

plot_model_comparison_mae(
    metrics_df,
    PLOTS_DIR
    / "02_model_comparison_mae.png",
)

plot_model_comparison_r2(
    metrics_df,
    PLOTS_DIR
    / "03_model_comparison_r2.png",
)

plot_train_test_rmse(
    generalization_df,
    PLOTS_DIR
    / "04_train_vs_test_rmse.png",
)

plot_train_test_mae(
    generalization_df,
    PLOTS_DIR
    / "05_train_vs_test_mae.png",
)


# ============================================================
# STEP 10 - SELECT MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 10: SELECT MODEL TO TUNE")
print("=" * 70)


selected_model_name, selection_df = select_best_model(
    metrics_df,
    generalization_df,
)


print(
    "\nSelected model:"
)

print(
    selected_model_name
)


print(
    "\nSelection results:"
)

print(
    selection_df.to_string(
        index=False
    )
)


selection_path = (
    OUTPUT_DIR
    / "model_selection.csv"
)

selection_df.to_csv(
    selection_path,
    index=False,
)


selection_evidence_path = (
    EVIDENCE_DIR
    / "model_selection.csv"
)

selection_df.to_csv(
    selection_evidence_path,
    index=False,
)


# ============================================================
# STEP 11 - GET EXACT SELECTED MODEL
# ============================================================
#
# IMPORTANT
# ------------------------------------------------------------
# We DO NOT create another Pipeline here.
#
# We save the EXACT model object that was used in:
#
#     predict_model(model, target_type, X_test)
#
# This prevents the saved model from following a different
# prediction path from the model evaluated above.
# ============================================================

print("\n" + "=" * 70)
print("STEP 11: SAVE EXACT SELECTED MODEL")
print("=" * 70)


selected_estimator = trained_models[
    selected_model_name
]

selected_target_type = target_types[
    selected_model_name
]


print(
    "\nSelected model object:"
)

print(
    type(selected_estimator)
)

print(
    "\nTarget type:"
)

print(
    selected_target_type
)


# ============================================================
# SAVE EXACT TRAINED MODEL
# ============================================================

selected_model_path = (
    MODELS_DIR
    / "selected_model.pkl"
)


joblib.dump(
    selected_estimator,
    selected_model_path,
)


print(
    "\nExact selected model saved to:"
)

print(
    selected_model_path
)


# ============================================================
# SAVE PREPROCESSOR SEPARATELY
# ============================================================

preprocessor_path = (
    MODELS_DIR
    / "preprocessor.pkl"
)


joblib.dump(
    preprocessor,
    preprocessor_path,
)


print(
    "\nPreprocessor saved to:"
)

print(
    preprocessor_path
)


# ============================================================
# STEP 12 - VERIFY SAVED MODEL
# ============================================================
#
# This is the important verification.
#
# We reload selected_model.pkl immediately and use the SAME
# X_test that produced the baseline metric.
#
# The reloaded model must reproduce the exact predictions.
# ============================================================

print("\n" + "=" * 70)
print("STEP 12: VERIFY SAVED SELECTED MODEL")
print("=" * 70)


reloaded_selected_model = joblib.load(
    selected_model_path
)


# Predict with the reloaded model
reloaded_predictions = np.asarray(
    predict_model(
        reloaded_selected_model,
        selected_target_type,
        X_test,
    )
)


# Compare original and reloaded predictions
prediction_difference = np.abs(
    test_predictions[selected_model_name]
    -
    reloaded_predictions
)


max_prediction_difference = (
    prediction_difference.max()
)


mean_prediction_difference = (
    prediction_difference.mean()
)


print(
    "\nMaximum prediction difference:"
)

print(
    max_prediction_difference
)


print(
    "\nMean prediction difference:"
)

print(
    mean_prediction_difference
)


# ------------------------------------------------------------
# Verify metrics
# ------------------------------------------------------------

reloaded_metrics = calculate_metrics(
    y_test,
    reloaded_predictions,
    selected_model_name,
)


print(
    "\nOriginal baseline metrics:"
)

print(
    selected_model_name
)

print(
    calculate_metrics(
        y_test,
        test_predictions[selected_model_name],
        selected_model_name,
    )
)


print(
    "\nReloaded model metrics:"
)

print(
    reloaded_metrics
)


# ------------------------------------------------------------
# Hard verification
# ------------------------------------------------------------

if not np.allclose(
    test_predictions[selected_model_name],
    reloaded_predictions,
    rtol=1e-10,
    atol=1e-10,
):

    raise RuntimeError(
        "\nERROR:\n"
        "The saved selected_model.pkl does NOT reproduce "
        "the predictions generated before saving.\n"
        "The model saving process is inconsistent."
    )


print(
    "\nMODEL SAVE VERIFICATION PASSED."
)

print(
    "The saved model reproduces the original "
    "baseline predictions exactly."
)


# ============================================================
# STEP 13 - SAVE SELECTED MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 13: SAVE SELECTED MODEL INFORMATION")
print("=" * 70)


selected_baseline_metrics = calculate_metrics(
    y_test,
    test_predictions[selected_model_name],
    selected_model_name,
)


selected_generalization = calculate_generalization(
    y_train,
    train_predictions[selected_model_name],
    y_test,
    test_predictions[selected_model_name],
    selected_model_name,
)


selected_model_info = {
    "model_name": selected_model_name,
    "target_type": selected_target_type,
    "model_class": type(
        selected_estimator
    ).__name__,
    "feature_names": list(
        feature_names
    ),
    "encoded_feature_count": int(
        X_train.shape[1]
    ),
    "train_start": str(
        train_df["Date"].min().date()
    ),
    "train_end": str(
        train_df["Date"].max().date()
    ),
    "test_start": str(
        test_df["Date"].min().date()
    ),
    "test_end": str(
        test_df["Date"].max().date()
    ),
    "train_rows": int(
        len(train_df)
    ),
    "test_rows": int(
        len(test_df)
    ),
    "test_metrics": selected_baseline_metrics,
    "generalization": selected_generalization,
    "workflow_role": (
        "baseline selected model "
        "for hyperparameter tuning"
    ),
}


selected_model_info_path = (
    MODELS_DIR
    / "selected_model_info.pkl"
)


joblib.dump(
    selected_model_info,
    selected_model_info_path,
)


print(
    "\nSelected model information saved to:"
)

print(
    selected_model_info_path
)


# ============================================================
# STEP 14 - SAVE BEFORE-TUNING METRICS
# ============================================================

print("\n" + "=" * 70)
print("STEP 14: SAVE BEFORE-TUNING EVIDENCE")
print("=" * 70)


before_metrics_df = pd.DataFrame(
    [
        selected_baseline_metrics
    ]
)


before_metrics_df.to_csv(
    EVIDENCE_DIR
    / "selected_model_before_tuning_metrics.csv",
    index=False,
)


before_generalization_df = pd.DataFrame(
    [
        selected_generalization
    ]
)


before_generalization_df.to_csv(
    EVIDENCE_DIR
    / "selected_model_before_tuning_generalization.csv",
    index=False,
)


# ============================================================
# SAVE SELECTED MODEL PREDICTIONS
# ============================================================

prediction_columns = [
    column
    for column in [
        "Date",
        "Region",
        "Country",
        "Product_Type",
        "Season",
    ]
    if column in test_df.columns
]


before_prediction_df = (
    test_df[prediction_columns]
    .copy()
)


before_prediction_df[
    "Actual_Sales_Volume"
] = y_test.to_numpy()


before_prediction_df[
    "Predicted_Sales_Volume"
] = test_predictions[
    selected_model_name
]


before_prediction_df[
    "Residual"
] = (
    before_prediction_df[
        "Actual_Sales_Volume"
    ]
    -
    before_prediction_df[
        "Predicted_Sales_Volume"
    ]
)


before_prediction_df[
    "Absolute_Error"
] = np.abs(
    before_prediction_df[
        "Residual"
    ]
)


before_prediction_path = (
    EVIDENCE_DIR
    / "selected_model_before_tuning_predictions.csv"
)


before_prediction_df.to_csv(
    before_prediction_path,
    index=False,
)


# ============================================================
# STEP 15 - FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 15: FEATURE IMPORTANCE")
print("=" * 70)


try:

    importance_values = np.asarray(
        selected_estimator.feature_importances_
    )

    if len(importance_values) == len(
        feature_names
    ):

        feature_importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance_values,
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )


        total_importance = (
            feature_importance_df[
                "Importance"
            ].sum()
        )


        if total_importance != 0:

            feature_importance_df[
                "Importance_Percent"
            ] = (
                feature_importance_df[
                    "Importance"
                ]
                / total_importance
                * 100
            )


        feature_importance_df.to_csv(
            EVIDENCE_DIR
            / "selected_model_before_tuning_feature_importance.csv",
            index=False,
        )


        print(
            "\nFeature importance saved."
        )


except Exception as error:

    print(
        "\nFeature importance could not be saved:"
    )

    print(
        error
    )


# ============================================================
# STEP 16 - FORECAST STATE
# ============================================================

print("\n" + "=" * 70)
print("STEP 16: SAVE FORECAST STATE")
print("=" * 70)


history_columns = [
    column
    for column in [
        "Date",
        "Region",
        "Country",
        "Product_Type",
        "Product_Type_ID",
        "Season",
        "Sales_Volume",
    ]
    if column in df.columns
]


forecast_state = {
    "history": df[
        history_columns
    ].copy(),

    "first_date": df[
        "Date"
    ].min(),

    "last_date": df[
        "Date"
    ].max(),

    "countries": sorted(
        df["Country"]
        .unique()
        .tolist()
    ),

    "products": sorted(
        df["Product_Type"]
        .unique()
        .tolist()
    ),

    "regions": sorted(
        df["Region"]
        .unique()
        .tolist()
    ),

    "country_region_mapping": (
        df[
            [
                "Country",
                "Region",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "Country"
        )[
            "Region"
        ]
        .to_dict()
    ),

    "season_mapping": (
        df.assign(
            Month=df["Date"].dt.month
        )
        .groupby(
            "Month"
        )[
            "Season"
        ]
        .agg(
            lambda x:
            x.mode().iloc[0]
            if not x.mode().empty
            else x.iloc[0]
        )
        .to_dict()
    ),

    "product_id_mapping": (
        df.groupby(
            "Product_Type"
        )[
            "Product_Type_ID"
        ]
        .first()
        .to_dict()
    ),
}


forecast_state_path = (
    MODELS_DIR
    / "forecast_state.pkl"
)


joblib.dump(
    forecast_state,
    forecast_state_path,
)


print(
    "\nForecast state saved to:"
)

print(
    forecast_state_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL EXPERIMENTS COMPLETED")
print("=" * 70)


print(
    "\nSelected model:"
)

print(
    selected_model_name
)


print(
    "\nBEFORE TUNING:"
)

print(
    f"MAE  : "
    f"{selected_baseline_metrics['MAE']:.6f}"
)

print(
    f"RMSE : "
    f"{selected_baseline_metrics['RMSE']:.6f}"
)

print(
    f"R2   : "
    f"{selected_baseline_metrics['R2']:.6f}"
)

print(
    f"MAPE : "
    f"{selected_baseline_metrics['MAPE']:.6f}%"
)

print(
    f"WAPE : "
    f"{selected_baseline_metrics['WAPE']:.6f}%"
)


print(
    "\nSaved selected model:"
)

print(
    selected_model_path
)


print(
    "\nSave verification:"
)

print(
    "PASSED"
)


print(
    "\nNext step:"
)

print(
    "python experiments\\tune_selected_model.py"
)

print(
    "=" * 70
)