# ============================================================
# TUNE SELECTED MODEL
# XGBoost - Log Target
# ============================================================

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EVIDENCE_DIR = PROJECT_ROOT / "experiments" / "evidence"

PRODUCT_PLOT_DIR = (
    EVIDENCE_DIR / "actual_vs_selected_tuned"
)

sys.path.append(str(SRC_DIR))


# ============================================================
# 2. IMPORT PROJECT FUNCTIONS
# ============================================================

from data_cleaning import load_and_clean_data
from feature_engineering import create_features

# ============================================================
# 3. FILE PATHS
# ============================================================

DATA_PATH = DATA_DIR / "sales_demand.csv"

SELECTED_MODEL_PATH = (
    MODELS_DIR / "selected_model.pkl"
)

SELECTED_MODEL_INFO_PATH = (
    MODELS_DIR / "selected_model_info.pkl"
)

TUNED_MODEL_PATH = (
    MODELS_DIR / "tuned_model.pkl"
)

TUNED_MODEL_INFO_PATH = (
    MODELS_DIR / "tuned_model_info.pkl"
)

PREDICTION_EVIDENCE_PATH = (
    EVIDENCE_DIR / "before_after_2024_predictions.csv"
)


# ============================================================
# 4. CREATE DIRECTORIES
# ============================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PRODUCT_PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 5. SETTINGS
# ============================================================

TARGET = "Sales_Volume"

RANDOM_STATE = 42

TEST_YEAR = 2024

VALIDATION_YEARS = [
    2021,
    2022,
    2023,
]


# ============================================================
# 6. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    non_zero_mask = actual != 0

    if non_zero_mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    actual[non_zero_mask]
                    - predicted[non_zero_mask]
                )
                / actual[non_zero_mask]
            )
        ) * 100

    else:

        mape = np.nan

    wape_denominator = np.sum(
        np.abs(actual)
    )

    if wape_denominator != 0:

        wape = (
            np.sum(
                np.abs(
                    actual - predicted
                )
            )
            / wape_denominator
        ) * 100

    else:

        wape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "WAPE": wape,
    }


# ============================================================
# 7. PRODUCT-WISE ACTUAL VS SELECTED VS TUNED PLOTS
# ============================================================

def create_product_prediction_plots(
    prediction_df,
):

    print("\n" + "=" * 70)
    print(
        "CREATING PRODUCT-WISE "
        "ACTUAL VS SELECTED VS TUNED PLOTS"
    )
    print("=" * 70)

    plot_df = prediction_df.copy()

    plot_df["Date"] = pd.to_datetime(
        plot_df["Date"],
        errors="coerce",
    )

    plot_df["Month"] = (
        plot_df["Date"].dt.month
    )

    plot_df["Month_Name"] = (
        plot_df["Date"]
        .dt.strftime("%b")
    )

    products = sorted(
        plot_df[
            "Product_Type"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "\nProducts found:",
        len(products),
    )

    for product in products:

        print(
            "  -",
            product,
        )

    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for product in products:

        print(
            f"\nCreating plot for: {product}"
        )

        product_df = plot_df[
            plot_df["Product_Type"]
            == product
        ].copy()

        monthly_df = (
            product_df
            .groupby(
                [
                    "Date",
                    "Month",
                    "Month_Name",
                ],
                as_index=False,
            )[
                [
                    "Actual_Sales_Volume",
                    "Selected_Model_Predicted",
                    "Tuned_Model_Predicted",
                ]
            ]
            .sum()
        )

        monthly_df = (
            monthly_df
            .sort_values("Month")
            .reset_index(drop=True)
        )

        monthly_df[
            "Month_Name"
        ] = pd.Categorical(
            monthly_df["Month_Name"],
            categories=month_order,
            ordered=True,
        )

        monthly_df = (
            monthly_df
            .sort_values("Month_Name")
            .reset_index(drop=True)
        )

        plt.figure(
            figsize=(12, 6)
        )

        # Actual
        plt.plot(
            monthly_df["Month_Name"],
            monthly_df[
                "Actual_Sales_Volume"
            ],
            marker="o",
            linewidth=2,
            label="Actual",
        )

        # Selected model
        plt.plot(
            monthly_df["Month_Name"],
            monthly_df[
                "Selected_Model_Predicted"
            ],
            marker="o",
            linewidth=2,
            linestyle="--",
            label="Selected Model",
        )

        # Tuned model
        plt.plot(
            monthly_df["Month_Name"],
            monthly_df[
                "Tuned_Model_Predicted"
            ],
            marker="o",
            linewidth=2,
            linestyle=":",
            label="Tuned Model",
        )

        plt.title(
            (
                "Actual vs Selected vs Tuned "
                f"Model Demand - {product} - 2024"
            ),
            fontsize=14,
            fontweight="bold",
        )

        plt.xlabel(
            "Month",
            fontsize=11,
        )

        plt.ylabel(
            "Sales Volume",
            fontsize=11,
        )

        plt.xticks(
            month_order
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        safe_product_name = (
            str(product)
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        plot_path = (
            PRODUCT_PLOT_DIR
            / f"{safe_product_name}.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "  Plot saved:",
            plot_path,
        )

    print(
        "\nAll product plots created successfully."
    )

    print(
        "Plot directory:",
        PRODUCT_PLOT_DIR,
    )


# ============================================================
# 8. TUNED MODEL PREDICTED VS ACTUAL SCATTER
# ============================================================

def create_predicted_vs_actual_plot(
    prediction_df,
):

    print("\n" + "=" * 70)
    print(
        "CREATING TUNED MODEL "
        "PREDICTED VS ACTUAL PLOT"
    )
    print("=" * 70)

    actual = prediction_df[
        "Actual_Sales_Volume"
    ]

    predicted = prediction_df[
        "Tuned_Model_Predicted"
    ]

    plt.figure(
        figsize=(8, 7)
    )

    plt.scatter(
        actual,
        predicted,
        alpha=0.5,
    )

    min_value = min(
        actual.min(),
        predicted.min(),
    )

    max_value = max(
        actual.max(),
        predicted.max(),
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    plt.xlabel(
        "Actual Sales Volume",
        fontsize=11,
    )

    plt.ylabel(
        "Tuned Model Predicted Sales Volume",
        fontsize=11,
    )

    plt.title(
        "Tuned Model - Predicted vs Actual",
        fontsize=14,
        fontweight="bold",
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        EVIDENCE_DIR
        / "tuned_predicted_vs_actual.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Predicted vs Actual plot saved:",
        plot_path,
    )


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

def create_feature_importance_plot(
    tuned_model,
):

    print("\n" + "=" * 70)
    print(
        "CREATING TUNED MODEL FEATURE IMPORTANCE"
    )
    print("=" * 70)

    target_model = (
        tuned_model.regressor_
    )

    xgb_model = (
        target_model
        .named_steps["model"]
    )

    preprocessor = (
        target_model
        .named_steps["preprocessor"]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance_values = (
        xgb_model.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # Save complete feature importance
    importance_csv_path = (
        EVIDENCE_DIR
        / "tuned_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_csv_path,
        index=False,
    )

    # Plot top 20
    top_features = (
        importance_df
        .head(20)
        .sort_values(
            "Importance",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    plt.xlabel(
        "Feature Importance",
        fontsize=11,
    )

    plt.ylabel(
        "Feature",
        fontsize=11,
    )

    plt.title(
        "Tuned XGBoost - Top 20 Feature Importance",
        fontsize=14,
        fontweight="bold",
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    plot_path = (
        EVIDENCE_DIR
        / "tuned_feature_importance.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Feature importance plot saved:",
        plot_path,
    )

    print(
        "Feature importance CSV saved:",
        importance_csv_path,
    )


# ============================================================
# 10. TRAIN VS TEST PERFORMANCE
# ============================================================

def create_train_test_performance_plot(
    train_metrics,
    test_metrics,
):

    print("\n" + "=" * 70)
    print(
        "CREATING TRAIN VS TEST PERFORMANCE PLOT"
    )
    print("=" * 70)

    metrics = [
        "MAE",
        "RMSE",
        "R2",
    ]

    train_values = [
        train_metrics["MAE"],
        train_metrics["RMSE"],
        train_metrics["R2"],
    ]

    test_values = [
        test_metrics["MAE"],
        test_metrics["RMSE"],
        test_metrics["R2"],
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        train_values,
        width,
        label="Train",
    )

    plt.bar(
        x + width / 2,
        test_values,
        width,
        label="Test",
    )

    plt.xticks(
        x,
        metrics,
    )

    plt.xlabel(
        "Evaluation Metric",
        fontsize=11,
    )

    plt.ylabel(
        "Metric Value",
        fontsize=11,
    )

    plt.title(
        "Tuned Model - Train vs Test Performance",
        fontsize=14,
        fontweight="bold",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        EVIDENCE_DIR
        / "tuned_train_vs_test_performance.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    performance_df = pd.DataFrame(
        {
            "Metric": metrics,
            "Train": train_values,
            "Test": test_values,
        }
    )

    csv_path = (
        EVIDENCE_DIR
        / "tuned_train_test_performance.csv"
    )

    performance_df.to_csv(
        csv_path,
        index=False,
    )

    print(
        "Train vs Test plot saved:",
        plot_path,
    )

    print(
        "Train vs Test CSV saved:",
        csv_path,
    )


# ============================================================
# 11. SELECTED MODEL VS TUNED MODEL METRICS
# ============================================================

def create_selected_vs_tuned_metrics_plot(
    selected_metrics,
    tuned_metrics,
):

    print("\n" + "=" * 70)
    print(
        "CREATING SELECTED VS TUNED "
        "MODEL METRICS PLOT"
    )
    print("=" * 70)

    metrics = [
        "MAE",
        "RMSE",
        "R2",
    ]

    selected_values = [
        selected_metrics["MAE"],
        selected_metrics["RMSE"],
        selected_metrics["R2"],
    ]

    tuned_values = [
        tuned_metrics["MAE"],
        tuned_metrics["RMSE"],
        tuned_metrics["R2"],
    ]

    x = np.arange(  # noqa: F841
        len(metrics)
    )

    width = 0.35  # noqa: F841

    # --------------------------------------------------------
    # Create three separate metric panels
    # This avoids putting MAE/RMSE and R2 on the same scale.
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    for i, metric in enumerate(metrics):

        values = [
            selected_values[i],
            tuned_values[i],
        ]

        bars = axes[i].bar(
            [
                "Selected Model",
                "Tuned Model",
            ],
            values,
            width=0.6,
        )

        axes[i].set_title(
            metric,
            fontsize=13,
            fontweight="bold",
        )

        axes[i].set_ylabel(
            "Value",
            fontsize=10,
        )

        axes[i].grid(
            axis="y",
            alpha=0.3,
        )

        axes[i].tick_params(
            axis="x",
            rotation=15,
        )

        # Add value labels
        for bar in bars:

            height = bar.get_height()

            axes[i].text(
                bar.get_x()
                + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "Selected Model vs Tuned Model Evaluation Metrics",
        fontsize=15,
        fontweight="bold",
    )

    plt.tight_layout()

    plot_path = (
        EVIDENCE_DIR
        / "selected_vs_tuned_evaluation_metrics.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Selected vs Tuned metrics plot saved:",
        plot_path,
    )


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # STEP 1: LOAD DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    df = load_and_clean_data(
        DATA_PATH
    )

    print(
        "\nCleaned data shape:",
        df.shape,
    )

    # ========================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    feature_df = create_features(
        df
    )

    feature_df["Date"] = pd.to_datetime(
        feature_df["Date"],
        errors="coerce",
    )

    print(
        "\nFeature dataset shape:",
        feature_df.shape,
    )

    # ========================================================
    # STEP 3: TRAIN / TEST SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 3: TRAIN / TEST SPLIT")
    print("=" * 70)

    train_df = feature_df[
        feature_df["Date"].dt.year < TEST_YEAR
    ].copy()

    test_df = feature_df[
        feature_df["Date"].dt.year == TEST_YEAR
    ].copy()

    print(
        "\nTraining rows:",
        len(train_df),
    )

    print(
        "Test rows:",
        len(test_df),
    )

    print(
        "\nTraining period:",
        train_df["Date"].min(),
        "to",
        train_df["Date"].max(),
    )

    print(
        "Test period:",
        test_df["Date"].min(),
        "to",
        test_df["Date"].max(),
    )

    # ========================================================
    # STEP 4: DEFINE X AND Y
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 4: PREPARING FEATURES")
    print("=" * 70)

    DROP_COLUMNS = [
        "Sales_Volume",
        "Date",
        "Month_Name",
        "Year_Month",
        "Country_ID",
        "Product_Type_ID",
    ]

    X_train = train_df.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
    )

    y_train = train_df[
        TARGET
    ]

    X_test = test_df.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
    )

    y_test = test_df[
        TARGET
    ]

    print(
        "\nX_train shape:",
        X_train.shape,
    )

    print(
        "X_test shape:",
        X_test.shape,
    )

    print(
        "y_train shape:",
        y_train.shape,
    )

    print(
        "y_test shape:",
        y_test.shape,
    )

    # ========================================================
    # STEP 5: IDENTIFY CATEGORICAL / NUMERICAL FEATURES
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 5: IDENTIFYING FEATURE TYPES"
    )
    print("=" * 70)

    categorical_features = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numerical_features = (
        X_train
        .select_dtypes(
            exclude=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    print(
        "\nCategorical features:"
    )

    for feature in categorical_features:
        print(
            "  -",
            feature,
        )

    print(
        "\nNumerical features:"
    )

    for feature in numerical_features:
        print(
            "  -",
            feature,
        )

    # ========================================================
    # STEP 6: PREPROCESSOR
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 6: CREATING PREPROCESSOR")
    print("=" * 70)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    print(
        "\nPreprocessor created successfully."
    )

    # ========================================================
    # STEP 7: SELECTED / BASELINE XGBOOST
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 7: CREATING SELECTED MODEL"
    )
    print("=" * 70)

    baseline_xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    baseline_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                baseline_xgb,
            ),
        ]
    )

    baseline_model = (
        TransformedTargetRegressor(
            regressor=baseline_pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    )

    print(
        "\nSelected model created:"
    )

    print(
        "XGBoost - Log Target"
    )

    # ========================================================
    # STEP 8: WALK-FORWARD CV
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 8: CREATING WALK-FORWARD CV"
    )
    print("=" * 70)

    cv_folds = []

    for validation_year in VALIDATION_YEARS:

        train_indices = (
            train_df[
                train_df["Date"].dt.year
                < validation_year
            ].index
        )

        validation_indices = (
            train_df[
                train_df["Date"].dt.year
                == validation_year
            ].index
        )

        train_positions = (
            train_df.index.get_indexer(
                train_indices
            )
        )

        validation_positions = (
            train_df.index.get_indexer(
                validation_indices
            )
        )

        cv_folds.append(
            (
                train_positions,
                validation_positions,
            )
        )

        print(
            f"\nFold validation year: "
            f"{validation_year}"
        )

        print(
            "Training rows:",
            len(train_positions),
        )

        print(
            "Validation rows:",
            len(validation_positions),
        )

    # ========================================================
    # STEP 9: HYPERPARAMETER SEARCH
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 9: HYPERPARAMETER TUNING"
    )
    print("=" * 70)

    param_distributions = {

        "regressor__model__n_estimators": [
            300,
            500,
            700,
            900,
            1200,
        ],

        "regressor__model__max_depth": [
            3,
            4,
            5,
            6,
            7,
        ],

        "regressor__model__learning_rate": [
            0.01,
            0.02,
            0.03,
            0.05,
            0.07,
            0.1,
        ],

        "regressor__model__subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "regressor__model__colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "regressor__model__min_child_weight": [
            1,
            3,
            5,
            7,
        ],

        "regressor__model__gamma": [
            0,
            0.1,
            0.2,
            0.5,
        ],

        "regressor__model__reg_alpha": [
            0,
            0.01,
            0.1,
            0.5,
        ],

        "regressor__model__reg_lambda": [
            1,
            1.5,
            2,
            3,
            5,
        ],
    }

    random_search = RandomizedSearchCV(
        estimator=baseline_model,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    random_search.fit(
        X_train,
        y_train,
    )

    tuned_model = (
        random_search.best_estimator_
    )

    print(
        "\nHyperparameter tuning completed."
    )

    print(
        "\nBest CV RMSE:",
        -random_search.best_score_,
    )

    print(
        "\nBest parameters:"
    )

    for key, value in (
        random_search.best_params_.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # STEP 10: TRAIN SELECTED MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 10: TRAINING SELECTED MODEL"
    )
    print("=" * 70)

    baseline_model.fit(
        X_train,
        y_train,
    )

    selected_train_pred = (
        baseline_model.predict(
            X_train
        )
    )

    selected_test_pred = (
        baseline_model.predict(
            X_test
        )
    )

    print(
        "\nSelected model trained."
    )

    # ========================================================
    # STEP 11: TUNED MODEL PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 11: GENERATING TUNED MODEL PREDICTIONS"
    )
    print("=" * 70)

    tuned_train_pred = (
        tuned_model.predict(
            X_train
        )
    )

    tuned_test_pred = (
        tuned_model.predict(
            X_test
        )
    )

    print(
        "\nTuned predictions generated."
    )

    # ========================================================
    # STEP 12: CALCULATE METRICS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 12: CALCULATING METRICS"
    )
    print("=" * 70)

    selected_train_metrics = (
        calculate_metrics(
            y_train.values,
            selected_train_pred,
        )
    )

    selected_test_metrics = (
        calculate_metrics(
            y_test.values,
            selected_test_pred,
        )
    )

    tuned_train_metrics = (
        calculate_metrics(
            y_train.values,
            tuned_train_pred,
        )
    )

    tuned_test_metrics = (
        calculate_metrics(
            y_test.values,
            tuned_test_pred,
        )
    )

    # ========================================================
    # STEP 13: PRINT MODEL COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 13: MODEL PERFORMANCE COMPARISON"
    )
    print("=" * 70)

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "XGBoost - Log Target",
                "XGBoost - Log Target - Tuned",
            ],
            "Stage": [
                "Before Tuning",
                "After Tuning",
            ],
            "MAE": [
                selected_test_metrics["MAE"],
                tuned_test_metrics["MAE"],
            ],
            "RMSE": [
                selected_test_metrics["RMSE"],
                tuned_test_metrics["RMSE"],
            ],
            "R2": [
                selected_test_metrics["R2"],
                tuned_test_metrics["R2"],
            ],
            "MAPE": [
                selected_test_metrics["MAPE"],
                tuned_test_metrics["MAPE"],
            ],
            "WAPE": [
                selected_test_metrics["WAPE"],
                tuned_test_metrics["WAPE"],
            ],
        }
    )

    print(
        "\n",
        comparison_df.to_string(
            index=False
        ),
    )

    # ========================================================
    # STEP 14: SAVE PREDICTION EVIDENCE
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 14: SAVING 2024 PREDICTIONS"
    )
    print("=" * 70)

    prediction_df = (
        test_df[
            [
                "Date",
                "Product_Type",
                "Country",
            ]
        ].copy()
    )

    prediction_df[
        "Actual_Sales_Volume"
    ] = y_test.values

    prediction_df[
        "Selected_Model_Predicted"
    ] = selected_test_pred

    prediction_df[
        "Tuned_Model_Predicted"
    ] = tuned_test_pred

    prediction_df.to_csv(
        PREDICTION_EVIDENCE_PATH,
        index=False,
    )

    print(
        "\nPrediction evidence saved:",
        PREDICTION_EVIDENCE_PATH,
    )

    # ========================================================
    # STEP 15: PRODUCT-WISE PLOTS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 15: PRODUCT-WISE ACTUAL VS PREDICTED PLOTS"
    )
    print("=" * 70)

    create_product_prediction_plots(
        prediction_df
    )

    # ========================================================
    # STEP 16: TUNED PREDICTED VS ACTUAL
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 16: TUNED PREDICTED VS ACTUAL"
    )
    print("=" * 70)

    create_predicted_vs_actual_plot(
        prediction_df
    )

    # ========================================================
    # STEP 17: FEATURE IMPORTANCE
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 17: TUNED FEATURE IMPORTANCE"
    )
    print("=" * 70)

    create_feature_importance_plot(
        tuned_model
    )

    # ========================================================
    # STEP 18: TRAIN VS TEST
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 18: TRAIN VS TEST PERFORMANCE"
    )
    print("=" * 70)

    create_train_test_performance_plot(
        tuned_train_metrics,
        tuned_test_metrics,
    )

    # ========================================================
    # STEP 19: SELECTED VS TUNED METRICS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 19: SELECTED VS TUNED METRICS"
    )
    print("=" * 70)

    create_selected_vs_tuned_metrics_plot(
        selected_test_metrics,
        tuned_test_metrics,
    )

    # ========================================================
    # STEP 20: SAVE TUNED MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEP 20: SAVING TUNED MODEL"
    )
    print("=" * 70)

    joblib.dump(
        tuned_model,
        TUNED_MODEL_PATH,
    )

    tuned_model_info = {
        "model_name": (
            "XGBoost - Log Target - Tuned"
        ),
        "selected_model": (
            "XGBoost - Log Target"
        ),
        "estimator": "XGBRegressor",
        "best_cv_rmse": (
            -random_search.best_score_
        ),
        "best_params": (
            random_search.best_params_
        ),
        "test_year": TEST_YEAR,
        "validation_years": VALIDATION_YEARS,
        "target": TARGET,
    }

    joblib.dump(
        tuned_model_info,
        TUNED_MODEL_INFO_PATH,
    )

    print(
        "\nTuned model saved:",
        TUNED_MODEL_PATH,
    )

    print(
        "Tuned model information saved:",
        TUNED_MODEL_INFO_PATH,
    )

    # ========================================================
    # STEP 21: FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 70)

    print(
        "\nSelected Model:"
    )

    print(
        "  MAE :",
        selected_test_metrics["MAE"],
    )

    print(
        "  RMSE:",
        selected_test_metrics["RMSE"],
    )

    print(
        "  R2  :",
        selected_test_metrics["R2"],
    )

    print(
        "\nTuned Model:"
    )

    print(
        "  MAE :",
        tuned_test_metrics["MAE"],
    )

    print(
        "  RMSE:",
        tuned_test_metrics["RMSE"],
    )

    print(
        "  R2  :",
        tuned_test_metrics["R2"],
    )

    print(
        "\nAll evidence generated successfully."
    )

    print(
        "\nEvidence directory:"
    )

    print(
        EVIDENCE_DIR
    )

    print(
        "\nProduct plot directory:"
    )

    print(
        PRODUCT_PLOT_DIR
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "TUNING AND ANALYSIS COMPLETED"
    )
    print(
        "=" * 70
    )