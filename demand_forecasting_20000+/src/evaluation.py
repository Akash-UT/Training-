import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# MAPE
# =========================================================

def calculate_mape(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    # Avoid division by zero
    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    -
                    y_pred[mask]
                )
                /
                y_true[mask]
            )
        )
        * 100
    )


# =========================================================
# WAPE
# =========================================================

def calculate_wape(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    denominator = np.sum(
        np.abs(y_true)
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(
                y_true - y_pred
            )
        )
        /
        denominator
    ) * 100


# =========================================================
# OVERALL MODEL METRICS
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
    model_name
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = calculate_mape(
        y_true,
        y_pred
    )

    wape = calculate_wape(
        y_true,
        y_pred
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "WAPE": wape
    }


# =========================================================
# GENERALIZATION / OVERFITTING CHECK
# =========================================================

def calculate_generalization(
    y_train,
    train_pred,
    y_test,
    test_pred,
    model_name
):

    # -----------------------------------------------------
    # TRAIN METRICS
    # -----------------------------------------------------

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )

    train_r2 = r2_score(
        y_train,
        train_pred
    )

    # -----------------------------------------------------
    # TEST METRICS
    # -----------------------------------------------------

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )

    test_r2 = r2_score(
        y_test,
        test_pred
    )

    # -----------------------------------------------------
    # GENERALIZATION GAP
    # -----------------------------------------------------

    r2_gap = (
        train_r2
        -
        test_r2
    )

    # -----------------------------------------------------
    # ERROR RATIOS
    # -----------------------------------------------------

    mae_ratio = (
        test_mae / train_mae
        if train_mae != 0
        else np.inf
    )

    rmse_ratio = (
        test_rmse / train_rmse
        if train_rmse != 0
        else np.inf
    )

    return {
        "Model": model_name,

        "Train_MAE": train_mae,
        "Test_MAE": test_mae,

        "Train_RMSE": train_rmse,
        "Test_RMSE": test_rmse,

        "Train_R2": train_r2,
        "Test_R2": test_r2,

        "MAE_Ratio": mae_ratio,
        "RMSE_Ratio": rmse_ratio,

        "R2_Gap": r2_gap
    }


# =========================================================
# PRODUCT-LEVEL EVALUATION
# =========================================================

def calculate_product_metrics(
    test_df,
    y_pred
):

    results = test_df[
        [
            "Product_Type",
            "Sales_Volume"
        ]
    ].copy()

    results["Predicted_Sales"] = y_pred

    product_results = []

    for product_type, group in results.groupby(
        "Product_Type"
    ):

        y_true = group[
            "Sales_Volume"
        ]

        predictions = group[
            "Predicted_Sales"
        ]

        product_results.append({

            "Product_Type": product_type,

            "MAE": mean_absolute_error(
                y_true,
                predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions
                )
            ),

            "R2": r2_score(
                y_true,
                predictions
            ),

            "MAPE": calculate_mape(
                y_true,
                predictions
            ),

            "WAPE": calculate_wape(
                y_true,
                predictions
            ),

            "Actual_Total": y_true.sum(),

            "Predicted_Total": predictions.sum()
        })

    return pd.DataFrame(
        product_results
    )


# =========================================================
# COUNTRY-LEVEL EVALUATION
# =========================================================

def calculate_country_metrics(
    test_df,
    y_pred
):

    results = test_df[
        [
            "Region",
            "Country",
            "Sales_Volume"
        ]
    ].copy()

    results["Predicted_Sales"] = y_pred

    country_results = []

    for (
        region,
        country
    ), group in results.groupby(
        [
            "Region",
            "Country"
        ]
    ):

        y_true = group[
            "Sales_Volume"
        ]

        predictions = group[
            "Predicted_Sales"
        ]

        country_results.append({

            "Region": region,

            "Country": country,

            "MAE": mean_absolute_error(
                y_true,
                predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions
                )
            ),

            "R2": r2_score(
                y_true,
                predictions
            ),

            "MAPE": calculate_mape(
                y_true,
                predictions
            ),

            "WAPE": calculate_wape(
                y_true,
                predictions
            ),

            "Actual_Total": y_true.sum(),

            "Predicted_Total": predictions.sum()
        })

    return pd.DataFrame(
        country_results
    )


# =========================================================
# REGION-LEVEL EVALUATION
# =========================================================

def calculate_region_metrics(
    test_df,
    y_pred
):

    results = test_df[
        [
            "Region",
            "Sales_Volume"
        ]
    ].copy()

    results["Predicted_Sales"] = y_pred

    region_results = []

    for region, group in results.groupby(
        "Region"
    ):

        y_true = group[
            "Sales_Volume"
        ]

        predictions = group[
            "Predicted_Sales"
        ]

        region_results.append({

            "Region": region,

            "MAE": mean_absolute_error(
                y_true,
                predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions
                )
            ),

            "R2": r2_score(
                y_true,
                predictions
            ),

            "MAPE": calculate_mape(
                y_true,
                predictions
            ),

            "WAPE": calculate_wape(
                y_true,
                predictions
            ),

            "Actual_Total": y_true.sum(),

            "Predicted_Total": predictions.sum()
        })

    return pd.DataFrame(
        region_results
    )


# =========================================================
# COUNTRY + PRODUCT EVALUATION
# =========================================================

def calculate_country_product_metrics(
    test_df,
    y_pred
):

    results = test_df[
        [
            "Region",
            "Country",
            "Product_Type",
            "Sales_Volume"
        ]
    ].copy()

    results["Predicted_Sales"] = y_pred

    series_results = []

    for (
        region,
        country,
        product_type
    ), group in results.groupby(
        [
            "Region",
            "Country",
            "Product_Type"
        ]
    ):

        y_true = group[
            "Sales_Volume"
        ]

        predictions = group[
            "Predicted_Sales"
        ]

        series_results.append({

            "Region": region,

            "Country": country,

            "Product_Type": product_type,

            "MAE": mean_absolute_error(
                y_true,
                predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions
                )
            ),

            "R2": r2_score(
                y_true,
                predictions
            ),

            "MAPE": calculate_mape(
                y_true,
                predictions
            ),

            "WAPE": calculate_wape(
                y_true,
                predictions
            ),

            "Actual_Total": y_true.sum(),

            "Predicted_Total": predictions.sum()
        })

    return pd.DataFrame(
        series_results
    )


# =========================================================
# MODEL SELECTION
# =========================================================

def select_best_model(
    predictive_results,
    generalization_results,
    predictive_weight=0.60,
    generalization_weight=0.40
):

    predictive = (
        predictive_results.copy()
    )

    generalization = (
        generalization_results.copy()
    )

    # =====================================================
    # PREDICTIVE RANKS
    # =====================================================

    predictive["MAE_Rank"] = (
        predictive["MAE"]
        .rank(
            method="min",
            ascending=True
        )
    )

    predictive["RMSE_Rank"] = (
        predictive["RMSE"]
        .rank(
            method="min",
            ascending=True
        )
    )

    predictive["R2_Rank"] = (
        predictive["R2"]
        .rank(
            method="min",
            ascending=False
        )
    )

    predictive["MAPE_Rank"] = (
        predictive["MAPE"]
        .rank(
            method="min",
            ascending=True
        )
    )

    predictive["WAPE_Rank"] = (
        predictive["WAPE"]
        .rank(
            method="min",
            ascending=True
        )
    )

    # -----------------------------------------------------
    # AVERAGE PREDICTIVE SCORE
    # -----------------------------------------------------

    predictive["Predictive_Score"] = (
        predictive[
            [
                "MAE_Rank",
                "RMSE_Rank",
                "R2_Rank",
                "MAPE_Rank",
                "WAPE_Rank"
            ]
        ]
        .mean(axis=1)
    )

    # =====================================================
    # GENERALIZATION RANKS
    # =====================================================

    generalization[
        "R2_Gap_Rank"
    ] = (
        generalization["R2_Gap"]
        .rank(
            method="min",
            ascending=True
        )
    )

    generalization[
        "MAE_Ratio_Rank"
    ] = (
        generalization["MAE_Ratio"]
        .rank(
            method="min",
            ascending=True
        )
    )

    generalization[
        "RMSE_Ratio_Rank"
    ] = (
        generalization["RMSE_Ratio"]
        .rank(
            method="min",
            ascending=True
        )
    )

    # -----------------------------------------------------
    # AVERAGE GENERALIZATION SCORE
    # -----------------------------------------------------

    generalization[
        "Generalization_Score"
    ] = (
        generalization[
            [
                "R2_Gap_Rank",
                "MAE_Ratio_Rank",
                "RMSE_Ratio_Rank"
            ]
        ]
        .mean(axis=1)
    )

    # =====================================================
    # COMBINE SCORES
    # =====================================================

    selection = predictive[
        [
            "Model",
            "Predictive_Score"
        ]
    ].merge(
        generalization[
            [
                "Model",
                "Generalization_Score"
            ]
        ],
        on="Model",
        how="inner"
    )

    selection["Final_Score"] = (
        predictive_weight
        *
        selection["Predictive_Score"]
        +
        generalization_weight
        *
        selection["Generalization_Score"]
    )

    selection = (
        selection
        .sort_values(
            "Final_Score",
            ascending=True
        )
        .reset_index(drop=True)
    )

    selected_model = (
        selection.iloc[0]["Model"]
    )

    return (
        selected_model,
        selection
    )