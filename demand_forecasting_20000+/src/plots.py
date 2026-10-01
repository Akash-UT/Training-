import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. MODEL COMPARISON - RMSE
# ============================================================

def plot_model_comparison_rmse(
    results,
    output_path
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        results["Model"],
        results["RMSE"]
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Model Comparison - RMSE")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 2. MODEL COMPARISON - MAE
# ============================================================

def plot_model_comparison_mae(
    results,
    output_path
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        results["Model"],
        results["MAE"]
    )

    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("Model Comparison - MAE")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 3. MODEL COMPARISON - R2
# ============================================================

def plot_model_comparison_r2(
    results,
    output_path
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        results["Model"],
        results["R2"]
    )

    plt.xlabel("Model")
    plt.ylabel("R²")
    plt.title("Model Comparison - R²")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 4. TRAIN VS TEST RMSE
# ============================================================

def plot_train_test_rmse(
    results,
    output_path
):

    x = np.arange(
        len(results)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        results["Train_RMSE"],
        width,
        label="Train RMSE"
    )

    plt.bar(
        x + width / 2,
        results["Test_RMSE"],
        width,
        label="Test RMSE"
    )

    plt.xticks(
        x,
        results["Model"],
        rotation=30,
        ha="right"
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Train vs Test RMSE")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 5. TRAIN VS TEST MAE
# ============================================================

def plot_train_test_mae(
    results,
    output_path
):

    x = np.arange(
        len(results)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        results["Train_MAE"],
        width,
        label="Train MAE"
    )

    plt.bar(
        x + width / 2,
        results["Test_MAE"],
        width,
        label="Test MAE"
    )

    plt.xticks(
        x,
        results["Model"],
        rotation=30,
        ha="right"
    )

    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("Train vs Test MAE")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 6. PRODUCT TYPE ACTUAL VS PREDICTED
# ============================================================

def plot_product_type_actual_vs_predicted(
    dates,
    y_true,
    y_pred,
    product_type,
    model_name,
    output_path
):

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        dates,
        y_true,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        dates,
        y_pred,
        label="Predicted",
        linewidth=2
    )

    plt.xlabel("Date")
    plt.ylabel("Sales Volume")

    plt.title(
        f"{product_type} - Actual vs Predicted\n"
        f"{model_name}"
    )

    plt.legend()

    plt.xticks(
        rotation=45
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. COUNTRY ACTUAL VS PREDICTED
# ============================================================

def plot_country_actual_vs_predicted(
    dates,
    y_true,
    y_pred,
    country,
    model_name,
    output_path
):

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        dates,
        y_true,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        dates,
        y_pred,
        label="Predicted",
        linewidth=2
    )

    plt.xlabel("Date")
    plt.ylabel("Sales Volume")

    plt.title(
        f"{country} - Actual vs Predicted\n"
        f"{model_name}"
    )

    plt.legend()

    plt.xticks(
        rotation=45
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. COUNTRY × PRODUCT MAE HEATMAP
# ============================================================

def plot_country_product_mae_heatmap(
    results,
    output_path
):

    pivot_data = results.pivot(
        index="Country",
        columns="Product_Type",
        values="MAE"
    )

    plt.figure(
        figsize=(14, 9)
    )

    plt.imshow(
        pivot_data,
        aspect="auto"
    )

    plt.colorbar(
        label="MAE"
    )

    plt.title(
        "Country × Product - MAE"
    )

    plt.xlabel(
        "Product Type"
    )

    plt.ylabel(
        "Country"
    )

    plt.xticks(
        range(len(pivot_data.columns)),
        pivot_data.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(pivot_data.index)),
        pivot_data.index
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 9. COUNTRY × PRODUCT WAPE HEATMAP
# ============================================================

def plot_country_product_wape_heatmap(
    results,
    output_path
):

    pivot_data = results.pivot(
        index="Country",
        columns="Product_Type",
        values="WAPE"
    )

    plt.figure(
        figsize=(14, 9)
    )

    plt.imshow(
        pivot_data,
        aspect="auto"
    )

    plt.colorbar(
        label="WAPE (%)"
    )

    plt.title(
        "Country × Product - WAPE"
    )

    plt.xlabel(
        "Product Type"
    )

    plt.ylabel(
        "Country"
    )

    plt.xticks(
        range(len(pivot_data.columns)),
        pivot_data.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(pivot_data.index)),
        pivot_data.index
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    feature_names,
    importances,
    model_name,
    output_path,
    top_n=15
):

    importance_data = list(
        zip(
            feature_names,
            importances
        )
    )

    importance_data.sort(
        key=lambda x: x[1],
        reverse=True
    )

    importance_data = (
        importance_data[:top_n]
    )

    names = [
        item[0]
        for item in importance_data
    ]

    values = [
        item[1]
        for item in importance_data
    ]

    names = names[::-1]
    values = values[::-1]

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        names,
        values
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Top {top_n} Feature Importance - "
        f"{model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. ACTUAL VS PREDICTED SCATTER
# ============================================================

def plot_actual_vs_predicted_scatter(
    y_true,
    y_pred,
    model_name,
    output_path
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    min_value = min(
        y_true.min(),
        y_pred.min()
    )

    max_value = max(
        y_true.max(),
        y_pred.max()
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(
        "Actual Sales Volume"
    )

    plt.ylabel(
        "Predicted Sales Volume"
    )

    plt.title(
        f"Actual vs Predicted\n{model_name}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. PREDICTION ERROR DISTRIBUTION
# ============================================================

def plot_error_distribution(
    y_true,
    y_pred,
    model_name,
    output_path
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    errors = (
        y_true - y_pred
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        errors,
        bins=40,
        alpha=0.75
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(
        "Prediction Error "
        "(Actual - Predicted)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"Prediction Error Distribution\n"
        f"{model_name}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()