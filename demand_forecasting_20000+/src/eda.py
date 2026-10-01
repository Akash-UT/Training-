import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def run_eda(df, output_path):
    """
    Generate and save EDA plots for the apparel demand dataset.

    Dataset dimensions:
        Date
        Region
        Country
        Product_Type
        Sales_Volume
    """

    os.makedirs(output_path, exist_ok=True)

    print("\nGenerating EDA plots...")

    # ==================================================
    # 1. TOTAL DEMAND OVER TIME
    # ==================================================

    monthly_demand = (
        df.groupby("Date")["Sales_Volume"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_demand["Date"],
        monthly_demand["Sales_Volume"],
        linewidth=2
    )

    plt.title("Total Sales Demand Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales Volume")

    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "01_total_demand_over_time.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 2. PRODUCT TYPE TRENDS
    # ==================================================

    product_trends = (
        df.groupby(
            ["Date", "Product_Type"]
        )["Sales_Volume"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 7))

    for product_type in sorted(
        product_trends["Product_Type"].unique()
    ):

        product_data = product_trends[
            product_trends["Product_Type"] == product_type
        ]

        plt.plot(
            product_data["Date"],
            product_data["Sales_Volume"],
            label=product_type,
            linewidth=1.5
        )

    plt.title("Sales Demand by Product Type")
    plt.xlabel("Date")
    plt.ylabel("Sales Volume")

    plt.legend(
        title="Product Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "02_product_type_trends.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 3. PRODUCT TYPE DISTRIBUTION
    # ==================================================

    product_distribution = (
        df.groupby("Product_Type")["Sales_Volume"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))

    product_distribution.plot(
        kind="bar"
    )

    plt.title("Total Sales Volume by Product Type")
    plt.xlabel("Product Type")
    plt.ylabel("Total Sales Volume")

    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "03_product_type_distribution.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 4. MONTHLY SEASONALITY
    # ==================================================

    monthly_seasonality = (
        df.groupby("Month")["Sales_Volume"]
        .mean()
        .reset_index()
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_seasonality["Month"],
        monthly_seasonality["Sales_Volume"],
        marker="o",
        linewidth=2
    )

    plt.title("Average Monthly Sales Seasonality")
    plt.xlabel("Month")
    plt.ylabel("Average Sales Volume")

    plt.xticks(
        range(1, 13),
        month_names
    )

    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "04_monthly_seasonality.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 5. REGION DISTRIBUTION
    # ==================================================

    region_distribution = (
        df.groupby("Region")["Sales_Volume"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))

    region_distribution.plot(
        kind="bar"
    )

    plt.title("Total Sales Volume by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Sales Volume")

    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "05_region_distribution.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 6. COUNTRY DISTRIBUTION
    # ==================================================

    country_distribution = (
        df.groupby("Country")["Sales_Volume"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(14, 7))

    country_distribution.plot(
        kind="bar"
    )

    plt.title("Total Sales Volume by Country")
    plt.xlabel("Country")
    plt.ylabel("Total Sales Volume")

    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "06_country_distribution.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 7. REGION TRENDS
    # ==================================================

    region_trends = (
        df.groupby(
            ["Date", "Region"]
        )["Sales_Volume"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 7))

    for region in sorted(
        region_trends["Region"].unique()
    ):

        region_data = region_trends[
            region_trends["Region"] == region
        ]

        plt.plot(
            region_data["Date"],
            region_data["Sales_Volume"],
            label=region,
            linewidth=1.5
        )

    plt.title("Sales Demand by Region Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales Volume")

    plt.legend(
        title="Region",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "07_region_trends.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 8. COUNTRY TRENDS
    # ==================================================

    country_trends = (
        df.groupby(
            ["Date", "Country"]
        )["Sales_Volume"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(15, 8))

    for country in sorted(
        country_trends["Country"].unique()
    ):

        country_data = country_trends[
            country_trends["Country"] == country
        ]

        plt.plot(
            country_data["Date"],
            country_data["Sales_Volume"],
            label=country,
            linewidth=1.2
        )

    plt.title("Sales Demand by Country Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales Volume")

    plt.legend(
        title="Country",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "08_country_trends.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 9. REGION × PRODUCT HEATMAP
    # ==================================================

    region_product = (
        df.groupby(
            ["Region", "Product_Type"]
        )["Sales_Volume"]
        .sum()
        .unstack()
    )

    plt.figure(figsize=(14, 7))

    plt.imshow(
        region_product,
        aspect="auto"
    )

    plt.colorbar(
        label="Total Sales Volume"
    )

    plt.title(
        "Sales Demand by Region and Product Type"
    )

    plt.xlabel("Product Type")
    plt.ylabel("Region")

    plt.xticks(
        range(len(region_product.columns)),
        region_product.columns,
        rotation=45
    )

    plt.yticks(
        range(len(region_product.index)),
        region_product.index
    )

    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "09_region_product_heatmap.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 10. COUNTRY × PRODUCT HEATMAP
    # ==================================================

    country_product = (
        df.groupby(
            ["Country", "Product_Type"]
        )["Sales_Volume"]
        .sum()
        .unstack()
    )

    plt.figure(figsize=(14, 9))

    plt.imshow(
        country_product,
        aspect="auto"
    )

    plt.colorbar(
        label="Total Sales Volume"
    )

    plt.title(
        "Sales Demand by Country and Product Type"
    )

    plt.xlabel("Product Type")
    plt.ylabel("Country")

    plt.xticks(
        range(len(country_product.columns)),
        country_product.columns,
        rotation=45
    )

    plt.yticks(
        range(len(country_product.index)),
        country_product.index
    )

    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "10_country_product_heatmap.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 11. YEARLY DEMAND
    # ==================================================

    yearly_demand = (
        df.groupby("Year")["Sales_Volume"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        yearly_demand["Year"],
        yearly_demand["Sales_Volume"],
        marker="o",
        linewidth=2
    )

    plt.title("Total Yearly Sales Demand")
    plt.xlabel("Year")
    plt.ylabel("Sales Volume")

    plt.xticks(
        yearly_demand["Year"]
    )

    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "11_yearly_demand.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_path}")

    # ==================================================
    # 12. SEASON × PRODUCT HEATMAP
    # ==================================================

    season_product = (
        df.groupby(
            ["Season", "Product_Type"]
        )["Sales_Volume"]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(14, 6))

    plt.imshow(
        season_product,
        aspect="auto"
    )

    plt.colorbar(
        label="Average Sales Volume"
    )

    plt.title(
        "Average Sales Demand by Season and Product Type"
    )

    plt.xlabel("Product Type")
    plt.ylabel("Season")

    plt.xticks(
        range(len(season_product.columns)),
        season_product.columns,
        rotation=45
    )

    plt.yticks(
        range(len(season_product.index)),
        season_product.index
    )

    plt.tight_layout()

    file_path = os.path.join(
        output_path,
        "12_season_product_heatmap.png"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==================================================
    # COMPLETED
    # ==================================================

    print("\nEDA generation completed successfully.")
    print("All EDA plots are saved in:")
    print(output_path)