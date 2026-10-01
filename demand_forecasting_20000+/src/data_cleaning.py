import pandas as pd


def load_and_clean_data(file_path):
    """
    Load the sales demand dataset and perform basic cleaning.
    """

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    df = pd.read_csv(file_path)

    print(f"Original shape: {df.shape}")

    # --------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------
    required_columns = [
        "Date",
        "Region",
        "Country",
        "Product_Type",
        "Sales_Volume"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # 3. Convert Date column
    # --------------------------------------------------
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Remove rows where Date could not be converted
    df = df.dropna(subset=["Date"])

    # --------------------------------------------------
    # 4. Standardize Region
    # --------------------------------------------------
    df["Region"] = (
        df["Region"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------
    # 5. Standardize Country
    # --------------------------------------------------
    df["Country"] = (
        df["Country"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------
    # 6. Standardize Product_Type
    # --------------------------------------------------
    df["Product_Type"] = (
        df["Product_Type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------
    # 7. Convert Sales_Volume to numeric
    # --------------------------------------------------
    df["Sales_Volume"] = pd.to_numeric(
        df["Sales_Volume"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 8. Remove duplicate observations
    #
    # One observation represents:
    # Date + Country + Product_Type
    # --------------------------------------------------
    df = df.drop_duplicates(
        subset=[
            "Date",
            "Country",
            "Product_Type"
        ]
    )

    # --------------------------------------------------
    # 9. Fill missing Sales_Volume
    #    using Country + Product_Type median
    # --------------------------------------------------
    df["Sales_Volume"] = (
        df.groupby(
            ["Country", "Product_Type"]
        )["Sales_Volume"]
        .transform(
            lambda x: x.fillna(x.median())
        )
    )

    # --------------------------------------------------
    # 10. Remove remaining missing target values
    # --------------------------------------------------
    df = df.dropna(
        subset=["Sales_Volume"]
    )

    # --------------------------------------------------
    # 11. Prevent negative sales values
    # --------------------------------------------------
    df["Sales_Volume"] = df[
        "Sales_Volume"
    ].clip(lower=0)

    # --------------------------------------------------
    # 12. Create basic date features
    # --------------------------------------------------
    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Month_Name"] = (
        df["Date"]
        .dt.month_name()
    )

    df["Quarter"] = (
        "Q"
        + df["Date"]
        .dt.quarter
        .astype(str)
    )

    df["Year_Month"] = (
        df["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    # --------------------------------------------------
    # 13. Sort data
    #
    # Forecasting history is maintained separately
    # for every Country + Product_Type combination.
    # --------------------------------------------------
    df = df.sort_values(
        [
            "Country",
            "Product_Type",
            "Date"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 14. Print summary
    # --------------------------------------------------
    print(f"Cleaned shape: {df.shape}")

    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Number of regions: "
        f"{df['Region'].nunique()}"
    )

    print(
        f"Number of countries: "
        f"{df['Country'].nunique()}"
    )

    print(
        f"Number of product types: "
        f"{df['Product_Type'].nunique()}"
    )

    print("\nRegions:")
    print(
        sorted(
            df["Region"].unique()
        )
    )

    print("\nCountries:")
    print(
        sorted(
            df["Country"].unique()
        )
    )

    print("\nProduct Types:")
    print(
        sorted(
            df["Product_Type"].unique()
        )
    )

    # --------------------------------------------------
    # 15. Check Country → Region consistency
    # --------------------------------------------------
    country_region_counts = (
        df.groupby("Country")["Region"]
        .nunique()
    )

    inconsistent_countries = (
        country_region_counts[
            country_region_counts > 1
        ]
    )

    if not inconsistent_countries.empty:
        print("\nWARNING:")
        print(
            "Some countries belong to multiple regions:"
        )
        print(
            inconsistent_countries
        )
    else:
        print(
            "\nCountry → Region mapping is consistent."
        )

    # --------------------------------------------------
    # 16. Check duplicate forecasting observations
    # --------------------------------------------------
    duplicate_count = df.duplicated(
        subset=[
            "Date",
            "Country",
            "Product_Type"
        ]
    ).sum()

    print(
        "\nDuplicate "
        "Date + Country + Product_Type rows:",
        duplicate_count
    )

    # --------------------------------------------------
    # 17. Missing values
    # --------------------------------------------------
    print("\nMissing values:")
    print(df.isnull().sum())

    print("=" * 70)

    return df