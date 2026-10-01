import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# CREATE FEATURES
# =========================================================

def create_features(df):

    df = df.copy()

    # =====================================================
    # SORT BY FORECASTING SERIES
    # =====================================================

    # Each forecasting series is:
    # Country + Product_Type

    df = df.sort_values(
        [
            "Country",
            "Product_Type",
            "Date"
        ]
    ).reset_index(drop=True)


    # =====================================================
    # TIME FEATURES
    # =====================================================

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Quarter"] = df["Date"].dt.quarter


    # =====================================================
    # CYCLICAL MONTH FEATURES
    # =====================================================

    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )


    # =====================================================
    # TIME INDEX
    # =====================================================

    df["Time_Index"] = (
        df["Date"] - df["Date"].min()
    ).dt.days


    # =====================================================
    # LAG FEATURES
    # =====================================================

    grouping_columns = [
        "Country",
        "Product_Type"
    ]

    for lag in [1, 2, 3, 6, 12]:

        df[f"Lag_{lag}"] = (
            df.groupby(grouping_columns)["Sales_Volume"]
            .shift(lag)
        )


    # =====================================================
    # ROLLING MEAN FEATURES
    # =====================================================

    df["Rolling_Mean_3"] = (
        df.groupby(grouping_columns)["Sales_Volume"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    df["Rolling_Mean_6"] = (
        df.groupby(grouping_columns)["Sales_Volume"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(6)
            .mean()
        )
    )

    df["Rolling_Mean_12"] = (
        df.groupby(grouping_columns)["Sales_Volume"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(12)
            .mean()
        )
    )


    # =====================================================
    # ROLLING STANDARD DEVIATION
    # =====================================================

    df["Rolling_Std_3"] = (
        df.groupby(grouping_columns)["Sales_Volume"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .std()
        )
    )

    df["Rolling_Std_6"] = (
        df.groupby(grouping_columns)["Sales_Volume"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(6)
            .std()
        )
    )


    # =====================================================
    # YEAR-OVER-YEAR CHANGE
    # =====================================================

    df["YoY_Change"] = (
        (
            df["Lag_1"] - df["Lag_12"]
        )
        /
        df["Lag_12"].replace(0, np.nan)
    )


    # =====================================================
    # REMOVE ROWS WITHOUT ENOUGH HISTORY
    # =====================================================

    df = df.dropna().reset_index(drop=True)


    # =====================================================
    # VALIDATION
    # =====================================================

    required_columns = [
        "Region",
        "Country",
        "Product_Type",
        "Date",
        "Sales_Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing after "
            f"feature engineering: {missing_columns}"
        )


    # =====================================================
    # FEATURE SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print(
        f"Feature-engineered shape: {df.shape}"
    )

    print("\nCreated features:")

    feature_columns = [
        "Year",
        "Month",
        "Quarter",
        "Month_Sin",
        "Month_Cos",
        "Time_Index",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_6",
        "Lag_12",
        "Rolling_Mean_3",
        "Rolling_Mean_6",
        "Rolling_Mean_12",
        "Rolling_Std_3",
        "Rolling_Std_6",
        "YoY_Change"
    ]

    for column in feature_columns:
        print(f"  - {column}")

    print("=" * 70)

    return df


# =========================================================
# PREPARE MODEL DATA
# =========================================================

def prepare_model_data(df_features):

    df = df_features.copy()


    # =====================================================
    # CHRONOLOGICAL TRAIN / TEST SPLIT
    # =====================================================

    # Last 12 months = test set.
    #
    # Train:
    # 2016-01 -> 2023-12
    #
    # Test:
    # 2024-01 -> 2024-12

    test_start_date = (
        df["Date"].max()
        -
        pd.DateOffset(months=11)
    )

    train_df = df[
        df["Date"] < test_start_date
    ].copy()

    test_df = df[
        df["Date"] >= test_start_date
    ].copy()


    # =====================================================
    # TARGET
    # =====================================================

    target = "Sales_Volume"


    # =====================================================
    # REMOVE NON-MODEL COLUMNS
    # =====================================================

    columns_to_drop = [
        target,
        "Date",
        "Month_Name",
        "Year_Month",
        "Country_ID",
        "Product_Type_ID"
    ]

    existing_drop_columns = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    X_train = train_df.drop(
        columns=existing_drop_columns
    )

    X_test = test_df.drop(
        columns=existing_drop_columns
    )

    y_train = train_df[target].copy()
    y_test = test_df[target].copy()


    # =====================================================
    # IDENTIFY CATEGORICAL FEATURES
    # =====================================================

    categorical_columns = (
        X_train.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numerical_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]


    # =====================================================
    # DISPLAY FEATURES
    # =====================================================

    print("\n" + "=" * 70)
    print("MODEL FEATURES")
    print("=" * 70)

    print("\nCategorical features:")

    for column in categorical_columns:
        print(f"  - {column}")

    print("\nNumerical features:")

    for column in numerical_columns:
        print(f"  - {column}")

    print(
        f"\nNumber of categorical columns: "
        f"{len(categorical_columns)}"
    )

    print(
        f"Number of numerical columns: "
        f"{len(numerical_columns)}"
    )

    print(
        f"\nTraining rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    print(
        f"\nTraining period: "
        f"{train_df['Date'].min().date()} "
        f"to "
        f"{train_df['Date'].max().date()}"
    )

    print(
        f"Testing period: "
        f"{test_df['Date'].min().date()} "
        f"to "
        f"{test_df['Date'].max().date()}"
    )


    # =====================================================
    # PREPROCESSOR
    # =====================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )


    # =====================================================
    # FIT ONLY ON TRAINING DATA
    # =====================================================

    X_train_encoded = (
        preprocessor.fit_transform(
            X_train
        )
    )


    # =====================================================
    # TRANSFORM TEST DATA
    # =====================================================

    X_test_encoded = (
        preprocessor.transform(
            X_test
        )
    )


    # =====================================================
    # FEATURE NAMES
    # =====================================================

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )


    # =====================================================
    # CONVERT TO DATAFRAME
    # =====================================================

    X_train_encoded = pd.DataFrame(
        X_train_encoded,
        columns=feature_names,
        index=X_train.index
    )

    X_test_encoded = pd.DataFrame(
        X_test_encoded,
        columns=feature_names,
        index=X_test.index
    )


    # =====================================================
    # FINAL FEATURE INFORMATION
    # =====================================================

    print(
        f"\nTotal encoded features: "
        f"{len(feature_names)}"
    )

    print("=" * 70)


    return (
        X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
        preprocessor,
        feature_names,
        train_df,
        test_df
    )
