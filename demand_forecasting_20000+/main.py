from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_cleaning import load_and_clean_data

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "sales_demand.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "future_forecasting"

TUNED_MODEL_PATH = MODEL_DIR / "tuned_model.pkl"
TUNED_INFO_PATH = MODEL_DIR / "tuned_model_info.pkl"


# ============================================================
# FORECAST SETTINGS
# ============================================================

FORECAST_START = pd.Timestamp("2025-01-01")
FORECAST_MONTHS = 12


# ============================================================
# COLUMNS DROPPED BEFORE MODEL PREDICTION
# ============================================================

DROP_COLUMNS = [
    "Sales_Volume",
    "Date",
    "Month_Name",
    "Year_Month",
    "Country_ID",
    "Product_Type_ID",
]


# ============================================================
# SEASON FUNCTION
# ============================================================

def get_season(month):
    """
    Convert month number into season.
    """

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    elif month in [9, 10, 11]:
        return "Autumn"

    return "Unknown"


# ============================================================
# GET COUNTRY ID
# ============================================================

def get_country_id(history_df, country):
    """
    Get Country_ID for a country from historical data.
    """

    rows = history_df.loc[
        history_df["Country"] == country,
        "Country_ID"
    ]

    if len(rows) > 0:
        return rows.iloc[0]

    return 0


# ============================================================
# GET PRODUCT TYPE ID
# ============================================================

def get_product_type_id(history_df, product_type):
    """
    Get Product_Type_ID for a product type.
    """

    rows = history_df.loc[
        history_df["Product_Type"] == product_type,
        "Product_Type_ID"
    ]

    if len(rows) > 0:
        return rows.iloc[0]

    return 0


# ============================================================
# CREATE FUTURE FEATURES
# ============================================================

def create_future_features(
    history_df,
    future_date,
    country,
    product_type
):
    """
    Create one future row containing the same features
    used during model training.
    """

    series = history_df[
        (history_df["Country"] == country)
        & (history_df["Product_Type"] == product_type)
    ].sort_values("Date")

    sales = series["Sales_Volume"].astype(float).tolist()

    if len(sales) < 12:
        raise ValueError(
            f"Not enough historical data for "
            f"{country} - {product_type}"
        )

    # --------------------------------------------------------
    # Basic date features
    # --------------------------------------------------------

    year = future_date.year
    month = future_date.month

    month_name = future_date.strftime("%B")

    # IMPORTANT:
    # Quarter is numeric because the training data uses
    # numeric quarter values.
    quarter = future_date.quarter

    year_month = future_date.strftime("%Y-%m")

    season = get_season(month)

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Number of months from the first historical month
    first_date = pd.to_datetime(history_df["Date"].min())

    time_index = (
        (future_date.year - first_date.year) * 12
        + (future_date.month - first_date.month)
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    lag_1 = sales[-1]
    lag_2 = sales[-2]
    lag_3 = sales[-3]
    lag_6 = sales[-6]
    lag_12 = sales[-12]

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    rolling_mean_3 = np.mean(sales[-3:])
    rolling_mean_6 = np.mean(sales[-6:])
    rolling_mean_12 = np.mean(sales[-12:])

    rolling_std_3 = np.std(sales[-3:], ddof=1)
    rolling_std_6 = np.std(sales[-6:], ddof=1)
    rolling_std_12 = np.std(sales[-12:], ddof=1)

    # --------------------------------------------------------
    # Year-over-year change
    # --------------------------------------------------------

    if sales[-12] != 0:
        yoy_change = (
            (sales[-1] - sales[-12])
            / sales[-12]
        )
    else:
        yoy_change = 0.0

    # --------------------------------------------------------
    # Country / product information
    # --------------------------------------------------------

    region_rows = history_df.loc[
        history_df["Country"] == country,
        "Region"
    ]

    if len(region_rows) > 0:
        region = region_rows.iloc[0]
    else:
        region = "Unknown"

    country_id = get_country_id(
        history_df,
        country
    )

    product_type_id = get_product_type_id(
        history_df,
        product_type
    )

    # --------------------------------------------------------
    # Create future row
    # --------------------------------------------------------

    future_row = {
        "Date": future_date,
        "Year": year,
        "Month": month,
        "Month_Name": month_name,
        "Quarter": quarter,
        "Year_Month": year_month,
        "Season": season,
        "Region": region,
        "Country": country,
        "Country_ID": country_id,
        "Product_Type": product_type,
        "Product_Type_ID": product_type_id,
        "Month_Sin": month_sin,
        "Month_Cos": month_cos,
        "Time_Index": time_index,
        "Lag_1": lag_1,
        "Lag_2": lag_2,
        "Lag_3": lag_3,
        "Lag_6": lag_6,
        "Lag_12": lag_12,
        "Rolling_Mean_3": rolling_mean_3,
        "Rolling_Mean_6": rolling_mean_6,
        "Rolling_Mean_12": rolling_mean_12,
        "Rolling_Std_3": rolling_std_3,
        "Rolling_Std_6": rolling_std_6,
        "Rolling_Std_12": rolling_std_12,
        "YoY_Change": yoy_change,
    }

    return future_row


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(feature_df):
    """
    Remove columns that were not used by the model.
    """

    X = feature_df.copy()

    for column in DROP_COLUMNS:
        if column in X.columns:
            X = X.drop(columns=column)

    return X


# ============================================================
# MONTHLY FORECAST
# ============================================================

def generate_forecast(
    model,
    history_df,
    start_date,
    forecast_months
):
    """
    Generate recursive monthly forecasts.

    Forecasting is performed separately for every
    Country + Product_Type combination.
    """

    history_df = history_df.copy()

    history_df["Date"] = pd.to_datetime(
        history_df["Date"]
    )

    history_df = history_df.sort_values(
        ["Country", "Product_Type", "Date"]
    )

    countries = sorted(
        history_df["Country"].unique()
    )

    product_types = sorted(
        history_df["Product_Type"].unique()
    )

    print(
        f"Number of country/product series: "
        f"{len(countries) * len(product_types)}"
    )

    expected_rows = (
        len(countries)
        * len(product_types)
        * forecast_months
    )

    print(
        f"Expected forecast rows: {expected_rows}"
    )

    forecast_rows = []

    # --------------------------------------------------------
    # Forecast month by month
    # --------------------------------------------------------

    for month_number in range(forecast_months):

        future_date = (
            start_date
            + pd.DateOffset(months=month_number)
        )

        print(
            f"Forecasting: "
            f"{future_date.strftime('%Y-%m')}"
        )

        for country in countries:

            for product_type in product_types:

                # --------------------------------------------
                # Create future features
                # --------------------------------------------

                future_row = create_future_features(
                    history_df=history_df,
                    future_date=future_date,
                    country=country,
                    product_type=product_type
                )

                future_df = pd.DataFrame(
                    [future_row]
                )

                # --------------------------------------------
                # Prepare model input
                # --------------------------------------------

                X_future = prepare_model_input(
                    future_df
                )

                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                prediction = model.predict(
                    X_future
                )[0]

                # Make sure forecast cannot be negative
                prediction = max(
                    0,
                    float(prediction)
                )

                # --------------------------------------------
                # Store forecast
                # --------------------------------------------

                future_row["Sales_Volume"] = prediction

                forecast_rows.append(
                    future_row.copy()
                )

                # --------------------------------------------
                # Add prediction to history
                #
                # This makes the forecasting recursive.
                # --------------------------------------------

                history_row = future_row.copy()

                history_row["Sales_Volume"] = prediction

                history_df = pd.concat(
                    [
                        history_df,
                        pd.DataFrame([history_row])
                    ],
                    ignore_index=True
                )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    return forecast_df


# ============================================================
# CREATE MONTHLY FORECAST CSV
# ============================================================

def create_monthly_forecast(forecast_df):

    monthly_df = forecast_df.copy()

    monthly_df = monthly_df[
        [
            "Date",
            "Year",
            "Month",
            "Month_Name",
            "Year_Month",
            "Region",
            "Country",
            "Product_Type",
            "Season",
            "Sales_Volume",
        ]
    ]

    monthly_df = monthly_df.sort_values(
        [
            "Date",
            "Country",
            "Product_Type"
        ]
    )

    output_path = (
        OUTPUT_DIR
        / "2025_monthly_forecast.csv"
    )

    monthly_df.to_csv(
        output_path,
        index=False
    )

    return monthly_df, output_path


# ============================================================
# CREATE SEASONAL FORECAST CSV
# ============================================================

def create_seasonal_forecast(forecast_df):

    seasonal_df = (
        forecast_df
        .groupby(
            [
                "Year",
                "Season",
                "Region",
                "Country",
                "Product_Type",
            ],
            as_index=False
        )["Sales_Volume"]
        .sum()
    )

    seasonal_df = seasonal_df.sort_values(
        [
            "Year",
            "Season",
            "Country",
            "Product_Type",
        ]
    )

    output_path = (
        OUTPUT_DIR
        / "2025_seasonal_forecast.csv"
    )

    seasonal_df.to_csv(
        output_path,
        index=False
    )

    return seasonal_df, output_path


# ============================================================
# CREATE YEARLY FORECAST CSV
# ============================================================

def create_yearly_forecast(forecast_df):

    yearly_df = (
        forecast_df
        .groupby(
            [
                "Year",
                "Region",
                "Country",
                "Product_Type",
            ],
            as_index=False
        )["Sales_Volume"]
        .sum()
    )

    yearly_df = yearly_df.sort_values(
        [
            "Year",
            "Country",
            "Product_Type",
        ]
    )

    output_path = (
        OUTPUT_DIR
        / "2025_yearly_forecast.csv"
    )

    yearly_df.to_csv(
        output_path,
        index=False
    )

    return yearly_df, output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("2025 APPAREL DEMAND FORECASTING")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load tuned model
    # --------------------------------------------------------

    print("\nLoading tuned model...")

    if not TUNED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tuned model not found:\n"
            f"{TUNED_MODEL_PATH}"
        )

    tuned_model = joblib.load(
        TUNED_MODEL_PATH
    )

    print(
        f"Tuned model loaded:\n"
        f"{TUNED_MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load model information
    # --------------------------------------------------------

    if TUNED_INFO_PATH.exists():

        tuned_info = joblib.load(
            TUNED_INFO_PATH
        )

        print("\nModel information:")
        print(
            f"Model: "
            f"{tuned_info.get('model_name', 'Unknown')}"
        )

        print(
            f"Selected model: "
            f"{tuned_info.get('selected_model', 'Unknown')}"
        )

        print(
            f"Target type: "
            f"{tuned_info.get('target_type', 'Unknown')}"
        )

    # --------------------------------------------------------
    # Load historical data
    # --------------------------------------------------------

    print("\nLoading historical data...")

    history_df = load_and_clean_data(
        DATA_PATH
    )

    history_df["Date"] = pd.to_datetime(
        history_df["Date"]
    )

    history_df = history_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    print(
        f"Historical data shape: "
        f"{history_df.shape}"
    )

    print(
        f"Historical date range: "
        f"{history_df['Date'].min().strftime('%Y-%m')} "
        f"to "
        f"{history_df['Date'].max().strftime('%Y-%m')}"
    )

    # --------------------------------------------------------
    # Determine forecast period
    # --------------------------------------------------------

    last_historical_date = history_df["Date"].max()

    first_forecast_date = (
        last_historical_date
        + pd.DateOffset(months=1)
    )

    last_forecast_date = (
        first_forecast_date
        + pd.DateOffset(
            months=FORECAST_MONTHS - 1
        )
    )

    print(
        f"\nLast historical month: "
        f"{last_historical_date.strftime('%Y-%m')}"
    )

    print(
        f"First forecast month: "
        f"{first_forecast_date.strftime('%Y-%m')}"
    )

    print(
        f"Last forecast month: "
        f"{last_forecast_date.strftime('%Y-%m')}"
    )

    print(
        f"Forecast months: "
        f"{FORECAST_MONTHS}"
    )

    # --------------------------------------------------------
    # Ensure forecast starts in 2025
    # --------------------------------------------------------

    if first_forecast_date.year != 2025:

        raise ValueError(
            "The next historical month is not January 2025. "
            f"Found {first_forecast_date.strftime('%Y-%m')}."
        )

    # --------------------------------------------------------
    # Generate forecast
    # --------------------------------------------------------

    print("\nGenerating 2025 forecasts...")

    forecast_df = generate_forecast(
        model=tuned_model,
        history_df=history_df,
        start_date=first_forecast_date,
        forecast_months=FORECAST_MONTHS
    )

    # --------------------------------------------------------
    # Verify forecast
    # --------------------------------------------------------

    print("\nForecast generated successfully.")

    print(
        f"Forecast shape: "
        f"{forecast_df.shape}"
    )

    print(
        f"Forecast date range: "
        f"{forecast_df['Date'].min().strftime('%Y-%m')} "
        f"to "
        f"{forecast_df['Date'].max().strftime('%Y-%m')}"
    )

    # --------------------------------------------------------
    # Save monthly forecast
    # --------------------------------------------------------

    monthly_df, monthly_path = (
        create_monthly_forecast(
            forecast_df
        )
    )

    # --------------------------------------------------------
    # Save seasonal forecast
    # --------------------------------------------------------

    seasonal_df, seasonal_path = (  # noqa: RUF059
        create_seasonal_forecast(
            forecast_df
        )
    )

    # --------------------------------------------------------
    # Save yearly forecast
    # --------------------------------------------------------

    yearly_df, yearly_path = (  # noqa: RUF059
        create_yearly_forecast(
            forecast_df
        )
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("MONTHLY FORECAST")
    print("=" * 70)

    monthly_summary = (
        monthly_df
        .groupby(
            [
                "Year",
                "Month",
                "Month_Name"
            ],
            as_index=False
        )["Sales_Volume"]
        .sum()
    )

    print(
        monthly_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("SEASONAL FORECAST")
    print("=" * 70)

    seasonal_summary = (
        forecast_df
        .groupby(
            [
                "Year",
                "Season"
            ],
            as_index=False
        )["Sales_Volume"]
        .sum()
    )

    print(
        seasonal_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("YEARLY FORECAST")
    print("=" * 70)

    yearly_summary = (
        forecast_df
        .groupby(
            ["Year"],
            as_index=False
        )["Sales_Volume"]
        .sum()
    )

    print(
        yearly_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # TOTAL FORECAST
    # ========================================================

    total_forecast = (
        forecast_df["Sales_Volume"].sum()
    )

    print("\n" + "=" * 70)
    print("TOTAL 2025 FORECAST")
    print("=" * 70)

    print(
        f"Total predicted sales volume: "
        f"{total_forecast:,.2f}"
    )

    # ========================================================
    # FILE LOCATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("CSV FILES SAVED")
    print("=" * 70)

    print(
        f"\nMonthly forecast:\n"
        f"{monthly_path}"
    )

    print(
        f"\nSeasonal forecast:\n"
        f"{seasonal_path}"
    )

    print(
        f"\nYearly forecast:\n"
        f"{yearly_path}"
    )

    print("\n" + "=" * 70)
    print("FORECASTING COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()