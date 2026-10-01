from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_cleaning import load_and_clean_data


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "tuned_model.pkl"
MODEL_INFO_FILE = PROJECT_ROOT / "models" / "tuned_model_info.pkl"
DATA_FILE = PROJECT_ROOT / "data" / "sales_demand.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zara Apparel Demand Forecasting",
    page_icon="Z",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Tuned model was not found at:\n{MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)
    info = joblib.load(MODEL_INFO_FILE) if MODEL_INFO_FILE.exists() else None
    return model, info


try:
    model, model_info = load_model()
except Exception as error:
    st.error("Unable to load the tuned model.")
    st.exception(error)
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_history():
    data = load_and_clean_data(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Sales_Volume"] = pd.to_numeric(data["Sales_Volume"], errors="coerce")

    data = data.dropna(
        subset=["Date", "Region", "Country", "Product_Type", "Sales_Volume"]
    )

    return data.sort_values(
        ["Country", "Product_Type", "Date"]
    ).reset_index(drop=True)


try:
    history = load_history()
except Exception as error:
    st.error("Unable to load historical data.")
    st.exception(error)
    st.stop()


# ============================================================
# BASIC VALUES
# ============================================================

products = sorted(history["Product_Type"].unique().tolist())
regions = sorted(history["Region"].unique().tolist())
countries = sorted(history["Country"].unique().tolist())

first_date = history["Date"].min()
last_date = history["Date"].max()

month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

season_months = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
}

seasons = list(season_months.keys())


def get_season(month):
    for season, months in season_months.items():
        if month in months:
            return season
    return "Unknown"


def countries_for_region(region):
    return sorted(
        history.loc[history["Region"] == region, "Country"].unique().tolist()
    )


def region_for_country(country):
    rows = history.loc[history["Country"] == country, "Region"]
    return rows.iloc[0] if not rows.empty else None


# ============================================================
# FEATURE CREATION FOR FUTURE MONTH
# ============================================================

def make_features(current_history, forecast_date, country, product_type):
    series = current_history.loc[
        (current_history["Country"] == country)
        & (current_history["Product_Type"] == product_type)
    ].sort_values("Date")

    values = series["Sales_Volume"].astype(float).tolist()

    if len(values) < 12:
        raise ValueError(
            f"Not enough historical data for {country} - {product_type}."
        )

    month = forecast_date.month
    lag_1 = values[-1]
    lag_12 = values[-12]

    if lag_12 != 0:
        yoy = (lag_1 - lag_12) / lag_12
    else:
        yoy = 0.0

    return {
        "Year": forecast_date.year,
        "Month": month,
        "Quarter": forecast_date.quarter,
        "Season": get_season(month),
        "Region": series["Region"].iloc[-1],
        "Country": country,
        "Product_Type": product_type,
        "Month_Sin": np.sin(2 * np.pi * month / 12),
        "Month_Cos": np.cos(2 * np.pi * month / 12),
        "Time_Index": (
            (forecast_date.year - first_date.year) * 12
            + (month - first_date.month)
        ),
        "Lag_1": lag_1,
        "Lag_2": values[-2],
        "Lag_3": values[-3],
        "Lag_6": values[-6],
        "Lag_12": lag_12,
        "Rolling_Mean_3": np.mean(values[-3:]),
        "Rolling_Mean_6": np.mean(values[-6:]),
        "Rolling_Mean_12": np.mean(values[-12:]),
        "Rolling_Std_3": np.std(values[-3:], ddof=1),
        "Rolling_Std_6": np.std(values[-6:], ddof=1),
        "Rolling_Std_12": np.std(values[-12:], ddof=1),
        "YoY_Change": yoy,
    }


def model_input(features):
    data = pd.DataFrame(features)
    return data


# ============================================================
# RECURSIVE FORECAST
# ============================================================

@st.cache_data(show_spinner=False)
def generate_forecast(target_date):
    target_date = pd.Timestamp(target_date)

    if target_date <= last_date:
        raise ValueError(
            f"Forecast date must be after {last_date.strftime('%B %Y')}."
        )

    working = history.copy()

    combinations = (
        working[["Country", "Product_Type"]]
        .drop_duplicates()
        .sort_values(["Country", "Product_Type"])
        .reset_index(drop=True)
    )

    dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        end=target_date,
        freq="MS",
    )

    all_predictions = []

    # Predict all country/product combinations for one month together.
    for forecast_date in dates:
        feature_rows = []
        metadata_rows = []

        for row in combinations.itertuples(index=False):
            country = row.Country
            product = row.Product_Type

            features = make_features(
                working,
                forecast_date,
                country,
                product,
            )

            feature_rows.append(features)
            metadata_rows.append((country, product))

        X = model_input(feature_rows)
        predictions = np.maximum(model.predict(X), 0)

        new_rows = []

        for i, (country, product) in enumerate(metadata_rows):
            series = working.loc[
                (working["Country"] == country)
                & (working["Product_Type"] == product)
            ]

            country_id = series["Country_ID"].iloc[-1]
            product_id = series["Product_Type_ID"].iloc[-1]
            region = series["Region"].iloc[-1]
            prediction = float(predictions[i])

            all_predictions.append(
                {
                    "Date": forecast_date,
                    "Year": forecast_date.year,
                    "Month": forecast_date.month,
                    "Month_Name": forecast_date.strftime("%B"),
                    "Quarter": forecast_date.quarter,
                    "Season": get_season(forecast_date.month),
                    "Region": region,
                    "Country": country,
                    "Product_Type": product,
                    "Predicted_Sales_Volume": prediction,
                }
            )

            new_rows.append(
                {
                    "Date": forecast_date,
                    "Year": forecast_date.year,
                    "Month": forecast_date.month,
                    "Month_Name": forecast_date.strftime("%B"),
                    "Quarter": forecast_date.quarter,
                    "Year_Month": forecast_date.strftime("%Y-%m"),
                    "Season": get_season(forecast_date.month),
                    "Region": region,
                    "Country": country,
                    "Country_ID": country_id,
                    "Product_Type": product,
                    "Product_Type_ID": product_id,
                    "Sales_Volume": prediction,
                }
            )

        working = pd.concat(
            [working, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

    return pd.DataFrame(all_predictions)


# ============================================================
# TABLE / SUMMARY HELPERS
# ============================================================

TABLE_COLUMNS = [
    "Date",
    "Year",
    "Month",
    "Month_Name",
    "Quarter",
    "Season",
    "Region",
    "Country",
    "Product_Type",
    "Predicted_Sales_Volume",
]


def show_table(data):
    display = data[TABLE_COLUMNS].copy()
    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%Y-%m-%d")
    display["Predicted_Sales_Volume"] = (
        display["Predicted_Sales_Volume"].round(0).astype(int)
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def show_metrics(data):
    total = data["Predicted_Sales_Volume"].sum()
    average = data["Predicted_Sales_Volume"].mean()
    maximum = data["Predicted_Sales_Volume"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predicted Demand", f"{total:,.0f}")
    c2.metric("Average Prediction", f"{average:,.0f}")
    c3.metric("Maximum Prediction", f"{maximum:,.0f}")
    c4.metric("Prediction Records", f"{len(data):,}")


def product_bar(data, title):
    chart = (
        data.groupby("Product_Type", as_index=False)["Predicted_Sales_Volume"]
        .sum()
        .sort_values("Predicted_Sales_Volume", ascending=False)
    )

    return px.bar(
        chart,
        x="Product_Type",
        y="Predicted_Sales_Volume",
        title=title,
        text_auto=".0f",
    )


def country_bar(data, title):
    chart = (
        data.groupby("Country", as_index=False)["Predicted_Sales_Volume"]
        .sum()
        .sort_values("Predicted_Sales_Volume", ascending=False)
    )

    return px.bar(
        chart,
        x="Country",
        y="Predicted_Sales_Volume",
        title=title,
        text_auto=".0f",
    )


def region_bar(data, title):
    chart = (
        data.groupby("Region", as_index=False)["Predicted_Sales_Volume"]
        .sum()
        .sort_values("Predicted_Sales_Volume", ascending=False)
    )

    return px.bar(
        chart,
        x="Region",
        y="Predicted_Sales_Volume",
        title=title,
        text_auto=".0f",
    )


def product_pie(data, title):
    chart = data.groupby(
        "Product_Type", as_index=False
    )["Predicted_Sales_Volume"].sum()

    return px.pie(
        chart,
        names="Product_Type",
        values="Predicted_Sales_Volume",
        title=title,
        hole=0.35,
    )


def country_pie(data, title):
    chart = data.groupby(
        "Country", as_index=False
    )["Predicted_Sales_Volume"].sum()

    return px.pie(
        chart,
        names="Country",
        values="Predicted_Sales_Volume",
        title=title,
        hole=0.35,
    )


def monthly_line(data, title):
    chart = (
        data.groupby("Date", as_index=False)["Predicted_Sales_Volume"]
        .sum()
        .sort_values("Date")
    )

    return px.line(
        chart,
        x="Date",
        y="Predicted_Sales_Volume",
        markers=True,
        title=title,
    )


def historical_line(data, title):
    chart = (
        data.groupby("Date", as_index=False)["Sales_Volume"]
        .sum()
        .sort_values("Date")
    )

    return px.line(
        chart,
        x="Date",
        y="Sales_Volume",
        title=title,
        markers=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("Zara Demand Forecast")
    st.caption("ML-Based Apparel Demand Forecasting")
    st.divider()

    page = st.radio(
        "Select Analysis",
        [
            "Dashboard",
            "Month Based",
            "Season Based",
            "Year Based",
            "Region Based",
            "Country Based",
        ],
    )

    st.divider()
    st.caption(
        "Prediction dimensions: Year | Month | Season | "
        "Region | Country | Product Type"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.title("Zara Apparel Demand Forecasting")
    st.write(
        "Historical demand analysis and future demand forecasting."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(history):,}")
    c2.metric("Products", len(products))
    c3.metric("Countries", len(countries))
    c4.metric("Regions", len(regions))
    c5.metric("Years", history["Year"].nunique())

    st.divider()

    selected_product = st.selectbox(
        "Product Type",
        products,
        key="dashboard_product",
    )

    product_history = history.loc[
        history["Product_Type"] == selected_product
    ]

    st.plotly_chart(
        historical_line(
            product_history,
            f"{selected_product} Historical Demand",
        ),
        use_container_width=True,
    )

    yearly = (
        product_history.groupby("Year", as_index=False)["Sales_Volume"]
        .sum()
    )

    fig = px.bar(
        yearly,
        x="Year",
        y="Sales_Volume",
        title=f"{selected_product} Year-wise Demand",
        text_auto=".0f",
    )

    st.plotly_chart(fig, use_container_width=True)

    totals = history.groupby(
        "Product_Type", as_index=False
    )["Sales_Volume"].sum()

    fig = px.pie(
        totals,
        names="Product_Type",
        values="Sales_Volume",
        hole=0.35,
        title="Historical Demand Share by Product",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MONTH BASED
# ============================================================

elif page == "Month Based":
    st.title("Month Based Prediction")
    st.write(
        "Predict demand for one future month. "
        "All important categorical dimensions are shown in the result."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        year = st.number_input(
            "Forecast Year",
            min_value=last_date.year + 1,
            max_value=2100,
            value=last_date.year + 1,
            step=1,
            key="mb_year",
        )

    with c2:
        month = st.selectbox(
            "Forecast Month",
            range(1, 13),
            format_func=lambda x: month_names[x - 1],
            key="mb_month",
        )

    with c3:
        product = st.selectbox(
            "Product Filter",
            ["All Products"] + products,
            key="mb_product",
        )

    region = st.selectbox(
        "Region Filter",
        ["All Regions"] + regions,
        key="mb_region",
    )

    available_countries = (
        countries if region == "All Regions"
        else countries_for_region(region)
    )

    country = st.selectbox(
        "Country Filter",
        ["All Countries"] + available_countries,
        key="mb_country",
    )

    target = pd.Timestamp(year, month, 1)

    if st.button(
        "GENERATE MONTH PREDICTION",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Generating monthly prediction..."):
                result = generate_forecast(target)

            result = result[result["Date"] == target].copy()

            if product != "All Products":
                result = result[result["Product_Type"] == product]
            if region != "All Regions":
                result = result[result["Region"] == region]
            if country != "All Countries":
                result = result[result["Country"] == country]

            st.session_state["month_result"] = result
            st.session_state["month_target"] = target
        except Exception as error:
            st.error("Unable to generate prediction.")
            st.exception(error)

    if "month_result" in st.session_state:
        result = st.session_state["month_result"]
        target = st.session_state["month_target"]

        if result.empty:
            st.warning("No records match the selected filters.")
        else:
            st.success("Monthly prediction generated successfully.")
            show_metrics(result)
            st.subheader("Prediction Table")
            show_table(result)

            st.plotly_chart(
                product_bar(result, f"Product Demand - {target.strftime('%B %Y')}"),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    country_bar(result, "Country-wise Demand"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    product_pie(result, "Product Demand Contribution"),
                    use_container_width=True,
                )


# ============================================================
# SEASON BASED
# ============================================================

elif page == "Season Based":
    st.title("Season Based Prediction")
    st.write(
        "Predict demand for a selected season. "
        "The result keeps month, season, region, country and product."
    )

    c1, c2 = st.columns(2)

    with c1:
        year = st.number_input(
            "Forecast Year",
            min_value=last_date.year + 1,
            max_value=2100,
            value=last_date.year + 1,
            step=1,
            key="sb_year",
        )

    with c2:
        season = st.selectbox(
            "Season",
            seasons,
            key="sb_season",
        )

    region = st.selectbox(
        "Region Filter",
        ["All Regions"] + regions,
        key="sb_region",
    )

    available_countries = (
        countries if region == "All Regions"
        else countries_for_region(region)
    )

    country = st.selectbox(
        "Country Filter",
        ["All Countries"] + available_countries,
        key="sb_country",
    )

    product = st.selectbox(
        "Product Filter",
        ["All Products"] + products,
        key="sb_product",
    )

    target = pd.Timestamp(year, 12, 1)

    if st.button(
        "GENERATE SEASON PREDICTION",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Generating seasonal prediction..."):
                result = generate_forecast(target)

            result = result[
                (result["Year"] == year)
                & (result["Season"] == season)
            ].copy()

            if product != "All Products":
                result = result[result["Product_Type"] == product]
            if region != "All Regions":
                result = result[result["Region"] == region]
            if country != "All Countries":
                result = result[result["Country"] == country]

            st.session_state["season_result"] = result
        except Exception as error:
            st.error("Unable to generate prediction.")
            st.exception(error)

    if "season_result" in st.session_state:
        result = st.session_state["season_result"]

        if result.empty:
            st.warning("No records match the selected filters.")
        else:
            st.success("Seasonal prediction generated successfully.")
            show_metrics(result)
            st.subheader(f"{season} {year} Prediction Table")
            show_table(result)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    monthly_line(result, f"{season} {year} Monthly Demand"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    product_bar(result, f"{season} {year} Product Demand"),
                    use_container_width=True,
                )

            st.plotly_chart(
                product_pie(result, f"{season} {year} Product Contribution"),
                use_container_width=True,
            )


# ============================================================
# YEAR BASED
# ============================================================

elif page == "Year Based":
    st.title("Year Based Prediction")
    st.write(
        "Generate the complete January-December forecast for a selected year."
    )

    year = st.number_input(
        "Forecast Year",
        min_value=last_date.year + 1,
        max_value=2100,
        value=last_date.year + 1,
        step=1,
        key="yb_year",
    )

    region = st.selectbox(
        "Region Filter",
        ["All Regions"] + regions,
        key="yb_region",
    )

    available_countries = (
        countries if region == "All Regions"
        else countries_for_region(region)
    )

    country = st.selectbox(
        "Country Filter",
        ["All Countries"] + available_countries,
        key="yb_country",
    )

    product = st.selectbox(
        "Product Filter",
        ["All Products"] + products,
        key="yb_product",
    )

    target = pd.Timestamp(year, 12, 1)

    if st.button(
        "GENERATE YEAR PREDICTION",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Generating yearly prediction..."):
                result = generate_forecast(target)

            result = result[result["Year"] == year].copy()

            if product != "All Products":
                result = result[result["Product_Type"] == product]
            if region != "All Regions":
                result = result[result["Region"] == region]
            if country != "All Countries":
                result = result[result["Country"] == country]

            st.session_state["year_result"] = result
        except Exception as error:
            st.error("Unable to generate prediction.")
            st.exception(error)

    if "year_result" in st.session_state:
        result = st.session_state["year_result"]

        if result.empty:
            st.warning("No records match the selected filters.")
        else:
            st.success("Yearly prediction generated successfully.")
            show_metrics(result)
            st.subheader(f"{year} Prediction Table")
            show_table(result)

            st.plotly_chart(
                monthly_line(result, f"{year} Monthly Demand"),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    product_bar(result, f"{year} Product Demand"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    product_pie(result, f"{year} Product Contribution"),
                    use_container_width=True,
                )


# ============================================================
# REGION BASED
# ============================================================

elif page == "Region Based":
    st.title("Region Based Prediction")
    st.write(
        "Select a region. The prediction contains all countries and products in that region."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        region = st.selectbox(
            "Region",
            regions,
            key="rb_region",
        )

    with c2:
        year = st.number_input(
            "Forecast Year",
            min_value=last_date.year + 1,
            max_value=2100,
            value=last_date.year + 1,
            step=1,
            key="rb_year",
        )

    with c3:
        month = st.selectbox(
            "Forecast Month",
            range(1, 13),
            format_func=lambda x: month_names[x - 1],
            key="rb_month",
        )

    product = st.selectbox(
        "Product Filter",
        ["All Products"] + products,
        key="rb_product",
    )

    region_countries = countries_for_region(region)
    st.info(
        f"Countries in {region}: {', '.join(region_countries)}"
    )

    target = pd.Timestamp(year, month, 1)

    if st.button(
        "GENERATE REGION PREDICTION",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Generating region prediction..."):
                result = generate_forecast(target)

            result = result[
                (result["Date"] == target)
                & (result["Region"] == region)
            ].copy()

            if product != "All Products":
                result = result[result["Product_Type"] == product]

            st.session_state["region_result"] = result
        except Exception as error:
            st.error("Unable to generate prediction.")
            st.exception(error)

    if "region_result" in st.session_state:
        result = st.session_state["region_result"]

        if result.empty:
            st.warning("No records match the selected filters.")
        else:
            st.success("Region prediction generated successfully.")
            show_metrics(result)
            st.subheader(
                f"{region} - {target.strftime('%B %Y')} Prediction Table"
            )
            show_table(result)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    country_bar(result, f"{region} Country-wise Demand"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    product_pie(result, f"{region} Product Contribution"),
                    use_container_width=True,
                )

            st.plotly_chart(
                product_bar(result, f"{region} Product-wise Demand"),
                use_container_width=True,
            )


# ============================================================
# COUNTRY BASED
# ============================================================

elif page == "Country Based":
    st.title("Country Based Prediction")
    st.write(
        "Select a country. The result contains its region, products, month and season."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        country = st.selectbox(
            "Country",
            countries,
            key="cb_country",
        )

    with c2:
        year = st.number_input(
            "Forecast Year",
            min_value=last_date.year + 1,
            max_value=2100,
            value=last_date.year + 1,
            step=1,
            key="cb_year",
        )

    with c3:
        month = st.selectbox(
            "Forecast Month",
            range(1, 13),
            format_func=lambda x: month_names[x - 1],
            key="cb_month",
        )

    product = st.selectbox(
        "Product Filter",
        ["All Products"] + products,
        key="cb_product",
    )

    region = region_for_country(country)
    st.info(f"Region: {region}")

    target = pd.Timestamp(year, month, 1)

    if st.button(
        "GENERATE COUNTRY PREDICTION",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Generating country prediction..."):
                result = generate_forecast(target)

            result = result[
                (result["Date"] == target)
                & (result["Country"] == country)
            ].copy()

            if product != "All Products":
                result = result[result["Product_Type"] == product]

            st.session_state["country_result"] = result
        except Exception as error:
            st.error("Unable to generate prediction.")
            st.exception(error)

    if "country_result" in st.session_state:
        result = st.session_state["country_result"]

        if result.empty:
            st.warning("No records match the selected filters.")
        else:
            st.success("Country prediction generated successfully.")
            show_metrics(result)
            st.subheader(
                f"{country} - {target.strftime('%B %Y')} Prediction Table"
            )
            show_table(result)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    product_bar(result, f"{country} Product-wise Demand"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    product_pie(result, f"{country} Product Contribution"),
                    use_container_width=True,
                )

            historical_country = history[
                history["Country"] == country
            ]

            st.plotly_chart(
                historical_line(
                    historical_country,
                    f"{country} Historical Demand",
                ),
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Zara Apparel Demand Forecasting | Historical Data: 2015-2024 | Tuned XGBoost"
)
