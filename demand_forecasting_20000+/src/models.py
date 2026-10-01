import numpy as np

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor


# =========================================================
# RANDOM FOREST
# =========================================================

def train_random_forest(
    X_train,
    y_train
):

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# =========================================================
# XGBOOST
# =========================================================

def train_xgboost(
    X_train,
    y_train
):

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=-1,

        eval_metric="rmse"
    )

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    return model


# =========================================================
# CATBOOST
# =========================================================

def train_catboost(
    X_train,
    y_train
):

    model = CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.04,

        loss_function="RMSE",

        random_seed=42,

        verbose=False,

        thread_count=-1
    )

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    return model


# =========================================================
# TRAIN SELECTED MODEL
# =========================================================

def train_selected_model(
    model_name,
    X_train,
    y_train
):

    # -----------------------------------------------------
    # RANDOM FOREST - ORIGINAL TARGET
    # -----------------------------------------------------

    if model_name == "Random Forest - Original":

        model = train_random_forest(
            X_train,
            y_train
        )

        target_type = "original"

    # -----------------------------------------------------
    # XGBOOST - ORIGINAL TARGET
    # -----------------------------------------------------

    elif model_name == "XGBoost - Original":

        model = train_xgboost(
            X_train,
            y_train
        )

        target_type = "original"

    # -----------------------------------------------------
    # CATBOOST - ORIGINAL TARGET
    # -----------------------------------------------------

    elif model_name == "CatBoost - Original":

        model = train_catboost(
            X_train,
            y_train
        )

        target_type = "original"

    # -----------------------------------------------------
    # XGBOOST - LOG TARGET
    # -----------------------------------------------------

    elif model_name == "XGBoost - Log Target":

        y_train_log = np.log1p(
            y_train
        )

        model = train_xgboost(
            X_train,
            y_train_log
        )

        target_type = "log"

    # -----------------------------------------------------
    # CATBOOST - LOG TARGET
    # -----------------------------------------------------

    elif model_name == "CatBoost - Log Target":

        y_train_log = np.log1p(
            y_train
        )

        model = train_catboost(
            X_train,
            y_train_log
        )

        target_type = "log"

    # -----------------------------------------------------
    # INVALID MODEL
    # -----------------------------------------------------

    else:

        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    return (
        model,
        target_type
    )


# =========================================================
# PREDICTION
# =========================================================

def predict_model(
    model,
    target_type,
    X
):

    predictions = model.predict(
        X
    )

    # -----------------------------------------------------
    # CONVERT LOG PREDICTIONS
    # BACK TO ORIGINAL SCALE
    # -----------------------------------------------------

    if target_type == "log":

        predictions = np.expm1(
            predictions
        )

    # -----------------------------------------------------
    # DEMAND CANNOT BE NEGATIVE
    # -----------------------------------------------------

    predictions = np.clip(
        predictions,
        0,
        None
    )

    return predictions