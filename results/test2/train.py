import numpy as np
import pandas as pd
import time
import xgboost as xgb
from pathlib import Path
from harness import save_and_evaluate


data_dir = Path(__file__).parent / "data"
train = pd.read_csv(f"{data_dir}/train.csv")

date_cols = ["Month", "DayofMonth", "DayOfWeek"]
cat_cols = ["UniqueCarrier", "Origin", "Dest"]
num_cols = ["CRSDepTime", "Distance"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

def prepare(df):
    X = df[num_cols].copy()
    # ordinal calendar features ("c-11" -> 11)
    for col in date_cols:
        X[col] = df[col].str[2:].astype(int)
    X["Hour"] = df["CRSDepTime"] // 100
    X["Minute"] = df["CRSDepTime"] % 100
    X["HourOfWeek"] = (X["DayOfWeek"] - 1) * 24 + X["Hour"]
    X["MonthHour"] = (X["Month"] - 1) * 24 + X["Hour"]
    X["DepMinutes"] = (df["CRSDepTime"] // 100) * 60 + df["CRSDepTime"] % 100
    for col in cat_cols:
        X[col] = pd.Categorical(
            df[col].where(df[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    grow_policy="lossguide",
    max_leaves=20,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=30,
    reg_lambda=25,
    num_parallel_tree=4,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)


t0 = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


# Saves {model, prepare} to artifacts/ and scores eval.csv row by row. Keep this call last.
save_and_evaluate(model, prepare)
