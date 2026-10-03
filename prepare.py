from pathlib import Path

import pandas as pd

data_dir = Path(__file__).parent / "data"
# source data, read straight from S3 (not stored locally)
source_url = "https://xgboost-autoresearch--airline-dataset.s3.us-west-2.amazonaws.com/2005.csv"

keep_cols = ["Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "UniqueCarrier",
             "Origin", "Dest", "Distance", "dep_delayed_15min"]

df = pd.read_csv(source_url, na_values="NA",
                 usecols=[c for c in keep_cols if c != "dep_delayed_15min"] + ["DepTime", "DepDelay"])

# keep only flights that departed (DepTime missing = cancelled); the actual DepTime itself is
# not kept, DepTime minus the scheduled CRSDepTime is DepDelay, which would leak the target
df = df[df["DepTime"].notna()]

df["dep_delayed_15min"] = (pd.to_numeric(df["DepDelay"], errors="coerce") >= 15).map({True: "Y", False: "N"})
for col in ["Month", "DayofMonth", "DayOfWeek"]:
    df[col] = "c-" + df[col].astype("Int64").astype("string")

df = df[keep_cols].dropna().reset_index(drop=True)
df = df.astype({"CRSDepTime": "int64", "Distance": "int64"})

print(df.shape)
print(df.head())
print(df["dep_delayed_15min"].value_counts().sort_index())


def split_4_1_1(d):
    n_train = len(d) * 4 // 6
    n_eval = len(d) // 6
    return d.iloc[:n_train], d.iloc[n_train:n_train + n_eval], d.iloc[n_train + n_eval:]


df_neg = df[df["dep_delayed_15min"] == "N"].sample(n=150_000, random_state=123)
df_pos = df[df["dep_delayed_15min"] == "Y"].sample(n=150_000, random_state=123)

df_train, df_eval, df_holdout = [
    pd.concat([neg, pos]).sample(frac=1.0, random_state=123)
    for neg, pos in zip(split_4_1_1(df_neg), split_4_1_1(df_pos))
]

for name, d in [("train", df_train), ("eval", df_eval), ("holdout", df_holdout)]:
    print(f"\n{name}: {d.shape}")
    print(d["dep_delayed_15min"].value_counts().sort_index())
    d.to_csv(data_dir / f"{name}.csv", index=False)
