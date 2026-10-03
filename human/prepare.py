from pathlib import Path

import pandas as pd

data_dir = Path(__file__).parent.parent / "data"
# source data, read straight from S3 (not stored locally): train is sampled from 2005,
# eval and holdout from 2006
source_url = "https://xgboost-autoresearch--airline-dataset.s3.us-west-2.amazonaws.com/{year}.csv"
train_year, test_year = 2005, 2006

keep_cols = ["Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "UniqueCarrier",
             "Origin", "Dest", "Distance", "dep_delayed_15min"]


def load(year):
    df = pd.read_csv(source_url.format(year=year), na_values="NA",
                     usecols=[c for c in keep_cols if c != "dep_delayed_15min"] + ["DepTime", "DepDelay"])

    # keep only flights that departed (DepTime missing = cancelled); the actual DepTime itself is
    # not kept, DepTime minus the scheduled CRSDepTime is DepDelay, which would leak the target
    df = df[df["DepTime"].notna()]

    df["dep_delayed_15min"] = (pd.to_numeric(df["DepDelay"], errors="coerce") >= 15).map({True: "Y", False: "N"})
    for col in ["Month", "DayofMonth", "DayOfWeek"]:
        df[col] = "c-" + df[col].astype("Int64").astype("string")

    df = df[keep_cols].dropna().reset_index(drop=True)
    df = df.astype({"CRSDepTime": "int64", "Distance": "int64"})

    print(f"\n{year}: {df.shape}")
    print(df.head())
    print(df["dep_delayed_15min"].value_counts().sort_index())
    return df


def sample_balanced(df, n_per_class):
    return [df[df["dep_delayed_15min"] == label].sample(n=n_per_class, random_state=123)
            for label in ["N", "Y"]]


def shuffle(neg, pos):
    return pd.concat([neg, pos]).sample(frac=1.0, random_state=123)


df_train = shuffle(*sample_balanced(load(train_year), 100_000))

# eval and holdout are disjoint halves of one balanced sample
neg, pos = sample_balanced(load(test_year), 50_000)
df_eval = shuffle(neg.iloc[:25_000], pos.iloc[:25_000])
df_holdout = shuffle(neg.iloc[25_000:], pos.iloc[25_000:])

for name, d in [("train", df_train), ("eval", df_eval), ("holdout", df_holdout)]:
    print(f"\n{name}: {d.shape}")
    print(d["dep_delayed_15min"].value_counts().sort_index())
    d.to_csv(data_dir / f"{name}.csv", index=False)
