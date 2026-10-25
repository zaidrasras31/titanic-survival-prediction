
import pandas as pd

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Has_Cabin",
]


def add_cabin_feature(df):
    """Create a binary feature indicating cabin availability."""
    df = df.copy()
    df["Has_Cabin"] = df["Cabin"].notna().astype(int)
    return df


def fill_missing_values(df, age_median=None, fare_median=None):
    """Fill missing numerical values using supplied or calculated medians."""
    df = df.copy()

    if age_median is None:
        age_median = df["Age"].median()

    if fare_median is None:
        fare_median = df["Fare"].median()

    df["Age"] = df["Age"].fillna(age_median)
    df["Fare"] = df["Fare"].fillna(fare_median)
    df["Embarked"] = df["Embarked"].fillna("S")

    return df


def encode_features(df):
    """One-hot encode categorical features."""
    df = df.copy()
    df = add_cabin_feature(df)
    df = fill_missing_values(df)

    X = df[FEATURES].copy()

    X = pd.get_dummies(
        X,
        columns=["Sex", "Embarked"],
        dtype=int
    )

    return X
