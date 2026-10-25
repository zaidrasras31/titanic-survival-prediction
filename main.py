
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.titanic.preprocessing import add_cabin_feature
from src.titanic.train import train_random_forest
from src.titanic.evaluate import print_evaluation


def main():
    # 1. Load the Titanic dataset
    data_path = Path("data/raw/train.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Download train.csv from Kaggle "
            "and place it inside data/raw/."
        )

    df = pd.read_csv(data_path)

    # 2. Feature engineering
    df = add_cabin_feature(df)

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Has_Cabin",
    ]

    X = df[features]
    y = df["Survived"]

    # 3. Split BEFORE fitting preprocessing
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Define feature types
    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Has_Cabin",
    ]

    categorical_features = [
        "Sex",
        "Embarked",
    ]

    # 5. Numerical preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # 6. Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    # 7. Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 8. Fit preprocessing on training data only
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the SAME transformations to validation data
    X_val_processed = preprocessor.transform(X_val)

    # 9. Train Random Forest
    model = train_random_forest(
        X_train_processed,
        y_train
    )

    # 10. Evaluate the model
    print("=" * 45)
    print("Titanic Survival Prediction")
    print("=" * 45)

    print("Model: Random Forest")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    print("\nModel Evaluation:")
    print_evaluation(
        model,
        X_val_processed,
        y_val
    )


if __name__ == "__main__":
    main()
