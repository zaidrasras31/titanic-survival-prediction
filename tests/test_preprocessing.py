
import pandas as pd

from src.titanic.preprocessing import (
    add_cabin_feature,
    fill_missing_values,
    encode_features,
)


def sample_dataframe():
    return pd.DataFrame({
        "Pclass": [1, 2, 3],
        "Sex": ["female", "male", "female"],
        "Age": [25.0, None, 35.0],
        "SibSp": [0, 1, 0],
        "Parch": [0, 0, 1],
        "Fare": [100.0, 20.0, None],
        "Embarked": ["S", None, "C"],
        "Cabin": ["C85", None, "E46"],
    })


def test_add_cabin_feature():
    df = sample_dataframe()
    result = add_cabin_feature(df)

    assert result["Has_Cabin"].tolist() == [1, 0, 1]


def test_fill_missing_values():
    df = sample_dataframe()
    result = fill_missing_values(df)

    assert result["Age"].isna().sum() == 0
    assert result["Fare"].isna().sum() == 0
    assert result["Embarked"].isna().sum() == 0
    assert result.loc[1, "Age"] == 30.0
    assert result.loc[2, "Fare"] == 60.0


def test_encode_features():
    df = sample_dataframe()
    result = encode_features(df)

    assert "Has_Cabin" in result.columns
    assert "Sex_female" in result.columns
    assert "Sex_male" in result.columns
    assert "Embarked_S" in result.columns
    assert not result.isna().any().any()
  
