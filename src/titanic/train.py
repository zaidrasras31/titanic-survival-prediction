
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters with GridSearchCV."""
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search
