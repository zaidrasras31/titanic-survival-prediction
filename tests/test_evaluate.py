import numpy as np
import pytest

from src.titanic.evaluate import evaluate_model


class DummyModel:
    def predict(self, X):
        return np.array([1, 0, 1, 0])


def test_evaluate_model():
    model = DummyModel()

    X_test = np.zeros((4, 2))
    y_test = np.array([1, 0, 0, 0])

    results = evaluate_model(model, X_test, y_test)

    assert results["accuracy"] == pytest.approx(0.75)
    assert results["precision"] == pytest.approx(0.5)
    assert results["recall"] == pytest.approx(1.0)
    assert results["f1_score"] == pytest.approx(2 / 3)

    expected_matrix = np.array([
        [2, 1],
        [0, 1],
    ])

    np.testing.assert_array_equal(
        results["confusion_matrix"],
        expected_matrix,
    )
