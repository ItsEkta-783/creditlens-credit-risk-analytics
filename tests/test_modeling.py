import numpy as np
import pandas as pd

from src.clean_data import TARGET
from src.modeling import evaluate_predictions, make_preprocessor, split_features


def test_split_excludes_identifier_and_target_and_is_stratified():
    frame = pd.DataFrame({
        "id": range(100),
        "age": range(100),
        "sex": [1, 2] * 50,
        TARGET: [0] * 80 + [1] * 20,
    })
    X_train, X_test, y_train, y_test = split_features(frame)
    assert "id" not in X_train.columns and TARGET not in X_train.columns
    assert len(X_test) == 20
    assert y_test.mean() == y_train.mean() == 0.2
    assert X_train.index.equals(y_train.index)


def test_preprocessor_one_hot_encodes_categories_and_drops_id():
    X = pd.DataFrame({"id": [1, 2], "age": [20, 30], "sex": [1, 2]})
    transformer = make_preprocessor(["age", "sex"])
    transformed = transformer.fit_transform(X)
    names = transformer.get_feature_names_out().tolist()
    assert transformed.shape[1] == 3
    assert "id" not in names
    assert any(name.startswith("sex_") for name in names)


def test_evaluation_uses_probabilities_and_returns_confusion_matrix():
    result = evaluate_predictions([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.9], threshold=0.5)
    assert result["confusion_matrix"].tolist() == [[1, 1], [0, 2]]
    assert result["recall"] == 1.0
    assert result["precision"] == 2 / 3
    assert np.isclose(result["roc_auc"], 0.75)
