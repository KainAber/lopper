from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def compute_node_stats(model: DecisionTreeRegressor, df: pd.DataFrame, target_col: str) -> list[dict[str, float]]:
    tree = model.tree_
    node_stats: list[dict[str, float]] = []

    features = df.drop(columns=[target_col]).values
    target = df[target_col].values
    node_indicator = model.decision_path(features)

    for node_id in range(tree.node_count):
        sample_indices = node_indicator[:, node_id].toarray().ravel().astype(bool)
        node_target = target[sample_indices]
        if node_target.size == 0:
            stats = {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        else:
            stats = {
                "count": float(node_target.size),
                "mean": float(np.mean(node_target)),
                "std": float(np.std(node_target)),
                "min": float(np.min(node_target)),
                "max": float(np.max(node_target)),
            }
        node_stats.append(stats)

    return node_stats


def pickle_tree_with_stats(
    model: DecisionTreeRegressor,
    df: pd.DataFrame,
    target_col: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.node_stats = compute_node_stats(model, df, target_col)
    joblib.dump(model, output_path)
    return output_path


def make_example_dataset(seed: int = 42, size: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1.0, size)
    x2 = rng.normal(2.0, 1.5, size)
    x3 = rng.uniform(-2.0, 2.0, size)
    noise = rng.normal(0, 0.5, size)
    y = 3.0 * x1 - 1.5 * x2 + 0.75 * x3 + noise

    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def train_and_save_example(output_path: str | Path) -> Path:
    df = make_example_dataset()
    features = df.drop(columns=["y"])
    target = df["y"]

    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(features, target)

    return pickle_tree_with_stats(model, df, "y", output_path)


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "data" / "example_tree.pkl"
    saved_path = train_and_save_example(output_path)
    print(f"Saved example tree to {saved_path}")


if __name__ == "__main__":
    main()
