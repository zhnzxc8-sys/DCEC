"""Deterministic smoke test for the retrieval Recall@K implementation."""

import torch

from retrieval_metrics import compute_retrieval_metrics


def main():
    # Three unit image features and three fixed text features; no random data.
    image_features = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    text_features = torch.tensor(
        [[0.6, 0.7, 0.2], [0.9, 0.5, 0.3], [0.1, 0.4, 0.8]]
    )
    image_to_text = [[0], [1], [2]]
    text_to_image = [0, 1, 2]

    metrics = compute_retrieval_metrics(
        image_features, text_features, image_to_text, text_to_image
    )

    # In each direction, only the third pair ranks first; all pairs rank within 5/10.
    expected = {
        "i2t_R@1": 100.0 / 3.0,
        "i2t_R@5": 100.0,
        "i2t_R@10": 100.0,
        "t2i_R@1": 100.0 / 3.0,
        "t2i_R@5": 100.0,
        "t2i_R@10": 100.0,
    }
    for name, expected_value in expected.items():
        assert abs(metrics[name] - expected_value) < 1e-6, (
            f"{name}: expected {expected_value}, got {metrics[name]}"
        )

    print("Retrieval metric assertions passed.")


if __name__ == "__main__":
    main()
