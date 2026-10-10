"""Offline unit test for the Flickr30K Karpathy data reader."""

from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, Features, Image as DatasetImage, Sequence, Value
from PIL import Image

from flickr30k_dataset import load_flickr30k_split, prepare_retrieval_samples


def main():
    # This deterministic fixture tests only the local Parquet data interface.
    features = Features(
        {
            "image_id": Value("string"),
            "image": DatasetImage(),
            "captions": Sequence(Value("string")),
        }
    )
    fixture = Dataset.from_dict(
        {
            "image_id": ["fixture-0", "fixture-1", "fixture-2"],
            "image": [
                Image.new("RGB", (2, 2), color=(255, 0, 0)),
                Image.new("RGB", (2, 2), color=(0, 255, 0)),
                Image.new("RGB", (2, 2), color=(0, 0, 255)),
            ],
            "captions": [
                [f"fixture image {image_index} caption {caption_index}" for caption_index in range(5)]
                for image_index in range(3)
            ],
        },
        features=features,
    )

    with TemporaryDirectory() as temporary_directory:
        parquet_path = Path(temporary_directory) / "flickr30k_fixture.parquet"
        cache_dir = Path(temporary_directory) / "cache"
        fixture.to_parquet(str(parquet_path))
        dataset = load_flickr30k_split(
            "test", local_parquet=parquet_path, cache_dir=cache_dir
        )
        images, captions_flat, image_to_text, text_to_image = prepare_retrieval_samples(
            dataset, num_samples=3
        )

        assert len(images) == 3
        assert all(len(text_indices) == 5 for text_indices in image_to_text)
        assert len(captions_flat) == 15
        assert image_to_text == [list(range(0, 5)), list(range(5, 10)), list(range(10, 15))]
        assert text_to_image == [0] * 5 + [1] * 5 + [2] * 5

    print("Flickr30K local Parquet mapping assertions passed.")


if __name__ == "__main__":
    main()
