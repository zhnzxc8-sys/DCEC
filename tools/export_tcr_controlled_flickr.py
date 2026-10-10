import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from PIL import Image

from flickr30k_dataset import load_flickr30k_split
from image_degradations import apply_degradation, get_degradation_metadata


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_token(value) -> str:
    text = str(value)
    return "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in text
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local-parquet",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--degradation",
        required=True,
    )

    parser.add_argument(
        "--severity",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    dataset = load_flickr30k_split(
        split="validation",
        local_parquet=args.local_parquet,
        cache_dir=args.cache_dir,
        streaming=False,
    )

    if len(dataset) != 1014:
        raise RuntimeError(
            f"Expected exactly 1014 validation images, got {len(dataset)}"
        )

    metadata = get_degradation_metadata(
        args.degradation,
        args.severity,
    )

    if args.degradation == "clean":
        tag = "clean"
    else:
        tag = f"{args.degradation}_s{args.severity}"

    condition_root = args.output_root / tag
    image_root = condition_root / "images"
    ann_root = args.output_root / "annotations_val_only"

    image_root.mkdir(parents=True, exist_ok=True)
    ann_root.mkdir(parents=True, exist_ok=True)

    annotations = []
    image_tree_hash = hashlib.sha256()

    print("===== EXPORT START =====")
    print("split       : validation")
    print("condition   :", tag)
    print("seed        :", args.seed)
    print("images      :", len(dataset))
    print("image_root  :", image_root)
    print("ann_root    :", ann_root)

    for image_index, sample in enumerate(dataset):
        if "image" not in sample:
            raise RuntimeError(
                f"Sample {image_index} missing image"
            )

        if "image_id" not in sample:
            raise RuntimeError(
                f"Sample {image_index} missing image_id"
            )

        if "captions" not in sample:
            raise RuntimeError(
                f"Sample {image_index} missing captions"
            )

        captions = list(sample["captions"])

        if len(captions) != 5:
            raise RuntimeError(
                f"Image {image_index} has {len(captions)} captions, expected 5"
            )

        if not all(
            isinstance(c, str) and len(c) > 0
            for c in captions
        ):
            raise RuntimeError(
                f"Image {image_index} has invalid caption"
            )

        source_image = sample["image"].convert("RGB")

        degraded_image = apply_degradation(
            source_image,
            degradation=args.degradation,
            severity=args.severity,
            seed=args.seed,
            image_index=image_index,
        ).convert("RGB")

        image_id = safe_token(sample["image_id"])

        filename = (
            f"{image_index:04d}_{image_id}.png"
        )

        output_path = image_root / filename

        # PNG is deliberate:
        # it stores the already-generated DCEC degraded pixels losslessly.
        # Saving JPEG here would add another compression pass.
        degraded_image.save(
            output_path,
            format="PNG",
        )

        expected_array = np.asarray(degraded_image)

        with Image.open(output_path) as reloaded:
            actual_array = np.asarray(
                reloaded.convert("RGB")
            )

        if not np.array_equal(
            expected_array,
            actual_array,
        ):
            raise RuntimeError(
                f"Pixel round-trip mismatch: {filename}"
            )

        file_hash = sha256_file(output_path)

        image_tree_hash.update(
            f"{filename}:{file_hash}\n".encode("utf-8")
        )

        annotations.append(
            {
                "image": filename,
                "caption": captions,
            }
        )

        if (
            image_index == 0
            or (image_index + 1) % 100 == 0
        ):
            print(
                f"exported {image_index + 1}/{len(dataset)}"
            )

    if len(annotations) != 1014:
        raise RuntimeError(
            "Annotation image count mismatch"
        )

    total_captions = sum(
        len(x["caption"])
        for x in annotations
    )

    if total_captions != 5070:
        raise RuntimeError(
            f"Expected 5070 captions, got {total_captions}"
        )

    annotation_path = (
        ann_root / "flickr30k_test.json"
    )

    annotation_path.write_text(
        json.dumps(
            annotations,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    source_parquet_sha256 = sha256_file(
        args.local_parquet
    )

    annotation_sha256 = sha256_file(
        annotation_path
    )

    manifest = {
        "protocol": "DCEC controlled Flickr30K validation export for TCR",
        "source_split": "validation",
        "final_test_accessed": False,

        "source_parquet": str(
            args.local_parquet
        ),
        "source_parquet_sha256": (
            source_parquet_sha256
        ),

        "number_of_images": 1014,
        "number_of_captions": 5070,
        "captions_per_image": 5,

        "degradation": (
            metadata["degradation"]
        ),
        "severity": metadata["severity"],
        "degradation_parameter_name": (
            metadata["parameter_name"]
        ),
        "degradation_parameter_value": (
            metadata["parameter_value"]
        ),
        "degradation_seed": args.seed,

        "image_root": str(image_root),
        "annotation_root": str(ann_root),
        "annotation_file": str(
            annotation_path
        ),

        "annotation_sha256": (
            annotation_sha256
        ),

        "image_tree_sha256": (
            image_tree_hash.hexdigest()
        ),

        "materialization_format": "PNG",
        "pixel_roundtrip_verified": True,

        "note": (
            "PNG is used only as a lossless container "
            "for the already-generated DCEC degraded RGB pixels."
        ),
    }

    manifest_path = (
        condition_root / "EXPORT_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== EXPORT PASS =====")
    print("images              = 1014")
    print("captions            = 5070")
    print("pixel_roundtrip     = PASS")
    print(
        "annotation_sha256   =",
        annotation_sha256,
    )
    print(
        "image_tree_sha256   =",
        image_tree_hash.hexdigest(),
    )
    print(
        "annotation          =",
        annotation_path,
    )
    print(
        "manifest            =",
        manifest_path,
    )


if __name__ == "__main__":
    main()
