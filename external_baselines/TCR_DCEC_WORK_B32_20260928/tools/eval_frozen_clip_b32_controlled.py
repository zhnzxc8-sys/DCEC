import json
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import ruamel.yaml as yaml

from data import create_dataset, create_loader
import models.clip_tta as CLIP


CONFIG = "configs/controlled/i2t/flickr_gaussian_blur_s1_clip_b32.yaml"
OUTPUT = Path(
    "output/controlled_frozen_b32/"
    "frozen_clip_b32_gaussian_blur_s1.json"
)


def recall_at_k(similarity, positives, k):
    ranking = similarity.argsort(dim=1, descending=True)
    hits = 0

    for q, pos in enumerate(positives):
        topk = set(ranking[q, :k].tolist())
        if topk.intersection(set(pos)):
            hits += 1

    return 100.0 * hits / len(positives)


def main():
    config = yaml.load(
        open(CONFIG, "r", encoding="utf-8"),
        Loader=yaml.Loader,
    )

    device = torch.device("cuda")

    print("===== DATASET =====")

    ds_text, ds_image = create_dataset(
        f'retrieval_{config["dataset"]}',
        config,
        resize=224,
    )

    loader_text, loader_image = create_loader(
        [ds_text, ds_image],
        [None, None],
        batch_size=[
            config["batch_size_text"],
            config["batch_size_image"],
        ],
        num_workers=[1, 1],
        is_trains=[False, False],
        collate_fns=[None, None],
        config=config,
    )

    print("images   =", len(ds_image.image))
    print("captions =", len(ds_image.text))

    assert len(ds_image.image) == 1014
    assert len(ds_image.text) == 5070

    print()
    print("===== LOAD OFFICIAL TCR CLIP ViT-B/32 =====")

    model = CLIP.clip_retrieval(
        device=device
    )

    model.eval()

    args = SimpleNamespace(
        distributed=False
    )

    print()
    print("===== COMPUTE TEXT EMBEDDINGS =====")

    with torch.no_grad():
        text_features = CLIP.get_text_embeds_clip(
            loader_text,
            model,
            device,
            args,
        )

        print()
        print("===== COMPUTE IMAGE EMBEDDINGS =====")

        image_features = CLIP.get_image_embeds_clip(
            loader_image,
            model,
            device,
            args,
        )

    assert image_features.shape == (1014, 512)
    assert text_features.shape == (5070, 512)

    similarity = (
        image_features.float()
        @ text_features.float().T
    ).cpu()

    image_to_text = [
        list(ds_image.img2txt[i])
        for i in range(1014)
    ]

    text_to_image = []

    for j in range(5070):
        value = ds_image.txt2img[j]

        if isinstance(value, int):
            text_to_image.append([value])
        else:
            text_to_image.append(list(value))

    metrics = {}

    for k in (1, 5, 10):
        metrics[f"i2t_R@{k}"] = recall_at_k(
            similarity,
            image_to_text,
            k,
        )

        metrics[f"t2i_R@{k}"] = recall_at_k(
            similarity.T,
            text_to_image,
            k,
        )

    metrics["primary"] = 0.5 * (
        metrics["i2t_R@1"]
        + metrics["t2i_R@1"]
    )

    result = {
        "method": "Frozen OpenAI CLIP ViT-B/32 via official TCR code",
        "condition": "gaussian_blur_s1",
        "source_split": "Flickr30K validation",
        "number_of_images": 1014,
        "number_of_captions": 5070,
        "final_test_accessed": False,
        "metrics": metrics,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== FROZEN CLIP-B32 RESULT =====")

    for key, value in metrics.items():
        print(
            f"{key:10s} = {value:.6f}"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
