import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import ruamel.yaml as yaml
from transformers import CLIPModel
import models.clip_tta as TCR_CLIP

from data import create_dataset, create_loader

ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs/controlled/i2t/flickr_gaussian_blur_s1_clip_b32.yaml"
OUTPUT = ROOT / "output/controlled_frozen_b32/hf_b32_exact_tcr_loader_gaussian_blur_s1.json"

MODEL_NAME = r"/root/autodl-tmp/hf_cache/models--openai--clip-vit-base-patch32/snapshots/d15b5f29721ca72dac15f8526b284be910de18be"


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

    assert len(ds_image.image) == 1014
    assert len(ds_image.text) == 5070

    device = torch.device("cuda")

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    ).to(device)

    model.eval()

    text_features = torch.zeros(
        len(ds_image.text),
        512,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for text, index in loader_text:
            if isinstance(text, (list, tuple)):
                text = TCR_CLIP.tokenize(list(text), truncate=True)
            text = text.to(device).long()

            out = model.text_model(
                input_ids=text
            )

            feat = model.text_projection(
                out.pooler_output
            )

            feat = F.normalize(
                feat.float(),
                p=2,
                dim=1,
            )

            text_features[index] = feat.cpu()

    image_features = torch.zeros(
        len(ds_image.image),
        512,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for image, index in loader_image:
            image = image.to(
                device=device,
                dtype=torch.float32,
            )

            out = model.vision_model(
                pixel_values=image
            )

            feat = model.visual_projection(
                out.pooler_output
            )

            feat = F.normalize(
                feat.float(),
                p=2,
                dim=1,
            )

            image_features[index] = feat.cpu()

    similarity = image_features @ text_features.T

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
        "method": "HF CLIP B32 with exact TCR loader tensors/tokens",
        "condition": "gaussian_blur_s1",
        "images": 1014,
        "captions": 5070,
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

    print("===== EXACT TCR-LOADER RESULT =====")
    print("HF_EXACT_TCR_INPUT_PRIMARY =", metrics["primary"])


if __name__ == "__main__":
    main()
