"""Run one CLIP image-text similarity inference without training."""

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
TEXTS = [
    "a photo of a cat",
    "a photo of a car",
    "a photo of a building",
]


def main():
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / "data" / "samples" / "test.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use a GPU when available; otherwise CLIP runs normally on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained CLIP model and its matching image/text processor.
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Prepare one image and the candidate text prompts as model inputs.
    with Image.open(image_path) as image:
        inputs = processor(
            text=TEXTS,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        ).to(device)

    # Disable gradients because this is inference only, not training.
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)[0]

    print(f"Device: {device}")
    for text, probability in zip(TEXTS, probabilities.tolist()):
        print(f"{text}: {probability:.4%}")


if __name__ == "__main__":
    main()
