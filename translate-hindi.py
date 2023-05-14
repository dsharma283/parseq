import argparse
import string
import sys
from dataclasses import dataclass
from typing import List
import torch
from tqdm import tqdm
from parseq.strhub.data.module import SceneTextDataModule
from parseq.strhub.models.utils import load_from_checkpoint
from nltk import edit_distance

device = "cuda"
checkpoint = checkpoint_path
model = load_from_checkpoint(checkpoint).eval().to(device)

def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0):
    transforms = []
    if augment:
        from .augment import rand_augment_transform
        transforms.append(rand_augment_transform())
    if rotation:
        transforms.append(lambda img: img.rotate(rotation, expand=True))
    transforms.extend(
        [T.Resize(img_size, T.InterpolationMode.BICUBIC),
         T.ToTensor(),
         T.Normalize(0.5, 0.5)
         ])
    return T.Compose(transforms)


hp = model.hparams
transform = get_transform(hp.img_size, rotation=0)
model.hparams.charset_test = "अआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसह़ऽािीुूृॄॅॆेैॉॊोौ्ॐ॒॑॓॔ॕॖॠॡॢॣ।॥०१२३४५६७८९॰ॱॲॻॿ"
predictions = []
for image in os.listdir("testing-data/new_crops"):
    image_path = os.path.join("testing-data/new_crops", image)
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    logits = model(img.unsqueeze(0).to(device))
    probs = logits.softmax(-1)
    preds, probs = model.tokenizer.decode(probs)
    text = model.charset_adapter(preds[0])
    predictions.append({"image": image, "prediction": text})
print(predictions)
