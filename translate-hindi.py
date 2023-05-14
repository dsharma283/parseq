import argparse
import string
import os
from typing import Tuple
import torch
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from torchvision import transforms as T
from PIL import Image


device = "cpu"
if torch.cuda.is_available() is True:
    device = "cuda"


charsets = [
    "अआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसह़ऽािीुूृॄॅॆेैॉॊोौ्ॐ॒॑॓॔ॕॖॠॡॢॣ।॥०१२३४५६७८९॰ॱॲॻॿ",
    None
]


def get_charset(lang):
    if lang == 'Devanagari':
        return charsets[0]
    else:
        return charsets[-1]


def save_output(op_path, results):
    if op_path is None:
        print(results)


def get_transform(img_size:Tuple[int], augment:bool = False, rotation:int = 0):
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


def load_and_update_model(chkpt, lang):
    if os.path.exists(chkpt) is False:
        print(f'Specified checkpoint does not exist')
        return None
    model = load_from_checkpoint(chkpt).eval().to(device)
    hp = model.hparams
    transform = get_transform(hp.img_size, rotation=0)
    hp.charset_test = get_charset(lang)
    return model, transform


def predict_results(model, transform, images):
    predictions = []
    for image in os.listdir(images):
        imp = os.path.join(images, image)
        img = Image.open(imp).convert('RGB')
        img = transform(img)
        logits = model(img.unsqueeze(0).to(device))
        probs = logits.softmax(-1)
        preds, probs = model.tokenizer.decode(probs)
        text = model.charset_adapter(preds[0])
        predictions.append({"image": image, "prediction": text})
    #print(predictions)
    return predictions


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', help="Pretrained model path", required=True)
    parser.add_argument('--images', '-i', help='Images to read', required=True)
    parser.add_argument('--output', '-o', help='Output directory to save the predictions', required=False)
    parser.add_argument('--language', '-l',
                        help='The language script in use, supported: Devanagari',
                        default="Devanagari")
    args = parser.parse_args()
    return args


def start_main():
    args = process_args()
    model, xform = load_and_update_model(args.checkpoint, args.language)
    results = predict_results(model, xform, args.images)
    save_output(args.output, results)


if __name__ == '__main__':
    start_main()

