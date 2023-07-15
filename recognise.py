'''
    Author: Devesh Sharma
    email: sharma.98@iitj.ac.in
'''
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


def save_output(fname, results):
    if fname is None:
        for idx, item in enumerate(results):
            print(f"{item['image']}:{item['prediction']}")
        return
    with open(fname, 'a') as of:
        for idx, item in enumerate(results):
            of.writelines(f"{item['image']}:{item['prediction']}\n")


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
        try:
            img = Image.open(imp).convert('RGB')
        except:
            continue
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
    return parser


def handle_paths(args):
    o_path = args.output
    if o_path is None:
        return None

    if os.path.exists(o_path) is False:
        os.makedirs(o_path)
    fname = os.path.join(o_path, "prediction.txt")
    if os.path.isfile(fname) is True:
        os.remove(fname)
    return fname


def start_main():
    args = process_args().parse_args()
    fname = handle_paths(args)
    model, xform = load_and_update_model(args.checkpoint, args.language)
    results = predict_results(model, xform, args.images)
    save_output(fname, results)


if __name__ == '__main__':
    start_main()

