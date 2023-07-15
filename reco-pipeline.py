'''
    Author: Devesh Sharma
    Email: sharma.98@iitj.ac.in
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
import numpy as np
from recognise import (charsets, get_charset, handle_paths,
                       load_and_update_model, process_args,
                       save_output)


device = "cpu"
if torch.cuda.is_available() is True:
    device = "cuda"


def process_args_extended():
    parser = process_args()
    parser.add_argument('--data', '-d', help="Data path", required=False)
    return parser


def is_detection_result_file(imname):
    tokens = imname.split('.')
    is_txt = tokens[-1] == 'txt'
    if is_txt:
        return True
    tokens = tokens[0].split('_')
    return tokens[-1] == 'res' and is_txt is False


def convert_to_pthw(bb, reshape_only=False):
    coor = np.reshape(bb, (4, 2))
    if reshape_only:
        return coor

    pt = coor[0]
    w = coor[1][0] - coor[0][0]
    h = coor[3][1] - coor[0][1]
    return np.asarray([pt, w, h])


def read_bbfile(bbfile):
    bblist = []
    with open(bbfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip('\n').split(',')[:-1]
        line = np.asarray([int(x) for x in line])
        #line = convert_to_pthw(line, reshape_only=True)
        bblist.append(line)
    return bblist


def recognise_one(model, im_descr, transform):
    prediction = []
    im = im_descr[0]
    imname = im_descr[1]
    bblist = im_descr[2]
    for idx, bb in enumerate(bblist):
        #crop = im.crop((bb[0][0], bb[0][1], bb[2][0], bb[2][1]))
        crop = im.crop((bb[0], bb[1], bb[4], bb[5]))
        crop = transform(crop)
        logits = model(crop.unsqueeze(0).to(device))
        probs = logits.softmax(-1)
        preds, probs = model.tokenizer.decode(probs)
        text = model.charset_adapter(preds[0])
        prediction.append([idx, bb, text])
    return {"image": imname, "prediction": prediction}


def recognise_multiple(model, transform, images):
    predictions = []
    for image in os.listdir(images):
        # Skip DBNet images with BB marked
        if is_detection_result_file(image):
            continue

        bbfile = f"res_%s.txt"%(image.split('.')[0])
        bbfile = os.path.join(images, bbfile)
        if os.path.isfile(bbfile) is False:
            print(f"bb descriptor {bbfile} not found, skipped recognition")
            continue
        print(f"{image};{bbfile.split('/')[-1]}")
        imp = os.path.join(images, image)
        try:
            img = Image.open(imp).convert('RGB')
        except:
            continue
        bblist = read_bbfile(bbfile)
        im_descr = (img, image, bblist)
        preds = recognise_one(model, im_descr, transform)
        predictions.append(preds)
    return predictions


def start_main():
    args = process_args_extended().parse_args()
    fname = handle_paths(args)
    model, xform = load_and_update_model(args.checkpoint, args.language)
    results = recognise_multiple(model, xform, args.images)
    save_output(fname, results, new_format=True)


if __name__ == '__main__':
    start_main()

