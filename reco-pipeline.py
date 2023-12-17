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
from predict_crops import identify_script

device = "cpu"
if torch.cuda.is_available() is True:
    device = "cuda"


def process_args_extended():
    parser = process_args()
    parser.add_argument('--with-scriptid', '-s', required=False, action='store_true',
                        help="run the prediction with script identification", default=False)
    parser.add_argument('--scriptid-mod-path', '-m', required=False, type=str,
                        help="Script identification pretrained model path",
                        default='./checkpoints')
    parser.add_argument('--save-crops', '-S', required=False, action='store_true',
                        help="Enable saving the cropped bounding boxes", default=False)
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


def save_crop(output, imname, crop, bbid):
    base = os.path.join(output, 'crops', imname)
    if os.path.isdir(base) is False:
        os.makedirs(base)
    fname = os.path.join(base, f'{imname}_{bbid}.jpg')
    crop.resize((224, 224)).save(fname)


def predict_text(model, xform, crop):
    logits = model(xform(crop).unsqueeze(0).to(device))
    probs = logits.softmax(-1)
    preds, probs = model.tokenizer.decode(probs)
    text = model.charset_adapter(preds[0])
    return text


def recognise_one(model, im_descr, transform):
    prediction = []
    im = im_descr[0]
    imname = im_descr[1]
    bblist = im_descr[2]
    for idx, bb in enumerate(bblist):
        #crop = im.crop((bb[0][0], bb[0][1], bb[2][0], bb[2][1]))
        try:
            crop = im.crop((bb[0], bb[1], bb[4], bb[5]))
        except:
            continue
        text = predict_text(model, transform, crop)
        prediction.append([idx, bb, text])
    return {"image": imname, "prediction": prediction}


def recognise_one_with_scriptid(args, im_descr):
    im, imname, bblist = im_descr[0], im_descr[1], im_descr[2]
    modpath = args.scriptid_mod_path
    crops, prediction = [], []

    for idx, bb in enumerate(bblist):
        crop = im.crop((bb[0], bb[1], bb[4], bb[5]))
        if args.save_crops is True:
            imgn = imname.split('.')[0]
            save_crop(args.output, imgn, crop, idx)
        crops.append(crop)
    scriptids = identify_script(modpath, crops)

    for key in scriptids.keys():
        if key != 'unknown':
            model, transform = load_and_update_model(args.checkpoint, key)

        for bbid in scriptids[key]:
            bb = bblist[bbid]
            text = ''
            if key != 'unknown':
                crop = im.crop((bb[0], bb[1], bb[4], bb[5]))
                text = predict_text(model, transform, crop)
            prediction.append([bbid, bb, text, key])
    return {"image": imname, "prediction": prediction}


def recognise_multiple(args, fname):
    predictions = []
    model, transform = None, None

    if args.with_scriptid is False:
        model, transform = load_and_update_model(args.checkpoint, args.language)
    images = args.images

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
        if args.with_scriptid is False:
            preds = recognise_one(model, im_descr, transform)
        else:
            preds = recognise_one_with_scriptid(args, im_descr)
        #print(preds)
        save_output(fname, [preds], new_format=True)
        predictions.append(preds)
    return predictions


def start_main():
    args = process_args_extended().parse_args()
    fname = handle_paths(args)
    results = recognise_multiple(args, fname)


if __name__ == '__main__':
    start_main()

