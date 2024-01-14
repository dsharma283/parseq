'''
    Author: Devesh Sharma
    Email: sharma.98@iitj.ac.in
'''
import argparse
import string
import os, cv2
from typing import Tuple
import torch
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from torchvision import transforms as T
from PIL import Image
import numpy as np
from recognise import (charsets, get_charset, handle_paths,
                       load_and_update_model, process_args)
from recopipeline import predict_text, generate_crop

device = "cpu"
if torch.cuda.is_available() is True:
    device = "cuda"


def process_args_extended():
    parser = process_args()
    parser.add_argument('--save-crops', '-S', required=False, action='store_true',
                        help="Enable saving the cropped bounding boxes", default=False)
    return parser


def save_output(fname, results):
    prefix = fname[1]
    for result in results:
        imname = result['image'].split('.')[0] + '.txt'
        oppath = os.path.join(prefix, imname)
        with open(oppath, 'w') as of:
            for pred in result['prediction']:
                cnf = ','.join(map(str, pred[3]))
                s = f"{cnf},{pred[2]},{pred[0]}\n"
                of.writelines(s)


def is_detection_result_file(imname):
    tokens = imname.split('.')
    is_txt = tokens[-1] == 'txt'
    if is_txt:
        return True
    tokens = tokens[0].split('_')
    return tokens[-1] == 'res' and is_txt is False


def read_bbfile(bbfile):
    bblist = []
    with open(bbfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip('\n').split(',')[:-1]
        line = np.asarray([int(x) for x in line])
        bblist.append(line)
    return bblist


def generate_crop(im, bb, is_poly=False):
    bb = np.reshape(bb, (-1, 2))
    rect = cv2.boundingRect(bb)
    x, y, w, h = rect
    croped = np.array(im)[y:y+h, x:x+w]

    bb = bb - bb.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [bb], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    return Image.fromarray(dst)


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
    return text, probs[0].detach().to('cpu').numpy()


def recognise_one(args, model, im_descr, transform):
    prediction = []
    im = im_descr[0]
    imname = im_descr[1]
    bblist = im_descr[2]
    for idx, bb in enumerate(bblist):
        try:
            crop = generate_crop(im, bb)
        except:
            print(f'invalid cropping points {imname}-{idx}')
            continue
        if args.save_crops:
            imgn = imname.split('.')[0]
            save_crop(args.output, imgn, crop, idx)
        text, probs = predict_text(model, transform, crop)
        prediction.append([idx, bb, text, probs])
    return {"image": imname,
            "prediction": sorted(prediction, key=lambda x : int(x[0]))}


def recognise_multiple(args, fname):
    predictions = []
    model, transform = None, None

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
        preds = recognise_one(args, model, im_descr, transform)
        save_output(fname, [preds])
        predictions.append(preds)
    return predictions


def start_main():
    args = process_args_extended().parse_args()
    fname = handle_paths(args)
    results = recognise_multiple(args, fname)


if __name__ == '__main__':
    start_main()

