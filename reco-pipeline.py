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
    parser.add_argument('--skip-unknown', '-U', required=False, action='store_true',
                        help="Skip saving the recognition taged as Unknown", default=False)
    parser.add_argument('--force-unknown', '-F', required=False, action='store_true', default=False,
                        help='''Force the processing of Unknows through multiple '''
                             '''recognizers and select the heigest confident prediction''')
    parser.add_argument('--with-conf', '-C', required=False, action='store_true', default=False,
                        help="Generated the prediction output with confidence in .conf file")
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
    return text


def handle_unknown_forced(crop, cnf, classmap, ckpt_path):
    topk = np.argpartition(cnf, -2)[-2:]
    tlang = classmap[topk[0]]
    model, xform = load_and_update_model(ckpt_path, tlang)
    return predict_text(model, xform, crop)


def recognise_one(model, im_descr, transform):
    prediction = []
    im = im_descr[0]
    imname = im_descr[1]
    bblist = im_descr[2]
    for idx, bb in enumerate(bblist):
        try:
            crop = generate_crop(im, bb)
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
        try:
            crop = generate_crop(im, bb)
        except:
            crop = None
            print(f'invalid cropping points {imname}-{idx}')
        if args.save_crops is True and crop is not None:
            imgn = imname.split('.')[0]
            save_crop(args.output, imgn, crop, idx)
        crops.append(crop)
    scriptids, classmap = identify_script(modpath, crops)

    for key in scriptids.keys():
        if key == 'unknown' and args.skip_unknown is True:
            continue

        if key != 'unknown':
            model, transform = load_and_update_model(args.checkpoint, key)

        for idx, entry in enumerate(scriptids[key]):
            bbid = next(iter(entry))
            bb = bblist[bbid]
            conf = entry[bbid]
            text = ''
            try:
                crop = generate_crop(im, bb)
            except:
                crop = None
            if key != 'unknown':
                text = predict_text(model, transform, crop)
            elif args.force_unknown and crop:
                text = handle_unknown_forced(crop, conf, classmap, args.checkpoint)
            prediction.append([bbid, bb, text, key, conf])
    return {"image": imname, "prediction": sorted(prediction, key=lambda x : int(x[0]))}


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
        save_output(fname, [preds], new_format=True,
                    withconf=args.with_conf)
        predictions.append(preds)
    return predictions


def start_main():
    args = process_args_extended().parse_args()
    fname = handle_paths(args, args.with_conf)
    results = recognise_multiple(args, fname)


if __name__ == '__main__':
    start_main()

