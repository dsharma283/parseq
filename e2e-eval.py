import argparse
import os, sys
import tqdm


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', '-g', required=True,
                        help="Path to ground truth in txt format")
    parser.add_argument('--predicted', '-p', required=True,
                        help='Path to predicted text from model')
    parser.add_argument('--output', '-o', required=False, default='./output',
                        help='The evaluation output will be written to this path')
    return parser


def process_files(path, pos, prefix=''):
    imgs = {}
    flist = os.listdir(path)
    flist = [item for item in flist if item.endswith('.txt')]
    pbar = tqdm.tqdm(flist)
    for idx, fname in enumerate(pbar):
        pbar.set_postfix_str(fname)
        fn = os.path.join(path, fname)
        with open(fn, 'r') as in_f:
            lines = in_f.readlines()
        bbs = {}
        for lid, line in enumerate(lines):
            line = line.strip('\n').split(',')
            word = line[pos]
            if pos == -3:
                lid = line[-1]
            bbs[lid] = word
        fname = fname.split('.')[0]
        if prefix != '':
            fname = fname.strip(prefix)
        imgs[fname] = dict(sorted(bbs.items(), key=lambda v: int(v[0])))
    return imgs


def process_gts(args):
    gts = None
    if not os.path.exists(args.ground_truth):
        return None
    gtpath = args.ground_truth
    gts = process_files(gtpath, pos=-1, prefix='res_')
    return gts


def process_preds(args):
    preds = None
    if not os.path.exists(args.predicted):
        return preds
    ppath = args.predicted
    preds = process_files(ppath, pos=-3)
    return preds


def evaluate(gts, preds):
    print(gts['image_1271'])
    print(preds['image_1271'])
    return None


def save_output(args, res):
    pass


def start_main():
    args = process_args().parse_args()
    gts = process_gts(args)
    preds = process_preds(args)
    res = evaluate(gts, preds)
    save_output(args, res)


if __name__ == '__main__':
    start_main()
