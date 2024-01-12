import argparse, json
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
    parser.add_argument('--jsonpath', '-j', required=False, default=None,
                        help='Json file path for annotations')
    return parser


def process_json(path):
    with open(path, 'r') as inf:
        data = json.load(inf)
    return data


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
    fgts, jgts = None, None
    if not os.path.exists(args.ground_truth):
        return None
    gtpath = args.ground_truth
    if args.jsonpath is None:
        fgts = process_files(gtpath, pos=-3, prefix='res_')
    else:
        jgts = process_json(args.jsonpath)
        fgts = process_files(gtpath, pos=-3, prefix='res_')
    return fgts, jgts


def process_preds(args):
    preds = None
    if not os.path.exists(args.predicted):
        return preds
    ppath = args.predicted
    preds = process_files(ppath, pos=-3)
    return preds


def search_gts(jgts, fgts, jgtkey, fgtkey, bbid):
    gtword = None
    if jgts != None:
        gt = [item for item in jgts['files'] if item['filename'] == jgtkey]
        if len(gt) == 0:
            gtword = None
        else:
            gtword = gt[0]['text'].strip(' ').strip('/').strip('.').strip('-').lower()
    if fgts != None and gtword is None:
        gtword = fgts[fgtkey].get(bbid)
        if not gtword is None:
            gtword = gtword.strip(' ').strip('/').strip('.').strip('-').lower()

    return gtword


def evaluate(gts, preds):
    gt_count, correct, fail, detected = 0, 0, 0, 0
    fgts, jgts = gts[0], gts[1]

    pbar = tqdm.tqdm(preds.keys())
    for idx, key in enumerate(pbar):
        pbar.set_postfix_str(key)
        for bbid in preds[key].keys():
            jgtkey = f'{key}_{bbid}.jpg'
            fgtkey = key 
            pred = preds[key][bbid]
            #if not pred == '':
            detected += 1
            gtword = search_gts(jgts, fgts, jgtkey, fgtkey, bbid)
            if gtword == None:
                continue
            gt_count += 1
            if pred == gtword:
                correct += 1
            else:
                fail += 1
    recall = correct / gt_count
    precision = correct / detected
    fscore = 2 * precision * recall / (precision + recall)

    print(f'predicted word count:\t\t{detected}')
    print(f'GT word count:\t\t\t{gt_count}')
    print(f'Correct prediction count:\t{correct}')
    print(f'Failed prediction count:\t{fail}')
    print(f'Precision:\t\t\t{precision}')
    print(f'Recall:\t\t\t\t{recall}')
    print(f'Fscore:\t\t\t\t{fscore}')
    return None


'''
def save_output(args, res):
    pass
'''


def start_main():
    args = process_args().parse_args()
    gts = process_gts(args)
    preds = process_preds(args)
    _ = evaluate(gts, preds)


if __name__ == '__main__':
    start_main()

