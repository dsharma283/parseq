import os, sys, tqdm, argparse, json


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        help='path to gt annotations in json format')
    parser.add_argument('--bbfiles', '-b', required=True,
                        help='')
    parser.add_argument('--output', '-o', required=False, default=f'./results',
                        help='Output directory to save the predictions')
    return parser


def handle_paths(args):
    if not os.path.exists(args.input):
        print(f'{args.input} path not found')
        return False
    if not os.path.exists(args.bbfiles):
        print(f'{args.bbfiles} path not found')
        return False
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    return True


def process_json(args):
    with open(args.input, 'r') as inf:
        data = json.load(inf)
    pbar = tqdm.tqdm(data['files'])
    for idx, entry in enumerate(pbar):
        imname = entry['filename'].
        pbar.set_postfix_str(entry['filename'])


def start_main():
    args = process_args().parse_args()
    if not handle_paths(args):
        exit(-1)
    process_json(args)
    #save_textfiles(args)


if __name__ == "__main__":
    start_main()

