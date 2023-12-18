import pandas as pd
import os, tqdm
import argparse


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        help='path to recognition new-format output files')
    parser.add_argument('--output', '-o', required=False, default=f'./results',
                        help='Output directory to save the predictions')
    return parser

def generate_datafram(args):
    if os.path.isdir(args.input) is False or os.path.exists(args.input) is False:
        print('Input directory not found')
        return None
    if os.path.exists(args.output) is False:
        os.makedirs(args.output)
    flist = os.listdir(args.input)
    flist = [item for item in flist if item.endswith('.txt')]

    df = pd.DataFrame(columns=['Scene_Image_name', 'Word_image_filename', 'Language', 'Recognition'])
    pbar = tqdm.tqdm(flist)
    for idx, img in enumerate(pbar):
        pbar.set_postfix_str(img)
        base = img.split('.')[0]
        fname = os.path.join(args.input, img)
        with open(fname, 'r') as in_f:
            lines = in_f.readlines()
        for line in lines:
            items = line.strip('\n').split(',')
            simg_fname = base + '.jpg'

            wimg_base = base + f'_{items[-1]}.jpg'
            wimg_fname = os.path.join(args.input, 'crops', base, wimg_base)
            if os.path.exists(wimg_fname) is False:
                wimg_base = ''
                wimg_fname = ''

            lang = items[-2]
            pred = items[-3]
            entry = {
                        'Scene_Image_name': simg_fname,
                        'Word_image_filename': wimg_fname,
                        'Language': lang,
                        'Recognition': pred,
                    }
            tdf = pd.DataFrame(entry, index=[0])
            df = pd.concat([df, tdf], ignore_index=True)
    return df


def generate_csv(args, df):
    of = os.path.join(args.output, 'recognised.csv')
    if os.path.exists(of):
        os.remove(of)
    df.to_csv(of, index=False)


def start_main():
    args = process_args().parse_args()
    df = generate_datafram(args)
    print(df)
    generate_csv(args, df)


if __name__ == '__main__':
    start_main()
