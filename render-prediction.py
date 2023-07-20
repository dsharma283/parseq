'''
    Author: Devesh Sharma
    Email: sharma.98@iitj.ac.in
'''
import argparse
import pygame
import sys, os


def should_skip(imname):
    is_txt = imname.split('.')[-1] == 'txt'
    is_res = imname.split('_')[-1] != 'res.jpg'
    return is_txt or is_res


def setup_screen_sprite(impath):
    img = pygame.image.load(impath)
    rect = img.get_rect()

    surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    screen = pygame.display.set_mode((rect.w, rect.h))

    sprite = pygame.sprite.Sprite()
    sprite.image = img
    sprite.rect = rect
    return screen, sprite, surf


def setup_font(fontname, size):
    mf = pygame.font.Font(fontname, size=size)
    mf.set_script("Deva")
    mf.set_bold(1)
    return mf

def get_words_and_points(predpath):
    words = []
    points = []
    with open(predpath, 'r') as bbf:
        lines = bbf.readlines()
    for line in lines:
        line = line.strip('\n').strip(' ').split(',')
        words.append(line[-1])
        line = [int(x) for x in line[:-1]]
        points.append((line[0], line[1]))
    return words, points


def render_text_on_images(args):
    imdir = args.images
    #txtdir = args.images
    fontname = args.fontname
    savedir = args.output

    pygame.init()
    mf = setup_font(fontname, size=25)
    for imname in os.listdir(imdir):
        if should_skip(imname):
            continue
        impath = os.path.join(imdir, imname)
        tmp = imname.split('.')[0].split('_')
        predname = tmp[0]+ '_' + tmp[1] + '.txt'
        predpath = os.path.join(imdir, predname)
        savename = 'rend-' + imname
        savepath = os.path.join(savedir, savename)
        print(imname, predname)
        words, points = get_words_and_points(predpath)
        screen, sprite, surf = setup_screen_sprite(impath)

        sequence = []
        for word, point in zip(words, points):
            ren = mf.render(word, False, (255, 0, 0))
            sequence.append((ren, (point[0], point[1] - mf.get_height())))
            surf.blits(sequence)
            sprite.image.blit(surf, sprite.rect)
        grp = pygame.sprite.Group()
        grp.add(sprite)
        grp.draw(screen)
        pygame.display.flip()
        pygame.image.save_extended(screen, savepath)
        if args.interactive:
            _ = input("Press enter to continue...")


def handle_paths(args):

    if os.path.exists(args.images) is False:
        print(f'{args.images} not found')
        exit(-1)
    if os.path.exists(args.prediction) is False:
        print(f'{args.prediction} not found')
        exit(-1)
    if os.path.exists(args.fontname) is False:
        print(f'{args.fontname} not found')
        exit(-1)

    if os.path.exists(args.output) is False:
        os.mkdirs(args.output)


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', '-i',
                        help='Scene text detected images with bounding box in-painted', required=True)
    parser.add_argument('--prediction', '-p',
                        help='Scene text recognized text file containing bb,pred', required=True)
    parser.add_argument('--output', '-o',
                        help='Output directory to save the prediction inpainted images', required=True)
    parser.add_argument('--fontname', '-f',
                        help='Font name to used for rendering', required=False,
                        default="./fonts/EkMukta-Regular.ttf")
    parser.add_argument('--language', '-l',
                        help='The language script in use, supported: Devanagari', default="Devanagari")
    parser.add_argument('--interactive', '-w',
                        help='Wait and ask to move forward', default=0, action='store_true')
    return parser


def start_main():
    args = process_args().parse_args()
    handle_paths(args)
    render_text_on_images(args)


if __name__ == '__main__':
    start_main()

