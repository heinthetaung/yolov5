import argparse
import json
import glob
import os
import sys
from pathlib import Path
from multiprocessing.pool import Pool, ThreadPool
from tqdm import tqdm
from itertools import repeat
import numpy as np
import torch
from PIL import Image

BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (print_args, LOGGER, NUM_THREADS, xywhn2xyxy, xyxy2xywhn)
from utils.metrics import bbox_ioa
from utils.dataloaders import (img2label_paths, verify_image_label)

prefix=''
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

#start_path = '/home/xyz/Documents/learning/machine-learning/datasets/Vis/VisDrone2019-DET-val/images'
start_path = '/home/xyz/Documents/learning/machine-learning/datasets/Vis/VisDrone2019-DET-train/images'
# start_path = '/home/xyz/Documents/learning/machine-learning/datasets/Vis/VisDrone2019-DET-test-dev/images'

path = Path(start_path)
images = list(path.glob('*.jpg'))
images.sort()

def split_img(im, labels, split_count=1):

    h, w = im.shape[:2]
    H = int(h / split_count)
    W = int(w / split_count)
    im_arr = []
    labels_arr = []
    for x in range(0,w,W):
        for y in range(0,h,H):
            im_arr.append(im[y:y+H,x:x+W])

            # split labels
            box = np.array([x, y, x+W, y+H], dtype=np.float32)
            ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
            new_labels = labels[ioa > 0.1]  # take labels with >60% intersection
            tmp_labels = xywhn2xyxy(new_labels[:, 1:5], w, h) # at original image size
            tmp_labels[:, 0] = tmp_labels[:, 0] - x
            tmp_labels[:, 2] = tmp_labels[:, 2] - x
            tmp_labels[:, 1] = tmp_labels[:, 1] - y
            tmp_labels[:, 3] = tmp_labels[:, 3] - y
            new_labels[:, 1:5] = xyxy2xywhn(tmp_labels, W, H) # normalize to new image size
            labels_arr.append(new_labels)

    return im_arr, labels_arr

def img2angle_labels_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}angle_labels{os.sep}'  # /images/, /angle_labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--split-count', type=int, default=1, help='split count')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise FileNotFoundError(f'{prefix}{p} does not exist')
        im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert im_files, f'{prefix}No images found'
    except Exception as e:
        raise Exception(f'{prefix}Error loading data from {path}: {e}\n')

    label_files = img2label_paths(im_files)
    x = {}  # dict
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
    with Pool(NUM_THREADS) as pool:
        pbar = tqdm(pool.imap(verify_image_label, zip(im_files, label_files, repeat(prefix))),
                    desc=desc,
                    total=len(im_files),
                    bar_format=BAR_FORMAT)
        for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if im_file:
                x[im_file] = [lb, shape, segments]
            if msg:
                msgs.append(msg)
            pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

    pbar.close()
    if msgs:
        LOGGER.info('\n'.join(msgs))
    if nf == 0:
        LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}.')
    x['image'] = im_files
    x['label'] = label_files
    x['results'] = nf, nm, ne, nc, len(im_files)
    x['msgs'] = msgs  # warnings

    im_target = path.parent.parent / (str(path.parent.stem) +'_generated') / 'images'
    label_target = path.parent.parent / (str(path.parent.stem) +'_generated') / 'labels'
    for im_file in im_files:
        img = Image.open(im_file)
        im = np.array(img)

        angle_label_file = img2angle_labels_paths([im_file])
        angle_label = np.loadtxt(angle_label_file[0], delimiter=',')
        label = x[im_file][0]
        angle, height = angle_label
        if height >= 7:
            split_count = 5
        elif height <= 7 and height >= 3:
            split_count = 2
        elif height < 3:
            split_count = 1

        im_arr, label_arr = split_img(im, label, split_count=split_count)
        
        Path(im_target).mkdir(parents=True, exist_ok=True)
        Path(label_target).mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(im_arr):
            im_fname = (im_target / (str(Path(im_file).stem) + '_' + str(i))).with_suffix('.jpg')
            label_fname = (label_target / (str(Path(im_file).stem) + '_' + str(i))).with_suffix('.txt')
            out_img = Image.fromarray(im)
            out_img.save(im_fname)
            np.savetxt(label_fname, label_arr[i], fmt='%d %.6f %.6f %.6f %.6f')
    print('end of program')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)