import cv2
import numpy as np
from pathlib import Path
import os
import csv

def add_angle_data(img, angle, height):
    range_set = [(0, 0), (10, 0), (20, 10),
                (127, 1), (160, 90), (200, 160),
                (220, 127), (250, 150) ,(255, 255)]
    range = range_set[int(angle)-1]
    # range = (127, 1)
    img_height = img.shape[0]
    img_width = img.shape[1]
    step = (range[0]-range[1])/(img_height-1)
    img += range[1]
    for index, i in enumerate(img):
        i += round(step*(index))
    return img

img_sample1 = '../datasets/VisDrone/VisDrone2019-DET-val/images'
img_sample2 = '../datasets/VisDrone/VisDrone2019-DET-train/images'
img_sample3 = '../datasets/VisDrone/VisDrone2019-DET-test-dev/images'
img_samples = [img_sample1 img_sample2 img_sample3]
for img_sample in img_samples:
    p = Path(img_sample)
    img_files = [x for x in p.glob("*.jpg") if x.is_file()]

    depth_folder = p.parent / 'depth'
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)
        
    for f in img_files:
        angle_labels = (p.parent / 'angle_labels' / f.stem).with_suffix('.txt')
        with open(angle_labels, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            row = next(csv_reader)
        print(f)
        # print(angle_labels)
        img = cv2.imread(str(f))
        depth_img = np.zeros((img.shape[0], img.shape[1]))
        depth_img = add_angle_data(depth_img, row[0], row[1])
        depth_img = np.uint8(depth_img)
        cv2.imwrite(str((depth_folder / f.stem).with_suffix(".png")), depth_img)


# # reference
# # https://stackoverflow.com/questions/44606257/imwrite-16-bit-png-depth-image
# # https://en.wikipedia.org/wiki/Portable_Network_Graphics#Pixel_format