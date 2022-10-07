import cv2
import numpy as np
from pathlib import Path
import os

def add_angle_data(img):
    return img

img_sample1 = '/home/xyz/Documents/learning/machine-learning/datasets/VisDrone/VisDrone2019-DET-val/images'
img_sample2 = '/home/xyz/Documents/learning/machine-learning/datasets/VisDrone/VisDrone2019-DET-train/images'
img_sample3 = '/home/xyz/Documents/learning/machine-learning/datasets/VisDrone/VisDrone2019-DET-test-dev/images'
img_samples = [img_sample1, img_sample2, img_sample3]
for img_sample in img_samples:
    p = Path(img_sample)
    img_files = [x for x in p.glob("*.jpg") if x.is_file()]

    depth_folder = p.parent / 'depth'
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)
        
    for f in img_files:
        img = cv2.imread(str(f))
        depth_img = np.zeros((img.shape[0], img.shape[1]))
        depth_img = add_angle_data(depth_img)
        depth_img = np.uint8(depth_img)
        cv2.imwrite(str((depth_folder / f.stem).with_suffix(".png")), depth_img)


# # reference
# # https://stackoverflow.com/questions/44606257/imwrite-16-bit-png-depth-image
# # https://en.wikipedia.org/wiki/Portable_Network_Graphics#Pixel_format