import pyvips
from PIL import Image
import numpy as np
import os
import cv2

base_path ="/home/sead/projects/vessel_seg/data/vessmap/"
images_path ="/home/sead/projects/vessel_seg/data/vessmap/images"
annotations_path ="/home/sead/projects/vessel_seg/data/vessmap/annotator1/labels"
output_path = "/home/sead/projects/vessel_seg/data/output_viz"

if not os.path.exists(output_path):
    os.makedirs(output_path)

slides = []
for img in os.listdir(images_path):
    img_name = img[:-5]
    slide = pyvips.Image.new_from_file(os.path.join(images_path, img)).numpy(dtype=np.uint8)
    slide = np.stack((slide,) *3, axis=-1)
    # slide = np.concatenate((slide, np.ones((slide.shape[0], slide.shape[1], 1))) * 255, axis=2)

    mask = cv2.imread(f"{os.path.join(annotations_path, img_name)}.png")
    mask[np.all(mask == np.array([255,255,255]), axis=-1)] = [255, 0, 0]

    viz = cv2.addWeighted(slide, .9, mask, 0.3, 0)

    cv2.imwrite(f"{os.path.join(output_path, img[:-4])}.png", viz)



