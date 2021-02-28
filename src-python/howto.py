from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np

def main(cfg):

    irisRec = irisRecognition(cfg)

    image_list = []
    filename_list = []
    mask_list = []
    polar_list = []

    # Get the list of images to process
    for filename in glob.glob("../data/*.bmp"):
        print(filename)
        im = Image.open(filename)
        image_list.append(im)
        filename_list.append(os.path.basename(filename))

    # Segmentation, normalization and encoding
    for im,fn in zip(image_list,filename_list):
        
        # segmentation mask: 
        mask = irisRec.segment(im)
        im_mask = Image.fromarray(mask)
        mask_list.append(im_mask)
        
        # circular approximation:
        pupil_xyr, iris_xyr = irisRec.circApprox(mask)

        # cartesian to polar transformation:

        # DEBUG: save selected processing results 
        im_mask.save("../dataProcessed/" + os.path.splitext(fn)[0] + "_seg_CCNet_mask.png")
        imVis = irisRec.segmentVis(im,mask,pupil_xyr,iris_xyr)
        cv2.imwrite("../dataProcessed/" + os.path.splitext(fn)[0] + "_seg_CCNet_vis.png",imVis)

        # human-driven BSIF encoding:



    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))