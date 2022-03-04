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

    # Get the list of images to process
    filename_list = []
    image_list = []
    for filename in glob.glob("../data/*.bmp"):
        im = Image.open(filename)
        image_list.append(im)
        filename_list.append(os.path.basename(filename))

    # Segmentation, normalization and encoding
    polar_mask_list = []
    code_list = []
    for im,fn in zip(image_list,filename_list):
        
        print(fn)

        # segmentation mask: 
        mask = irisRec.segment(im)
        im_mask = Image.fromarray(mask)
        
        # circular approximation:
        pupil_xyr, iris_xyr = irisRec.circApprox(mask)

        # cartesian to polar transformation:
        im_polar, mask_polar = irisRec.cartToPol_prev(im, mask, pupil_xyr, iris_xyr)
        im_polar2, mask_polar2 = irisRec.cartToPol(im, mask, pupil_xyr, iris_xyr)

        ims_polar = np.concatenate([im_polar, im_polar2], axis=0)
        masks_polar = np.concatenate([mask_polar, mask_polar2], axis=0)

        cv2.imwrite('cartToPol_samples/im_polar_sample_'+fn, ims_polar)
        cv2.imwrite('cartToPol_samples/im_mask_sample_'+fn, masks_polar)
        
     
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))