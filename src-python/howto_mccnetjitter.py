from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np

def main(cfg):

    irisRec = irisRecognition(cfg, use_hough=False)

    # Get the list of images to process
    filename_list = []
    image_list = []
    extensions = ["bmp", "png", "gif", "jpg", "jpeg", "tiff", "tif"]
    for ext in extensions:
        for filename in glob.glob("../data/*." + ext):
            im = Image.open(filename)
            image_list.append(im)
            filename_list.append(os.path.basename(filename))

    # Segmentation, normalization and encoding
    polar_masks_list = []
    codes_list = []
    for im,fn in zip(image_list,filename_list):
        
        print(fn)

        # convert to ISO-compliant aspect ratio (4:3)
        im = irisRec.fix_image(im)

        # segmentation mask and circular approximation:
        mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
        im_mask = Image.fromarray(np.where(mask > 0.5, 255, 0).astype(np.uint8), 'L')

        # cartesian to polar transformation:
        images_polar, masks_polar = irisRec.cartToPol_torch_jitter(im, mask, pupil_xyr, iris_xyr)
        polar_masks_list.append(masks_polar)

        # human-driven BSIF encoding:
        codes = irisRec.extractMultipleCodes(images_polar)
        #print(code.shape)
        codes_list.append(codes)

    # Matching (all-vs-all, as an example)
    for codes1,masks1,fn1,i in zip(codes_list,polar_masks_list,filename_list,range(len(codes_list))):
        for codes2,masks2,fn2,j in zip(codes_list,polar_masks_list,filename_list,range(len(codes_list))):
            if i < j:
                score = irisRec.matchCodesJitter(codes1, codes2[int((len(codes2)-1)/2)], masks1, masks2[int((len(codes2)-1)/2)])
                print("{} <-> {} : {:.3f}".format(fn1,fn2,score))
     
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))