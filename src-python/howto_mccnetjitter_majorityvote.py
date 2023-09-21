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
    polar_mask_list = []
    code_list = []
    for im,fn in zip(image_list,filename_list):
        
        print(fn)

        # convert to ISO-compliant aspect ratio (4:3)
        im = irisRec.fix_image(im)

        # segmentation mask and circular approximation:
        mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
        im_mask = Image.fromarray(np.where(mask > 0.5, 255, 0).astype(np.uint8), 'L')

        # cartesian to polar transformation:
        images_polar, masks_polar = irisRec.cartToPol_torch_jitter(im, mask, pupil_xyr, iris_xyr)

        path = "../dataProcessed/" + os.path.splitext(fn)[0]

        im_mask.save("../dataProcessed/" + os.path.splitext(fn)[0] + "_seg_MCCNet_mask.png")

        for i, (im_polar, mask_polar) in enumerate(zip(images_polar, masks_polar)):
            cv2.imwrite("../dataProcessed/im_polar_jitter/" + os.path.splitext(fn)[0] + "_im_polar_MCCNet_jitter_" + str(i) + ".png",im_polar)
            cv2.imwrite("../dataProcessed/mask_polar_jitter/" + os.path.splitext(fn)[0] + "_mask_polar_MCCNet_jitter_" + str(i) + ".png",mask_polar)

        # human-driven BSIF encoding:
        codes = irisRec.extractMultipleCodes(images_polar)

        majorityVoteCode, mask_polar = irisRec.findMajorityVoteCode(codes, masks_polar)
        polar_mask_list.append(mask_polar)

        code_list.append(majorityVoteCode)

        # DEBUG: save selected processing results
        
        cv2.imwrite(path + "_mask_polar_MCCNet_majvote.png", (mask_polar.astype(int) * 255).astype(np.uint8))

    # Matching (all-vs-all, as an example)
    for code1,mask1,fn1,i in zip(code_list,polar_mask_list,filename_list,range(len(code_list))):
        for code2,mask2,fn2,j in zip(code_list,polar_mask_list,filename_list,range(len(code_list))):
            if i < j:
                score, shift = irisRec.matchCodes(code1, code2, mask1, mask2)
                print("{} <-> {} : {:.3f} (mutual rot: {:.2f} deg)".format(fn1,fn2,score,360*shift/irisRec.polar_width))
     
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))