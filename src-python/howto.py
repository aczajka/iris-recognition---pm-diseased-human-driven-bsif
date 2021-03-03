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
        im_polar, mask_polar = irisRec.cartToPol(im, mask, pupil_xyr, iris_xyr)
        polar_mask_list.append(mask_polar)

        # human-driven BSIF encoding:
        code = irisRec.extractCode(im_polar)
        code_list.append(code)

        # DEBUG: save selected processing results 
        im_mask.save("../dataProcessed/" + os.path.splitext(fn)[0] + "_seg_CCNet_mask.png")
        imVis = irisRec.segmentVis(im,mask,pupil_xyr,iris_xyr)
        cv2.imwrite("../dataProcessed/" + os.path.splitext(fn)[0] + "_seg_CCNet_vis.png",imVis)
        cv2.imwrite("../dataProcessed/" + os.path.splitext(fn)[0] + "_im_polar_CCNet.png",im_polar)
        cv2.imwrite("../dataProcessed/" + os.path.splitext(fn)[0] + "_mask_polar_CCNet.png",mask_polar)
        np.savez_compressed("./templates/" + os.path.splitext(fn)[0] + "_tmpl.npz",code)
        
    # Matching (all-vs-all, as an example)
    for code1,mask1,fn1,i in zip(code_list,polar_mask_list,filename_list,range(len(code_list))):
        for code2,mask2,fn2,j in zip(code_list,polar_mask_list,filename_list,range(len(code_list))):
            if i < j:
                score, shift = irisRec.matchCodes(code1, code2, mask1, mask2)
                print("{} <-> {} : {:.3f} (mutual rot: {:.2f} deg)".format(fn1,fn2,score,360*shift/irisRec.polar_width))

                b = irisRec.visualizeMatchingResult(code1, code2, mask1, mask2, shift)
                cv2.imwrite("../dataProcessed/" + os.path.splitext(fn1)[0] + "_" + os.path.splitext(fn2)[0] + "_b.png",b)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))