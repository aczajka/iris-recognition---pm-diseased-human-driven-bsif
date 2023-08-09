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

    # Get the image pair
    filename_list = []
    filename_list.append("0002_L_2_1.bmp")
    filename_list.append("0002_L_3_1.bmp")
    image_list = []

    for fn in filename_list:
        im = Image.open("../data/" + fn)
        image_list.append(im)
    
    # Segmentation, normalization and encoding
    polar_mask_list = []
    code_list = []
    pupil_xyr_list = []
    iris_xyr_list = []
    for im,fn in zip(image_list,filename_list):
        
        print(fn)

        # segmentation mask and circular approximation:
        mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
        im_mask = Image.fromarray(np.where(mask > 0.5, 255, 0).astype(np.uint8), 'L')
        
        pupil_xyr_list.append(pupil_xyr)
        iris_xyr_list.append(iris_xyr)

        # cartesian to polar transformation:
        im_polar, mask_polar = irisRec.cartToPol_torch(im, mask, pupil_xyr, iris_xyr)
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
        
    # Matching
    score, shift = irisRec.matchCodes(code_list[0], code_list[1], polar_mask_list[0], polar_mask_list[1])
    print(score)

    # Visualization: heat map (most similar regions in red, least similar regions in blue)
    imVis = irisRec.visualizeMatchingResult(code_list[0], code_list[1], polar_mask_list[0], polar_mask_list[1], shift, image_list[0], pupil_xyr_list[0], iris_xyr_list[0])
    cv2.imwrite("../dataProcessed/" + filename_list[0] + "_HeatMapVis.png",imVis)

    imVis = irisRec.visualizeMatchingResult(code_list[0], code_list[1], polar_mask_list[0], polar_mask_list[1], shift, image_list[1], pupil_xyr_list[1], iris_xyr_list[1])
    cv2.imwrite("../dataProcessed/" + filename_list[1] + "_HeatMapVis.png",imVis)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))