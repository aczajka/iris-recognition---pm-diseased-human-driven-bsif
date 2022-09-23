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
    for filename in glob.glob("../data/*.bmp"):
        im = Image.open(filename)
        image_list.append(im)
        filename_list.append(os.path.basename(filename))

    # Segmentation, normalization and encoding
    polar_mask_list = []
    code_list = []
    for im,fn in zip(image_list,filename_list):
        
        print(fn)

        # segmentation mask and circular approximation:
        mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
        im_mask = Image.fromarray(mask)

        # cartesian to polar transformation:
        im_polar, mask_polar = irisRec.cartToPol(im, mask, pupil_xyr, iris_xyr)
        polar_mask_list.append(mask_polar)

        # human-driven BSIF encoding:
        code = irisRec.extractCode(im_polar)
        code_list.append(code)

        # DEBUG: save selected processing results
        im_mask.save("../dataProcessed/" + os.path.splitext(fn)[0] + "_seg_CCNet_mask.png")
        imVis = irisRec.segmentVis(im,mask,pupil_xyr,iris_xyr)
        path = "../dataProcessed/" + os.path.splitext(fn)[0]
        cv2.imwrite(path + "_seg_CCNet_vis.png",imVis)
        cv2.imwrite(path + "_im_polar_CCNet.png",im_polar)
        cv2.imwrite(path + "_mask_polar_CCNet.png",mask_polar)
        np.savez_compressed("./templates/" + os.path.splitext(fn)[0] + "_tmpl.npz",code)
        for i in range(irisRec.num_filters):
            cv2.imwrite(("%s_code_filter%d.png" % (path,i)),255*code[:,:,i])

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