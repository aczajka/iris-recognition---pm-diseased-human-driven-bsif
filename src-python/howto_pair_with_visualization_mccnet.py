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
    pupil_xyr_list = []
    iris_xyr_list = []
    for im,fn in zip(image_list,filename_list):
        
        print(fn)
        
        # convert to ISO-compliant aspect ratio (4:3)
        im = irisRec.fix_image(im)

        # segmentation mask and circular approximation:
        mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
        im_mask = Image.fromarray(mask)
        
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
    '''    
    # Matching
    for code1,mask1,fn1,i in zip(code_list,polar_mask_list,filename_list,range(len(code_list))):
        for code2,mask2,fn2,j in zip(code_list,polar_mask_list,filename_list,range(len(code_list))):
            if i < j:
                score, shift = irisRec.matchCodes(code1, code2, mask1, mask2)
                print("{} <-> {} : {:.3f} (mutual rot: {:.2f} deg)".format(fn1,fn2,score,360*shift/irisRec.polar_width))
                # Visualization: heat map (most similar regions in red, least similar regions in blue)
                imVis = irisRec.visualizeMatchingResult(code_list[i], code_list[j], polar_mask_list[i], polar_mask_list[j], shift, image_list[i], pupil_xyr_list[i], iris_xyr_list[i])
                cv2.imwrite("../dataProcessed/" + filename_list[i] + "_HeatMapVis.png",imVis)

                imVis = irisRec.visualizeMatchingResult(code_list[i], code_list[j], polar_mask_list[i], polar_mask_list[j], shift, image_list[j], pupil_xyr_list[j], iris_xyr_list[j])
                cv2.imwrite("../dataProcessed/" + filename_list[1] + "_HeatMapVis.png",imVis)
    '''
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))