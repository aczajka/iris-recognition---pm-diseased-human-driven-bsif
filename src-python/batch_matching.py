from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np

def main(args):
    
    irisRec = irisRecognition(get_cfg(args.cfg_path))

    with open(args.pairs,'r') as pfile:
        data = pfile.readlines()
    pairs_list = [l.strip('\n').split(' ') for l in data]

    # Get the list of images to process
    filename_list = list(set([y for x in pairs_list for y in x]))
    image_list = []
    mask_list = []
    i = 0
    filename_map = {}
    for filename in filename_list:
        im = Image.open(os.path.join(args.images, filename)).convert('L')
        image_list.append(im)

        maskname = os.path.join(args.masks, filename).replace('bmp','png')
        mask = np.array(Image.open(maskname)).astype(np.bool)
        mask_list.append(mask)
        filename_map[filename]=i;
        i += 1
        

    # Segmentation, normalization and encoding
    polar_mask_list = []
    code_list = []
    for im, mask,fn in zip(image_list, mask_list, filename_list):
        
        print(fn)

        # circular approximation:
        pupil_xyr, iris_xyr = irisRec.circApprox(mask)

        # cartesian to polar transformation:
        im_polar, mask_polar = irisRec.cartToPol(im, mask, pupil_xyr, iris_xyr)
        polar_mask_list.append(mask_polar)

        # human-driven BSIF encoding:
        code = irisRec.extractCode(im_polar)
        code_list.append(code)

        # DEBUG: save selected processing results 
        np.savez_compressed("./templates/" + os.path.splitext(fn)[0] + "_tmpl.npz",code)
        
    # Matching (all-vs-all, as an example)
    for reference, probe in pairs_list:
        
        refix = filename_map[reference]
        code1 = code_list[refix]
        mask1 = polar_mask_list[refix]
        
        probix = filename_map[probe]
        code2 = code_list[probix]
        mask2 = polar_mask_list[probix]

        score, shift = irisRec.matchCodes(code1, code2, mask1, mask2)
        print(f"{reference} {probe} {score}")

     
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg.yaml", help="path of the configuration file")
    parser.add_argument("--pairs", type=str, help="File containing image pairs to be compared.", required=True)
    parser.add_argument("--images", type=str, help="Directory containing the images to be compared.", default='../data')
    parser.add_argument("--masks", type=str, help="Directory containing the masks of the images.")
    args = parser.parse_args()
    main(args)