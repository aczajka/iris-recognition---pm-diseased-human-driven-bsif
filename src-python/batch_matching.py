from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np

from multiprocessing import Pool

def preprocess_image(im, mask, irisRec, name, dir):
    pmaskname = os.path.join(dir, name.replace('.bmp','_pmask.png'))
    codename = os.path.join(dir, name.replace('.bmp','_tmpl.npz'))
    try:
        # load the calculated code/polar mask
        mask_polar = cv2.imread(pmaskname, 0)
        with np.load(codename) as npz:
            code = npz['code']
    except:
        # circular approximation:
        pupil_xyr, iris_xyr = irisRec.circApprox(mask)

        # cartesian to polar transformation:
        im_polar, mask_polar = irisRec.cartToPol(im, mask, pupil_xyr, iris_xyr)
        cv2.imwrite(pmaskname,mask_polar)

        # human-driven BSIF encoding:
        code = irisRec.extractCode(im_polar)
        np.savez_compressed(codename,code=code)
    return code, mask_polar


def match_pair(file1, file2, imgdir, maskdir, tmpdir, cfg_path):
    # load images/masks
    reference = Image.open(os.path.join(imgdir, file1)).convert('L')
    probe = Image.open(os.path.join(imgdir, file2)).convert('L')
    
    maskname = os.path.join(maskdir, file1).replace('bmp','png')
    maskref = np.array(Image.open(maskname)).astype(np.bool)
    maskname = os.path.join(maskdir, file2).replace('bmp','png')
    maskprb = np.array(Image.open(maskname)).astype(np.bool)
    
    # failed segmentations
    if np.count_nonzero(maskref)==0:
        print(f'{file1} failed segmentation.')
        maskref = None
    if np.count_nonzero(maskprb)==0:
        print(f'{file2} Failed segmentation.')
        maskprb = None

    irisRec = irisRecognition(get_cfg(cfg_path))

    # preprocessing
    coderef, pmaskref = preprocess_image(reference, maskref, irisRec, file1, tmpdir)
    codeprb, pmaskprb = preprocess_image(probe, maskprb, irisRec, file2, tmpdir)

    # matching
    score, shift = irisRec.matchCodes(coderef, codeprb, pmaskref, pmaskprb)

    return file1, file2, score, shift


def main(args):
    
    with open(args.pairs,'r') as pfile:
        data = pfile.readlines()
    pairs_list = [l.strip('\n').split(' ') for l in data]

    # parallelize matching
    with Pool(8) as pool:
        def callback(*args):
            print(f"{args[0][0]} {args[0][1]} {args[0][2]}")
            return
        results = [
            pool.apply_async(
                match_pair, 
                args = (reference, probe, args.images, args.masks, args.tempdir, args.cfg_path),
                callback=callback
            )
            for reference, probe in pairs_list]
        results = [r.get for r in results]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg.yaml", help="path of the configuration file")
    parser.add_argument("--pairs", type=str, help="File containing image pairs to be compared.", required=True)
    parser.add_argument("--images", type=str, help="Directory containing the images to be compared.", default='../data')
    parser.add_argument("--masks", type=str, help="Directory containing the masks of the images.")
    parser.add_argument("--tempdir", type=str, help="Directory to store masks and iriscodes.", default="../dataProcessed")
    args = parser.parse_args()
    main(args)