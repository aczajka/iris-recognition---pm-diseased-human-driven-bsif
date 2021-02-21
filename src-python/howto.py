from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os

def main(cfg):

    irisRec = irisRecognition(cfg)

    image_list = []
    filename_list = []
    mask_list = []
    polar_list = []

    # Get images to process
    
    for filename in glob.glob("../data/*.bmp"):
        print(filename)
        im = Image.open(filename)
        image_list.append(im)
        filename_list.append(os.path.basename(filename))

    # Segment
    for im,fn in zip(image_list,filename_list):
        mask = irisRec.segment(im)
        im_mask = Image.fromarray(mask)
        mask_list.append(im_mask)
        im_mask.save("../dataProcessed/" + os.path.splitext(fn)[0] + "_seg_CCNet_mask.png")

    # Cartesian to polar

    # Coding



    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))