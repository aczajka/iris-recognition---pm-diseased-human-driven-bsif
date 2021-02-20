from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os

def main(cfg):

    irisRec = irisRecognition(cfg)

    # Get images to process
    image_list = []
    filename_list = []
    for filename in glob.glob("../data/*.bmp"):
        print(filename)
        im = Image.open(filename)
        image_list.append(im)
        filename_list.append(os.path.basename(filename))

    # Segment
    for im,fn in zip(image_list,filename_list):
        mask = irisRec.segment(im)
        im_mask = Image.fromarray(mask)
        im_mask.save("../dataProcessed/" + fn + "_seg_CCNet_mask.png")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default="cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path))