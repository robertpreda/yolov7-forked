import argparse
import os
import zipfile

from pylabel import importer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-file', help="Path to coco.json file", type=str, required=True)
    parser.add_argument('--images', help="Path to images directory, if different than annotations path")
    parser.add_argument('--split', help="Train or val")

    return parser.parse_args()

def main():
    opt = get_args()
    dataset = importer.ImportCoco(opt.ann_file, path_to_images=opt.images, name="COCO")
    #dataset.path_to_annotations = f"../data/darknet/{opt.split}"
    dataset.export.ExportToYoloV5(output_path=f"../dataset/darknet/{opt.split}")

if __name__ == '__main__':
    main()