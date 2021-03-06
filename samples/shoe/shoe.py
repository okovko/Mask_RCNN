"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco
    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True
    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last
    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import skimage
import json
import os
import sys
import numpy as np
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as maskUtils

# import zipfile
# import urllib.request
# import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class ShoeConfig(Config):
    """Configuration for training on     MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class ShoeDataset(utils.Dataset):
    def load_shoe(self, dataset_dir, subset):
        """Load dataset
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val)
        """
        
        self.add_class("menwithfootwear", 1, "shoe")

        assert subset in ['train', 'val']
        
        data = json.load(open(os.path.join(dataset_dir, 'annotations.json')))
        images = data['images']
        segments = data['annotations']

        for current_image in images:
            image_id = current_image['id']
            path = os.path.join(dataset_dir, current_image['file_name'])
            width = current_image['width']
            height = current_image['height']
            # num_ids = sum(1 for x in segments if x['image_id'] == image_id)
            annotations = []
            for x in [y for y in segments if y['image_id'] == image_id]:
                annotation_id = x['id']
                xs = segments[annotation_id]['segmentation'][0][::2]
                ys = segments[annotation_id]['segmentation'][0][1::2]
                anno = {}
                anno['id'] = annotation_id
                anno['class'] = 'shoe'
                anno['xs'] = xs
                anno['ys'] = ys
                annotations.append(anno)
            self.add_image('menwithfootwear', image_id, path, width=width, height=height, annotations=annotations)

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        # If not a COCO image, delegate to parent class.
        current_image = self.image_info[image_id]
        if current_image['source'] != 'menwithfootwear':
            return super(ShoeDataset, self).load_mask(image_id)

        h = current_image['height']
        w = current_image['width']
        annotations = current_image['annotations']
        
        image_masks = np.zeros([h, w, len(annotations)], dtype=np.uint8)
        class_ids = np.array([1] * len(annotations), dtype=np.int32)
        
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask
        for i, anno in enumerate(annotations):
            fillxs, fillys = skimage.draw.polygon(anno['xs'], anno['ys'])
            image_masks[fillxs, fillys, i] = True
        
        self.image_info[image_id]['image_masks'] = image_masks
        
        # for anno in annotations:
        #     h = current_image['height']
        #     w = current_image['width']
        #     mask = np.zeros([h, w], dtype=np.uint8)
        #     fillxs, fillys = skimage.draw.polygon(anno['xs'], anno['ys'])
        #     mask[fillxs, fillys] = True
        #     image_masks.append(mask)
        #     class_ids.append(1) # class id for shoe
        
        return image_masks, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        current_image = self.image_info[image_id]
        if current_image["source"] == "shoe":
            return current_image["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Configurations
    config = ShoeConfig()
    # Model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_MODEL_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)


    # datasets for training and for validation
    dataset_train = ShoeDataset()
    dataset_train.load_shoe(args.dataset, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShoeDataset()
    dataset_val.load_shoe(args.dataset, 'val')
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')