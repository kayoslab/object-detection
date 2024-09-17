# import the necessary packages
import os

###########################################################################
##
## Paths and filenames
##
###########################################################################

DATASET_PATH = "dataset"

# define the base path to the unprecessed dataset for preprocessing operations
LABELLED_PATH = os.path.sep.join([DATASET_PATH, "labelled"])

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label binarizer, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.keras"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])

###########################################################################
##
## Model training configuration
##
###########################################################################

LEARNING_RATE = 1e-4
SPLIT_RATIO = 0.2
NUM_EPOCHS = 5000
BATCH_SIZE = 16
GLOBAL_CLIPNORM = 10.0
IMAGE_DIMENSIONS_WIDTH = 640
IMAGE_DIMENSIONS_HEIGHT = 640
BOUNDING_BOX_FORMAT = "xyxy"
CLASS_IDS = [
    "A",
    "B"
]

"""
YOLOV8 Backbones available in KerasCV:

1.   Without Weights:

    1.   yolo_v8_xs_backbone
    2.   yolo_v8_s_backbone
    3.   yolo_v8_m_backbone
    4.   yolo_v8_l_backbone
    5.   yolo_v8_xl_backbone

2. With Pre-trained coco weight:

    1.   yolo_v8_xs_backbone_coco
    2.   yolo_v8_s_backbone_coco
    2.   yolo_v8_m_backbone_coco
    2.   yolo_v8_l_backbone_coco
    2.   yolo_v8_xl_backbone_coco
"""
MODEL_BACKBONE = "yolo_v8_m_backbone_coco" 

"""
A specification of the depth of the CSP blocks in the Feature Pyramid Network.
This is usually 1, 2, or 3, depending on the size of your YOLOV8Detector model.
We recommend using 3 for "yolo_v8_l_backbone" and "yolo_v8_xl_backbone". 

Defaults to 2.
"""
FPN_DEPTH = 2

###########################################################################
##
## Validation configuration
##
###########################################################################

IOU_THRESHOLD=0.2

CONFIDENCE_THRESHOLD = 0.7