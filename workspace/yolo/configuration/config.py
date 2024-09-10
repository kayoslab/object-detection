# import the necessary packages
import os

DATASET_PATH = "dataset"

# define the base path to the unprecessed dataset for preprocessing operations
LABELLED_PATH = os.path.sep.join([DATASET_PATH, "labelled"])

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label binarizer, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.keras"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])

LEARNING_RATE = 1e-4
SPLIT_RATIO = 0.2
NUM_EPOCHS = 32
BATCH_SIZE = 4
GLOBAL_CLIPNORM = 10.0
IMAGE_DIMENSIONS_WIDTH = 640
IMAGE_DIMENSIONS_HEIGHT = 640

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