# import the necessary packages
import os

DATASET_PATH = "dataset"

# define the base path to the unprecessed dataset for preprocessing operations
LABELLED_PATH = os.path.sep.join([DATASET_PATH, "labelled"])


# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
PROCESSED_PATH = os.path.sep.join([DATASET_PATH, "processed"])
PROCESSED_IMAGES_PATH = os.path.sep.join([PROCESSED_PATH, "images"])
PROCESSED_ANNOTS_PATH = os.path.sep.join([PROCESSED_PATH, "annotations"])


# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label binarizer, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.keras"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


LEARNING_RATE = 1e-4
SPLIT_RATIO = 0.2
NUM_EPOCHS = 10
BATCH_SIZE = 16
GLOBAL_CLIPNORM = 10.0
IMAGE_DIMENSIONS_WIDTH = 224
IMAGE_DIMENSIONS_HEIGHT = 6224

CLASS_IDS = [
    "A",
    "B"
]