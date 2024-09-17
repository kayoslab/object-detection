# import the necessary packages
from configuration import config

import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np

import keras_cv
from keras_cv import visualization

# Imports for annotation parsing from XML files
import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

# construct the argument parser and parse the arguments
import mimetypes
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", 
    "--input", 
    required=True,
	help="path to input xml/text file of xml paths"
)
args = vars(ap.parse_args())

# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
xml_paths = [args["input"]]
# if the file type is a text file, then we need to process *multiple*
# xml
if "text/plain" == filetype:
	# load the xml paths in our testing file
	xml_paths = open(args["input"]).read().strip().split("\n")

# load our object detector and label binarizer from disk
print("[INFO] loading object detector...")
yolo = load_model(config.MODEL_PATH)
# map each class name to a unique numerical identifier. 
# This mapping is used to encode and decode the class labels during training and inference in
# object detection tasks.
class_ids = config.CLASS_IDS
class_mapping = dict(zip(range(len(class_ids)), class_ids))


########################################################################################################################
##
## Image loading
##
## We will load the images from the input file and preprocess them for the object detection model.
##
########################################################################################################################


# map each class name to a unique numerical identifier. 
# This mapping is used to encode and decode the class labels during training and inference in
# object detection tasks.
class_ids = config.CLASS_IDS
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Get all XML file paths in labelled path and sort them
path_labelled = config.LABELLED_PATH
xml_files = sorted(
    [
        os.path.join(path_labelled, file_name)
        for file_name in os.listdir(path_labelled)
        if file_name.endswith(".xml")
    ]
)

def parse_annotation(xml_file):
    """Reads XML files, finds image name and paths and iterates over each object in the XML file.
    It extracts the bounding box coordinates and class labels for each object.

    The function returns three values: the image path, a list of bounding boxes (each
    represented as a list of four floats: xmin, ymin, xmax, ymax), and a list of class IDs
    (represented as integers) corresponding to each bounding box. The class IDs are obtained
    by mapping the class labels to integer values using a dictionary called `class_mapping`.

    Parameters
    ----------
        xml_file (str): the XML file to parse

    Returns
    -------
        image_path (str): the path to the image
        boxes (list): a list of bounding boxes
        class_ids (list): a list of class IDs
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(path_labelled, image_name)

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids


image_paths = []
bbox = []
classes = []

for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)

bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)
data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))


def load_image(image_path):
    """Reads and decodes an image from the given file path.

    Parameters
    ----------
        image_path (str): the file path to the image

    Returns
    -------
        image (Tensor): A Tensor of type uint8.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    """Loads an image and its associated bounding boxes and class labels.

    Parameters
    ----------
        image_path (str): the file path to the image
        classes (Tensor): a Tensor containing the class labels
        bbox (Tensor): a Tensor containing the bounding boxes

    Returns
    -------
        dict: a dictionary containing the image and bounding boxes
    """
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

resizing = keras_cv.layers.JitteredResize(
    target_size=(
        config.IMAGE_DIMENSIONS_WIDTH,
        config.IMAGE_DIMENSIONS_HEIGHT
    ),
    scale_factor=(
        0.75, 
        1.3
    ),
    bounding_box_format=config.BOUNDING_BOX_FORMAT,
)

val_ds = data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(config.BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(config.BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

# prepare the inputs for the model
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

########################################################################################################################
##
## Visualizing predictions
##
## Finally, we will visualize the predictions made by the model on the validation dataset.
## We will plot the images with the predicted bounding boxes and class labels.
##
########################################################################################################################

def visualize_detections(model, dataset):
    """Visualizes the predictions made by the model on the dataset.

    Parameters
    ----------
        model (tf.keras.Model): the object detection model
        dataset (tf.data.Dataset): the dataset to visualize
        bounding_box_format (str): the format of the bounding boxes
    """
    images, y_true = next(iter(dataset.take(10)))

    model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
        bounding_box_format=config.BOUNDING_BOX_FORMAT,
        from_logits=True,
        # Decrease the required threshold to make predictions get pruned out
        iou_threshold=config.IOU_THRESHOLD,
        # Tune confidence threshold for predictions to pass NMS
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )

    y_pred = model.predict(images)

    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=config.BOUNDING_BOX_FORMAT,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        legend=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )

print("[INFO] Visualizing predictions...")
visualize_detections(
    yolo,
    dataset=val_ds
)