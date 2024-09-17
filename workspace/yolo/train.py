########################################################################################################################
##
## YOLO Object Detection with KerasCV
##
## In this notebook, we will train a YOLO (You Only Look Once) object detection model using the KerasCV library.
## We will use the YOLOV8 architecture, which is a state-of-the-art object detection model that is highly regarded
## for its accuracy and speed. We will train the model on a custom dataset of wound images and evaluate its performance
## using the COCO metric. We will also visualize the predictions made by the model on the validation dataset.
##
## YOLOv8 is a cutting-edge YOLO model that is used for a variety of computer vision tasks, such as object detection, 
## image classification, and instance segmentation. Ultralytics, the creators of YOLOv5, also developed YOLOv8,
## which incorporates many improvements and changes in architecture and developer experience compared to its predecessor. 
## 
## Below table compares the performance metrics of five different YOLOv8 models with different sizes (measured in pixels): 
## YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x. The metrics include mean average precision (mAP) values at different
## intersection-over-union (IoU) thresholds for validation data, inference speed on CPU with ONNX format and A100 TensorRT, 
## number of parameters, and number of floating-point operations (FLOPs) (both in millions and billions, respectively).
## As the size of the model increases, the mAP, parameters, and FLOPs generally increase while the speed decreases. 
## YOLOv8x has the highest mAP, parameters, and FLOPs but also the slowest inference speed, while YOLOv8n has the smallest size, 
## fastest inference speed, and lowest mAP, parameters, and FLOPs.
##
## | Model   | size(pixels) | mAP val 50-95 | Speed CPU ONNX(ms) | Speed A100 TensorRT (ms) | params(M) | FLOPs(B) |
## | ------- | ------------ | ------------- | ------------------ | ------------------------ | --------- | -------- |
## | YOLOv8n | 640          | 37.3          | 80.4               | 0.99                     | 3.2       | 8.7      |
## | YOLOv8s | 640          | 44.9          | 128.4              | 1.20                     | 11.2      | 28.6     |
## | YOLOv8m | 640          | 50.2          | 234.7              | 1.83                     | 25.9      | 78.9     |
## | YOLOv8l | 640          | 52.9          | 375.2              | 2.39                     | 43.7      | 165.2    |
## | YOLOv8x | 640          | 53.9          | 479.1              | 3.53                     | 68.2      | 257.8    |
##
## For further reading on the YOLO model and its changes in comparison to previous versions, please refer to the 
## [YOLOv8 blog post](https://blog.roboflow.com/whats-new-in-yolov8/).
## [Choosing Models](https://medium.com/@sohaib.zafar522/choosing-the-right-pre-trained-model-a-guide-to-vggnet-resnet-googlenet-alexnet-and-inception-db7a8c918510)
##
##
## The notebook is divided into the following sections:
## 1. Data loading and preprocessing
## 2. Model creation
## 3. Model training
## 4. Model evaluation
##
## The code is based in parts on the yolov8 example from the keras-io repository
## source: https://github.com/keras-team/keras-io/blob/master/examples/vision/yolov8.py
##
## The code is distributed under the Apache License 2.0. You may obtain a copy of the License at
## https://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root directory of this repository.
##
########################################################################################################################

# Configuration file for the wound-test project
from configuration import config

# Imports for argument parsing
import os, sys, argparse

# Imports for annotation parsing from XML files
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

# Imports for data loading and processing
import tensorflow as tf
from tensorflow import keras
import keras_cv

# Imports for data evaluation
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
##
## Argument parsing
##
## Construct an argument parser, parses the arguments and sets the configuration parameters for 
## the model training accordingly.
##
########################################################################################################################

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", 
    "--input", 
    required=False,
	help="Path to dataset directory. Default is set to ./dataset/labelled"
)
ap.add_argument(
    "-o", 
    "--output", 
    required=False,
	help="Path to output directory. Default is set to ./output"
)
ap.add_argument(
    "--model-backbone", 
    required=False,
	help="Specifies the model backboe used. Default is set to yolo_v8_m_backbone_coco,"
)
ap.add_argument(
    "--batch-size", 
    required=False,
	help="Batch size for training. Default is set to 16"
)
ap.add_argument(
    "--epochs",
    required=False,
	help="Number of epochs for training. Default is set to 5000"
)

try:
    args = vars(ap.parse_args())
except:
    ap.print_help()
    sys.exit(0)

# Set the dataset path
if args["input"]:
    config.DATASET_PATH = args["input"]
if args["output"]:
    config.BASE_OUTPUT = args["output"]
if args["model_backbone"]:
    if args["model_backbone"] in config.AVAILABLE_BACKBONES:
        config.MODEL_BACKBONE = args["model_backbone"]
    else:
        print(f"Model backbone {args['model_backbone']} not available. Using default yolo_v8_m_backbone_coco")
if args["batch_size"]:
    config.BATCH_SIZE = int(args["batch_size"])
if args["epochs"]:
    config.NUM_EPOCHS = int(args["epochs"])

# Create folder structure if not existing
if not os.path.exists(config.BASE_OUTPUT):
    os.makedirs(config.BASE_OUTPUT)
if not os.path.exists(config.PLOTS_PATH):
    os.makedirs(config.PLOTS_PATH)

########################################################################################################################
##
## Data loading and preprocessing
##
## The first step is to load the labelled data from disk. 
## We will load the images, class labels, and bounding box coordinates from disk and store them in memory.
## We will then create a tf.data.Dataset from the loaded data to create an input pipeline for the model.
## The dataset will be split into training and validation sets, and the images will be preprocessed and augmented
## using KerasCV data augmentation layers.
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


"""
We are using `tf.ragged.constant` to create ragged tensors from the `bbox` and
`classes` lists. A ragged tensor is a type of tensor that can handle varying lengths of
data along one or more dimensions. This is useful when dealing with data that has
variable-length sequences, such as text or time series data.

```python
classes = [
    [8, 8, 8, 8, 8],      # 5 classes
    [12, 14, 14, 14],     # 4 classes
    [1],                  # 1 class
    [7, 7],               # 2 classes
 ...]
```

```python
bbox = [
    [[199.0, 19.0, 390.0, 401.0],
    [217.0, 15.0, 270.0, 157.0],
    [393.0, 18.0, 432.0, 162.0],
    [1.0, 15.0, 226.0, 276.0],
    [19.0, 95.0, 458.0, 443.0]],     #image 1 has 4 objects
    [[52.0, 117.0, 109.0, 177.0]],   #image 2 has 1 object
    [[88.0, 87.0, 235.0, 322.0],
    [113.0, 117.0, 218.0, 471.0]],   #image 3 has 2 objects
 ...]
```

In this case, the `bbox` and `classes` lists have different lengths for each image,
depending on the number of objects in the image and the corresponding bounding boxes and
classes. To handle this variability, ragged tensors are used instead of regular tensors.

Later, these ragged tensors are used to create a `tf.data.Dataset` using the
`from_tensor_slices` method. This method creates a dataset from the input tensors by
slicing them along the first dimension. By using ragged tensors, the dataset can handle
varying lengths of data for each image and provide a flexible input pipeline for further
processing.
"""

bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)
data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))


# Determine the number of validation samples
num_val = int(len(xml_files) * config.SPLIT_RATIO)

# Split the dataset into train and validation sets
val_data = data.take(num_val)
train_data = data.skip(num_val)



"""
Bounding boxes in KerasCV have a predetermined format. To do this, you must bundle your bounding
boxes into a dictionary that complies with the requirements listed below:

```python
bounding_boxes = {
    # num_boxes may be a Ragged dimension
    'boxes': Tensor(shape=[batch, num_boxes, 4]),
    'classes': Tensor(shape=[batch, num_boxes])
}
```

The dictionary has two keys, `'boxes'` and `'classes'`, each of which maps to a
TensorFlow RaggedTensor or Tensor object. Ragged tensors are the TensorFlow equivalent 
of nested variable-length lists. They make it easy to store and process data with 
non-uniform shapes. The `'boxes'` Tensor has a shape of `[batch, num_boxes, 4]`, 
where batch is the number of images in the batch and num_boxes is the maximum number 
of bounding boxes in any image. The 4 represents the four values needed to define a
bounding box:  xmin, ymin, xmax, ymax.

The `'classes'` Tensor has a shape of `[batch, num_boxes]`, where each element represents
the class label for the corresponding bounding box in the `'boxes'` Tensor. The num_boxes
dimension may be ragged, which means that the number of boxes may vary across images in
the batch.

Final dict should be:
```python
{"images": images, "bounding_boxes": bounding_boxes}
```
"""


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
    image = tf.image.decode_png(image, channels=3)
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
    return {
        "images": tf.cast(image, tf.float32),
        "bounding_boxes": bounding_boxes
    }

"""
## Data Augmentation

One of the most challenging tasks when constructing object detection pipelines is data
augmentation. It involves applying various transformations to the input images to
increase the diversity of the training data and improve the model's ability to
generalize. However, when working with object detection tasks, it becomes even more
complex as these transformations need to be aware of the underlying bounding boxes and
update them accordingly.

KerasCV provides native support for bounding box augmentation. KerasCV offers an
extensive collection of data augmentation layers specifically designed to handle bounding
boxes. These layers intelligently adjust the bounding box coordinates as the image is
transformed, ensuring that the bounding boxes remain accurate and aligned with the
augmented images.

Here we create a layer that resizes images, while maintaining the original aspect ratio.
The bounding boxes associated with the image are specified in the `xyxy` format.
If necessary, the resized image will be padded with zeros to maintain the original 
aspect ratio.
"""

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(
            mode="horizontal",
            bounding_box_format=config.BOUNDING_BOX_FORMAT
        ),
        keras_cv.layers.RandomShear(
            x_factor=0.2, 
            y_factor=0.2, 
            bounding_box_format=config.BOUNDING_BOX_FORMAT
        ),
        keras_cv.layers.JitteredResize(
            target_size = (
                config.IMAGE_DIMENSIONS_WIDTH,
                config.IMAGE_DIMENSIONS_HEIGHT
            ), 
            scale_factor = (
                0.75, 
                1.3
            ), 
            bounding_box_format = config.BOUNDING_BOX_FORMAT
        ),
    ]
)

# Creating Training Dataset
train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(config.BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(config.BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

# Creating Validation Dataset
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

val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(config.BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(config.BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

# prepare the inputs for the model
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]

train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

########################################################################################################################
##
## Model creation
##
## Next, we will build a YOLOV8 model using the YOLOV8Detector class from KerasCV.
## The YOLOV8Detector class accepts a feature extractor as the backbone argument, a num_classes argument that specifies
## the number of object classes to detect based on the size of the class_mapping list, a bounding_box_format argument that
## informs the model of the format of the bbox in the dataset, and a finally, the feature pyramid network (FPN) depth is
## specified by the fpn_depth argument.
##
########################################################################################################################

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    config.MODEL_BACKBONE
)

"""
Next, let's build a YOLOV8 model using the `YOLOV8Detector`, which accepts a feature
extractor as the `backbone` argument, a `num_classes` argument that specifies the number
of object classes to detect based on the size of the `class_mapping` list, a
`bounding_box_format` argument that informs the model of the format of the bbox in the
dataset, and a finally, the feature pyramid network (FPN) depth is specified by the
`fpn_depth` argument.
"""
yolo = keras_cv.models.YOLOV8Detector(
    backbone=backbone,
    num_classes=len(class_mapping),
    bounding_box_format=config.BOUNDING_BOX_FORMAT,
    fpn_depth=config.FPN_DEPTH,
)

"""
Loss used for YOLOV8 is a combination of two loss functions:

1. Classification Loss: This loss function calculates the discrepancy between anticipated
class probabilities and actual class probabilities. In this instance,
`binary_crossentropy`, a prominent solution for binary classification issues, is
Utilized. We Utilized binary crossentropy since each thing that is identified is either
classed as belonging to or not belonging to a certain object class (such as a person, a
car, etc.).

2. Box Loss: `box_loss` is the loss function used to measure the difference between the
predicted bounding boxes and the ground truth. In this case, the Complete IoU (CIoU)
metric is used, which not only measures the overlap between predicted and ground truth
bounding boxes but also considers the difference in aspect ratio, center distance, and
box size. Together, these loss functions help optimize the model for object detection by
minimizing the difference between the predicted and ground truth class probabilities and
bounding boxes.

For more specific settings consult the KerasCV documentation on the 
[Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
"""
optimizer = tf.keras.optimizers.Adam(
    learning_rate=config.LEARNING_RATE,
    global_clipnorm=config.GLOBAL_CLIPNORM,
)

yolo.compile(
    classification_loss="binary_crossentropy",
    box_loss="ciou",
    optimizer=optimizer,
    jit_compile=False,
)

########################################################################################################################
##
## Model training
##
## We will train the YOLOV8 model on the training dataset and evaluate its performance on the validation dataset.
## We will use the COCO metric to evaluate the model's performance and save the best model based on the MaP score.
##
########################################################################################################################

"""
## COCO Metric Callback

We will be using `BoxCOCOMetrics` to evaluate the model and calculate the MaP(Mean Average Precision) 
score, Recall and Precision. We also save our model when the MaP score improves.
"""
class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format=config.BOUNDING_BOX_FORMAT,
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        if current_map >= self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)

        return logs

# Train the model on the training dataset and evaluate it on the validation dataset
H = yolo.fit(
    train_ds,
    epochs=config.NUM_EPOCHS,
    validation_data=val_ds,
    shuffle=True,
    callbacks=[
        EvaluateCOCOMetricsCallback(
            val_ds,
            config.MODEL_PATH
        )
    ],
    validation_freq=1,
)

# Print the model summary to get an overview of the model architecture and the number of parameters
yolo.summary()

########################################################################################################################
##
## Model evaluation
##
## We will plot the training and validation loss and accuracy to visualize the training process.
## The plot will be saved to disk.
##
########################################################################################################################

# plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_loss", "box_loss"]
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(N, H.history[l], label=l)
	ax[i].plot(N, H.history["val_" + l], label="val_" + l)
	ax[i].legend()

# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plotPath = os.path.sep.join([config.PLOTS_PATH, "losses.png"])
plt.savefig(plotPath)
plt.close()

# create a new figure for the MaP
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["MaP"], label="MaP")
plt.plot(N, H.history["MaP@[IoU=50]"], label="MaP (IoU=50)")
plt.plot(N, H.history["MaP@[IoU=75]"], label="MaP (IoU=75)")
plt.title("Mean Average Precision")
plt.xlabel("Epoch #")
plt.ylabel("MaP")
plt.legend(loc="lower left")

# save the MaP plot
plotPath = os.path.sep.join([config.PLOTS_PATH, "map.png"])
plt.savefig(plotPath)
plt.close()