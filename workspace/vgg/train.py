# Configuration file for the wound-test project
from configuration import config

# Keras related imports for the training script
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV3Large
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

# scikit-learn related imports for the training script
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# additional imports for importing the dataset
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os


########################################################################################################################
##
## Data loading and preprocessing
##
## The first step is to load the dataset from disk. We will load the images, class labels, and bounding box coordinates
## from disk and store them in memory. We will also preprocess the images by resizing them to 224×224 pixels and scaling
## the pixel intensities to the range [0, 1] in order to prepare them for the VGG16 network.
##
##
########################################################################################################################


# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
data = []
labels = []
bboxes = []
imagePaths = []

print("[INFO] loading dataset...")

# loop over all CSV files in the annotations directory
for csvPath in paths.list_files(config.PROCESSED_ANNOTS_PATH, validExts=(".csv")):
	# load the contents of the current CSV annotations file
	rows = open(csvPath).read().strip().split("\n")
	# loop over the rows
	for row in rows:
		# break the row into the filename, bounding box coordinates,
		# and class label
		row = row.split(",")
		(filename, startX, startY, endX, endY, label) = row

		# derive the path to the input image, load the image (in
		# OpenCV format), and grab its dimensions
		imagePath = os.path.sep.join([config.PROCESSED_IMAGES_PATH, label, filename])
		image = cv2.imread(imagePath)
		(h, w) = image.shape[:2]
		# scale the bounding box coordinates relative to the spatial
		# dimensions of the input image
		startX = float(startX) / w
		startY = float(startY) / h
		endX = float(endX) / w
		endY = float(endY) / h

		# load the image and preprocess it
		# resizing step forces our image to 224×224 pixels for our VGG16-based CNN.
		image = load_img(
			imagePath, 
			target_size=(
				config.IMAGE_DIMENSIONS_WIDTH, 
				config.IMAGE_DIMENSIONS_HEIGHT,
			)
		)
		image = img_to_array(image)
		# update our list of data, class labels, bounding boxes, and
		# image paths
		data.append(image)
		labels.append(label)
		bboxes.append((startX, startY, endX, endY))
		imagePaths.append(imagePath)

# convert the data, class labels, bounding boxes, and image paths to
# NumPy arrays, scaling the input pixel intensities from the range
# [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

# perform one-hot encoding on the labels
###################### ToDo: Check for documentation on one-hot encoding ######################
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(lb.classes_) == 2:
	labels = to_categorical(labels)


########################################################################################################################
##
## Data preparation
##
## Splitting the data into training and testing splits. We will use 80% of the data for training and the remaining 20%
## for testing. We will also write the testing image paths to disk so that we can use them when evaluating/testing our
## object detector.
##
########################################################################################################################


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, labels, bboxes, imagePaths, test_size=0.20, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()


########################################################################################################################
##
## Model creation
##
## We will use the VGG16 network as a feature extractor, removing the fully connected layer head 
## and replacing them with our own custom layer head. The body of the network will be frozen
## so that the weights are not updated during the training process. This is a common practice.
## We will then flatten the layer to the body of the network.
## This way we will create a multi-output (two-branch) model for multi-class bounding box regression
## and classification. The model will output the bounding box coordinates and the class label.
##
########################################################################################################################


# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

# put together our model which accept an input image and then output
# bounding box coordinates and a class label
vgg_model = Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))

# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error
# for the bounding box head
losses = {
	"class_label": "categorical_crossentropy",
	"bounding_box": "mean_squared_error",
}

# define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
	"class_label": 1.0,
	"bounding_box": 1.0
}

# define the metrics used to evaluate during training and testing.
# The metrics passed here are evaluated without sample weighting;
# if you would like sample weighting to apply, you can specify your 
# metrics via the weighted_metrics argument instead.
metrics = [
	"accuracy",
	["accuracy", "mse"]
]

# initialize the optimizer, compile the model, and show the model summary
vgg_opt = Adam(learning_rate=config.LEARNING_RATE)
vgg_model.compile(loss=losses, optimizer=vgg_opt, metrics=metrics , loss_weights=lossWeights)
print(vgg_model.summary())


########################################################################################################################
##
## Model training
##
## We will train the model using the training data and validate it using the testing data.
## Once the model is trained we serialize the model to disk and save the label binarizer.
##
########################################################################################################################

# construct a dictionary for our target training outputs
trainTargets = {
	"class_label": trainLabels,
	"bounding_box": trainBBoxes
}
# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
	"class_label": testLabels,
	"bounding_box": testBBoxes
}

# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
H = vgg_model.fit(
	trainImages, 
	trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1
)

# serialize the model to disk
print("[INFO] saving object detector model...")
vgg_model.save(config.MODEL_PATH)

# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
f = open(config.LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()


########################################################################################################################
##
## Model evaluation
##
## We will plot the training and validation loss and accuracy to visualize the training process.
## The plot will be saved to disk.
##
########################################################################################################################

# plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
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

# create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"], label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"], label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

# save the accuracies plot
plotPath = os.path.sep.join([config.PLOTS_PATH, "accs.png"])
plt.savefig(plotPath)

