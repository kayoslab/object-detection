# object-detection

A simple comparison and example implementation of object detection algorithms covering VGG and YOLO v8 for comparison of their functionality and as a starting point for a detection project. The implementation expects images to be labelled using [Label Studio](https://github.com/HumanSignal/label-studio) and their XML file exports.

## VGG Implementation

Due to the earlier implementation of the VGG algorithm the labelled information has to be transformed into a CSV file format. In order to transpose the data into the expected format run the `preprocess.py` function. It will create an index of the given dataset.

## YOLO Implementation

The code is based in parts on the yolov8 example from the keras-io [repository](https://github.com/keras-team/keras-io/blob/master/examples/vision/yolov8.py) and adapted for these purposes. That code in specific is distributed under the Apache License 2.0. You may obtain a copy of the License at [apache.org](https://www.apache.org/licenses/LICENSE-2.0) or in the LICENSE file in the root directory of its repository.