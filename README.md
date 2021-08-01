# Drowsiness Detection

## Problem statement

Build a model that will alert the driver before he falls asleep based on the realtime drowsiness detection.

## Model Description

Drowsiness detection using LSTM model which takes a sequence of 50 extracted feature frame to determine the level of drowsiness.

## Packages Used

* Face Landmark Detection: To detect the face and crop the region of interest i.e. eye region.

![Face landmark example](https://raw.githubusercontent.com/patlevin/face-detection-tflite/main/docs/portrait_fl.jpg)

* Inception V3: To extract features from the cropped image of eyes.

* LSTM: To learn the relation between 50 extracted feature frames and predict the drowsiness level.
