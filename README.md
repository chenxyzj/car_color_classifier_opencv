# Car color classification using OpenCV - C++ example

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A C++ example for using [Spectrico's car color classifier](http://spectrico.com/car-color-recognition.html). Tested on Windows 10 and Ubuntu Linux 16.04 LTS.  It takes as input a cropped single car image. The size doesn't matter. Because the car image is not always square, it is center-cropped by the demo to become square and then it is resized to 224x224 pixels, so the aspect ratio is keeped. The demo returns the first 3 color probabilities. The classifier is based on Mobilenet v2 (OpenCV backend). It takes 35 milliseconds on Intel Core i5-7600 CPU for single classification. This classifier is not accurate enough yet and serves as a proof-of-concept demo.

---

#### Usage
The demo is started using:
```
$ opencv_car_color_classifier car.jpg
```
The output is printed to the console:
```
  Inference time, ms: 52.3272
  Probabilities:
  orange: 94.7113 %
  white:  2.91052 %
  yellow: 0.689009 %
```

![image](https://github.com/spectrico/car_color_classifier_flask_server/blob/master/car-color.png?raw=true)

---
## Requirements
  - C++ compiler
  - OpenCV

---
## Credits
The car color classifier is based on MobileNetV2 mobile architecture: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
