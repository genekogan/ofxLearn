# ofxLearn

ofxLearn is a general-purpose machine learning library for OpenFrameworks, built on top of [dlib](http://dlib.net/).


### About

ofxLearn supports classification, regression, and unsupervised clustering. The goal is to be a high-level wrapper for dlib's machine learning routines, taking care of the ugly stuff, i.e. determining a model, kernel, and parameter selection).

The library contains a basic example for each of classification, regression, and clustering. Because training can take a long time, there are also examples for placing each of these tasks into its own separate thread.


### Features

ofxLearn supports classification (using [kernel ridge regression](http://en.wikipedia.org/wiki/Kernel_method)), regression (using kernel ridge or [multilayer perceptron (neural network)](http://en.wikipedia.org/wiki/Multilayer_perceptron)), and [k-means clustering](http://en.wikipedia.org/wiki/K-means_clustering). 

Each has a separate class for threading (see the `_threaded` examples).

### Usage

Add **only** ofxLearn's source folder (ofxLearn/src) to your project, but don't add the "libs" folder containing dlib, as it will cause naming collisions. Instead, add `../../../addons/ofxLearn/libs/` to your header search paths. In XCode, this can be found in the Build Settings of your project file.

Note also that classification and regression can sometimes take several minutes to complete, depending on the complexity and size of the data.


### To-do

* implement grid-parameter search and cross-validation routines for parameter selection
* principal component analysis 