# ofxLearn

ofxLearn is a general-purpose machine learning library for OpenFrameworks, built on top of [dlib](http://dlib.net/).


### About

ofxLearn supports classification, regression, and unsupervised clustering tasks. The goal is to be a high-level wrapper for dlib's machine learning routines, taking care of the ugly stuff, i.e. determining a model, kernel, and parameter selection).

The library contains basic examples for classification, regression, and clustering, and a simple mouse-based gesture recognition (classification) example.


### Features

ofxLearn supports classification (using [kernel ridge regression](http://en.wikipedia.org/wiki/Kernel_method)), regression using (kernel ridge or [multilayer perceptron (neural network)](http://en.wikipedia.org/wiki/Multilayer_perceptron)), and [k-means clustering](http://en.wikipedia.org/wiki/K-means_clustering). 


### Usage

Add ofxLearn's source folder (ofxLearn/src) to your project, but don't add the "libs" folder containing dlib, as it will cause naming collisions. Instead, add "../../../addons/ofxLearn/libs/" to your user header search paths.

Note also that classification and regression can sometimes take several minutes to complete, especially in `ACCURATE` mode -- check the log (set verbose) in the console for progress.  
