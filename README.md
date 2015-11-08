# ofxLearn

ofxLearn is a general-purpose machine learning library for OpenFrameworks, built on top of [dlib](http://dlib.net/).


### About

ofxLearn supports classification, regression, and unsupervised clustering. The goal is to be a high-level wrapper for dlib's machine learning routines, taking care of the ugly stuff, i.e. determining a model, kernel, and parameter selection).

The library contains a basic example for each of classification, regression, and clustering. Because training can take a long time, there are also examples for placing each of these tasks into its own separate thread.


### Features

ofxLearn supports classification (using [kernel ridge regression](http://en.wikipedia.org/wiki/Kernel_method)), regression (using kernel ridge or [multilayer perceptron (neural network)](http://en.wikipedia.org/wiki/Multilayer_perceptron)), and [k-means clustering](http://en.wikipedia.org/wiki/K-means_clustering). 

Each has a separate class for threading (see the `_threaded` examples).

### Usage

See examples for usage of classification, regression, and clustering. Depending on the size and complexity of your data, training can take a long time, and it will freeze the application, unless you use the threaded learners. The examples ending with _threaded run the training in a separate thread and alert you with a callback function when they are done. 


### To-do

* grid-parameter search / cross-validation 
* PCA