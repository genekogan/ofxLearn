# ofxLearn

ofxLearn is a general-purpose machine learning library for OpenFrameworks, built on top of [dlib](http://dlib.net/).

### About

ofxLearn supports classification, regression, and unsupervised clustering tasks. The goal for it is to be a high-level wrapper to powerful machine learning routines which takes care of the ugly stuff (e.g. determining  model, kernel, and parameter selection) while at the same time providing optional accessors for dlib's more advanced functionality for more power users.

At this time, the library contains basic examples for classification and clustering, with more examples to come, including regression task, and applications to gesture recognition.

### Features

ofxLearn supports classification via support vector machine, regression via multilayer perceptron (neural network), and k-means clustering. In the future, more models and learning algorithms will be wrapped as well.

### Usage

Please note the following compilation instructions. You should add ofxLearn's source folder (ofxLearn/src) to your project, but *do not* add the "libs" folder which contains dlib, as it will cause naming collisions. Instead, go to the project's build settings, and add "../../../addons/ofxLearn/libs/" to your user header search paths.  
