# ofxLearn

ofxLearn is a general-purpose machine learning library for OpenFrameworks, built on top of [dlib](http://dlib.net/).


### About

ofxLearn supports classification, regression, and unsupervised clustering tasks. The goal for it is to be a high-level wrapper for dlib's machine learning routines, taking care of the ugly stuff, e.g. determining a model, kernel, and parameter selection).

At this time, the library contains basic examples for classification, regression, and clustering, and a more sophisticated gesture recognition (classification) example.


### Features

ofxLearn supports classification and regression (both using kernel ridge regression model), and k-means clustering. In the future, more models and learning algorithms will be wrapped as well, and more examples of higher level usage will be provided.


### Usage

Note the following compilation instructions. You should add ofxLearn's source folder (ofxLearn/src) to your project, but *do not* add the "libs" folder containing dlib, as it will cause naming collisions. Instead, go to the project's build settings, and add "../../../addons/ofxLearn/libs/" to your user header search paths.

Note also that classification and regression can sometimes take several minutes to complete -- check the console for progress.  
