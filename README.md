# ofxLearn

ofxLearn is a general-purpose machine learning library for OpenFrameworks, built on top of [dlib](http://dlib.net/).  It is currently a work in progress. 

## About

ofxLearn will support classification, regression, and unsupervised clustering tasks. The goal for it is to be a high-level wrapper to powerful machine learning routines which takes care of the ugly stuff (e.g. determining  model, kernel, and parameter selection) while at the same time providing optional accessors for dlib's more advanced functionality for more power users.

At this time, the library contains basic examples for classification and clustering, with more examples to come, including regression task, and applications to gesture recognition.

## Usage

The library is a work in progress and not everything works properly yet.  Use at own risk or check back later.

Please note the following compilation instructions. You should add ofxLearn's source to your project, but do not add the "libs" folder which contains dlib.  Dlib's source itself cannot be pulled into the project because it will cause naming collisions. Instead, go to the project's build settings, and add "../../../addons/ofxLearn/libs/" to your user header search paths.  
