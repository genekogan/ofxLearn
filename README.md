# ofxLearn

ofxLearn will be a general-purpose machine learning library for OpenFrameworks, built on top of [dlib](http://dlib.net/).  It is currently under development and does not yet function properly.  Please check back later.

## Promises

ofxLearn will support classification, regression, and unsupervised clustering tasks. The goal for it is to be a high-level wrapper to powerful machine learning routines which takes care of the ugly stuff (e.g. determining  model, kernel, and parameter selection) while at the same time providing optional accessors for dlib's more advanced functionality for more power users.

## Usage

The library is not yet in a ready state, but if you want to use it anyway, please note the following instructions.  You should add ofxLearn's source to your project, but do not add the "libs" folder; dlib's source itself cannot be pulled into the project because it will cause naming collisions. Instead, go to the project's build settings, and add "../../../addons/ofxLearn/libs/" to your user header search paths.  
