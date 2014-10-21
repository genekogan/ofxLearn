#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    ofSetLogLevel(OF_LOG_VERBOSE);

    // create a training set with 500 examples
    // objective is to predict class of new (x, y) inputs

    for (int i=0; i<500; i++) {
        double x = ofRandom(ofGetWidth());
        double y = ofRandom(ofGetHeight());
        
        vector<double> instance;
        int label;
        
        instance.push_back(x);
        instance.push_back(y);
        
        // we assign each example one of three classes (1/2/3) depending roughly
        // on its distance from center. but we give it a little bit of noise so
        // the classification is not too obvious and can generalize better
        
        float distFromCenter = ofDist(x, y, ofGetWidth()/2, ofGetHeight()/2);
        if (distFromCenter < ofRandom(100, 240))
            label = 1;
        else if (distFromCenter < ofRandom(150, 450))
            label = 2;
        else
            label = 3;
        
        trainingExamples.push_back(instance);
        trainingLabels.push_back(label);
        classifier.addTrainingInstance(instance, label);
    }
    
    
    // can train either FAST or ACCURATE. FAST uses
    // default parameters whereas ACCURATE attempts
    // a grid parameter search to find optimal parameters.
    // CLASSIFICATION does classification using a support vector machine (default).

    classifier.trainClassifier(CLASSIFICATION, ACCURATE);
    
    
    // after calling trainClassifier(), the learning algorithm begins
    // processing in its own thread. the amount of time required to finish
    // training can vary greatly from nearly instantly to several minutes,
    // depending on the size of the training set, the accuracy setting used,
    // and the properties of the data.

}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw() {
    
    // draw training set
    ofBackground(255);
    for (int i=0; i<trainingExamples.size(); i++) {
        vector<double> trainingExample = trainingExamples[i];
        int trainingLabel = trainingLabels[i];
        if (trainingLabel == 1) {
            ofSetColor(255, 0, 0);
        } else if (trainingLabel == 2) {
            ofSetColor(0, 255, 0);
        } else if (trainingLabel == 3) {
            ofSetColor(0, 0, 255);
        }
        ofCircle(trainingExample[0], trainingExample[1], 5);
    }
    

    // while the classification is training, you can get a status update on it
    if (classifier.getTraining()) {
        ofSetColor(255, 240);
        ofRect(5, 5, 500, 50);
        ofSetColor(0);
        ofDrawBitmapString(classifier.getStatusString(), 10, 20);
        ofDrawBitmapString("Classifier progress: "+ofToString(int(100 * classifier.getProgress()))+"%", 10, 40);
    }
    
    
    // once classification is finished, we can try to predict the output of new inputs
    if (classifier.getTrained()) {
        vector<double> instance;
        instance.push_back(ofGetMouseX());
        instance.push_back(ofGetMouseY());
        
        int label = classifier.predict(instance);
        
        if (label == 1) {
            ofSetColor(255, 0, 0);
        } else if (label == 2) {
            ofSetColor(0, 255, 0);
        } else if (label == 3) {
            ofSetColor(0, 0, 255);
        }
        ofCircle(ofGetMouseX(), ofGetMouseY(), ofMap(sin(0.1*ofGetFrameNum()), -1, 1, 5, 35));
    }
    
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    
}