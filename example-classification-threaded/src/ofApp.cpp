#include "ofApp.h"

/*
 This example is identical to example-classification, except that it
 uses an ofxLearnThreaded instead of ofxLearn.  This class allows you
 to run learning procedures inside of a separate thread, which can
 be useful because ofxLearn can take anywhere from seconds to minutes to
 run, depending on the size and properties of the data, and the training mode.
 After beginning a training procedure, you can track the status of the thread
 using ofxLearnThreaded::getTrained()
 */

//--------------------------------------------------------------
void ofApp::setup() {
    ofSetLogLevel(OF_LOG_VERBOSE);

    for (int i=0; i<1000; i++) {
        double x = ofRandom(ofGetWidth());
        double y = ofRandom(ofGetHeight());
        
        vector<double> instance;
        int label;
        
        instance.push_back(x);
        instance.push_back(y);
        
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
    
    classifier.beginTrainClassifier(CLASSIFICATION, FAST);   // can be FAST or ACCURATE
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
    
    else {
        ofSetColor(0, 255, 0);
        ofRect(40, 25, 450, 40);
        ofSetColor(0);
        ofDrawBitmapString("Currently training classification... please wait.", 50, 50);
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