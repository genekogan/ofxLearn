#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    ofSetLogLevel(OF_LOG_VERBOSE);
    
    // add 5000 samples to training set
    for (int i=0; i<5000; i++)
    {
        // our samples have two features: x, and y,
        // which are bound between (0, 1).
        // note: your feature values don't need to be between 0, 1
        // but best practice is to pre-normalize because it's faster
        // and ensures parity of feature influences

        vector<double> sample;
        sample.push_back(ofRandom(1));
        sample.push_back(ofRandom(1));

        // our label contains 3 possible classes, which roughly
        // correspond to the distance from the center of the screen
        // with some noise thrown in
        int label;
        float distFromCenter = ofDist(sample[0], sample[1], 0.5, 0.5);
        if (distFromCenter < ofRandom(0.1, 0.25)) {
            label = 1;
        }
        else if (distFromCenter < ofRandom(0.15, 0.45)) {
            label = 2;
        }
        else {
            label = 3;
        }
        
        // save our samples
        trainingExamples.push_back(sample);
        trainingLabels.push_back(label);
        
        // add sample to our classifier
        classifier.addTrainingInstance(sample, label);
    }
    
    classifier.train();
    
    // or you can make a grid parameter search to find the
    // best parameters for training. this takes much longer
    // but should be more accurate.
    //classifier.trainWithGridParameterSearch()
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
        }
        else if (trainingLabel == 2) {
            ofSetColor(0, 255, 0);
        }
        else if (trainingLabel == 3) {
            ofSetColor(0, 0, 255);
        }
        ofCircle(trainingExample[0] * ofGetWidth(), trainingExample[1] * ofGetHeight(), 5);
    }

    // classify a new sample
    vector<double> sample;
    sample.push_back((double)ofGetMouseX()/ofGetWidth());
    sample.push_back((double)ofGetMouseY()/ofGetHeight());

    int label = classifier.predict(sample);
    
    if (label == 1) {
        ofSetColor(255, 0, 0);
    }
    else if (label == 2) {
        ofSetColor(0, 255, 0);
    }
    else if (label == 3) {
        ofSetColor(0, 0, 255);
    }
    ofCircle(ofGetMouseX(), ofGetMouseY(), ofMap(sin(0.1*ofGetFrameNum()), -1, 1, 5, 35));
    ofSetColor(0);
    ofDrawBitmapString("class "+ofToString(label), ofGetMouseX()-25, ofGetMouseY());

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