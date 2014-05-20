#include "testApp.h"

//--------------------------------------------------------------
void testApp::setup() {
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
    
    classifier.trainClassifier();    
}

//--------------------------------------------------------------
void testApp::update(){
}

//--------------------------------------------------------------
void testApp::draw() {
    
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
    
    vector<double> instance;
    instance.push_back(ofGetMouseX());
    instance.push_back(ofGetMouseY());
    
    int label = classifier.classify(instance);
    
    
    if (label == 1) {
        ofSetColor(255, 0, 0);
    } else if (label == 2) {
        ofSetColor(0, 255, 0);
    } else if (label == 3) {
        ofSetColor(0, 0, 255);
    }
    ofCircle(ofGetMouseX(), ofGetMouseY(), ofMap(sin(0.1*ofGetFrameNum()), -1, 1, 5, 35));
    
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y){
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){
    
}