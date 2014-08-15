#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    
    // create a noisy training set
    // objective is to predict y as function of x
    for (int i=0; i<300; i++) {
        double x = ofRandom(ofGetWidth());
        double y = 0.00074 * pow(x, 2) + 0.0095*x + ofRandom(-80, 80);
        
        // for this example, each instance contains one feature.
        // in general, an instance vector can contain any number
        // of elements, but must stay fixed for a single classifier/regressor
        vector<double> instance;
        instance.push_back(x);
        
        regression.addTrainingInstance(instance, y);
        
        trainingExamples.push_back(instance);
        trainingLabels.push_back(y);
    }
    
    // can train either FAST or ACCURATE. FAST uses
    // default parameters whereas ACCURATE attempts
    // a grid parameter search to find optimal parameters.
    regression.trainRegression(ACCURATE);
}

//--------------------------------------------------------------
void ofApp::update(){
    
}

//--------------------------------------------------------------
void ofApp::draw() {
    
    // draw training set
    ofBackground(255);
    ofSetColor(255, 0, 0);
    for (int i=0; i<trainingExamples.size(); i++) {
        vector<double> trainingExample = trainingExamples[i];
        ofCircle(trainingExample[0], trainingLabels[i], 5);
    }
    
    // predict regression for mouseX
    float x = ofGetMouseX();
    
    vector<double> instance;
    instance.push_back(x);
    double label = regression.predict(instance);
    
    ofSetColor(0, 255, 0);
    ofCircle(x, label, ofMap(sin(0.1*ofGetFrameNum()), -1, 1, 3, 30));
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key==' ') {
        regression.saveModel("testsave.dat");
    }
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