#include "testApp.h"

//--------------------------------------------------------------
void testApp::setup() {

    // create a noisy training set
    // objective is to predict y as function of x
    for (int i=0; i<300; i++) {
        double x = ofRandom(ofGetWidth());
        double y = 0.00074 * pow(x, 2) + 0.0095*x + ofRandom(-80, 80);
        
        // each instance contains one feature
        vector<double> instance;
        instance.push_back(x);
        
        trainingExamples.push_back(instance);
        trainingLabels.push_back(y);

        regression.addTrainingInstance(instance, y);
    }

    regression.trainRegression();
 
}

//--------------------------------------------------------------
void testApp::update(){
}

//--------------------------------------------------------------
void testApp::draw() {
    
    // draw training set
    ofBackground(255);
    ofSetColor(255, 0, 0);
    for (int i=0; i<trainingExamples.size(); i++) {
        vector<double> trainingExample = trainingExamples[i];
        ofCircle(trainingExample[0], trainingLabels[i], 5);
    }

    // predict regression for mouseX
    vector<double> instance;
    instance.push_back(ofGetMouseX());
    double label = regression.predict(instance);

    ofSetColor(0, 255, 0);
    ofCircle(ofGetMouseX(), label, ofMap(sin(0.1*ofGetFrameNum()), -1, 1, 5, 35));
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