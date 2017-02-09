#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    ofSetLogLevel(OF_LOG_VERBOSE);
    
    // create a noisy training set
    // objective is to predict y as function of x
    for (int i=0; i<1000; i++)
    {
        double x = ofRandom(ofGetWidth());
        double y = 0.00074 * pow(x, 2) + 0.0095*x + ofRandom(-80, 80);
        
        // in this example, we bound all input and output variables to (0,1).
        // this isn't strictly required but is best practice because it ensures
        // features are at parity in influence, and training is generally faster.
        // for MLP normalization to (0, 1) is required.
        x = ofClamp(ofMap(x, 0, ofGetWidth(), 0, 1), 0, 1);
        y = ofClamp(ofMap(y, 0, ofGetHeight(), 0, 1), 0, 1);
        
        // for this example, each instance contains one feature.
        // in general, an instance vector can contain any number
        // of elements, but must stay fixed for a single classifier/regressor
        vector<double> sample;
        sample.push_back(x);
        
        mlp.addSample(sample, y);
        svr.addSample(sample, y);
        
        trainingExamples.push_back(sample);
        trainingLabels.push_back(y);
    }
    
    // we have two different algorithms for doing regression:
    // MLP = multilayer perceptron (neural network)
    // SVR = support vector regression
    
    mlp.beginTraining();
    svr.beginTraining();
}

//--------------------------------------------------------------
void ofApp::update(){
    
}

//--------------------------------------------------------------
void ofApp::draw() {
    ofBackground(255);
    
    if (!mlp.isTrained() || !svr.isTrained()) {
        ofSetColor(0);
        ofDrawBitmapString("Please wait... Training clusterer in its own thread! ("+ofToString(ofGetFrameNum())+")", 50, 50);
        return;
    }
    
    // draw training set
    ofSetColor(255, 0, 0, 50);
    for (int i=0; i<trainingExamples.size(); i++) {
        vector<double> trainingExample = trainingExamples[i];
        ofCircle(ofGetWidth() * trainingExample[0], ofGetHeight() * trainingLabels[i], 5);
    }
    
    // predict regression for mouseX
    float x = (float) ofGetMouseX() / ofGetWidth();
    
    vector<double> sample;
    sample.push_back(x);
    
    double mlpPrediction = mlp.predict(sample);
    
    ofSetColor(0, 255, 0, 150);
    ofCircle(ofGetWidth() * x, ofGetHeight() * mlpPrediction, 20);
    ofSetColor(0);
    ofDrawBitmapString("MLP", ofGetWidth() * x, ofGetHeight() * mlpPrediction);
    
    double svrPrediction = svr.predict(sample);
    
    ofSetColor(0, 0, 255, 150);
    ofCircle(ofGetWidth() * x, ofGetHeight() * svrPrediction, 20);
    ofSetColor(0);
    ofDrawBitmapString("SVR", ofGetWidth() * x, ofGetHeight() * svrPrediction);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key==' ') {
        //regression.saveModel(ofToDataPath("testSave.dat"));
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