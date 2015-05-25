#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetLogLevel(OF_LOG_VERBOSE);
    
    // We are randomly distributing NUMPOINTS instance vectors as points
    // in 3-d space, and sending them to our classifier object
    for (int i = 0; i < NUMPOINTS; i++) {
        instances[i].push_back( ofRandom(-500,500) );
        instances[i].push_back( ofRandom(-500,500) );
        instances[i].push_back( ofRandom(-500,500) );
        clusterer.addTrainingInstance( instances[i] );
    }
    
    // we tell ofxLearn to assign our NUMPOINTS points into NUMCLUSTERS clusters.
    // It returns a vector of integers specifying the cluster for each of the points
    clusterer.setNumClusters(NUMCLUSTERS);

    // We randomize NUMCLUSTERS colors to visualize the clusters in the draw loop
    for (int i = 0; i < NUMCLUSTERS; i++) {
        colors[i] = ofColor( ofRandom(255), ofRandom(255), ofRandom(255) );
    }
    
    // beginTraining() sets trainer off in its own thread, as
    // opposed to train() which will block the frameloop until
    // it's done training
    //clusterer.beginTraining();
    
    // you can also supply a callback function to be alerted once training is done
    clusterer.beginTraining(this, &ofApp::callbackTrainingDone);
}

//--------------------------------------------------------------
void ofApp::update(){
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(255);
    
    if (!clusterer.isTrained()) {
        ofSetColor(0);
        ofDrawBitmapString("Please wait... Training clusterer in its own thread! ("+ofToString(ofGetFrameNum())+")", 50, 50);
        return;
    }

    // once training is done, we can access the clusters
    
    clusters = clusterer.getClusters();
    
    // We display the NUMPOINTS points on the screen, and color them
    // according to whichever cluster they were assigned to by
    // the classifier.
    cam.begin();
    for (int i = 0; i < NUMPOINTS; i++) {
        ofPushMatrix();
        ofEnableDepthTest();
        ofTranslate(instances[i][0], instances[i][1], instances[i][2]);
        ofSetColor( colors[clusters[i]] );
        ofDrawSphere(10);
        ofDisableDepthTest();
        ofPopMatrix();
    }
    cam.end();
    
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
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
