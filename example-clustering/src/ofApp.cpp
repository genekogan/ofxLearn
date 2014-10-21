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
        learn.addTrainingInstance( instances[i] );
    }
    
    
    // we tell ofxLearn to assign our NUMPOINTS points into NUMCLUSTERS clusters.
    // It begins the clustering process in a separate thread
    learn.trainClusters(NUMCLUSTERS);
    
    
    // We randomize NUMCLUSTERS colors to visualize the clusters in the draw loop
    // once they have been returned by ofxLearn
    for (int i = 0; i < NUMCLUSTERS; i++) {
        colors[i] = ofColor( ofRandom(255), ofRandom(255), ofRandom(255) );
    }
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
    
    ofBackground(255);
    
    // while the clustering is training, it's not ready yet
    if (learn.getTraining()) {
        ofSetColor(0);
        ofDrawBitmapString("Currently clustering points. Please wait...", 50, 50);
    }
    
  
    // once clustering is finished, we can display the points in their clusters
    else if (learn.getTrained()) {
        
        // get the clusters
        clusters = learn.getClusters();
        
        // We display the NUMPOINTS points on the screen, and color them
        // according to whichever cluster they were assigned to by the classifier.
        cam.begin();
        ofEnableDepthTest();
        for (int i = 0; i < NUMPOINTS; i++) {
            ofPushMatrix();
            ofTranslate(instances[i][0], instances[i][1], instances[i][2]);
            ofSetColor( colors[clusters[i]] );
            ofDrawSphere(10);
            ofPopMatrix();
        }
        ofDisableDepthTest();
        cam.end();
    }

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
