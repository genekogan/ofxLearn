#include "ofApp.h"

/*
 This example is identical to example-clustering, except that it
 uses an ofxLearnThreaded instead of ofxLearn.  This class allows you
 to run learning procedures inside of a separate thread, which can
 be useful because ofxLearn can take anywhere from seconds to minutes to
 run, depending on the size and properties of the data, and the training mode.
 After beginning a training procedure, you can track the status of the thread
 using ofxLearnThreaded::getTrained()
*/

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
    clusters.resize(NUMPOINTS);

    // We randomize NUMCLUSTERS colors to visualize the clusters in the draw loop
    for (int i = 0; i < NUMCLUSTERS; i++) {
        colors[i] = ofColor( ofRandom(255), ofRandom(255), ofRandom(255) );
    }

    // we tell ofxLearn to assign our NUMPOINTS points into NUMCLUSTERS clusters.
    // It returns a vector of integers specifying the cluster for each of the points
    learn.beginTrainClusters(NUMCLUSTERS);
    
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
    
    ofBackground(255);
    
    // if clusters are available, get them
    if (learn.getTrained()) {
        clusters = learn.getClusters();
    }
    else {
        ofSetColor(0, 255, 0);
        ofFill();
        ofRect(40, 25, 400, 40);
        ofSetColor(0);
        ofDrawBitmapString("Currently determining clusters... please wait.", 50, 50);
    }
    
    
    // We display the NUMPOINTS points on the screen, and color them
    // according to whichever cluster they were assigned to by
    // the classifier.
    cam.begin();
    for (int i = 0; i < NUMPOINTS; i++) {
        ofPushMatrix();
        ofEnableDepthTest();
        ofTranslate(instances[i][0], instances[i][1], instances[i][2]);
        
        if (learn.getTrained()) {
            ofSetColor(colors[clusters[i]]);
        }
        else {
            ofSetColor(0);
        }
        
        ofFill();
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
