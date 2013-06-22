#include "testApp.h"

//--------------------------------------------------------------
void testApp::setup(){
    
    // We are randomly distributing NUMPOINTS instance vectors as points
    // in 3-d space, and sending them to our classifier object
    for (int i = 0; i < NUMPOINTS; i++) {
        instances[i].push_back( ofRandom(-500,500) );
        instances[i].push_back( ofRandom(-500,500) );
        instances[i].push_back( ofRandom(-500,500) );
        classifier.addTrainingInstance( instances[i] );
    }

    // we tell ofxLearn to assign our NUMPOINTS points into NUMCLUSTERS clusters.  
    // It returns a vector of integers specifying the cluster for each of the points
    clusters = classifier.findClusters(NUMCLUSTERS);
    
    for (int i = 0; i < clusters.size(); i++)
        cout << "Instance " << ofToString(i) << " " << ofToString(instances[i]) << " assigned to cluster " << ofToString(clusters[i]) << endl;
    
    // We randomize NUMCLUSTERS colors to visualize the clusters in the draw loop
    for (int i = 0; i < NUMCLUSTERS; i++)
        colors[i] = ofColor( ofRandom(255), ofRandom(255), ofRandom(255) );
}

//--------------------------------------------------------------
void testApp::update(){
}

//--------------------------------------------------------------
void testApp::draw() {
    glEnable(GL_DEPTH_TEST);
        
    // We display the NUMPOINTS points on the screen, and color them
    // according to whichever cluster they were assigned to by
    // the classifier.
    cam.begin();    
    for (int i = 0; i < NUMPOINTS; i++) {
        ofPushMatrix();
        ofTranslate(instances[i][0], instances[i][1], instances[i][2]);
        ofSetColor( colors[clusters[i]] );        
        ofSphere(10);
        ofPopMatrix();
    }
    cam.end();
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