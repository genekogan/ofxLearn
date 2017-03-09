#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    
    // add 7 samples with 5 dimensions
    learn.addSample(vector<double>{1, 0.5, 3, 1, 5});
    learn.addSample(vector<double>{2, 0.7, 3.3, 0.9, 4.2});
    learn.addSample(vector<double>{1, 4, -0.4, 9, 3.2});
    learn.addSample(vector<double>{1.5, 5.2, 0.1, 8, 2.6});
    learn.addSample(vector<double>{0.2, -0.5, 0.3, -1.2, 7.4});
    learn.addSample(vector<double>{0.2, -0.5, 0.31, -1.2, 7.3});
    learn.addSample(vector<double>{9, 8, 6, 3, -1.0});
    
    // Run PCA to reduce to 3 dimensions
    learn.pca(3);
    
    // Get original dataset projected onto principal components
    vector<vector<double> > projectedSamples = learn.getProjectedSamples();
    for (int i=0; i<projectedSamples.size(); i++) {
        cout << "Dataset sample " << i << " projected: " << ofToString(projectedSamples[i]) << endl;
    }
    
    // Project a new point onto the principal components
    vector<double> newSample = {5, 4, 1, -2, 3.2};
    vector<double> newSampleProjected = learn.project(newSample);
    cout << "New sample projected " << ofToString(newSampleProjected) << endl;

    // save PCA results
    learn.save(ofToDataPath("myPca.dat"));
    
    // load PCA back
    learn.load(ofToDataPath("myPca.dat"));

    
    // Somtimes, it's better to reduce the dimensionality of your dataset by
    // simply multiplying it by a random matrix with as many columns as components
    // you want to keep. This might reconstruct the dataset more poorly than PCA, but
    // when dimensionality is high to begin with, the difference is very small and the
    // random projection is orders of magnitude faster than PCA.
    
    rp.addSample(vector<double>{1, 0.5, 3, 1, 5});
    rp.addSample(vector<double>{2, 0.7, 3.3, 0.9, 4.2});
    rp.addSample(vector<double>{1, 4, -0.4, 9, 3.2});
    rp.addSample(vector<double>{1.5, 5.2, 0.1, 8, 2.6});
    rp.addSample(vector<double>{0.2, -0.5, 0.3, -1.2, 7.4});
    rp.addSample(vector<double>{0.2, -0.5, 0.31, -1.2, 7.3});
    rp.addSample(vector<double>{9, 8, 6, 3, -1.0});

    rp.randomProjection(3);
    
    // Get original dataset projected onto principal components
    vector<vector<double> > projectedSamples2 = rp.getProjectedSamples();
    for (int i=0; i<projectedSamples2.size(); i++) {
        cout << "Dataset sample " << i << " projected: " << ofToString(projectedSamples2[i]) << endl;
    }
    
    // Project a new point onto the principal components
    vector<double> newSample2 = {5, 4, 1, -2, 3.2};
    vector<double> newSampleProjected2 = rp.project(newSample2);
    cout << "New sample projected " << ofToString(newSampleProjected2) << endl;

    // save random projection results
    learn.save(ofToDataPath("myRp.dat"));
    
    // load random projection back
    learn.load(ofToDataPath("myRp.dat"));

}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

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
