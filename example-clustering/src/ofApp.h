#pragma once

#include "ofMain.h"
#include "ofxLearn.h"

#define NUMPOINTS 500
#define NUMCLUSTERS 5

class ofApp : public ofBaseApp{
public:
    void setup();
    void update();
    void draw();
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y);
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    ofxLearn learn;
    vector<double> instances[NUMPOINTS];
    vector<int> clusters;
    
    ofColor colors[NUMCLUSTERS];
    ofEasyCam cam;
};
