#pragma once

#include "ofMain.h"
#include "ofxLearn.h"


class ofxGraphicsFeatureMaker {
public:
    ofxGraphicsFeatureMaker();
    vector<double>      createInstanceFromPointArray(vector<ofVec2f> &points);
    void                drawInstanceFromPointArray(vector<double> &instance,
                                                   int x=0, int y=0, int width=100, int height=100);
protected:
    ofFbo               fbo;
    ofPixels            fboPixels;
    float               hop;
    int                 n;
};



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
    
    ofxLearn classifier;
    ofxGraphicsFeatureMaker maker;
    vector<double> instance;
    vector<ofVec2f> points;
    bool isCreatingInstance, lastInstanceIsTraining, isTrained;
    int lastLabel;
    
};


/*
 TO-DO
 =====
 save and load training instances to disk
 regression
 clustering
 opencv hand-tracking example
 leapmotion example
 --
 tutorials
 readme
 */