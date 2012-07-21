#pragma once

#include "ofMain.h"
#include "dlib/svm.h"

typedef dlib::matrix<double, 64, 1> sample_type;
typedef dlib::radial_basis_kernel<sample_type> kernel_type;
typedef dlib::krr_trainer<kernel_type> binary_trainer_type;
typedef dlib::one_vs_one_trainer<dlib::any_trainer<sample_type> > ovo_trainer;
typedef dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<kernel_type> > dec_funct_type;

class ofxLearn {
public:
    ofxLearn();
    void                addTrainingInstance(vector<double> instance, int label);
    void                addTrainingInstance(vector<double> instance);
    void                clearTrainingSet();
    void                trainClassifier();
    void                trainRegressor();
    vector<int>         findClusters(int k);
    int                 predict(vector<double> instance);
    void                optimizeClassifier();
    int                 getNumberTrainingInstances() { return numInstances; }
    
    void                saveModel(string filename);
    void                loadModel(string filename);
    void                saveDataset(string filename);
    void                loadDataset(string filename);
        
    bool                isTrained;

protected:
    vector<sample_type> samples;
    vector<double>      labels;
    binary_trainer_type svm_trainer;
    ovo_trainer         trainer;
    dec_funct_type      decision_function;
    int                 numInstances;
    
    int                 mode;

};



