#pragma once

#include "ofMain.h"
#include "dlib/svm.h"
#include "dlib/mlp.h"

// samples
typedef dlib::matrix<double, 0, 1> sample_type;

// support vector machine (classifier)
typedef dlib::radial_basis_kernel<sample_type> kernel_type;
typedef dlib::krr_trainer<kernel_type> svm_trainer_type;

// multilayer perceptron (regressor)
typedef dlib::mlp::kernel_1a_c mlp_trainer_type;

// optimization
typedef dlib::one_vs_one_trainer<dlib::any_trainer<sample_type> > ovo_trainer;
typedef dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<kernel_type> > dec_funct_type;


class ofxLearn {
public:
    ofxLearn();
    void                addTrainingInstance(vector<double> instance, int label);
    void                addTrainingInstance(vector<double> instance, float label);
    void                addTrainingInstance(vector<double> instance);
    void                clearTrainingSet();
    void                trainClassifier();
    void                trainRegressor();
    vector<int>         findClusters(int k);    // todo: pointer to vector as argument

    int                 classify(vector<double> instance);
    float               predict(vector<double> instance);
    void                optimizeClassifier();
    int                 getNumberTrainingInstances() { return numInstances; }
    
    void                saveModel(string filename);
    void                loadModel(string filename);
    void                saveDataset(string filename);
    void                loadDataset(string filename);
    
    bool                isTrained;


    
    // alpha (learning rate), momentum
    
private:
    int                 numFeatures;
    int                 numInstances;
    vector<sample_type> samples;
    vector<double>      labels;
    
    mlp_trainer_type    *mlp_trainer;
    svm_trainer_type    svm_trainer;
    
    ovo_trainer         trainer;
    dec_funct_type      decision_function;
};

// TODO
// - destructor/clean up
// - updat save and mode for regressor



