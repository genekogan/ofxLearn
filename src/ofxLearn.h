#pragma once

#include "ofMain.h"
#include "dlib/svm.h"
#include "dlib/mlp.h"


// sample data
typedef dlib::matrix<double, 0, 1>                  sample_type;
typedef dlib::radial_basis_kernel<sample_type>      kernel_type;
typedef dlib::vector_normalizer<sample_type>        normalizer_type;

// learning algorithm
typedef dlib::krr_trainer<kernel_type>              svm_trainer_type;

// decision function
typedef dlib::decision_function<kernel_type>        dec_funct_type;
typedef dlib::normalized_function<dec_funct_type>   funct_type;

// multiclass classification
typedef dlib::any_trainer<sample_type>              any_trainer;
typedef dlib::one_vs_one_trainer<any_trainer>       ovo_trainer;
typedef dlib::one_vs_one_decision_function
                <ovo_trainer, dec_funct_type>       ovo_d_funct_type;
typedef dlib::normalized_function<ovo_d_funct_type> ovo_funct_type;


// FAST means choose default parameters
// ACCURATE does grid parameter search to determine best parameters
enum TrainMode { FAST, ACCURATE };


class ofxLearn
{
public:
    
    // data
    void                addTrainingInstance(vector<double> instance, double label);
    void                addTrainingInstance(vector<double> instance);
    void                clearTrainingInstances();
    int                 getNumberTrainingInstances() { return samples.size(); }
    
    // model
    void                trainClassifier(TrainMode trainMode = ACCURATE);
    void                trainRegression(TrainMode trainMode = ACCURATE);
    
    int                 classify(vector<double> instance);
    double              predict(vector<double> instance);
    vector<int>         getClusters(int k);
    
    // IO
    void                saveModel(string filename);
    void                loadModel(string filename);

    
private:
    
    // data
    vector<sample_type> samples;
    vector<double>      labels;
    normalizer_type     normalizer;
    
    // classification
    svm_trainer_type    svm_trainer;
    ovo_trainer         trainer;
    ovo_funct_type      classification_function;
    
    // regression
    funct_type          regression_function;
};
