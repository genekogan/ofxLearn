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



typedef dlib::mlp::kernel_1a_c mlp_trainer_type;



class ofxLearn
{
public:
    
    // data
    void                addTrainingInstance(vector<double> instance, double label);
    void                addTrainingInstance(vector<double> instance);
    void                clearTrainingSet() { samples.clear(); }
    int                 getNumberTrainingInstances() { return samples.size(); }
    
    // model
    void                trainClassifier();
    void                trainRegression();
    
    int                 classify(vector<double> instance);
    double              predict(vector<double> instance);
    vector<int>         getClusters(int k);
    
    // IO
    void                saveModel(char *filename);
    void                loadModel(char *filename);
    
    
    
    
    // MLP
    void                trainRegression2();
    double              predict2(vector<double> instance);

    
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
    
    

    // MLP
    mlp_trainer_type    *mlp_trainer;

};
