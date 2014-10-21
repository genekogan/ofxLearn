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

// multilayer perceptron (neural net)
typedef dlib::mlp::kernel_1a_c                      mlp_trainer_type;


// choose training mode: fast or accurate
enum TrainMode {
    FAST,
    ACCURATE };

// choose algorithm: classification, regression, clustering
enum LearnMode {
    CLASSIFICATION,
    REGRESSION_SVM, // support vector machine
    REGRESSION_MLP, // multilayer perceptron
    CLUSTERING };


class ofxLearn : public ofThread
{
public:
    ~ofxLearn();
    ofxLearn();
    
    // data
    void                addTrainingInstance(vector<double> instance, double label);
    void                addTrainingInstance(vector<double> instance);
    void                clearTrainingInstances();
    int                 getNumberTrainingInstances() { return samples.size(); }
    
    // model
    void                trainClassifier(LearnMode learnMode, TrainMode trainMode = ACCURATE);
    void                trainClusters(int numClusters);
    
    double              predict(vector<double> instance);
    vector<int> &       getClusters();

    // IO
    void                saveModel(string path);
    void                loadModel(LearnMode learnMode, string path);
    
    // get classifiers
    ovo_funct_type      getClassifier()    { return classification_function; }
    funct_type          getRegressionSvm() { return regression_function; }
    mlp_trainer_type*   getRegressionMlp() { return mlp_trainer; }
    
    // mlp paramrters
    void                setMlpNumHiddenLayers(int n) { mlpNumHiddenLayers = n; }
    void                setMlpMaxSamples(int n) { mlpMaxSamples = n; }
    void                setMlpTargetRmse(float rmse) { mlpTargetRmse = rmse; }
    int                 getMlpNumHiddenLayers() { return mlpNumHiddenLayers; }
    int                 getMlpMaxSamples() { return mlpMaxSamples; }
    int                 getMlpNumSamples() { return mlpSamples; }
    float               getMlpTargetRmse() { return mlpTargetRmse; }
    float               getMlpRmse() { return mlpRmse; }

    
    // get status
    bool                getTraining() { return isTraining; }
    bool                getTrained() { return isTrained; }
    float               getProgress() { return progress; }
    string              getStatusString() { return status; }
    

private:
    
    void                threadedFunction();
    
    void                trainClassifierSvm(TrainMode trainMode);
    void                trainRegressionSvm(TrainMode trainMode);
    void                trainRegressionMlp(TrainMode trainMode);
    void                trainKMeansClusters();
    
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
    mlp_trainer_type    *mlp_trainer;
    int                 mlpNumHiddenLayers;
    float               mlpTargetRmse, mlpRmse;
    int                 mlpMaxSamples, mlpSamples;
    
    // clustering
    int                 numClusters;
    vector<int>         clusters;
        
    // mode + status
    LearnMode           learnMode;
    TrainMode           trainMode;
    bool                isTraining;
    bool                isTrained;
    float               progress;
    string              status;
};
