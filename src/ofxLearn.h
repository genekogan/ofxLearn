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


// enums for training options
enum TrainMode { FAST, ACCURATE };
enum LearnMode { CLASSIFICATION, REGRESSION_SVM, REGRESSION_MLP, CLUSTERING };



class ofxLearn
{
public:
    ofxLearn();
    
    // data
    void                addTrainingInstance(vector<double> instance, double label);
    void                addTrainingInstance(vector<double> instance);
    void                clearTrainingInstances();
    int                 getNumberTrainingInstances() { return samples.size(); }
    
    // model
    void                trainClassifier(LearnMode learnMode = CLASSIFICATION, TrainMode trainMode = ACCURATE);
    void                trainRegression(LearnMode learnMode = REGRESSION_SVM, TrainMode trainMode = ACCURATE);
    double              predict(vector<double> instance);
    
    // cluster
    void                trainClusters(int numClusters);
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
    void                setMlpTargetRmse(float t) { mlpTargetRmse = t; }
    int                 getMlpNumHiddenLayers() { return mlpNumHiddenLayers; }
    int                 getMlpMaxSamples() { return mlpMaxSamples; }
    float               getMlpTargetRmse() { return mlpTargetRmse; }
    
    
protected:
    
    void                trainRegressionSvm(TrainMode trainMode);
    void                trainRegressionMlp(TrainMode trainMode);
    
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
    float               mlpTargetRmse;
    int                 mlpMaxSamples;
    
    // clustering
    vector<int>         clusters;
    int                 numClusters;
    
    // learn mode
    LearnMode           learnMode;
    TrainMode           trainMode;
};




class ofxLearnThreaded : public ofxLearn, public ofThread
{
public:
    ofxLearnThreaded() : ofxLearn() {
        trained = false;
    }
    
    bool getTrained() {return trained;}

    void beginTrainClassifier(LearnMode learnMode = CLASSIFICATION, TrainMode trainMode = ACCURATE) {
        this->learnMode = learnMode;
        this->trainMode = trainMode;
        trained = false;
        startThread();
    }
    
    void beginTrainRegression(LearnMode learnMode = REGRESSION_SVM, TrainMode trainMode = ACCURATE) {
        this->learnMode = learnMode;
        this->trainMode = trainMode;
        trained = false;
        startThread();
    }
    
    void beginTrainClusters(int numClusters) {
        this->learnMode = CLUSTERING;
        this->numClusters = numClusters;
        trained = false;
        startThread();
    }
    
    
private:
    
    void threadedFunction() {
        while (isThreadRunning()) {
            if (lock()) {
                if      (learnMode == CLASSIFICATION) {
                    trainClassifier(learnMode, trainMode);
                }
                else if (learnMode == REGRESSION_SVM || learnMode == REGRESSION_MLP) {
                    trainRegression(learnMode, trainMode);
                }
                else if (learnMode == CLUSTERING) {
                    trainClusters(numClusters);
                }
                trained = true;
                unlock();
                stopThread();
            }
            else {
                ofLogWarning("threadedFunction()") << "Unable to lock mutex.";
            }
        }
    }
    
    bool trained;
};

