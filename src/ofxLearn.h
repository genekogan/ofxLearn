#pragma once

#include "ofMain.h"

#include "dlib/svm.h"
#include "dlib/mlp.h"
#include "dlib/svm/svm_threaded.h"
#include "dlib/matrix/matrix_abstract.h"

#include "dlib/statistics/statistics.h"


// sample type
typedef dlib::matrix<double, 0, 1>                  sample_type;
typedef dlib::matrix<double, 0, 0>                  matrix;

// kernel types
typedef dlib::radial_basis_kernel<sample_type>      rbf_kernel_type;
typedef dlib::polynomial_kernel<sample_type>        poly_kernel_type;

// trainer types
typedef dlib::any_trainer<sample_type>              any_trainer;
typedef dlib::one_vs_one_trainer<any_trainer>       ovo_trainer;
typedef dlib::mlp::kernel_1a_c                      mlp_trainer_type;


class ofxLearn {
public:
    ofxLearn() { }
    virtual ~ofxLearn() { }
    virtual void train() { }
    virtual void saveModel(string path) { }
    virtual void loadModel(string path) { }
    inline sample_type vectorToSample(vector<double> sample_);
};


class ofxLearnSupervised : public ofxLearn {
public:
    ofxLearnSupervised() : ofxLearn() {}
    void addSample(vector<double> sample, double label);
    void addSample(sample_type sample, double label);
    void clearTrainingInstances();
    virtual double predict(vector<double> & sample) { }
    virtual double predict(sample_type & sample) { }
protected:
    vector<sample_type> samples;
    vector<double> labels;
};

class ofxLearnUnsupervised : public ofxLearn{
public:
    ofxLearnUnsupervised() : ofxLearn() {}
    void addSample(vector<double> sample);
    void addSample(sample_type sample);
    void clearTrainingInstances();
protected:
    vector<sample_type> samples;
};

class ofxLearnMLP : public ofxLearnSupervised {
public:
    ofxLearnMLP();
    ~ofxLearnMLP();
    void train();
    double predict(vector<double> & sample);
    double predict(sample_type & sample);
    void setHiddenLayers(int hiddenLayers) {this->hiddenLayers = hiddenLayers;}
    void setTargetRmse(float targetRmse) {this->targetRmse = targetRmse;}
    void setMaxSamples(int maxSamples) {this->maxSamples = maxSamples;}
    int getHiddenLayers() {return hiddenLayers;}
    float getTargetRmse() {return targetRmse;}
    int getMaxSamples() {return maxSamples;}
    mlp_trainer_type * getTrainer() {return mlp_trainer;}
private:
    mlp_trainer_type *mlp_trainer;
    int hiddenLayers;
    float targetRmse;
    int maxSamples;
};


class ofxLearnSVR : public ofxLearnSupervised
{
public:
    ofxLearnSVR();
    ~ofxLearnSVR();
    void train();
    void trainWithGridParameterSearch();
    double predict(vector<double> & sample);
    double predict(sample_type & sample);
private:
    dlib::svr_trainer<rbf_kernel_type> trainer;
    dlib::decision_function<rbf_kernel_type> df;
};


class ofxLearnSVM : public ofxLearnSupervised
{
public:
    ofxLearnSVM();
    ~ofxLearnSVM();
    void train();
    void trainWithGridParameterSearch();
    double predict(vector<double> & sample);
    double predict(sample_type & sample);
    void saveModel(string path);
    void loadModel(string path);
private:
    ovo_trainer trainer;
    //dlib::svm_nu_trainer<poly_kernel_type> poly_trainer;
    dlib::krr_trainer<rbf_kernel_type> rbf_trainer;
    dlib::one_vs_one_decision_function<ovo_trainer> df;
};

class ofxLearnKMeans : public ofxLearnUnsupervised {
public:
    ofxLearnKMeans();
    int getNumClusters() {return numClusters;}
    void setNumClusters(int numClusters);
    void train();
    vector<int> & getClusters() {return clusters;}
private:
    vector<int> clusters;
    int numClusters;
};

class ofxLearnPCA : public ofxLearnUnsupervised {
public:
    ofxLearnPCA();
    void pca(int numComponents);
    vector<double> project(vector<double> sample);
    vector<vector<double> > getProjectedSamples();
protected:
    
    
    matrix U, E, V;
};



////////////////////////
class ofxLearnThreaded : public ofxLearn, public ofThread {
public:
    ofxLearnThreaded();
    ~ofxLearnThreaded();
    void beginTraining();
    template <typename L, typename M> void beginTraining(L *listener, M method);
    bool isTrained() {return trained;}
private:
    void threadedFunction();
    virtual void threadedTrainer() {};
    bool trained;
    ofEvent<void> finishedTrainingE;
};

template <typename L, typename M>
void ofxLearnThreaded::beginTraining(L *listener, M method){
    ofAddListener(finishedTrainingE, listener, method);
    beginTraining();
}

class ofxLearnMLPThreaded : public ofxLearnMLP, public ofxLearnThreaded {
    void threadedTrainer() {ofxLearnMLP::train();}
};

class ofxLearnSVRThreaded : public ofxLearnSVR, public ofxLearnThreaded {
    void threadedTrainer() {ofxLearnSVR::train();}
};

class ofxLearnSVMThreaded : public ofxLearnSVM, public ofxLearnThreaded {
    void threadedTrainer() {ofxLearnSVM::train();}
};

class ofxLearnKMeansThreaded : public ofxLearnKMeans, public ofxLearnThreaded {
    void threadedTrainer() {ofxLearnKMeans::train();}
};

