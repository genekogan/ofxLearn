#pragma once

#include "ofMain.h"

#include "dlib/svm.h"
#include "dlib/mlp.h"
#include "dlib/svm/svm_threaded.h"
#include "dlib/matrix/matrix_abstract.h"



// http://dlib.net/mlp_ex.cpp.html
// http://dlib.net/svr_ex.cpp.html
// http://dlib.net/multiclass_classification_ex.cpp.html



// TODO
//  x mlp
//  x svr
//  x multiclass svm
//  - cross validation
//  - grid parameter search
//  - pca
//  - sample from gaussian (http://dlib.net/3d_point_cloud_ex.cpp.html)
//  -






// sample type
typedef dlib::matrix<double, 0, 1>                  sample_type;

// kernel types
typedef dlib::radial_basis_kernel<sample_type>      rbf_kernel_type;
typedef dlib::polynomial_kernel<sample_type>        poly_kernel_type;

// trainer types
typedef dlib::any_trainer<sample_type>              any_trainer;
typedef dlib::one_vs_one_trainer<any_trainer>       ovo_trainer;
typedef dlib::mlp::kernel_1a_c                      mlp_trainer_type;

// decision type?
//typedef dlib::decision_function<poly_kernel_type>   df_poly_type;
//typedef dlib::one_vs_one_decision_function<ovo_trainer, df_poly_type> ovo_df_poly_type;
//typedef dlib::one_vs_one_decision_function<ovo_trainer, rbf_kernel_type> ovo_df_rbf_type;


class ofxLearn
{
public:
    ofxLearn() { }
    virtual ~ofxLearn() { }
    void svd();
    
    virtual void train() { }
    
    inline sample_type vectorToSample(vector<double> sample_);
    
    double sinc(double x){return x == 0 ? 1 : sin(x)/x;}
};



class ofxLearnSupervised : public ofxLearn
{
public:
    ofxLearnSupervised() : ofxLearn() {}
    
    void addTrainingInstance(vector<double> sample, double label);
    void addSample(sample_type sample, double label);
    void clearTrainingInstances();
    
    virtual double predict(vector<double> sample) { }
    
protected:

    vector<sample_type> samples;
    vector<double> labels;
    
};

class ofxLearnUnsupervised : public ofxLearn
{
public:
    ofxLearnUnsupervised() : ofxLearn() {}
    
    void addTrainingInstance(vector<double> sample);
    void addSample(sample_type sample);
    void clearTrainingInstances();
    
protected:
    
    vector<sample_type> samples;
    
};


class ofxLearnMLP : public ofxLearnSupervised
{
public:
    ofxLearnMLP();
    ~ofxLearnMLP();
    
    void train();
    double predict(vector<double> sample);

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
    double predict(vector<double> sample);

private:
    
    void trainCrossValidate() {
        randomize_samples(samples, labels);
        cout << "MSE and R-Squared: "<< cross_validate_regression_trainer(trainer, samples, labels, 5) << endl;
    }
    
    dlib::svr_trainer<rbf_kernel_type> trainer;
    dlib::decision_function<rbf_kernel_type> df;
};


class ofxLearnSVM : public ofxLearnSupervised
{
public:
    ofxLearnSVM();
    ~ofxLearnSVM();
    
    void train();
    double predict(vector<double> sample);

private:
    
    void generate_data (std::vector<sample_type>& samples,std::vector<double>& labels);
    
    ovo_trainer trainer;
    
    dlib::krr_trainer<rbf_kernel_type> rbf_trainer;
    dlib::svm_nu_trainer<poly_kernel_type> poly_trainer;
    
    dlib::one_vs_one_decision_function<ovo_trainer> df;
};


class ofxLearnKMeans : public ofxLearnUnsupervised
{
public:
    ofxLearnKMeans();
    
    void setNumClusters(int numClusters);
    void train();
    vector<int> & getClusters() {return clusters;}

private:
    vector<int> clusters;
    int numClusters;
};




////////////////////////

class ofxLearnThreaded : public ofxLearn, public ofThread
{
public:
    ofxLearnThreaded() : ofxLearn()
    {
        trained = false;
    }
    
    ~ofxLearnThreaded() {
        finishedTrainingE.clear();
        finishedTrainingE.disable();
    }
    
    void beginTraining()
    {
        trained = false;
        startThread();
    }
    
    template <typename L, typename M>
    void beginTraining(L *listener, M method) {
        ofAddListener(finishedTrainingE, listener, method);
        beginTraining();
    }
    
    bool isTrained() {return trained;}
    
private:
    
    void threadedFunction()
    {
        while (isThreadRunning())
        {
            if (lock())
            {
                threadedTrainer();
                trained = true;
                ofNotifyEvent(finishedTrainingE);
                unlock();
                stopThread();
            }
            else
            {
                ofLogWarning("threadedFunction()") << "Unable to lock mutex.";
            }
        }
    }
    
    virtual void threadedTrainer() {};
    
    bool trained;
    
    ofEvent<void> finishedTrainingE;
};

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

