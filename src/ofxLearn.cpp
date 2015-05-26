#include "ofxLearn.h"


inline sample_type ofxLearn::vectorToSample(vector<double> sample_) {
    sample_type sample(sample_.size());
    for (int i=0; i<sample_.size(); i++) {
        sample(i) = sample_[i];
    }
    return sample;
}

//////////////////////////////////////////////////////////////////////
////  Supervised

void ofxLearnSupervised::addTrainingInstance(vector<double> sample, double label)
{
    if (label < 0.0 || label > 1.0) {
        ofLog(OF_LOG_ERROR, "MLP can only take labels between 0.0 and 1.0");
        //return;
    }
    sample_type samp(sample.size());
    for (int i = 0; i < sample.size(); i++) {
        samp(i) = sample[i];
    }
    samples.push_back(samp);
    labels.push_back(label);
}

void ofxLearnSupervised::addSample(sample_type sample, double label)
{
    if (label < 0.0 || label > 1.0) {
        ofLog(OF_LOG_ERROR, "MLP can only take labels between 0.0 and 1.0");
        //return;
    }
    samples.push_back(sample);
    labels.push_back(label);
}

void ofxLearnSupervised::clearTrainingInstances()
{
    samples.clear();
    labels.clear();
}

//////////////////////////////////////////////////////////////////////
////  Unsupervised

void ofxLearnUnsupervised::addTrainingInstance(vector<double> sample)
{
    sample_type samp(sample.size());
    for (int i = 0; i < sample.size(); i++) {
        samp(i) = sample[i];
    }
    samples.push_back(samp);
}

void ofxLearnUnsupervised::addSample(sample_type sample)
{
    samples.push_back(sample);
}

void ofxLearnUnsupervised::clearTrainingInstances()
{
    samples.clear();
}

//////////////////////////////////////////////////////////////////////
////  Regression: MLP

ofxLearnMLP::ofxLearnMLP() : ofxLearnSupervised()
{
    hiddenLayers = 2;
    targetRmse = 0.01;
    maxSamples = 100000;
}

ofxLearnMLP::~ofxLearnMLP()
{
    delete mlp_trainer;
}

void ofxLearnMLP::train()
{
    if (samples.size() == 0) {
        ofLog(OF_LOG_ERROR, "You have not added any samples yet. Can not train.");
        return;
    }
    
    ofLog(OF_LOG_VERBOSE, "beginning to train regression function... ");
    
    vector<int> index;
    for (int i=0; i<samples.size(); i++) {
        index.push_back(i);
    }
    
    mlp_trainer = new mlp_trainer_type(samples[0].size(), hiddenLayers);
    
    int iterations = 0;
    bool stoppingCriteria = false;
    while (!stoppingCriteria)
    {
        iterations++;
        
        randomize_samples(samples, labels);
        for (int i=0; i<samples.size(); i++) {
            mlp_trainer->train(samples[i], labels[i]);
        }
        
        float rmse = 0.0;
        for (int i=0; i<samples.size(); i++) {
            rmse += pow((*mlp_trainer)(samples[i]) - labels[i], 2);
        }
        rmse = sqrt(rmse / samples.size());
        
        ofLog(OF_LOG_VERBOSE, "MLP, "+ofToString(hiddenLayers)+" layers, iteration "+ofToString(iterations)+", rmse "+ofToString(rmse));
        if (rmse <= targetRmse || iterations * samples.size() >= maxSamples) {
            stoppingCriteria = true;
        }
    }
}

double ofxLearnMLP::predict(vector<double> sample)
{
    return (*mlp_trainer)(vectorToSample(sample));
}


//////////////////////////////////////////////////////////////////////
////  Regression: SVR

ofxLearnSVR::ofxLearnSVR() : ofxLearnSupervised()
{
    
}

ofxLearnSVR::~ofxLearnSVR()
{
    
}

double ofxLearnSVR::predict(vector<double> sample)
{
    return df(vectorToSample(sample));
}

void ofxLearnSVR::train()
{
    trainer.set_kernel(rbf_kernel_type(0.1));
    trainer.set_c(10);
    trainer.set_epsilon_insensitivity(0.001);
    df = trainer.train(samples, labels);
}



//////////////////////////////////////////////////////////////////////
////  Classification: SVM

ofxLearnSVM::ofxLearnSVM() : ofxLearnSupervised() {
    
}

ofxLearnSVM::~ofxLearnSVM() {
    
    
}

void ofxLearnSVM::train()
{
    poly_trainer.set_kernel(poly_kernel_type(0.1, 1, 2));
    rbf_trainer.set_kernel(rbf_kernel_type(0.1));

    trainer.set_trainer(rbf_trainer);
    //trainer.set_trainer(poly_trainer, 1, 2);
    
    randomize_samples(samples, labels);
    df = trainer.train(samples, labels);
}

double ofxLearnSVM::predict(vector<double> sample) {
    return df(vectorToSample(sample));
}


//////////////////////////////////////////////////////////////////////
////  Clustering: K-Means

ofxLearnKMeans::ofxLearnKMeans() : ofxLearnUnsupervised()
{
    
}

void ofxLearnKMeans::setNumClusters(int numClusters)
{
    this->numClusters = numClusters;
}

void ofxLearnKMeans::train()
{
    vector<sample_type> initial_centers;
    dlib::kcentroid<rbf_kernel_type> kc(rbf_kernel_type(0.00001), 0.00001, 64);
    dlib::kkmeans<rbf_kernel_type> kmeans(kc);
    kmeans.set_number_of_centers(numClusters);
    pick_initial_centers(numClusters, initial_centers, samples, kmeans.get_kernel());
    kmeans.train(samples,initial_centers);
    clusters.clear();
    for (int i = 0; i < samples.size(); ++i) {
        clusters.push_back(kmeans(samples[i]));
    }
    return clusters;
}


/*
 // some notes...
void ofxLearn::svd()
{
    // matrix expressions: http://dlib.net/matrix_ex.cpp.html
    
    
    sample_type m;
    sample_type a;
    sample_type b;
    sample_type c;
    
    
    
    dlib::matrix<double,2,2> mm;
    mm(0, 0) = 5;
    mm(0, 1) = 7;
    mm(1, 0) = 1;
    mm(1, 1) = 3;
    
    //    cout << mm << endl;
    
    dlib::matrix<double,2,2> mmi = dlib::inv(mm);
    cout << mm << endl;
    cout << mmi << endl;
    
    
    
    dlib::matrix<double,3,4> mm1;
    dlib::matrix<double,3,4> mms;
    dlib::matrix<double,4,4> mmv;
    dlib::matrix<double,4,4> mmd;
    
    mm1(0, 0) = 4;
    mm1(0, 1) = 2;
    mm1(0, 2) = 8;
    mm1(0, 3) = 9;
    mm1(1, 0) = 3;
    mm1(1, 1) = 5;
    mm1(1, 2) = 2;
    mm1(1, 3) = 1;
    mm1(2, 0) = 9;
    mm1(2, 1) = 4;
    mm1(2, 2) = 4;
    mm1(2, 3) = 3;
    
    
    dlib::svd(mm1, mms, mmv, mmd);
    cout << "=========="<<endl;
    cout << mm1 << endl;
    cout << mms << endl;
    cout << mmv << endl;
    cout << mmd << endl;
    
    
    
    // SVD links
    // http://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca
    // http://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
    
    // code at bottom:
    // http://arxiv.org/pdf/1404.1100.pdf
    
    // http://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    // http://www3.cs.stonybrook.edu/~sael/teaching/cse549/Slides/CSE549_16.pdf
    // http://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/IDAPILecture15.pdf
    
    // numerical example
    // http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc552.htm
    
    // matrix calc
    // http://www.bluebit.gr/matrix-calculator/
    
    
    
    //http://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
    
    dlib::matrix<double,4,2> m1;
    dlib::matrix<double,4,2> ms;
    dlib::matrix<double,2,2> mv;
    dlib::matrix<double,2,2> md;
    
    m1(0, 0) = 2;
    m1(0, 1) = 4;
    m1(1, 0) = 1;
    m1(1, 1) = 3;
    m1(2, 0) = 0;
    m1(2, 1) = 0;
    m1(3, 0) = 0;
    m1(3, 1) = 0;
    
    
    dlib::svd(m1, ms, mv, md);
    cout << "=========="<<endl;
    cout << m1 << endl;
    cout << ms << endl;
    cout << mv << endl;
    cout << md << endl;
}
*/
