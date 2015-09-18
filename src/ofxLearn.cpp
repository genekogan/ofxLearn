
//https://www.projectrhea.org/rhea/index.php/PCA


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
        ofLog(OF_LOG_WARNING, "Warning: continuous labels should be between 0.0 and 1.0");
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
        ofLog(OF_LOG_WARNING, "Warning: continuous labels should be between 0.0 and 1.0");
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
    
    ofLog(OF_LOG_VERBOSE, "beginning to train regression ... ");
    
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
        
        ofLog(OF_LOG_NOTICE, "MLP, "+ofToString(hiddenLayers)+" layers, iteration "+ofToString(iterations)+", rmse "+ofToString(rmse));
        if (rmse <= targetRmse || iterations * samples.size() >= maxSamples) {
            stoppingCriteria = true;
        }
    }
}

double ofxLearnMLP::predict(vector<double> & sample)
{
    return (*mlp_trainer)(vectorToSample(sample));
}

double ofxLearnMLP::predict(sample_type & sample)
{
    double theValue = (*mlp_trainer)(sample);
    return theValue;
}


//////////////////////////////////////////////////////////////////////
////  Regression: SVR

ofxLearnSVR::ofxLearnSVR() : ofxLearnSupervised()
{
    
}

ofxLearnSVR::~ofxLearnSVR()
{
    
}

double ofxLearnSVR::predict(vector<double> & sample)
{
    return df(vectorToSample(sample));
}

double ofxLearnSVR::predict(sample_type & sample)
{
    return df(sample);
}

void ofxLearnSVR::train()
{
    trainer.set_kernel(rbf_kernel_type(0.1));
    trainer.set_c(10);
    trainer.set_epsilon_insensitivity(0.001);
    df = trainer.train(samples, labels);
}

void ofxLearnSVR::trainWithGridParameterSearch()
{
    ofLog(OF_LOG_NOTICE, "SVR cross validation is broken for now, just reverting to default train()... sorry!");
    train();
    return;
    
    // why is this not working right???  overfitting?
    // just use train() for now.....
    ofLog(OF_LOG_NOTICE, "Optimizing SVR via cross validation. this may take a while... ");
    
    randomize_samples(samples, labels);
    
    float best_gamma, best_c;
    float best_accuracy = 0;
    for (double gamma = 0.01; gamma <= 1.0; gamma *= 10) {
        for (double c = 0.01; c <= 100.0; c *= 10){
            trainer.set_kernel(rbf_kernel_type(gamma));
            trainer.set_c(c);
            trainer.set_epsilon_insensitivity(0.001);
            const dlib::matrix<double> confusion_matrix = dlib::cross_validate_regression_trainer(trainer, samples, labels, 10);
            double accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix);
            ofLog(OF_LOG_NOTICE, "SVR accuracy (gamma = "+ofToString(gamma)+", C = "+ofToString(c)+" : accuracy = "+ofToString(accuracy));
            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_gamma = gamma;
                best_c = c;
            }
        }
    }
    ofLog(OF_LOG_NOTICE, "Finished SVR grid parameter search with top accuracy : "+ofToString(best_accuracy)+", gamma = "+ofToString(best_gamma)+", C = "+ofToString(best_c));
    
    // finally, set best parameters and retrain
    trainer.set_kernel(rbf_kernel_type(best_gamma));
    trainer.set_c(best_c);
    trainer.set_epsilon_insensitivity(0.001);
    df = trainer.train(samples, labels);
}



//////////////////////////////////////////////////////////////////////
////  Classification: SVM

ofxLearnSVM::ofxLearnSVM() : ofxLearnSupervised() {
    
}

ofxLearnSVM::~ofxLearnSVM() {
    
    
}

void ofxLearnSVM::trainWithGridParameterSearch()
{
    ofLog(OF_LOG_NOTICE, "Optimizing SVM via cross validation. this may take a while... ");

    randomize_samples(samples, labels);
    
    float best_gamma, best_lambda;
    float best_accuracy = 0;
    for (double gamma = 0.01; gamma <= 1.0; gamma *= 10) {
        for (double lambda = 0.001; lambda <= 1.0; lambda *= 10){
            rbf_trainer.set_kernel(rbf_kernel_type(gamma));
            rbf_trainer.set_lambda(lambda);
            trainer.set_trainer(rbf_trainer);
            const dlib::matrix<double> confusion_matrix = dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 10);
            double accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix);
            ofLog(OF_LOG_NOTICE, "SVM accuracy (gamma = "+ofToString(gamma)+", lambda = "+ofToString(lambda)+" : accuracy = "+ofToString(accuracy));
            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_gamma = gamma;
                best_lambda = lambda;
            }
        }
    }
    ofLog(OF_LOG_NOTICE, "Finished SVM grid parameter search with top accuracy : "+ofToString(best_accuracy)+", gamma = "+ofToString(best_gamma)+", lambda = "+ofToString(best_lambda));

    // finally, set best parameters and retrain
    rbf_trainer.set_kernel(rbf_kernel_type(best_gamma));
    rbf_trainer.set_lambda(best_lambda);
    trainer.set_trainer(rbf_trainer);
    df = trainer.train(samples, labels);
}

void ofxLearnSVM::train()
{
    rbf_trainer.set_kernel(rbf_kernel_type(0.1));
    rbf_trainer.set_lambda(0.01);
    trainer.set_trainer(rbf_trainer);

    //poly_trainer.set_kernel(poly_kernel_type(0.1, 1, 2));
    //trainer.set_trainer(poly_trainer, 1, 2);
    
    randomize_samples(samples, labels);
    df = trainer.train(samples, labels);
}

double ofxLearnSVM::predict(vector<double> & sample) {
    return df(vectorToSample(sample));
}

double ofxLearnSVM::predict(sample_type & sample) {
    return df(sample);
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


//////////////////////////////////////////////////////////////////////
////  Threaded learners

ofxLearnThreaded::ofxLearnThreaded() : ofxLearn() {
    trained = false;
}

ofxLearnThreaded::~ofxLearnThreaded() {
    finishedTrainingE.clear();
    finishedTrainingE.disable();
}

void ofxLearnThreaded::beginTraining()
{
    trained = false;
    startThread();
}

void ofxLearnThreaded::threadedFunction()
{
    while (isThreadRunning())
    {
        if (lock())
        {
            threadedTrainer();
            trained = true;
            sleep(1000);
            unlock();
            stopThread();
            ofNotifyEvent(finishedTrainingE);
        }
        else
        {
            ofLogWarning("threadedFunction()") << "Unable to lock mutex.";
        }
    }
}






// some notes...
void ofxLearn::svd()
{
    // matrix expressions: http://dlib.net/matrix_ex.cpp.html
    
    
    // verifying....
    
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

    
    /////////////////////
    dlib::running_stats<double> rs;
    
    double tp1 = 0;
    double tp2 = 0;
    
    // We first generate the data and add it sequentially to our running_stats object.  We
    // then print every fifth data point.
    for (int x = 1; x <= 100; x++)
    {
        tp1 = x/100.0;
        tp2 = pi*x == 0 ? 1 : sin(pi * x) / (pi * x);
        
        rs.add(tp2);
    }
    
    // Finally, we compute and print the mean, variance, skewness, and excess kurtosis of
    // our data.
    
    cout << endl;
    cout << "Mean:           " << rs.mean() << endl;
    cout << "Variance:       " << rs.variance() << endl;
//    cout << "Skewness:       " << rs.skewness() << endl;
//    cout << "Excess Kurtosis " << rs.ex_kurtosis() << endl;
  
    
    dlib::vector_normalizer_pca<sample_type> pca1;
    
    vector<sample_type> vects;
    
    for (int i=0; i<850; i++) {
        sample_type m2(5);
        m2(0) = ofRandom(1);
        m2(1) = m2(0) * (0.35 + ofRandom(-0.05, 0.05));
        m2(2) = m2(0) * (0.35 + ofRandom(-0.15, 0.15)) + m2(1) * (0.45 + ofRandom(-0.11, 0.19));
        m2(3) = m2(1) * (0.15 + ofRandom(-0.03, 0.03)) + m2(2) * (0.12 + ofRandom(-0.1, 0.1)) + m2(0) * (0.05 + ofRandom(-0.04, 0.04));
        m2(4) = ofRandom(1);
        vects.push_back(m2);
    }


    pca1.train(vects);

    dlib::matrix<double> pca = pca1.pca_matrix();
    
    cout << "pca is " << endl;
    cout << pca << endl;
}

