
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

void ofxLearnSupervised::addSample(vector<double> sample, double label)
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

void ofxLearnUnsupervised::addSample(vector<double> sample)
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
    sample_type samp(vectorToSample(sample));
    return predict(samp);
}

double ofxLearnMLP::predict(sample_type & sample)
{
    double theValue = (*mlp_trainer)(sample);
    cout << "pred 1 ST = " <<theValue << endl;

    return theValue;
}


double ofxLearnMLP::predict2(vector<double> & sample)
{

    sample_type samp(vectorToSample(sample));
    return predict2(samp);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double ofxLearnMLP::predict2(sample_type & sample)
{
    int numFeatures = samples[0].size();
    int numLayers = getHiddenLayers();
    vector<double> w1, w3;
    
    w1.clear();
    w3.clear();
    
    dlib::matrix<double> w1m = mlp_trainer->get_w1();
    dlib::matrix<double> w3m = mlp_trainer->get_w3();
    
    for (int i=0; i<numLayers+1; i++) {
        for (int j = 0; j < numFeatures + 1; j++) {
            w1.push_back(w1m(i, j));
        }
    }
    for (int i=0; i<numLayers+1; i++) {
        w3.push_back(w3m(i));
    }
    
    
    
    
    
    
    vector<double> z;
    for (int f=0; f<numFeatures; f++) {
        z.push_back((double) sample(f));
    }
    z.push_back(-1.0);
    
    vector<double> tmp1;
    for (int i=0; i<numLayers+1; i++) {
        float tmp0 = 0.0;
        for (int j=0; j<numFeatures+1; j++) {
            tmp0 += (w1[i*(numFeatures+1) + j] * z[j]);
        }
        tmp1.push_back(sigmoid(tmp0));
    }
    tmp1[numLayers] = -1.0;
    
    float tmp2 = 0.0;
    for (int j=0; j<numLayers+1; j++) {
        tmp2 += (w3[j] * tmp1[j]);
    }
    
    cout << "pred 2 MANUAL = " <<sigmoid(tmp2)<< endl;
    
    //*output->parameter = ofMap(sigmoid(tmp2), 0, 1, output->getMin(), output->getMax());
    return sigmoid(tmp2);

    
    
    
    
    
    

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

void ofxLearnSVM::save(string path) {
    const char *filepath = path.c_str();
    ofstream fout(filepath, ios::binary);
    dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<rbf_kernel_type> > df2, df3;
    df2 = df;
    serialize(df2, fout);
}

void ofxLearnSVM::load(string path) {
    const char *filepath = path.c_str();
    ifstream fin(filepath, ios::binary);
    dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<rbf_kernel_type> > df2;
    dlib::deserialize(df2, fin);
    df = df2;
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
}


//////////////////////////////////////////////////////////////////////
////  Principal component analysis via singular value decomposition

ofxLearnPCA::ofxLearnPCA() : ofxLearnUnsupervised()
{
    
}

void ofxLearnPCA::pca(int numComponents)
{
    if (samples.size() == 0) {
        ofLog(OF_LOG_ERROR, "No samples added yet");
        return;
    }
    
    int numFeatures = samples[0].size();
    int numSamples = samples.size();
        
    matrix_type data;
    data.set_size(numSamples, numFeatures);

    // copy all samples into a matrix
    for (int i=0; i<numSamples; i++) {
        for (int j=0; j<numFeatures; j++) {
            data(i, j) = samples[i](j);
        }
    }
    
    // calculate column means and subtract from data
    column_means.resize(numFeatures);
    for (int i=0; i<numFeatures; i++) {
        column_means[i] = dlib::mean(dlib::colm(data, i));
        for (int j=0; j<numSamples; j++) {
            data(j, i) -= column_means[i];
        }
    }

    // temporarily erase vector of samples to save memory, push back to them later
    samples.clear();

    // compute singular value decomposition
    dlib::svd(data, U, E, V);
    
    // sort eigenvalues
    vector<size_t> idx(E.nc());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [this](size_t i1, size_t i2) {return E(i1, i1) > E(i2, i2);});
    
    // copy top-numComponents vectors of U, E, and V
    // into smaller Ur, Er, and Vr, then overwrite
    
    matrix_type Ur, Er, Vr;
    
    Ur.set_size(numSamples, numComponents);
    Er.set_size(numComponents, numComponents);
    Vr.set_size(numComponents, numComponents);
    
    Er = dlib::zeros_matrix<double>(numComponents, numComponents);
    Vr = dlib::zeros_matrix<double>(numFeatures, numComponents);
    
    for (int c=0; c<numComponents; c++) {
        for (int r=0; r<numSamples; r++) {
            Ur(r, c) = U(r, idx[c]);
        }
        for (int r=0; r<numFeatures; r++) {
            Vr(r, c) = V(r, idx[c]);
        }
        Er(c, c) = E(idx[c], idx[c]);
    }
    
    // copy U, E, and V
    U = Ur;
    E = Er;
    V = Vr;

    // erase temp variables
    Ur.set_size(0, 0);
    Er.set_size(0, 0);
    Vr.set_size(0, 0);
    
    // copy samples back to vector
    for (int i=0; i<numSamples; i++) {
        sample_type sample(numFeatures);
        for (int j=0; j<numFeatures; j++) {
            sample(j) = data(i, j) + column_means[j];
        }
        samples.push_back(sample);
    }

    // erase temp matrix
    data.set_size(0, 0);
}

vector<double> ofxLearnPCA::project(vector<double> sample)
{
    matrix_type p, q;
    p.set_size(1, sample.size());
    for (int i=0; i<sample.size(); i++) {
        p(0, i) = sample[i] - column_means[i];
    }
    //q = (p * dlib::inv(E) * V);
    q = (p * V);
    vector<double> projectedSample;
    for (int i=0; i<q.nc(); i++) {
        projectedSample.push_back(q(0, i));
    }
    return projectedSample;
}

vector<vector<double> > ofxLearnPCA::getProjectedSamples()
{
    int numFeatures = samples[0].size();
    int numSamples = samples.size();
    vector<vector<double> > projectedSamples;

    for (int i=0; i<numSamples; i++) {
        matrix_type p, q;
        p.set_size(1, numFeatures);
        for (int j=0; j<numFeatures; j++) {
            p(0, j) = samples[i](j) - column_means[j];
        }
        //q = (p * dlib::inv(E) * V);
        q = (p * V);
        vector<double> projectedSample;
        for (int i=0; i<q.nc(); i++) {
            projectedSample.push_back(q(0, i));
        }
        projectedSamples.push_back(projectedSample);
    }
    return projectedSamples;
}

void ofxLearnPCA::save(string path) {
    const char *filepath = path.c_str();
    ofstream fout(filepath, ios::binary);
    dlib::serialize(U, fout);
    dlib::serialize(E, fout);
    dlib::serialize(V, fout);
    dlib::serialize(column_means, fout);
}

void ofxLearnPCA::load(string path) {
    const char *filepath = path.c_str();
    ifstream fin(filepath, ios::binary);
    dlib::deserialize(U, fin);
    dlib::deserialize(E, fin);
    dlib::deserialize(V, fin);
    dlib::deserialize(column_means, fin);
}



//////////////////////////////////////////////////////////////////////
////  Random projection (instead of PCA)

ofxLearnRandomProjection::ofxLearnRandomProjection() : ofxLearnUnsupervised()
{
    
}

void ofxLearnRandomProjection::randomProjection(int numComponents)
{
    if (samples.size() == 0) {
        ofLog(OF_LOG_ERROR, "No samples added yet");
        return;
    }
    
    int numFeatures = samples[0].size();
    int numSamples = samples.size();
    
    matrix_type data;
    data.set_size(numSamples, numFeatures);
    
    // copy all samples into a matrix
    for (int i=0; i<numSamples; i++) {
        for (int j=0; j<numFeatures; j++) {
            data(i, j) = samples[i](j);
        }
    }
    
    // calculate column means and subtract from data
    column_means.resize(numFeatures);
    for (int i=0; i<numFeatures; i++) {
        column_means[i] = dlib::mean(dlib::colm(data, i));
        for (int j=0; j<numSamples; j++) {
            data(j, i) -= column_means[i];
        }
    }
    
    // temporarily erase vector of samples to save memory, push back to them later
    samples.clear();
    
    // make random matrix
    vector<double> column_lengths(numComponents);
    V.set_size(numFeatures, numComponents);
    for (int i=0; i<numComponents; i++) {
        column_lengths[i] = 0.0;
        for (int j=0; j<numFeatures; j++) {
            V(j, i) = ofRandom(1);
            column_lengths[i] += V(j, i) * V(j, i);
        }
        column_lengths[i] = sqrt(column_lengths[i]);
    }
    
    // make columns of V have unit distance
    for (int i=0; i<numComponents; i++) {
        for (int j=0; j<numFeatures; j++) {
            V(j, i) = V(j, i) / column_lengths[i];
        }
    }
    
    // copy samples back to vector
    for (int i=0; i<numSamples; i++) {
        sample_type sample(numFeatures);
        for (int j=0; j<numFeatures; j++) {
            sample(j) = data(i, j) + column_means[j];
        }
        samples.push_back(sample);
    }

    // erase temp matrix
    data.set_size(0, 0);
}

vector<double> ofxLearnRandomProjection::project(vector<double> sample)
{
    matrix_type p, q;
    p.set_size(1, sample.size());
    for (int i=0; i<sample.size(); i++) {
        p(0, i) = sample[i] - column_means[i];
    }
    //q = (p * dlib::inv(E) * V);
    q = (p * V);
    vector<double> projectedSample;
    for (int i=0; i<q.nc(); i++) {
        projectedSample.push_back(q(0, i));
    }
    return projectedSample;
}

vector<vector<double> > ofxLearnRandomProjection::getProjectedSamples()
{
    int numFeatures = samples[0].size();
    int numSamples = samples.size();
    vector<vector<double> > projectedSamples;
    
    for (int i=0; i<numSamples; i++) {
        matrix_type p, q;
        p.set_size(1, numFeatures);
        for (int j=0; j<numFeatures; j++) {
            p(0, j) = samples[i](j) - column_means[j];
        }
        //q = (p * dlib::inv(E) * V);
        q = (p * V);
        vector<double> projectedSample;
        for (int i=0; i<q.nc(); i++) {
            projectedSample.push_back(q(0, i));
        }
        projectedSamples.push_back(projectedSample);
    }
    return projectedSamples;
}

void ofxLearnRandomProjection::save(string path) {
    const char *filepath = path.c_str();
    ofstream fout(filepath, ios::binary);
    dlib::serialize(V, fout);
    dlib::serialize(column_means, fout);
}

void ofxLearnRandomProjection::load(string path) {
    const char *filepath = path.c_str();
    ifstream fin(filepath, ios::binary);
    dlib::deserialize(V, fin);
    dlib::deserialize(column_means, fin);
}


//////////////////////////////////////////////////////////////////////
////  Threaded learners

ofxLearnThreaded::ofxLearnThreaded() : ofxLearn() {
    trained = false;
}

ofxLearnThreaded::~ofxLearnThreaded() {
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
