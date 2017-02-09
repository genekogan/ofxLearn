
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

void ofxLearnSVM::saveModel(string path) {
    const char *filepath = path.c_str();
    ofstream fout(filepath, ios::binary);
    dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<rbf_kernel_type> > df2, df3;
    df2 = df;
    serialize(df2, fout);
}

void ofxLearnSVM::loadModel(string path) {
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
    return clusters;
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
    
    matrix data;
    data.set_size(numSamples, numFeatures);
    
    for (int i=0; i<numSamples; i++) {
        for (int j=0; j<numFeatures; j++) {
            data(i, j) = samples[i](j);
        }
    }
    
    // compute singular value decomposition
    dlib::svd(data, U, E, V);
    
    // sort eigenvalues
    vector<size_t> idx(E.nc());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [this](size_t i1, size_t i2) {return E(i1, i1) > E(i2, i2);});
    
    // copy top-numComponents vectors of U, E, and V
    // into smaller Ur, Er, and Vr, then overwrite
    
    matrix Ur, Er, Vr;
    
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
    
    U = Ur;
    E = Er;
    V = Vr;
}

vector<double> ofxLearnPCA::project(vector<double> sample)
{
    matrix p, q;
    p.set_size(1, sample.size());
    for (int i=0; i<sample.size(); i++) {
        p(0, i) = sample[i];
    }
    q = (p * dlib::inv(E) * V);
    vector<double> projectedSample;
    for (int i=0; i<q.nc(); i++) {
        projectedSample.push_back(q(0, i));
    }
    return projectedSample;
}

vector<vector<double> > ofxLearnPCA::getProjectedSamples()
{
    int numSamples = U.nr();
    int numComponents = U.nc();
    vector<vector<double> > projectedSamples;
    for (int i=0; i<numSamples; i++) {
        vector<double> projectedSample;
        for (int j=0; j<numComponents; j++){
            projectedSample.push_back(U(i, j));
        }
        projectedSamples.push_back(projectedSample);
    }
    return projectedSamples;
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





/*
 subtract mean of columns
 divide data by sqrt(N-1);
 */


// some notes...
void ofxLearn::svd()
{
    int numPoints = 7;
    int numColumns = 5;
    int numComponents = 4;
    
    
    
    
    
    dlib::matrix<double,0,0> DATA;
    dlib::matrix<double,0,0> U;
    dlib::matrix<double,0,0> E;
    dlib::matrix<double,0,0> V;

    DATA.set_size(numPoints, numColumns);
    
    
    
    DATA(0, 0) = 1;         // high 1,3,5
    DATA(0, 1) = 0.5;
    DATA(0, 2) = 3;
    DATA(0, 3) = 1;
    DATA(0, 4) = 5;
    
    DATA(1, 0) = 2;         // 1+2 similar
    DATA(1, 1) = 0.7;
    DATA(1, 2) = 3.3;
    DATA(1, 3) = 0.9;
    DATA(1, 4) = 4.2;
    
    DATA(2, 0) = 1;         // high 2,4,5
    DATA(2, 1) = 4;
    DATA(2, 2) = -0.4;
    DATA(2, 3) = 9;
    DATA(2, 4) = 3.2;
    
    DATA(3, 0) = 1.5;       // 3+4 similar
    DATA(3, 1) = 5.2;
    DATA(3, 2) = 0.1;
    DATA(3, 3) = 8;
    DATA(3, 4) = 2.6;
    
    DATA(4, 0) = 0.2;       // high 5
    DATA(4, 1) = -0.5;
    DATA(4, 2) = 0.3;
    DATA(4, 3) = -1.2;
    DATA(4, 4) = 7.4;
    
    DATA(5, 0) = 0.2;      // 5+6 similar
    DATA(5, 1) = -0.5;
    DATA(5, 2) = 0.31;
    DATA(5, 3) = -1.2;
    DATA(5, 4) = 7.3;
    
    DATA(6, 0) = 9;       // descending from 1 to 5
    DATA(6, 1) = 8;
    DATA(6, 2) = 6;
    DATA(6, 3) = 3;
    DATA(6, 4) = -1.0;
    
    
    dlib::svd(DATA, U, E, V);
    
    
    
    cout << "===== DATA ====" <<endl;
    cout << DATA << endl;
    cout << "U"<<endl;
    cout << U << endl;
    cout << "E"<<endl;
    cout << E << endl;
    cout << "V"<<endl;
    cout << V << endl;
    cout << "===== DATA ====" <<endl;
    
    
    
    // SORT EIGENVECTORS
    vector<size_t> idx(E.nc());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&E](size_t i1, size_t i2) {return E(i1, i1) > E(i2, i2);});
    
    
    // PCA KEEP COMPONENTS
    
//    dlib::matrix<double, 7, numComponents> Us;
//    dlib::matrix<double, numComponents, numComponents> Es;
//    dlib::matrix<double, numColumns, numComponents> Vs;

    dlib::matrix<double, 0, 0> Us;
    dlib::matrix<double, 0, 0> Es;
    dlib::matrix<double, 0, 0> Vs;

    Us.set_size(7, numComponents);
    Es.set_size(numComponents, numComponents);
    Vs.set_size(numComponents, numComponents);
    //    for (int c=0; c<numComponents; c++) {
    //        dlib::set_colm(Us, dlib::colm(U, c));
    //    }
    
    
    Es = dlib::zeros_matrix<double>(numComponents, numComponents);
    Vs = dlib::zeros_matrix<double>(numColumns, numComponents);
    
    // reduce U
    for (int c=0; c<numComponents; c++) {
        for (int r=0; r<numPoints; r++) {
            Us(r, c) = U(r, idx[c]);
        }
    }
    
    // reduce E
    for (int i=0; i<numComponents; i++) {
        Es(i, i) = E(idx[i], idx[i]);
    }
    
    // reduce V
    for (int c=0; c<numComponents; c++) {
        for (int r=0; r<numColumns; r++) {
            Vs(r, c) = V(r, idx[c]);
        }
    }
    
    
    
    cout << "------"<<endl;
    cout << Us << endl;
    cout << "------"<<endl;
    cout << Es << endl;
    cout << "------"<<endl;
    cout << Vs << endl;
    
    cout << "------"<<endl;
    cout << "------ REMIND DATA =-----"<<endl;
    cout << DATA << endl;
    cout << " - correlations " << endl;
//    myCor(U);
    
    cout << "------ RECONS =-----"<<endl;
    dlib::matrix<double,0,0> Dr = (U * E * dlib::trans(V));
    cout << Dr << endl;
    cout << " - correlations " << endl;
//    myCor(Dr);
    
    
    cout << "------ LAST =-----"<<endl;
    dlib::matrix<double,0,0> Drr = (Us * Es * dlib::trans(Vs));
    cout << Drr << endl;
    cout << " - correlations " << endl;
//    myCor(Drr);
    
    
    cout << "-----PROJECTION -----"<<endl;
    
    dlib::matrix<double, 0, 0> q;
    q.set_size(1, numColumns);
    q(0, 0) = 1;
    q(0, 1) = 0.5;
    q(0, 2) = 3;
    q(0, 3) = 1;
    q(0, 4) = 5;
    
    cout << q << endl;
    
    cout << " --- " << endl;
    
    cout << (q * Vs) << endl;
    
    
    
    ofExit();
    
    
    
    
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
    
    
}

