#include "ofxLearn.h"

//--------------------------------------------------------------
ofxLearn::ofxLearn() {
    numInstances = 0;
    isTrained = false;
    svm_trainer.set_kernel(kernel_type(0.0001));
    svm_trainer.set_lambda(0.0001);
    trainer.set_trainer(svm_trainer);  
}

//--------------------------------------------------------------
void ofxLearn::addTrainingInstance(vector<double> instance) {
    sample_type samp;
    for (int i = 0; i < instance.size(); i++)
        samp(i) = instance[i];    
    samples.push_back(samp);
    numInstances++;    
}

//--------------------------------------------------------------
void ofxLearn::addTrainingInstance(vector<double> instance, int label) {
    addTrainingInstance(instance);
    labels.push_back(label);
}

//--------------------------------------------------------------
void ofxLearn::trainClassifier() {    
    cout << "beginning to train model... ";
    randomize_samples(samples, labels);
    decision_function = trainer.train(samples, labels);
    isTrained = true;
    cout << " finished training model, ready to use." << endl;
}

//--------------------------------------------------------------
void ofxLearn::trainRegressor() {    
    cout << "beginning to train model... ";
    // tbd <-- regression model
    cout << " finished training model, ready to use." << endl;
}

//--------------------------------------------------------------
vector<int> ofxLearn::findClusters(int k) {
    cout << "Running kmeans clustering... this could take a few moments... " << endl;
    vector<sample_type> initial_centers;
    dlib::kcentroid<kernel_type> kc(kernel_type(0.00001), 0.00001, 64);
    dlib::kkmeans<kernel_type> kmeans(kc);    
    kmeans.set_number_of_centers(k);
    pick_initial_centers(k, initial_centers, samples, kmeans.get_kernel());
    kmeans.train(samples,initial_centers);
    vector<int> clusters;
    for (int i = 0; i < samples.size(); ++i)
        clusters.push_back(kmeans(samples[i]));
    cout << "Finished clustering!" << endl;
    return clusters;
}

//--------------------------------------------------------------
void ofxLearn::clearTrainingSet() {
    samples.clear();
    labels.clear();
}

//--------------------------------------------------------------
void ofxLearn::optimizeClassifier() {
    cout << "optimizing via cross validation. this may take a while... " << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma *= 10)
    {
        for (double lambda = 0.00001; lambda <=1 ; lambda *= 10)
        {
            svm_trainer.set_kernel(kernel_type(gamma));
            svm_trainer.set_lambda(lambda);
   
            const dlib::matrix<double> confusion_matrix = dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 10);
            double accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix);
            cout << "gamma: " << gamma << ", lambda: " << lambda << ", accuracy: " << accuracy << endl; 
            
        }
    }
    
    // normalize?
    /* 
    dlib::vector_normalizer<sample_type> normalizer;
    normalizer.train(samples);
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 
    */
    
    // set optimal parameters
    svm_trainer.set_kernel(kernel_type(0.0001));
    svm_trainer.set_lambda(0.0001);
}
     
//--------------------------------------------------------------
int ofxLearn::predict(vector<double> instance) {
    sample_type samp;
    for (int j=0; j < instance.size(); j++)
        samp(j) = instance[j];
    return decision_function(samp);
}

//--------------------------------------------------------------
void ofxLearn::saveModel(string filename) {
    //dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<kernel_type> > df;
    dec_funct_type df;
    df = decision_function;    
    ofstream fout("data/df.dat", ios::binary);
    serialize(df, fout);
    fout.close();
    cout << "saved" << endl;
    
}

//--------------------------------------------------------------
void ofxLearn::loadModel(string filename) {
    ifstream fin("data/df.dat", ios::binary);
    deserialize(decision_function, fin);
    cout << "loaded" << endl;
    isTrained = true;
}


//--------------------------------------------------------------
void ofxLearn::saveDataset(string filename) {
    //tbd
}

//--------------------------------------------------------------
void ofxLearn::loadDataset(string filename) {
    //tbd
}