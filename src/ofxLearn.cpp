#include "ofxLearn.h"


//--------------------------------------------------------------
void ofxLearn::addTrainingInstance(vector<double> instance) {
    sample_type samp(instance.size());
    for (int i = 0; i < instance.size(); i++)
        samp(i) = instance[i];
    samples.push_back(samp);
}

//--------------------------------------------------------------
void ofxLearn::addTrainingInstance(vector<double> instance, double label) {
    addTrainingInstance(instance);
    labels.push_back(label);
}

//--------------------------------------------------------------
void ofxLearn::trainClassifier() {
    // normalize training set
    normalizer.train(samples);
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);
    
    // randomize training set
    randomize_samples(samples, labels);
    
    // grid parameter search to determine best parameters
    float best_gamma, best_lambda;
    float best_accuracy = 0;
    cout << "optimizing via cross validation. this may take a while... " << endl;
    for (double gamma = 0.01; gamma <= 0.1; gamma *= 10) {
        for (double lambda = 0.001; lambda <=1 ; lambda *= 10){
            svm_trainer.set_kernel(kernel_type(gamma));
            svm_trainer.set_lambda(lambda);
            trainer.set_trainer(svm_trainer);
            const dlib::matrix<double> confusion_matrix = dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 10);
            double accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix);
            //cout << confusion_matrix << endl;
            cout << "gamma: " << gamma << ", lambda: " << lambda << ", accuracy: " << accuracy << endl;
            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_gamma = gamma;
                best_lambda = lambda;
            }
        }
    }
    cout << "finished training with best parameters: gamma "<<best_gamma
            <<", lambda "<< best_lambda << endl;
    
    svm_trainer.set_kernel(kernel_type(best_gamma));
    svm_trainer.set_lambda(best_lambda);
    trainer.set_trainer(svm_trainer);
    
    // set normalized multiclass decision function
    classification_function.function = trainer.train(samples, labels);
    classification_function.normalizer = normalizer;
}

//--------------------------------------------------------------
void ofxLearn::trainRegression() {
    cout << "beginning to train regressor... ";

    // normalize training set
    normalizer.train(samples);
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);
    
    // randomize training set
    randomize_samples(samples, labels);
    
    double max_gamma = 10.0 * 3.0/compute_mean_squared_distance(randomly_subsample(samples, 2000));
    float best_gamma, best_lambda;
    float best_mse = 1000000;
    for (double gamma = 0.01; gamma <= max_gamma; gamma *= 10) {
        for (double lambda = 0.001; lambda <=1.0 ; lambda *= 10){
            svm_trainer.set_kernel(kernel_type(gamma));
            svm_trainer.set_lambda(lambda);    
            vector<double> loo_values;
            svm_trainer.train(samples, labels, loo_values);
            float mse = dlib::mean_squared_error(labels,loo_values);
            if (mse < best_mse) {
                best_mse = mse;
                best_gamma = gamma;
                best_lambda = lambda;
            }
        }
    }
    cout << "finished training with best parameters: gamma "<<best_gamma
            <<", lambda "<< best_lambda << endl;
    
    svm_trainer.set_kernel(kernel_type(best_gamma));
    svm_trainer.set_lambda(best_lambda);
    regression_function.function = svm_trainer.train(samples, labels);
    regression_function.normalizer = normalizer;
}

//--------------------------------------------------------------
int ofxLearn::classify(vector<double> instance) {
    sample_type sample(instance.size());
    for (int i=0; i<instance.size(); i++)
        sample(i) = instance[i];
    return classification_function(sample);
}

//--------------------------------------------------------------
double ofxLearn::predict(vector<double> instance) {
    sample_type sample(instance.size());
    for (int j=0; j < instance.size(); j++)
        sample(j) = instance[j];
    return regression_function(sample);    
}

//--------------------------------------------------------------
vector<int> ofxLearn::getClusters(int k) {
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
void ofxLearn::saveModel(char *filename) {
    ovo_funct_type df;
    df = classification_function;
    ofstream fout(filename, ios::binary);
    dlib::serialize(df, fout);
    fout.close();
    cout << "saved " << filename << endl;
}

//--------------------------------------------------------------
void ofxLearn::loadModel(char *filename) {
    ifstream fin(filename, ios::binary);
    dlib::deserialize(classification_function, fin);
    cout << "loaded "<< filename << endl;
}
