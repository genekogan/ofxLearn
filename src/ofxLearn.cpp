#include "ofxLearn.h"


//---------
ofxLearn::ofxLearn() {
    mlpNumHiddenLayers = 2;
    mlpTargetRmse = 0.01;
    mlpMaxSamples = 100000;
}

//---------
void ofxLearn::addTrainingInstance(vector<double> instance) {
    sample_type samp(instance.size());
    for (int i = 0; i < instance.size(); i++) {
        samp(i) = instance[i];
    }
    samples.push_back(samp);
}

//---------
void ofxLearn::addTrainingInstance(vector<double> instance, double label) {
    addTrainingInstance(instance);
    labels.push_back(label);
}

//---------
void ofxLearn::clearTrainingInstances() {
    samples.clear();
    labels.clear();
}

//---------
void ofxLearn::trainClassifier(TrainMode trainMode, LearnMode learnMode) {
    this->learnMode = CLASSIFICATION;   // only one learnMode at the moment
    
    if (samples.size() == 0) {
        ofLog(OF_LOG_ERROR, "You have not added any samples yet. Can not train.");
        return;
    }
    ofLog(OF_LOG_VERBOSE, "beginning to train classifier function... ");
    
    // normalize training set
    normalizer.train(samples);
    vector<sample_type> normalized_samples;
    normalized_samples.resize(samples.size());
    for (unsigned long i = 0; i < samples.size(); ++i) {
        normalized_samples[i] = normalizer(samples[i]);
    }
    
    // randomize training set
    randomize_samples(normalized_samples, labels);

    // choose default parameters
    float best_gamma, best_lambda;
    if (trainMode == FAST) {
        best_gamma = 1;
        best_lambda = 0.001;
    }
    
    // grid parameter search to determine best parameters
    else if (trainMode == ACCURATE) {
        ofLog(OF_LOG_NOTICE, "Optimizing via cross validation. this may take a while...");
        float best_accuracy = 0;
        for (double gamma = 0.01; gamma <= 1.0; gamma *= 10) {
            for (double lambda = 0.001; lambda <= 1.0; lambda *= 10){
                svm_trainer.set_kernel(kernel_type(gamma));
                svm_trainer.set_lambda(lambda);
                trainer.set_trainer(svm_trainer);
                const dlib::matrix<double> confusion_matrix = dlib::cross_validate_multiclass_trainer(trainer, normalized_samples, labels, 10);
                double accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix);
                ofLog(OF_LOG_VERBOSE, "gamma: "+ofToString(gamma)+", lambda: "+ofToString(lambda)+", accuracy: "+ofToString(accuracy));
                if (accuracy > best_accuracy) {
                    best_accuracy = accuracy;
                    best_gamma = gamma;
                    best_lambda = lambda;
                }
            }
        }
        ofLog(OF_LOG_VERBOSE, "finished training with best parameters: gamma "+ofToString(best_gamma)+", lambda "+ofToString(best_lambda));
    }

    // final train using best parameters
    svm_trainer.set_kernel(kernel_type(best_gamma));
    svm_trainer.set_lambda(best_lambda);
    trainer.set_trainer(svm_trainer);
    
    // set normalized multiclass decision function
    classification_function.function = trainer.train(normalized_samples, labels);
    classification_function.normalizer = normalizer;
    
    ofLog(OF_LOG_VERBOSE, "finished training classifier function.");
}

//---------
void ofxLearn::trainRegression(TrainMode trainMode, LearnMode learnMode) {
    this->learnMode = learnMode;
    
    if (samples.size() == 0) {
        ofLog(OF_LOG_ERROR, "You have not added any samples yet. Can not train.");
        return;
    }
    ofLog(OF_LOG_VERBOSE, "beginning to train regression function... ");
    
    switch (learnMode) {
        case REGRESSION_SVM:
            trainRegressionSvm(trainMode);
            break;

        case REGRESSION_MLP:
            trainRegressionMlp(trainMode);
            break;
            
        default:
            break;
    }
    ofLog(OF_LOG_VERBOSE, "finished training regression function.");
}

//---------
void ofxLearn::trainRegressionSvm(TrainMode trainMode) {
    // normalize training set
    normalizer.train(samples);
    vector<sample_type> normalized_samples;
    normalized_samples.resize(samples.size());
    for (unsigned long i = 0; i < samples.size(); ++i) {
        normalized_samples[i] = normalizer(samples[i]);
    }
    
    // randomize training set
    randomize_samples(normalized_samples, labels);
    
    // choose default parameters
    float best_gamma, best_lambda;
    if (trainMode == FAST) {
        best_gamma = 1;
        best_lambda = 0.001;
    }
    
    // grid parameter search to determine best parameters
    else if (trainMode == ACCURATE) {
        double max_gamma = 10.0 * 3.0/compute_mean_squared_distance(randomly_subsample(normalized_samples, 2000));
        float best_mse = 1000000;
        for (double gamma = 0.01; gamma <= max_gamma; gamma *= 10) {
            for (double lambda = 0.001; lambda <= 1.0; lambda *= 10){
                svm_trainer.set_kernel(kernel_type(gamma));
                svm_trainer.set_lambda(lambda);
                vector<double> loo_values;
                svm_trainer.train(normalized_samples, labels, loo_values);
                float mse = dlib::mean_squared_error(labels, loo_values);
                if (mse < best_mse) {
                    best_mse = mse;
                    best_gamma = gamma;
                    best_lambda = lambda;
                }
            }
        }
        ofLog(OF_LOG_VERBOSE, "finished training with best parameters: gamma "+ofToString(best_gamma)+", lambda "+ofToString(best_lambda));
    }

    // final train using best parameters
    svm_trainer.set_kernel(kernel_type(best_gamma));
    svm_trainer.set_lambda(best_lambda);
    regression_function.function = svm_trainer.train(normalized_samples, labels);
    regression_function.normalizer = normalizer;
}

//---------
void ofxLearn::trainRegressionMlp(TrainMode trainMode) {
    randomize_samples(samples, labels);
    
    vector<int> index;
    for (int i=0; i<samples.size(); i++) {
        index.push_back(i);
    }

    mlp_trainer = new mlp_trainer_type(samples[0].size(), mlpNumHiddenLayers);
        
    int iterations = 0;
    bool stoppingCriteria = false;
    while (!stoppingCriteria) {
        iterations++;
        random_shuffle(index.begin(), index.end());
        for (int i=0; i<samples.size(); i++) {
            if (labels[index[i]] < 0 || labels[index[i]] > 1) {
                ofLog(OF_LOG_ERROR, "Error: MLP can only take labels between 0.0 and 1.0");
            }
            mlp_trainer->train(samples[index[i]], labels[index[i]]);
        }
        float rmse = 0.0;
        for (int i=0; i<samples.size(); i++) {
            rmse += pow((*mlp_trainer)(samples[index[i]]) - labels[index[i]], 2);
        }
        rmse = sqrt(rmse / samples.size());

        ofLog(OF_LOG_VERBOSE, "MLP, "+ofToString(mlpNumHiddenLayers)+" layers, iteration "+ofToString(iterations)+", rmse "+ofToString(rmse));
        if (rmse <= mlpTargetRmse || iterations*samples.size() >= mlpMaxSamples) {
            stoppingCriteria = true;
        }
    }
}

//---------
double ofxLearn::predict(vector<double> instance) {
    double prediction;
    
    // create sample
    sample_type sample(instance.size());
    for (int i=0; i<instance.size(); i++) {
        sample(i) = instance[i];
    }

    // make prediction
    switch (learnMode) {
        case CLASSIFICATION:
            prediction = classification_function(sample);
            break;
        case REGRESSION_SVM:
            prediction = regression_function(sample);
            break;
        case REGRESSION_MLP:
            prediction = (*mlp_trainer)(sample);
            break;
            
        default:
            break;
    }    
    return prediction;
}

//---------
vector<int> ofxLearn::getClusters(int k) {
    ofLog(OF_LOG_NOTICE, "Running kmeans clustering... this could take a few moments... ");
    learnMode = CLUSTERING;
    vector<sample_type> initial_centers;
    dlib::kcentroid<kernel_type> kc(kernel_type(0.00001), 0.00001, 64);
    dlib::kkmeans<kernel_type> kmeans(kc);
    kmeans.set_number_of_centers(k);
    pick_initial_centers(k, initial_centers, samples, kmeans.get_kernel());
    kmeans.train(samples,initial_centers);
    vector<int> clusters;
    for (int i = 0; i < samples.size(); ++i) {
        clusters.push_back(kmeans(samples[i]));
    }
    return clusters;
}

//---------
void ofxLearn::saveModel(string path) {
    const char *filepath = path.c_str();
    ofstream fout(filepath, ios::binary);

    switch (learnMode) {
        case CLASSIFICATION:
            dlib::serialize(classification_function, fout);
            break;
        case REGRESSION_SVM:
            dlib::serialize(regression_function, fout);
            break;
        case REGRESSION_MLP:
            // this needs to be fixed...
            dlib::serialize(mlp_trainer, fout);
            break;
        default:
            break;
    }
    fout.close();
    ofLog(OF_LOG_VERBOSE, "saved model: "+path);
}

//---------
void ofxLearn::loadModel(LearnMode learnMode, string path) {
    this->learnMode = learnMode;
    const char *filepath = path.c_str();
    ifstream fin(filepath, ios::binary);
    switch (learnMode) {
        case CLASSIFICATION:
            dlib::deserialize(classification_function, fin);
            break;
        case REGRESSION_SVM:
            dlib::deserialize(regression_function, fin);
            break;
        case REGRESSION_MLP:
            // this needs to be fixed...
//            dlib::deserialize(*mlp_trainer, fin);
            break;
        default:
            break;
    }
    ofLog(OF_LOG_VERBOSE, "loaded model: "+path);
}
