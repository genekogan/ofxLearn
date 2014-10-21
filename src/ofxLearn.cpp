#include "ofxLearn.h"


//---------
ofxLearn::ofxLearn() {
    mlpNumHiddenLayers = 2;
    mlpTargetRmse = 0.001;
    mlpMaxSamples = 1000000;
    status = "";
    isTraining = false;
    isTrained = true;
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
void ofxLearn::trainClassifier(LearnMode learnMode, TrainMode trainMode) {
    this->learnMode = learnMode;
    this->trainMode = trainMode;
    if (samples.size() == 0) {
        ofLog(OF_LOG_ERROR, "You have not added any samples yet. Can not train.");
        return;
    }
    progress = 0;
    isTraining = true;
    isTrained = false;
    startThread();
}

//---------
void ofxLearn::trainClusters(int numClusters) {
    learnMode = CLUSTERING;
    this->numClusters = numClusters;
    progress = 0;
    isTraining = true;
    isTrained = false;
    startThread();
}

//---------
void ofxLearn::trainClassifierSvm(TrainMode trainMode) {
    ofLog(OF_LOG_VERBOSE, "beginning to train SVM classifier function... ");
    
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
        ofLog(OF_LOG_VERBOSE, "Optimizing via cross validation. this may take a while...");
        vector<double> gamma, lambda;
        double max_gamma = 10.0 * 3.0/compute_mean_squared_distance(randomly_subsample(normalized_samples, 2000));
        for (double g = 0.01; g <= max_gamma; g *= 10) {
            for (double l = 0.001; l <= 1.0; l *= 10){
                gamma.push_back(g);
                lambda.push_back(l);
            }
        }
        int numParameterSets = gamma.size();
        float best_accuracy = 0;
        status = "SVM: have searched 0/"+ofToString(numParameterSets)+" parameter sets";
        for (int i=0; i<numParameterSets; i++) {
            svm_trainer.set_kernel(kernel_type(gamma[i]));
            svm_trainer.set_lambda(lambda[i]);
            trainer.set_trainer(svm_trainer);
            const dlib::matrix<double> confusion_matrix = dlib::cross_validate_multiclass_trainer(trainer, normalized_samples, labels, 10);
            double accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix);
            ofLog(OF_LOG_VERBOSE, "gamma: "+ofToString(gamma[i])+", lambda: "+ofToString(lambda[i])+", accuracy: "+ofToString(accuracy));
            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_gamma = gamma[i];
                best_lambda = lambda[i];
            }
            status = "SVM: have searched "+ofToString(i+1)+"/"+ofToString(numParameterSets)+" parameter sets, best accuracy "+ofToString(best_accuracy);
            progress = (float) (i+1) / numParameterSets;
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
    
    isTrained = true;
    isTraining = false;

    ofLog(OF_LOG_VERBOSE, "finished training SVM classifier function.");
}

//---------
void ofxLearn::trainRegressionSvm(TrainMode trainMode) {
    ofLog(OF_LOG_VERBOSE, "finished training SVM regression function.");

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
        ofLog(OF_LOG_VERBOSE, "Optimizing via cross validation. this may take a while...");
        vector<double> gamma, lambda;
        double max_gamma = 10.0 * 3.0/compute_mean_squared_distance(randomly_subsample(normalized_samples, 2000));
        for (double g = 0.01; g <= max_gamma; g *= 10) {
            for (double l = 0.001; l <= 1.0; l *= 10){
                gamma.push_back(g);
                lambda.push_back(l);
            }
        }
        int numParameterSets = gamma.size();
        float best_mse = 1000000;
        status = "SVM: have searched 0/"+ofToString(numParameterSets)+" parameter sets";
        for (int i=0; i<numParameterSets; i++) {
            svm_trainer.set_kernel(kernel_type(gamma[i]));
            svm_trainer.set_lambda(lambda[i]);
            vector<double> loo_values;
            svm_trainer.train(normalized_samples, labels, loo_values);
            float mse = dlib::mean_squared_error(labels, loo_values);
            if (mse < best_mse) {
                best_mse = mse;
                best_gamma = gamma[i];
                best_lambda = lambda[i];
            }
            status = "SVM: have searched "+ofToString(i+1)+"/"+ofToString(numParameterSets)+" parameter sets, best MSE "+ofToString(best_mse);
            progress = (float) (i+1) / numParameterSets;
        }
        ofLog(OF_LOG_VERBOSE, "finished training with best parameters: gamma "+ofToString(best_gamma)+", lambda "+ofToString(best_lambda));
    }

    // final train using best parameters
    svm_trainer.set_kernel(kernel_type(best_gamma));
    svm_trainer.set_lambda(best_lambda);
    regression_function.function = svm_trainer.train(normalized_samples, labels);
    regression_function.normalizer = normalizer;
    
    isTrained = true;
    isTraining = false;

    ofLog(OF_LOG_VERBOSE, "finished training SVM regression function.");
}

//---------
void ofxLearn::trainRegressionMlp(TrainMode trainMode) {
    ofLog(OF_LOG_VERBOSE, "beginning to train MLP regression function... ");
    
    vector<int> index;
    for (int i=0; i<samples.size(); i++)    index.push_back(i);

    randomize_samples(samples, labels);
    mlp_trainer = new mlp_trainer_type(samples[0].size(), mlpNumHiddenLayers);
    
    mlpSamples = 0;
    mlpRmse = 0.0;

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
            mlpSamples++;
        }
        float rmse = 0.0;
        for (int i=0; i<samples.size(); i++) {
            rmse += pow((*mlp_trainer)(samples[index[i]]) - labels[index[i]], 2);
        }
        mlpRmse = sqrt(rmse / samples.size());
        status = "MLP: rmse "+ofToString(mlpRmse)+" (target "+ofToString(mlpTargetRmse)+"), "+ofToString(mlpMaxSamples)+" samples (max "+ofToString(mlpMaxSamples)+")";
        progress = (float) mlpSamples / mlpMaxSamples;
        
        ofLog(OF_LOG_VERBOSE, "MLP, "+ofToString(mlpNumHiddenLayers)+" layers, iteration "+ofToString(iterations)+", rmse "+ofToString(mlpRmse));
        if (mlpRmse <= mlpTargetRmse || mlpSamples >= mlpMaxSamples) {
            stoppingCriteria = true;
        }
    }

    isTrained = true;
    isTraining = false;
    
    ofLog(OF_LOG_VERBOSE, "finished training MLP regression function.");
}

//---------
void ofxLearn::trainKMeansClusters() {
    ofLog(OF_LOG_NOTICE, "Running kmeans clustering... this could take a few moments... ");
    
    vector<sample_type> initial_centers;
    dlib::kcentroid<kernel_type> kc(kernel_type(0.00001), 0.00001, 64);
    dlib::kkmeans<kernel_type> kmeans(kc);
    kmeans.set_number_of_centers(numClusters);
    pick_initial_centers(numClusters, initial_centers, samples, kmeans.get_kernel());
    kmeans.train(samples,initial_centers);
    
    clusters.clear();
    for (int i = 0; i < samples.size(); ++i) {
        clusters.push_back(kmeans(samples[i]));
    }

    isTrained = true;
    isTraining = false;

    ofLog(OF_LOG_NOTICE, "Finished kmeans clustering.");
}

//---------
void ofxLearn::threadedFunction() {
    while (isThreadRunning()) {
        if (lock()) {
            switch (learnMode) {
                case CLASSIFICATION:
                    trainClassifierSvm(trainMode);
                    break;
                    
                case REGRESSION_SVM:
                    trainRegressionSvm(trainMode);
                    break;
                    
                case REGRESSION_MLP:
                    trainRegressionMlp(trainMode);
                    break;
                    
                case CLUSTERING:
                    trainKMeansClusters();
                    break;
                    
                default:
                    break;
            }
            
            unlock();
            stopThread();
        }
        else {
            ofLogWarning("threadedFunction()") << "Unable to lock mutex.";
            stopThread();
        }
    }
}

//---------
double ofxLearn::predict(vector<double> instance) {
    if (!isTrained) {
        ofLog(OF_LOG_ERROR, "Error: can't predict instance because no classifier is trained.");
        return 0.0;
    }
    
    // create sample
    sample_type sample(instance.size());
    for (int i=0; i<instance.size(); i++) {
        sample(i) = instance[i];
    }

    // make prediction
    double prediction;
    
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
vector<int> & ofxLearn::getClusters() {
    if (!isTrained) {
        ofLog(OF_LOG_ERROR, "Error: no clusters have been found yet.");
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

        case CLUSTERING:
            // this needs to be implemented
            //
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
            isTrained = true;
            break;
            
        case REGRESSION_SVM:
            dlib::deserialize(regression_function, fin);
            isTrained = true;
            break;
            
        case REGRESSION_MLP:
            // this needs to be fixed...
            //dlib::deserialize(*mlp_trainer, fin);
            break;
            
        case CLUSTERING:
            // this needs to be implemented
            //
            break;
            
        default:
            return;
    }
    ofLog(OF_LOG_VERBOSE, "Loaded model: "+path);
}

//---------
ofxLearn::~ofxLearn() {
    if (learnMode == REGRESSION_MLP && isTrained) {
        delete mlp_trainer;
    }
}
