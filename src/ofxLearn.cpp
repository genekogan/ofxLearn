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
    // Now setup a SVR trainer object.  It has three parameters, the kernel and
    // two parameters specific to SVR.
    
    // This parameter is the usual regularization parameter.  It determines the trade-off
    // between trying to reduce the training error or allowing more errors but hopefully
    // improving the generalization of the resulting function.  Larger values encourage exact
    // fitting while smaller values of C may encourage better generalization.
    
    // Epsilon-insensitive regression means we do regression but stop trying to fit a data
    // point once it is "close enough" to its target value.  This parameter is the value that
    // controls what we mean by "close enough".  In this case, I'm saying I'm happy if the
    // resulting regression function gets within 0.001 of the target value.
    
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
    // First, get our labeled set of training data
    //generate_data(samples, labels);
    
    cout << "samples.size(): "<< samples.size() << endl;
    
    // The main object in this example program is the one_vs_one_trainer.  It is essentially
    // a container class for regular binary classifier trainer objects.  In particular, it
    // uses the any_trainer object to store any kind of trainer object that implements a
    // .train(samples,labels) function which returns some kind of learned decision function.
    // It uses these binary classifiers to construct a voting multiclass classifier.  If
    // there are N classes then it trains N*(N-1)/2 binary classifiers, one for each pair of
    // labels, which then vote on the label of a sample.
    //
    // In this example program we will work with a one_vs_one_trainer object which stores any
    // kind of trainer that uses our sample_type samples.
    
    
    // Finally, make the one_vs_one_trainer.
    
    
    // Next, we will make two different binary classification trainer objects.  One
    // which uses kernel ridge regression and RBF kernels and another which uses a
    // support vector machine and polynomial kernels.  The particular details don't matter.
    // The point of this part of the example is that you can use any kind of trainer object
    // with the one_vs_one_trainer.
    
    
    
    // make the binary trainers and set some parameters
    
    poly_trainer.set_kernel(poly_kernel_type(0.1, 1, 2));
    rbf_trainer.set_kernel(rbf_kernel_type(0.1));
    
    
    // Now tell the one_vs_one_trainer that, by default, it should use the rbf_trainer
    // to solve the individual binary classification subproblems.
    trainer.set_trainer(rbf_trainer);
    // We can also get more specific.  Here we tell the one_vs_one_trainer to use the
    // poly_trainer to solve the class 1 vs class 2 subproblem.  All the others will
    // still be solved with the rbf_trainer.
    
    //trainer.set_trainer(poly_trainer, 1, 2);
    
    // Now let's do 5-fold cross-validation using the one_vs_one_trainer we just setup.
    // As an aside, always shuffle the order of the samples before doing cross validation.
    // For a discussion of why this is a good idea see the svm_ex.cpp example.
    randomize_samples(samples, labels);
    
    
    //cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
    
    
    // The output is shown below.  It is the confusion matrix which describes the results.  Each row
    // corresponds to a class of data and each column to a prediction.  Reading from top to bottom,
    // the rows correspond to the class labels if the labels have been listed in sorted order.  So the
    // top row corresponds to class 1, the middle row to class 2, and the bottom row to class 3.  The
    // columns are organized similarly, with the left most column showing how many samples were predicted
    // as members of class 1.
    //
    // So in the results below we can see that, for the class 1 samples, 60 of them were correctly predicted
    // to be members of class 1 and 0 were incorrectly classified.  Similarly, the other two classes of data
    // are perfectly classified.
    //  cross validation:
    //  60  0  0
    //   0 70  0
    //  0  0 80
    
    // Next, if you wanted to obtain the decision rule learned by a one_vs_one_trainer you
    // would store it into a one_vs_one_decision_function.
    
    cout << "go 1 " << endl;
    df = trainer.train(samples, labels);
    cout << "go 2 " << endl;
    
    
    //cout << "predicted label: "<< df(samples[0])  << ", true label: "<< labels[0] << endl;
    
    
    return;
    
    cout << "predicted label: "<< df(samples[90]) << ", true label: "<< labels[90] << endl;
    
    
    
    // The output is:
    
    // predicted label: 2, true label: 2
    // predicted label: 1, true label: 1
    
    
    
    // If you want to save a one_vs_one_decision_function to disk, you can do
    // so.  However, you must declare what kind of decision functions it contains.
    dlib::one_vs_one_decision_function<ovo_trainer,
    dlib::decision_function<poly_kernel_type>,  // This is the output of the poly_trainer
    dlib::decision_function<rbf_kernel_type>    // This is the output of the rbf_trainer
    > df2, df3;
    
    
    // Put df into df2 and then save df2 to disk.  Note that we could have also said
    // df2 = trainer.train(samples, labels);  But doing it this way avoids retraining.
    df2 = df;
    
    //serialize("df.dat") << df2;
    
    // load the function back in from disk and store it in df3.
    //deserialize("df.dat") >> df3;
    
    
    // Test df3 to see that this worked.
    cout << endl;
    cout << "predicted label: "<< df3(samples[0])  << ", true label: "<< labels[0] << endl;
    cout << "predicted label: "<< df3(samples[90]) << ", true label: "<< labels[90] << endl;
    // Test df3 on the samples and labels and print the confusion matrix.
    cout << "test deserialized function: \n" << test_multiclass_decision_function(df3, samples, labels) << endl;
    
    
    
    
    // Finally, if you want to get the binary classifiers from inside a multiclass decision
    // function you can do it by calling get_binary_decision_functions() like so:
    //     dlib::one_vs_one_decision_function<ovo_trainer>::binary_function_table functs;
    //     functs = df.get_binary_decision_functions();
    //     cout << "number of binary decision functions in df: " << functs.size() << endl;
    //     // The functs object is a std::map which maps pairs of labels to binary decision
    // functions.  So we can access the individual decision functions like so:
    //     dlib::decision_function<poly_kernel> df_1_2 = any_cast<decision_function<poly_kernel> >(functs[make_unordered_pair(1,2)]);
    //     dlib::decision_function<rbf_kernel>  df_1_3 = any_cast<decision_function<rbf_kernel>  >(functs[make_unordered_pair(1,3)]);
    // df_1_2 contains the binary decision function that votes for class 1 vs. 2.
    // Similarly, df_1_3 contains the classifier that votes for 1 vs. 3.
    
    // Note that the multiclass decision function doesn't know what kind of binary
    // decision functions it contains.  So we have to use any_cast to explicitly cast
    // them back into the concrete type.  If you make a mistake and try to any_cast a
    // binary decision function into the wrong type of function any_cast will throw a
    // bad_any_cast exception.
    
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


//////////////////////////////////////////////////////////
//
//
//
//
//
//
//
//
//
//
//
//
//
////
//
//
//
//
//
////
//
//
//
//
//
//
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


void ofxLearnSVM::generate_data (
                                 std::vector<sample_type>& samples,
                                 std::vector<double>& labels
                                 )
{
    const long num = 50;
    
    sample_type m(2);
    
    dlib::rand rnd;
    
    
    // make some samples near the origin
    double radius = 0.5;
    for (long i = 0; i < num+10; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));
        
        // add this sample to our set of training samples
        samples.push_back(m);
        labels.push_back(1);
    }
    
    // make some samples in a circle around the origin but far away
    radius = 10.0;
    for (long i = 0; i < num+20; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));
        
        // add this sample to our set of training samples
        samples.push_back(m);
        labels.push_back(2);
    }
    
    // make some samples in a circle around the point (25,25)
    radius = 4.0;
    for (long i = 0; i < num+30; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));
        
        // translate this point away from the origin
        m(0) += 25;
        m(1) += 25;
        
        // add this sample to our set of training samples
        samples.push_back(m);
        labels.push_back(3);
    }
}