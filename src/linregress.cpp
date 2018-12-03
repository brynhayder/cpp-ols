//
// Created by Bryn Elesedy on 12/11/2018.
// Linear regression with a single output dimension


#include <iostream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>


using Eigen::MatrixXd;
using Eigen::VectorXd;


VectorXd ridge_weights(MatrixXd X, VectorXd y, double lambda);
VectorXd predict(VectorXd weights, MatrixXd xvals);

int main() {

    int n_examples = 5;
    int n_features = 1;
    double s = 2;
    double err_size = 0.1;

    VectorXd v = VectorXd::Constant(n_examples, 1);
    VectorXd u = VectorXd::LinSpaced(n_examples, 0, 4);


    MatrixXd X(v.rows(), v.cols() + 2 * u.cols());
    X << v,
         -u,
         u;

    VectorXd eps = err_size * VectorXd::Random(n_examples);
    VectorXd y = u + eps;

    std::cout << "X = " << X << std::endl;
    std::cout << "Y = " << y << std::endl;

    double l = 1.0;
    VectorXd b = ridge_weights(X, y, l);
    std::cout << "beta = " << b << std::endl;

    VectorXd predictions = predict(b, u);

    std::cout << "predictions = " << predictions << std::endl;
    return 0;
}

// need to add a constant to X (!!!)
// These are column stacked training examples

// would be good if we could enforce the size of these two things to be the same
VectorXd ridge_weights(MatrixXd X, VectorXd y, double lambda){
    // NEED TO CHANGE THIS TO USE QR (or something) WHEN LAMBDA \NEQ 0
    if (lambda <= 0){
        throw domain_error("Ridge parameter lambda must be > 0.");
    }
    Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(X.cols(), X.cols());
    return (lambda * eye + X.transpose() * X).llt().solve(X.transpose() * y);
}


VectorXd ols_weights(MatrixXd X, VectorXd y){

}

/*
 * There is some confusion on how the multi output dim stuff works and whether X should
 * be column or row stacked. Need to work this out, but stuff should (surely?) still work for
 * multiple input and output dims and also should be okay as long as consistency in maintained.
 *
 * Now in the process of making the predictions. Might be worth doing that and getting a couple
 * of other statistics on the fit (i.e. RMSE or something) and then put it into a class. Maybe
 * that would be enough?
 *
 * When you do the class maybe you can figure out the number of dims in advance and save some
 * compile time?
 *
 *
 * Notes:
 * - figure out this thing about column major ordering in Eigen
 * (is it normal and is it just indexing?)
 * - get the QR decomp for when lambda is non-zero. maybe want householder QR,
 *   maybe also need to make sure the rank is checked?
 *
 * Goal here is to get something that can do linear regression on data from a text file
 * and can be used from the command line
 *
 */


VectorXd predict(VectorXd weights, MatrixXd xvals) {
    double offset = weights(0);
    VectorXd slope = weights.tail(weights.rows() - 1);
    VectorXd output = ((xvals * slope).array() + offset).matrix();

    std::cout << xvals << slope << output << std::endl;

    return output;
}