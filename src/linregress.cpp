//
// Created by Bryn Elesedy on 12/11/2018.
// Linear regression with a single output dimension


#include <iostream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>


using Eigen::MatrixXd;
using Eigen::VectorXd;


VectorXd ols_weights(MatrixXd X, VectorXd y);


int main() {

    int n_examples = 5;
    int n_features = 1;
    double s = 2;
    double err_size = 0.1;

    VectorXd v = VectorXd::Constant(n_examples, 1);
    VectorXd u = VectorXd::LinSpaced(n_examples, 0, 4);

    MatrixXd X(v.rows(), v.cols() + u.cols());
    X << v,
         u;

    VectorXd eps = err_size * VectorXd::Random(n_examples);
    VectorXd y = u + eps;

    std::cout << "X = " << X << std::endl;
    std::cout << "Y = " << y << std::endl;

    VectorXd b = ols_weights(X, y);
    std::cout << "beta = " << b << std::endl;

    std::cout << b .dot(b) <<std::endl;

    VectorXd predictions = predict(ols_weights, u);

    std::cout << "predictions = " << predictions << std::endl;

    return 0;
}

// need to add a constant to X (!!!)
// These are column stacked training examples

// would be good if we could enforce the size of these two things to be the same
VectorXd ols_weights(MatrixXd X, VectorXd y){
    // y_hat = (XX^T)^-1 Xy etc etc use cholesky! LLT
    return (X.transpose() * X).llt().solve(X.transpose() * y);
}

/*
 * There is some confusion on how the multi output dim stuff works and whether X should
 * be column or row stacked. Need to work this out, but stuff should (surely?) still work for
 * multiple input and output dims and also should be okay as long as consistency in maintained.
 *
 * Now in the process of making the predictions. Might be worth doing that and getting a couple
 * of other statistics on the fit (i.e. RMSE or something) and then put it into a class. Maybe
 * that would be enough?
 */



VectorXd predict(VectorXd weights, VectorXd x_vals) {
    double offset = weights(0); // Need to change this to take the row rather than the element...
    VectorXd slope = weights.tail(weights.size() - 1);

    VectorXd output = (slope.dot(x_vals) + offset).matrix();
    return output;
}