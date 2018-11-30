//
// Created by Bryn Elesedy on 12/11/2018.
// Linear regression with a single output dimension


#include <iostream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;


MatrixXd append_ones_column(MatrixXd X) {
    MatrixXd new_x(X.rows(), X.cols() + 1);
    VectorXd c = VectorXd::Constant(X.rows(), 1.);
    new_x << X, c;
    return new_x;
}

class RidgeRegression {
private:
    double lambda;
    bool has_constant;
    bool is_fitted;
    /*
     * Some of these aren't in the constructor, so it might be worth thinking about their
     * initialisation.
     */
//    MatrixXd x_train;
//    VectorXd y_train;


    Eigen::LLT <MatrixXd> cholesky_decomp; // Will only need this for errors etc. (i think)
    VectorXd weights;

public:
    RidgeRegression(double lambda) : lambda(lambda), is_fitted(false) {}

    void fit(MatrixXd X, VectorXd y, bool add_constant);

    VectorXd predict(MatrixXd x_vals);

    inline VectorXd get_weights() { return this->weights; }
};


void RidgeRegression::fit(MatrixXd X, VectorXd y, bool add_constant) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have same number of rows.");
    }

    MatrixXd X_train = (add_constant) ? append_ones_column(X) : X;

    Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(X_train.cols(), X_train.cols());

    this->cholesky_decomp = (this->lambda * eye + X_train.transpose() * X_train).llt();
    this->weights = cholesky_decomp.solve(X_train.transpose() * y);
    this->has_constant = add_constant;
    this->is_fitted = true;
    return;
}

VectorXd RidgeRegression::predict(MatrixXd xvals) {
    if (!this->is_fitted) {
        throw std::domain_error("Need to fit regression before prediction.");
    }

    if (has_constant) {
        double offset = this->weights(this->weights.rows() - 1);
        VectorXd slope = this->weights.head(this->weights.rows() - 1);

        std::cout << "offset is " << offset << std::endl;
        std::cout << "slope is \n" << slope << std::endl;

        return ((xvals * slope).array() + offset).matrix();
    } else {
        return xvals * this->weights;
    }
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
 *   need to implement the ols stuff as well
 *   Possibly worth just couching this as bayes lin regress
 *   potentially some ordering issues with eigen? also need to make sure we stack
 *   examples correctly
 *
 *   decide which stats to show, how to write output etc.
 *   can pick a few from the python version
 */



int main() {

    int n_examples = 5;
    double err_size = 0.1;

    VectorXd u = VectorXd::LinSpaced(n_examples, 0, 4);
    VectorXd u2 = VectorXd::LinSpaced(n_examples, 4, 0);

    MatrixXd X(n_examples, 2 * u.cols());
    X << u, u2;

    std::srand(0);
    VectorXd eps = err_size * VectorXd::Random(n_examples);
    VectorXd y = u + eps;

    std::cout << "X = \n" << X << std::endl;
    std::cout << "Y = \n" << y << std::endl;

    RidgeRegression regressor(1.0);
    regressor.fit(X, y, true);

    VectorXd b = regressor.get_weights();

    std::cout << "beta = \n" << b << std::endl;

    MatrixXd xvals = u.replicate(1, 2);

    std::cout << xvals << std::endl;

    VectorXd predictions = regressor.predict(xvals);
    std::cout << "predictions = \n" << predictions << std::endl;

    return 0;
}