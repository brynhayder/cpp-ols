//
// Created by Bryn Elesedy on 12/11/2018.
//

#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    MatrixXd m(2, 2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    return 0;
}


double slope(MatrixXd X, VectorXd y){
    // y_hat = (XX^T)^-1 Xy etc etc use cholesky! LLT
    return
}