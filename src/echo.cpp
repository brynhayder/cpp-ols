//
// Created by Bryn Elesedy on 03/12/2018.
//

#include <getopt.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>


struct input_data {
    MatrixXd X;
    VectorXd y;
};

void read_input_file(std::string input_filename){
    input_data inputs;

    std::ifstream infile(input_filename);

    if (!infile.is_open()) {
        std::cerr << "Failed to open " << input_filename << std::endl;
        exit(EXIT_FAILURE);
    }

    while (std::getline(infile, line)) {
        std::cout << line << std::endl;
    }

    // read file

    return ;
}

int main(){
    read_input_file("test_inputs.txt")
}