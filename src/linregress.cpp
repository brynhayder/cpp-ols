//
// Created by Bryn Elesedy on 12/11/2018.
// Linear regression with a single output dimension

#include <getopt.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

const double DEFAULT_LAMBDA_VALUE = 0;
const std::streamsize DEFAULT_PREC = 2;
const char DELIM = ',';
const std::string INPUT_FILE_EXTENSION = ".csv";
const std::string OUTPUT_PREDICTION_EXTENSION = ".fittedvalues";
const std::string OUTPUT_WEIGHTS_EXTENSION = ".weights";



// ----------------
// Parse command line arguments

struct command_line_args {
    command_line_args() : lambda(DEFAULT_LAMBDA_VALUE), prec(DEFAULT_PREC) {}

    double lambda;
    std::streamsize prec;
    std::string input_filename;

};

void print_help_message() {
    std::cout << "==========================================================\n";
    std::cout << ">>>>>>>>>>>>>>>>>L2 Regularised Regression<<<<<<<<<<<<<<<<\n";
    std::cout << "==========================================================\n";
    std::cout << "Usage: linregress <path/to/input_file> [--lambda] [--prec]\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Parameters:\n";
    std::cout << "input_file: File containing training data, must be csv.\n"
                 "            Row stacked training examples.\n"
                 "            Final column the y values, others form X.\n";
    std::cout << "--lambda (numeric): Ridge regularisation parameter.\n"
                 "                    Must be >=0, 0 gives OLS regression.\n"
                 "                    Default is " << DEFAULT_LAMBDA_VALUE << "\n";
    std::cout << "--prec (int): Significant figures for console output.\n"
                 "              Default is " << DEFAULT_PREC << "\n";
    std::cout << "==========================================================\n";
    return;
}



command_line_args parse_args(int argc, char *argv[]) {
    const char *const short_opts = "l:h";
    const option long_opts[] = {
            {"lambda", required_argument, nullptr, 'l'},
            {"prec",   required_argument,       nullptr, 'p'},
            {"help",   no_argument,       nullptr, 'h'},
    };

    command_line_args output;
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, &option_index)) != -1) {
        switch (opt) {
            case 'l':
                output.lambda = std::stod(optarg);
                break;
            case 'p':
                output.prec = (std::streamsize) std::stoi(optarg);
                break;
            case 'h':
            case '?':
            default:
                print_help_message();
                exit(EXIT_SUCCESS);
        }
    }

    if (optind < argc) {
        output.input_filename = argv[optind++];
    } else {
        std::cerr << "Failure: must give an input file." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (optind < argc) {
        std::cerr << "Warning: " << argc - optind << " additional arguments given, all ignored!\n" << std::endl;
    }

    return output;
}

void check_args(command_line_args &args) {

    if (args.lambda < 0) {
        std::cerr << "Error: Ridge regularisation parameter lambda must be >= 0" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (args.prec < 0) {
        std::cerr << "Warning: given output precision < 0." << std::endl;
    }

    return;
}

// ----------------



// ----------------
// Parse inputs into Eigen matrices

struct data_size {
    std::size_t n_features;
    std::size_t n_examples;
};

struct input_data {
    MatrixXd X;
    VectorXd y;
};


data_size get_data_size(std::string input_filename, char delim) {
    std::ifstream infile(input_filename);

    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open " << input_filename << std::endl;
        exit(EXIT_FAILURE);
    }

    data_size input_sizes;
    std::string line;
    std::size_t lc = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) {
            continue;

        }

        if (lc == 0) {
            input_sizes.n_features = std::count(line.begin(), line.end(), delim);
        }
        lc++;
    }
    input_sizes.n_examples = lc;

    infile.close();
    return input_sizes;
}


input_data read_input_file(std::string input_filename, data_size input_sizes, char delim) {

    std::ifstream infile(input_filename);

    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open " << input_filename << std::endl;
        exit(EXIT_FAILURE);
    }

    MatrixXd X(input_sizes.n_examples, input_sizes.n_features);
    VectorXd y(input_sizes.n_examples);

    std::size_t col = 0;
    std::size_t row = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::stringstream this_line(line);
        std::string item;

        while (std::getline(this_line, item, delim)) {
            if (item.empty()) {
                continue;
            }

            if (col == input_sizes.n_features) {
                y(row) = std::stod(item);
                col = 0;
                row++;
            } else {
                X(row, col) = std::stod(item);
                col++;
            }
        }

    }
    infile.close();

    input_data output;
    output.X = X;
    output.y = y;
    return output;
}

// ----------------



// ----------------
// Do the Regression

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

    Eigen::HouseholderQR <MatrixXd> qr_decomp;
    VectorXd weights;

public:
    RidgeRegression(double lambda) : lambda(lambda), is_fitted(false) {}

    void fit(MatrixXd X, VectorXd y, bool add_constant);

    VectorXd predict(MatrixXd x_vals);

    inline VectorXd get_weights() { return this->weights; }

    inline bool has_const() { return this->has_constant; }
};


void RidgeRegression::fit(MatrixXd X, VectorXd y, bool add_constant) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have same number of rows.");
    }

    MatrixXd X_train = (add_constant) ? append_ones_column(X) : X;

    Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(X_train.cols(), X_train.cols());

    this->qr_decomp = (this->lambda * eye + X_train.transpose() * X_train).householderQr();
    this->weights = this->qr_decomp.solve(X_train.transpose() * y);
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
        return ((xvals * slope).array() + offset).matrix();
    } else {
        return xvals * this->weights;
    }
}
// ----------------



// ----------------
// Output

double rmse(VectorXd y1, VectorXd y2) {
    return std::sqrt((y1 - y2).array().pow(2).mean());
}


void print_results(RidgeRegression regressor, double rmse_, std::streamsize prec) {

    VectorXd weights = regressor.get_weights();

    std::streamsize curr_prec = std::cout.precision();
    std::cout << "Fitted Model: y = ";
    std::cout << std::setprecision(prec);
    int i = 0;
    while (i < weights.size() - 1) {
        std::cout << weights(i) << " x" << i + 1 << " + ";
        i++;
    }

    std::cout << weights(i) << std::endl;
    std::cout << "RMSE: " << rmse_ << std::endl;
    std::cout << std::setprecision(curr_prec);
    return;
}

std::string get_output_filename(std::string input_filename, std::string old_ext, std::string new_ext) {
    std::string prefix;
    int d = input_filename.size() - old_ext.size();
    if (d > 0 && input_filename.compare(d, old_ext.size(), old_ext) == 0) {
        prefix = input_filename.substr(0, d);
    } else {
        prefix = input_filename;
    }
    return prefix + new_ext;
}

void save_vector(VectorXd fitted_values, std::string filename) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Failed out open output file " << filename << " values not written." << std::endl;
        return;
    }

    for (std::size_t i = 0; i < fitted_values.rows(); i++) {
        outfile << fitted_values(i) << std::endl;
    }

    outfile.close();
    return;
}

// ----------------



int main(int argc, char *argv[]) {

    command_line_args args = parse_args(argc, argv);

    check_args(args);

    data_size dims = get_data_size(args.input_filename, DELIM);
    input_data data = read_input_file(args.input_filename, dims, DELIM);

    std::cout << "=================================================" << std::endl;
    std::cout << "Least Squares Regression. L2 regularisation = " << args.lambda << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Read data from " << args.input_filename << ", found " << dims.n_features << " features and ";
    std::cout << dims.n_examples << " examples" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    RidgeRegression regressor(args.lambda);
    // Previously considered giving user option to not include constant
    // in regression but I changed my mind.
    regressor.fit(data.X, data.y, true);

    VectorXd fitted_values = regressor.predict(data.X);

    print_results(regressor, rmse(data.y, fitted_values), args.prec);
    save_vector(
            fitted_values,
            get_output_filename(args.input_filename, INPUT_FILE_EXTENSION, OUTPUT_PREDICTION_EXTENSION)
            );

    save_vector(
            regressor.get_weights(),
            get_output_filename(args.input_filename, INPUT_FILE_EXTENSION, OUTPUT_WEIGHTS_EXTENSION)
    );
    std::cout << "=================================================" << std::endl;

    return 0;

}
