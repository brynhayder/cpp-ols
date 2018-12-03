//
// Created by Bryn Elesedy on 12/11/2018.
// Linear regression with a single output dimension

#include <getopt.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

double DEFAULT_LAMBDA_VALUE = 0;
char DELIM = ',';



// ----------------
// Parse command line arguments

struct command_line_args {
    command_line_args() : lambda(DEFAULT_LAMBDA_VALUE) {}

    double lambda;
    std::string input_filename;

};

void print_help_message() {
    std::cout << "help!" << std::endl;
    return;
}

command_line_args parse_args(int argc, char *argv[]) {
    const char *const short_opts = "l:h";
    const option long_opts[] = {
            {"lambda", required_argument, nullptr, 'l'},
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

            case 'h':
            case '?':
            default:
                print_help_message();
                exit(EXIT_SUCCESS);
        }
    }

    // sort out the below!!!
    if (optind < argc) { // IS THIS THE BEST WAY?
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
        std::cerr << "Failed to open " << input_filename << std::endl;
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
        std::cerr << "Failed to open " << input_filename << std::endl;
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
};


void RidgeRegression::fit(MatrixXd X, VectorXd y, bool add_constant) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have same number of rows.");
    }

    MatrixXd X_train = (add_constant) ? append_ones_column(X) : X;

    Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(X_train.cols(), X_train.cols());


    this->qr_decomp = (this->lambda * eye + X_train.transpose() * X_train).householderQr();


    MatrixXd w = this->qr_decomp.solve(X_train.transpose() * y);
    this->weights = Eigen::Map<VectorXd>(w.data(), w.size());

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



void print_results(VectorXd weights, std::streamsize prec) {
    std::streamsize curr_prec = std::cout.precision();
    std::cout << std::internal;
    std::cout << "Fitted Model\n";
    std::cout << std::left;
    std::cout << "y = ";
    std::cout << std::setprecision(prec);

    int i = 0;
    while (i < weights.size() - 1) {
        std::cout << weights(i) << " x" << i + 1 << " + ";
        i++;
    }

    std::cout << weights(i) << std::endl;

    std::cout << std::setprecision(curr_prec);
    return;
}


int main(int argc, char *argv[]) {

    command_line_args args = parse_args(argc, argv);
    data_size dims = get_data_size(args.input_filename, DELIM);
    input_data data = read_input_file(args.input_filename, dims, DELIM);

    RidgeRegression regressor(args.lambda);
    regressor.fit(data.X, data.y, true);

    VectorXd b = regressor.get_weights();

    VectorXd u = VectorXd::LinSpaced(dims.n_examples, 0, 4);
    MatrixXd xvals = u.replicate(1, 2);

    VectorXd predictions = regressor.predict(xvals);

    std::streamsize prec = 2;
    print_results(b, prec);

    return 0;

}


/*
 * APPARENTLY THE USE OF QR IS MORE COMMON FOR LEAST SQUARES PROBLEMS !!!!!!!!! BALLS

 *  Maybe it is easier to assume full rank of the X matrix?
 *  need ot do something witht the predictions
 *  also need to take the output prec from the command line args.
 *
 *
 * TODO:
 * - need to take the adding a constant from the command line args
 * - reading input fule
 * - write usage/help message
 * - use QR rather than cholesky?
 * - print table, give output, make predictions, etc.
 * - ability to do OLS regression too!
 * - may need some CMAKE type stuff. probs worth reading the embedded systems labs stuff
 *
 * The program can print a table that has
 * < regression results>
 * < model: y ~ a1 x1 ... etc.>
 * t-stats (for each coeff)
 * RMSE
 * log likelihood
 * t-stat calc from (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-372)
 *
 *
 *
 * Now in the process of making the predictions. Might be worth doing that and getting a couple
 * of other statistics on the fit (i.e. RMSE or something) and then put it into a class. Maybe
 * that would be enough? (just get a couple of sample stats like beta, R^2, T-stat, etc.
 * could print the model like y = a1x1 + ... + b
 *
 *
 * When you do the class maybe you can figure out the number of dims in advance and save some
 * compile time?
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
