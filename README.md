# cpp-ols

## Usage
`g++ -std=c++11 linregress.cpp -o linregress`

`./linregress ../example_data/single_inputs.csv`

## Examples
The directory `scripts` contains the Python code used to generate the
examples. 
1. Multiple inputs used y = 5 * x1 + 2 * x2 + N(0, 0.1)
2. Single input used y = x - 2 + N(0, 1)  


## Dependencies
- Eigen 3.3.5
- C++11 (I think)
