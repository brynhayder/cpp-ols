import os
import numpy as np

base_path = r"/Users/bryn/Programming/cpp-ols/example_data"

if __name__ == "__main__":
    np.random.seed(0)


    n_examples = 1000
    sigma1 = 0.1
    sigma2 = 1

    x = np.linspace(-10, 10, n_examples)


    X = np.column_stack([x, x[::-1]]) #, np.random.shuffle(x)])
    y = sum([5 * x, 2 *  x[::-1]])  + np.random.randn(n_examples) * sigma1

    out = np.column_stack([X, y])

    np.savetxt(
        X=out,
        fname=os.path.join(base_path, "multiple_inputs.csv"),
        delimiter=','
    )

    y_single = x - 2 + np.random.randn(n_examples) * sigma2

    out_single = np.column_stack([x, y_single])
    np.savetxt(
        X=out_single,
        fname=os.path.join(base_path, "single_input.csv"),
        delimiter=','
    )


