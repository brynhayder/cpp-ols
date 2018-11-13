

- io
    - could just get some generic thing to take data from .csv
    - can you spit stuff out at the end to .csv?
    - what about the program making charts??? (surely this sucks!)
        - this is probably a step too far
- handling data
    - hopefully can load in as double and put into a matrix
    - There may be something to think about with the dimensions of the data


- the problem itself
    - get in the matrices
    - transpose etc, calculate cholesky decomp or w/e
    - get predictions
    - calculate other statistics
    - make some benchmarking tests (maybe you can just do this from the command line?)
    i.e. using timeit util or something (or maybe you can write a timeit yourself?)


- Design:
    - you could make a function that returns a struct
    - you could make a class


Notes:
- project structure
- tools like Make or CMake
- how to handle this eigen file
    - don't want to put it in usr/local/include since that is global
    - seems odd to have all of this raw code just sitting there
    - maybe it is a good idea to just compile it and include it as a library or something?
- is there a C++ requirements file?
- is there a way of making a virtual environment for compiling C++ programs, so that
the compiler knows where to look? (is this just achieved with environment variables?)

