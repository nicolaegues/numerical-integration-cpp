
## Numerical Integration and Parallelization with OpenMP

This project implements and evaluates the accuracy and performance of four numerical integration methods - Midpoint, Leftpoint, Trapezoidal, and Simpson's rule - using C++ and with support for parallelization using OpenMP. 

By estimating the integral of a function whose analytical result is known, each method is compared in terms of absolute error and mean execution time. Additionally, the script analyses the effect of spacing (the width of the subinterval) on the accuracy of these methods as well as how the execution time of the Midpoint method scales with an increasing number of threads. 

Optionally, the calculations can be run and plotted by using a Python (ctypes) wrapper for the main C++ code. The produced plots can be found below. 

### How to run

- To run only the C++ code: 
    - Compile: 
        ```
        g++ -std=c++17 -fopenmp integration_methods.cpp -o integration_methods
        ```

    - Execute: 
        ```   
        ./integration_methods
        ```

- Alternatively, to run the code and plot the results: 

    - Compile the C++ source code into a shared library:
        ```
        g++ -std=c++17 -fopenmp -fPIC -shared integration_methods.cpp -o integration_library.so
        ```

    - Run the Python wrapper: 
        ```
        python integration_wrapper.py
        ```

The slurm script to run the code in a High-Performance Computing (HPC) environment is included under  `run_integration_hpc.sh`.

### Output plots

- Absolute Error vs Spacing: 

    <img src="https://github.com/nicolaegues/numerical-integration-cpp/blob/main/plots/error_vs_spacing.png" width="70%" height="70%">

- Time and Speedup vs Number of Threads: 

    <img src="https://github.com/nicolaegues/numerical-integration-cpp/blob/main/plots/time_vs_threads.png" width="70%" height="70%">



