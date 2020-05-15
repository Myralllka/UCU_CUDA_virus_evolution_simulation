## Lab 2 (performance testing)

    Topic:      Two variables integral parallel calculation.
    
    Task:       Find the integral of the given Langermann function, 
                where m, ci, a1i, a2i, i = 1..m -- f_params.
                
                    m=5
                    a1 = [1, 2, 1, 1, 5]
                    a2 = [4, 5, 1, 2, 4]
                    c = [2, 1, 4, 7, 2]
                    
                In intervals: x = -10..10; y = -10 .. 10.
                
    Latex Function Formula: 
    $$ f(x_1, x_2) = \sum_{i=1}^{m} c_i exp(- \frac{1}{\pi} ((x_1 - a_{1i})^2 + (x_2 - a_{2i})^2)) cos(\pi((x_1 - a_{1i})^2 + (x_2 - a_{2i})^2)) $$
    
    Expected result: -1.604646665
    Task Number:    4
    Author:         Yuriy Pasichnyk 
----
### Function:
![formula](https://render.githubusercontent.com/render/math?math=f(x_1,x_2)=\sum_{i=1}^{m}c_i%20exp(-\frac{1}{\pi}%20((x_1%20-%20a_{1i})^2%20+%20(x_2%20-%20a_{2i})^2))%20cos(\pi((x_1%20-%20a_{1i})^2%20+%20(x_2%20-%20a_{2i})^2)))

### Run: './start.sh --help' to view possible options
First time running you must use `-c` option to compile the project.

### Project Structure

    - [cmake-build-debug]   # build directory
    - include               # headers
    - src                   # source files
        
    ## Names in [] are generated. 

    start.sh - interface to work with the project (run './start.sh --help' for more info)
    execution.conf - is the file with all configuration values (well self documented)
    
    
#### Conf file structure  
    - Absolute precision
    - Relative precision (not less than 0.001)
    - Number of parallel execution flows
    - Initial number of integration steps
    - Initial max number of integration steps
    - x region start
    - x region end
    - y region start
    - x region end
    - function parameters (arrays - a1, a2, c, coefitient - m)
