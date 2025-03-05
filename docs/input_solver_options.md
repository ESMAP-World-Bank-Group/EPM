# Solver options

Solving an instance of the EPM model requires solving a mixed-integer programming instance with thousands of variables and constraints. The solver options chosen will then be determining to reduce computational time.

The solver options are defined in the `cplex.opt` file. Checkout [here](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/cplex.opt) a simple example of the file.

## Important parameters
- **lpmethod**: determining the algorithm used to solve the LP problem. Options include primal simplex and barrier algorithm.
- **threads**: number of parallel threads allowed. This parameter can speed up the performance of the algorithm. The speedup is far from linear, there is usually a sweet spot between 8 and 12. When setting the number of threads in the config file, you can specify a value higher than the number of logical threads on your computer, as solvers can benefit from oversubscription through better load balancing, asynchronous task execution, and OS-level scheduling. However, setting the thread count too high leads to excessive context switching, increased memory and cache contention, and diminishing returns due to solver synchronization overhead. In practice, performance often improves when exceeding the available threads slightly (e.g., using 12 threads on an 8-thread machine) but degrades if set too high (e.g., 16 threads), as the system spends more time managing threads than solving the problem. Testing different values is key to finding the optimal balance.
- **predual**: Switch to give dual problem to the optimizer. The setting for this parameter can either be a big winner or big looser or no impact if default makes the “right decision”.
- **barcolnz**: dense column handling. Determines whether or not columns are considered dense for special barrier algorithm handling. At the default setting of 0, this parameter is determined dynamically. Values above 0 specify the number of entries in columns to be considered as dense. This parameter can have a huge impact, but is usually difficult to tune, and is specific to each model instance.

## Other parameters


### Good Options for Large Scale Models
- **threads**: `4`
- **lpmethod**: `4`
- **solutiontype**: `2`
- **bardisplay**: `2`

---

### Long Presolve Time:
One can suppress certain "auxiliary tasks" in the root node via the option `auxrootthreads`. After doing so, CPLEX starts solving the root node after a moderate presolve time.

---

### Long RIP Run Time
The `barcolnz` parameter can have a big impact. For instance, with a value of `500`, CPLEX is significantly faster than with `700`.  
The option `lpmethod` specifies the algorithm for solving LPs, but there is a dedicated option `startalg` for specifying the algorithm for the root LP when solving a MIP.  

**Note**: `solutiontype 2` should be used with caution when solving MIPs. A basic solution (e.g., from the simplex algorithm or after crossover when using the barrier) is usually beneficial.

---

### Detailed Options

### `startalg`
- **Description**: Specifies the algorithm to use for starting the solution process.
- **Possible Values**:
  - `0`: Primal simplex
  - `1`: Dual simplex
  - `2`: Network
  - `3`: Barrier
  - `4`: Automatic (CPLEX chooses the algorithm)
- **Recommended**: `4` (Automatic)

---

### `scaind`
- **Description**: Controls the use of scaling for the constraint matrices.
- **Possible Values**:
  - `-1`: No scaling
  - `0`: No scaling
  - `1`: Automatic scaling
- **Recommended**: `1` (Automatic)

---

### `lpmethod`
- **Description**: Chooses the method for solving linear programs.
- **Possible Values**:
  - `0`: Primal simplex
  - `1`: Dual simplex
  - `2`: Network method
  - `3`: Barrier
  - `4`: Automatic (CPLEX chooses the algorithm)
- **Recommended**: `4` (Automatic)

---

### `threads`
- **Description**: Sets the number of threads to use for solving.
- **Possible Values**: Any positive integer (e.g., `1`, `2`, `4`, `8`)
- **Recommended**: Depends on hardware; more threads reduce solution time but increase memory usage.

---

### `predual`
- **Description**: Controls whether CPLEX solves the dual or primal problem during the presolve phase.
- **Possible Values**:
  - `-1`: Automatic
  - `1`: Dual
  - `0`: Primal
- **Recommended**: `-1` (Automatic)

---

### `baralg`
- **Description**: Specifies the barrier algorithm to use.
- **Possible Values**:
  - `0`: Standard barrier
  - `1`: Barrier with crossover
- **Recommended**: `1` (Barrier with crossover)

---

### `barcolnz`
- **Description**: Limits the number of nonzeros in columns during the barrier solution.
- **Possible Values**: Any positive integer
- **Recommended**: `500` for faster solutions.

---

### Additional Options:
- **`barepcomp`**: Sets the barrier algorithm's complementarity tolerance.  
  Recommended: `1e-5`.

- **`barcrossalg`**: Determines the crossover algorithm after the barrier.  
  Recommended: `1` (Dual simplex).

- **`solutiontype`**: Specifies the type of solution to produce.  
  Recommended: `1` for a basic solution, `2` for an interior point solution.

- **`memoryemphasis`**: Emphasizes memory saving during solving.  
  Recommended: `1` for large models.

- **`bardisplay`**: Controls the level of detail displayed during the barrier solution.  
  Recommended: `2` for detailed information.



**Note**: this list of parameters is far from exhaustive. Detailed information on how to choose the solver options are discussed [here](https://worldbankgroup.sharepoint.com/:b:/t/PowerSystemPlanning-WBGroup/EU2NwUyeOo9CljzcBCJThbsBac_sVZWv7GWmuUWf0XDIyw?e=wLkYhH).