
# Run EPM with Python

1. Activate environement using `requirements.txt` file:
    - You need to have Python installed in your computer. If you don't have it, you can download Anaconda distribution from [here](https://www.anaconda.com/products/distribution).
    - Use the terminal to run the command: `pip install -r requirements.txt`
    

2. Create `config.csv` file that contains the baseline scenario. Look at the `input` folder for an example.

3. Create `scenarios.csv` file that contains the specification of the scenarios. Look at the `input` folder for an example. By default, the model will run the baseline scenario. 

4. Use `run_epm.py`or a notebook to launch the code.

For example:
```python 

# Define the path to the folder
PATH_GAMS = {
    'path_main_file': 'WB_EPM_v8_5_main.gms',
    'path_base_file': 'WB_EPM_v8_5_base.gms',
    'path_report_file': 'WB_EPM_v8_5_Report.gms',
    'path_reader_file': 'WB_EPM_input_readers.gms',
    'path_cplex_file': 'cplex.opt'
}

# Select the scenarios to run in the selected_scenarios list
launch_epm_multi_scenarios(
    scenario_baseline='input/scenario_baseline.csv',
    scenarios_specification='input/scenarios_specification.csv',
    selected_scenarios=['baseline'],
    cpu=1
    )
```

## Running jobs in parallel with python

The function `launch_epm_multi_scenarios` allows to launch jobs in parallel, by specifying the number of `cpus`.

### Recap on parallel runs

#### Key components of a System

Machine (Local Computer or Cluster):

1. The physical or virtual system where your code runs

Has a specific CPU (central processing unit), RAM (random access memory), and possibly multiple cores.

2. CPU and Cores

CPU: The main processor; a chip responsible for executing instructions. On local machines, there is usually a single CPU.

Cores: CPUs often have multiple cores, which allow for simultaneous processing of different tasks. For example, a quad-core CPU has 4 cores. Recent local machines (such as MacBook pro) often include multiple cores.

Threads: Logical divisions of a core. Some CPUs support hyper-threading, where each core can handle two threads simultaneously (e.g., a quad-core CPU with hyper-threading supports 8 threads).

3. RAM:

Temporary memory used to store data required by running programs. Jobs with high memory usage (like optimization models) can quickly exhaust this.

#### Key considerations

**Thread Overhead**:

If your local machine has 8 cores, and you set cpu=8 for the Pool, all cores are occupied.
If each GAMS optimization uses threads=12, you’re asking for more threads than your system can physically support. This leads to:
- Thread contention: Processes fight for CPU resources, reducing efficiency.
- Slower execution.

**RAM Usage**:

Each worker process runs a separate instance of GAMS, consuming its own RAM for the optimization model.
If your machine has 16 GB of RAM and each GAMS job requires 4 GB, running 8 parallel jobs will exceed your memory capacity, leading to swapping or crashing

### Example

If you launch `launch_epm_multi_scenarios` with `cpu=1`, you will only run one simulation at a time on the machine where you are launching this. 

**Note**: if the simulations are then launched on the Engine cluster, they may be executed in parallel. Each job you submit to the cluster is an independent task. The cluster's scheduler then takes over, running these jobs in parallel based on the available resources on the cluster.

### Running EPM in parallel

There are two sources of parallel resources:
- the number of jobs/simulations you launch in parallel.
- the number of threads used by each simulation for the optimization in GAMS.

Depending on the available resources, you should carefully balance both threads and cpu to make sure that resources are used as efficiently as possible.