# Threading and Parallelism in EPM

---

## Understanding CPU, Core, Thread, and vCPU

Efficient use of computing resources requires clarity on how modern processors are structured and virtualized:

- **CPU (Central Processing Unit):** The physical processor. A system may have one or more CPUs.
- **Core:** A processing unit within a CPU that executes instructions. Modern CPUs typically have multiple cores.
- **Thread:** A stream of execution handled by a core. With hyperthreading, one core can manage multiple threads (usually 2).
- **vCPU (Virtual CPU):** In virtualized/cloud environments, a vCPU typically maps to a single thread. It represents the smallest unit of compute allocated to a process.

| Term      | Description                        | Relation to Others                 |
|-----------|------------------------------------|------------------------------------|
| **CPU**   | Physical chip                       | Contains multiple **cores**        |
| **Core**  | Independent compute unit            | Can run 1–2 **threads**            |
| **Thread**| Stream of execution                 | Mapped to a **vCPU**               |
| **vCPU**  | Virtualized thread in the cloud     | ≈ 1 **thread** (part of a **core**) |

**Note:** A vCPU does not represent a full physical core. When a cloud provider offers "4 vCPUs", this usually means access to 4 hardware threads — not 4 full cores — and therefore you can run 4 parallel tasks, assuming no other bottlenecks.

---

## Framework to Define the Right Simulation Setup

Performance is often limited more by **available memory** than by CPU speed. Follow this step-by-step method to size your simulations appropriately:

### 1. Determine Available Memory

Start with the total available RAM on your server or VM.

> **Example**:  
> For the World Bank Planning Team server:  
> **Available RAM**: 128 GB

### 2. Check CPU and Core Information

Identify how many physical CPUs and cores you have.

> **Example**:  
> 1 physical CPU with **4 cores**, each supporting **2 threads**  
> → Total: **8 threads**, **8 vCPUs**

### 3. Estimate Memory Usage per Simulation

Run one standard model and monitor its peak memory usage using **GAMS Studio**:

- Open the `.lst` file or console output.
- Look for the **`ProcTreeMemMonitor`** entry.
- Use the **`VSS` (Virtual Set Size)** value — this is the total memory footprint.

> **Example**:  
> A typical simulation uses **32 GB RAM** (VSS value).

### 4. Calculate Max Parallel Jobs Based on Memory

Divide available RAM by the estimated memory per job:

```math
Max Parallel Simulations = Total RAM / Memory per Simulation
```

> **Example**:  
> 128 GB / 32 GB = **4 parallel simulations**

### 5. Set Number of Threads per Simulation

Each simulation can then use a subset of threads, depending on your compute layout:

- If your machine has 8 threads total and you run 4 simulations:
- Each simulation can use up to **2 threads**

You control this in GAMS using the CPLEX thread option:
```gams
option threads = 2;
```

---

### Example Setup Summary

| Parameter              | Value          |
|------------------------|----------------|
| Total RAM              | 128 GB         |
| Per-job RAM usage      | 32 GB          |
| Max parallel jobs      | 4              |
| Total threads available| 8              |
| Threads per simulation | 2              |
| Cores used             | 4              |
