# Performance & Threads

EPM can run several scenarios in parallel with the `--parallel` flag. This page explains how to size that parameter correctly for your machine.

---

## Key concepts

| Term | Meaning |
|---|---|
| **Core** | Physical compute unit of the processor |
| **Thread** | Execution stream; with hyperthreading, one core handles 2 threads |
| **vCPU** | In cloud/VM terms, 1 vCPU ≈ 1 thread |
| **`--parallel`** | Number of EPM scenarios launched simultaneously |
| **`threads`** | Number of threads allocated to each CPLEX solve, set in the solver options file |

> **Note** — the `--parallel` flag actually controls the number of **parallel jobs**, not CPUs directly. It will be renamed in a future version to avoid the confusion.

---

## The two ceilings to respect

Running `--parallel N` means N scenarios run at the same time. Each scenario consumes RAM **and** CPU threads. There are therefore **two independent ceilings**:

```
RAM ceiling  = total RAM / RAM per scenario
CPU ceiling  = total vCPUs / threads per scenario

--parallel = min(RAM ceiling, CPU ceiling)
```

The lower ceiling is the binding one. Exceeding either causes resource contention and slows every job down.

---

## How to work it out in practice

**Step 1 — Know your machine**

Note the total RAM and the number of available vCPUs.  
On Linux: `free -h` (RAM) and `nproc` (vCPUs).

**Step 2 — Measure RAM per scenario**

Run a single scenario on its own and look in the `.lst` file or the GAMS Studio console for:
```
ProcTreeMemMonitor → VSS
```
That is the peak memory footprint of that scenario. Use that value.

**Step 3 — Know your `threads`**

Look at your CPLEX options file (`cplex_baseline.opt`):
```
threads = 8
```
If the line is missing, CPLEX uses every available thread — to be avoided in a parallel context.  
See [Solver options](options_solver.md) to change the value.

**Step 4 — Compute `--parallel`**

```
RAM ceiling  = total RAM / RAM per scenario
CPU ceiling  = total vCPUs / threads

--parallel = min(RAM ceiling, CPU ceiling)
```

---

## Worked example

Machine: **256 GB RAM, 32 vCPUs**, scenario of ~32 GB, `threads = 8`

```
RAM ceiling  = 256 / 32  = 8 jobs
CPU ceiling  = 32 / 8    = 4 jobs

--parallel = min(8, 4) = 4
```

Here the CPU is binding. Run with `--parallel 4`, which leaves ~64 GB of RAM unused.

```sh
python epm.py --folder_input my_country --config config.csv --scenarios --parallel 4
```

---

## Trade-off: threads vs. parallel scenarios

The number of threads per solve is a parameter to tune to your usage.

**Many scenarios to get through (long queue)**  
→ Prefer **fewer threads, more parallel jobs**.  
Parallelisation across scenarios is near-perfect (each job is independent), whereas adding threads within a single solve has diminishing returns — going from 4 to 8 threads speeds a given solve up only slightly. More solves in parallel finish a long queue faster.

*Example: lowering to `threads = 5` → CPU ceiling = 32 / 5 = 6 jobs → `--parallel 6` instead of 4.*

**Few heavy scenarios (MIP without `--simple`)**  
→ Prefer **more threads, fewer parallel jobs**.  
Concentrate the resources on each solve to finish it faster.

> **Note** — you can overshoot the CPU ceiling slightly (e.g. `--parallel 6` with `threads = 8` on 32 vCPUs). The OS then time-shares the CPU between threads and the jobs run more slowly. Results stay correct, but overall throughput drops compared with a balanced allocation.

---

## Summary

| Situation | Recommendation |
|---|---|
| Long queue of RMIP scenarios | Lower `threads`, raise `--parallel` |
| A few heavy MIP scenarios | Keep `threads` high, `--parallel` lower |
| Shared machine (2 modellers) | Halve `--parallel` |
