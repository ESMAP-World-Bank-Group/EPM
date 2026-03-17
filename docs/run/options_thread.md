# Performance & Threads

EPM can run multiple scenarios in parallel using the `--cpu` flag. This page explains how to size that correctly for your machine.

---

## Key concepts

| Term | What it means |
|---|---|
| **Core** | A physical compute unit on the CPU |
| **Thread** | A stream of execution; with hyperthreading, one core handles 2 threads |
| **vCPU** | In cloud/VM environments, one vCPU ≈ one thread |

> A machine with "8 vCPUs" can run 8 parallel tasks, but memory is usually the real bottleneck.

---

## How to size your parallel runs

The limiting factor is almost always **RAM**, not CPU.

**Step 1:Check available RAM**
Note your total system memory (e.g. 128 GB).

**Step 2:Measure memory per simulation**
Run one scenario and check the `.lst` file or GAMS Studio console for `ProcTreeMemMonitor` → `VSS` (Virtual Set Size). This is the peak memory footprint per run.

**Step 3:Calculate max parallel jobs**

```
Max parallel jobs = Total RAM / Memory per simulation
```

**Step 4:Set threads per simulation**

```
Threads per simulation = Total threads / Max parallel jobs
```

Set this in your CPLEX option file: `threads = <value>` (see [Solver Options](options_solver.md)).

---

## Example

| Parameter | Value |
|---|---|
| Total RAM | 128 GB |
| Memory per simulation (VSS) | 32 GB |
| Max parallel jobs (`--cpu`) | 4 |
| Total threads | 8 |
| Threads per simulation (`threads`) | 2 |

```sh
python epm.py --folder_input my_country --config config.csv --scenarios --cpu 4
```
