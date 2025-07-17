# CPLEX Options


TODO 
- What it is 
- What are the main parameters
- But also other parameters 


## 1. Number of threads

TODO: Th thread option

### Reminder - What are CPU, Core, Thread, and vCPU?

- **CPU** (Central Processing Unit): The physical processor. A machine may have one or multiple CPUs.
- **Core**: Each CPU contains one or more cores. Each core can execute tasks independently.
- **Thread**: A core can run one or more threads (typically 2 with hyperthreading). A thread is a stream of execution.
- **vCPU** (Virtual CPU): A virtualized unit of compute, typically mapped to a single thread, not an entire core. It is the basic unit of processing in cloud environments.


| Term      | Description                              | Relation to Others                               |
|:----------|:------------------------------------------|:--------------------------------------------------|
| **CPU**   | Physical processor                        | Contains multiple **cores**                      |
| **Core**  | Independent processing unit               | May run 1–2 **threads**                          |
| **Thread**| Smallest unit of execution                | Mapped to 1 **vCPU**                             |
| **vCPU**  | Virtualized thread (cloud environment)    | ≈ 1 **thread** (part of a **core**)              |


Note: A vCPU does not include multiple cores. It typically corresponds to a single thread within a core. When cloud providers offer "4 vCPUs", this means your virtual machine can run 4 tasks in parallel — but not necessarily with access to 4 full cores.

---

## 2. TO COMPLETE

